//! Tiny HTTP broker on the game host. The GPU worker connects OUT to it:
//!   GET  /next    long-poll (25 s): 200 + {"id","game","prompt","options"} or 204 when idle
//!   POST /answer  {"id","choice"}
//!   GET  /health  200
//! A request handed out but unanswered for 90 s is handed out again (the worker may have been stopped
//! for a game); the duel just waits, it never times out.
use crate::decide::Decision;
use crate::duel::{Agent, View};
use crate::rng::Rng;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    next_id: u64,
    pending: Option<(u64, Value, Option<Instant>)>, // id, body, when handed out
    answer: Option<(u64, usize)>,
    served: u64,
    answered: u64,
}

#[derive(Clone)]
pub struct Broker(Arc<(Mutex<State>, Condvar)>);

impl Broker {
    pub fn start(addr: &str) -> Result<Broker, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
        let b = Broker(Arc::new((Mutex::new(State::default()), Condvar::new())));
        let bc = b.clone();
        std::thread::spawn(move || {
            for s in listener.incoming().flatten() {
                let bc = bc.clone();
                std::thread::spawn(move || {
                    let _ = bc.handle(s);
                });
            }
        });
        Ok(b)
    }

    pub fn counts(&self) -> (u64, u64) {
        let st = self.0 .0.lock().unwrap();
        (st.served, st.answered)
    }

    fn handle(&self, mut s: TcpStream) -> std::io::Result<()> {
        s.set_read_timeout(Some(Duration::from_secs(30)))?;
        let mut rd = BufReader::new(s.try_clone()?);
        let mut line = String::new();
        rd.read_line(&mut line)?;
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let mut clen = 0usize;
        loop {
            let mut h = String::new();
            if rd.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                clen = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; clen.min(1 << 20)];
        rd.read_exact(&mut body)?;
        let (code, out) = match (method, path) {
            ("GET", "/health") => (200, "ok".to_string()),
            ("GET", "/next") => match self.next() {
                Some(v) => (200, v.to_string()),
                None => (204, String::new()),
            },
            ("POST", "/answer") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => match (v["id"].as_u64(), v["choice"].as_u64()) {
                    (Some(id), Some(c)) => {
                        self.answer(id, c as usize);
                        (200, "ok".into())
                    }
                    _ => (400, "need id and choice".into()),
                },
                Err(e) => (400, e.to_string()),
            },
            _ => (404, "not found".into()),
        };
        let reason = match code { 200 => "OK", 204 => "No Content", 400 => "Bad Request", _ => "Not Found" };
        write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len())?;
        s.flush()
    }

    fn next(&self) -> Option<Value> {
        let (m, cv) = &*self.0;
        let deadline = Instant::now() + Duration::from_secs(25);
        let mut st = m.lock().unwrap();
        loop {
            let st_ref = &mut *st;
            if let Some((_, body, handed)) = st_ref.pending.as_mut() {
                if handed.map(|t| t.elapsed() > Duration::from_secs(90)).unwrap_or(true) {
                    *handed = Some(Instant::now());
                    st_ref.served += 1;
                    return Some(body.clone());
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            st = cv.wait_timeout(st, deadline - now).unwrap().0;
        }
    }

    fn answer(&self, id: u64, choice: usize) {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        if st.pending.as_ref().map(|p| p.0) == Some(id) {
            st.answer = Some((id, choice));
            st.answered += 1;
            cv.notify_all();
        }
    }

    /// Post one decision and block until the worker answers it.
    pub fn ask(&self, game: u32, prompt: String, options: Vec<String>) -> usize {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        st.next_id += 1;
        let id = st.next_id;
        st.pending = Some((id, json!({"id": id, "game": game, "prompt": prompt, "options": options}), None));
        st.answer = None;
        cv.notify_all();
        let mut waited = Instant::now();
        loop {
            if let Some((aid, c)) = st.answer {
                if aid == id {
                    st.pending = None;
                    st.answer = None;
                    return c;
                }
            }
            st = cv.wait_timeout(st, Duration::from_secs(5)).unwrap().0;
            if waited.elapsed() > Duration::from_secs(60) {
                eprintln!("{{\"event\":\"waiting_for_worker\",\"id\":{id}}}");
                waited = Instant::now();
            }
        }
    }
}

const PHASES: [(u16, &str); 10] = [
    (0x01, "draw phase"), (0x02, "standby phase"), (0x04, "main phase 1"), (0x08, "battle phase"),
    (0x10, "battle step"), (0x20, "damage step"), (0x40, "damage calculation"), (0x80, "end of battle"),
    (0x100, "main phase 2"), (0x200, "end phase"),
];

/// Player whose decisions (two or more choices) go to the GPU worker.
pub struct RemoteAgent {
    pub broker: Broker,
    pub game: u32,
    pub requests: u64,
}

impl Agent for RemoteAgent {
    fn choose(&mut self, d: &Decision, v: &View, _rng: &mut Rng) -> Result<usize, String> {
        if d.choices.len() == 1 {
            return Ok(0);
        }
        let me = (d.player & 1) as usize;
        let phase = PHASES.iter().find(|p| p.0 == v.phase).map(|p| p.1).unwrap_or("start of turn");
        let prompt = format!(
            "Yu-Gi-Oh duel, turn {}, {phase}. My life points {}, opponent's {}. Decision: {}. Best choice:",
            v.turn, v.lp[me], v.lp[1 - me], d.kind
        );
        let options = d.choices.iter().map(|c| c.text.clone()).collect();
        self.requests += 1;
        let c = self.broker.ask(self.game, prompt, options);
        if c >= d.choices.len() {
            return Err(format!("worker answered {c} for {} options", d.choices.len()));
        }
        Ok(c)
    }
}
