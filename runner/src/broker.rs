//! Game server on the game host. The GPU worker connects OUT to it; nothing on the GPU workstation listens.
//!   POST /job     {"name", "seed", "games": [{"seats": ["learner", "random"], "replay": false}, ...]} -> {"job": id}
//!                 A seat is "random" (played here) or any other policy tag, answered by the worker.
//!   GET  /next?max=N&wait=S&gather=MS   long-poll (default 25 s); once a decision is waiting it keeps
//!                      gathering for up to MS ms (default 0) or until every running game is waiting: {"requests": [{"id","job","idx","seat","policy","lp","prompt","n"}],
//!                      "results": [finished games]}; results are handed out once
//!   POST /answer  [{"id","choice"}, ...] (or one object)
//!   POST /cancel  drop games not started yet -> {"cancelled": n}
//!   GET  /live    the latest move (board, options, choice) of every running game marked "live"
//!   GET  /status (with "active": {job: games queued or running}), GET /health
//! Job ids start from the clock, so they never repeat across restarts.
//! A request handed out but unanswered for 90 s is handed out again (the worker may have been stopped
//! for a game); a duel waiting on the worker just waits, it never times out.
use crate::data::Library;
use crate::decide::Decision;
use crate::duel::{self, Agent, Ctx, Outcome, RandomAgent, WindBotAgent};
use crate::rng::Rng;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub struct GameSpec {
    pub job: u64,
    pub idx: u32,
    pub seed: u64,
    pub seats: [String; 2],
    pub replay: bool,
    pub live: bool,
    pub name: String,
}

struct Pending {
    body: Value,
    handed: Option<Instant>,
    answer: Option<usize>,
}

#[derive(Default)]
struct State {
    next_id: u64,
    next_job: u64,
    queue: VecDeque<GameSpec>,
    running: usize,
    pending: BTreeMap<u64, Pending>,
    active: HashMap<u64, usize>, // job -> games queued or running
    live: BTreeMap<(u64, u32), (Value, Instant)>, // (job, idx) -> latest move; finished games linger 60 s
    results: Vec<Value>,
    served: u64,
    answered: u64,
    games_done: u64,
    errors: u64,
}

#[derive(Clone)]
pub struct Broker(Arc<(Mutex<State>, Condvar)>);

impl Broker {
    pub fn start(addr: &str) -> Result<Broker, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let st = State { next_job: now * 1000, ..Default::default() };
        let b = Broker(Arc::new((Mutex::new(st), Condvar::new())));
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

    fn handle(&self, mut s: TcpStream) -> std::io::Result<()> {
        s.set_read_timeout(Some(Duration::from_secs(30)))?;
        let mut rd = BufReader::new(s.try_clone()?);
        let mut line = String::new();
        rd.read_line(&mut line)?;
        let mut parts = line.split_whitespace();
        let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
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
        let mut body = vec![0u8; clen.min(64 << 20)];
        rd.read_exact(&mut body)?;
        let q = |k: &str| query.split('&').find_map(|kv| kv.strip_prefix(k)).and_then(|v| v.parse::<u64>().ok());
        let (qmax, qwait) = (q("max=").unwrap_or(64) as usize, q("wait=").unwrap_or(25).min(25));
        let qgather = q("gather=").unwrap_or(0).min(2000);
        let (code, out) = match (method, path) {
            ("GET", "/health") => (200, "ok".to_string()),
            ("GET", "/status") => (200, self.status().to_string()),
            ("GET", "/live") => (200, self.live_games().to_string()),
            ("GET", "/next") => (200, self.next(qmax, qwait, qgather).to_string()),
            ("POST", "/answer") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => {
                    let list = if v.is_array() { v.as_array().unwrap().clone() } else { vec![v] };
                    let mut bad = 0;
                    for a in list {
                        match (a["id"].as_u64(), a["choice"].as_u64()) {
                            (Some(id), Some(c)) => self.answer(id, c as usize),
                            _ => bad += 1,
                        }
                    }
                    if bad == 0 { (200, "ok".into()) } else { (400, "need id and choice".into()) }
                }
                Err(e) => (400, e.to_string()),
            },
            ("POST", "/job") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => match self.add_job(&v) {
                    Ok(id) => (200, json!({"job": id}).to_string()),
                    Err(e) => (400, e),
                },
                Err(e) => (400, e.to_string()),
            },
            ("POST", "/cancel") => (200, json!({"cancelled": self.cancel()}).to_string()),
            _ => (404, "not found".into()),
        };
        let reason = match code { 200 => "OK", 400 => "Bad Request", _ => "Not Found" };
        write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len())?;
        s.flush()
    }

    fn status(&self) -> Value {
        let st = self.0 .0.lock().unwrap();
        let active: serde_json::Map<String, Value> = st.active.iter().map(|(j, n)| (j.to_string(), json!(n))).collect();
        json!({"active": active, "queued": st.queue.len(), "running": st.running, "pending": st.pending.len(), "results_waiting": st.results.len(),
            "games_done": st.games_done, "errors": st.errors, "served": st.served, "answered": st.answered})
    }

    fn live_games(&self) -> Value {
        let mut st = self.0 .0.lock().unwrap();
        st.live.retain(|_, (v, t)| !(v["end"].as_bool().unwrap_or(false) && t.elapsed() > Duration::from_secs(60)));
        Value::Array(st.live.values().map(|(v, _)| v.clone()).collect())
    }

    fn publish(&self, job: u64, idx: u32, v: Value) {
        let mut st = self.0 .0.lock().unwrap();
        st.live.insert((job, idx), (v, Instant::now()));
    }

    fn add_job(&self, v: &Value) -> Result<u64, String> {
        let games = v["games"].as_array().ok_or("need games")?;
        let seed = v["seed"].as_u64().unwrap_or(1);
        let name = v["name"].as_str().unwrap_or("job").to_string();
        let mut specs = Vec::new();
        for (i, g) in games.iter().enumerate() {
            let seats = g["seats"].as_array().ok_or("need seats")?;
            let seat = |k: usize| seats.get(k).and_then(|s| s.as_str()).map(String::from).ok_or("seat must be a string");
            specs.push((i as u32, [seat(0)?, seat(1)?], g["replay"].as_bool().unwrap_or(false), g["live"].as_bool().unwrap_or(false)));
        }
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        st.next_job += 1;
        let job = st.next_job;
        for (idx, seats, replay, live) in specs {
            let seed = seed.wrapping_mul(1_000_003).wrapping_add(job << 32).wrapping_add(idx as u64);
            st.queue.push_back(GameSpec { job, idx, seed, seats, replay, live, name: name.clone() });
            *st.active.entry(job).or_insert(0) += 1;
        }
        cv.notify_all();
        Ok(job)
    }

    fn cancel(&self) -> usize {
        let mut st = self.0 .0.lock().unwrap();
        let n = st.queue.len();
        let dropped: Vec<u64> = st.queue.drain(..).map(|g| g.job).collect();
        for j in dropped {
            if let Some(c) = st.active.get_mut(&j) {
                *c -= 1;
                if *c == 0 {
                    st.active.remove(&j);
                }
            }
        }
        n
    }

    /// Up to `max` requests not handed out yet (or handed out 90+ s ago), plus all finished results.
    fn next(&self, max: usize, wait: u64, gather: u64) -> Value {
        let (m, cv) = &*self.0;
        let deadline = Instant::now() + Duration::from_secs(wait);
        let mut first_seen: Option<Instant> = None;
        let mut st = m.lock().unwrap();
        loop {
            let open = |p: &Pending| p.answer.is_none() && p.handed.map(|t| t.elapsed() > Duration::from_secs(90)).unwrap_or(true);
            let ready = st.pending.values().filter(|p| open(p)).count();
            let now = Instant::now();
            if ready > 0 && first_seen.is_none() {
                first_seen = Some(now);
            }
            let gathered = first_seen.map(|t| now.duration_since(t) >= Duration::from_millis(gather)).unwrap_or(false);
            let all_waiting = ready >= st.running.max(1) || ready >= max;
            if (ready > 0 && (gathered || all_waiting)) || !st.results.is_empty() && ready == 0 || now >= deadline {
                let mut reqs = Vec::new();
                for p in st.pending.values_mut() {
                    if reqs.len() >= max {
                        break;
                    }
                    if open(p) {
                        p.handed = Some(Instant::now());
                        reqs.push(p.body.clone());
                    }
                }
                st.served += reqs.len() as u64;
                let results: Vec<Value> = std::mem::take(&mut st.results);
                return json!({"requests": reqs, "results": results});
            }
            let until = match first_seen {
                Some(t) => (t + Duration::from_millis(gather)).min(deadline),
                None => deadline,
            };
            st = cv.wait_timeout(st, until.saturating_duration_since(now).max(Duration::from_millis(1))).unwrap().0;
        }
    }

    fn answer(&self, id: u64, choice: usize) {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        if let Some(p) = st.pending.get_mut(&id) {
            if p.answer.is_none() {
                p.answer = Some(choice);
                st.answered += 1;
                cv.notify_all();
            }
        }
    }

    /// Post one decision and block until the worker answers it.
    pub fn ask(&self, body: Value) -> usize {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        st.next_id += 1;
        let id = st.next_id;
        let mut body = body;
        body["id"] = json!(id);
        st.pending.insert(id, Pending { body, handed: None, answer: None });
        cv.notify_all();
        loop {
            if let Some(c) = st.pending.get(&id).and_then(|p| p.answer) {
                st.pending.remove(&id);
                return c;
            }
            st = cv.wait_timeout(st, Duration::from_secs(5)).unwrap().0;
        }
    }

    fn take_game(&self) -> GameSpec {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        loop {
            if let Some(g) = st.queue.pop_front() {
                st.running += 1;
                return g;
            }
            st = cv.wait(st).unwrap();
        }
    }

    fn finish_game(&self, job: u64, result: Value, error: bool) {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        st.running -= 1;
        if let Some(c) = st.active.get_mut(&job) {
            *c -= 1;
            if *c == 0 {
                st.active.remove(&job);
            }
        }
        st.games_done += 1;
        st.errors += error as u64;
        st.results.push(result);
        cv.notify_all();
    }
}

/// A seat whose decisions (two or more choices) go to the worker, tagged with its policy.
pub struct RemoteAgent<'a> {
    pub broker: &'a Broker,
    pub job: u64,
    pub idx: u32,
    pub seat: u8,
    pub policy: String,
    pub requests: u64,
}

impl Agent for RemoteAgent<'_> {
    fn wants_board(&self) -> bool {
        true
    }
    fn choose(&mut self, d: &Decision, ctx: &Ctx, _rng: &mut Rng) -> Result<usize, String> {
        if d.choices.len() == 1 {
            return Ok(0);
        }
        let me = (d.player & 1) as usize;
        let prompt = ctx.prompt.ok_or("remote seat without a prompt")?;
        self.requests += 1;
        let c = self.broker.ask(json!({"job": self.job, "idx": self.idx, "seat": self.seat, "policy": self.policy,
            "lp": [ctx.view.lp[me], ctx.view.lp[1 - me]], "turn": ctx.view.turn, "prompt": prompt, "n": d.choices.len()}));
        if c >= d.choices.len() {
            return Err(format!("worker answered {c} for {} options", d.choices.len()));
        }
        Ok(c)
    }
}

/// Game threads: each takes the next queued game, plays it, and posts the result.
pub fn run_games(broker: Broker, lib: Arc<Library>, deck: Arc<(Vec<u32>, Vec<u32>)>, threads: usize, replay_dir: PathBuf) {
    let _ = std::fs::create_dir_all(&replay_dir);
    for _ in 0..threads {
        let (b, lib, deck, dir) = (broker.clone(), lib.clone(), deck.clone(), replay_dir.clone());
        std::thread::spawn(move || loop {
            let g = b.take_game();
            let started = Instant::now();
            let mut agents: Vec<Box<dyn Agent + '_>> = Vec::new();
            let mut seat_error: Option<String> = None;
            for (k, pol) in g.seats.iter().enumerate() {
                if pol == "random" {
                    agents.push(Box::new(RandomAgent));
                } else if pol == "windbot" {
                    let server = std::env::var("YGO_WINDBOT").unwrap_or("127.0.0.1:2399".into());
                    match crate::windbot::WindBot::connect(&server, "CyberDragon", k as u8, "WindBot") {
                        Ok(w) => agents.push(Box::new(WindBotAgent(w))),
                        Err(e) => {
                            seat_error = Some(e);
                            agents.push(Box::new(RandomAgent));
                        }
                    }
                } else {
                    agents.push(Box::new(RemoteAgent { broker: &b, job: g.job, idx: g.idx, seat: k as u8, policy: pol.clone(), requests: 0 }));
                }
            }
            let (a0, rest) = agents.split_at_mut(1);
            let header = json!({"job": g.job, "game": g.idx, "name": g.name, "seats": g.seats});
            let publish = |mut v: Value| {
                for (k, x) in header.as_object().unwrap() {
                    v[k] = x.clone();
                }
                b.publish(g.job, g.idx, v);
            };
            let live: Option<&dyn Fn(Value)> = if g.live { Some(&publish) } else { None };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &seat_error {
                Some(e) => Err(e.clone()),
                None => duel::play(&lib, &deck, g.seed, &mut [a0[0].as_mut(), rest[0].as_mut()], g.replay, live),
            }));
            let res = match res {
                Ok(r) => r,
                Err(p) => Err(format!(
                    "panic: {}",
                    p.downcast_ref::<String>().cloned().or(p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
                )),
            };
            let mut out = json!({"job": g.job, "idx": g.idx, "name": g.name, "seed": g.seed, "seats": g.seats,
                "seconds": started.elapsed().as_secs_f64()});
            let error = res.is_err();
            match res {
                Ok(r) => {
                    let (winner, reason) = match r.outcome {
                        Outcome::Win { player, reason } => (player as i32, reason.to_string()),
                        Outcome::Draw { reason } => (-1, reason.to_string()),
                    };
                    out["winner"] = json!(winner);
                    out["reason"] = json!(reason);
                    out["turns"] = json!(r.turns);
                    out["decisions"] = json!(r.decisions);
                    out["lp"] = json!(r.lp);
                    out["script_errors"] = json!(r.script_errors);
                    out["max_choices"] = json!(r.max_choices);
                    out["loop_breaks"] = json!(r.loop_breaks);
                    out["loop_turns"] = json!(r.loop_turns);
                    if g.replay {
                        let file = format!("{}-j{}-g{}.json", g.name, g.job, g.idx);
                        let rep = json!({"name": g.name, "job": g.job, "game": g.idx, "seed": g.seed, "seats": g.seats,
                            "winner": winner, "reason": reason, "turns": r.turns, "lp": r.lp, "names": r.names, "steps": r.steps,
                            "finished": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)});
                        if std::fs::write(dir.join(&file), rep.to_string()).is_ok() {
                            out["replay"] = json!(file);
                        }
                    }
                }
                Err(e) => out["error"] = json!(e),
            }
            let wb: Vec<Value> = agents.iter_mut().filter_map(|a| a.windbot().map(|w| json!({"seat": w.seat, "decisions": w.decisions, "fallbacks": w.fallbacks}))).collect();
            if !wb.is_empty() {
                out["windbot"] = json!(wb);
            }
            drop(agents);
            b.finish_game(g.job, out, error);
        });
    }
}
