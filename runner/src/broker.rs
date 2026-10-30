//! Game server on the game host. The GPU worker connects OUT to it; nothing on the GPU workstation listens.
//!   POST /job     {"name", "seed", "token", "games": [{"seats": ["learner", "random"], "replay": false}, ...]}
//!                 -> {"job": id, "token": t}. "name" is [A-Za-z0-9._-[]:+=@,], at most 128 characters. "token" is
//!                 optional (16-128 of [A-Za-z0-9_-]); without one the server makes a random one. Only a /next that
//!                 sends "X-Job-Token: t" gets this job's results.
//!                 A seat is "random" (played here) or any other policy tag, answered by the worker.
//!                 Round 5: a game may carry "decks": [d0, d1], WindBot deck names (AI_<d>.ydk in YGO_DECK_DIR,
//!                 default /srv/ygo/windbot/Decks); null or no "decks" is the server's --deck. WindBot seats play
//!                 their seat's deck.
//!   GET  /next?max=N&wait=S&gather=MS   long-poll (default 25 s); once a decision is waiting it keeps
//!                      gathering for up to MS ms (default 0) or until every running game is waiting: {"requests": [{"id","job","idx","seat","policy","lp","prompt","n"}],
//!                      "results": [finished games of the caller's X-Job-Token]}; results are handed out once.
//!                      Every request carries a "lease" that only this /next response holds.
//!   POST /answer  [{"id","lease","choice"}, ...] (or one object) -> {"accepted": n}, or 409 with
//!                 {"accepted": n, "rejected": [ids]} for an unknown, already answered or stale (re-leased) id
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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const MAX_NAME: usize = 128;
const MAX_GAMES_PER_JOB: usize = 10_000;
const MAX_QUEUE: usize = 50_000;
const MAX_RESULTS: usize = 50_000; // results nobody collected: the oldest are dropped past this
const MAX_CONNS: usize = 256;
const MAX_BODY: usize = 8 << 20;
const REQUEST_DEADLINE: Duration = Duration::from_secs(30); // whole request head and body, not per read

pub struct GameSpec {
    pub job: u64,
    pub idx: u32,
    pub seed: u64,
    pub seats: [String; 2],
    pub replay: bool,
    pub live: bool,
    pub name: Arc<str>,
    pub owner: Arc<str>,
    pub decks: [Option<String>; 2],
}

struct Pending {
    body: Value,
    handed: Option<Instant>,
    lease: Option<String>,
    answer: Option<usize>,
}

/// 128 random bits as hex, for job tokens and decision leases.
fn token() -> String {
    let mut b = [0u8; 16];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).expect("read /dev/urandom");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A job name is part of the replay file name, so it may not hold a path separator.
fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= MAX_NAME && n.chars().all(|c| c.is_ascii_alphanumeric() || "._-[]:+=@,".contains(c))
}

fn valid_token(t: &str) -> bool {
    (16..=128).contains(&t.len()) && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Reads with one deadline for the whole request, so a peer trickling bytes cannot hold a connection open.
struct DeadlineReader {
    s: TcpStream,
    until: Instant,
}

impl Read for DeadlineReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "request deadline"));
        }
        self.s.set_read_timeout(Some(left))?;
        self.s.read(buf)
    }
}

/// Counts a live connection; the count drops when the handler (or a thread that never started) is dropped.
struct ConnGuard(Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
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
    results: Vec<(Arc<str>, Value)>,                // (job token, result)
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
        let conns = Arc::new(AtomicUsize::new(0));
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loop {
                let s = match listener.accept() {
                    Ok((s, _)) => s,
                    Err(e) => {
                        eprintln!("broker accept: {e}"); // e.g. out of file descriptors: back off, don't spin
                        std::thread::sleep(Duration::from_millis(200));
                        continue;
                    }
                };
                if conns.fetch_add(1, Ordering::SeqCst) >= MAX_CONNS {
                    conns.fetch_sub(1, Ordering::SeqCst);
                    continue; // over the cap: the connection is closed at once
                }
                let (bc, guard) = (bc.clone(), ConnGuard(conns.clone()));
                if std::thread::Builder::new()
                    .spawn(move || {
                        let _guard = guard;
                        let _ = bc.handle(s);
                    })
                    .is_err()
                {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }));
            eprintln!("broker accept thread died: {r:?}");
            std::process::exit(1); // let the service manager restart a broker that no longer listens
        });
        Ok(b)
    }

    fn handle(&self, mut s: TcpStream) -> std::io::Result<()> {
        s.set_write_timeout(Some(REQUEST_DEADLINE))?;
        let mut rd = BufReader::new(DeadlineReader { s: s.try_clone()?, until: Instant::now() + REQUEST_DEADLINE });
        let mut line = String::new();
        rd.read_line(&mut line)?;
        let mut parts = line.split_whitespace();
        let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let mut clen = 0usize;
        let mut job_token = String::new();
        loop {
            let mut h = String::new();
            if rd.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                clen = v.trim().parse().unwrap_or(0);
            }
            if h.to_ascii_lowercase().starts_with("x-job-token:") {
                job_token = h["x-job-token:".len()..].trim().to_string();
            }
        }
        let mut body = vec![0u8; clen.min(MAX_BODY)];
        rd.read_exact(&mut body)?;
        let q = |k: &str| query.split('&').find_map(|kv| kv.strip_prefix(k)).and_then(|v| v.parse::<u64>().ok());
        let (qmax, qwait) = (q("max=").unwrap_or(64) as usize, q("wait=").unwrap_or(25).min(25));
        let qgather = q("gather=").unwrap_or(0).min(2000);
        let (code, out) = match (method, path) {
            ("GET", "/health") => (200, "ok".to_string()),
            ("GET", "/status") => (200, self.status().to_string()),
            ("GET", "/live") => (200, self.live_games().to_string()),
            ("GET", "/next") => (200, self.next(qmax, qwait, qgather, Some(job_token.as_str()).filter(|t| !t.is_empty())).to_string()),
            ("POST", "/answer") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => {
                    let list = if v.is_array() { v.as_array().unwrap().clone() } else { vec![v] };
                    let parsed: Vec<_> = list.iter().map(|a| (a["id"].as_u64(), a["lease"].as_str(), a["choice"].as_u64())).collect();
                    if parsed.iter().any(|(i, l, c)| i.is_none() || l.is_none() || c.is_none()) {
                        (400, "need id, lease and choice".into())
                    } else {
                        let rejected: Vec<u64> = parsed.iter().filter(|(i, l, c)| !self.answer(i.unwrap(), l.unwrap(), c.unwrap() as usize)).map(|(i, _, _)| i.unwrap()).collect();
                        let accepted = parsed.len() - rejected.len();
                        if rejected.is_empty() {
                            (200, json!({"accepted": accepted}).to_string())
                        } else {
                            (409, json!({"accepted": accepted, "rejected": rejected}).to_string())
                        }
                    }
                }
                Err(e) => (400, e.to_string()),
            },
            ("POST", "/job") => match serde_json::from_slice::<Value>(&body) {
                Ok(v) => match self.add_job(&v) {
                    Ok((id, tok)) => (200, json!({"job": id, "token": &*tok}).to_string()),
                    Err(e) => (400, e),
                },
                Err(e) => (400, e.to_string()),
            },
            ("POST", "/cancel") => (200, json!({"cancelled": self.cancel()}).to_string()),
            _ => (404, "not found".into()),
        };
        let reason = match code { 200 => "OK", 400 => "Bad Request", 409 => "Conflict", _ => "Not Found" };
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

    fn add_job(&self, v: &Value) -> Result<(u64, Arc<str>), String> {
        let games = v["games"].as_array().ok_or("need games")?;
        if games.len() > MAX_GAMES_PER_JOB {
            return Err(format!("at most {MAX_GAMES_PER_JOB} games per job"));
        }
        let seed = v["seed"].as_u64().unwrap_or(1);
        let name: Arc<str> = match &v["name"] {
            Value::Null => "job".into(),
            Value::String(n) if valid_name(n) => n.as_str().into(),
            _ => return Err(format!("bad job name: at most {MAX_NAME} of [A-Za-z0-9._-[]:+=@,]")),
        };
        let owner: Arc<str> = match &v["token"] {
            Value::Null => token().into(),
            Value::String(t) if valid_token(t) => t.as_str().into(),
            _ => return Err("bad token: 16-128 of [A-Za-z0-9_-]".into()),
        };
        let mut specs = Vec::new();
        for (i, g) in games.iter().enumerate() {
            let seats = g["seats"].as_array().ok_or("need seats")?;
            let seat = |k: usize| seats.get(k).and_then(|s| s.as_str()).map(String::from).ok_or("seat must be a string");
            let deck = |k: usize| -> Result<Option<String>, String> {
                match g["decks"].get(k) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::String(d)) if !d.is_empty() && d.chars().all(|c| c.is_ascii_alphanumeric() || "-_[]".contains(c)) => Ok(Some(d.clone())),
                    Some(d) => Err(format!("bad deck name {d}")),
                }
            };
            specs.push((i as u32, [seat(0)?, seat(1)?], g["replay"].as_bool().unwrap_or(false), g["live"].as_bool().unwrap_or(false),
                g["seed"].as_u64(), [deck(0)?, deck(1)?]));
        }
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        if st.queue.len() + specs.len() > MAX_QUEUE {
            return Err(format!("queue full ({} games queued, at most {MAX_QUEUE})", st.queue.len()));
        }
        st.next_job += 1;
        let job = st.next_job;
        for (idx, seats, replay, live, own_seed, decks) in specs {
            // round 4: a game may carry its own seed (paired games: the same deal with the seats swapped)
            let seed = own_seed.unwrap_or(seed.wrapping_mul(1_000_003).wrapping_add(job << 32).wrapping_add(idx as u64));
            st.queue.push_back(GameSpec { job, idx, seed, seats, replay, live, name: name.clone(), owner: owner.clone(), decks });
            *st.active.entry(job).or_insert(0) += 1;
        }
        cv.notify_all();
        Ok((job, owner))
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

    /// Up to `max` requests not handed out yet (or handed out 90+ s ago), each under a fresh lease, plus the
    /// finished results of the jobs whose token is `owner`.
    fn next(&self, max: usize, wait: u64, gather: u64, owner: Option<&str>) -> Value {
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
            let mine = |st: &State| owner.is_some_and(|o| st.results.iter().any(|(t, _)| &**t == o));
            if (ready > 0 && (gathered || all_waiting)) || mine(&st) && ready == 0 || now >= deadline {
                let lease = token();
                let mut reqs = Vec::new();
                for p in st.pending.values_mut() {
                    if reqs.len() >= max {
                        break;
                    }
                    if open(p) {
                        p.handed = Some(Instant::now());
                        p.lease = Some(lease.clone());
                        let mut body = p.body.clone();
                        body["lease"] = json!(lease);
                        reqs.push(body);
                    }
                }
                st.served += reqs.len() as u64;
                let (results, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut st.results).into_iter().partition(|(t, _)| Some(&**t) == owner);
                st.results = rest;
                let results: Vec<Value> = results.into_iter().map(|(_, v)| v).collect();
                return json!({"requests": reqs, "results": results});
            }
            let until = match first_seen {
                Some(t) => (t + Duration::from_millis(gather)).min(deadline),
                None => deadline,
            };
            st = cv.wait_timeout(st, until.saturating_duration_since(now).max(Duration::from_millis(1))).unwrap().0;
        }
    }

    /// Accepts the first answer under the request's current lease; false for an unknown, answered or stale one.
    fn answer(&self, id: u64, lease: &str, choice: usize) -> bool {
        let (m, cv) = &*self.0;
        let mut st = m.lock().unwrap();
        match st.pending.get_mut(&id) {
            Some(p) if p.answer.is_none() && p.lease.as_deref() == Some(lease) => {
                p.answer = Some(choice);
                st.answered += 1;
                cv.notify_all();
                true
            }
            _ => false,
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
        st.pending.insert(id, Pending { body, handed: None, lease: None, answer: None });
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

    fn finish_game(&self, job: u64, owner: Arc<str>, result: Value, error: bool) {
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
        st.results.push((owner, result));
        if st.results.len() > MAX_RESULTS {
            let excess = st.results.len() - MAX_RESULTS;
            st.results.drain(..excess);
        }
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
    fn game_id(&self) -> Option<(u64, u32)> {
        Some((self.job, self.idx))
    }
    fn choose(&mut self, d: &Decision, ctx: &Ctx, _rng: &mut Rng) -> Result<usize, String> {
        if d.choices.len() == 1 {
            return Ok(0);
        }
        let me = (d.player & 1) as usize;
        let prompt = ctx.prompt.ok_or("remote seat without a prompt")?;
        self.requests += 1;
        let c = self.broker.ask(json!({"job": self.job, "idx": self.idx, "seat": self.seat, "policy": self.policy,
            "lp": [ctx.view.lp[me], ctx.view.lp[1 - me]], "turn": ctx.view.turn, "prompt": prompt, "n": d.choices.len(),
            "counts": ctx.counts}));
        if c >= d.choices.len() {
            return Err(format!("worker answered {c} for {} options", d.choices.len()));
        }
        Ok(c)
    }
}

/// Round 4 DAgger: a remote seat (policy "shadow:<tag>", the worker sees <tag>) with a WindBot watching it.
pub struct ShadowAgent<'a>(pub RemoteAgent<'a>, pub crate::windbot::WindBot);

impl Agent for ShadowAgent<'_> {
    fn wants_board(&self) -> bool {
        true
    }
    fn choose(&mut self, d: &Decision, ctx: &Ctx, rng: &mut Rng) -> Result<usize, String> {
        self.0.choose(d, ctx, rng)
    }
    fn game_id(&self) -> Option<(u64, u32)> {
        self.0.game_id()
    }
    fn shadow(&mut self) -> Option<&mut crate::windbot::WindBot> {
        Some(&mut self.1)
    }
}

/// Round 5: a WindBot deck by name (AI_<name>.ydk in YGO_DECK_DIR), checked for card data and scripts; cached.
pub fn named_deck(lib: &Library, cache: &Mutex<HashMap<String, Arc<duel::Deck>>>, name: &str) -> Result<Arc<duel::Deck>, String> {
    if let Some(d) = cache.lock().unwrap().get(name) {
        return Ok(d.clone());
    }
    let dir = std::env::var("YGO_DECK_DIR").unwrap_or("/srv/ygo/windbot/Decks".into());
    let d = crate::data::read_ydk(&PathBuf::from(dir).join(format!("AI_{name}.ydk")))?;
    let missing: Vec<u32> = d.0.iter().chain(&d.1).copied()
        .filter(|c| !lib.cards.contains_key(c) || !lib.has_script(&format!("c{c}.lua"))).collect();
    if d.0.is_empty() || !missing.is_empty() {
        return Err(format!("deck {name}: {} main cards, missing data/script {missing:?}", d.0.len()));
    }
    let d = Arc::new(d);
    cache.lock().unwrap().insert(name.to_string(), d.clone());
    Ok(d)
}

/// Replay disk budget: YGO_REPLAY_MAX_FILES (default 10000) files and YGO_REPLAY_MAX_BYTES (default 10 GiB).
/// Once the directory is over either, new replays are skipped until the operator prunes it.
fn replay_budget_left(dir: &Path) -> bool {
    let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (max_files, max_bytes) = (env("YGO_REPLAY_MAX_FILES", 10_000), env("YGO_REPLAY_MAX_BYTES", 10 << 30));
    let (mut files, mut bytes) = (0u64, 0u64);
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        files += 1;
        bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
    }
    files < max_files && bytes < max_bytes
}

/// The replay path for a job name, or None if it would not land directly in `dir`.
fn replay_path(dir: &Path, file: &str) -> Option<PathBuf> {
    let p = dir.join(file);
    (!file.contains('/') && p.parent() == Some(dir)).then_some(p)
}

/// Game threads: each takes the next queued game, plays it, and posts the result.
pub fn run_games(broker: Broker, lib: Arc<Library>, deck: Arc<(Vec<u32>, Vec<u32>)>, threads: usize, replay_dir: PathBuf) {
    let _ = std::fs::create_dir_all(&replay_dir);
    let cache: Arc<Mutex<HashMap<String, Arc<duel::Deck>>>> = Arc::new(Mutex::new(HashMap::new()));
    for _ in 0..threads {
        let (b, lib, deck, dir, cache) = (broker.clone(), lib.clone(), deck.clone(), replay_dir.clone(), cache.clone());
        std::thread::spawn(move || loop {
            let g = b.take_game();
            let started = Instant::now();
            let mut seat_error: Option<String> = None;
            let mut seat_decks: Vec<Arc<duel::Deck>> = Vec::new();
            for k in 0..2 {
                match &g.decks[k] {
                    None => seat_decks.push(deck.clone()),
                    Some(n) => match named_deck(&lib, &cache, n) {
                        Ok(d) => seat_decks.push(d),
                        Err(e) => {
                            seat_error = Some(e);
                            seat_decks.push(deck.clone());
                        }
                    },
                }
            }
            let wb_deck = |k: usize| g.decks[k].clone().unwrap_or("CyberDragon".into());
            let mut agents: Vec<Box<dyn Agent + '_>> = Vec::new();
            for (k, pol) in g.seats.iter().enumerate() {
                if pol == "random" {
                    agents.push(Box::new(RandomAgent));
                } else if pol == "windbot" || pol == "windbot-labels" {
                    let server = std::env::var("YGO_WINDBOT").unwrap_or("127.0.0.1:2399".into());
                    match crate::windbot::WindBot::connect(&server, &wb_deck(k), k as u8, "WindBot") {
                        Ok(w) => agents.push(Box::new(WindBotAgent(w, pol == "windbot-labels"))),
                        Err(e) => {
                            seat_error = Some(e);
                            agents.push(Box::new(RandomAgent));
                        }
                    }
                } else if let Some(inner) = pol.strip_prefix("shadow:") {
                    let server = std::env::var("YGO_WINDBOT").unwrap_or("127.0.0.1:2399".into());
                    let remote = RemoteAgent { broker: &b, job: g.job, idx: g.idx, seat: k as u8, policy: inner.to_string(), requests: 0 };
                    match crate::windbot::WindBot::connect(&server, &wb_deck(k), k as u8, "Shadow") {
                        Ok(w) => agents.push(Box::new(ShadowAgent(remote, w))),
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
            let header = json!({"job": g.job, "game": g.idx, "name": &*g.name, "seats": g.seats});
            let publish = |mut v: Value| {
                for (k, x) in header.as_object().unwrap() {
                    v[k] = x.clone();
                }
                b.publish(g.job, g.idx, v);
            };
            let live: Option<&dyn Fn(Value)> = if g.live { Some(&publish) } else { None };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &seat_error {
                Some(e) => Err(e.clone()),
                None => duel::play2(&lib, [&seat_decks[0], &seat_decks[1]], g.seed, &mut [a0[0].as_mut(), rest[0].as_mut()], g.replay, live),
            }));
            let res = match res {
                Ok(r) => r,
                Err(p) => Err(format!(
                    "panic: {}",
                    p.downcast_ref::<String>().cloned().or(p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
                )),
            };
            let mut out = json!({"job": g.job, "idx": g.idx, "name": &*g.name, "seed": g.seed, "seats": g.seats,
                "seconds": started.elapsed().as_secs_f64()});
            if g.decks.iter().any(|d| d.is_some()) {
                out["decks"] = json!(g.decks);
            }
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
                    out["responses"] = json!(r.responses);
                    if g.seats.iter().any(|p| p.starts_with("shadow:")) {
                        out["shadow"] = json!(r.shadow); // per seat: [asked, advice among the options, model chose otherwise]
                    }
                    if g.replay {
                        let file = format!("{}-j{}-g{}.json", g.name, g.job, g.idx);
                        let rep = json!({"name": &*g.name, "job": g.job, "game": g.idx, "seed": g.seed, "seats": g.seats, "decks": g.decks,
                            "winner": winner, "reason": reason, "turns": r.turns, "lp": r.lp, "names": r.names, "steps": r.steps,
                            "finished": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)});
                        match replay_path(&dir, &file) {
                            Some(_) if !replay_budget_left(&dir) => out["replay_skipped"] = json!("replay directory over its budget"),
                            Some(p) => {
                                if std::fs::write(p, rep.to_string()).is_ok() {
                                    out["replay"] = json!(file);
                                }
                            }
                            None => out["replay_skipped"] = json!("bad replay name"),
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
            b.finish_game(g.job, g.owner.clone(), out, error);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn broker() -> Broker {
        Broker(Arc::new((Mutex::new(State::default()), Condvar::new())))
    }

    #[test]
    fn job_names_that_leave_the_replay_dir_are_rejected() {
        let b = broker();
        for bad in ["../x", "/abs/x", "a/b", "", &"n".repeat(MAX_NAME + 1)] {
            assert!(b.add_job(&json!({"name": bad, "games": [{"seats": ["random", "random"]}]})).is_err(), "{bad}");
        }
        assert!(b.add_job(&json!({"name": "eval-step3-windbot", "games": [{"seats": ["random", "random"]}]})).is_ok());
        let dir = Path::new("/srv/ygo/replays");
        assert!(replay_path(dir, "../x-j1-g0.json").is_none());
        assert!(replay_path(dir, "/abs/x-j1-g0.json").is_none());
        assert_eq!(replay_path(dir, "ok-j1-g0.json"), Some(dir.join("ok-j1-g0.json")));
    }

    #[test]
    fn job_size_and_token_are_checked() {
        let b = broker();
        let games: Vec<Value> = (0..=MAX_GAMES_PER_JOB).map(|_| json!({"seats": ["random", "random"]})).collect();
        assert!(b.add_job(&json!({"games": games})).is_err());
        assert!(b.add_job(&json!({"token": "short", "games": []})).is_err());
        let (_, t) = b.add_job(&json!({"games": []})).unwrap();
        assert!(valid_token(&t));
        let (_, t) = b.add_job(&json!({"token": "cccccccccccccccc-_", "games": []})).unwrap();
        assert_eq!(&*t, "cccccccccccccccc-_");
    }

    #[test]
    fn answers_need_the_current_lease() {
        let b = broker();
        let bc = b.clone();
        let asker = std::thread::spawn(move || bc.ask(json!({"job": 1, "n": 3})));
        let reqs = loop {
            let v = b.next(8, 1, 0, None);
            if !v["requests"].as_array().unwrap().is_empty() {
                break v["requests"].clone();
            }
        };
        let (id, lease) = (reqs[0]["id"].as_u64().unwrap(), reqs[0]["lease"].as_str().unwrap().to_string());
        assert!(!b.answer(id, "not-the-lease", 0));
        assert!(!b.answer(id + 1, &lease, 0));
        assert!(b.answer(id, &lease, 2));
        assert!(!b.answer(id, &lease, 1), "a second answer is refused");
        assert_eq!(asker.join().unwrap(), 2);
    }

    /// One raw HTTP exchange through Broker::handle.
    fn http(b: &Broker, req: &str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let bc = b.clone();
        let t = std::thread::spawn(move || bc.handle(s));
        c.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        c.read_to_string(&mut out).unwrap();
        t.join().unwrap().unwrap();
        out
    }

    #[test]
    fn http_job_token_and_answer_lease() {
        let b = broker();
        let body = r#"{"name":"../x","games":[]}"#;
        assert!(http(&b, &format!("POST /job HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len())).starts_with("HTTP/1.1 400"));
        let body = r#"{"name":"t","token":"aaaaaaaaaaaaaaaaaaaa","games":[]}"#;
        let r = http(&b, &format!("POST /job HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()));
        assert!(r.starts_with("HTTP/1.1 200") && r.contains(r#""token":"aaaaaaaaaaaaaaaaaaaa""#), "{r}");
        b.0 .0.lock().unwrap().running = 1;
        b.finish_game(5, "aaaaaaaaaaaaaaaaaaaa".into(), json!({"job": 5}), false);
        assert!(http(&b, "GET /next?wait=0 HTTP/1.1\r\n\r\n").contains(r#""results":[]"#));
        assert!(http(&b, "GET /next?wait=0 HTTP/1.1\r\nX-Job-Token: aaaaaaaaaaaaaaaaaaaa\r\n\r\n").contains(r#""results":[{"job":5}]"#));
        let body = r#"[{"id":1,"choice":0}]"#;
        assert!(http(&b, &format!("POST /answer HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len())).starts_with("HTTP/1.1 400"));
        let body = r#"[{"id":1,"lease":"x","choice":0}]"#;
        let r = http(&b, &format!("POST /answer HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()));
        assert!(r.starts_with("HTTP/1.1 409") && r.contains(r#""rejected":[1]"#), "{r}");
    }

    #[test]
    fn results_go_only_to_the_job_token() {
        let b = broker();
        b.0 .0.lock().unwrap().running = 1;
        b.finish_game(7, "aaaaaaaaaaaaaaaaaaaa".into(), json!({"job": 7}), false);
        assert!(b.next(8, 0, 0, None)["results"].as_array().unwrap().is_empty());
        assert!(b.next(8, 0, 0, Some("bbbbbbbbbbbbbbbbbbbb"))["results"].as_array().unwrap().is_empty());
        assert_eq!(b.next(8, 0, 0, Some("aaaaaaaaaaaaaaaaaaaa"))["results"], json!([{"job": 7}]));
        assert!(b.next(8, 0, 0, Some("aaaaaaaaaaaaaaaaaaaa"))["results"].as_array().unwrap().is_empty());
    }
}
