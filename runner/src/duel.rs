//! One duel on the engine: create, load decks, step messages, hand each decision to an agent.
use crate::board::{self, Board};
use crate::data::{self, Library};
use crate::decide::{self, Decision, Reader};
use crate::ffi::*;
use crate::rng::Rng;
use std::ffi::CString;
use std::os::raw::c_void;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const TURN_LIMIT: u32 = 150;
pub const DECISION_LIMIT: u32 = 20_000;
pub const RETRY_LIMIT: u32 = 50;
/// A player making this many decisions in one turn is looping (a fixed policy can re-pick the same
/// activate-then-cancel path forever); the rest of that turn is played randomly and counted in loop_breaks.
pub const TURN_DECISION_LIMIT: u32 = 100;

/// Turn, phase and life points, tracked from engine messages.
pub struct View {
    pub turn: u32,
    pub turn_player: u8,
    pub phase: u16,
    pub lp: [i64; 2],
}

/// What an agent gets for one decision. `prompt` is the full text (board, decision, lettered options)
/// when the agent asked for the board.
pub struct Ctx<'a> {
    pub view: &'a View,
    pub prompt: Option<&'a str>,
}

pub trait Agent {
    /// Index into d.choices.
    fn choose(&mut self, d: &Decision, ctx: &Ctx, rng: &mut Rng) -> Result<usize, String>;
    fn wants_board(&self) -> bool {
        false
    }
}

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn choose(&mut self, d: &Decision, _: &Ctx, rng: &mut Rng) -> Result<usize, String> {
        Ok(rng.below(d.choices.len()))
    }
}

#[derive(Debug)]
pub enum Outcome {
    Win { player: u8, reason: u8 },
    Draw { reason: &'static str },
}

pub struct GameResult {
    pub outcome: Outcome,
    pub turns: u32,
    pub decisions: u32,
    pub retries: u32,
    pub script_errors: usize,
    pub lp: [i64; 2],
    pub max_choices: usize,
    pub loop_breaks: u32,
    /// One entry per decision when recording: options, the choice made, and the board when it changed.
    pub steps: Vec<Value>,
    pub names: BTreeMap<u32, String>,
}

struct DuelHandle(OCG_Duel);
impl Drop for DuelHandle {
    fn drop(&mut self) {
        unsafe { OCG_DestroyDuel(self.0) }
    }
}

pub fn play(
    lib: &Library, deck: &(Vec<u32>, Vec<u32>), seed: u64, agents: &mut [&mut dyn Agent; 2], record: bool,
    live: Option<&dyn Fn(Value)>,
) -> Result<GameResult, String> {
    let script_errors = AtomicUsize::new(0);
    let mut rng = Rng::new(seed);
    let player = OCG_Player { starting_lp: 8000, starting_draw_count: 5, draw_count_per_turn: 1 };
    let payload = lib as *const Library as *mut c_void;
    let opts = OCG_DuelOptions {
        seed: [rng.next() | 1, rng.next(), rng.next(), rng.next()],
        flags: DUEL_MODE_MR5,
        team1: player,
        team2: player,
        card_reader: data::card_reader,
        payload1: payload,
        script_reader: data::script_reader,
        payload2: payload,
        log_handler: data::log_handler,
        payload3: &script_errors as *const AtomicUsize as *mut c_void,
        card_reader_done: data::card_reader_done,
        payload4: std::ptr::null_mut(),
        enable_unsafe_libraries: 0,
    };
    let mut raw: OCG_Duel = std::ptr::null_mut();
    let rc = unsafe { OCG_CreateDuel(&mut raw, &opts) };
    if rc != OCG_DUEL_CREATION_SUCCESS {
        return Err(format!("OCG_CreateDuel failed: {rc}"));
    }
    let duel = DuelHandle(raw);
    for name in ["constant.lua", "utility.lua"] {
        let c = CString::new(name).unwrap();
        if data::script_reader(payload, duel.0, c.as_ptr()) == 0 {
            return Err(format!("failed to load {name}"));
        }
    }
    for team in 0..2u8 {
        let mut main = deck.0.clone();
        rng.shuffle(&mut main);
        for (codes, loc) in [(&main, LOCATION_DECK), (&deck.1, LOCATION_EXTRA)] {
            for &code in codes.iter() {
                let info = OCG_NewCardInfo { team, duelist: 0, code, con: team, loc, seq: 0, pos: POS_FACEDOWN_DEFENSE };
                unsafe { OCG_DuelNewCard(duel.0, &info) };
            }
        }
    }
    unsafe { OCG_StartDuel(duel.0) };

    let mut view = View { turn: 0, turn_player: 0, phase: 0, lp: [8000, 8000] };
    let (mut decisions, mut retries, mut retry_streak, mut max_choices) = (0u32, 0u32, 0u32, 0usize);
    let mut steps: Vec<Value> = Vec::new();
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut last_board = String::new();
    let mut last: Option<Vec<u8>> = None; // last decision message, re-asked after MSG_RETRY
    let mut unselected: Vec<decide::CardKey> = Vec::new();
    let (mut turn_seen, mut turn_decisions, mut loop_breaks) = (0u32, [0u32; 2], 0u32);
    let finish = |outcome, view: &View, decisions, retries, max_choices, steps, names, loop_breaks| GameResult {
        outcome,
        turns: view.turn,
        decisions,
        retries,
        script_errors: script_errors.load(Ordering::Relaxed),
        lp: view.lp,
        max_choices,
        steps,
        names,
        loop_breaks,
    };
    loop {
        let status = unsafe { OCG_DuelProcess(duel.0) };
        let mut len = 0u32;
        let ptr = unsafe { OCG_DuelGetMessage(duel.0, &mut len) } as *const u8;
        let buf = if len == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec() };
        let mut pending: Option<Vec<u8>> = None;
        let mut retry = false;
        let mut pos = 0usize;
        while pos + 4 <= buf.len() {
            let n = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + n > buf.len() || n == 0 {
                return Err(format!("bad message framing at {pos}"));
            }
            let msg = &buf[pos..pos + n];
            pos += n;
            let mut r = Reader::new(&msg[1..]);
            match msg[0] {
                decide::MSG_RETRY => retry = true,
                decide::MSG_WIN => {
                    let (p, reason) = (r.u8()?, r.u8()?);
                    let outcome = if p > 1 { Outcome::Draw { reason: "engine draw" } } else { Outcome::Win { player: p, reason } };
                    if let Some(f) = live {
                        let mut n = BTreeMap::new();
                        let b = Board::read(duel.0, view.turn, view.turn_player, view.phase, view.lp)?.to_json(lib, &mut n);
                        f(json!({"i": decisions, "end": true, "winner": if p > 1 { -1 } else { p as i32 }, "board": b, "names": n}));
                    }
                    if record {
                        let b = Board::read(duel.0, view.turn, view.turn_player, view.phase, view.lp)?;
                        steps.push(json!({"i": decisions, "end": true, "winner": if p > 1 { -1 } else { p as i32 }, "board": b.to_json(lib, &mut names)}));
                    }
                    return Ok(finish(outcome, &view, decisions, retries, max_choices, steps, names, loop_breaks));
                }
                decide::MSG_NEW_TURN => {
                    view.turn += 1;
                    view.turn_player = r.u8()?;
                    if view.turn > TURN_LIMIT {
                        return Ok(finish(Outcome::Draw { reason: "turn limit" }, &view, decisions, retries, max_choices, steps, names, loop_breaks));
                    }
                }
                decide::MSG_NEW_PHASE => view.phase = r.u16()?,
                decide::MSG_LPUPDATE => {
                    let p = r.u8()? as usize & 1;
                    view.lp[p] = r.u32()? as i64;
                }
                decide::MSG_DAMAGE | decide::MSG_PAY_LPCOST => {
                    let p = r.u8()? as usize & 1;
                    view.lp[p] = (view.lp[p] - r.u32()? as i64).max(0); // the engine floors LP at 0 too
                }
                decide::MSG_RECOVER => {
                    let p = r.u8()? as usize & 1;
                    view.lp[p] += r.u32()? as i64;
                }
                t if decide::is_decision(t) => pending = Some(msg.to_vec()),
                _ => {}
            }
        }
        match status {
            OCG_DUEL_STATUS_END => return Err("duel ended without MSG_WIN".into()),
            OCG_DUEL_STATUS_CONTINUE => continue,
            OCG_DUEL_STATUS_AWAITING => {}
            s => return Err(format!("unknown duel status {s}")),
        }
        let msg = match (pending, retry) {
            (Some(m), _) => {
                retry_streak = 0;
                m
            }
            (None, true) => {
                retries += 1;
                retry_streak += 1;
                if retry_streak > RETRY_LIMIT {
                    return Err(format!("engine rejected {RETRY_LIMIT} answers in a row (msg {:?})", last.as_ref().map(|m| m[0])));
                }
                last.clone().ok_or("MSG_RETRY before any decision")?
            }
            (None, false) => return Err("engine awaiting a response but sent no decision message".into()),
        };
        let d = decide::parse(&msg, lib, &mut rng, &unselected)?;
        if d.msg != 26 {
            unselected.clear();
        }
        if d.choices.is_empty() {
            return Err(format!("no legal choice found for message {} ({})", d.msg, d.kind));
        }
        decisions += 1;
        if decisions > DECISION_LIMIT {
            return Err(format!("decision limit {DECISION_LIMIT} hit on turn {}", view.turn));
        }
        if d.choices.len() > board::LETTERS.len() {
            return Err(format!("{} choices for {} (only {} letters)", d.choices.len(), d.kind, board::LETTERS.len()));
        }
        max_choices = max_choices.max(d.choices.len());
        let who = (d.player & 1) as usize;
        let need_board = record || live.is_some() || (d.choices.len() > 1 && agents[who].wants_board());
        let board = if need_board { Some(Board::read(duel.0, view.turn, view.turn_player, view.phase, view.lp)?) } else { None };
        let options: Vec<String> = d.choices.iter().map(|c| c.text.clone()).collect();
        let prompt = board.as_ref().map(|b| board::prompt(&b.describe(lib, d.player), d.kind, &options));
        if view.turn != turn_seen {
            turn_seen = view.turn;
            turn_decisions = [0, 0];
        }
        turn_decisions[who] += 1;
        let looping = turn_decisions[who] > TURN_DECISION_LIMIT;
        if turn_decisions[who] == TURN_DECISION_LIMIT + 1 {
            loop_breaks += 1;
        }
        let i = if looping {
            rng.below(d.choices.len())
        } else {
            agents[who].choose(&d, &Ctx { view: &view, prompt: prompt.as_deref() }, &mut rng)?
        };
        if let Some(f) = live {
            let mut n = BTreeMap::new();
            let b = board.as_ref().unwrap().to_json(lib, &mut n);
            f(json!({"i": decisions, "player": d.player & 1, "kind": d.kind, "options": options, "chosen": i, "board": b, "names": n}));
        }
        if let Some(Some(k)) = d.unselects.get(i) {
            unselected.push(*k);
        }
        let c = d.choices.get(i).ok_or_else(|| format!("agent chose {i} of {}", d.choices.len()))?;
        if record {
            let b = board.as_ref().unwrap().to_json(lib, &mut names);
            let text = b.to_string();
            let mut step = json!({"i": decisions, "player": d.player & 1, "kind": d.kind, "options": options, "chosen": i});
            if text != last_board {
                step["board"] = b;
                last_board = text;
            }
            steps.push(step);
        }
        unsafe { OCG_DuelSetResponse(duel.0, c.resp.as_ptr() as *const c_void, c.resp.len() as u32) };
        last = Some(msg);
    }
}
