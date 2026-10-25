//! ygo-runner: Cyber Dragon mirror games on EDOPro's engine.
//!   ygo-runner random [--games 100] [--seed 1] [--replays N]   random vs random; first N games saved as replays
//!   ygo-runner prompt [--seed 1] [--turn 3]                    print one real model prompt from a random game
//!   ygo-runner serve [--listen 0.0.0.0:8765] [--threads 48]    game server for the GPU worker (broker.rs)
//!   ygo-runner optcheck [--decisions 500]                       round 4: sampled options re-parse identically
//!   ygo-runner leakcheck --opp-deck X.ydk [--prompts 2000]      round 4: no hidden opponent card in player 0's prompts
//! Paths: --deck, --scripts, --cdb (repeatable), --log (one JSON line per game), --replay-dir.
mod board;
mod broker;
mod data;
mod decide;
mod duel;
mod ffi;
mod history;
mod rng;
mod windbot;

use duel::{Agent, Ctx, Outcome, RandomAgent};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

/// Always the first (or last) option: the worst case for loops, like a stuck greedy model.
struct FixedAgent(bool);

impl Agent for FixedAgent {
    fn choose(&mut self, d: &decide::Decision, _: &Ctx, _: &mut rng::Rng) -> Result<usize, String> {
        Ok(if self.0 { d.choices.len() - 1 } else { 0 })
    }
}

/// Plays randomly, but asks for the board so the prompt can be printed once.
struct PromptProbe {
    min_turn: u32,
    min_choices: usize,
    printed: bool,
}

impl Agent for PromptProbe {
    fn wants_board(&self) -> bool {
        !self.printed
    }
    fn choose(&mut self, d: &decide::Decision, ctx: &Ctx, rng: &mut rng::Rng) -> Result<usize, String> {
        if !self.printed && ctx.view.turn >= self.min_turn && d.choices.len() >= self.min_choices {
            if let Some(p) = ctx.prompt {
                println!("----- prompt (player {}, {} options) -----\n{p}\n-----", d.player & 1, d.choices.len());
                self.printed = true;
            }
        }
        Ok(rng.below(d.choices.len()))
    }
}

/// Round 4 leak check: plays randomly as player 0 and scans each of its prompts for the names of the opponent's
/// hidden cards (hand not revealed, face-down field and banished). The opponent plays a deck with no card names in
/// common, so a name can only get there through a leak, unless that card was public earlier (seen face-up, in a
/// GY or banished face-up, or revealed, on any board snapshot so far).
struct LeakProbe {
    opp_names: Vec<String>,
    /// Names, not codes: a deck can hold two prints of one card (different passcodes, one name).
    ever_public: std::collections::HashSet<String>,
    prompts: usize,
    want: usize,
    leaks: usize,
    control_fired: Option<bool>,
    examples: Vec<String>,
}

impl LeakProbe {
    /// The opponent-deck card names a text mentions; at each position the longest name wins.
    fn mentioned(&self, text: &str) -> std::collections::HashSet<String> {
        let mut out = std::collections::HashSet::new();
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let hit = self.opp_names.iter().filter(|n| b[i..].starts_with(n.as_bytes())).max_by_key(|n| n.len());
            match hit {
                Some(n) => {
                    out.insert(n.clone());
                    i += n.len();
                }
                None => i += 1,
            }
        }
        out
    }
}

impl Agent for LeakProbe {
    fn wants_board(&self) -> bool {
        true
    }
    fn choose(&mut self, d: &decide::Decision, ctx: &Ctx, rng: &mut rng::Rng) -> Result<usize, String> {
        let (Some(p), Some(b)) = (ctx.prompt, ctx.board) else { return Ok(rng.below(d.choices.len())) };
        let lib = LIB.get().unwrap();
        let o = &b.sides[1 - (d.player & 1) as usize];
        let up = |c: &&board::CardView| c.pos & 0x5 != 0;
        for c in o.mzone.iter().flatten().chain(o.szone.iter().flatten()).filter(up)
            .chain(o.grave.iter()).chain(o.banished.iter().filter(up)).chain(o.hand.iter().filter(|c| c.public)) {
            self.ever_public.insert(lib.name(c.code));
        }
        self.ever_public.extend(ctx.revealed.iter().chain(ctx.seen_public).map(|c| lib.name(*c)));
        let m = &b.sides[(d.player & 1) as usize]; // an opponent's card on my side (e.g. Lava Golem) is mine to see
        for c in m.hand.iter().chain(m.mzone.iter().flatten()).chain(m.szone.iter().flatten()).chain(m.grave.iter()).chain(m.banished.iter()) {
            self.ever_public.insert(lib.name(c.code));
        }
        let hidden: Vec<u32> = o.hand.iter().filter(|c| !c.public)
            .chain(o.mzone.iter().flatten().chain(o.szone.iter().flatten()).chain(o.banished.iter()).filter(|c| c.pos & 0x5 == 0))
            .map(|c| c.code).collect();
        let names: Vec<String> = hidden.iter().map(|c| lib.name(*c)).filter(|n| !self.ever_public.contains(n)).collect();
        if d.choices.len() >= 2 && self.prompts < self.want {
            let seen = self.mentioned(p);
            let leaked: Vec<&String> = names.iter().filter(|n| seen.contains(*n)).collect();
            self.prompts += 1;
            if !leaked.is_empty() {
                self.leaks += 1;
                if self.examples.len() < 3 {
                    self.examples.push(format!("{leaked:?} in:\n{p}"));
                }
            }
            if self.control_fired.is_none() && !names.is_empty() {
                // positive control: the same prompt with one hidden card's name planted must be flagged
                let planted = format!("{p}\nOpponent's hand: {}\n", names[0]);
                self.control_fired = Some(names.iter().any(|n| self.mentioned(&planted).contains(n)));
            }
        }
        Ok(rng.below(d.choices.len()))
    }
}

static LIB: std::sync::OnceLock<data::Library> = std::sync::OnceLock::new();

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let games: u32 = flag("--games").map(|g| g.parse().expect("--games")).unwrap_or(if mode == "remote" { 20 } else { 100 });
    let seed: u64 = flag("--seed").map(|s| s.parse().expect("--seed")).unwrap_or(1);
    let deck_path = PathBuf::from(flag("--deck").unwrap_or("/srv/ygo/CyberDragon.ydk".into()));
    let scripts = PathBuf::from(flag("--scripts").unwrap_or("/srv/ygo/scripts".into()));
    let mut cdbs: Vec<PathBuf> =
        args.windows(2).filter(|w| w[0] == "--cdb").map(|w| PathBuf::from(&w[1])).collect();
    if cdbs.is_empty() {
        cdbs.push("/srv/ygo/cdb/cards.cdb".into());
    }
    let log_path = PathBuf::from(flag("--log").unwrap_or(format!("/srv/ygo/logs/{mode}-games.jsonl")));
    let replay_dir = PathBuf::from(flag("--replay-dir").unwrap_or("/srv/ygo/replays".into()));
    let fixed = flag("--fixed").unwrap_or_default(); // test only: first | last | mixed
    let replays: u32 = flag("--replays").map(|g| g.parse().expect("--replays")).unwrap_or(0);
    if !["random", "prompt", "serve", "optcheck", "leakcheck"].contains(&mode.as_str()) {
        eprintln!("usage: ygo-runner random|prompt|serve|optcheck|leakcheck [--games N] [--seed S] [--listen ADDR]");
        std::process::exit(2);
    }

    let (mut maj, mut min) = (0, 0);
    unsafe { ffi::OCG_GetVersion(&mut maj, &mut min) };
    let lib = data::Library::load(&cdbs, &scripts).unwrap_or_else(|e| panic!("card data: {e}"));
    let deck = data::read_ydk(&deck_path).unwrap_or_else(|e| panic!("deck: {e}"));
    let missing: Vec<u32> = deck.0.iter().chain(&deck.1).copied()
        .filter(|c| !lib.cards.contains_key(c) || !lib.has_script(&format!("c{c}.lua"))).collect();
    println!(
        "engine ocgcore {maj}.{min}; cards {}; deck {} main + {} extra ({}); missing {:?}",
        lib.cards.len(), deck.0.len(), deck.1.len(), deck_path.display(), missing
    );
    if !missing.is_empty() {
        eprintln!("deck has cards without data or script");
        std::process::exit(1);
    }

    if mode == "optcheck" {
        // Sampled options (several cards, zones, tributes, orders, counters, sums, declares) recorded in random games,
        // then parsed again from the same game seed and message: they must come out identical. The old way (one
        // shared game RNG) is shown as the control: re-parsing with the RNG advanced elsewhere.
        let want: usize = flag("--decisions").map(|g| g.parse().expect("--decisions")).unwrap_or(500);
        const SAMPLED: &[u8] = &[15, 18, 20, 21, 22, 23, 24, 25, 140, 141, 142];
        *duel::OPT_RECORD.lock().unwrap() = Some(Vec::new());
        let mut g = 0u64;
        while duel::OPT_RECORD.lock().unwrap().as_ref().unwrap().iter().filter(|r| SAMPLED.contains(&r.1[0]) && r.4.len() > 1).count() < want {
            let (mut a0, mut a1) = (RandomAgent, RandomAgent);
            if let Err(e) = duel::play(&lib, &deck, seed.wrapping_mul(1_000_003).wrapping_add(g), &mut [&mut a0, &mut a1], false, None) {
                eprintln!("game error: {e}");
                std::process::exit(1);
            }
            g += 1;
        }
        let rec = duel::OPT_RECORD.lock().unwrap().take().unwrap();
        let picked: Vec<_> = rec.into_iter().filter(|r| SAMPLED.contains(&r.1[0]) && r.4.len() > 1).take(want).collect();
        let (mut same, mut old_same) = (0, 0);
        let mut shared = rng::Rng::new(seed ^ 0x5eed);
        for (gs, msg, unsel, asked, opts) in &picked {
            let public = |c: u8, l: u8, q: u32| asked.iter().find(|(k, _)| *k == (c, l, q)).map(|(_, v)| *v).unwrap_or(false);
            let again = decide::parse(msg, &lib, &mut rng::Rng::new(duel::option_seed(*gs, msg)), unsel, &public).expect("re-parse");
            let again: Vec<(String, Vec<u8>)> = again.choices.iter().map(|c| (c.text.clone(), c.resp.clone())).collect();
            same += (&again == opts) as usize;
            let old = decide::parse(msg, &lib, &mut shared, unsel, &public).expect("re-parse");
            let old: Vec<(String, Vec<u8>)> = old.choices.iter().map(|c| (c.text.clone(), c.resp.clone())).collect();
            old_same += (&old == opts) as usize;
        }
        println!("optcheck games={g} decisions={} (sampled kinds with 2+ options)", picked.len());
        println!("options_stable={same}/{} shared_rng_control_stable={old_same}/{}", picked.len(), picked.len());
        std::process::exit(if same == picked.len() { 0 } else { 1 });
    }

    if mode == "leakcheck" {
        let want: usize = flag("--prompts").map(|g| g.parse().expect("--prompts")).unwrap_or(2000);
        let opp_path = PathBuf::from(flag("--opp-deck").expect("--opp-deck"));
        let opp = data::read_ydk(&opp_path).unwrap_or_else(|e| panic!("opp deck: {e}"));
        let mine: std::collections::HashSet<String> = deck.0.iter().chain(&deck.1).map(|c| lib.name(*c)).collect();
        let mut opp_names: Vec<String> = opp.0.iter().chain(&opp.1).map(|c| lib.name(*c)).collect();
        opp_names.sort();
        opp_names.dedup();
        let shared: Vec<&String> = opp_names.iter().filter(|n| mine.iter().any(|m| m.contains(n.as_str()) || n.contains(m.as_str()))).collect();
        let missing: Vec<u32> = opp.0.iter().chain(&opp.1).copied()
            .filter(|c| !lib.cards.contains_key(c) || !lib.has_script(&format!("c{c}.lua"))).collect();
        if !shared.is_empty() || !missing.is_empty() {
            eprintln!("opponent deck unusable: names shared with mine {shared:?}, missing data/script {missing:?}");
            std::process::exit(1);
        }
        let lib = LIB.get_or_init(|| lib);
        duel::TRACE_PUBLIC.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut probe = LeakProbe { opp_names, ever_public: Default::default(), prompts: 0, want, leaks: 0, control_fired: None, examples: vec![] };
        let (mut g, mut errors) = (0u64, 0);
        while probe.prompts < want {
            probe.ever_public.clear();
            let mut other = RandomAgent;
            if let Err(e) = duel::play2(lib, [&deck, &opp], seed.wrapping_mul(1_000_003).wrapping_add(g), &mut [&mut probe, &mut other], false, None) {
                eprintln!("game error: {e}");
                errors += 1;
            }
            g += 1;
        }
        for e in &probe.examples {
            println!("LEAK {e}\n-----");
        }
        let fired = probe.control_fired.unwrap_or(false) as u8;
        println!("leakcheck games={g} errors={errors} opp_deck={}", opp_path.display());
        println!("leakcheck prompts={} leaks={} control_fired={fired}", probe.prompts, probe.leaks);
        std::process::exit(if probe.leaks == 0 && fired == 1 { 0 } else { 1 });
    }

    if mode == "prompt" {
        let turn: u32 = flag("--turn").map(|t| t.parse().expect("--turn")).unwrap_or(3);
        for g in 0..50u64 {
            let mut probe = PromptProbe { min_turn: turn, min_choices: 3, printed: false };
            let mut other = RandomAgent;
            let r = duel::play(&lib, &deck, seed.wrapping_mul(1_000_003).wrapping_add(g), &mut [&mut probe, &mut other], false, None);
            if let Err(e) = r {
                eprintln!("game error: {e}");
                std::process::exit(1);
            }
            if probe.printed {
                return;
            }
        }
        eprintln!("no decision with 3+ options on turn {turn}+ in 50 games");
        std::process::exit(1);
    }

    if mode == "serve" {
        let addr = flag("--listen").unwrap_or("0.0.0.0:8765".into());
        let threads: usize = flag("--threads").map(|t| t.parse().expect("--threads")).unwrap_or(48);
        let b = broker::Broker::start(&addr).unwrap_or_else(|e| panic!("{e}"));
        println!("game server on {addr}, {threads} game threads, replays in {}", replay_dir.display());
        broker::run_games(b, Arc::new(lib), Arc::new(deck), threads, replay_dir);
        loop {
            std::thread::park();
        }
    }

    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("log file");
    let (mut p0, mut p1, mut draws, mut errors, mut script_errors, mut max_choices) = (0, 0, 0, 0, 0usize, 0usize);
    let _ = std::fs::create_dir_all(&replay_dir);
    let started = std::time::Instant::now();
    for g in 0..games {
        let game_seed = seed.wrapping_mul(1_000_003).wrapping_add(g as u64);
        let (mut a0, mut a1) = (RandomAgent, RandomAgent);
        let (mut f0, mut f1) = (FixedAgent(fixed == "last"), FixedAgent(fixed == "last" || fixed == "mixed"));
        let record = g < replays;
        let res = catch_unwind(AssertUnwindSafe(|| {
            if fixed.is_empty() {
                duel::play(&lib, &deck, game_seed, &mut [&mut a0, &mut a1], record, None)
            } else {
                duel::play(&lib, &deck, game_seed, &mut [&mut f0, &mut f1], record, None)
            }
        }));
        let res = match res {
            Ok(r) => r,
            Err(p) => Err(format!(
                "panic: {}",
                p.downcast_ref::<String>().cloned().or(p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
            )),
        };
        let line = match &res {
            Ok(r) => {
                script_errors += r.script_errors;
                max_choices = max_choices.max(r.max_choices);
                let (result, winner, reason) = match r.outcome {
                    Outcome::Win { player: 0, reason } => { p0 += 1; ("win", 0, reason.to_string()) }
                    Outcome::Win { player, reason } => { p1 += 1; ("loss", player as i32, reason.to_string()) }
                    Outcome::Draw { reason } => { draws += 1; ("draw", -1, reason.to_string()) }
                };
                if record {
                    let rep = serde_json::json!({"name": "random", "job": 0, "game": g, "seed": game_seed, "seats": ["random", "random"],
                        "winner": winner, "reason": reason, "turns": r.turns, "lp": r.lp, "names": r.names, "steps": r.steps,
                        "finished": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)});
                    std::fs::write(replay_dir.join(format!("random-s{seed}-g{g}.json")), rep.to_string()).expect("replay file");
                }
                serde_json::json!({"game": g, "seed": game_seed, "result_p0": result, "winner": winner, "reason": reason,
                    "turns": r.turns, "decisions": r.decisions, "retries": r.retries, "script_errors": r.script_errors,
                    "max_choices": r.max_choices, "loop_breaks": r.loop_breaks})
            }
            Err(e) => {
                errors += 1;
                serde_json::json!({"game": g, "seed": game_seed, "error": e})
            }
        };
        println!("{line}");
        writeln!(log, "{line}").unwrap();
    }
    let secs = started.elapsed().as_secs_f64();
    let last_err = data::LAST_SCRIPT_ERROR.lock().map(|s| s.clone()).unwrap_or_default();
    println!(
        "games={games} errors={errors} wins_p0={p0} losses_p0={p1} draws={draws} script_errors={script_errors} seconds={secs:.1} max_choices={max_choices}"
    );
    if script_errors > 0 {
        println!("last_script_error={last_err:?}");
    }
}
