//! ygo-runner: Cyber Dragon mirror games on EDOPro's engine.
//!   ygo-runner random [--games 100] [--seed 1]                random vs random
//!   ygo-runner remote [--games 20] [--listen 0.0.0.0:8765]   player 0 = GPU worker, player 1 = random
//! Paths: --deck, --scripts, --cdb (repeatable), --log (one JSON line per game).
mod broker;
mod data;
mod decide;
mod duel;
mod ffi;
mod rng;

use duel::{Agent, Outcome, RandomAgent};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

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
    if mode != "random" && mode != "remote" {
        eprintln!("usage: ygo-runner random|remote [--games N] [--seed S] [--listen ADDR]");
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

    let broker = if mode == "remote" {
        let addr = flag("--listen").unwrap_or("0.0.0.0:8765".into());
        let b = broker::Broker::start(&addr).unwrap_or_else(|e| panic!("{e}"));
        println!("broker listening on {addr}");
        Some(b)
    } else {
        None
    };

    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("log file");
    let (mut p0, mut p1, mut draws, mut errors, mut script_errors, mut remote_requests) = (0, 0, 0, 0, 0usize, 0u64);
    let started = std::time::Instant::now();
    for g in 0..games {
        let game_seed = seed.wrapping_mul(1_000_003).wrapping_add(g as u64);
        let mut a1 = RandomAgent;
        let mut rnd0 = RandomAgent;
        let mut remote = broker.as_ref().map(|b| broker::RemoteAgent { broker: b.clone(), game: g, requests: 0 });
        let a0: &mut dyn Agent = match remote.as_mut() {
            Some(r) => r,
            None => &mut rnd0,
        };
        let res = catch_unwind(AssertUnwindSafe(|| duel::play(&lib, &deck, game_seed, &mut [a0, &mut a1])));
        let res = match res {
            Ok(r) => r,
            Err(p) => Err(format!(
                "panic: {}",
                p.downcast_ref::<String>().cloned().or(p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
            )),
        };
        let reqs = remote.as_ref().map(|r| r.requests).unwrap_or(0);
        remote_requests += reqs;
        let line = match &res {
            Ok(r) => {
                script_errors += r.script_errors;
                let (result, winner, reason) = match r.outcome {
                    Outcome::Win { player: 0, reason } => { p0 += 1; ("win", 0, reason.to_string()) }
                    Outcome::Win { player, reason } => { p1 += 1; ("loss", player as i32, reason.to_string()) }
                    Outcome::Draw { reason } => { draws += 1; ("draw", -1, reason.to_string()) }
                };
                serde_json::json!({"game": g, "seed": game_seed, "result_p0": result, "winner": winner, "reason": reason,
                    "turns": r.turns, "decisions": r.decisions, "retries": r.retries, "script_errors": r.script_errors,
                    "remote_requests": reqs})
            }
            Err(e) => {
                errors += 1;
                serde_json::json!({"game": g, "seed": game_seed, "error": e, "remote_requests": reqs})
            }
        };
        println!("{line}");
        writeln!(log, "{line}").unwrap();
    }
    let secs = started.elapsed().as_secs_f64();
    let last_err = data::LAST_SCRIPT_ERROR.lock().map(|s| s.clone()).unwrap_or_default();
    let prefix = if mode == "remote" { "remote_games" } else { "games" };
    println!(
        "{prefix}={games} errors={errors} wins_p0={p0} losses_p0={p1} draws={draws} script_errors={script_errors} seconds={secs:.1}"
    );
    if let Some(b) = &broker {
        let (served, answered) = b.counts();
        println!("remote_requests={remote_requests} broker_served={served} broker_answered={answered}");
    }
    if script_errors > 0 {
        println!("last_script_error={last_err:?}");
    }
}
