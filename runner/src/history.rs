//! Round 4: "Recent plays", the last HISTORY_LEN public events of the game, for the model's prompt.
//! A whitelist: summons, sets (the opponent's without a name), activations (chain links), attacks, cards sent to
//! the GY or banished face-up, and cards revealed from the hand. Anything else (draws, moves to the hand, deck or
//! a face-down zone, deck reveals) is left out, so nothing hidden from the decider can reach its prompt.
use crate::board;
use crate::data::Library;
use crate::ffi::*;
use std::collections::VecDeque;

pub const HISTORY_LEN: usize = 16;
const MSG_CONFIRM_CARDS: u8 = 31;
const MSG_MOVE: u8 = 50;
const MSG_SET: u8 = 54;
const MSG_SUMMONING: u8 = 60;
const MSG_SPSUMMONING: u8 = 62;
const MSG_FLIPSUMMONING: u8 = 64;
const MSG_CHAINING: u8 = 70;
const MSG_ATTACK: u8 = 110;

/// One public event. `actor` is the player it is about; `named` says whether the card name is public.
#[derive(Clone, Debug)]
pub struct Event {
    pub turn: u32,
    pub actor: u8,
    pub verb: &'static str,
    pub code: u32,
    pub named: bool,
    /// Attacks: the target (controller, code, named), or None for a direct attack.
    pub target: Option<(u8, u32, bool)>,
    pub chain: u32,
}

#[derive(Default)]
pub struct History(VecDeque<Event>);

fn u32_at(m: &[u8], at: usize) -> Option<u32> {
    m.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// The public event in an engine message, if it is one. Pure apart from `card` (looks up the card at a
/// location, for attacks: the message carries locations only).
pub fn event(msg: &[u8], turn: u32, card: &dyn Fn(u8, u8, u32) -> Option<board::CardView>) -> Option<Event> {
    let t = *msg.first()?;
    let ev = |actor: u8, verb, code, named| Some(Event { turn, actor: actor & 1, verb, code, named, target: None, chain: 0 });
    match t {
        // code u32, then loc_info: controller u8, location u8, sequence u32, position u32
        MSG_SUMMONING | MSG_FLIPSUMMONING if msg.len() >= 15 => {
            let verb = if t == MSG_SUMMONING { "Normal Summoned" } else { "Flip Summoned" };
            ev(msg[5], verb, u32_at(msg, 1)?, true)
        }
        MSG_SPSUMMONING if msg.len() >= 15 => {
            let pos = u32_at(msg, 11)?;
            let code = u32_at(msg, 1)?;
            ev(msg[5], "Special Summoned", code, code != 0 && pos & 0x5 != 0)
        }
        MSG_SET if msg.len() >= 15 => {
            let verb = if msg[6] as u32 & LOCATION_MZONE as u32 != 0 { "set a monster" } else { "set a spell/trap" };
            ev(msg[5], verb, u32_at(msg, 1)?, false) // the setter's own name is added at render time
        }
        // code, loc_info, triggering controller u8, location u8, sequence u32, description u64, chain size u32
        MSG_CHAINING if msg.len() >= 33 => {
            let mut e = ev(msg[5], "activated", u32_at(msg, 1)?, true)?;
            e.chain = u32_at(msg, 29)?;
            Some(e)
        }
        // attacker loc_info, target loc_info (all zero for a direct attack)
        MSG_ATTACK if msg.len() >= 21 => {
            let (ac, al, aseq) = (msg[1], msg[2], u32_at(msg, 3)?);
            let a = card(ac & 1, al, aseq)?;
            let (tc, tl, tseq) = (msg[11], msg[12], u32_at(msg, 13)?);
            let mut e = ev(ac, "attacked with", a.code, a.pos & 0x5 != 0)?;
            if tl != 0 {  // a target the lookup can't read is rendered as a face-down monster
                e.target = Some(card(tc & 1, tl, tseq).map(|c| (tc & 1, c.code, c.pos & 0x5 != 0)).unwrap_or((tc & 1, 0, false)));
            }
            Some(e)
        }
        // code, previous loc_info, current loc_info, reason
        MSG_MOVE if msg.len() >= 25 => {
            let (con, loc, pos) = (msg[15], msg[16] as u32, u32_at(msg, 21)?);
            let code = u32_at(msg, 1)?;
            if loc & LOCATION_GRAVE != 0 && msg[16] & 0x80 == 0 {
                ev(con, "sent to the GY", code, code != 0)
            } else if loc & LOCATION_REMOVED != 0 && pos & 0x5 != 0 {
                ev(con, "banished", code, code != 0)
            } else {
                None
            }
        }
        // player u8, count u32, then per card: code u32, controller u8, location u8, sequence u32
        MSG_CONFIRM_CARDS if msg.len() >= 6 => {
            let n = u32_at(msg, 2)? as usize;
            if n != 1 || msg.len() < 16 {
                return None; // several cards at once: hand reveals of one card only, kept simple
            }
            let (code, con, loc) = (u32_at(msg, 6)?, msg[10], msg[11] as u32);
            if loc & LOCATION_HAND == 0 {
                return None; // a deck or face-down reveal is shown to one player only
            }
            ev(con, "revealed from the hand", code, code != 0)
        }
        _ => None,
    }
}

impl History {
    pub fn push(&mut self, e: Event) {
        self.0.push_back(e);
        while self.0.len() > HISTORY_LEN {
            self.0.pop_front();
        }
    }

    /// The section for player `me`, oldest first; empty before the first event.
    pub fn render(&self, lib: &Library, me: u8) -> String {
        if self.0.is_empty() {
            return String::new();
        }
        let me = me & 1;
        let mut t = String::from("Recent plays (oldest first):\n");
        for e in &self.0 {
            let mine = e.actor == me;
            let who = if mine { "I" } else { "opponent" };
            let name = |code: u32, named: bool, own: bool| if named || own { lib.name(code) } else { String::new() };
            let line = match e.verb {
                "set a monster" | "set a spell/trap" if mine => format!("{who} {} ({})", e.verb, lib.name(e.code)),
                "set a monster" | "set a spell/trap" => format!("{who} {}", e.verb),
                "attacked with" => {
                    let a = name(e.code, e.named, mine);
                    let a = if a.is_empty() { "a face-down monster".into() } else { a };
                    match e.target {
                        None => format!("{who} attacked directly with {a}"),
                        Some((tc, code, named)) => {
                            let tn = name(code, named, tc == me);
                            let whose = if tc == me { "my" } else { "opponent's" };
                            let tn = if tn.is_empty() { "face-down monster".into() } else { tn };
                            format!("{who} attacked {whose} {tn} with {a}")
                        }
                    }
                }
                "activated" => format!("{who} activated {} (chain link {})", lib.name(e.code), e.chain),
                v => {
                    let n = name(e.code, e.named, mine);
                    if n.is_empty() {
                        format!("{who} {v} a face-down card")
                    } else if v == "sent to the GY" || v == "banished" {
                        format!("{} {n} was {}", if mine { "my" } else { "opponent's" }, if v == "banished" { "banished" } else { "sent to the GY" })
                    } else {
                        format!("{who} {v} {n}")
                    }
                }
            };
            t += &format!("- T{} {line}\n", e.turn);
        }
        t
    }
}
