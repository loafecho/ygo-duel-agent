//! The board as either player sees it, read from the engine with OCG_DuelQueryLocation.
//! One snapshot feeds both the model's prompt (hidden cards hidden) and the replay viewer (everything).
use crate::data::Library;
use crate::ffi::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct CardView {
    pub code: u32,
    pub pos: u32,
    pub type_: u32,
    pub level: u32,
    pub atk: i32,
    pub def: i32,
    pub public: bool,
}

#[derive(Clone, Default)]
pub struct Side {
    pub hand: Vec<CardView>,
    pub mzone: Vec<Option<CardView>>,
    pub szone: Vec<Option<CardView>>,
    pub grave: Vec<CardView>,
    pub banished: Vec<CardView>,
    pub deck: u32,
    pub extra: u32,
}

#[derive(Clone)]
pub struct Board {
    pub turn: u32,
    pub turn_player: u8,
    pub phase: u16,
    pub lp: [i64; 2],
    pub sides: [Side; 2],
}

const FLAGS: u32 = QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_ATTACK | QUERY_DEFENSE;

/// Records per card: [u16 size][u32 flag][size-4 bytes], ending with QUERY_END; an empty slot is a lone u16 0.
fn query(duel: OCG_Duel, con: u8, loc: u32) -> Result<Vec<Option<CardView>>, String> {
    let info = OCG_QueryInfo { flags: FLAGS, con, loc, seq: 0, overlay_seq: 0 };
    let mut len = 0u32;
    let ptr = unsafe { OCG_DuelQueryLocation(duel, &mut len, &info) } as *const u8;
    if len < 4 {
        return Err(format!("query {con}/{loc:#x}: {len} bytes"));
    }
    let b = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let total = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let b = &b[4..4 + total.min(b.len() - 4)];
    let mut out = Vec::new();
    let mut p = 0usize;
    let rd16 = |p: usize| -> Result<u16, String> {
        b.get(p..p + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap())).ok_or("query truncated".to_string())
    };
    let rd32 = |p: usize| -> Result<u32, String> {
        b.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or("query truncated".to_string())
    };
    while p < b.len() {
        let size = rd16(p)? as usize;
        if size == 0 {
            out.push(None);
            p += 2;
            continue;
        }
        let mut c = CardView { code: 0, pos: 0, type_: 0, level: 0, atk: 0, def: 0, public: false };
        loop {
            let size = rd16(p)? as usize;
            let flag = rd32(p + 2)?;
            let v = p + 6;
            match flag {
                QUERY_CODE => c.code = rd32(v)?,
                QUERY_POSITION => c.pos = rd32(v)?,
                QUERY_TYPE => c.type_ = rd32(v)?,
                QUERY_LEVEL => c.level = rd32(v)?,
                QUERY_ATTACK => c.atk = rd32(v)? as i32,
                QUERY_DEFENSE => c.def = rd32(v)? as i32,
                QUERY_IS_PUBLIC => c.public = *b.get(v).ok_or("query truncated")? != 0,
                _ => {}
            }
            p += 2 + size;
            if flag == QUERY_END {
                break;
            }
        }
        out.push(Some(c));
    }
    Ok(out)
}

/// The card at one location (history.rs: attack messages carry locations, not codes).
pub fn card_at(duel: OCG_Duel, con: u8, loc: u8, seq: u32) -> Option<CardView> {
    query(duel, con, loc as u32).ok()?.into_iter().nth(seq as usize).flatten()
}

impl Board {
    pub fn read(duel: OCG_Duel, turn: u32, turn_player: u8, phase: u16, lp: [i64; 2]) -> Result<Board, String> {
        let mut sides: [Side; 2] = Default::default();
        for p in 0..2u8 {
            let s = &mut sides[p as usize];
            s.hand = query(duel, p, LOCATION_HAND)?.into_iter().flatten().collect();
            s.mzone = query(duel, p, LOCATION_MZONE as u32)?;
            s.szone = query(duel, p, LOCATION_SZONE as u32)?;
            s.grave = query(duel, p, LOCATION_GRAVE)?.into_iter().flatten().collect();
            s.banished = query(duel, p, LOCATION_REMOVED)?.into_iter().flatten().collect();
            s.deck = unsafe { OCG_DuelQueryCount(duel, p, LOCATION_DECK) };
            s.extra = unsafe { OCG_DuelQueryCount(duel, p, LOCATION_EXTRA) };
        }
        Ok(Board { turn, turn_player, phase, lp, sides })
    }

    /// The decider's view, in words: own cards in full, opponent's hidden cards hidden.
    pub fn describe(&self, lib: &Library, me: u8) -> String {
        let me = (me & 1) as usize;
        let op = 1 - me;
        let whose_turn = if self.turn_player as usize & 1 == me { "my turn" } else { "opponent's turn" };
        let mut t = format!(
            "Yu-Gi-Oh duel. Turn {}, {} ({whose_turn}).\nLife points: me {}, opponent {}.\n",
            self.turn, phase_name(self.phase), self.lp[me], self.lp[op]
        );
        let (m, o) = (&self.sides[me], &self.sides[op]);
        let names = |v: &[CardView]| grouped(v.iter().map(|c| lib.name(c.code)));
        t += &format!("My hand ({}): {}\n", m.hand.len(), if m.hand.is_empty() { "empty".into() } else { names(&m.hand) });
        t += &format!("My monsters: {}\n", monsters(lib, &m.mzone, true));
        t += &format!("My spells/traps: {}\n", spells(lib, &m.szone, true));
        t += &format!("My graveyard ({}): {}\n", m.grave.len(), if m.grave.is_empty() { "empty".into() } else { names(&m.grave) });
        t += &format!("My banished ({}): {}\n", m.banished.len(), if m.banished.is_empty() { "none".into() } else { names(&m.banished) });
        t += &format!("My deck: {} cards. My extra deck: {} cards.\n", m.deck, m.extra);
        let shown: Vec<&CardView> = o.hand.iter().filter(|c| c.public).collect();
        let hand = if shown.is_empty() {
            format!("{} cards", o.hand.len())
        } else {
            format!("{} cards, revealed: {}", o.hand.len(), shown.iter().map(|c| lib.name(c.code)).collect::<Vec<_>>().join(", "))
        };
        t += &format!("Opponent's hand: {hand}\n");
        t += &format!("Opponent's monsters: {}\n", monsters(lib, &o.mzone, false));
        t += &format!("Opponent's spells/traps: {}\n", spells(lib, &o.szone, false));
        t += &format!("Opponent's graveyard ({}): {}\n", o.grave.len(), if o.grave.is_empty() { "empty".into() } else { names(&o.grave) });
        // Face-down banished cards are hidden from me: names for the face-up ones, a count for the rest.
        let up: Vec<CardView> = o.banished.iter().filter(|c| c.pos & 0x5 != 0).cloned().collect();
        let down = o.banished.len() - up.len();
        t += &format!("Opponent's banished ({}): {}{}\n", o.banished.len(),
            if up.is_empty() { if down == 0 { "none".into() } else { String::new() } } else { names(&up) },
            if down == 0 { String::new() } else { format!("{}{down} face-down", if up.is_empty() { "" } else { ", " }) });
        t += &format!("Opponent's deck: {} cards.\n", o.deck);
        t
    }

    /// Effect text, once per card: my hand and field, the opponent's face-up field cards, then any other card
    /// the options name. Graveyards stay names only.
    pub fn card_texts(&self, lib: &Library, me: u8, option_codes: &[u32]) -> String {
        let me = (me & 1) as usize;
        let (m, o) = (&self.sides[me], &self.sides[1 - me]);
        let mut codes: Vec<u32> = Vec::new();
        let faceup = |c: &&CardView| c.pos & 0x5 != 0;
        let mine = m.hand.iter().chain(m.mzone.iter().flatten()).chain(m.szone.iter().flatten());
        let theirs = o.mzone.iter().flatten().chain(o.szone.iter().flatten()).filter(faceup);
        for code in mine.chain(theirs).map(|c| c.code).chain(option_codes.iter().copied()) {
            if code != 0 && !codes.contains(&code) {
                codes.push(code);
            }
        }
        if codes.is_empty() {
            return String::new();
        }
        let mut t = String::from("Card text:\n");
        for c in codes {
            t += &format!("- {}\n", lib.text_line(c));
        }
        t
    }

    /// Full board for the viewer. A card is [code, position, atk, def, public, type]; names go in `names`.
    pub fn to_json(&self, lib: &Library, names: &mut BTreeMap<u32, String>) -> Value {
        let mut card = |c: &CardView| {
            names.entry(c.code).or_insert_with(|| lib.name(c.code));
            json!([c.code, c.pos, c.atk, c.def, c.public as u8, c.type_])
        };
        let mut sides = Vec::new();
        for s in &self.sides {
            let hand: Vec<Value> = s.hand.iter().map(&mut card).collect();
            let mzone: Vec<Value> = s.mzone.iter().map(|c| c.as_ref().map(&mut card).unwrap_or(Value::Null)).collect();
            let szone: Vec<Value> = s.szone.iter().map(|c| c.as_ref().map(&mut card).unwrap_or(Value::Null)).collect();
            let grave: Vec<Value> = s.grave.iter().map(&mut card).collect();
            let banished: Vec<Value> = s.banished.iter().map(&mut card).collect();
            sides.push(json!({"hand": hand, "mzone": mzone, "szone": szone, "grave": grave, "banished": banished,
                "deck": s.deck, "extra": s.extra}));
        }
        json!({"turn": self.turn, "turn_player": self.turn_player, "phase": phase_name(self.phase), "lp": self.lp, "sides": sides})
    }
}

/// "Maxx \"C\" x3, Raigeki": duplicates counted, first-seen order kept (keeps late-game prompts short).
fn grouped(names: impl Iterator<Item = String>) -> String {
    let mut seen: Vec<(String, usize)> = Vec::new();
    for n in names {
        match seen.iter_mut().find(|(s, _)| *s == n) {
            Some(e) => e.1 += 1,
            None => seen.push((n, 1)),
        }
    }
    seen.into_iter().map(|(n, k)| if k > 1 { format!("{n} x{k}") } else { n }).collect::<Vec<_>>().join(", ")
}

const PHASES: [(u16, &str); 10] = [
    (0x01, "draw phase"), (0x02, "standby phase"), (0x04, "main phase 1"), (0x08, "battle phase"),
    (0x10, "battle step"), (0x20, "damage step"), (0x40, "damage calculation"), (0x80, "end of battle"),
    (0x100, "main phase 2"), (0x200, "end phase"),
];

pub fn phase_name(p: u16) -> &'static str {
    PHASES.iter().find(|x| x.0 == p).map(|x| x.1).unwrap_or("start of turn")
}

fn position(pos: u32) -> &'static str {
    match pos {
        0x1 => "face-up attack",
        0x2 => "face-down attack",
        0x4 => "face-up defense",
        _ => "face-down defense",
    }
}

fn monsters(lib: &Library, zones: &[Option<CardView>], mine: bool) -> String {
    let mut parts = Vec::new();
    for (i, c) in zones.iter().enumerate() {
        let Some(c) = c else { continue };
        let zone = if i < 5 { format!("zone {}", i + 1) } else { format!("extra zone {}", i - 4) };
        let faceup = c.pos & 0x5 != 0;
        if faceup || mine {
            let hidden = if faceup { "" } else { ", set" };
            parts.push(format!("{zone} {} ({}, ATK {} DEF {}{hidden})", lib.name(c.code), position(c.pos), c.atk, c.def));
        } else {
            parts.push(format!("{zone} face-down {} monster", if c.pos & 0x2 != 0 { "attack" } else { "defense" }));
        }
    }
    if parts.is_empty() { "none".into() } else { parts.join("; ") }
}

fn spells(lib: &Library, zones: &[Option<CardView>], mine: bool) -> String {
    let mut parts = Vec::new();
    for (i, c) in zones.iter().enumerate() {
        let Some(c) = c else { continue };
        let zone = match i {
            0..=4 => format!("zone {}", i + 1),
            5 => "field zone".into(),
            _ => format!("pendulum zone {}", i - 5),
        };
        let faceup = c.pos & 0x5 != 0;
        if faceup {
            parts.push(format!("{zone} {} (face-up)", lib.name(c.code)));
        } else if mine {
            parts.push(format!("{zone} {} (set)", lib.name(c.code)));
        } else {
            parts.push(format!("{zone} face-down card"));
        }
    }
    if parts.is_empty() { "none".into() } else { parts.join("; ") }
}

/// Options lettered A-Z then a-z; the model answers with one letter (a single token in Qwen's vocabulary).
pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Round 4: `history` ("Recent plays", history.rs) sits between the board and the card text.
pub fn prompt(board_text: &str, history: &str, card_texts: &str, kind: &str, options: &[String]) -> String {
    let mut t = format!("{board_text}{history}{card_texts}Decision: {kind}.\nOptions:\n");
    for (i, o) in options.iter().enumerate() {
        t += &format!("{}. {o}\n", LETTERS.as_bytes()[i] as char);
    }
    t += "Answer:";
    t
}
