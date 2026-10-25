//! Turns every message the engine waits on (MSG_SELECT_*, MSG_SORT_*, MSG_ANNOUNCE_*, rock-paper-scissors)
//! into a list of legal choices, each carrying the exact response bytes the engine accepts.
//! Formats mirror playerop.cpp in edo9300/ygopro-core. Combinatorial selections (several cards, sums,
//! zones, counters, orderings) are offered as up to MAX_CHOICES sampled legal answers.
use crate::data::Library;
use crate::ffi::*;
use crate::rng::Rng;

pub const MAX_CHOICES: usize = 16;

pub const MSG_RETRY: u8 = 1;
pub const MSG_HINT: u8 = 2;
pub const MSG_WIN: u8 = 5;
pub const MSG_NEW_TURN: u8 = 40;
pub const MSG_NEW_PHASE: u8 = 41;
pub const MSG_DAMAGE: u8 = 91;
pub const MSG_RECOVER: u8 = 92;
pub const MSG_LPUPDATE: u8 = 94;
pub const MSG_PAY_LPCOST: u8 = 100;

const DECISIONS: &[u8] = &[10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 132, 140, 141, 142, 143];

pub fn is_decision(t: u8) -> bool {
    DECISIONS.contains(&t)
}

pub struct Choice {
    pub text: String,
    pub resp: Vec<u8>,
}

pub struct Decision {
    pub msg: u8,
    pub player: u8,
    pub kind: &'static str,
    pub choices: Vec<Choice>,
    /// Select/unselect (26) only: the card each "unselect" choice takes back, so the caller can stop it
    /// from being picked again in the same selection (a fixed policy would otherwise toggle it forever).
    pub unselects: Vec<Option<CardKey>>,
    /// Cards the options name, first-mentioned order (not the declare-a-card list: those are random picks).
    pub codes: Vec<u32>,
}

/// controller, location, sequence, code
pub type CardKey = (u8, u8, u32, u32);

pub struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() {
            return Err(format!("truncated message: need {n} bytes at {} of {}", self.pos, self.b.len()));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// loc_info: controler u8, location u8, sequence u32, position u32
    fn loc_info(&mut self) -> Result<(u8, u8, u32), String> {
        let (c, l, s) = (self.u8()?, self.u8()?, self.u32()?);
        self.u32()?;
        Ok((c, l, s))
    }
}

fn i32r(v: i32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// parse_response_cards type 0: [0 i32][count u32][index u32...]
fn cards_resp(idx: &[usize]) -> Vec<u8> {
    let mut r = i32r(0);
    r.extend((idx.len() as u32).to_le_bytes());
    for i in idx {
        r.extend((*i as u32).to_le_bytes());
    }
    r
}

fn loc_name(loc: u8) -> &'static str {
    match loc {
        0x01 => "deck",
        0x02 => "hand",
        0x04 => "monster zone",
        0x08 => "spell/trap zone",
        0x10 => "graveyard",
        0x20 => "banished",
        0x40 => "extra deck",
        0x80 => "material",
        _ => "field",
    }
}

thread_local! {
    /// Card codes the options of the decision being parsed mention (their effect text goes in the prompt).
    static MENTIONED: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn note(code: u32) {
    MENTIONED.with(|m| {
        let mut m = m.borrow_mut();
        if !m.contains(&code) {
            m.push(code);
        }
    });
}

/// Is the card at (controller, location, sequence) public? Asked for the opponent's cards only.
pub type Public<'a> = &'a dyn Fn(u8, u8, u32) -> bool;

/// Round 4: an opponent's card the decider can't see is named "a face-down card" and its text is not added
/// (round 3 named face-down cards in select options, a leak the leak check found). Graveyard cards are public;
/// field and banished cards are public when face-up; hand, deck and extra-deck cards are hidden.
fn card_text(lib: &Library, code: u32, me: u8, con: u8, loc: u8, seq: u32, public: Public) -> String {
    let whose = if con == me { "my" } else { "opponent's" };
    let shown = con == me || loc & 0x10 != 0
        || (loc & (0x04 | 0x08 | 0x20) != 0 && public(con, loc, seq));
    if !shown {
        return format!("a face-down card ({whose} {})", loc_name(loc));
    }
    note(code);
    format!("{} ({whose} {})", lib.name(code), loc_name(loc))
}

fn push_unique(out: &mut Vec<Choice>, c: Choice) {
    if out.len() < MAX_CHOICES && !out.iter().any(|o| o.resp == c.resp) {
        out.push(c);
    }
}

/// `unselected`: cards taken back earlier in the current select/unselect sequence (not offered again).
pub fn parse(msg: &[u8], lib: &Library, rng: &mut Rng, unselected: &[CardKey], public: Public) -> Result<Decision, String> {
    let t = msg[0];
    MENTIONED.with(|m| m.borrow_mut().clear());
    let mut r = Reader::new(&msg[1..]);
    let player = r.u8()?;
    let mut ch: Vec<Choice> = Vec::new();
    let mut unselects: Vec<Option<CardKey>> = Vec::new();
    let kind: &'static str;
    match t {
        10 => {
            kind = "battle phase command";
            let n = r.u32()?;
            for i in 0..n {
                let (code, con, loc, seq, desc) = (r.u32()?, r.u8()?, r.u8()?, r.u32()?, r.u64()?);
                r.u8()?;
                ch.push(Choice {
                    text: format!("activate {}: {}", card_text(lib, code, player, con, loc, seq, public), lib.desc(desc)),
                    resp: i32r(((i as i32) << 16) | 0),
                });
            }
            let n = r.u32()?;
            for i in 0..n {
                let (code, con, loc, seq, direct) = (r.u32()?, r.u8()?, r.u8()?, r.u8()? as u32, r.u8()?);
                let how = if direct != 0 { " (can attack directly)" } else { "" };
                ch.push(Choice {
                    text: format!("attack with {}{how}", card_text(lib, code, player, con, loc, seq, public)),
                    resp: i32r(((i as i32) << 16) | 1),
                });
            }
            if r.u8()? != 0 {
                ch.push(Choice { text: "go to main phase 2".into(), resp: i32r(2) });
            }
            if r.u8()? != 0 {
                ch.push(Choice { text: "end the turn".into(), resp: i32r(3) });
            }
        }
        11 => {
            kind = "main phase command";
            let verbs = ["normal summon", "special summon", "change position of", "set monster", "set spell/trap"];
            for (ti, verb) in verbs.iter().enumerate() {
                let n = r.u32()?;
                for i in 0..n {
                    let code = r.u32()?;
                    let (con, loc) = (r.u8()?, r.u8()?);
                    let seq = if ti == 2 { r.u8()? as u32 } else { r.u32()? };
                    ch.push(Choice {
                        text: format!("{verb} {}", card_text(lib, code, player, con, loc, seq, public)),
                        resp: i32r(((i as i32) << 16) | ti as i32),
                    });
                }
            }
            let n = r.u32()?;
            for i in 0..n {
                let (code, con, loc, seq, desc) = (r.u32()?, r.u8()?, r.u8()?, r.u32()?, r.u64()?);
                r.u8()?;
                ch.push(Choice {
                    text: format!("activate {}: {}", card_text(lib, code, player, con, loc, seq, public), lib.desc(desc)),
                    resp: i32r(((i as i32) << 16) | 5),
                });
            }
            if r.u8()? != 0 {
                ch.push(Choice { text: "go to battle phase".into(), resp: i32r(6) });
            }
            if r.u8()? != 0 {
                ch.push(Choice { text: "end the turn".into(), resp: i32r(7) });
            }
            r.u8()?; // shuffle hand (8): a no-op for play, not offered
        }
        12 => {
            kind = "use an effect?";
            let code = r.u32()?;
            let (con, loc, seq) = r.loc_info()?;
            let desc = r.u64()?;
            let what = format!("{}: {}", card_text(lib, code, player, con, loc, seq, public), lib.desc(desc));
            ch.push(Choice { text: format!("yes, use {what}"), resp: i32r(1) });
            ch.push(Choice { text: format!("no, skip {what}"), resp: i32r(0) });
        }
        13 => {
            kind = "yes or no";
            let desc = lib.desc(r.u64()?);
            ch.push(Choice { text: format!("yes: {desc}"), resp: i32r(1) });
            ch.push(Choice { text: format!("no: {desc}"), resp: i32r(0) });
        }
        14 => {
            kind = "choose an option";
            let n = r.u8()?;
            for i in 0..n {
                ch.push(Choice { text: lib.desc(r.u64()?), resp: i32r(i as i32) });
            }
        }
        15 => {
            kind = "select cards";
            let cancel = r.u8()? != 0;
            let (min, max, n) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
            let mut cards = Vec::with_capacity(n);
            for _ in 0..n {
                let code = r.u32()?;
                let (con, loc, seq) = r.loc_info()?;
                cards.push(card_text(lib, code, player, con, loc, seq, public));
            }
            if cancel {
                ch.push(Choice { text: "select nothing / cancel".into(), resp: i32r(-1) });
            }
            let lo = min.max(1).min(n);
            let hi = max.min(n);
            if hi == 1 {
                for (i, c) in cards.iter().enumerate() {
                    push_unique(&mut ch, Choice { text: format!("select {c}"), resp: cards_resp(&[i]) });
                }
            } else if hi >= lo && n > 0 {
                for _ in 0..64 {
                    let k = rng.range(lo, hi);
                    let mut idx: Vec<usize> = (0..n).collect();
                    rng.shuffle(&mut idx);
                    idx.truncate(k);
                    idx.sort();
                    let names: Vec<&str> = idx.iter().map(|i| cards[*i].as_str()).collect();
                    push_unique(&mut ch, Choice { text: format!("select {}", names.join(", ")), resp: cards_resp(&idx) });
                }
            }
        }
        16 => {
            kind = "chain an effect?";
            r.u8()?; // spe_count
            let forced = r.u8()? != 0;
            r.u32()?;
            r.u32()?;
            let n = r.u32()?;
            if !forced {
                ch.push(Choice { text: "do not chain".into(), resp: i32r(-1) });
            }
            for i in 0..n {
                let code = r.u32()?;
                let (con, loc, seq) = r.loc_info()?;
                let desc = r.u64()?;
                r.u8()?;
                ch.push(Choice {
                    text: format!("chain {}: {}", card_text(lib, code, player, con, loc, seq, public), lib.desc(desc)),
                    resp: i32r(i as i32),
                });
            }
        }
        18 | 24 => {
            kind = if t == 18 { "choose a zone" } else { "choose a zone to disable" };
            let count = r.u8()? as usize;
            let flag = r.u32()?;
            let mut zones: Vec<(u8, u8, u8, String)> = Vec::new();
            for b in 0..32u32 {
                if flag & (1 << b) != 0 || b % 8 == 7 && b % 16 < 8 {
                    continue; // taken, or monster-zone bit 7 (no such zone)
                }
                let p = if b >= 16 { 1 - player } else { player };
                let w = b % 16;
                let (loc, seq) = if w < 8 { (LOCATION_MZONE, w as u8) } else { (LOCATION_SZONE, (w - 8) as u8) };
                let whose = if p == player { "my" } else { "opponent's" };
                zones.push((p, loc, seq, format!("{whose} {} {}", loc_name(loc), seq + 1)));
            }
            if count >= 1 && zones.len() >= count {
                for _ in 0..64 {
                    let mut idx: Vec<usize> = (0..zones.len()).collect();
                    rng.shuffle(&mut idx);
                    idx.truncate(count);
                    idx.sort();
                    let mut resp = Vec::new();
                    let mut names = Vec::new();
                    for i in &idx {
                        let z = &zones[*i];
                        resp.extend([z.0, z.1, z.2]);
                        names.push(z.3.as_str());
                    }
                    push_unique(&mut ch, Choice { text: names.join(", "), resp });
                }
            }
        }
        19 => {
            kind = "choose a position";
            let code = r.u32()?;
            let pos = r.u8()?;
            for (bit, name) in [(1u8, "face-up attack"), (2, "face-down attack"), (4, "face-up defense"), (8, "face-down defense")] {
                if pos & bit != 0 {
                    note(code);
                    ch.push(Choice { text: format!("{} in {name}", lib.name(code)), resp: i32r(bit as i32) });
                }
            }
        }
        20 => {
            kind = "select tributes";
            let cancel = r.u8()? != 0;
            let (min, max, n) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
            let mut cards = Vec::with_capacity(n);
            for _ in 0..n {
                let (code, con, loc, seq, rel) = (r.u32()?, r.u8()?, r.u8()?, r.u32()?, r.u8()?);
                cards.push((card_text(lib, code, player, con, loc, seq, public), rel as usize));
            }
            if cancel {
                ch.push(Choice { text: "cancel".into(), resp: i32r(-1) });
            }
            let need = min.max(1);
            for _ in 0..64 {
                let mut idx: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut idx);
                let (mut pick, mut sum) = (Vec::new(), 0);
                for i in idx {
                    if sum >= need {
                        break;
                    }
                    pick.push(i);
                    sum += cards[i].1;
                }
                if sum >= min && pick.len() <= max && !pick.is_empty() {
                    pick.sort();
                    let names: Vec<&str> = pick.iter().map(|i| cards[*i].0.as_str()).collect();
                    push_unique(&mut ch, Choice { text: format!("tribute {}", names.join(", ")), resp: cards_resp(&pick) });
                }
            }
        }
        21 | 25 => {
            kind = if t == 21 { "order chain" } else { "order cards" };
            let n = r.u32()? as usize;
            let mut names = Vec::with_capacity(n);
            for _ in 0..n {
                let (code, _con, _loc, _seq) = (r.u32()?, r.u8()?, r.u32()?, r.u32()?);
                names.push(lib.name(code));
            }
            ch.push(Choice { text: "keep the default order".into(), resp: vec![0xff] });
            for _ in 0..32 {
                let mut perm: Vec<usize> = (0..n).collect();
                rng.shuffle(&mut perm);
                let order: Vec<&str> = perm.iter().map(|i| names[*i].as_str()).collect();
                // response byte i = new position of card i
                let mut resp = vec![0u8; n];
                for (pos, card) in perm.iter().enumerate() {
                    resp[*card] = pos as u8;
                }
                push_unique(&mut ch, Choice { text: format!("order: {}", order.join(", ")), resp });
            }
        }
        22 => {
            kind = "remove counters";
            r.u16()?;
            let count = r.u16()? as usize;
            let n = r.u32()? as usize;
            let mut cards = Vec::with_capacity(n);
            for _ in 0..n {
                let (code, _con, _loc, _seq, c) = (r.u32()?, r.u8()?, r.u8()?, r.u8()?, r.u16()?);
                cards.push((lib.name(code), c as usize));
            }
            let total: usize = cards.iter().map(|c| c.1).sum();
            if total >= count {
                for _ in 0..64 {
                    let mut order: Vec<usize> = (0..n).collect();
                    rng.shuffle(&mut order);
                    let mut take = vec![0usize; n];
                    let mut left = count;
                    let mut cap_after: usize = total;
                    for &i in &order {
                        cap_after -= cards[i].1;
                        let lo = left.saturating_sub(cap_after);
                        let hi = cards[i].1.min(left);
                        let k = rng.range(lo, hi);
                        take[i] = k;
                        left -= k;
                    }
                    let mut resp = Vec::new();
                    let mut parts = Vec::new();
                    for (i, k) in take.iter().enumerate() {
                        resp.extend((*k as i16).to_le_bytes());
                        if *k > 0 {
                            parts.push(format!("{k} from {}", cards[i].0));
                        }
                    }
                    push_unique(&mut ch, Choice { text: format!("remove {}", parts.join(", ")), resp });
                }
            }
        }
        23 => {
            kind = "select cards for a sum";
            let at_least = r.u8()? != 0;
            let (acc, min, max) = (r.u32()? as i32, r.u32()? as usize, r.u32()? as usize);
            let nm = r.u32()?;
            let mut must = Vec::new();
            for _ in 0..nm {
                r.u32()?;
                r.loc_info()?;
                must.push(r.u32()? as i32);
            }
            let n = r.u32()? as usize;
            let mut cards = Vec::with_capacity(n);
            for _ in 0..n {
                let code = r.u32()?;
                let (con, loc, seq) = r.loc_info()?;
                cards.push((card_text(lib, code, player, con, loc, seq, public), r.u32()? as i32));
            }
            let mut order: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut order);
            let mut found: Vec<Vec<usize>> = Vec::new();
            let mut budget = 200_000usize;
            let mut cur = Vec::new();
            sum_dfs(&order, 0, &mut cur, &cards, &must, acc, min, max, at_least, &mut found, &mut budget);
            for mut pick in found {
                pick.sort();
                let names: Vec<&str> = pick.iter().map(|i| cards[*i].0.as_str()).collect();
                push_unique(&mut ch, Choice { text: format!("select {}", names.join(", ")), resp: cards_resp(&pick) });
            }
        }
        26 => {
            // Unselect is offered only when nothing can be selected, and a card taken back is not offered
            // again in the same selection, so a fixed policy cannot toggle one card forever.
            kind = "select or unselect a card";
            let finishable = r.u8()? != 0;
            let cancelable = r.u8()? != 0;
            r.u32()?;
            r.u32()?;
            let mut i = 0i32;
            let mut sel: Vec<(Choice, CardKey)> = Vec::new();
            let n1 = r.u32()?;
            for _ in 0..n1 {
                let code = r.u32()?;
                let (con, loc, seq) = r.loc_info()?;
                let mut resp = i32r(1);
                resp.extend(i32r(i));
                sel.push((Choice { text: format!("select {}", card_text(lib, code, player, con, loc, seq, public)), resp }, (con, loc, seq, code)));
                i += 1;
            }
            let mut unsel: Vec<(Choice, CardKey)> = Vec::new();
            let n2 = r.u32()?;
            for _ in 0..n2 {
                let code = r.u32()?;
                let (con, loc, seq) = r.loc_info()?;
                let mut resp = i32r(1);
                resp.extend(i32r(i));
                unsel.push((Choice { text: format!("unselect {}", card_text(lib, code, player, con, loc, seq, public)), resp }, (con, loc, seq, code)));
                i += 1;
            }
            let (fresh, stale): (Vec<(Choice, CardKey)>, Vec<(Choice, CardKey)>) =
                sel.into_iter().partition(|(_, k)| !unselected.contains(k));
            if !fresh.is_empty() {
                for (c, _) in fresh {
                    ch.push(c);
                    unselects.push(None);
                }
            } else if !(finishable || cancelable) {
                for (c, k) in unsel {
                    ch.push(c);
                    unselects.push(Some(k));
                }
            }
            if finishable || cancelable {
                ch.push(Choice { text: if finishable { "finish selecting".into() } else { "cancel".into() }, resp: i32r(-1) });
                unselects.push(None);
            }
            if ch.is_empty() {
                for (c, _) in stale {
                    ch.push(c);
                    unselects.push(None);
                }
            }
        }
        132 => {
            kind = "rock paper scissors";
            for (v, name) in [(1, "scissors"), (2, "rock"), (3, "paper")] {
                ch.push(Choice { text: name.into(), resp: i32r(v) });
            }
        }
        140 | 141 => {
            kind = if t == 140 { "declare a monster type" } else { "declare an attribute" };
            let count = r.u8()? as usize;
            let avail = if t == 140 { r.u64()? } else { r.u32()? as u64 };
            let bits: Vec<u64> = (0..64).map(|b| 1u64 << b).filter(|b| avail & b != 0).collect();
            if bits.len() >= count && count > 0 {
                for _ in 0..64 {
                    let mut pick = bits.clone();
                    rng.shuffle(&mut pick);
                    pick.truncate(count);
                    let v: u64 = pick.iter().fold(0, |a, b| a | b);
                    let resp = if t == 140 { v.to_le_bytes().to_vec() } else { (v as u32).to_le_bytes().to_vec() };
                    push_unique(&mut ch, Choice { text: format!("declare {kind} mask {v:#x}"), resp });
                }
            }
        }
        142 => {
            kind = "declare a card name";
            let n = r.u8()?;
            let mut ops = Vec::with_capacity(n as usize);
            for _ in 0..n {
                ops.push(r.u64()?);
            }
            let mut ok: Vec<u32> = lib.cards.values().filter(|c| is_declarable(c, &ops)).map(|c| c.code).collect();
            ok.sort();
            rng.shuffle(&mut ok);
            for code in ok.into_iter().take(MAX_CHOICES) {
                ch.push(Choice { text: format!("declare {}", lib.name(code)), resp: i32r(code as i32) });
            }
        }
        143 => {
            kind = "declare a number";
            let n = r.u8()?;
            for i in 0..n {
                ch.push(Choice { text: format!("{}", r.u64()?), resp: i32r(i as i32) });
            }
        }
        _ => return Err(format!("unhandled decision message {t}")),
    }
    unselects.resize(ch.len(), None);
    let codes = MENTIONED.with(|m| std::mem::take(&mut *m.borrow_mut()));
    Ok(Decision { msg: t, player, kind, choices: ch, unselects, codes })
}

/// select_sum_check1 from playerop.cpp: every card uses one of its two values and they add up to acc exactly.
fn sum_check1(op: &[i32], index: usize, acc: i32) -> bool {
    if acc == 0 || index == op.len() {
        return false;
    }
    let (o1, o2) = (op[index] & 0xffff, op[index] >> 16);
    if index == op.len() - 1 {
        return acc == o1 || acc == o2;
    }
    (acc > o1 && sum_check1(op, index + 1, acc - o1)) || (o2 > 0 && acc > o2 && sum_check1(op, index + 1, acc - o2))
}

/// The "at least acc, nothing redundant" rule of the SelectSum branch with max == 0.
fn sum_at_least(op: &[i32], acc: i32) -> (bool, bool) {
    let (mut sum, mut mx, mut mn) = (0, 0, i32::MAX);
    for &p in op {
        let (o1, o2) = (p & 0xffff, p >> 16);
        let ms = if o2 != 0 && o2 < o1 { o2 } else { o1 };
        sum += ms;
        mx += if o2 > o1 { o2 } else { o1 };
        mn = mn.min(ms);
    }
    let ok = !(mx < acc || sum - mn >= acc);
    let dead = !op.is_empty() && sum - mn >= acc; // adding cards never lowers sum - mn
    (ok, dead)
}

#[allow(clippy::too_many_arguments)]
fn sum_dfs(
    order: &[usize], start: usize, cur: &mut Vec<usize>, cards: &[(String, i32)], must: &[i32], acc: i32,
    min: usize, max: usize, at_least: bool, found: &mut Vec<Vec<usize>>, budget: &mut usize,
) {
    if found.len() >= MAX_CHOICES || *budget == 0 {
        return;
    }
    *budget -= 1;
    if !cur.is_empty() {
        let mut op: Vec<i32> = must.to_vec();
        op.extend(cur.iter().map(|i| cards[*i].1));
        if at_least {
            let (ok, dead) = sum_at_least(&op, acc);
            if ok {
                found.push(cur.clone());
            }
            if dead {
                return;
            }
        } else {
            if cur.len() >= min && cur.len() <= max && sum_check1(&op, 0, acc) {
                found.push(cur.clone());
            }
            if cur.len() >= max {
                return;
            }
        }
    }
    for k in start..order.len() {
        cur.push(order[k]);
        sum_dfs(order, k + 1, cur, cards, must, acc, min, max, at_least, found, budget);
        cur.pop();
        if found.len() >= MAX_CHOICES || *budget == 0 {
            return;
        }
    }
}

const OPCODE_ADD: u64 = 0x4000000000000000;
const OPCODE_SUB: u64 = 0x4000000100000000;
const OPCODE_MUL: u64 = 0x4000000200000000;
const OPCODE_DIV: u64 = 0x4000000300000000;
const OPCODE_AND: u64 = 0x4000000400000000;
const OPCODE_OR: u64 = 0x4000000500000000;
const OPCODE_NEG: u64 = 0x4000000600000000;
const OPCODE_NOT: u64 = 0x4000000700000000;
const OPCODE_BAND: u64 = 0x4000000800000000;
const OPCODE_BOR: u64 = 0x4000000900000000;
const OPCODE_BNOT: u64 = 0x4000001000000000;
const OPCODE_BXOR: u64 = 0x4000001100000000;
const OPCODE_LSHIFT: u64 = 0x4000001200000000;
const OPCODE_RSHIFT: u64 = 0x4000001300000000;
const OPCODE_ALLOW_ALIASES: u64 = 0x4000001400000000;
const OPCODE_ALLOW_TOKENS: u64 = 0x4000001500000000;
const OPCODE_ISCODE: u64 = 0x4000010000000000;
const OPCODE_ISSETCARD: u64 = 0x4000010100000000;
const OPCODE_ISTYPE: u64 = 0x4000010200000000;
const OPCODE_ISRACE: u64 = 0x4000010300000000;
const OPCODE_ISATTRIBUTE: u64 = 0x4000010400000000;
const OPCODE_GETCODE: u64 = 0x4000010500000000;
const OPCODE_GETTYPE: u64 = 0x4000010700000000;
const OPCODE_GETRACE: u64 = 0x4000010800000000;
const OPCODE_GETATTRIBUTE: u64 = 0x4000010900000000;
const CARD_MARINE_DOLPHIN: u32 = 78734254;
const CARD_TWINKLE_MOSS: u32 = 13857930;

/// Port of is_declarable() in playerop.cpp (the stack machine behind "declare a card name").
fn is_declarable(cd: &crate::data::Card, ops: &[u64]) -> bool {
    let mut st: Vec<i64> = Vec::new();
    let (mut alias, mut token) = (false, false);
    macro_rules! bin {
        ($f:expr) => {{
            if st.len() >= 2 {
                let rhs = st.pop().unwrap();
                let lhs = st.pop().unwrap();
                st.push($f(lhs, rhs));
            }
        }};
    }
    macro_rules! un {
        ($f:expr) => {{
            if let Some(v) = st.pop() {
                st.push($f(v));
            }
        }};
    }
    for &op in ops {
        match op {
            OPCODE_ADD => bin!(|a: i64, b: i64| a.wrapping_add(b)),
            OPCODE_SUB => bin!(|a: i64, b: i64| a.wrapping_sub(b)),
            OPCODE_MUL => bin!(|a: i64, b: i64| a.wrapping_mul(b)),
            OPCODE_DIV => bin!(|a: i64, b: i64| if b == 0 { 0 } else { a.wrapping_div(b) }),
            OPCODE_AND => bin!(|a: i64, b: i64| (a != 0 && b != 0) as i64),
            OPCODE_OR => bin!(|a: i64, b: i64| (a != 0 || b != 0) as i64),
            OPCODE_NEG => un!(|v: i64| v.wrapping_neg()),
            OPCODE_NOT => un!(|v: i64| (v == 0) as i64),
            OPCODE_BAND => bin!(|a: i64, b: i64| a & b),
            OPCODE_BOR => bin!(|a: i64, b: i64| a | b),
            OPCODE_BXOR => bin!(|a: i64, b: i64| a ^ b),
            OPCODE_BNOT => un!(|v: i64| !v),
            OPCODE_LSHIFT => bin!(|a: i64, b: i64| a.wrapping_shl(b as u32)),
            OPCODE_RSHIFT => bin!(|a: i64, b: i64| a.wrapping_shr(b as u32)),
            OPCODE_ISCODE => un!(|v: i64| (cd.code == v as u32) as i64),
            OPCODE_ISTYPE => un!(|v: i64| (cd.type_ as i64) & v),
            OPCODE_ISRACE => un!(|v: i64| (cd.race as i64) & v),
            OPCODE_ISATTRIBUTE => un!(|v: i64| (cd.attribute as i64) & v),
            OPCODE_GETCODE => st.push(cd.code as i64),
            OPCODE_GETTYPE => st.push(cd.type_ as i64),
            OPCODE_GETRACE => st.push(cd.race as i64),
            OPCODE_GETATTRIBUTE => st.push(cd.attribute as i64),
            OPCODE_ISSETCARD => {
                if let Some(v) = st.pop() {
                    let set_code = v as i32;
                    let settype = (set_code & 0xfff) as u16;
                    let setsub = (set_code & 0xf000) as u16;
                    let hit = cd.setcodes.iter().take_while(|s| **s != 0).any(|&sc| (sc & 0xfff) == settype && (sc & 0xf000 & setsub) == setsub);
                    st.push(hit as i64);
                }
            }
            OPCODE_ALLOW_ALIASES => alias = true,
            OPCODE_ALLOW_TOKENS => token = true,
            other => st.push(other as i64),
        }
    }
    if st.len() != 1 || st[0] == 0 {
        return false;
    }
    cd.code == CARD_MARINE_DOLPHIN
        || cd.code == CARD_TWINKLE_MOSS
        || ((alias || cd.alias == 0) && (token || (cd.type_ & (TYPE_MONSTER | TYPE_TOKEN)) != (TYPE_MONSTER | TYPE_TOKEN)))
}
