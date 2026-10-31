//! A WindBot seat. WindBot (ProjectIgnis/windbot) is a network client of an EDOPro host, so for each game
//! this seat plays the host's part on a loopback port: it asks the WindBot server (ServerMode, HTTP :2399)
//! to send a bot, does the join handshake, then forwards the engine's messages the way EDOPro's host does
//! (decisions and private hints only to their player, the other player's hidden card codes zeroed) and
//! refreshes the field with MSG_UPDATE_DATA before each of the bot's decisions. The bot's CTOS_RESPONSE
//! bytes go straight to OCG_DuelSetResponse.
//! Any local process can see and connect to the loopback port, so each call carries a fresh room password
//! (WindBot's "password" parameter, sent back in CTOS_JOIN_GAME) and a connection without it is dropped.
use crate::decide;
use crate::ffi::*;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const CTOS_RESPONSE: u8 = 0x01;
const CTOS_UPDATE_DECK: u8 = 0x02;
const CTOS_PLAYER_INFO: u8 = 0x10;
const CTOS_JOIN_GAME: u8 = 0x12;
const CTOS_HS_READY: u8 = 0x22;
const CTOS_TIME_CONFIRM: u8 = 0x15;
const CTOS_CHAT: u8 = 0x16;
const STOC_GAME_MSG: u8 = 0x01;
const STOC_JOIN_GAME: u8 = 0x12;
const STOC_TYPE_CHANGE: u8 = 0x13;
const STOC_DUEL_START: u8 = 0x15;
const STOC_DUEL_END: u8 = 0x16;
const MSG_START: u8 = 4;
const MSG_UPDATE_DATA: u8 = 6;
const MSG_SHUFFLE_HAND: u8 = 33;
const MSG_MOVE: u8 = 50;
const MSG_SET: u8 = 54;
const MSG_DRAW: u8 = 90;
const MSG_MISSED_EFFECT: u8 = 120;
const HANDSHAKE: u32 = 4043399681;
const ROOM_PASS_LEN: usize = 16;
const POS_FACEDOWN: u32 = 0xA;
/// The query fields WindBot's ClientCard.Update reads. Any other field would hit its ReadChars fallback,
/// which decodes text and can misread binary, so only these are sent.
const WB_FIELDS: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800
    | 0x10000 | 0x40000 | 0x80000 | 0x200000 | 0x400000 | 0x800000;
const QUERY_ALL: u32 = WB_FIELDS | QUERY_IS_PUBLIC;

pub struct WindBot {
    pub seat: u8,
    s: TcpStream,
    /// Decisions where the engine rejected the bot's answer and the seat answered randomly instead.
    pub fallbacks: u32,
    pub decisions: u32,
}

fn io<E: std::fmt::Display>(what: &'static str) -> impl Fn(E) -> String {
    move |e| format!("windbot {what}: {e}")
}

impl WindBot {
    /// Calls a bot from the WindBot server at `server` (host:port) and completes the room handshake.
    pub fn connect(server: &str, deck: &str, seat: u8, name: &str) -> Result<WindBot, String> {
        let l = TcpListener::bind("127.0.0.1:0").map_err(io("listen"))?;
        let port = l.local_addr().map_err(io("listen"))?.port();
        let pass = &crate::broker::token()[..ROOM_PASS_LEN]; // 64 random bits; WindBot sends at most 19 characters
        let mut h = TcpStream::connect(server).map_err(io("server connect"))?;
        h.set_read_timeout(Some(Duration::from_secs(10))).ok();
        write!(h, "GET /?name={name}&deck={deck}&host=127.0.0.1&port={port}&password={pass} HTTP/1.0\r\nHost: {server}\r\n\r\n")
            .map_err(io("server request"))?;
        let mut resp = String::new();
        let _ = h.read_to_string(&mut resp);
        if !resp.starts_with("HTTP/1.1 200") && !resp.starts_with("HTTP/1.0 200") {
            return Err(format!("windbot server answered {:?}", resp.lines().next().unwrap_or("")));
        }
        l.set_nonblocking(true).map_err(io("listen"))?;
        let t0 = Instant::now();
        let mut wb = loop {
            match l.accept() {
                Ok((mut s, _)) => {
                    // Only the bot we called knows the password; anything else is dropped and we keep waiting.
                    s.set_nonblocking(false).map_err(io("socket"))?;
                    s.set_nodelay(true).ok();
                    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
                    match expect_on(&mut s, CTOS_PLAYER_INFO).and_then(|_| expect_on(&mut s, CTOS_JOIN_GAME)) {
                        Ok(join) if join_password(&join).as_deref() == Some(pass) => break WindBot { seat, s, fallbacks: 0, decisions: 0 },
                        r => {
                            eprintln!("windbot seat {seat}: dropped a loopback connection without the room password ({:?})", r.err());
                            let _ = s.shutdown(std::net::Shutdown::Both);
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(format!("windbot accept: {e}")),
            }
            if t0.elapsed() > Duration::from_secs(30) {
                return Err("windbot never connected with the room password (30 s)".into());
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        wb.s.set_read_timeout(Some(Duration::from_secs(60))).ok();
        // HostInfo as OnJoinGame reads it: banlist, rule, mode, duel_rule, no-check, no-shuffle, pad 3, lp,
        // hand, draw, time limit, pad 4, handshake, version, team sizes, best-of, duel flags, forbidden, extra.
        let mut p = Vec::new();
        p.extend(0u32.to_le_bytes());
        p.extend([0u8, 0, 5, 1, 0, 0, 0, 0]);
        p.extend(8000u32.to_le_bytes());
        p.extend([5u8, 1]);
        p.extend(0u16.to_le_bytes());
        p.extend([0u8; 4]);
        p.extend(HANDSHAKE.to_le_bytes());
        p.extend(0u32.to_le_bytes());
        for v in [1i32, 1, 1, (DUEL_MODE_MR5 & 0xffff_ffff) as i32, 0, 0] {
            p.extend(v.to_le_bytes());
        }
        wb.send(STOC_JOIN_GAME, &p)?;
        wb.expect(CTOS_UPDATE_DECK)?;
        wb.send(STOC_TYPE_CHANGE, &[seat])?; // its position, not the host (so it never sends HsStart)
        wb.expect(CTOS_HS_READY)?;
        wb.send(STOC_DUEL_START, &[])?;
        Ok(wb)
    }

    fn send(&mut self, id: u8, payload: &[u8]) -> Result<(), String> {
        let n = payload.len() + 1;
        if n > 0xffff {
            return Err(format!("windbot packet too large ({n} bytes)"));
        }
        let mut b = Vec::with_capacity(n + 2);
        b.extend((n as u16).to_le_bytes());
        b.push(id);
        b.extend(payload);
        self.s.write_all(&b).map_err(io("send"))
    }

    fn game_msg(&mut self, msg: &[u8]) -> Result<(), String> {
        self.send(STOC_GAME_MSG, msg)
    }

    fn read(&mut self) -> Result<(u8, Vec<u8>), String> {
        read_on(&mut self.s)
    }

    fn expect(&mut self, id: u8) -> Result<Vec<u8>, String> {
        expect_on(&mut self.s, id)
    }

    /// MSG_START: who goes first (for this bot), both life points, deck and extra deck sizes.
    pub fn start(&mut self, duel: OCG_Duel) -> Result<(), String> {
        let mut m = vec![MSG_START, self.seat];
        m.extend(8000i32.to_le_bytes());
        m.extend(8000i32.to_le_bytes());
        for p in 0..2u8 {
            for loc in [LOCATION_DECK, LOCATION_EXTRA] {
                m.extend((unsafe { OCG_DuelQueryCount(duel, p, loc) } as i16).to_le_bytes());
            }
        }
        self.game_msg(&m)
    }

    /// One engine message, filtered the way EDOPro's host filters it for this player.
    pub fn observe(&mut self, msg: &[u8]) -> Result<(), String> {
        let me = self.seat;
        let t = msg[0];
        if decide::is_decision(t) || t == decide::MSG_RETRY {
            return Ok(()); // its own decisions go through respond(); retries are handled there
        }
        let mut m = msg.to_vec();
        match t {
            decide::MSG_HINT if m.len() >= 3 => {
                let (kind, player) = (m[1], m[2] & 1);
                match kind {
                    1 | 2 | 3 | 5 if player != me => return Ok(()),
                    4 | 6 | 7 | 8 | 9 if player == me => return Ok(()),
                    _ => {}
                }
            }
            MSG_MISSED_EFFECT if m.len() >= 2 && m[1] & 1 != me => return Ok(()),
            MSG_DRAW if m.len() >= 6 && m[1] & 1 != me => {
                let n = u32::from_le_bytes(m[2..6].try_into().unwrap()) as usize;
                for i in 0..n {
                    let at = 6 + i * 8;
                    if at + 8 > m.len() {
                        break;
                    }
                    let pos = u32::from_le_bytes(m[at + 4..at + 8].try_into().unwrap());
                    if pos & 0x5 == 0 {
                        m[at..at + 4].fill(0);
                    }
                }
            }
            MSG_SHUFFLE_HAND if m.len() >= 6 && m[1] & 1 != me => {
                for b in &mut m[6..] {
                    *b = 0;
                }
            }
            // code, previous loc_info (10 bytes), current loc_info: controller, location, sequence u32, position u32
            MSG_MOVE if m.len() >= 25 => {
                let (con, loc) = (m[15] & 1, m[16] as u32);
                let pos = u32::from_le_bytes(m[21..25].try_into().unwrap());
                let hidden = loc & (LOCATION_GRAVE | 0x80) == 0
                    && (loc & (LOCATION_DECK | LOCATION_HAND) != 0 || pos & POS_FACEDOWN != 0);
                if con != me && hidden {
                    m[1..5].fill(0);
                }
            }
            MSG_SET if m.len() >= 6 && m[5] & 1 != me => m[1..5].fill(0),
            _ => {}
        }
        self.game_msg(&m)
    }

    /// The field as this bot may see it: everything of its own, the opponent's public cards only.
    fn refresh(&mut self, duel: OCG_Duel) -> Result<(), String> {
        let me = self.seat;
        let locs: [(u8, u32); 11] = [
            (me, LOCATION_HAND), (me, LOCATION_MZONE as u32), (me, LOCATION_SZONE as u32), (me, LOCATION_GRAVE),
            (me, LOCATION_REMOVED), (me, LOCATION_EXTRA),
            (1 - me, LOCATION_HAND), (1 - me, LOCATION_MZONE as u32), (1 - me, LOCATION_SZONE as u32),
            (1 - me, LOCATION_GRAVE), (1 - me, LOCATION_REMOVED),
        ];
        for (p, loc) in locs {
            let info = OCG_QueryInfo { flags: QUERY_ALL, con: p, loc, seq: 0, overlay_seq: 0 };
            let mut len = 0u32;
            let ptr = unsafe { OCG_DuelQueryLocation(duel, &mut len, &info) } as *const u8;
            if len < 4 {
                return Err(format!("windbot refresh query {p}/{loc:#x}: {len} bytes"));
            }
            let raw = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
            let body = filter_query(&raw[4..], p != me, loc)?;
            let mut m = vec![MSG_UPDATE_DATA, p, loc as u8];
            m.extend((body.len() as u32 + 4).to_le_bytes());
            m.extend(body);
            self.game_msg(&m)?;
        }
        Ok(())
    }

    /// Sends the bot its decision (after a field refresh) and returns its answer bytes.
    pub fn respond(&mut self, duel: OCG_Duel, msg: &[u8]) -> Result<Vec<u8>, String> {
        self.decisions += 1;
        self.refresh(duel)?;
        self.game_msg(msg)?;
        loop {
            let (id, b) = self.read()?;
            if id == CTOS_RESPONSE {
                return Ok(b);
            }
            // chat, time confirm and the like: nothing to do
        }
    }

}

impl Drop for WindBot {
    /// Ends the bot's duel however the game ended (MSG_WIN was already forwarded by observe).
    fn drop(&mut self) {
        let _ = self.send(STOC_DUEL_END, &[]);
        let _ = self.s.shutdown(std::net::Shutdown::Both);
    }
}

fn read_on(s: &mut TcpStream) -> Result<(u8, Vec<u8>), String> {
    let mut h = [0u8; 2];
    s.read_exact(&mut h).map_err(io("read"))?;
    let n = u16::from_le_bytes(h) as usize;
    if n == 0 {
        return Err("windbot sent an empty packet".into());
    }
    let mut b = vec![0u8; n];
    s.read_exact(&mut b).map_err(io("read"))?;
    Ok((b[0], b[1..].to_vec()))
}

fn expect_on(s: &mut TcpStream, id: u8) -> Result<Vec<u8>, String> {
    let (mut got, mut b) = read_on(s)?;
    while got == CTOS_CHAT || got == CTOS_TIME_CONFIRM {
        (got, b) = read_on(s)?;
    }
    if got != id {
        return Err(format!("windbot sent packet {got:#x}, expected {id:#x}"));
    }
    Ok(b)
}

/// The room password in a CTOS_JOIN_GAME payload as WindBot's GameClient writes it: version (2), padding (2),
/// room id (4), then 20 UTF-16LE characters, NUL-padded.
fn join_password(b: &[u8]) -> Option<String> {
    let raw = b.get(8..48)?;
    let units: Vec<u16> = raw.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&u| u != 0).collect();
    String::from_utf16(&units).ok()
}

/// Re-encodes a location query: only the fields WindBot reads, and for the opponent's hidden cards
/// (face-down, or in hand and not revealed) only the position.
fn filter_query(b: &[u8], opponent: bool, loc: u32) -> Result<Vec<u8>, String> {
    let rd16 = |p: usize| b.get(p..p + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap())).ok_or("query truncated");
    let rd32 = |p: usize| b.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or("query truncated");
    let mut out = Vec::new();
    let mut p = 0usize;
    while p < b.len() {
        if rd16(p)? == 0 {
            out.extend([0u8, 0]);
            p += 2;
            continue;
        }
        let mut fields: Vec<(u32, &[u8])> = Vec::new();
        let (mut pos, mut public) = (0u32, false);
        loop {
            let size = rd16(p)? as usize;
            let flag = rd32(p + 2)?;
            let v = b.get(p + 6..p + 2 + size).ok_or("query truncated")?;
            match flag {
                QUERY_POSITION => pos = rd32(p + 6)?,
                QUERY_IS_PUBLIC => public = v.first().copied().unwrap_or(0) != 0,
                _ => {}
            }
            p += 2 + size;
            if flag == QUERY_END {
                break;
            }
            fields.push((flag, v));
        }
        let in_hand = loc & (LOCATION_HAND | LOCATION_EXTRA) != 0;
        let hidden = opponent && !public && (in_hand || pos & POS_FACEDOWN != 0);
        for (flag, v) in fields {
            if flag & WB_FIELDS == 0 || flag.count_ones() != 1 || (hidden && flag != QUERY_POSITION) {
                continue;
            }
            out.extend(((v.len() + 4) as u16).to_le_bytes());
            out.extend(flag.to_le_bytes());
            out.extend(v);
        }
        out.extend(4u16.to_le_bytes());
        out.extend(QUERY_END.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(id: u8, payload: &[u8]) -> Vec<u8> {
        let mut b = ((payload.len() + 1) as u16).to_le_bytes().to_vec();
        b.push(id);
        b.extend(payload);
        b
    }

    /// CTOS_PLAYER_INFO then CTOS_JOIN_GAME laid out the way WindBot's GameClient writes them.
    fn join(c: &mut TcpStream, pass: &str) {
        c.write_all(&packet(CTOS_PLAYER_INFO, &[0u8; 40])).unwrap();
        let mut j = vec![0x40, 0x13, 0xAA, 0xBB, 0, 0, 0, 0];
        let mut room = [0u8; 40];
        for (i, u) in pass.encode_utf16().take(19).enumerate() {
            room[2 * i..2 * i + 2].copy_from_slice(&u.to_le_bytes());
        }
        j.extend(room);
        j.extend(0x1340u32.to_le_bytes());
        c.write_all(&packet(CTOS_JOIN_GAME, &j)).unwrap();
    }

    fn param<'a>(req: &'a str, k: &str) -> &'a str {
        let q = req.split_whitespace().nth(1).unwrap();
        q.split(['?', '&']).find_map(|kv| kv.strip_prefix(k)).unwrap()
    }

    /// A stub WindBot server: answers 200, lets an impostor connect first, then joins with the password it was given.
    #[test]
    fn a_loopback_peer_without_the_room_password_is_dropped() {
        let srv = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = srv.local_addr().unwrap().to_string();
        let bot = std::thread::spawn(move || {
            let (mut h, _) = srv.accept().unwrap();
            let mut req = String::new();
            let mut buf = [0u8; 1024];
            while !req.ends_with("\r\n\r\n") {
                let n = h.read(&mut buf).unwrap();
                assert!(n > 0, "request cut short: {req:?}");
                req.push_str(&String::from_utf8_lossy(&buf[..n]));
            }
            let (port, pass) = (param(&req, "port=").parse::<u16>().unwrap(), param(&req, "password=").to_string());
            assert_eq!(pass.len(), ROOM_PASS_LEN);
            h.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").unwrap();
            drop(h);
            let mut impostor = TcpStream::connect(("127.0.0.1", port)).unwrap();
            join(&mut impostor, "not-the-password");
            let mut rest = Vec::new();
            impostor.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let _ = impostor.read_to_end(&mut rest);
            assert!(rest.is_empty(), "the impostor got no STOC_JOIN_GAME");
            let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
            join(&mut c, &pass);
            assert_eq!(read_on(&mut c).unwrap().0, STOC_JOIN_GAME);
            c.write_all(&packet(CTOS_UPDATE_DECK, &[])).unwrap();
            assert_eq!(read_on(&mut c).unwrap(), (STOC_TYPE_CHANGE, vec![1]));
            c.write_all(&packet(CTOS_HS_READY, &[])).unwrap();
            assert_eq!(read_on(&mut c).unwrap().0, STOC_DUEL_START);
            c
        });
        let wb = WindBot::connect(&addr, "CyberDragon", 1, "WindBot").unwrap();
        assert_eq!(wb.seat, 1);
        drop(bot.join().unwrap());
    }

    #[test]
    fn join_password_reads_the_room_field() {
        let mut j = vec![0u8; 8];
        let mut room = [0u8; 40];
        for (i, u) in "abc".encode_utf16().enumerate() {
            room[2 * i..2 * i + 2].copy_from_slice(&u.to_le_bytes());
        }
        j.extend(room);
        assert_eq!(join_password(&j).as_deref(), Some("abc"));
        assert_eq!(join_password(&j[..20]), None);
    }
}
