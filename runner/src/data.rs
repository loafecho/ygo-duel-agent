//! Card data (ProjectIgnis BabelCDB, read with rusqlite) and card scripts (ProjectIgnis CardScripts),
//! plus the three callbacks the engine calls back into.
use crate::ffi::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct Card {
    pub code: u32,
    pub alias: u32,
    pub setcodes: Vec<u16>, // zero-terminated, the engine reads until 0
    pub type_: u32,
    pub level: u32,
    pub lscale: u32,
    pub rscale: u32,
    pub attribute: u32,
    pub race: u64,
    pub attack: i32,
    pub defense: i32,
    pub link_marker: u32,
    pub name: String,
    pub strs: Vec<String>,
    /// Effect text (texts.desc), newlines collapsed to spaces.
    pub text: String,
}

pub struct Library {
    pub cards: HashMap<u32, Card>,
    scripts: HashMap<String, PathBuf>,
}

/// Script errors the engine reported through the log handler (Lua errors in card scripts).
pub static SCRIPT_ERRORS: AtomicUsize = AtomicUsize::new(0);
pub static LAST_SCRIPT_ERROR: Mutex<String> = Mutex::new(String::new());

impl Library {
    pub fn load(cdbs: &[PathBuf], script_root: &Path) -> Result<Library, String> {
        let mut cards = HashMap::new();
        for db in cdbs {
            let conn = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .map_err(|e| format!("{}: {e}", db.display()))?;
            let cols: String = (1..=16).map(|i| format!(",t.str{i}")).collect();
            let sql = format!(
                "select d.id,d.alias,d.setcode,d.type,d.atk,d.def,d.level,d.race,d.attribute,t.name{cols},t.desc \
                 from datas d join texts t on t.id=d.id"
            );
            let mut st = conn.prepare(&sql).map_err(|e| e.to_string())?;
            let rows = st
                .query_map([], |r| {
                    let setcode: i64 = r.get(2)?;
                    let level: i64 = r.get(6)?;
                    let type_: i64 = r.get(3)?;
                    let def: i64 = r.get(5)?;
                    let mut setcodes: Vec<u16> =
                        (0..4).map(|i| ((setcode as u64 >> (16 * i)) & 0xffff) as u16).filter(|s| *s != 0).collect();
                    setcodes.push(0);
                    let link = (type_ as u32) & TYPE_LINK != 0;
                    let mut strs = Vec::with_capacity(16);
                    for i in 0..16 {
                        strs.push(r.get::<_, Option<String>>(10 + i)?.unwrap_or_default());
                    }
                    Ok(Card {
                        code: r.get::<_, i64>(0)? as u32,
                        alias: r.get::<_, i64>(1)? as u32,
                        setcodes,
                        type_: type_ as u32,
                        level: (level & 0xff) as u32,
                        lscale: ((level >> 24) & 0xff) as u32,
                        rscale: ((level >> 16) & 0xff) as u32,
                        attribute: r.get::<_, i64>(8)? as u32,
                        race: r.get::<_, i64>(7)? as u64,
                        attack: r.get::<_, i64>(4)? as i32,
                        defense: if link { 0 } else { def as i32 },
                        link_marker: if link { def as u32 } else { 0 },
                        name: r.get::<_, Option<String>>(9)?.unwrap_or_default(),
                        strs,
                        text: r.get::<_, Option<String>>(26)?.unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" "),
                    })
                })
                .map_err(|e| e.to_string())?;
            for c in rows {
                let c = c.map_err(|e| e.to_string())?;
                cards.entry(c.code).or_insert(c);
            }
        }
        // Script index by file name. Root files first, then official, then the rest: first one wins.
        let mut scripts = HashMap::new();
        let mut dirs = vec![script_root.to_path_buf(), script_root.join("official")];
        if let Ok(rd) = std::fs::read_dir(script_root) {
            let mut rest: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && !p.ends_with("official") && !p.file_name().unwrap().to_string_lossy().starts_with('.'))
                .collect();
            rest.sort();
            dirs.extend(rest);
        }
        for d in dirs {
            if let Ok(rd) = std::fs::read_dir(&d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().map(|x| x == "lua").unwrap_or(false) {
                        scripts.entry(p.file_name().unwrap().to_string_lossy().into_owned()).or_insert(p);
                    }
                }
            }
        }
        Ok(Library { cards, scripts })
    }

    pub fn name(&self, code: u32) -> String {
        self.cards.get(&code).map(|c| c.name.clone()).unwrap_or_else(|| format!("card {code}"))
    }

    /// Effect descriptions are aux.Stringid(code, i) = code<<20 | i; anything smaller is a system string.
    pub fn desc(&self, d: u64) -> String {
        if d > 0xfffff {
            let code = (d >> 20) as u32;
            let i = (d & 0xfffff) as usize;
            if let Some(c) = self.cards.get(&code) {
                if let Some(s) = c.strs.get(i).filter(|s| !s.is_empty()) {
                    return s.clone();
                }
                return format!("{} effect", c.name);
            }
        }
        if d == 0 { String::new() } else { format!("system string {d}") }
    }

    /// "Name: effect text" for the prompt's card-text section.
    pub fn text_line(&self, code: u32) -> String {
        match self.cards.get(&code) {
            Some(c) if !c.text.is_empty() => format!("{}: {}", c.name, c.text),
            _ => self.name(code),
        }
    }

    pub fn has_script(&self, name: &str) -> bool {
        self.scripts.contains_key(name)
    }
}

pub extern "C" fn card_reader(payload: *mut c_void, code: u32, data: *mut OCG_CardData) {
    let lib = unsafe { &*(payload as *const Library) };
    let out = unsafe { &mut *data };
    match lib.cards.get(&code) {
        Some(c) => {
            out.code = c.code;
            out.alias = c.alias;
            out.setcodes = c.setcodes.as_ptr() as *mut u16; // lives as long as the Library
            out.type_ = c.type_;
            out.level = c.level;
            out.attribute = c.attribute;
            out.race = c.race;
            out.attack = c.attack;
            out.defense = c.defense;
            out.lscale = c.lscale;
            out.rscale = c.rscale;
            out.link_marker = c.link_marker;
        }
        None => {
            *out = OCG_CardData {
                code: 0, alias: 0, setcodes: std::ptr::null_mut(), type_: 0, level: 0, attribute: 0, race: 0,
                attack: 0, defense: 0, lscale: 0, rscale: 0, link_marker: 0,
            };
        }
    }
}

pub extern "C" fn card_reader_done(_payload: *mut c_void, _data: *mut OCG_CardData) {}

pub extern "C" fn script_reader(payload: *mut c_void, duel: OCG_Duel, name: *const c_char) -> c_int {
    let lib = unsafe { &*(payload as *const Library) };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    let file = Path::new(&name).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(name.clone());
    let Some(path) = lib.scripts.get(&file) else { return 0 };
    let Ok(buf) = std::fs::read(path) else { return 0 };
    let cname = CString::new(name).unwrap();
    unsafe { OCG_LoadScript(duel, buf.as_ptr() as *const c_char, buf.len() as u32, cname.as_ptr()) }
}

/// payload: this duel's own error counter (games run in parallel), or null.
pub extern "C" fn log_handler(payload: *mut c_void, s: *const c_char, type_: c_int) {
    if type_ == OCG_LOG_TYPE_ERROR {
        SCRIPT_ERRORS.fetch_add(1, Ordering::Relaxed);
        if !payload.is_null() {
            unsafe { &*(payload as *const AtomicUsize) }.fetch_add(1, Ordering::Relaxed);
        }
        let msg = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
        if let Ok(mut l) = LAST_SCRIPT_ERROR.lock() {
            *l = msg;
        }
    }
}

/// .ydk deck file: codes between #main and #extra are the main deck, #extra..!side the extra deck.
pub fn read_ydk(path: &Path) -> Result<(Vec<u32>, Vec<u32>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut main, mut extra) = (Vec::new(), Vec::new());
    let mut section = 0;
    for line in text.lines().map(str::trim) {
        match line {
            "#main" => section = 1,
            "#extra" => section = 2,
            "!side" => section = 3,
            l if l.starts_with('#') || l.is_empty() => {}
            l => {
                if let Ok(code) = l.parse::<u32>() {
                    match section {
                        1 => main.push(code),
                        2 => extra.push(code),
                        _ => {}
                    }
                }
            }
        }
    }
    Ok((main, extra))
}
