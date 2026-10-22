//! Hand-written bindings for ocgapi.h / ocgapi_types.h (edo9300/ygopro-core, API 11.0).
#![allow(non_camel_case_types, dead_code)]
use std::os::raw::{c_char, c_int, c_void};

pub type OCG_Duel = *mut c_void;

#[repr(C)]
pub struct OCG_CardData {
    pub code: u32,
    pub alias: u32,
    pub setcodes: *mut u16,
    pub type_: u32,
    pub level: u32,
    pub attribute: u32,
    pub race: u64,
    pub attack: i32,
    pub defense: i32,
    pub lscale: u32,
    pub rscale: u32,
    pub link_marker: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OCG_Player {
    pub starting_lp: u32,
    pub starting_draw_count: u32,
    pub draw_count_per_turn: u32,
}

pub type OCG_DataReader = extern "C" fn(payload: *mut c_void, code: u32, data: *mut OCG_CardData);
pub type OCG_DataReaderDone = extern "C" fn(payload: *mut c_void, data: *mut OCG_CardData);
pub type OCG_ScriptReader = extern "C" fn(payload: *mut c_void, duel: OCG_Duel, name: *const c_char) -> c_int;
pub type OCG_LogHandler = extern "C" fn(payload: *mut c_void, string: *const c_char, type_: c_int);

#[repr(C)]
pub struct OCG_DuelOptions {
    pub seed: [u64; 4],
    pub flags: u64,
    pub team1: OCG_Player,
    pub team2: OCG_Player,
    pub card_reader: OCG_DataReader,
    pub payload1: *mut c_void,
    pub script_reader: OCG_ScriptReader,
    pub payload2: *mut c_void,
    pub log_handler: OCG_LogHandler,
    pub payload3: *mut c_void,
    pub card_reader_done: OCG_DataReaderDone,
    pub payload4: *mut c_void,
    pub enable_unsafe_libraries: u8,
}

#[repr(C)]
pub struct OCG_NewCardInfo {
    pub team: u8,
    pub duelist: u8,
    pub code: u32,
    pub con: u8,
    pub loc: u32,
    pub seq: u32,
    pub pos: u32,
}

pub const OCG_DUEL_CREATION_SUCCESS: c_int = 0;
pub const OCG_DUEL_STATUS_END: c_int = 0;
pub const OCG_DUEL_STATUS_AWAITING: c_int = 1;
pub const OCG_DUEL_STATUS_CONTINUE: c_int = 2;
pub const OCG_LOG_TYPE_ERROR: c_int = 0;

extern "C" {
    pub fn OCG_GetVersion(major: *mut c_int, minor: *mut c_int);
    pub fn OCG_CreateDuel(out: *mut OCG_Duel, options: *const OCG_DuelOptions) -> c_int;
    pub fn OCG_DestroyDuel(duel: OCG_Duel);
    pub fn OCG_DuelNewCard(duel: OCG_Duel, info: *const OCG_NewCardInfo);
    pub fn OCG_StartDuel(duel: OCG_Duel);
    pub fn OCG_DuelProcess(duel: OCG_Duel) -> c_int;
    pub fn OCG_DuelGetMessage(duel: OCG_Duel, length: *mut u32) -> *mut c_void;
    pub fn OCG_DuelSetResponse(duel: OCG_Duel, buffer: *const c_void, length: u32);
    pub fn OCG_LoadScript(duel: OCG_Duel, buffer: *const c_char, length: u32, name: *const c_char) -> c_int;
    pub fn OCG_DuelQueryCount(duel: OCG_Duel, team: u8, loc: u32) -> u32;
    pub fn OCG_DuelQueryLocation(duel: OCG_Duel, length: *mut u32, info: *const OCG_QueryInfo) -> *mut c_void;
}

#[repr(C)]
pub struct OCG_QueryInfo {
    pub flags: u32,
    pub con: u8,
    pub loc: u32,
    pub seq: u32,
    pub overlay_seq: u32,
}

// ocgapi_constants.h
pub const LOCATION_DECK: u32 = 0x01;
pub const LOCATION_HAND: u32 = 0x02;
pub const LOCATION_MZONE: u8 = 0x04;
pub const LOCATION_SZONE: u8 = 0x08;
pub const LOCATION_EXTRA: u32 = 0x40;
pub const LOCATION_GRAVE: u32 = 0x10;
pub const LOCATION_REMOVED: u32 = 0x20;
pub const QUERY_CODE: u32 = 0x1;
pub const QUERY_POSITION: u32 = 0x2;
pub const QUERY_TYPE: u32 = 0x8;
pub const QUERY_LEVEL: u32 = 0x10;
pub const QUERY_ATTACK: u32 = 0x100;
pub const QUERY_DEFENSE: u32 = 0x200;
pub const QUERY_IS_PUBLIC: u32 = 0x100000;
pub const QUERY_END: u32 = 0x80000000;
pub const TYPE_SPELL: u32 = 0x2;
pub const TYPE_TRAP: u32 = 0x4;
pub const POS_FACEDOWN_DEFENSE: u32 = 0x8;
pub const TYPE_MONSTER: u32 = 0x1;
pub const TYPE_TOKEN: u32 = 0x4000;
pub const TYPE_LINK: u32 = 0x4000000;
pub const DUEL_PZONE: u64 = 0x800;
pub const DUEL_EMZONE: u64 = 0x2000;
pub const DUEL_FSX_MMZONE: u64 = 0x4000;
pub const DUEL_TRAP_MONSTERS_NOT_USE_ZONE: u64 = 0x8000;
pub const DUEL_TRIGGER_ONLY_IN_LOCATION: u64 = 0x20000;
pub const DUEL_MODE_MR5: u64 =
    DUEL_PZONE | DUEL_EMZONE | DUEL_FSX_MMZONE | DUEL_TRAP_MONSTERS_NOT_USE_ZONE | DUEL_TRIGGER_ONLY_IN_LOCATION;
