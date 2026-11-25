//! Replacements for the console's system calls.
//!
//! A title reaches the kernel (`xboxkrnl.exe`) and the system library
//! (`xam.xex`) only through imports by number. Each imported function has a
//! thunk in the image; [`install`] puts a host function on every thunk
//! address and writes the imported variables. Arguments arrive in r3..r10
//! (more on the guest stack), the result goes back in r3.
//!
//! Guest address layout used here:
//!
//! | Range | Use |
//! |---|---|
//! | `0x00100000..0x3F000000` | kernel pool: kernel structures, pool and system-library allocations |
//! | `0x40000000..0x7F000000` | virtual memory and thread stacks, 64 KB granularity |
//! | `0x7FC80000` | GPU registers (plain memory, polled by [`video`]) |
//! | `0x82000000..` | the executable image |
//! | `0xA0000000..0xC0000000` | physical memory, 4 KB granularity |
//!
//! Physical memory is not mirrored at `0xC0000000` and `0xE0000000`; every
//! physical allocation is handed out in the `0xA0000000` view.

pub mod audio;
pub mod files;
pub mod gpu;
pub mod input;
pub mod memory;
pub mod net;
pub mod objects;
pub mod profile;
pub mod rtl;
pub mod sound;
pub mod table;
pub mod threads;
pub mod video;
pub mod xam;

use crate::heap::Heap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use xenon_rt::{Context, GuestFn, Runtime};

pub const STATUS_SUCCESS: u32 = 0;
pub const STATUS_TIMEOUT: u32 = 0x102;
pub const STATUS_PENDING: u32 = 0x103;
pub const STATUS_NO_MORE_FILES: u32 = 0x8000_0006;
pub const STATUS_UNSUCCESSFUL: u32 = 0xC000_0001;
pub const STATUS_NOT_IMPLEMENTED: u32 = 0xC000_0002;
pub const STATUS_INVALID_HANDLE: u32 = 0xC000_0008;
pub const STATUS_INVALID_PARAMETER: u32 = 0xC000_000D;
pub const STATUS_NO_SUCH_FILE: u32 = 0xC000_000F;
pub const STATUS_END_OF_FILE: u32 = 0xC000_0011;
pub const STATUS_NO_MEMORY: u32 = 0xC000_0017;
pub const STATUS_ACCESS_DENIED: u32 = 0xC000_0022;
pub const STATUS_OBJECT_NAME_NOT_FOUND: u32 = 0xC000_0034;
pub const STATUS_OBJECT_NAME_COLLISION: u32 = 0xC000_0035;
pub const STATUS_OBJECT_PATH_NOT_FOUND: u32 = 0xC000_003A;
pub const STATUS_MUTANT_NOT_OWNED: u32 = 0xC000_0046;
pub const STATUS_SEMAPHORE_LIMIT_EXCEEDED: u32 = 0xC000_0047;

pub const ERROR_SUCCESS: u32 = 0;
pub const ERROR_NO_MORE_FILES: u32 = 18;
pub const ERROR_IO_PENDING: u32 = 997;
pub const ERROR_DEVICE_NOT_CONNECTED: u32 = 1167;
pub const ERROR_NOT_LOGGED_ON: u32 = 1245;
pub const ERROR_NO_SUCH_USER: u32 = 1317;
pub const ERROR_FUNCTION_FAILED: u32 = 1627;

/// Handles: a guest object (by the guest address of its dispatcher header),
/// an open file, or something only the host tracks.
#[derive(Clone)]
pub enum Handle {
    Object(u32),
    File(std::sync::Arc<Mutex<files::File>>),
    Other(u32),
}

/// The pseudo handle for the calling thread.
pub const CURRENT_THREAD: u32 = 0xFFFF_FFFE;

pub struct Kernel {
    pub game_dir: PathBuf,
    pub data_dir: PathBuf,
    pub pool: Mutex<Heap>,
    pub virt: Mutex<Heap>,
    pub phys: Mutex<Heap>,
    handles: Mutex<HashMap<u32, Handle>>,
    next_handle: AtomicU32,
    /// Guest copy of the executable's header, for header-field queries.
    pub xex_header: u32,
    /// Guest module record (`XexExecutableModuleHandle` points at it).
    pub module: u32,
    /// Thread-local storage layout from the executable header.
    pub tls_slots: u32,
    pub tls_data: (u32, u32, u32),
    pub default_stack: u32,
    pub threads: Mutex<HashMap<u32, threads::ThreadInfo>>,
    pub tls_used: Mutex<Vec<bool>>,
    pub links: Mutex<Vec<(String, PathBuf, bool)>>,
    /// Import calls by thunk address: (name, count).
    pub calls: Mutex<HashMap<u32, (String, u64)>>,
    pub swaps: AtomicU64,
    pub video: Mutex<video::Video>,
    pub time_stamp_bundle: u32,
}

static K: OnceLock<Kernel> = OnceLock::new();

pub fn k() -> &'static Kernel {
    K.get().expect("kernel not installed")
}

/// Argument `i` (0 = r3). Arguments past the eighth are on the guest
/// stack, one 8-byte slot each with a 32-bit value in its low word.
pub fn arg(rt: &Runtime, c: &Context, i: usize) -> u32 {
    if i < 8 {
        c.r[3 + i] as u32
    } else {
        rt.mem
            .read_u32((c.r[1] as u32).wrapping_add(0x54 + 8 * (i as u32 - 8)))
    }
}

/// Sets the 32-bit result. Like the console's kernel it is sign-extended
/// to the full register: titles test results with 64-bit record compares
/// (`or. rX, r3, r3`), so a failure status must read as negative.
pub fn ret(c: &mut Context, v: u32) {
    c.r[3] = v as i32 as i64 as u64;
}

impl Kernel {
    pub fn add_handle(&self, h: Handle) -> u32 {
        let id = self.next_handle.fetch_add(4, Ordering::Relaxed);
        self.handles.lock().unwrap().insert(id, h);
        id
    }

    pub fn handle(&self, id: u32) -> Option<Handle> {
        self.handles.lock().unwrap().get(&id).cloned()
    }

    /// Removes handle `id`; gives what it held.
    pub fn close(&self, id: u32) -> Option<Handle> {
        self.handles.lock().unwrap().remove(&id)
    }

    /// Whether any open handle refers to the object at `a`.
    pub fn referenced(&self, a: u32) -> bool {
        self.handles
            .lock()
            .unwrap()
            .values()
            .any(|h| matches!(h, Handle::Object(x) if *x == a))
    }

    /// Open handles, for the memory line of the run log.
    pub fn handle_count(&self) -> usize {
        self.handles.lock().unwrap().len()
    }

    /// Kernel pool allocation, zeroed.
    pub fn pool_alloc(&self, rt: &Runtime, size: u32) -> u32 {
        let r = self.pool.lock().unwrap().alloc(size, 16);
        let (a, reused) = r.unwrap_or_else(|| panic!("kernel pool exhausted ({size} bytes asked)"));
        if reused {
            fill(rt, a, size, 0);
        }
        a
    }

    pub fn usage_line(&self) -> String {
        let mb = |h: &Mutex<Heap>| {
            let (u, p, t) = h.lock().unwrap().usage();
            format!(
                "{} {}/{} MB (peak {})",
                h.lock().unwrap().name,
                u >> 20,
                t >> 20,
                p >> 20
            )
        };
        // The pool is small and leaks there are small, so it is in KB.
        let (u, p, t) = self.pool.lock().unwrap().usage();
        let pool = format!("pool {}/{} KB (peak {})", u >> 10, t >> 10, p >> 10);
        format!("{}, {}, {pool}", mb(&self.virt), mb(&self.phys))
    }
}

/// Fills guest memory.
pub fn fill(rt: &Runtime, addr: u32, len: u32, byte: u8) {
    let w = u64::from_ne_bytes([byte; 8]);
    let mut a = addr;
    let end = addr + len;
    while a < end && a % 8 != 0 {
        rt.mem.write_u8(a, byte);
        a += 1;
    }
    while a + 8 <= end {
        rt.mem.write_u64(a, w);
        a += 8;
    }
    while a < end {
        rt.mem.write_u8(a, byte);
        a += 1;
    }
}

/// Reads a zero-terminated byte string.
pub fn cstr(rt: &Runtime, mut a: u32) -> String {
    let mut s = Vec::new();
    while a != 0 && s.len() < 4096 {
        let b = rt.mem.read_u8(a);
        if b == 0 {
            break;
        }
        s.push(b);
        a += 1;
    }
    String::from_utf8_lossy(&s).into_owned()
}

/// Reads a counted string (`ANSI_STRING`: length, maximum, buffer).
pub fn ansi(rt: &Runtime, p: u32) -> String {
    if p == 0 {
        return String::new();
    }
    let len = rt.mem.read_u16(p) as usize;
    let buf = rt.mem.read_u32(p + 4);
    let mut b = vec![0; len];
    rt.mem.read_bytes(buf, &mut b);
    String::from_utf8_lossy(&b).into_owned()
}

pub fn write_cstr(rt: &Runtime, a: u32, s: &str) {
    rt.mem.write_bytes(a, s.as_bytes());
    rt.mem.write_u8(a + s.len() as u32, 0);
}

/// What the host was given to run.
pub struct Setup<'a> {
    pub file: &'a [u8],
    pub image: &'a xenon_recomp::xex::Image,
    pub game_dir: PathBuf,
    pub data_dir: PathBuf,
    /// Unpacked title update, read as `update:`.
    pub update_dir: Option<PathBuf>,
}

const POOL: (u32, u32) = (0x0010_0000, 0x3F00_0000);
const VIRT: (u32, u32) = (0x4000_0000, 0x7F00_0000);
const PHYS: (u32, u32) = (0xA000_0000, 0xC000_0000);

/// The function every unknown import gets: stops with its name.
fn missing(_: &Runtime, _: &mut Context) {
    let at = crate::last_call();
    let name = k()
        .calls
        .lock()
        .unwrap()
        .get(&at)
        .map(|x| x.0.clone())
        .unwrap_or_default();
    panic!("unimplemented import {name} (thunk {at:#010x})");
}

/// Writes the kernel's guest structures and puts host functions on every
/// import. Call before [`Runtime::share`].
pub fn install(rt: &mut Runtime, s: &Setup) {
    let h = xenon_recomp::xex::Header::parse(s.file).expect("executable header");
    // Fixed ranges outside the heaps (no-ops without guarded memory).
    xenon_rt::guard::commit(s.image.base, s.image.data.len() as u32);
    xenon_rt::guard::commit(video::GPU_REGS, 0x8_0000);
    let mut pool = Heap::new("pool", POOL.0, POOL.1, 0x10);
    let mut alloc = |size: u32| pool.alloc(size, 16).unwrap().0;

    // The header, as the loader leaves it in memory.
    let xex_header = alloc(h.header_size);
    rt.mem
        .write_bytes(xex_header, &s.file[..h.header_size as usize]);
    let tls = h
        .data(s.file, 0x0002_0104)
        .ok()
        .flatten()
        .map(|(o, _)| {
            let r =
                |i: usize| u32::from_be_bytes(s.file[o + 4 * i..o + 4 * i + 4].try_into().unwrap());
            (r(0), (r(1), r(2), r(3)))
        })
        .unwrap_or((64, (0, 0, 0)));
    let default_stack = h.value(0x0002_0200).unwrap_or(256 << 10);

    // Module record: the image base at 0x18, entry point at 0x1C, image
    // size at 0x20, the header at 0x58.
    let module = alloc(0x100);
    rt.mem.write_u32(module + 0x18, s.image.base);
    rt.mem.write_u32(module + 0x1C, s.image.entry);
    rt.mem.write_u32(module + 0x20, s.image.data.len() as u32);
    rt.mem.write_u32(module + 0x58, xex_header);

    let mut calls = HashMap::new();
    let mut vars = HashMap::new();
    let mut tsb = 0;
    for i in &s.image.imports {
        let (tab, short) = match i.library.as_str() {
            "xboxkrnl.exe" => (table::XBOXKRNL, "xboxkrnl"),
            "xam.xex" => (table::XAM, "xam"),
            _ => (&[][..], "?"),
        };
        let found = tab.iter().find(|e| e.0 == i.ordinal);
        let name = found.map_or(format!("{short}#{}", i.ordinal), |e| e.1.to_string());
        if i.thunk {
            rt.set_override(i.address, found.map_or(missing as GuestFn, |e| e.2));
            calls.insert(i.address, (name, 0));
        } else if i.library == "xboxkrnl.exe" {
            vars.insert(i.ordinal, i.address);
        }
    }
    // Imported variables: the slot receives the variable's guest address.
    for (&ord, &slot) in &vars {
        let a = match ord {
            // XboxKrnlVersion: 2.0.17559.0
            344 => {
                let a = alloc(8);
                rt.mem.write_u64(a, 0x0002_0000_4497_0000);
                a
            }
            // XexExecutableModuleHandle: a pointer to the module record.
            403 => {
                let a = alloc(4);
                rt.mem.write_u32(a, module);
                a
            }
            // ExLoadedCommandLine
            430 => {
                let a = alloc(1024);
                write_cstr(rt, a, "\"default.xex\"");
                a
            }
            // VdGpuClockInMHz
            448 => {
                let a = alloc(4);
                rt.mem.write_u32(a, 500);
                a
            }
            // KeTimeStampBundle: interrupt time, system time, tick count.
            173 => {
                tsb = alloc(0x18);
                tsb
            }
            // Object types, monitor data, the global device pointer, the
            // calibration lock: zeroed blocks.
            _ => alloc(0x100),
        };
        rt.mem.write_u32(slot, a);
    }
    let thunks: Vec<u32> = calls.keys().copied().collect();
    crate::set_thunk_range(
        thunks.iter().min().copied().unwrap_or(0),
        thunks.iter().max().copied().unwrap_or(0),
    );

    let ok = K.set(Kernel {
        game_dir: s.game_dir.clone(),
        data_dir: s.data_dir.clone(),
        pool: Mutex::new(pool),
        virt: Mutex::new(Heap::new("virtual", VIRT.0, VIRT.1, 0x1_0000)),
        phys: Mutex::new(Heap::new("physical", PHYS.0, PHYS.1, 0x1000)),
        handles: Mutex::new(HashMap::new()),
        next_handle: AtomicU32::new(0x1000),
        xex_header,
        module,
        tls_slots: tls.0.max(64),
        tls_data: tls.1,
        default_stack,
        threads: Mutex::new(HashMap::new()),
        tls_used: Mutex::new(Vec::new()),
        links: Mutex::new(files::default_links(
            &s.game_dir,
            &s.data_dir,
            s.update_dir.as_deref(),
        )),
        calls: Mutex::new(calls),
        swaps: AtomicU64::new(0),
        video: Mutex::new(video::Video::default()),
        time_stamp_bundle: tsb,
    });
    assert!(ok.is_ok(), "kernel installed twice");
}

pub fn import_name(addr: u32) -> Option<String> {
    k().calls.lock().unwrap().get(&addr).map(|x| x.0.clone())
}

/// Name of the import this thread called last.
pub fn current_import() -> String {
    let at = crate::last_call();
    k().calls
        .lock()
        .unwrap()
        .get(&at)
        .map_or_else(|| format!("{at:#x}"), |x| x.0.clone())
}

/// Counts an import call (called from the runtime's trace hook).
/// Returns the import's name when the call is traced (`XENON_TRACE_IMPORTS`).
pub fn count_call(addr: u32) -> Option<String> {
    // Unset: off. "1": every import but critical sections. Otherwise only
    // imports whose name contains one of the comma-separated words.
    static TRACE: OnceLock<Option<Vec<String>>> = OnceLock::new();
    let filter = TRACE.get_or_init(|| {
        std::env::var("XENON_TRACE_IMPORTS").ok().map(|v| {
            if v == "1" {
                vec![]
            } else {
                v.split(',').map(str::to_string).collect()
            }
        })
    });
    let mut calls = k().calls.lock().unwrap();
    let e = calls.get_mut(&addr)?;
    e.1 += 1;
    let f = filter.as_ref()?;
    let show = if f.is_empty() {
        !e.0.contains("CriticalSection")
    } else {
        f.iter().any(|w| e.0.contains(w.as_str()))
    };
    show.then(|| e.0.clone())
}

/// Import call counts, most called first.
pub fn call_report() -> Vec<(String, u64)> {
    let mut v: Vec<_> = k()
        .calls
        .lock()
        .unwrap()
        .values()
        .filter(|x| x.1 > 0)
        .cloned()
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}
