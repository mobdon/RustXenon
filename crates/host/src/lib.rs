//! `xenon-host`: runs recompiled code on `xenon-rt`.
//!
//! The first milestone is headless: load the player's own executable and
//! unpacked game folder, run the title with the system calls replaced
//! ([`kernel`]), and report how far it got. No window, no sound.
//!
//! A generated game crate calls [`run`] with the SHA-256 it was built from
//! and the register functions of its code crates.

pub mod heap;
pub mod kernel;

use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use xenon_rt::{Context, GuestMemory, Runtime};

/// Guest address space: the whole 32-bit range (pages are only touched
/// when used).
pub const GUEST_MEMORY: usize = 1 << 32;

static RT: OnceLock<Arc<Runtime>> = OnceLock::new();
static THUNKS: (AtomicU32, AtomicU32) = (AtomicU32::new(0), AtomicU32::new(0));
static START: OnceLock<Instant> = OnceLock::new();

thread_local! {
    /// The last call targets on this host thread, newest last.
    static RECENT: Cell<[u32; 8]> = const { Cell::new([0; 8]) };
    /// Guest stack pointer at the newest call.
    static SP: Cell<u32> = const { Cell::new(0) };
    /// This thread's entry in [`ACTIVITY`].
    static MINE: Arc<Activity> = {
        let a = Arc::new(Activity::default());
        let name = std::thread::current().name().unwrap_or("?").to_string();
        ACTIVITY.lock().unwrap().push((name, a.clone()));
        a
    };
}

/// What a thread last did, for finding where a stalled title waits.
#[derive(Default)]
struct Activity {
    /// Last import thunk called, and when (ms since start).
    import: AtomicU32,
    at_ms: AtomicU32,
    /// Last guest call target and the guest stack pointer then.
    call: AtomicU32,
    sp: AtomicU32,
}

static ACTIVITY: std::sync::Mutex<Vec<(String, Arc<Activity>)>> = std::sync::Mutex::new(Vec::new());

fn ms() -> u32 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u32
}

/// One line per guest thread: its last import and how long ago, and its
/// last call target.
fn activity_report() {
    let now = ms();
    for (name, a) in ACTIVITY.lock().unwrap().iter() {
        let imp = a.import.load(Ordering::Relaxed);
        let what = kernel::import_name(imp).unwrap_or_else(|| "-".into());
        log(&format!(
            "  {name:<10} last import {what} {} ms ago, last call {:#010x}",
            now.saturating_sub(a.at_ms.load(Ordering::Relaxed)),
            a.call.load(Ordering::Relaxed)
        ));
        if let Some(rt) = RT.get() {
            let bt: Vec<String> = backtrace_from(rt, a.sp.load(Ordering::Relaxed))
                .iter()
                .take(12)
                .map(|x| format!("{x:#x}"))
                .collect();
            log(&format!("             returns to {}", bt.join(" ")));
        }
    }
}

/// Return addresses found by walking the guest stack's back chain from the
/// newest call (each frame links to its caller's; the caller's saved link
/// register sits 8 bytes below that).
fn backtrace(rt: &Runtime) -> Vec<u32> {
    backtrace_from(rt, SP.with(Cell::get))
}

fn backtrace_from(rt: &Runtime, mut sp: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..24 {
        if sp == 0 || sp % 4 != 0 || sp as usize + 4 > GUEST_MEMORY {
            break;
        }
        let prev = rt.mem.read_u32(sp);
        if prev <= sp || prev < 8 {
            break;
        }
        out.push(rt.mem.read_u32(prev - 8));
        sp = prev;
    }
    out
}

pub fn runtime() -> Arc<Runtime> {
    RT.get().expect("runtime not started").clone()
}

/// Makes `rt` the runtime kernel calls use (once per process; the kernel
/// must be installed in it). [`run`] does this; tests call it directly.
pub fn set_runtime(rt: Arc<Runtime>) {
    let _ = RT.set(rt);
}

/// The call hook the host installs: recent calls, per-thread activity and
/// import counts.
pub fn install_trace(rt: &mut Runtime) {
    rt.trace = Some(trace);
}

pub fn set_thunk_range(lo: u32, hi: u32) {
    THUNKS.0.store(lo, Ordering::Relaxed);
    THUNKS.1.store(hi, Ordering::Relaxed);
}

/// Target of the newest call on this thread.
pub fn last_call() -> u32 {
    RECENT.with(|r| r.get()[7])
}

fn trace(c: &Context, addr: u32) {
    SP.with(|s| s.set(c.r[1] as u32));
    RECENT.with(|r| {
        let mut v = r.get();
        v.copy_within(1.., 0);
        v[7] = addr;
        r.set(v);
    });
    MINE.with(|m| {
        m.call.store(addr, Ordering::Relaxed);
        m.sp.store(c.r[1] as u32, Ordering::Relaxed);
    });
    if (THUNKS.0.load(Ordering::Relaxed)..=THUNKS.1.load(Ordering::Relaxed)).contains(&addr) {
        MINE.with(|m| {
            m.import.store(addr, Ordering::Relaxed);
            m.at_ms.store(ms(), Ordering::Relaxed);
        });
        if let Some(name) = kernel::count_call(addr) {
            log(&format!(
                "[call] {name}({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}) from {:#010x}",
                c.r[3], c.r[4], c.r[5], c.r[6], c.r[7], c.r[8], c.lr as u32
            ));
        }
    }
}

/// A line on stderr with the time since start.
pub fn log(s: &str) {
    let t = START.get_or_init(Instant::now).elapsed();
    eprintln!("[{:8.3}] {s}", t.as_secs_f64());
}

const USAGE: &str = "\
USAGE: game --exe <patched executable> --game <unpacked game folder>
            [--update <unpacked title update>] [--data <writable folder>]
            [--seconds <n>]

Runs the title headless for --seconds (default 30) and reports how far it
got: frames presented, threads, imports called, memory in use.";

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Entry point of a generated game crate.
pub fn run(sha256: &str, regs: &[fn(&mut Runtime)]) -> ExitCode {
    START.get_or_init(Instant::now);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(exe), Some(game)) = (opt(&args, "--exe"), opt(&args, "--game")) else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    let data = opt(&args, "--data").map_or_else(
        || PathBuf::from(&game).with_extension("data"),
        PathBuf::from,
    );
    let seconds: u64 = opt(&args, "--seconds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let file = match std::fs::read(&exe) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{exe}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let got: String = Sha256::digest(&file)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    log(&format!("executable sha256 {got}"));
    if !got.eq_ignore_ascii_case(sha256) {
        eprintln!("this build is for executable {sha256}; refusing a different one");
        return ExitCode::FAILURE;
    }
    let image = match xenon_recomp::xex::load(&file, &[]) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{exe}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut mem = GuestMemory::new(GUEST_MEMORY);
    let base = image.base as usize;
    mem.bytes_mut()[base..base + image.data.len()].copy_from_slice(&image.data);
    let mut rt = Runtime::new(mem);
    for r in regs {
        r(&mut rt);
    }
    rt.trace = Some(trace);
    kernel::install(
        &mut rt,
        &kernel::Setup {
            file: &file,
            image: &image,
            game_dir: game.into(),
            data_dir: data,
            update_dir: opt(&args, "--update").map(PathBuf::from),
        },
    );
    let rt = rt.share();
    set_runtime(rt.clone());

    std::panic::set_hook(Box::new(|info| {
        let recent = RECENT.with(|r| r.get());
        let name = std::thread::current().name().unwrap_or("?").to_string();
        log(&format!("STOP in {name}: {info}"));
        log(&format!(
            "recent calls: {}",
            recent
                .iter()
                .filter(|&&a| a != 0)
                .map(|a| format!("{a:#010x}"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
        if let Some(rt) = RT.get() {
            let bt: Vec<String> = backtrace(rt).iter().map(|a| format!("{a:#010x}")).collect();
            log(&format!("guest return addresses: {}", bt.join(" ")));
        }
        report();
        std::process::exit(101);
    }));

    start_clock(rt.clone());
    kernel::video::start(rt.clone());
    kernel::threads::register_main(&rt, image.entry);
    log(&format!("started at {:#010x}", image.entry));

    let t0 = Instant::now();
    let mut last = 0;
    let mut stalled = 0;
    while t0.elapsed() < Duration::from_secs(seconds) {
        std::thread::sleep(Duration::from_secs(1));
        let swaps = kernel::k().swaps.load(Ordering::Relaxed);
        log(&format!(
            "frames {swaps} (+{}), {}",
            swaps - last,
            kernel::k().usage_line()
        ));
        if swaps == last {
            stalled += 1;
            // A stall with memory to spare is a wait, not exhaustion: say
            // where each thread is.
            if stalled == 3 {
                log(&format!(
                    "no new frames for 3 s; command processor state {}, last unmet wait on {:#x}, {} rounds, {} interrupts; threads:",
                    kernel::gpu::STATE.load(Ordering::Relaxed),
                    kernel::gpu::WAIT_ON.load(Ordering::Relaxed),
                    kernel::gpu::ROUNDS.load(Ordering::Relaxed),
                    kernel::gpu::RAISED.load(Ordering::Relaxed)
                ));
                activity_report();
            }
        } else {
            stalled = 0;
        }
        last = swaps;
    }
    report();
    ExitCode::SUCCESS
}

fn report() {
    let kx = kernel::k();
    log(&format!(
        "frames presented: {}",
        kx.swaps.load(Ordering::Relaxed)
    ));
    log(&format!(
        "threads started: {}",
        kx.threads.lock().map(|t| t.len()).unwrap_or(0)
    ));
    log(&format!("memory: {}", kx.usage_line()));
    let calls = kernel::call_report();
    log(&format!("imports called: {} distinct", calls.len()));
    for (n, c) in calls.iter().take(40) {
        log(&format!("  {c:>10} {n}"));
    }
}

/// Keeps the time stamp bundle (interrupt time, system time, tick count in
/// milliseconds) current, as the kernel's clock interrupt does.
fn start_clock(rt: Arc<Runtime>) {
    let p = kernel::k().time_stamp_bundle;
    if p == 0 {
        return;
    }
    std::thread::spawn(move || {
        let t0 = Instant::now();
        loop {
            let e = t0.elapsed();
            rt.mem.write_u64(p, (e.as_nanos() / 100) as u64);
            rt.mem.write_u64(p + 8, kernel::rtl::system_time());
            rt.mem.write_u32(p + 16, e.as_millis() as u32);
            std::thread::sleep(Duration::from_millis(1));
        }
    });
}
