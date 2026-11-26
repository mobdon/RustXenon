//! `xenon-host`: runs recompiled code on `xenon-rt`.
//!
//! The first milestone is headless: load the player's own executable and
//! unpacked game folder, run the title with the system calls replaced
//! ([`kernel`]), and report how far it got. The default run is headless:
//! no window or sound unless `--window` or `--sound` is given.
//!
//! A generated game crate calls [`run`] with the SHA-256 it was built from
//! and the register functions of its code crates; on Android its
//! `android_main` calls [`android_main`] instead.

pub mod cpu_profile;
pub mod heap;
pub mod kernel;
pub mod settings;

use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use xenon_rt::{Context, GuestMemory, Runtime};
#[cfg(target_os = "android")]
pub use xenon_window::android::AndroidApp;

/// Guest address space: the whole 32-bit range (pages are only touched
/// when used).
pub const GUEST_MEMORY: usize = 1 << 32;

static RT: OnceLock<Arc<Runtime>> = OnceLock::new();
static THUNKS: (AtomicU32, AtomicU32) = (AtomicU32::new(0), AtomicU32::new(0));
static START: OnceLock<Instant> = OnceLock::new();

thread_local! {
    /// Guest stack pointer at the newest call.
    static SP: Cell<u32> = const { Cell::new(0) };
    /// This thread's entry in [`ACTIVITY`].
    static MINE: Arc<Activity> = {
        let a = Arc::new(Activity::default());
        let t = std::thread::current();
        let name = t.name().unwrap_or("?").to_string();
        ACTIVITY.lock().unwrap().push((name, t.id(), a.clone()));
        a
    };
}

/// What a thread last did, for finding where a stalled title waits.
/// Written on every guest call by its own thread; aligned to two cache lines
/// (adjacent-line prefetch) so threads' entries never share one.
#[derive(Default)]
#[repr(align(128))]
struct Activity {
    /// Last import thunk called, and when (ms since start).
    import: AtomicU32,
    at_ms: AtomicU32,
    /// Last guest call target and the guest stack pointer then.
    call: AtomicU32,
    sp: AtomicU32,
    /// First argument of the last import call.
    arg0: AtomicU32,
    /// r3 at the last guest call: `this` of a virtual call, so a stop at a
    /// wild call target can show the object the pointer came from.
    r3: AtomicU32,
    /// The last call targets, written only by the owning thread at
    /// `head % 8`, so any thread can print them on a stop.
    ring: [AtomicU32; 8],
    head: AtomicU32,
    /// Guest object this thread is blocked on (0: none), and how many
    /// objects the wait covers.
    wait: AtomicU32,
    wait_n: AtomicU32,
}

impl Activity {
    /// The last call targets, oldest first.
    fn recent(&self) -> Vec<u32> {
        let h = self.head.load(Ordering::Relaxed);
        (0..8)
            .map(|i| self.ring[(h.wrapping_add(i) % 8) as usize].load(Ordering::Relaxed))
            .filter(|&a| a != 0)
            .collect()
    }
}

type ThreadEntry = (String, std::thread::ThreadId, Arc<Activity>);
static ACTIVITY: std::sync::Mutex<Vec<ThreadEntry>> = std::sync::Mutex::new(Vec::new());

/// Name of the host thread `id`, for saying who holds a lock.
pub fn thread_name(id: std::thread::ThreadId) -> String {
    ACTIVITY
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.1 == id)
        .map_or_else(|| format!("{id:?}"), |e| e.0.clone())
}

/// Marks this thread as blocked on the guest object `addr` (one of `n`)
/// until the returned guard drops, so a stall report can say what it waits on.
pub fn waiting(addr: u32, n: u32) -> impl Drop {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            MINE.with(|m| m.wait.store(0, Ordering::Relaxed));
        }
    }
    MINE.with(|m| {
        m.wait_n.store(n, Ordering::Relaxed);
        m.wait.store(addr, Ordering::Relaxed);
    });
    Clear
}

fn hex_list(v: &[u32]) -> String {
    v.iter()
        .map(|a| format!("{a:#010x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Source commit this host was built from (`-dirty` with local edits).
pub const BUILD: &str = env!("XENON_BUILD");

/// Milliseconds since the host started.
pub fn ms() -> u32 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u32
}

/// One line per guest thread: its last import and how long ago, and its
/// last call target.
fn activity_report() {
    let now = ms();
    // Copied out: describing a lock owner looks the list up again.
    let all: Vec<_> = ACTIVITY.lock().unwrap().clone();
    for (name, _, a) in all.iter() {
        let imp = a.import.load(Ordering::Relaxed);
        let what = kernel::import_name(imp).unwrap_or_else(|| "-".into());
        log(&format!(
            "  {name:<10} last import {what}({:#x}) {} ms ago, last call {:#010x}",
            a.arg0.load(Ordering::Relaxed),
            now.saturating_sub(a.at_ms.load(Ordering::Relaxed)),
            a.call.load(Ordering::Relaxed)
        ));
        log(&format!(
            "             recent calls {}",
            hex_list(&a.recent())
        ));
        let w = a.wait.load(Ordering::Relaxed);
        if w != 0 {
            let n = a.wait_n.load(Ordering::Relaxed);
            let more = if n > 1 {
                format!(" (first of {n})")
            } else {
                String::new()
            };
            let what = RT
                .get()
                .map_or_else(String::new, |rt| kernel::objects::describe(rt, w));
            log(&format!("             waiting on {w:#010x}{more}: {what}"));
        }
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
    use xenon_rt::guard::readable;
    let mut out = Vec::new();
    for _ in 0..24 {
        if sp == 0 || sp % 4 != 0 || sp as usize + 4 > GUEST_MEMORY || !readable(sp, 4) {
            break;
        }
        let prev = rt.mem.read_u32(sp);
        if prev <= sp || prev < 8 || !readable(prev - 8, 4) {
            break;
        }
        out.push(rt.mem.read_u32(prev - 8));
        sp = prev;
    }
    out
}

/// The eight words at `a`, as far as guest memory is readable there.
fn words_at(rt: &Runtime, a: u32) -> Vec<u32> {
    use xenon_rt::guard::readable;
    (0..8)
        .map(|i| a.wrapping_add(4 * i))
        .take_while(|&p| p % 4 == 0 && p as usize + 4 <= GUEST_MEMORY && readable(p, 4))
        .map(|p| rt.mem.read_u32(p))
        .collect()
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
    let on = |v| std::env::var_os(v).is_some();
    DEBUG_HOOKS.store(
        on("XENON_CALL_PROFILE") || on("XENON_WATCH"),
        Ordering::Relaxed,
    );
    if cpu_profile::on() {
        rt.after = Some(after);
    }
}

fn is_thunk(addr: u32) -> bool {
    (THUNKS.0.load(Ordering::Relaxed)..=THUNKS.1.load(Ordering::Relaxed)).contains(&addr)
}

/// The return hook, installed only for `XENON_CPU_PROFILE`.
fn after(_: &Context, addr: u32) {
    if is_thunk(addr) {
        cpu_profile::leave(addr);
    }
}

pub fn set_thunk_range(lo: u32, hi: u32) {
    THUNKS.0.store(lo, Ordering::Relaxed);
    THUNKS.1.store(hi, Ordering::Relaxed);
}

/// Target of the newest call on this thread.
pub fn last_call() -> u32 {
    MINE.with(|m| {
        m.ring[(m.head.load(Ordering::Relaxed).wrapping_sub(1) % 8) as usize]
            .load(Ordering::Relaxed)
    })
}

/// `XENON_CALL_PROFILE=<thread>:<from ms>:<to ms>`: counts the guest
/// call targets of one thread in a time window (printed in the report).
static PROFILE: std::sync::Mutex<Option<std::collections::HashMap<u32, u64>>> =
    std::sync::Mutex::new(None);

/// Set by [`install_trace`] when `XENON_CALL_PROFILE` or `XENON_WATCH` is
/// given, so the per-call hook skips both (and their config cells) otherwise.
static DEBUG_HOOKS: AtomicBool = AtomicBool::new(false);

#[cold]
fn profile_call(addr: u32) {
    static CFG: OnceLock<Option<(String, u32, u32)>> = OnceLock::new();
    let Some((who, from, to)) = CFG.get_or_init(|| {
        let v = std::env::var("XENON_CALL_PROFILE").ok()?;
        let mut p = v.split(':');
        Some((
            p.next()?.to_string(),
            p.next()?.parse().ok()?,
            p.next()?.parse().ok()?,
        ))
    }) else {
        return;
    };
    let t = ms();
    if t < *from || t >= *to || std::thread::current().name() != Some(who.as_str()) {
        return;
    }
    *PROFILE
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .entry(addr)
        .or_default() += 1;
}

/// `XENON_WATCH=<hex addr>:<what>`: at each call to `addr`, logs `what`
/// when it changes. `what` is a register number (`3`), a guest word
/// (`m82bf99d4`), a byte at a register plus an offset (`b3+14c7`), or the
/// text a register points at (`s3`); a leading `*` logs every call, not
/// only changes.
#[cold]
fn watch(c: &Context, addr: u32) {
    enum What {
        Reg(usize),
        Word(u32),
        Byte(usize, u32),
        Text(usize),
    }
    static CFG: OnceLock<Option<(u32, What, bool)>> = OnceLock::new();
    thread_local!(static LAST: std::cell::Cell<u32> = const { std::cell::Cell::new(0xDEAD_BEEF) });
    let Some((a, what, every)) = CFG.get_or_init(|| {
        let v = std::env::var("XENON_WATCH").ok()?;
        let (a, r) = v.split_once(':')?;
        let (every, r) = match r.strip_prefix('*') {
            Some(r) => (true, r),
            None => (false, r),
        };
        let what = if let Some(m) = r.strip_prefix('m') {
            What::Word(u32::from_str_radix(m, 16).ok()?)
        } else if let Some(t) = r.strip_prefix('s') {
            What::Text(t.parse().ok()?)
        } else if let Some(b) = r.strip_prefix('b') {
            let (reg, off) = b.split_once('+')?;
            What::Byte(reg.parse().ok()?, u32::from_str_radix(off, 16).ok()?)
        } else {
            What::Reg(r.parse().ok()?)
        };
        Some((
            u32::from_str_radix(a.trim_start_matches("0x"), 16).ok()?,
            what,
            every,
        ))
    }) else {
        return;
    };
    // Address 0 watches at every call, to find the call after a change.
    if addr != *a && *a != 0 {
        return;
    }
    let Some(rt) = RT.get() else { return };
    let (v, name) = match what {
        What::Reg(r) => (c.r[*r] as u32, format!("r{r}")),
        What::Word(m) => (rt.mem.read_u32(*m), format!("[{m:#x}]")),
        What::Byte(r, o) => {
            let at = (c.r[*r] as u32).wrapping_add(*o);
            (
                rt.mem.read_u8(at) as u32,
                format!("byte [r{r}+{o:#x}] ({at:#x})"),
            )
        }
        What::Text(r) => {
            let at = c.r[*r] as u32;
            let text: String = (0..120)
                .map(|i| rt.mem.read_u8(at + i))
                .take_while(|&b| b != 0)
                .map(|b| {
                    if (0x20..0x7F).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            // Fold the text in so an edit in place at the same address still logs.
            let v = text.bytes().fold(at, |h, b| h.rotate_left(5) ^ b as u32);
            (v, format!("text at r{r} ({at:#x}) {text:?}"))
        }
    };
    if LAST.with(|l| l.replace(v)) != v || *every {
        log(&format!(
            "[watch] {name} = {v:#x} at call to {addr:#010x} from {:#010x} in {} after {:x?}",
            c.lr as u32,
            std::thread::current().name().unwrap_or("?"),
            MINE.with(|m| m.recent())
        ));
    }
}

fn trace(c: &Context, addr: u32) {
    if DEBUG_HOOKS.load(Ordering::Relaxed) {
        profile_call(addr);
        watch(c, addr);
    }
    SP.with(|s| s.set(c.r[1] as u32));
    MINE.with(|m| {
        let h = m.head.load(Ordering::Relaxed);
        m.ring[(h % 8) as usize].store(addr, Ordering::Relaxed);
        m.head.store(h.wrapping_add(1), Ordering::Relaxed);
        m.call.store(addr, Ordering::Relaxed);
        m.sp.store(c.r[1] as u32, Ordering::Relaxed);
        m.r3.store(c.r[3] as u32, Ordering::Relaxed);
        if is_thunk(addr) {
            import_call(m, c, addr);
        }
    });
}

/// The import half of [`trace`], out of line so the per-call hook stays small.
#[inline(never)]
fn import_call(m: &Activity, c: &Context, addr: u32) {
    if cpu_profile::on() {
        cpu_profile::enter(addr);
    }
    m.import.store(addr, Ordering::Relaxed);
    m.arg0.store(c.r[3] as u32, Ordering::Relaxed);
    m.at_ms.store(ms(), Ordering::Relaxed);
    if let Some(name) = kernel::count_call(addr) {
        log_call(c, &name);
    }
}

#[cold]
#[inline(never)]
fn log_call(c: &Context, name: &str) {
    log(&format!(
        "[call] {} {name}({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}) from {:#010x}",
        std::thread::current().name().unwrap_or("?"),
        c.r[3],
        c.r[4],
        c.r[5],
        c.r[6],
        c.r[7],
        c.r[8],
        c.lr as u32
    ));
}

/// A line on stderr and in the log file with the time since start.
pub fn log(s: &str) {
    let t = START.get_or_init(Instant::now).elapsed();
    say(&format!("[{:8.3}] {s}", t.as_secs_f64()));
}

/// `rustxenon-play.log` beside the program, once opened.
static LOG_FILE: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
/// The newest log lines, for the stop screen.
static TAIL: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
const TAIL_LINES: usize = 40;

/// Text on stderr and in the log file. Locks are taken even if poisoned:
/// this runs inside the panic hook.
fn say(s: &str) {
    eprintln!("{s}");
    if let Some(f) = LOG_FILE.get() {
        let _ = writeln!(f.lock().unwrap_or_else(|e| e.into_inner()), "{s}");
    }
    let mut t = TAIL.lock().unwrap_or_else(|e| e.into_inner());
    for line in s.lines() {
        if t.len() == TAIL_LINES {
            t.pop_front();
        }
        t.push_back(line.into());
    }
}

/// Exit code of a run that stopped (0 while running), for the window's
/// close handler.
static STOPPED: AtomicI32 = AtomicI32::new(0);

/// Marks the run stopped with `code`. Only the first stop goes on: a
/// later one (another thread faulting behind the stop screen) parks for
/// good, so the first reason stays on screen.
fn claim_stop(code: i32) {
    if STOPPED
        .compare_exchange(0, code, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        if on_window_thread() {
            quit(STOPPED.load(Ordering::SeqCst));
        }
        loop {
            std::thread::park();
        }
    }
}

/// The log's newest lines.
fn tail() -> Vec<String> {
    let t = TAIL.lock().unwrap_or_else(|e| e.into_inner());
    t.iter().cloned().collect()
}

/// Whether this is the thread that pumps the window, which must not park.
fn on_window_thread() -> bool {
    xenon_window::is_pump_thread()
}

/// Ends the run with `code` (see [`show`]).
fn hold(code: i32) -> ! {
    claim_stop(code);
    show(code, tail())
}

/// Ends a claimed stop. With a window open (and this not its thread) the
/// sound stops and the window shows `lines` until the player closes it;
/// its close handler then exits.
fn show(code: i32, lines: Vec<String>) -> ! {
    if let Some(w) = kernel::input::WINDOW.get() {
        if !on_window_thread() {
            kernel::sound::silence();
            w.show_text(lines);
            loop {
                std::thread::park();
            }
        }
    }
    quit(code);
}

/// Ends the process with `code`. On Android with `_exit`: exit() would run
/// C++ static destructors while ART and the game's threads still run.
/// shortcut: the activity is not finished first (android-activity only
/// finishes it when `android_main` returns); do that if Android restarts it.
fn quit(code: i32) -> ! {
    #[cfg(target_os = "android")]
    {
        let _ = std::io::stdout().flush();
        // SAFETY: ends the process at once; the log file is unbuffered.
        unsafe { libc::_exit(code) }
    }
    #[cfg(not(target_os = "android"))]
    std::process::exit(code)
}

/// Logs why the run cannot go on and ends it (see [`hold`]).
fn fail(why: &str) -> ExitCode {
    log(why);
    hold(1)
}

const USAGE: &str = "\
USAGE: game --exe <patched executable> --game <unpacked game folder>
            [--update <unpacked title update>] [--data <writable folder>]
            [--seconds <n>] [--press <ms>:<button>[+<button>],...]
            [--no-controller] [--window] [--capture <dir> [--capture-at <s>,<s>...] [--capture-bundle]] [--render]
            [--capture-frames <n>,<n>... --capture-dir <dir>] [--audio-wav <file>]
            [--sound] [--mute] [--guard-memory] [--home <dir>]

Runs the title headless for --seconds (default 30) and reports how far it
got: frames presented, threads, imports called, memory in use. A
virtual controller on port 0 presses Start, then A, every four seconds
(from the fifth second) unless --press gives a script or --no-controller
unplugs it. Buttons: up down left right start back lthumb rthumb lb rb a b
x y. --capture writes, for one frame at each given second (default 20),
the textures bound at its draws and a list of the draws: local files,
derived from the game. --render draws every frame headless on the host GPU
and adds the frame image to each capture; --capture-bundle also
writes a frame bundle (commands and 512 MB of guest memory) for the
drawer's replay_frame example. --capture-frames writes the listed
frames (counted from 1, one per swap) as <dir>/frame-<n>.png; it implies
--render. --window opens a window that shows each drawn frame (it implies
--render), plugs in the controller and keyboard as the pad on port 0 (unless
--press or --no-controller is given) and runs until the window is closed
(or for --seconds when given). --audio-wav writes the game's sound (the
render clients' frames mixed to 48 kHz stereo) to a .wav file; compressed
sound is decoded through an LGPL FFmpeg found at run time (XENON_FFMPEG_DIR),
silence without one. --window also plays the sound on the default output
device (Windows only), as does --sound without a window; --mute keeps it
silent. --guard-memory (or
XENON_GUARD_MEMORY=1) reserves guest memory without access and commits only
what the kernel hands out, so a wild pointer stops the run with its address
and every thread's last calls instead of reading zeros. XENON_CPU_PROFILE=1
logs, every 5 s, the ms per frame each guest thread spends in its own code,
in kernel calls and waiting, the top kernel calls by time, and a frame rate
bound with drawing free; run it without --render to keep drawing off.
With --render, drawing runs on a thread of its own and the GPU command
processor runs up to XENON_DRAW_AHEAD=<frames> frames ahead of it (default
1); XENON_DRAW_AHEAD=0 draws inline.

Everything logged also goes to rustxenon-play.log beside this program;
with a window open, a stop shows the log's last lines in it until it is
closed (Esc, B or Back). --home is the player's folder (default: the folder
above games\\<sha256>\\ when this program is game.exe there, else its own
folder). Its settings.toml, then this game's run.toml (beside this program,
else <home>\\games\\<sha256>\\run.toml), give [window] width, height,
fullscreen and refresh, and [draw] aniso, force_linear and scale
(XENON_ANISO, XENON_FORCE_LINEAR and XENON_SCALE when set win); run.toml
also gives exe, game, update and data when the options do not.";

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Entry point of a generated game crate.
pub fn run(sha256: &str, regs: &[fn(&mut Runtime)]) -> ExitCode {
    run_with(
        sha256,
        regs,
        std::env::args().skip(1).collect(),
        std::env::current_exe().unwrap_or_default(),
    )
}

/// Entry point of a generated game crate on Android, called from its
/// `android_main`. The player's folder is the app's external files folder
/// (else its internal one); this game's log, run.toml, stdout.log and an
/// optional args.txt (more options, split at whitespace) live in
/// `<home>/games/<sha256>/`, which is also the current directory, so
/// relative paths in run.toml start there. The game runs on its own thread
/// and its end exits the process; this thread pumps the window.
#[cfg(target_os = "android")]
pub fn android_main(app: AndroidApp, sha256: &'static str, regs: &'static [fn(&mut Runtime)]) {
    let home = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
        .unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join("games").join(sha256);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::env::set_current_dir(&dir);
    redirect_output(&dir.join("stdout.log"));
    let mut args = vec![
        "--window".to_string(),
        "--home".to_string(),
        home.display().to_string(),
    ];
    if let Ok(more) = std::fs::read_to_string(dir.join("args.txt")) {
        args.extend(more.split_whitespace().map(String::from));
    }
    let me = dir.join("game");
    std::thread::Builder::new()
        .name("game".into())
        .stack_size(16 << 20)
        .spawn(move || {
            let code = run_with(sha256, regs, args, me);
            quit(if code == ExitCode::SUCCESS { 0 } else { 1 });
        })
        .expect("game thread");
    xenon_window::android::run(app)
}

/// Sends stdout and stderr (Android drops both) to `path`.
#[cfg(target_os = "android")]
fn redirect_output(path: &std::path::Path) {
    use std::os::fd::AsRawFd;
    let Ok(f) = std::fs::File::create(path) else {
        return;
    };
    // SAFETY: dup2 onto the standard descriptors; `f` stays open until
    // they hold their own copies.
    unsafe {
        libc::dup2(f.as_raw_fd(), 1);
        libc::dup2(f.as_raw_fd(), 2);
    }
}

/// [`run`] with the options (without the program name) and this program's
/// path given: the log and run.toml go beside `me`.
pub fn run_with(
    sha256: &str,
    regs: &[fn(&mut Runtime)],
    args: Vec<String>,
    me: PathBuf,
) -> ExitCode {
    START.get_or_init(Instant::now);
    let me_dir = me
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let opened = std::fs::File::create(me_dir.join("rustxenon-play.log"))
        .map(|f| LOG_FILE.set(Mutex::new(f)));
    log(&format!("build {BUILD}"));
    // Early, so a panic while starting up reaches the log and the stop screen.
    std::panic::set_hook(Box::new(|info| stop(&info.to_string())));
    if let Err(e) = opened {
        log(&format!("rustxenon-play.log: {e}; logging to stderr only"));
    }
    let home = settings::home(opt(&args, "--home").as_deref(), &me);
    let mut warn = Vec::new();
    let conf = settings::load(&home, &me_dir, sha256, &mut warn);
    for w in &warn {
        log(w);
    }
    // Before any thread starts: the drawer reads these when it opens.
    for (k, v) in conf
        .layer
        .draw_env(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()))
    {
        log(&format!("{k}={v} from the settings files"));
        std::env::set_var(k, v);
    }
    let arg_or = |name: &str, file: &Option<String>| opt(&args, name).or_else(|| file.clone());
    let (Some(exe), Some(game)) = (arg_or("--exe", &conf.exe), arg_or("--game", &conf.game)) else {
        say(USAGE);
        return ExitCode::FAILURE;
    };
    let data = arg_or("--data", &conf.data).map_or_else(
        || PathBuf::from(&game).with_extension("data"),
        PathBuf::from,
    );
    let window = args.iter().any(|a| a == "--window");
    let seconds: u64 = opt(&args, "--seconds")
        .and_then(|s| s.parse().ok())
        .unwrap_or(if window { u64::MAX / 4 } else { 30 });
    if let Some(list) = opt(&args, "--capture-frames") {
        let dir = opt(&args, "--capture-dir").unwrap_or_else(|| "captures".into());
        let n: Vec<u32> = list
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let _ = kernel::gpu::capture::FRAMES.set((PathBuf::from(dir), n));
        kernel::gpu::capture::RENDER.store(true, Ordering::Relaxed);
    }
    if args.iter().any(|a| a == "--render") {
        kernel::gpu::capture::RENDER.store(true, Ordering::Relaxed);
    }
    if let Some(d) = opt(&args, "--capture") {
        let at = opt(&args, "--capture-at").unwrap_or_else(|| "20".into());
        let at: Vec<u32> = at
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let _ = kernel::gpu::capture::SETTINGS.set((PathBuf::from(d), at));
        if args.iter().any(|a| a == "--capture-bundle") {
            kernel::gpu::capture::BUNDLE.store(true, Ordering::Relaxed);
        }
    }
    if window {
        kernel::gpu::capture::RENDER.store(true, Ordering::Relaxed);
        if opt(&args, "--press").is_none() && !args.iter().any(|a| a == "--no-controller") {
            kernel::input::set_script(kernel::input::Script::Live);
        }
        let link = xenon_window::spawn(format!("RustXenon {BUILD}"), conf.layer.window(), || {
            let code = STOPPED.load(Ordering::SeqCst);
            if code == 0 {
                report();
            }
            quit(code);
        });
        let _ = kernel::input::WINDOW.set(link);
    }
    if args.iter().any(|a| a == "--no-controller") {
        kernel::input::set_script(kernel::input::Script::Unplugged);
    } else if let Some(p) = opt(&args, "--press") {
        match kernel::input::parse(&p) {
            Ok(s) => kernel::input::set_script(s),
            Err(e) => return fail(&format!("--press: {e}")),
        }
    }

    let file = match std::fs::read(&exe) {
        Ok(f) => f,
        Err(e) => return fail(&format!("{exe}: {e}")),
    };
    let got: String = Sha256::digest(&file)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    log(&format!("executable sha256 {got}"));
    if !got.eq_ignore_ascii_case(sha256) {
        return fail(&format!(
            "this build is for executable {sha256}; refusing a different one"
        ));
    }
    let image = match xenon_recomp::xex::load(&file, &[]) {
        Ok(i) => i,
        Err(e) => return fail(&format!("{exe}: {e}")),
    };

    let guarded = args.iter().any(|a| a == "--guard-memory")
        || std::env::var_os("XENON_GUARD_MEMORY").is_some();
    // Write watch is on unless XENON_WRITE_WATCH=0; where the OS cannot
    // watch, memory is allocated as before.
    let watch_env = std::env::var_os("XENON_WRITE_WATCH");
    let watch = watch_env.as_ref().is_none_or(|v| v != "0");
    let mem = if guarded {
        log("guarded guest memory: a never-allocated address stops the run");
        if watch_env.is_some() && watch {
            log("XENON_WRITE_WATCH: not used with guarded guest memory");
        }
        GuestMemory::new_guarded(GUEST_MEMORY)
    } else if watch {
        let mem = GuestMemory::new_watched(GUEST_MEMORY);
        if mem.watched() {
            log("watched guest memory: the drawer re-reads only what the CPU wrote (XENON_WRITE_WATCH=0 turns it off)");
        } else {
            if watch_env.is_some() {
                log("XENON_WRITE_WATCH: the OS could not watch guest memory, ignored");
            }
            GUEST_IN_HEAP.store(GUEST_MEMORY, Ordering::Relaxed);
        }
        mem
    } else {
        GUEST_IN_HEAP.store(GUEST_MEMORY, Ordering::Relaxed);
        GuestMemory::new(GUEST_MEMORY)
    };
    xenon_rt::guard::commit(image.base, image.data.len() as u32);
    // Not bytes_mut: with guarded memory most of the range is inaccessible.
    mem.write_bytes(image.base, &image.data);
    let mut rt = Runtime::new(mem);
    for r in regs {
        r(&mut rt);
    }
    install_trace(&mut rt);
    kernel::install(
        &mut rt,
        &kernel::Setup {
            file: &file,
            image: &image,
            game_dir: game.into(),
            data_dir: data,
            update_dir: arg_or("--update", &conf.update).map(PathBuf::from),
        },
    );
    let rt = rt.share();
    set_runtime(rt.clone());
    log(&format!("call table: {} MB", rt.table_bytes() >> 20));

    xenon_rt::guard::on_wild_access(|a| {
        stop(&format!(
            "wild guest access at {a:#010x}, never allocated{}",
            xenon_rt::runtime::wild_hint(a)
        ))
    });

    if let Some(w) = opt(&args, "--audio-wav") {
        if let Err(e) = kernel::audio::start_wav(w.as_ref()) {
            return fail(&format!("{w}: {e}"));
        }
    }
    if (window || args.iter().any(|a| a == "--sound")) && !args.iter().any(|a| a == "--mute") {
        kernel::sound::start_speaker();
    }
    start_clock(rt.clone());
    start_memwatch(rt.clone());
    kernel::video::start(rt.clone());
    kernel::threads::register_main(&rt, image.entry);
    log(&format!("started at {:#010x}", image.entry));

    let t0 = Instant::now();
    let mut last = 0;
    let mut stalled = 0;
    while t0.elapsed() < Duration::from_secs(seconds) {
        std::thread::sleep(Duration::from_secs(1));
        let swaps = kernel::k().swaps.load(Ordering::Relaxed);
        cpu_profile::tick(swaps);
        log(&format!(
            "frames {swaps} (+{}), {}, {}",
            swaps - last,
            kernel::k().usage_line(),
            growth_line()
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

/// The system allocator, counting the bytes Rust code holds, so the run
/// log can tell Rust heap growth from memory a driver or C library takes.
struct Counting;

static RUST_HEAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// SAFETY: forwards to System with the same layouts; only adds counting.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc(l) };
        if !p.is_null() {
            RUST_HEAP.fetch_add(l.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc_zeroed(l) };
        if !p.is_null() {
            RUST_HEAP.fetch_add(l.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(p, l) };
        RUST_HEAP.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
        let q = unsafe { std::alloc::System.realloc(p, l, n) };
        if !q.is_null() {
            RUST_HEAP.fetch_add(n, Ordering::Relaxed);
            RUST_HEAP.fetch_sub(l.size(), Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes of guest memory allocated on the Rust heap (unguarded memory is
/// one heap block; guarded memory is reserved outside it).
static GUEST_IN_HEAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Bytes the Rust heap holds now.
pub fn rust_heap() -> usize {
    RUST_HEAP.load(Ordering::Relaxed)
}

/// Counts that grow if something is never released: open handles, live
/// guest threads and host resident memory.
fn growth_line() -> String {
    let kx = kernel::k();
    let started = kx.threads.lock().map(|t| t.len()).unwrap_or(0) as u32;
    let ended = kernel::threads::ENDED.load(Ordering::Relaxed);
    let rss = host_rss().map_or("?".into(), |b| (b >> 20).to_string());
    let r = |i: usize| kernel::files::READS[i].load(Ordering::Relaxed);
    format!(
        "handles {}, threads {} live, host {rss} MB (Rust heap {} MB besides guest memory), file reads {} ({} MB, {} ms), {}",
        kx.handle_count(),
        started.saturating_sub(ended),
        rust_heap().saturating_sub(GUEST_IN_HEAP.load(Ordering::Relaxed)) >> 20,
        r(0),
        r(1) >> 20,
        r(2) / 1000,
        kernel::sound::line()
    )
}

/// Resident memory of this process in bytes.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn host_rss() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    Some(s.split_whitespace().nth(1)?.parse::<u64>().ok()? * 4096)
}

#[cfg(windows)]
fn host_rss() -> Option<u64> {
    // PROCESS_MEMORY_COUNTERS; kernel32 exports K32GetProcessMemoryInfo.
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        rest: [usize; 6],
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(p: isize, c: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Default::default()
    };
    // SAFETY: c is a correctly sized PROCESS_MEMORY_COUNTERS.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0).then_some(c.working_set as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn host_rss() -> Option<u64> {
    None
}

/// Stops the run: the reason, this thread's last calls and guest return
/// addresses, then the full report (every thread). Never returns.
fn stop(why: &str) -> ! {
    claim_stop(101);
    let name = std::thread::current().name().unwrap_or("?").to_string();
    log(&format!("STOP in {name}: {why}"));
    log(&format!(
        "recent calls: {}",
        hex_list(&MINE.with(|m| m.recent()))
    ));
    if let Some(rt) = RT.get() {
        log(&format!(
            "guest return addresses: {}",
            hex_list(&backtrace(rt))
        ));
        // For a virtual call: the object, then its vtable.
        let r3 = MINE.with(|m| m.r3.load(Ordering::Relaxed));
        let object = words_at(rt, r3);
        log(&format!("last call's r3 {r3:#010x}: {}", hex_list(&object)));
        if let Some(&v) = object.first() {
            log(&format!(
                "words at {v:#010x}: {}",
                hex_list(&words_at(rt, v))
            ));
        }
    }
    // Taken before the report, whose lines would push these off the screen.
    let mut lines = tail();
    lines.push("The full report follows in rustxenon-play.log.".into());
    // The kernel is installed before RT is set; without it there is
    // nothing to report.
    if RT.get().is_some() {
        report();
    }
    show(101, lines)
}

fn report() {
    kernel::audio::finish_wav();
    let kx = kernel::k();
    log(&format!(
        "frames presented: {}",
        kx.swaps.load(Ordering::Relaxed)
    ));
    log(&format!(
        "threads started: {}",
        kx.threads.lock().map(|t| t.len()).unwrap_or(0)
    ));
    log(&format!(
        "memory: {}, {}, unknown frees {}",
        kx.usage_line(),
        growth_line(),
        kernel::memory::UNKNOWN_FREES.load(Ordering::Relaxed)
    ));
    if let Some(p) = PROFILE.lock().unwrap().as_ref() {
        let mut v: Vec<_> = p.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        log(&format!("call profile, {} targets:", v.len()));
        for (a, n) in v.iter().take(400) {
            log(&format!("  {n:>8} {a:#010x}"));
        }
    }
    log("threads:");
    activity_report();
    let calls = kernel::call_report();
    log(&format!("imports called: {} distinct", calls.len()));
    for (n, c) in calls.iter().take(40) {
        log(&format!("  {c:>10} {n}"));
    }
}

/// `XENON_MEMWATCH=<hex addr>:<hex len>`: polls a guest range every
/// millisecond and logs each word that changes, with every thread's last
/// call target (to find who fills a structure).
fn start_memwatch(rt: Arc<Runtime>) {
    let Some((a, n)) = std::env::var("XENON_MEMWATCH").ok().and_then(|v| {
        let (a, n) = v.split_once(':')?;
        Some((
            u32::from_str_radix(a, 16).ok()?,
            u32::from_str_radix(n, 16).ok()?,
        ))
    }) else {
        return;
    };
    std::thread::spawn(move || {
        let read = |i: u32| rt.mem.read_u32(a + 4 * i);
        let mut old: Vec<u32> = (0..n / 4).map(read).collect();
        loop {
            std::thread::sleep(Duration::from_millis(1));
            for (i, o) in old.iter_mut().enumerate() {
                let v = read(i as u32);
                if v != *o {
                    let who: Vec<String> = ACTIVITY
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|(t, _, x)| {
                            format!(
                                "{t}:{:#x}@{:#x}",
                                x.call.load(Ordering::Relaxed),
                                x.sp.load(Ordering::Relaxed)
                            )
                        })
                        .collect();
                    log(&format!(
                        "[memwatch] {:#x}: {:#x} -> {v:#x}; {}",
                        a + 4 * i as u32,
                        *o,
                        who.join(" ")
                    ));
                    *o = v;
                }
            }
        }
    });
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

#[cfg(test)]
mod tests {
    #[test]
    fn rust_heap_counts_allocations_and_frees() {
        // Other tests allocate and free at the same time, so one clean try
        // out of a few is enough.
        let clean = (0..5).any(|_| {
            let before = super::rust_heap();
            let mut v = std::hint::black_box(vec![1u8; 32 << 20]);
            let grown = super::rust_heap() >= before + (32 << 20);
            v.resize(64 << 20, 0);
            std::hint::black_box(&mut v);
            let resized = super::rust_heap() >= before + (64 << 20);
            drop(v);
            grown && resized && super::rust_heap() < before + (16 << 20)
        });
        assert!(clean);
    }

    #[test]
    fn ring_keeps_the_last_eight_calls_and_wait_marker_clears() {
        let mut c = xenon_rt::Context::new();
        for a in 1..=10u32 {
            c.r[3] = u64::from(0x4000_0000 + a);
            super::trace(&c, 0x8200_0000 + 4 * a);
        }
        assert_eq!(
            super::MINE.with(|m| m.r3.load(super::Ordering::Relaxed)),
            0x4000_000A
        );
        let want: Vec<u32> = (3..=10).map(|a| 0x8200_0000 + 4 * a).collect();
        assert_eq!(super::MINE.with(|m| m.recent()), want);
        assert_eq!(super::last_call(), 0x8200_0028);
        {
            let _w = super::waiting(0x4000_0010, 3);
            assert_eq!(
                super::MINE.with(|m| m.wait.load(super::Ordering::Relaxed)),
                0x4000_0010
            );
        }
        assert_eq!(
            super::MINE.with(|m| m.wait.load(super::Ordering::Relaxed)),
            0
        );
    }

    #[test]
    fn build_id_is_a_commit_or_unknown() {
        let c = super::BUILD.trim_end_matches("-dirty");
        assert!(
            c == "unknown" || (c.len() == 12 && c.bytes().all(|b| b.is_ascii_hexdigit())),
            "{}",
            super::BUILD
        );
    }
}
