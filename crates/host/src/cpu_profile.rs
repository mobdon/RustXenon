//! `XENON_CPU_PROFILE=1`: where the CPU time of each frame goes, logged
//! every five seconds as `[cpu]` lines. Off by default; when off it costs
//! one check per import call.
//!
//! Each guest thread's time is cut at its import calls: from an import's
//! return to the next import's entry is guest code, the import itself is
//! kernel time, and imports that block (waits, sleeps, lock entry) are
//! counted as waiting. Interrupt callbacks the `gpu` host thread runs are
//! timed as a whole. Guest code that spins on memory (a GPU fence, say)
//! shows as guest code, unless the spin yields through a waiting import:
//! guest code shorter than [`SPIN_NS`] between two calls of the same waiting
//! import (a loop of `KeDelayExecutionThread(0)` polling a frame counter,
//! say) is the wait loop itself and counts as waiting. That needs no
//! per-title data; a spin that never calls out cannot be told from work.
//!
//! Run without `--render` to read guest time with drawing off: the command
//! ring still runs and frames are still counted, nothing is drawn.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XENON_CPU_PROFILE").is_some_and(|v| v != "0"))
}

/// One thread's running totals, in nanoseconds since [`t0`].
#[derive(Default)]
struct Totals {
    guest: AtomicU64,
    /// Start of the current guest stretch or import call.
    seg: AtomicU64,
    /// Import being called now (0: in guest code).
    cur: AtomicU32,
    depth: AtomicU32,
    /// Per import thunk: (ns, calls).
    imports: Mutex<HashMap<u32, (u64, u64)>>,
    /// Import that last returned (at depth 0), for spotting a spin.
    last: AtomicU32,
    /// Per import thunk: guest ns spent between back-to-back calls of it
    /// (gaps under [`SPIN_NS`]), still counted in `guest`; the report moves
    /// it to waiting for waiting imports.
    spin: Mutex<HashMap<u32, u64>>,
}

/// Longest guest stretch between two calls of the same waiting import that
/// still counts as a spin on it rather than work.
pub const SPIN_NS: u64 = 50_000;

static THREADS: Mutex<Vec<(String, Arc<Totals>)>> = Mutex::new(Vec::new());
/// Interrupt callbacks on the `gpu` thread: ns, calls.
static INTERRUPTS: (AtomicU64, AtomicU64) = (AtomicU64::new(0), AtomicU64::new(0));

fn t0() -> Instant {
    static T0: OnceLock<Instant> = OnceLock::new();
    *T0.get_or_init(Instant::now)
}

fn now() -> u64 {
    t0().elapsed().as_nanos() as u64
}

thread_local! {
    /// This thread's totals; none for host threads that are not guest threads.
    static MINE: Option<Arc<Totals>> = {
        let name = std::thread::current().name().unwrap_or("?").to_string();
        name.starts_with("guest-").then(|| {
            let t = Arc::new(Totals::default());
            t.seg.store(now(), Ordering::Relaxed);
            THREADS.lock().unwrap().push((name, t.clone()));
            t
        })
    };
}

/// Import call entry (from the trace hook).
pub fn enter(addr: u32) {
    let _ = MINE.try_with(|m| {
        let Some(m) = m else { return };
        // Imports that call back into the guest: the outer call counts.
        if m.depth.fetch_add(1, Ordering::Relaxed) == 0 {
            let t = now();
            let s = m.seg.swap(t, Ordering::Relaxed);
            let gap = t.saturating_sub(s);
            m.guest.fetch_add(gap, Ordering::Relaxed);
            m.cur.store(addr, Ordering::Relaxed);
            if gap < SPIN_NS && m.last.load(Ordering::Relaxed) == addr {
                *m.spin.lock().unwrap().entry(addr).or_default() += gap;
            }
        }
    });
}

/// Import call return.
pub fn leave(addr: u32) {
    let _ = MINE.try_with(|m| {
        let Some(m) = m else { return };
        if m.depth.load(Ordering::Relaxed) == 0 {
            return; // profile turned on mid-call; nothing entered
        }
        if m.depth.fetch_sub(1, Ordering::Relaxed) == 1 {
            let t = now();
            let s = m.seg.swap(t, Ordering::Relaxed);
            let e = m.cur.swap(0, Ordering::Relaxed);
            let e = if e != 0 { e } else { addr };
            m.last.store(e, Ordering::Relaxed);
            let mut i = m.imports.lock().unwrap();
            let x = i.entry(e).or_default();
            x.0 += t.saturating_sub(s);
            x.1 += 1;
        }
    });
}

/// Times an interrupt callback run on a host thread.
pub fn interrupt(f: impl FnOnce()) {
    if !on() {
        return f();
    }
    let s = Instant::now();
    f();
    INTERRUPTS
        .0
        .fetch_add(s.elapsed().as_nanos() as u64, Ordering::Relaxed);
    INTERRUPTS.1.fetch_add(1, Ordering::Relaxed);
}

/// Imports that block until something else happens. An ended thread's
/// host thread stays parked in `ExTerminateThread` for the rest of the run.
pub fn is_wait(name: &str) -> bool {
    [
        "Wait",
        "Delay",
        "Sleep",
        "EnterCriticalSection",
        "SpinLock",
        "TerminateThread",
    ]
    .iter()
    .any(|w| name.contains(w))
}

#[derive(Clone, Default)]
struct Snap {
    guest: u64,
    imports: HashMap<u32, (u64, u64)>,
    spin: HashMap<u32, u64>,
}

/// Guest ns between `o` and `s` that was spin on a waiting import.
fn spin_on_waits(s: &Snap, o: &Snap, is_wait: impl Fn(u32) -> bool) -> u64 {
    s.spin
        .iter()
        .filter(|(a, _)| is_wait(**a))
        .map(|(a, ns)| ns.saturating_sub(o.spin.get(a).copied().unwrap_or_default()))
        .sum()
}

fn snap(t: &Totals, at: u64) -> Snap {
    let mut s = Snap {
        guest: t.guest.load(Ordering::Relaxed),
        imports: t.imports.lock().unwrap().clone(),
        spin: t.spin.lock().unwrap().clone(),
    };
    // The stretch in progress, so a thread that never calls out still shows.
    let open = at.saturating_sub(t.seg.load(Ordering::Relaxed));
    match t.cur.load(Ordering::Relaxed) {
        0 => s.guest += open,
        a => s.imports.entry(a).or_default().0 += open,
    }
    s
}

/// One window's numbers, per frame, before formatting.
struct Row {
    name: String,
    guest: f64,
    kernel: f64,
    wait: f64,
    /// Of `wait`, the guest spin between calls of a waiting import.
    spin: f64,
}

/// Fixed integer work: `n` dependent multiply-adds.
fn fixed_work(n: u64) -> u64 {
    let mut x = 1u64;
    for i in 0..n {
        x = std::hint::black_box(x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i));
    }
    x
}

/// Fastest of three runs of `fixed_work(n)`, in ns, so one preemption does
/// not read as a slow machine.
fn time_work(n: u64) -> u64 {
    (0..3)
        .map(|_| {
            let s = Instant::now();
            std::hint::black_box(fixed_work(n));
            s.elapsed().as_nanos() as u64
        })
        .min()
        .unwrap_or(0)
}

/// The fixed work's size and its time at the first tick, from [`calibrate`].
static CAL: OnceLock<(u64, u64)> = OnceLock::new();

/// Iteration count that takes about 1 ms now, and its time in ns.
fn calibrate() -> (u64, u64) {
    let mut n = 1024;
    loop {
        let t = time_work(n);
        if t >= 1_000_000 || n >= 1 << 40 {
            return (n, t.max(1));
        }
        n *= 2;
    }
}

/// Lines for one window of `frames` frames over `secs` seconds, from the
/// per-thread deltas and the per-import deltas (name, ms, calls).
fn lines(
    secs: f64,
    frames: u64,
    mut rows: Vec<Row>,
    imports: Vec<(String, f64, u64)>,
    irq: (f64, u64),
) -> Vec<String> {
    let n = frames.max(1) as f64;
    let per = if frames == 0 { "window" } else { "frame" };
    let mut out = vec![format!(
        "[cpu] {secs:.1} s, {frames} frames ({:.1}/s); ms per {per}: guest code / kernel calls (of which waiting) by thread",
        frames as f64 / secs
    )];
    rows.sort_by(|a, b| (b.guest + b.kernel - b.wait).total_cmp(&(a.guest + a.kernel - a.wait)));
    let busy = |r: &Row| r.guest + r.kernel - r.wait;
    let idle = rows.iter().filter(|r| busy(r) / n < 0.05).count();
    let (mut g, mut kk, mut w) = (0.0, 0.0, 0.0);
    for r in &rows {
        g += r.guest;
        kk += r.kernel;
        w += r.wait;
        if busy(r) / n >= 0.05 {
            out.push(format!(
                "[cpu]   {:<10} {:7.2} / {:7.2} ({:7.2})",
                r.name,
                r.guest / n,
                r.kernel / n,
                r.wait / n
            ));
        }
    }
    out.push(format!(
        "[cpu]   all        {:7.2} / {:7.2} ({:7.2}); {idle} threads under 0.05 ms busy; gpu interrupts {:.2} ms ({} calls)",
        g / n,
        kk / n,
        w / n,
        irq.0 / n,
        irq.1
    ));
    let top: Vec<String> = imports
        .iter()
        .take(12)
        .map(|(name, ms, c)| format!("{name} {:.2} ({:.0})", ms / n, *c as f64 / n))
        .collect();
    out.push(format!(
        "[cpu]   top kernel calls, ms (calls) per {per}: {}",
        top.join(", ")
    ));
    // A thread busy for the whole frame at any frame rate is polling (a
    // spin on a fence or the vertical blank), not doing that much work. So is
    // one that spends most of its time spinning on a waiting import: at a low
    // frame rate the loop around the spin can still look like the busiest.
    let frame_ms = secs * 1e3 / n;
    let (spin, work): (Vec<&Row>, Vec<&Row>) = rows.iter().partition(|r| {
        busy(r) / n >= 0.95 * frame_ms
            || r.spin >= 0.5 * (r.guest + r.kernel).max(f64::MIN_POSITIVE)
    });
    if let Some(r) = work.first().filter(|_| frames > 0) {
        let b = busy(r) / n;
        let spin: Vec<&str> = spin.iter().map(|r| r.name.as_str()).collect();
        out.push(format!(
            "[cpu]   busiest thread {} {b:.2} ms per frame busy (guest + kernel - waiting); with drawing free, about {:.0} fps at most{}",
            r.name,
            1000.0 / b.max(0.001),
            if spin.is_empty() {
                String::new()
            } else {
                format!(" (left out, busy the whole frame, likely polling: {})", spin.join(" "))
            }
        ));
    }
    out
}

/// Call once a second from the run loop with the frames presented so
/// far; logs the `[cpu]` lines every five seconds.
pub fn tick(swaps: u64) {
    if !on() {
        return;
    }
    type Prev = (u64, u64, HashMap<String, Snap>, (u64, u64));
    static PREV: Mutex<Option<Prev>> = Mutex::new(None);
    let at = now();
    let cur: HashMap<String, Snap> = THREADS
        .lock()
        .unwrap()
        .iter()
        .map(|(n, t)| (n.clone(), snap(t, at)))
        .collect();
    let irq = (
        INTERRUPTS.0.load(Ordering::Relaxed),
        INTERRUPTS.1.load(Ordering::Relaxed),
    );
    let mut prev = PREV.lock().unwrap();
    let Some((t_prev, s_prev, old, irq_prev)) = prev.as_ref() else {
        CAL.get_or_init(calibrate);
        *prev = Some((at, swaps, cur, irq));
        return;
    };
    if at - t_prev < 5_000_000_000 {
        return;
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    let mut rows = Vec::new();
    let mut by: HashMap<u32, (u64, u64)> = HashMap::new();
    let name = |a: u32| crate::kernel::import_name(a).unwrap_or_else(|| format!("{a:#x}"));
    for (n, s) in &cur {
        let o = old.get(n).cloned().unwrap_or_default();
        let (mut kernel, mut wait) = (0, 0);
        for (a, x) in &s.imports {
            let y = o.imports.get(a).copied().unwrap_or_default();
            let d = (x.0.saturating_sub(y.0), x.1.saturating_sub(y.1));
            kernel += d.0;
            if is_wait(&name(*a)) {
                wait += d.0;
            }
            let e = by.entry(*a).or_default();
            e.0 += d.0;
            e.1 += d.1;
        }
        // Spin between calls of a waiting import moves from guest code to
        // waiting (and so to kernel calls, which waiting is part of).
        let guest = s.guest.saturating_sub(o.guest);
        let spin = spin_on_waits(s, &o, |a| is_wait(&name(a))).min(guest);
        rows.push(Row {
            name: n.clone(),
            guest: ms(guest - spin),
            kernel: ms(kernel + spin),
            wait: ms(wait + spin),
            spin: ms(spin),
        });
    }
    let mut imports: Vec<_> = by
        .into_iter()
        .filter(|x| x.1 .1 > 0 || x.1 .0 > 0)
        .map(|(a, (ns, c))| (name(a), ms(ns), c))
        .collect();
    imports.sort_by(|a, b| b.1.total_cmp(&a.1));
    for l in lines(
        ms(at - t_prev) / 1e3,
        swaps - s_prev,
        rows,
        imports,
        (ms(irq.0 - irq_prev.0), irq.1 - irq_prev.1),
    ) {
        crate::log(&l);
    }
    // Same work every window: a higher ratio is a slower machine (heat, boost,
    // other load), not more work.
    let (n, base) = *CAL.get_or_init(calibrate);
    let t = time_work(n);
    crate::log(&format!(
        "[cpu]   fixed work {:.2} ms, {:.2}x the first window (above 1: the machine, not the guest, got slower)",
        ms(t),
        t as f64 / base as f64
    ));
    *prev = Some((at, swaps, cur, irq));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_are_told_from_work() {
        assert!(is_wait("KeWaitForSingleObject"));
        assert!(is_wait("RtlEnterCriticalSection"));
        assert!(is_wait("KeDelayExecutionThread"));
        assert!(!is_wait("RtlLeaveCriticalSection"));
        assert!(!is_wait("NtReadFile"));
        assert!(is_wait("ExTerminateThread"));
    }

    #[test]
    fn lines_are_per_frame_and_estimate_from_the_busiest_thread() {
        let row = |name: &str, guest, kernel, wait| Row {
            name: name.into(),
            guest,
            kernel,
            wait,
            spin: 0.0,
        };
        // 300 frames in 5 s; guest-1 is busy 3000 ms of guest code plus
        // 600 ms of kernel calls, 300 ms of which waiting: 11 ms a frame.
        let l = lines(
            5.0,
            300,
            vec![
                row("guest-2", 30.0, 4000.0, 4000.0),
                row("guest-1", 3000.0, 600.0, 300.0),
                row("guest-9", 1.0, 4999.0, 4999.0),
                row("guest-5", 4990.0, 1.0, 0.0),
            ],
            vec![("KeWaitForSingleObject".into(), 4299.0, 600)],
            (60.0, 300),
        );
        assert!(l[0].contains("300 frames (60.0/s)"), "{}", l[0]);
        assert!(l[1].contains("guest-5"));
        assert!(l[2].contains("guest-1") && l[2].contains("10.00 /    2.00 (   1.00)"));
        assert!(l[3].contains("guest-2"));
        assert!(l.iter().any(|x| x.contains("1 threads under 0.05 ms busy")));
        assert!(l
            .iter()
            .any(|x| x.contains("KeWaitForSingleObject 14.33 (2)")));
        let last = l.last().unwrap();
        assert!(
            last.contains("guest-1 11.00 ms")
                && last.contains("about 91 fps")
                && last.contains("likely polling: guest-5"),
            "{last}"
        );

        // 50 frames in 5 s (100 ms a frame). guest-3 spins 3000 of its 5000 ms
        // on a waiting import; the loop around the spin leaves it 40 ms a
        // frame busy, under the whole-frame bar, but it is polling, not work.
        let spinner = Row {
            spin: 3000.0,
            ..row("guest-3", 2000.0, 3000.0, 3000.0)
        };
        let l = lines(
            5.0,
            50,
            vec![spinner, row("guest-1", 1000.0, 600.0, 100.0)],
            vec![],
            (0.0, 0),
        );
        let last = l.last().unwrap();
        assert!(
            last.contains("busiest thread guest-1 30.00 ms")
                && last.contains("about 33 fps")
                && last.contains("likely polling: guest-3"),
            "{last}"
        );
    }

    #[test]
    fn fixed_work_is_calibrated_to_about_a_millisecond() {
        let (n, base) = calibrate();
        assert!(n >= 1024 && base >= 1_000_000, "{n} {base}");
        assert!(time_work(n) > 0);
        assert_ne!(fixed_work(n), fixed_work(n / 2));
    }

    #[test]
    fn a_thread_is_cut_at_its_imports() {
        let t = std::thread::Builder::new()
            .name("guest-test".into())
            .spawn(|| {
                MINE.with(|m| assert!(m.is_some()));
                std::thread::sleep(std::time::Duration::from_millis(20));
                enter(0x10);
                std::thread::sleep(std::time::Duration::from_millis(60));
                enter(0x20); // nested: belongs to the outer call
                leave(0x20);
                leave(0x10);
                MINE.with(|m| snap(m.as_ref().unwrap(), now()))
            })
            .unwrap();
        let s = t.join().unwrap();
        let (ns, calls) = s.imports[&0x10];
        assert!(s.guest >= 20_000_000 && s.guest < ns, "{} {ns}", s.guest);
        assert!(ns >= 60_000_000 && calls == 1, "{ns} {calls}");
        assert!(!s.imports.contains_key(&0x20));
        assert!(s.spin.is_empty(), "{:?}", s.spin);
    }

    #[test]
    fn back_to_back_calls_of_one_import_are_spin() {
        let t = std::thread::Builder::new()
            .name("guest-spin".into())
            .spawn(|| {
                // A yield loop on 0x30, then one call of 0x40 after it.
                for _ in 0..1000 {
                    enter(0x30);
                    leave(0x30);
                }
                enter(0x40);
                leave(0x40);
                // Work longer than a spin gap before the same import again.
                std::thread::sleep(std::time::Duration::from_nanos(2 * SPIN_NS));
                enter(0x40);
                leave(0x40);
                MINE.with(|m| snap(m.as_ref().unwrap(), now()))
            })
            .unwrap();
        let s = t.join().unwrap();
        assert!(s.spin[&0x30] > 0 && s.spin[&0x30] <= s.guest);
        assert!(!s.spin.contains_key(&0x40), "{:?}", s.spin);
        let none = Snap::default();
        assert_eq!(spin_on_waits(&s, &none, |a| a == 0x30), s.spin[&0x30]);
        assert_eq!(spin_on_waits(&s, &none, |_| false), 0);
        assert_eq!(spin_on_waits(&s, &s, |_| true), 0);
    }
}
