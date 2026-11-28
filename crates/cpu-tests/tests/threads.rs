//! Synthetic multi-thread guest programs: several guest threads at once on
//! host threads, sharing one runtime and one guest memory.
//!
//! Thread interleaving changes from run to run, so every check is on an
//! outcome that must hold for any interleaving (final counts, sums, which
//! waits succeeded), never on an order.

include!(concat!(env!("OUT_DIR"), "/threads.rs"));

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use xenon_rt::{CriticalSection, Event, GuestFn, GuestMemory, Semaphore};

/// More guest threads than the console's six hardware threads.
const THREADS: u64 = 8;

// Translated guest programs (see build.rs).
const ADD32: u32 = 0x8240_0000;
const ADD64: u32 = 0x8240_1000;
const SPIN: u32 = 0x8240_2000;

// Hooks standing in for the kernel's lock, event and semaphore calls; the
// guest object's address is in r3, as on the console.
const CS_ENTER: u32 = 0x8250_0000;
const CS_LEAVE: u32 = 0x8250_0010;
const EV_WAIT: u32 = 0x8250_0020;
const EV_WAIT_MS: u32 = 0x8250_0030;
const SEM_WAIT: u32 = 0x8250_0040;
const SEM_RELEASE: u32 = 0x8250_0050;

fn cs_enter(rt: &Runtime, c: &mut Context) {
    rt.object(c.r[3] as u32, CriticalSection::new).enter(None);
}
fn cs_leave(rt: &Runtime, c: &mut Context) {
    rt.object(c.r[3] as u32, CriticalSection::new).leave();
}
fn ev_wait(rt: &Runtime, c: &mut Context) {
    rt.object(c.r[3] as u32, || Event::new(false, false))
        .wait(None);
}
/// Waits up to r4 milliseconds; r3 = 1 if the event was taken.
fn ev_wait_ms(rt: &Runtime, c: &mut Context) {
    let e = rt.object(c.r[3] as u32, || Event::new(false, false));
    c.r[3] = e.wait(Some(Duration::from_millis(c.r[4]))) as u64;
}
fn sem(rt: &Runtime, addr: u64) -> Arc<Semaphore> {
    rt.object(addr as u32, || panic!("semaphore {addr:#x} not created"))
}
fn sem_wait(rt: &Runtime, c: &mut Context) {
    sem(rt, c.r[3]).wait(None);
}
fn sem_release(rt: &Runtime, c: &mut Context) {
    sem(rt, c.r[3])
        .release(1)
        .expect("semaphore over its maximum");
}

/// Calls `addr` with `r3` as the only argument, as guest code would.
fn call1(rt: &Runtime, c: &mut Context, addr: u32, r3: u64) {
    c.r[3] = r3;
    rt.call(c, addr);
}

fn runtime(setup: impl FnOnce(&mut Runtime)) -> Arc<Runtime> {
    let mut rt = Runtime::new(GuestMemory::new(0x10000));
    rt.register(ADD32, atomic_add32);
    rt.register(ADD64, atomic_add64);
    rt.register(SPIN, spin_lock_count);
    for (addr, f) in [
        (CS_ENTER, cs_enter as GuestFn),
        (CS_LEAVE, cs_leave),
        (EV_WAIT, ev_wait),
        (EV_WAIT_MS, ev_wait_ms),
        (SEM_WAIT, sem_wait),
        (SEM_RELEASE, sem_release),
    ] {
        rt.set_override(addr, f);
    }
    setup(&mut rt);
    rt.share()
}

fn args(r3: u64, r4: u64) -> Context {
    let mut c = Context::new();
    c.r[3] = r3;
    c.r[4] = r4;
    c
}

/// Starts `n` guest threads at `entry` with the same arguments, then joins
/// them all and returns their exit values.
fn run(rt: &Runtime, n: u64, entry: u32, r3: u64, r4: u64) -> Vec<u64> {
    let ids: Vec<u32> = (0..n)
        .map(|_| rt.spawn(entry, args(r3, r4), false))
        .collect();
    ids.into_iter().map(|id| rt.join(id)).collect()
}

#[test]
fn contended_32_bit_counter_loses_no_updates() {
    let rt = runtime(|_| {});
    run(&rt, THREADS, ADD32, 0x100, 20_000);
    assert_eq!(rt.mem.read_u32(0x100), (THREADS * 20_000) as u32);
}

#[test]
fn contended_64_bit_counter_carries_past_32_bits() {
    let rt = runtime(|_| {});
    rt.mem.write_u64(0x200, 0xFFFF_F000);
    run(&rt, THREADS, ADD64, 0x200, 20_000);
    assert_eq!(rt.mem.read_u64(0x200), 0xFFFF_F000 + THREADS * 20_000);
}

#[test]
fn guest_spin_lock_protects_plain_stores() {
    let rt = runtime(|_| {});
    run(&rt, THREADS, SPIN, 0x300, 5_000);
    assert_eq!(rt.mem.read_u32(0x300), 0, "lock left held");
    assert_eq!(rt.mem.read_u32(0x304), (THREADS * 5_000) as u32);
}

const WORKER: u32 = 0x8260_0000;

#[test]
fn critical_section_through_overrides_is_exclusive_and_recursive() {
    /// r3 = lock, r4 = count, r5 = counter: a read-modify-write with a
    /// yield in the middle, which loses updates unless the lock holds.
    fn worker(rt: &Runtime, c: &mut Context) {
        let (cs, n, counter) = (c.r[3], c.r[4], c.r[5] as u32);
        for _ in 0..n {
            call1(rt, c, CS_ENTER, cs);
            call1(rt, c, CS_ENTER, cs);
            let v = rt.mem.read_u32(counter);
            std::thread::yield_now();
            rt.mem.write_u32(counter, v + 1);
            call1(rt, c, CS_LEAVE, cs);
            call1(rt, c, CS_LEAVE, cs);
        }
    }
    let rt = runtime(|rt| rt.register(WORKER, worker));
    let ids: Vec<u32> = (0..THREADS)
        .map(|_| {
            let mut c = args(0x400, 2_000);
            c.r[5] = 0x410;
            rt.spawn(WORKER, c, false)
        })
        .collect();
    for id in ids {
        rt.join(id);
    }
    assert_eq!(rt.mem.read_u32(0x410), (THREADS * 2_000) as u32);
}

#[test]
fn event_wakes_a_waiting_thread() {
    /// Waits on the event at r3, then returns the word at 0x510.
    fn waiter(rt: &Runtime, c: &mut Context) {
        rt.call(c, EV_WAIT);
        c.r[3] = rt.mem.read_u32(0x510) as u64;
    }
    let rt = runtime(|rt| rt.register(WORKER, waiter));
    let id = rt.spawn(WORKER, args(0x500, 0), false);
    std::thread::sleep(Duration::from_millis(20));
    rt.mem.write_u32(0x510, 42);
    rt.object(0x500, || Event::new(false, false)).set();
    assert_eq!(rt.join(id), 42);
}

#[test]
fn auto_reset_event_releases_exactly_one_of_two_waiters() {
    let rt = runtime(|_| {});
    // Set once before either waits: one wait takes it, the other times out.
    rt.object(0x600, || Event::new(false, false)).set();
    let woken: u64 = run(&rt, 2, EV_WAIT_MS, 0x600, 300).iter().sum();
    assert_eq!(woken, 1);
}

#[test]
fn producer_consumer_queue_delivers_every_item_once() {
    const SLOTS: u64 = 8;
    const ITEMS: u64 = 10_000;
    const CONSUMERS: u64 = 3;
    const FREE: u64 = 0x700;
    const FULL: u64 = 0x710;
    const LOCK: u64 = 0x720;
    const HEAD: u32 = 0x730;
    const TAIL: u32 = 0x734;
    const RING: u32 = 0x740;

    fn producer(rt: &Runtime, c: &mut Context) {
        let values = (1..=ITEMS).chain((0..CONSUMERS).map(|_| 0));
        for v in values {
            call1(rt, c, SEM_WAIT, FREE);
            call1(rt, c, CS_ENTER, LOCK);
            let tail = rt.mem.read_u32(TAIL);
            rt.mem.write_u32(RING + 4 * (tail % SLOTS as u32), v as u32);
            rt.mem.write_u32(TAIL, tail + 1);
            call1(rt, c, CS_LEAVE, LOCK);
            call1(rt, c, SEM_RELEASE, FULL);
        }
    }
    /// Returns the sum of the values it took; stops at a 0.
    fn consumer(rt: &Runtime, c: &mut Context) {
        let mut sum = 0u64;
        loop {
            call1(rt, c, SEM_WAIT, FULL);
            call1(rt, c, CS_ENTER, LOCK);
            let head = rt.mem.read_u32(HEAD);
            let v = rt.mem.read_u32(RING + 4 * (head % SLOTS as u32));
            rt.mem.write_u32(HEAD, head + 1);
            call1(rt, c, CS_LEAVE, LOCK);
            call1(rt, c, SEM_RELEASE, FREE);
            if v == 0 {
                break;
            }
            sum += v as u64;
        }
        c.r[3] = sum;
    }

    let rt = runtime(|rt| {
        rt.register(WORKER, producer);
        rt.register(WORKER + 0x100, consumer);
    });
    rt.object(FREE as u32, || Semaphore::new(SLOTS as u32, SLOTS as u32));
    rt.object(FULL as u32, || Semaphore::new(0, SLOTS as u32));
    let consumers: Vec<u32> = (0..CONSUMERS)
        .map(|_| rt.spawn(WORKER + 0x100, Context::new(), false))
        .collect();
    let p = rt.spawn(WORKER, Context::new(), false);
    rt.join(p);
    let total: u64 = consumers.into_iter().map(|id| rt.join(id)).sum();
    assert_eq!(total, ITEMS * (ITEMS + 1) / 2);
    assert_eq!(rt.mem.read_u32(HEAD) as u64, ITEMS + CONSUMERS);
    assert_eq!(rt.mem.read_u32(TAIL) as u64, ITEMS + CONSUMERS);
}

#[test]
fn suspended_thread_waits_for_resume() {
    fn mark(rt: &Runtime, _: &mut Context) {
        rt.mem.write_u32(0x800, 1);
    }
    let rt = runtime(|rt| rt.register(WORKER, mark));
    let id = rt.spawn(WORKER, Context::new(), true);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(rt.mem.read_u32(0x800), 0, "suspended thread ran");
    assert_eq!(rt.resume(id), 1);
    rt.join(id);
    assert_eq!(rt.mem.read_u32(0x800), 1);
}

/// Waits (up to 10 s) until `f` holds.
fn eventually(mut f: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < end, "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn running_thread_stops_at_its_next_call_while_suspended() {
    const COUNT: u32 = 0x900;
    const STOP: u32 = 0x904;
    fn tick(rt: &Runtime, _: &mut Context) {
        rt.mem.write_u32(COUNT, rt.mem.read_u32(COUNT) + 1);
    }
    fn spin(rt: &Runtime, c: &mut Context) {
        while rt.mem.read_u32(STOP) == 0 {
            rt.call(c, WORKER + 0x100);
        }
    }
    let rt = runtime(|rt| {
        rt.register(WORKER, spin);
        rt.register(WORKER + 0x100, tick);
    });
    let id = rt.spawn(WORKER, Context::new(), false);
    eventually(|| rt.mem.read_u32(COUNT) > 0);
    rt.suspend(id);
    // A call already past its suspend check may still finish.
    let at_suspend = rt.mem.read_u32(COUNT);
    std::thread::sleep(Duration::from_millis(100));
    let settled = rt.mem.read_u32(COUNT);
    assert!(settled <= at_suspend + 1, "{at_suspend} -> {settled}");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        rt.mem.read_u32(COUNT),
        settled,
        "suspended thread kept running"
    );
    rt.resume(id);
    eventually(|| rt.mem.read_u32(COUNT) > settled);
    rt.mem.write_u32(STOP, 1);
    rt.join(id);
}

#[test]
fn each_thread_knows_its_id_and_returns_r3() {
    fn who(_: &Runtime, c: &mut Context) {
        c.r[3] = Runtime::current_thread().unwrap() as u64 * 100;
    }
    let rt = runtime(|rt| rt.register(WORKER, who));
    let ids: Vec<u32> = (0..4)
        .map(|_| rt.spawn(WORKER, Context::new(), false))
        .collect();
    for id in ids {
        assert_eq!(rt.join(id), id as u64 * 100);
    }
    assert_eq!(Runtime::current_thread(), None);
}

#[test]
fn trace_sees_calls_from_every_thread() {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    fn trace(_: &mut Context, addr: u32) {
        if addr == CS_ENTER {
            CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn worker(rt: &Runtime, c: &mut Context) {
        for _ in 0..100 {
            call1(rt, c, CS_ENTER, 0xA00);
            call1(rt, c, CS_LEAVE, 0xA00);
        }
    }
    let rt = runtime(|rt| {
        rt.register(WORKER, worker);
        rt.trace = Some(trace);
    });
    run(&rt, 4, WORKER, 0, 0);
    assert_eq!(CALLS.load(Ordering::Relaxed), 400);
}

#[test]
#[should_panic(expected = "call budget of 50")]
fn call_budget_applies_to_each_guest_thread() {
    fn forever(rt: &Runtime, c: &mut Context) {
        loop {
            call1(rt, c, CS_ENTER, 0xB00);
            call1(rt, c, CS_LEAVE, 0xB00);
        }
    }
    let rt = runtime(|rt| {
        rt.register(WORKER, forever);
        rt.call_budget = 50;
    });
    let id = rt.spawn(WORKER, Context::new(), false);
    rt.join(id);
}

#[test]
#[should_panic(expected = "no guest function at 0x82dead00")]
fn panic_in_a_guest_thread_reaches_the_joiner() {
    fn bad(rt: &Runtime, c: &mut Context) {
        rt.call(c, 0x82DE_AD00);
    }
    let rt = runtime(|rt| rt.register(WORKER, bad));
    let id = rt.spawn(WORKER, Context::new(), false);
    rt.join(id);
}
