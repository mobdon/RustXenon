//! Every guest call goes through one choke point so that hooks can override
//! any function by address, calls can be traced and budgeted, and suspend
//! requests take effect.
//!
//! Guest threads: each one runs on its own host thread at the same time as
//! the others (the console has three cores with two hardware threads each).
//! They share one [`Runtime`], so everything reachable from `&Runtime` is
//! safe to use from several threads. Set up functions and overrides first,
//! then call [`Runtime::share`] before starting threads.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::thread::{self, GuestThread};
use crate::{Context, GuestMemory};

/// A recompiled (or hand-written override) guest function.
pub type GuestFn = fn(&Runtime, &mut Context);

/// Host stack for each guest thread. Recompiled code nests host calls as
/// deep as the guest does, with larger frames; the guest's own stack lives
/// in guest memory.
const THREAD_STACK: usize = 16 << 20;

/// Largest flat call table, in slots (8 bytes each): 128 MB of host memory,
/// 64 MB of guest code. A larger span keeps the hash maps only.
const MAX_SLOTS: usize = 16 << 20;

pub struct Runtime {
    pub mem: GuestMemory,
    /// Game functions by guest address.
    functions: HashMap<u32, GuestFn>,
    /// Overrides installed by hooks; checked before `functions`.
    overrides: HashMap<u32, GuestFn>,
    /// Flat copy of both maps over the span of `functions`, one slot per
    /// instruction from `table_base`, built by [`Runtime::share`]. Overrides
    /// are already merged in, so a call is one index. Calls outside the span
    /// (or before `share`) use the maps.
    table: Vec<Option<GuestFn>>,
    table_base: u32,
    /// A guest thread stops with a panic once it has made more calls than
    /// this (counted in [`Context::calls`]); catches runaway loops in tests.
    pub call_budget: u64,
    /// Called before every guest call with the caller's context and target.
    pub trace: Option<fn(&Context, u32)>,
    /// Called after every guest call returns (not when it unwinds), with
    /// the callee's context and the target; for timing calls.
    pub after: Option<fn(&Context, u32)>,
    /// Whether a call must check `trace`, `after` or `call_budget`. Set by
    /// [`Runtime::share`] (the fields cannot change after it); true before,
    /// so calls on an unshared runtime always check.
    slow: bool,
    threads: Mutex<HashMap<u32, Arc<GuestThread>>>,
    next_thread: AtomicU32,
    /// Host-side objects (locks, events, semaphores) by guest address.
    objects: Mutex<HashMap<u32, Arc<dyn Any + Send + Sync>>>,
    me: OnceLock<Weak<Runtime>>,
}

impl Runtime {
    pub fn new(mem: GuestMemory) -> Self {
        Self {
            mem,
            functions: HashMap::new(),
            overrides: HashMap::new(),
            table: Vec::new(),
            table_base: 0,
            call_budget: u64::MAX,
            trace: None,
            after: None,
            slow: true,
            threads: Mutex::new(HashMap::new()),
            next_thread: AtomicU32::new(1),
            objects: Mutex::new(HashMap::new()),
            me: OnceLock::new(),
        }
    }

    /// Registers a recompiled function at its guest address.
    pub fn register(&mut self, addr: u32, f: GuestFn) {
        self.functions.insert(addr, f);
    }

    /// Replaces the function at `addr` with a hook.
    pub fn set_override(&mut self, addr: u32, f: GuestFn) {
        self.overrides.insert(addr, f);
    }

    /// Ends setup: the runtime moves behind an `Arc` so guest threads can
    /// share it. [`Runtime::spawn`] works only after this.
    pub fn share(mut self) -> Arc<Self> {
        self.build_table();
        self.slow = self.trace.is_some() || self.after.is_some() || self.call_budget != u64::MAX;
        let rt = Arc::new(self);
        rt.me.set(Arc::downgrade(&rt)).unwrap();
        rt
    }

    fn build_table(&mut self) {
        let (Some(&lo), Some(&hi)) = (self.functions.keys().min(), self.functions.keys().max())
        else {
            return;
        };
        let lo = lo & !3;
        let slots = ((hi - lo) / 4) as usize + 1;
        if slots > MAX_SLOTS {
            return;
        }
        let mut t = vec![None; slots];
        for (&a, &f) in &self.functions {
            if a & 3 == 0 && (lo..=hi).contains(&a) {
                t[((a - lo) / 4) as usize] = Some(f);
            }
        }
        // Overrides last, so they win.
        for (&a, &f) in &self.overrides {
            if a & 3 == 0 && (lo..=hi).contains(&a) {
                t[((a - lo) / 4) as usize] = Some(f);
            }
        }
        self.table = t;
        self.table_base = lo;
    }

    /// Host memory used by the flat call table, in bytes (0 before
    /// [`Runtime::share`]). Fixed for the run.
    pub fn table_bytes(&self) -> usize {
        self.table.len() * std::mem::size_of::<Option<GuestFn>>()
    }

    #[inline]
    fn lookup(&self, addr: u32) -> Option<GuestFn> {
        let i = addr.wrapping_sub(self.table_base) as usize / 4;
        if addr & 3 == 0 && i < self.table.len() {
            return self.table[i];
        }
        self.overrides
            .get(&addr)
            .or_else(|| self.functions.get(&addr))
            .copied()
    }

    /// Calls the guest function at `addr`. Panics with the address if nothing
    /// is registered there, so an unbuilt range reports where it was reached.
    #[inline]
    pub fn call(&self, ctx: &mut Context, addr: u32) {
        ctx.calls += 1;
        if self.slow {
            return self.call_checked(ctx, addr);
        }
        thread::checkpoint();
        let Some(f) = self.lookup(addr) else {
            missing(addr)
        };
        f(self, ctx);
    }

    /// [`Runtime::call`] with the budget, `trace` and `after` checks.
    #[inline(never)]
    fn call_checked(&self, ctx: &mut Context, addr: u32) {
        if ctx.calls > self.call_budget {
            over_budget(self.call_budget, addr);
        }
        if let Some(trace) = self.trace {
            trace(ctx, addr);
        }
        thread::checkpoint();
        let Some(f) = self.lookup(addr) else {
            missing(addr)
        };
        f(self, ctx);
        if let Some(after) = self.after {
            after(ctx, addr);
        }
    }

    /// Starts a guest thread running the function at `entry` with `ctx`
    /// (arguments in r3.., stack in r1, set up by the caller). It runs on its
    /// own host thread; a suspended thread waits for [`Runtime::resume`]
    /// before its first instruction. Returns the thread id.
    pub fn spawn(&self, entry: u32, mut ctx: Context, suspended: bool) -> u32 {
        let rt = self
            .me
            .get()
            .and_then(Weak::upgrade)
            .expect("Runtime::share must be called before starting guest threads");
        let id = self.next_thread.fetch_add(1, Ordering::Relaxed);
        let t = Arc::new(GuestThread::new(id, suspended as u32));
        let t2 = t.clone();
        let handle = std::thread::Builder::new()
            .name(format!("guest-{id}"))
            .stack_size(THREAD_STACK)
            .spawn(move || {
                let _exit = thread::enter(t2);
                thread::checkpoint();
                rt.call(&mut ctx, entry);
                thread::finish(ctx.r[3]);
            })
            .expect("cannot start a host thread");
        t.set_handle(handle);
        self.threads.lock().unwrap().insert(id, t);
        id
    }

    fn thread(&self, id: u32) -> Arc<GuestThread> {
        self.threads
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .unwrap_or_else(|| panic!("no guest thread {id}"))
    }

    /// Waits for guest thread `id` to return and gives its exit value (r3).
    /// A panic in that thread continues in the caller.
    pub fn join(&self, id: u32) -> u64 {
        self.thread(id).join()
    }

    /// Raises the suspend count of thread `id`; returns the previous count.
    /// The thread stops at its next guest call (or before its first
    /// instruction), not mid-function.
    pub fn suspend(&self, id: u32) -> u32 {
        self.thread(id).suspend()
    }

    /// Lowers the suspend count; the thread continues when it reaches zero.
    /// Returns the previous count.
    pub fn resume(&self, id: u32) -> u32 {
        self.thread(id).resume()
    }

    /// Id of the guest thread running on this host thread (`None` on a host
    /// thread the runtime did not start).
    pub fn current_thread() -> Option<u32> {
        thread::current_id()
    }

    /// The host-side object for the guest object at `addr`, made by `make`
    /// the first time. Panics if a different kind of object is already there.
    pub fn object<T: Any + Send + Sync>(&self, addr: u32, make: impl FnOnce() -> T) -> Arc<T> {
        let obj = self
            .objects
            .lock()
            .unwrap()
            .entry(addr)
            .or_insert_with(|| Arc::new(make()))
            .clone();
        obj.downcast()
            .unwrap_or_else(|_| panic!("guest object at {addr:#010x} has another type"))
    }

    /// The object at `addr` if one exists and has type `T`; creates nothing.
    pub fn find_object<T: Any + Send + Sync>(&self, addr: u32) -> Option<Arc<T>> {
        let obj = self.objects.lock().unwrap().get(&addr)?.clone();
        obj.downcast().ok()
    }

    /// Forgets the object at `addr` (the guest freed or reinitialised it).
    pub fn drop_object(&self, addr: u32) {
        self.objects.lock().unwrap().remove(&addr);
    }

    /// A trap instruction (`tw`, `td`, `twi`, `tdi`) whose condition held.
    /// Games use these for assertions, so stop with the address.
    #[cold]
    pub fn trap(&self, _ctx: &mut Context, pc: u32) -> ! {
        panic!("guest trap at {pc:#010x}");
    }

    /// `sc`. Titles reach the kernel through imports, not system calls.
    #[cold]
    pub fn syscall(&self, _ctx: &mut Context, pc: u32) -> ! {
        panic!("guest system call instruction at {pc:#010x}");
    }

    /// An instruction the translator does not know. Generated code still
    /// compiles; reaching it stops with the address and the word.
    #[cold]
    pub fn unknown(&self, _ctx: &mut Context, pc: u32, word: u32) -> ! {
        panic!("untranslated instruction {word:#010x} at {pc:#010x}");
    }

    /// Time base (`mftb`), ticking at the console's 49.875 MHz.
    pub fn timebase(&self) -> u64 {
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        let t = START.get_or_init(Instant::now).elapsed();
        (t.as_nanos() * 49_875 / 1_000_000) as u64
    }
}

#[cold]
#[inline(never)]
fn missing(addr: u32) -> ! {
    panic!("no guest function at {addr:#010x}{}", wild_hint(addr))
}

#[cold]
#[inline(never)]
fn over_budget(budget: u64, addr: u32) -> ! {
    panic!("guest thread used its call budget of {budget} at {addr:#010x}")
}

/// A call target that is really a common instruction word means code bytes
/// were read as a pointer: an overwritten function pointer or a stale object.
pub fn wild_hint(addr: u32) -> String {
    let what = match addr {
        a if a >> 16 == 0x9421 => "stwu r1,-n(r1), a function's first instruction",
        0x7D88_02A6 => "mflr r12, a function's first instruction",
        0x7C08_02A6 => "mflr r0",
        0x4E80_0020 => "blr",
        0x6000_0000 => "nop",
        _ => return String::new(),
    };
    format!(
        " (that is the instruction {what}, not an address: look for an overwritten function pointer or a stale object)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_one(_: &Runtime, c: &mut Context) {
        c.r[3] += 1;
    }
    fn add_ten(_: &Runtime, c: &mut Context) {
        c.r[3] += 10;
    }

    #[test]
    fn override_wins() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        let mut c = Context::new();
        rt.register(0x8200_0000, add_one);
        rt.call(&mut c, 0x8200_0000);
        assert_eq!(c.r[3], 1);
        rt.set_override(0x8200_0000, add_ten);
        rt.call(&mut c, 0x8200_0000);
        assert_eq!(c.r[3], 11);
    }

    #[test]
    fn shared_table_keeps_overrides_and_outside_calls() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        rt.register(0x8200_0000, add_one);
        rt.register(0x8200_0010, add_one);
        rt.set_override(0x8200_0010, add_ten);
        // An override outside the recompiled span (an import thunk, say).
        rt.set_override(0x9000_0000, add_ten);
        let rt = rt.share();
        assert_eq!(rt.table_bytes(), 5 * 8);
        let mut c = Context::new();
        for a in [0x8200_0000, 0x8200_0010, 0x9000_0000] {
            rt.call(&mut c, a);
        }
        assert_eq!(c.r[3], 21);
        assert!(rt.lookup(0x8200_0004).is_none());
        assert!(rt.lookup(0x8200_0011).is_none());
    }

    #[test]
    #[should_panic(expected = "0x82001000")]
    fn missing_function_reports_address() {
        let rt = Runtime::new(GuestMemory::new(16));
        rt.call(&mut Context::new(), 0x8200_1000);
    }

    #[test]
    #[should_panic(expected = "no guest function at 0x9421fff8 (that is the instruction stwu")]
    fn instruction_as_address_is_named() {
        let rt = Runtime::new(GuestMemory::new(16));
        rt.call(&mut Context::new(), 0x9421_FFF8);
    }

    #[test]
    fn ordinary_address_gets_no_hint() {
        assert_eq!(wild_hint(0x8200_1000), "");
        assert!(wild_hint(0x7D88_02A6).contains("mflr r12"));
    }

    #[test]
    #[should_panic(expected = "call budget of 2 at 0x82000000")]
    fn call_budget_stops_a_runaway_thread() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        rt.register(0x8200_0000, add_one);
        rt.call_budget = 2;
        let mut c = Context::new();
        for _ in 0..3 {
            rt.call(&mut c, 0x8200_0000);
        }
    }

    #[test]
    fn shared_runtime_keeps_hooks_and_budget() {
        use std::sync::atomic::AtomicU64;
        static SEEN: AtomicU64 = AtomicU64::new(0);
        fn before(_: &Context, _: u32) {
            SEEN.fetch_add(1, Ordering::Relaxed);
        }
        fn after(_: &Context, _: u32) {
            SEEN.fetch_add(100, Ordering::Relaxed);
        }
        let mut rt = Runtime::new(GuestMemory::new(16));
        rt.register(0x8200_0000, add_one);
        rt.trace = Some(before);
        rt.after = Some(after);
        rt.call_budget = 2;
        let rt = rt.share();
        let mut c = Context::new();
        rt.call(&mut c, 0x8200_0000);
        rt.call(&mut c, 0x8200_0000);
        assert_eq!((c.r[3], SEEN.load(Ordering::Relaxed)), (2, 202));
        let over = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.call(&mut c, 0x8200_0000)
        }));
        assert!(over.is_err());
        // Without hooks the shared runtime still counts calls.
        let mut plain = Runtime::new(GuestMemory::new(16));
        plain.register(0x8200_0000, add_one);
        let plain = plain.share();
        let mut c = Context::new();
        plain.call(&mut c, 0x8200_0000);
        assert_eq!((c.r[3], c.calls), (1, 1));
    }

    /// Speed of one guest call through `Runtime::call`, for comparing
    /// dispatch layouts: `cargo test -p xenon-rt --release -- --ignored dispatch_speed --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_speed() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        // A large title: 100k functions over 8 MB of code, 500 import thunks.
        for i in 0..100_000u32 {
            rt.register(0x8200_0000 + i * 80, add_one);
        }
        for i in 0..500u32 {
            rt.set_override(0x8280_0000 + i * 16, add_one);
        }
        let rt = rt.share();
        let mut c = Context::new();
        let n = 20_000_000u32;
        let t = std::time::Instant::now();
        let mut x = 1u32;
        for _ in 0..n {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            rt.call(&mut c, 0x8200_0000 + (x >> 8) % 100_000 * 80);
        }
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        assert_eq!(c.r[3], n as u64);
        eprintln!("dispatch: {ns:.2} ns per call");
    }

    #[test]
    #[should_panic(expected = "another type")]
    fn object_type_is_checked() {
        let rt = Runtime::new(GuestMemory::new(16));
        rt.object(0x40, crate::sync::CriticalSection::new);
        rt.object(0x40, || crate::sync::Semaphore::new(0, 1));
    }

    #[test]
    fn find_object_creates_nothing_and_reports_owner() {
        let rt = Runtime::new(GuestMemory::new(16));
        assert!(rt.find_object::<crate::sync::Event>(0x40).is_none());
        let cs = rt.object(0x40, crate::sync::CriticalSection::new);
        assert!(rt.find_object::<crate::sync::Event>(0x40).is_none());
        assert_eq!(cs.owner(), None);
        cs.enter(None);
        let found = rt
            .find_object::<crate::sync::CriticalSection>(0x40)
            .unwrap();
        assert_eq!(found.owner(), Some(std::thread::current().id()));
    }

    #[test]
    #[should_panic(expected = "Runtime::share")]
    fn spawn_needs_a_shared_runtime() {
        Runtime::new(GuestMemory::new(16)).spawn(0, Context::new(), false);
    }
}
