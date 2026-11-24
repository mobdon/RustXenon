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

pub struct Runtime {
    pub mem: GuestMemory,
    /// Game functions by guest address.
    functions: HashMap<u32, GuestFn>,
    /// Overrides installed by hooks; checked before `functions`.
    overrides: HashMap<u32, GuestFn>,
    /// A guest thread stops with a panic once it has made more calls than
    /// this (counted in [`Context::calls`]); catches runaway loops in tests.
    pub call_budget: u64,
    /// Called before every guest call with the caller's context and target.
    pub trace: Option<fn(&Context, u32)>,
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
            call_budget: u64::MAX,
            trace: None,
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
    pub fn share(self) -> Arc<Self> {
        let rt = Arc::new(self);
        rt.me.set(Arc::downgrade(&rt)).unwrap();
        rt
    }

    /// Calls the guest function at `addr`. Panics with the address if nothing
    /// is registered there, so an unbuilt range reports where it was reached.
    pub fn call(&self, ctx: &mut Context, addr: u32) {
        ctx.calls += 1;
        if ctx.calls > self.call_budget {
            panic!(
                "guest thread used its call budget of {} at {addr:#010x}",
                self.call_budget
            );
        }
        if let Some(trace) = self.trace {
            trace(ctx, addr);
        }
        thread::checkpoint();
        let f = self
            .overrides
            .get(&addr)
            .or_else(|| self.functions.get(&addr))
            .copied()
            .unwrap_or_else(|| panic!("no guest function at {addr:#010x}"));
        f(self, ctx);
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
    #[should_panic(expected = "0x82001000")]
    fn missing_function_reports_address() {
        let rt = Runtime::new(GuestMemory::new(16));
        rt.call(&mut Context::new(), 0x8200_1000);
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
    #[should_panic(expected = "another type")]
    fn object_type_is_checked() {
        let rt = Runtime::new(GuestMemory::new(16));
        rt.object(0x40, crate::sync::CriticalSection::new);
        rt.object(0x40, || crate::sync::Semaphore::new(0, 1));
    }

    #[test]
    #[should_panic(expected = "Runtime::share")]
    fn spawn_needs_a_shared_runtime() {
        Runtime::new(GuestMemory::new(16)).spawn(0, Context::new(), false);
    }
}
