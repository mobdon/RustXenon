//! Host-side synchronisation objects that guest locks, events and semaphores
//! map onto. The system call replacements (step 4) look them up by the guest
//! address of the guest object with [`crate::Runtime::object`].
//!
//! Waits take an optional timeout and return `true` when the object was
//! acquired, `false` when the timeout ran out.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Threads inside [`wait_any`], and a generation bumped (with a wake) each
/// time an object may have become signalled while there are any. Waits for
/// several objects sleep on this instead of polling, because a short sleep
/// lasts about 1 ms on Windows and made every such wait wake that late.
static ANY_WAITERS: AtomicUsize = AtomicUsize::new(0);
static ANY: (Mutex<u64>, Condvar) = (Mutex::new(0), Condvar::new());

/// Wakes [`wait_any`] callers. Call after an object's state change that may
/// satisfy a wait, with the object's own lock released. Costs one atomic
/// load when nobody waits for several objects. [`CriticalSection`] does not
/// call it, since guest critical sections are never in such a wait; mutant
/// releases call it themselves.
pub fn signal_any() {
    if ANY_WAITERS.load(Ordering::SeqCst) > 0 {
        *ANY.0.lock().unwrap() += 1;
        ANY.1.notify_all();
    }
}

/// Calls `ready` until it gives a value or `timeout` runs out, sleeping
/// between calls until some object here is signalled. `ready` must not
/// block (take objects with a zero timeout).
///
/// No wake is lost: this thread counts itself in before it reads the
/// generation, and reads the generation before it calls `ready`, so a
/// signal after `ready` looked changes the generation and the wait below
/// returns at once.
pub fn wait_any<T>(timeout: Option<Duration>, mut ready: impl FnMut() -> Option<T>) -> Option<T> {
    struct Leave;
    impl Drop for Leave {
        fn drop(&mut self) {
            ANY_WAITERS.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let deadline = timeout.map(|t| Instant::now() + t);
    ANY_WAITERS.fetch_add(1, Ordering::SeqCst);
    let _leave = Leave;
    let mut seen = *ANY.0.lock().unwrap();
    loop {
        if let Some(v) = ready() {
            return Some(v);
        }
        // Capped, so a state change made without signal_any is still seen.
        let mut nap = Duration::from_millis(1);
        if let Some(d) = deadline {
            let now = Instant::now();
            if now >= d {
                return None;
            }
            nap = nap.min(d - now);
        }
        let mut g = ANY.0.lock().unwrap();
        if *g == seen {
            g = ANY.1.wait_timeout(g, nap).unwrap().0;
        }
        seen = *g;
    }
}

/// Waits on `cv` until `ready` holds or the timeout runs out. Returns the
/// guard and whether `ready` held.
fn wait_until<'a, T>(
    cv: &Condvar,
    mut g: MutexGuard<'a, T>,
    timeout: Option<Duration>,
    mut ready: impl FnMut(&mut T) -> bool,
) -> (MutexGuard<'a, T>, bool) {
    let deadline = timeout.map(|t| Instant::now() + t);
    loop {
        if ready(&mut g) {
            return (g, true);
        }
        match deadline {
            None => g = cv.wait(g).unwrap(),
            Some(d) => {
                let now = Instant::now();
                if now >= d {
                    return (g, false);
                }
                g = cv.wait_timeout(g, d - now).unwrap().0;
            }
        }
    }
}

/// An event: manual reset (stays set until reset, wakes every waiter) or
/// auto reset (each set releases one waiter).
pub struct Event {
    manual: bool,
    set: Mutex<bool>,
    cv: Condvar,
}

impl Event {
    pub fn new(manual: bool, initially_set: bool) -> Self {
        Self {
            manual,
            set: Mutex::new(initially_set),
            cv: Condvar::new(),
        }
    }

    /// Signals the event; returns whether it was already set.
    pub fn set(&self) -> bool {
        let mut s = self.set.lock().unwrap();
        let was = std::mem::replace(&mut *s, true);
        if self.manual {
            self.cv.notify_all();
        } else {
            self.cv.notify_one();
        }
        drop(s);
        signal_any();
        was
    }

    /// Whether the event is set now (for reports; it may change at once).
    pub fn is_set(&self) -> bool {
        *self.set.lock().unwrap()
    }

    /// Clears the event; returns whether it was set.
    pub fn reset(&self) -> bool {
        std::mem::replace(&mut *self.set.lock().unwrap(), false)
    }

    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let manual = self.manual;
        let g = self.set.lock().unwrap();
        wait_until(&self.cv, g, timeout, |s| {
            let ok = *s;
            if ok && !manual {
                *s = false;
            }
            ok
        })
        .1
    }
}

/// A counting semaphore with a maximum count.
pub struct Semaphore {
    max: u32,
    count: Mutex<u32>,
    cv: Condvar,
}

impl Semaphore {
    pub fn new(initial: u32, max: u32) -> Self {
        assert!(
            initial <= max,
            "semaphore count {initial} above maximum {max}"
        );
        Self {
            max,
            count: Mutex::new(initial),
            cv: Condvar::new(),
        }
    }

    /// Adds `n`; returns the previous count, or `None` (and changes nothing)
    /// if that would pass the maximum, as the console's kernel refuses it.
    pub fn release(&self, n: u32) -> Option<u32> {
        let mut c = self.count.lock().unwrap();
        let prev = *c;
        *c = prev.checked_add(n).filter(|&v| v <= self.max)?;
        for _ in 0..n {
            self.cv.notify_one();
        }
        drop(c);
        signal_any();
        Some(prev)
    }

    /// The count now (for reports).
    pub fn count(&self) -> u32 {
        *self.count.lock().unwrap()
    }

    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let g = self.count.lock().unwrap();
        wait_until(&self.cv, g, timeout, |c| {
            let ok = *c > 0;
            if ok {
                *c -= 1;
            }
            ok
        })
        .1
    }
}

/// A recursive lock owned by one host thread at a time (guest critical
/// sections and mutants).
#[derive(Default)]
pub struct CriticalSection {
    state: Mutex<(Option<ThreadId>, u32)>,
    cv: Condvar,
}

impl CriticalSection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter(&self, timeout: Option<Duration>) -> bool {
        let me = thread::current().id();
        let g = self.state.lock().unwrap();
        wait_until(&self.cv, g, timeout, |(owner, depth)| match owner {
            Some(o) if *o != me => false,
            _ => {
                *owner = Some(me);
                *depth += 1;
                true
            }
        })
        .1
    }

    /// The host thread holding the lock now, if any (for reports).
    pub fn owner(&self) -> Option<ThreadId> {
        self.state.lock().unwrap().0
    }

    pub fn try_enter(&self) -> bool {
        self.enter(Some(Duration::ZERO))
    }

    /// Leaves once. Stops if the calling thread does not own the lock, which
    /// is a guest bug worth seeing.
    pub fn leave(&self) {
        assert!(
            self.try_leave(),
            "critical section left by a thread that does not own it"
        );
    }

    /// Leaves once if the calling thread owns the lock; false (and no
    /// change) otherwise, for callers that report it to the guest.
    pub fn try_leave(&self) -> bool {
        let mut g = self.state.lock().unwrap();
        if g.0 != Some(thread::current().id()) {
            return false;
        }
        g.1 -= 1;
        if g.1 == 0 {
            g.0 = None;
            self.cv.notify_one();
        }
        true
    }

    /// Leaves once on behalf of whichever thread holds the lock, as the
    /// console's critical sections allow (they only count); false if it
    /// is free.
    pub fn leave_for_owner(&self) -> bool {
        let mut g = self.state.lock().unwrap();
        if g.0.is_none() {
            return false;
        }
        g.1 -= 1;
        if g.1 == 0 {
            g.0 = None;
            self.cv.notify_one();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHORT: Option<Duration> = Some(Duration::from_millis(20));
    const LONG: Option<Duration> = Some(Duration::from_secs(5));

    /// Runs `f` on another thread after a short delay.
    fn later(f: impl FnOnce() + Send + 'static) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            f()
        })
    }

    #[test]
    fn wait_any_returns_the_object_that_was_set() {
        let a = std::sync::Arc::new([Event::new(false, false), Event::new(false, false)]);
        let b = a.clone();
        let t = later(move || {
            b[1].set();
        });
        let ready = || (0..2).find(|&i| a[i].wait(Some(Duration::ZERO)));
        assert_eq!(wait_any(LONG, ready), Some(1));
        t.join().unwrap();
        // The auto-reset event was taken, so nothing is ready now.
        assert_eq!(wait_any(SHORT, ready), None);
        assert_eq!(wait_any(Some(Duration::ZERO), ready), None);
    }

    #[test]
    fn wait_any_wakes_for_a_semaphore_release() {
        let s = std::sync::Arc::new(Semaphore::new(0, 1));
        let s2 = s.clone();
        let t = later(move || {
            s2.release(1);
        });
        assert_eq!(
            wait_any(LONG, || s.wait(Some(Duration::ZERO)).then_some(0)),
            Some(0)
        );
        t.join().unwrap();
    }

    #[test]
    fn wait_any_wakes_for_a_mutant_release() {
        let cs = std::sync::Arc::new(CriticalSection::new());
        let (held, take) = std::sync::mpsc::channel();
        let cs2 = cs.clone();
        let t = thread::spawn(move || {
            assert!(cs2.enter(None));
            held.send(()).unwrap();
            thread::sleep(Duration::from_millis(20));
            cs2.leave();
            signal_any();
        });
        take.recv().unwrap();
        assert_eq!(wait_any(LONG, || cs.try_enter().then_some(0)), Some(0));
        cs.leave();
        t.join().unwrap();
    }

    #[test]
    fn set_and_release_bump_the_generation_while_someone_waits() {
        let bumps = |f: &dyn Fn()| {
            ANY_WAITERS.fetch_add(1, Ordering::SeqCst);
            let before = *ANY.0.lock().unwrap();
            f();
            let after = *ANY.0.lock().unwrap();
            ANY_WAITERS.fetch_sub(1, Ordering::SeqCst);
            after > before
        };
        let e = Event::new(true, false);
        assert!(bumps(&|| {
            e.set();
        }));
        let s = Semaphore::new(0, 1);
        assert!(bumps(&|| {
            s.release(1);
        }));
    }

    #[test]
    fn auto_reset_event_releases_one_wait() {
        let e = Event::new(false, false);
        assert!(!e.wait(SHORT));
        e.set();
        assert!(e.wait(SHORT));
        assert!(!e.wait(SHORT));
    }

    #[test]
    fn manual_reset_event_stays_set() {
        let e = Event::new(true, true);
        assert!(e.wait(SHORT) && e.wait(SHORT));
        assert!(e.reset());
        assert!(!e.wait(SHORT));
    }

    #[test]
    fn semaphore_respects_its_maximum() {
        let s = Semaphore::new(1, 2);
        assert_eq!(s.release(1), Some(1));
        assert_eq!(s.release(1), None);
        assert!(s.wait(SHORT) && s.wait(SHORT));
        assert!(!s.wait(SHORT));
    }

    #[test]
    fn critical_section_is_recursive_and_exclusive() {
        let cs = std::sync::Arc::new(CriticalSection::new());
        assert!(cs.enter(None) && cs.enter(None));
        let other = cs.clone();
        assert!(!std::thread::spawn(move || other.try_enter())
            .join()
            .unwrap());
        cs.leave();
        cs.leave();
        let other = cs.clone();
        assert!(std::thread::spawn(move || {
            let ok = other.try_enter();
            other.leave();
            ok
        })
        .join()
        .unwrap());
    }

    #[test]
    fn try_leave_refuses_a_thread_that_does_not_own_it() {
        let cs = std::sync::Arc::new(CriticalSection::new());
        assert!(!cs.try_leave(), "unowned");
        assert!(cs.enter(None));
        let other = cs.clone();
        assert!(!std::thread::spawn(move || other.try_leave())
            .join()
            .unwrap());
        // Still held by this thread: another thread cannot take it.
        let other = cs.clone();
        assert!(!std::thread::spawn(move || other.try_enter())
            .join()
            .unwrap());
        assert!(cs.try_leave());
        assert!(!cs.try_leave());
    }

    #[test]
    fn leave_for_owner_frees_a_lock_another_thread_holds() {
        let cs = std::sync::Arc::new(CriticalSection::new());
        assert!(!cs.leave_for_owner(), "free");
        assert!(cs.enter(None) && cs.enter(None));
        let other = cs.clone();
        assert!(std::thread::spawn(move || other.leave_for_owner())
            .join()
            .unwrap());
        // Once more held, then free: another thread can take it.
        let other = cs.clone();
        assert!(!std::thread::spawn(move || other.try_enter())
            .join()
            .unwrap());
        let other = cs.clone();
        assert!(std::thread::spawn(move || other.leave_for_owner())
            .join()
            .unwrap());
        let other = cs.clone();
        assert!(std::thread::spawn(move || {
            let ok = other.try_enter();
            other.leave();
            ok
        })
        .join()
        .unwrap());
    }
}
