//! Host-side synchronisation objects that guest locks, events and semaphores
//! map onto. The system call replacements (step 4) look them up by the guest
//! address of the guest object with [`crate::Runtime::object`].
//!
//! Waits take an optional timeout and return `true` when the object was
//! acquired, `false` when the timeout ran out.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

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
        was
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
        Some(prev)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHORT: Option<Duration> = Some(Duration::from_millis(20));

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
}
