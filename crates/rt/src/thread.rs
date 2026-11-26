//! Per-guest-thread state: suspend count, exit value and host join handle.
//! The runtime's public API is on [`crate::Runtime`].

use std::cell::{Cell, OnceCell};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

#[derive(Default)]
struct State {
    suspend: u32,
    /// Set when the thread has stopped; `Some(r3)` if it returned normally.
    done: Option<Option<u64>>,
}

pub(crate) struct GuestThread {
    id: u32,
    /// Copy of `state.suspend` read without the lock on every guest call.
    suspend_hint: AtomicU32,
    state: Mutex<State>,
    cv: Condvar,
    handle: Mutex<Option<JoinHandle<()>>>,
}

thread_local! {
    // Set once per host thread and never borrowed mutably, so the per-call
    // checkpoint has no borrow flag that could panic (seen once in a game run).
    static CURRENT: OnceCell<Arc<GuestThread>> = const { OnceCell::new() };
    // The same thread as a plain pointer for the per-call checkpoint: a
    // const Cell with no destructor is one TLS load, with no lazy-init or
    // destructor-state check. Null when no guest thread is bound or after
    // `Exit` (the runtime's thread map keeps the `GuestThread` alive too).
    static FAST: Cell<*const GuestThread> = const { Cell::new(std::ptr::null()) };
}

impl GuestThread {
    pub(crate) fn new(id: u32, suspend: u32) -> Self {
        Self {
            id,
            suspend_hint: AtomicU32::new(suspend),
            state: Mutex::new(State {
                suspend,
                done: None,
            }),
            cv: Condvar::new(),
            handle: Mutex::new(None),
        }
    }

    pub(crate) fn set_handle(&self, h: JoinHandle<()>) {
        *self.handle.lock().unwrap() = Some(h);
    }

    pub(crate) fn suspend(&self) -> u32 {
        let mut s = self.state.lock().unwrap();
        s.suspend += 1;
        self.suspend_hint.store(s.suspend, Ordering::Relaxed);
        s.suspend - 1
    }

    pub(crate) fn resume(&self) -> u32 {
        let mut s = self.state.lock().unwrap();
        let prev = s.suspend;
        s.suspend = prev.saturating_sub(1);
        self.suspend_hint.store(s.suspend, Ordering::Relaxed);
        self.cv.notify_all();
        prev
    }

    #[inline]
    fn wait_while_suspended(&self) {
        if self.suspend_hint.load(Ordering::Relaxed) != 0 {
            self.wait_slow();
        }
    }

    #[cold]
    #[inline(never)]
    fn wait_slow(&self) {
        let mut s = self.state.lock().unwrap();
        while s.suspend > 0 {
            s = self.cv.wait(s).unwrap();
        }
    }

    pub(crate) fn join(&self) -> u64 {
        let mut s = self.state.lock().unwrap();
        while s.done.is_none() {
            s = self.cv.wait(s).unwrap();
        }
        let code = s.done.unwrap();
        drop(s);
        // The first joiner collects the host thread and any panic.
        if let Some(h) = self.handle.lock().unwrap().take() {
            if let Err(p) = h.join() {
                std::panic::resume_unwind(p);
            }
        }
        code.unwrap_or_else(|| panic!("guest thread {} stopped with a panic", self.id))
    }

    fn mark_done(&self, code: Option<u64>) {
        let mut s = self.state.lock().unwrap();
        if s.done.is_none() {
            s.done = Some(code);
        }
        self.cv.notify_all();
    }
}

/// Marks the thread stopped when dropped, so joiners wake even if the guest
/// code panicked.
pub(crate) struct Exit;

impl Drop for Exit {
    fn drop(&mut self) {
        FAST.with(|f| f.set(std::ptr::null()));
        CURRENT.with(|c| {
            if let Some(t) = c.get() {
                t.mark_done(None);
            }
        });
    }
}

/// Binds `t` to the calling host thread.
pub(crate) fn enter(t: Arc<GuestThread>) -> Exit {
    CURRENT.with(|c| {
        let p = Arc::as_ptr(&t);
        assert!(c.set(t).is_ok(), "host thread already runs a guest thread");
        FAST.with(|f| f.set(p));
    });
    Exit
}

/// Records a normal return with exit value `code`.
pub(crate) fn finish(code: u64) {
    CURRENT.with(|c| {
        if let Some(t) = c.get() {
            t.mark_done(Some(code));
        }
    });
}

/// Blocks while the current guest thread is suspended. Called on every
/// guest call.
#[inline]
pub(crate) fn checkpoint() {
    let t = FAST.with(Cell::get);
    // SAFETY: non-null only between `enter` and `Exit`, while `CURRENT` on
    // this same thread holds an `Arc` to it.
    if let Some(t) = unsafe { t.as_ref() } {
        t.wait_while_suspended();
    }
}

pub(crate) fn current_id() -> Option<u32> {
    CURRENT.with(|c| c.get().map(|t| t.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hammers the per-call checkpoint while another thread suspends and
    /// resumes it, then checks the exit value and that a host thread cannot
    /// be bound twice.
    #[test]
    fn checkpoint_survives_suspend_resume_storm() {
        let t = Arc::new(GuestThread::new(7, 0));
        let t2 = t.clone();
        let h = std::thread::spawn(move || {
            let _exit = enter(t2.clone());
            assert_eq!(current_id(), Some(7));
            for _ in 0..200_000 {
                checkpoint();
            }
            let again = std::panic::catch_unwind(|| drop(enter(t2)));
            assert!(again.is_err());
            finish(42);
        });
        for _ in 0..2_000 {
            t.suspend();
            t.resume();
        }
        assert_eq!(t.join(), 42);
        h.join().unwrap();
        assert_eq!(current_id(), None);
    }
}
