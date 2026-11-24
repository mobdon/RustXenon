//! Per-guest-thread state: suspend count, exit value and host join handle.
//! The runtime's public API is on [`crate::Runtime`].

use std::cell::RefCell;
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
    static CURRENT: RefCell<Option<Arc<GuestThread>>> = const { RefCell::new(None) };
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

    fn wait_while_suspended(&self) {
        if self.suspend_hint.load(Ordering::Relaxed) == 0 {
            return;
        }
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
        if let Some(t) = CURRENT.with(|c| c.borrow_mut().take()) {
            t.mark_done(None);
        }
    }
}

/// Binds `t` to the calling host thread.
pub(crate) fn enter(t: Arc<GuestThread>) -> Exit {
    CURRENT.with(|c| *c.borrow_mut() = Some(t));
    Exit
}

/// Records a normal return with exit value `code`.
pub(crate) fn finish(code: u64) {
    CURRENT.with(|c| {
        if let Some(t) = &*c.borrow() {
            t.mark_done(Some(code));
        }
    });
}

/// Blocks while the current guest thread is suspended. Called on every
/// guest call.
#[inline]
pub(crate) fn checkpoint() {
    CURRENT.with(|c| {
        if let Some(t) = &*c.borrow() {
            t.wait_while_suspended();
        }
    });
}

pub(crate) fn current_id() -> Option<u32> {
    CURRENT.with(|c| c.borrow().as_ref().map(|t| t.id))
}
