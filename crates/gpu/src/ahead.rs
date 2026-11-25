//! A queue that lets the command processor run ahead of a drawer on another
//! thread, by at most a set number of frames (swaps) and batches.
//!
//! The producer pushes each batch with the number of swaps it holds and
//! blocks while more than `frames` swaps or `batches` batches are not yet
//! finished. The consumer pops a batch, works on it, then calls
//! [`Ahead::done`] with its swaps. Either side can [`Ahead::close`] it:
//! pushes are then dropped and pops return `None` once the queue is empty.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

pub struct Ahead<T> {
    frames: u32,
    batches: usize,
    state: Mutex<State<T>>,
    changed: Condvar,
}

struct State<T> {
    queue: VecDeque<(T, u32)>,
    /// Swaps pushed and not yet done (queued or being worked on).
    swaps: u32,
    closed: bool,
}

impl<T> Ahead<T> {
    /// `frames` swaps may be unfinished while the producer goes on (0 makes
    /// it wait at each swap until the consumer is through it); `batches`
    /// (at least 1) bounds the queue between swaps.
    pub fn new(frames: u32, batches: usize) -> Self {
        Ahead {
            frames,
            batches: batches.max(1),
            state: Mutex::new(State {
                queue: VecDeque::new(),
                swaps: 0,
                closed: false,
            }),
            changed: Condvar::new(),
        }
    }

    /// Queues `item` holding `swaps` swaps, then waits while the consumer is
    /// too far behind. Returns false (dropping `item`) when closed.
    pub fn push(&self, item: T, swaps: u32) -> bool {
        let mut s = self.state.lock().unwrap();
        while !s.closed && s.queue.len() >= self.batches {
            s = self.changed.wait(s).unwrap();
        }
        if s.closed {
            return false;
        }
        s.queue.push_back((item, swaps));
        s.swaps += swaps;
        self.changed.notify_all();
        while !s.closed && s.swaps > self.frames {
            s = self.changed.wait(s).unwrap();
        }
        true
    }

    /// The next batch and its swaps; blocks while empty, `None` once closed
    /// and drained.
    pub fn pop(&self) -> Option<(T, u32)> {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(b) = s.queue.pop_front() {
                self.changed.notify_all();
                return Some(b);
            }
            if s.closed {
                return None;
            }
            s = self.changed.wait(s).unwrap();
        }
    }

    /// The consumer finished a popped batch holding `swaps` swaps.
    pub fn done(&self, swaps: u32) {
        let mut s = self.state.lock().unwrap();
        s.swaps = s.swaps.saturating_sub(swaps);
        self.changed.notify_all();
    }

    pub fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.changed.notify_all();
    }

    /// Swaps pushed and not yet done.
    pub fn swaps_behind(&self) -> u32 {
        self.state.lock().unwrap().swaps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    fn wait_until(f: impl Fn() -> bool) {
        for _ in 0..500 {
            if f() {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out");
    }

    #[test]
    fn batches_come_out_in_order() {
        let q = Ahead::new(4, 8);
        assert!(q.push("a", 0));
        assert!(q.push("b", 1));
        assert_eq!(q.pop(), Some(("a", 0)));
        assert_eq!(q.pop(), Some(("b", 1)));
        assert_eq!(q.swaps_behind(), 1);
        q.done(1);
        assert_eq!(q.swaps_behind(), 0);
    }

    /// With one frame ahead, the push of the second unfinished swap blocks
    /// until the consumer is done with the first.
    #[test]
    fn producer_waits_one_frame_ahead() {
        let q = Arc::new(Ahead::new(1, 64));
        assert!(q.push(1, 1));
        assert!(q.push(2, 0), "work between swaps does not wait");
        let back = Arc::new(AtomicBool::new(false));
        let t = {
            let (q, back) = (q.clone(), back.clone());
            std::thread::spawn(move || {
                q.push(3, 1);
                back.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(!back.load(Ordering::SeqCst), "two frames unfinished");
        let (_, s) = q.pop().unwrap();
        q.done(s);
        wait_until(|| back.load(Ordering::SeqCst));
        t.join().unwrap();
    }

    #[test]
    fn producer_waits_for_room_between_swaps() {
        let q = Arc::new(Ahead::new(1, 2));
        q.push(1, 0);
        q.push(2, 0);
        let back = Arc::new(AtomicBool::new(false));
        let t = {
            let (q, back) = (q.clone(), back.clone());
            std::thread::spawn(move || {
                q.push(3, 0);
                back.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(!back.load(Ordering::SeqCst));
        q.pop();
        wait_until(|| back.load(Ordering::SeqCst));
        t.join().unwrap();
    }

    /// Closing frees a waiting producer and drops later pushes; the
    /// consumer still drains what was queued.
    #[test]
    fn close_releases_both_sides() {
        let q = Arc::new(Ahead::new(0, 8));
        let t = {
            let q = q.clone();
            std::thread::spawn(move || q.push(1, 1))
        };
        wait_until(|| q.swaps_behind() == 1);
        q.close();
        assert!(t.join().unwrap(), "the batch was queued before the close");
        assert!(!q.push(2, 0));
        assert_eq!(q.pop(), Some((1, 1)));
        assert_eq!(q.pop(), None);
    }
}
