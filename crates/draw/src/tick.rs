//! A cheap clock for the draw thread's per-draw timings.
//!
//! `Instant::now` is QueryPerformanceCounter and a division on Windows:
//! tens of nanoseconds, and with several reads a draw it was a twentieth of
//! the draw thread on busy frames. On x86-64 this reads the time-stamp
//! counter instead (a few nanoseconds) and converts tick counts to time
//! only when two reads are subtracted.

use std::ops::Sub;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A point in time, comparable only with another `Tick`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Tick(u64);

impl Tick {
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn now() -> Self {
        // shortcut: assumes an invariant time-stamp counter (every x86-64
        // CPU of the last fifteen years); an older one only skews timings.
        #[allow(unused_unsafe)]
        Tick(unsafe { core::arch::x86_64::_rdtsc() })
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) fn now() -> Self {
        Tick(epoch().elapsed().as_nanos() as u64)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Nanoseconds per tick, measured once against `Instant`.
fn ns_per_tick() -> f64 {
    static RATE: OnceLock<f64> = OnceLock::new();
    *RATE.get_or_init(|| {
        if cfg!(not(target_arch = "x86_64")) {
            return 1.0;
        }
        let (i0, t0) = (Instant::now(), Tick::now());
        std::thread::sleep(Duration::from_millis(10));
        let (i1, t1) = (Instant::now(), Tick::now());
        let ticks = t1.0.saturating_sub(t0.0).max(1);
        (i1 - i0).as_nanos() as f64 / ticks as f64
    })
}

impl Sub for Tick {
    type Output = Duration;
    /// The time from `earlier` to `self`; zero if `earlier` is later.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, earlier: Tick) -> Duration {
        let ticks = self.0.saturating_sub(earlier.0);
        Duration::from_nanos((ticks as f64 * ns_per_tick()) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_measure_time_like_instant() {
        let (i0, t0) = (Instant::now(), Tick::now());
        std::thread::sleep(Duration::from_millis(30));
        let (t1, i1) = (Tick::now(), Instant::now());
        let (ticks, real) = (t1 - t0, i1 - i0);
        // Within a millisecond (and a tenth) of the real time.
        let diff = ticks.abs_diff(real);
        assert!(
            diff < Duration::from_millis(1) + real / 10,
            "{ticks:?} vs {real:?}"
        );
        assert_eq!(t0 - t1, Duration::ZERO);
    }
}
