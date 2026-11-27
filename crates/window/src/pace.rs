//! How evenly frames reach the screen: the spread of the intervals between
//! frames as they are presented, summed up once a second.

use std::time::{Duration, Instant};

/// An interval longer than this counts as a hitch: a frame shown three
/// 60 Hz refreshes after the one before it, or later.
const HITCH: Duration = Duration::from_millis(25);

/// The intervals between presented frames over the last second or so.
#[derive(Default)]
pub struct Pacing {
    last: Option<Instant>,
    since: Option<Instant>,
    gaps: Vec<Duration>,
}

/// One second's intervals: how many, their median, 99th percentile and
/// longest, and how many were hitches (longer than 25 ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub frames: usize,
    pub p50: Duration,
    pub p99: Duration,
    pub max: Duration,
    pub hitches: usize,
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        write!(
            f,
            "[pace] {} frames shown, interval ms p50 {:.1} p99 {:.1} max {:.1}, over 25 ms: {}",
            self.frames,
            ms(self.p50),
            ms(self.p99),
            ms(self.max),
            self.hitches
        )
    }
}

impl Pacing {
    /// A frame was presented `at`; once a second has passed since the last
    /// summary, the summary of the intervals since.
    pub fn frame(&mut self, at: Instant) -> Option<Spread> {
        if let Some(last) = self.last.replace(at) {
            self.gaps.push(at.saturating_duration_since(last));
        }
        let since = *self.since.get_or_insert(at);
        if at.saturating_duration_since(since) < Duration::from_secs(1) {
            return None;
        }
        self.since = Some(at);
        spread(std::mem::take(&mut self.gaps))
    }
}

/// The spread of `gaps`, none for none. Percentiles are nearest-rank.
fn spread(mut gaps: Vec<Duration>) -> Option<Spread> {
    gaps.sort_unstable();
    let max = *gaps.last()?;
    let n = gaps.len();
    let rank = |p: usize| gaps[(n * p).div_ceil(100).max(1) - 1];
    Some(Spread {
        frames: n,
        p50: rank(50),
        p99: rank(99),
        max,
        hitches: gaps.iter().filter(|&&g| g > HITCH).count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_of_even_and_hitching_frames() {
        let ms = Duration::from_millis;
        let t0 = Instant::now();
        let mut p = Pacing::default();
        // 16 ms frames with one 50 ms and one 30 ms hitch, past a second.
        let mut at = t0;
        let mut out = None;
        assert_eq!(p.frame(at), None);
        for i in 0..70 {
            at += match i {
                10 => ms(50),
                20 => ms(30),
                _ => ms(16),
            };
            if let Some(s) = p.frame(at) {
                out.get_or_insert(s);
            }
        }
        let s = out.expect("a summary after a second");
        assert_eq!(s.p50, ms(16));
        assert_eq!(s.max, ms(50));
        assert_eq!(s.hitches, 2);
        // 1 s of frames is about 60 intervals: the 99th is the worst.
        assert!(s.frames >= 58 && s.frames <= 62, "{s:?}");
        assert_eq!(s.p99, ms(50));
        let line = s.to_string();
        assert!(
            line.contains("p50 16.0 p99 50.0 max 50.0, over 25 ms: 2"),
            "{line}"
        );
    }

    #[test]
    fn spread_ranks() {
        let ms = Duration::from_millis;
        assert_eq!(spread(Vec::new()), None);
        let one = spread(vec![ms(40)]).unwrap();
        assert_eq!(
            (one.p50, one.p99, one.max, one.hitches),
            (ms(40), ms(40), ms(40), 1)
        );
        // 1..=100 ms: the median is 50, the 99th 99; 75 are over 25 ms.
        let s = spread((1..=100).rev().map(ms).collect()).unwrap();
        assert_eq!(
            (s.p50, s.p99, s.max, s.hitches),
            (ms(50), ms(99), ms(100), 75)
        );
        // A clock that steps back counts as no time, not a panic.
        let t = Instant::now() + ms(100);
        let mut p = Pacing::default();
        p.frame(t);
        p.frame(t - ms(5));
        assert_eq!(p.gaps, [Duration::ZERO]);
    }
}
