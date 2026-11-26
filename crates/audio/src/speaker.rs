//! Live output to the default audio device.
//!
//! [`Speaker::write`] (the mixing side) resamples the 48 kHz stereo mix to
//! the device rate and pushes it into a [`Ring`]; the device's output
//! callback pulls from that ring and spreads each stereo pair over the
//! device's channels ([`to_channels`]). The ring never blocks either side
//! for longer than a copy: when the callback asks for more than is queued
//! it plays silence (an underflow), and when the mix arrives faster than the
//! device plays it the ring cuts back once to its target level (an
//! overflow). Between those, [`Speaker::write`] steers the resampling rate
//! by up to [`MAX_STEER`] from how full the ring is, so a host clock that
//! runs a little fast or slow does not drift into either.
//!
//! Everything but [`Speaker::open`] is pure and tests anywhere. The device
//! side uses `cpal` on Windows only; elsewhere [`Speaker::open`] reports that
//! no output is available.

use crate::{Sink, SAMPLE_RATE};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Sound the ring may hold before it cuts back to its target, unless the
/// device pulls more than half of this at once.
pub const MAX_MS: usize = 100;
/// Sound the ring gathers before it plays, at start and after an underflow,
/// so the 5.33 ms mix ticks and the device's larger pulls do not collide.
/// It is also the level [`Speaker::write`] steers towards.
pub const START_MS: usize = 50;
/// Largest change to the resampling rate the fill level may make (0.5%).
pub const MAX_STEER: f64 = 0.005;
/// Rate change per unit of fill error ((fill - target) / target).
const STEER_GAIN: f64 = 0.01;

/// Interleaved stereo at the device rate, between the mixer and the device.
pub struct Ring {
    buf: VecDeque<f32>,
    cap: usize,
    start: usize,
    /// Largest pull seen, in samples.
    max_pull: usize,
    playing: bool,
    /// Whether anything was ever pushed: silence before the first push is
    /// not an underflow.
    started: bool,
    /// Stereo frames of silence played because the ring ran dry.
    pub underflow: u64,
    /// Stereo frames dropped because the ring was full.
    pub overflow: u64,
}

impl Ring {
    /// A ring for a device playing `rate` stereo frames a second.
    pub fn new(rate: u32) -> Self {
        let samples = |ms: usize| rate as usize * ms / 1000 * 2;
        Self {
            buf: VecDeque::with_capacity(samples(MAX_MS)),
            cap: samples(MAX_MS),
            start: samples(START_MS),
            max_pull: 0,
            playing: false,
            started: false,
            underflow: 0,
            overflow: 0,
        }
    }

    /// Stereo samples queued.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The level, in samples, to start playing at and to steer towards:
    /// [`START_MS`], or the largest pull if that is more.
    pub fn target(&self) -> usize {
        self.start.max(self.max_pull)
    }

    /// Queues interleaved stereo. Past the cap (or twice the largest pull)
    /// it drops the oldest down to [`Ring::target`] in one cut, so a burst
    /// costs one jump rather than a trim on every later push.
    pub fn push(&mut self, stereo: &[f32]) {
        self.started = true;
        self.buf.extend(stereo);
        if self.buf.len() > self.cap.max(2 * self.max_pull) {
            let over = (self.buf.len() - self.target()) & !1;
            self.buf.drain(..over);
            self.overflow += over as u64 / 2;
        }
    }

    /// Fills `out` (interleaved stereo) from the ring, silence for what is
    /// missing. It starts, and after running dry starts again, only once
    /// [`Ring::target`] is queued, so a whole pull is there to play.
    pub fn pop_into(&mut self, out: &mut [f32]) {
        self.max_pull = self.max_pull.max(out.len());
        if !self.playing && self.buf.len() >= self.target() {
            self.playing = true;
        }
        let n = if self.playing {
            self.buf.len().min(out.len())
        } else {
            0
        };
        for (o, s) in out.iter_mut().zip(self.buf.drain(..n)) {
            *o = s;
        }
        out[n..].fill(0.0);
        if n < out.len() {
            self.playing = false;
            if self.started {
                self.underflow += (out.len() - n) as u64 / 2;
            }
        }
    }
}

/// Linear resampling of interleaved stereo, kept continuous across calls.
pub struct Resampler {
    /// Input frames per output frame.
    step: f64,
    /// Position of the next output frame; 0 is `prev`, 1 the first frame of
    /// the next input.
    t: f64,
    prev: [f32; 2],
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Self {
            step: from as f64 / to as f64,
            t: 0.0,
            prev: [0.0; 2],
        }
    }

    /// Appends `input` resampled to `out`. Output runs one frame behind the
    /// input, also at equal rates, so the rate can be steered at any time.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let n = input.len() / 2;
        let at = |i: usize| match i {
            0 => self.prev,
            _ => [input[2 * i - 2], input[2 * i - 1]],
        };
        while self.t < n as f64 {
            let i = self.t as usize;
            let f = (self.t - i as f64) as f32;
            let (a, b) = (at(i), at(i + 1));
            out.push(a[0] + (b[0] - a[0]) * f);
            out.push(a[1] + (b[1] - a[1]) * f);
            self.t += self.step;
        }
        if n > 0 {
            self.prev = at(n);
        }
        self.t -= n as f64;
    }
}

/// Spreads interleaved stereo over `channels` outputs: one channel gets the
/// average, two or more get left and right first and silence after.
pub fn to_channels(stereo: &[f32], channels: usize, mut put: impl FnMut(f32)) {
    for f in stereo.chunks_exact(2) {
        if channels == 1 {
            put((f[0] + f[1]) * 0.5);
            continue;
        }
        put(f[0]);
        put(f[1]);
        for _ in 2..channels {
            put(0.0);
        }
    }
}

/// What the mixer and the device callback share.
struct Shared {
    ring: Mutex<Ring>,
    /// Set by the device's error callback: the stream has stopped.
    dead: AtomicBool,
    /// Stereo frames of silence played because the mixer held the ring.
    busy: AtomicU64,
}

impl Shared {
    #[cfg_attr(not(windows), allow(dead_code))] // only open and tests make one
    fn new(rate: u32) -> Arc<Self> {
        Arc::new(Self {
            ring: Mutex::new(Ring::new(rate)),
            dead: AtomicBool::new(false),
            busy: AtomicU64::new(0),
        })
    }
}

/// A [`Sink`] that plays on the default output device.
pub struct Speaker {
    shared: Arc<Shared>,
    resampler: Resampler,
    scratch: Vec<f32>,
    rate: u32,
    /// The unsteered resampling step.
    base: f64,
    /// Ring level after each write, smoothed, in samples.
    fill: f64,
}

fn lock(r: &Mutex<Ring>) -> std::sync::MutexGuard<'_, Ring> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}

impl Speaker {
    #[cfg_attr(not(windows), allow(dead_code))] // only open and tests make one
    fn new(shared: Arc<Shared>, rate: u32) -> Self {
        let resampler = Resampler::new(SAMPLE_RATE, rate);
        Self {
            shared,
            base: resampler.step,
            resampler,
            scratch: Vec::new(),
            rate,
            fill: 0.0,
        }
    }

    /// The device's rate in frames a second.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// (underflow, overflow) so far, in milliseconds of sound.
    pub fn counts_ms(&self) -> (u64, u64) {
        let (u, o) = {
            let r = lock(&self.shared.ring);
            (r.underflow, r.overflow)
        };
        let ms = |f: u64| f * 1000 / self.rate as u64;
        (ms(u + self.shared.busy.load(Ordering::Relaxed)), ms(o))
    }

    /// Opens the default output device and starts it playing silence.
    #[cfg(windows)]
    pub fn open() -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("speaker".into())
            .spawn(move || match device::start() {
                Ok((stream, rate, shared)) => {
                    let _ = tx.send(Ok((rate, shared.clone())));
                    // The stream plays while it lives, and it may not move
                    // to another thread, so this one keeps it until the
                    // device is lost.
                    let _stream = stream;
                    while !shared.dead.load(Ordering::Relaxed) {
                        std::thread::park_timeout(std::time::Duration::from_secs(1));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        let (rate, shared) = rx.recv().map_err(|e| e.to_string())??;
        Ok(Self::new(shared, rate))
    }

    /// No output device support on this platform.
    #[cfg(not(windows))]
    pub fn open() -> Result<Self, String> {
        Err("live sound is only built for Windows".into())
    }
}

impl Sink for Speaker {
    /// Fails once the device is lost; until then a full ring drops and
    /// counts instead.
    fn write(&mut self, stereo: &[f32]) -> std::io::Result<()> {
        if self.shared.dead.load(Ordering::Relaxed) {
            return Err(std::io::Error::other("the output device stopped"));
        }
        self.scratch.clear();
        self.resampler.process(stereo, &mut self.scratch);
        let (len, target) = {
            let mut r = lock(&self.shared.ring);
            r.push(&self.scratch);
            (r.len(), r.target())
        };
        // The level swings by a device pull between writes, so steer on an
        // average rather than on each reading.
        self.fill += (len as f64 - self.fill) / 32.0;
        let err = (self.fill - target as f64) / target as f64;
        self.resampler.step = self.base * (1.0 + (STEER_GAIN * err).clamp(-MAX_STEER, MAX_STEER));
        Ok(())
    }
}

#[cfg(windows)]
mod device {
    use super::{to_channels, Shared, MAX_MS};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, TryLockError};

    type Started = (cpal::Stream, u32, Arc<Shared>);

    pub fn start() -> Result<Started, String> {
        let dev = cpal::default_host()
            .default_output_device()
            .ok_or("no default output device")?;
        let supported = dev.default_output_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let rate = config.sample_rate.0;
        let shared = Shared::new(rate);
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&dev, &config, shared.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&dev, &config, shared.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&dev, &config, shared.clone()),
            f => return Err(format!("device sample format {f} is not handled")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok((stream, rate, shared))
    }

    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        dev: &cpal::Device,
        config: &cpal::StreamConfig,
        shared: Arc<Shared>,
    ) -> Result<cpal::Stream, String> {
        let channels = config.channels as usize;
        // Allocated here, never in the callback: pulls go in chunks of at
        // most this many frames.
        let frames = (config.sample_rate.0 as usize * MAX_MS / 1000).max(1);
        let mut stereo = Vec::with_capacity(frames * 2);
        let on_error = shared.clone();
        dev.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                for part in data.chunks_mut(frames * channels) {
                    stereo.clear();
                    stereo.resize(part.len() / channels * 2, 0.0);
                    // Never wait on the mixer from the device thread: if it
                    // holds the ring (it is preempted mid-copy), this pull
                    // plays silence and is counted, rather than the device
                    // missing its period behind a lower-priority thread.
                    let ring = match shared.ring.try_lock() {
                        Ok(r) => Some(r),
                        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
                        Err(TryLockError::WouldBlock) => None,
                    };
                    match ring {
                        Some(mut r) => r.pop_into(&mut stereo),
                        None => {
                            let n = stereo.len() as u64 / 2;
                            shared.busy.fetch_add(n, Ordering::Relaxed);
                        }
                    }
                    let mut out = part.iter_mut();
                    to_channels(&stereo, channels, |v| {
                        if let Some(o) = out.next() {
                            *o = T::from_sample(v);
                        }
                    });
                }
            },
            move |e| {
                eprintln!("audio: speaker: {e}");
                on_error.dead.store(true, Ordering::Relaxed);
            },
            None,
        )
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_wraps_in_order() {
        // 1 kHz: cap 200 samples, start at 100.
        let mut r = Ring::new(1000);
        let mut out = [0.0; 60];
        for round in 0..20 {
            let v: Vec<f32> = (0..60).map(|i| (round * 60 + i) as f32).collect();
            r.push(&v);
            if round == 0 {
                r.push(&[0.0; 60]); // reach the start level once
                r.pop_into(&mut out);
                r.pop_into(&mut out);
                continue;
            }
            r.pop_into(&mut out);
            assert_eq!(out[0], (round * 60) as f32, "round {round}");
            assert!(out.windows(2).all(|w| w[1] == w[0] + 1.0));
        }
        assert_eq!((r.underflow, r.overflow), (0, 0));
    }

    #[test]
    fn ring_overflow_cuts_once_to_target() {
        let mut r = Ring::new(1000);
        let v: Vec<f32> = (0..250).map(|i| i as f32).collect();
        r.push(&v);
        assert_eq!((r.len(), r.overflow), (100, 75));
        let mut out = [0.0; 4];
        r.pop_into(&mut out);
        assert_eq!(out, [150.0, 151.0, 152.0, 153.0]);
        // Below the cap again, later pushes are not trimmed.
        r.push(&[0.0; 90]);
        assert_eq!((r.len(), r.overflow), (186, 75));
    }

    #[test]
    fn ring_underflow_plays_silence_and_refills() {
        let mut r = Ring::new(1000);
        let mut out = [1.0; 20];
        r.pop_into(&mut out);
        assert_eq!(r.underflow, 0, "silence before any push is not counted");
        assert!(out.iter().all(|&s| s == 0.0));
        r.push(&[0.5; 120]);
        r.pop_into(&mut out[..10]);
        assert!(out[..10].iter().all(|&s| s == 0.5));
        let mut big = [1.0; 120];
        r.pop_into(&mut big);
        assert_eq!(r.underflow, 5);
        assert!(big[110..].iter().all(|&s| s == 0.0));
        // Below the start level it waits rather than play a sliver.
        r.push(&[0.5; 40]);
        r.pop_into(&mut out);
        assert!(out.iter().all(|&s| s == 0.0));
        assert_eq!(r.underflow, 15);
    }

    #[test]
    fn ring_waits_for_a_whole_pull() {
        // Start level 100 samples, but the device pulls 120 at a time.
        let mut r = Ring::new(1000);
        let mut out = [1.0; 120];
        r.pop_into(&mut out);
        r.push(&[0.5; 110]);
        r.pop_into(&mut out);
        assert!(out.iter().all(|&s| s == 0.0), "110 queued is not a pull");
        assert_eq!(r.len(), 110);
        r.push(&[0.5; 10]);
        r.pop_into(&mut out);
        assert!(out.iter().all(|&s| s == 0.5));
        // From here, a pull's worth a pull plays without a gap.
        let u = r.underflow;
        for _ in 0..10 {
            r.push(&[0.5; 60]);
            r.push(&[0.5; 60]);
            r.pop_into(&mut out);
        }
        assert_eq!((r.underflow, r.overflow), (u, 0));
    }

    #[test]
    fn ring_serves_pulls_larger_than_the_cap() {
        // Cap 200 samples; the device pulls 240.
        let mut r = Ring::new(1000);
        let mut out = [1.0; 240];
        let mut counts = Vec::new();
        for _ in 0..10 {
            r.push(&[0.5; 120]);
            r.push(&[0.5; 120]);
            r.pop_into(&mut out);
            counts.push((r.underflow, r.overflow));
        }
        assert!(out.iter().all(|&s| s == 0.5));
        assert!(counts[1..].iter().all(|&c| c == counts[1]), "{counts:?}");
    }

    #[test]
    fn resample_48k_to_44k1_length() {
        let mut rs = Resampler::new(48_000, 44_100);
        let mut out = Vec::new();
        let chunk = [0.25f32; 512];
        // Three seconds of input.
        for _ in 0..3 * 48_000 / 256 {
            rs.process(&chunk, &mut out);
        }
        let frames = out.len() / 2;
        assert!(frames.abs_diff(562 * 256 * 441 / 480) <= 1, "{frames}");
    }

    #[test]
    fn resample_is_continuous_across_chunks() {
        let n = 4096;
        let ramp: Vec<f32> = (0..n).flat_map(|i| [i as f32, -(i as f32)]).collect();
        let mut whole = Vec::new();
        Resampler::new(48_000, 44_100).process(&ramp, &mut whole);
        let mut chunked = Vec::new();
        let mut rs = Resampler::new(48_000, 44_100);
        for c in ramp.chunks(2 * 256) {
            rs.process(c, &mut chunked);
        }
        assert_eq!(whole.len(), chunked.len());
        let step = 48_000.0 / 44_100.0;
        for (j, (a, b)) in whole.chunks(2).zip(chunked.chunks(2)).enumerate() {
            assert!((a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3);
            if j >= 1 {
                // A ramp stays a ramp: x(t) = t - 1 past the first frame.
                let want = (j as f64 * step - 1.0) as f32;
                assert!((b[0] - want).abs() < 1e-2, "{j}: {} vs {want}", b[0]);
                assert_eq!(b[1], -b[0]);
            }
        }
    }

    #[test]
    fn resample_same_rate_keeps_samples_one_frame_late() {
        let mut rs = Resampler::new(48_000, 48_000);
        let mut out = Vec::new();
        rs.process(&[0.1, 0.2, 0.3, 0.4], &mut out);
        rs.process(&[0.5, 0.6], &mut out);
        assert_eq!(out, [0.0, 0.0, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn channel_mapping() {
        let s = [0.5, -0.5, 1.0, 0.0];
        let map = |c| {
            let mut v = Vec::new();
            to_channels(&s, c, |x| v.push(x));
            v
        };
        assert_eq!(map(1), [0.0, 0.5]);
        assert_eq!(map(2), s);
        assert_eq!(
            map(6),
            [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn speaker_resamples_into_ring() {
        let shared = Shared::new(44_100);
        let mut sp = Speaker::new(shared.clone(), 44_100);
        for _ in 0..10 {
            sp.write(&[0.1; 512]).unwrap();
        }
        let frames = lock(&shared.ring).len() / 2;
        // Within the steering range of 2560 frames at 44.1 kHz.
        let want = 2560 * 441 / 480;
        assert!(frames.abs_diff(want) <= want / 100, "{frames}");
        assert_eq!(sp.counts_ms(), (0, 0));
    }

    #[test]
    fn speaker_steers_rate_from_fill() {
        let shared = Shared::new(48_000);
        let mut sp = Speaker::new(shared.clone(), 48_000);
        // Nearly empty: play slower than the mix arrives, to fill up.
        sp.write(&[0.1; 512]).unwrap();
        assert!((sp.resampler.step - (1.0 - MAX_STEER)).abs() < 1e-9);
        // Nobody pulls, so the ring stays above its target: speed up.
        for _ in 0..500 {
            sp.write(&[0.1; 512]).unwrap();
        }
        let step = sp.resampler.step;
        assert!(step > 1.0 && step <= 1.0 + MAX_STEER + 1e-9, "{step}");
        // Drained so each write lands on the target, the rate settles back
        // to the device's own.
        for _ in 0..2000 {
            sp.write(&[0.1; 512]).unwrap();
            let mut r = lock(&shared.ring);
            let keep = r.target() - 512;
            let extra = r.len().saturating_sub(keep) & !1;
            r.buf.drain(..extra);
        }
        assert!(
            (sp.resampler.step - 1.0).abs() < 1e-4,
            "{}",
            sp.resampler.step
        );
    }

    #[test]
    fn speaker_fails_once_device_is_lost() {
        let shared = Shared::new(48_000);
        let mut sp = Speaker::new(shared.clone(), 48_000);
        sp.write(&[0.1; 512]).unwrap();
        shared.dead.store(true, Ordering::Relaxed);
        assert!(sp.write(&[0.1; 512]).is_err());
    }
}
