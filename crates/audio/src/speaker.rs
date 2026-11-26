//! Live output to the default audio device.
//!
//! [`Speaker::write`] (the mixing side) resamples the 48 kHz stereo mix to
//! the device rate and pushes it into a [`Ring`]; the device's output
//! callback pulls from that ring and spreads each stereo pair over the
//! device's channels ([`to_channels`]). The ring never blocks either side
//! for longer than a copy: when the callback asks for more than is queued
//! it plays silence (an underflow), and when the mix arrives faster than the
//! device plays it the oldest samples go (an overflow), so latency stays
//! under [`MAX_MS`].
//!
//! Everything but [`Speaker::open`] is pure and tests anywhere. The device
//! side uses `cpal` on Windows only; elsewhere [`Speaker::open`] reports that
//! no output is available.

use crate::{Sink, SAMPLE_RATE};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Most sound the ring holds; older samples are dropped past it.
pub const MAX_MS: usize = 100;
/// Sound the ring gathers before it plays, at start and after an underflow,
/// so the 5.33 ms mix ticks and the device's larger pulls do not collide.
pub const START_MS: usize = 50;

/// Interleaved stereo at the device rate, between the mixer and the device.
pub struct Ring {
    buf: VecDeque<f32>,
    cap: usize,
    start: usize,
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

    /// Queues interleaved stereo, dropping the oldest past the cap.
    pub fn push(&mut self, stereo: &[f32]) {
        self.started = true;
        self.buf.extend(stereo);
        let over = self.buf.len().saturating_sub(self.cap) & !1;
        self.buf.drain(..over);
        self.overflow += over as u64 / 2;
    }

    /// Fills `out` (interleaved stereo) from the ring, silence for what is
    /// missing. After running dry it stays silent until [`START_MS`] is
    /// queued again.
    pub fn pop_into(&mut self, out: &mut [f32]) {
        if !self.playing && self.buf.len() >= self.start {
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

    /// Appends `input` resampled to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.step == 1.0 {
            out.extend_from_slice(input);
            return;
        }
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

/// A [`Sink`] that plays on the default output device.
pub struct Speaker {
    ring: Arc<Mutex<Ring>>,
    resampler: Resampler,
    scratch: Vec<f32>,
    rate: u32,
}

fn lock(r: &Mutex<Ring>) -> std::sync::MutexGuard<'_, Ring> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}

impl Speaker {
    #[cfg_attr(not(windows), allow(dead_code))] // only open and tests make one
    fn new(ring: Arc<Mutex<Ring>>, rate: u32) -> Self {
        Self {
            ring,
            resampler: Resampler::new(SAMPLE_RATE, rate),
            scratch: Vec::new(),
            rate,
        }
    }

    /// The device's rate in frames a second.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// (underflow, overflow) so far, in milliseconds of sound.
    pub fn counts_ms(&self) -> (u64, u64) {
        let r = lock(&self.ring);
        let ms = |f: u64| f * 1000 / self.rate as u64;
        (ms(r.underflow), ms(r.overflow))
    }

    /// Opens the default output device and starts it playing silence.
    #[cfg(windows)]
    pub fn open() -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("speaker".into())
            .spawn(move || match device::start() {
                Ok((stream, rate, ring)) => {
                    let _ = tx.send(Ok((rate, ring)));
                    // The stream plays while it lives, and it may not move
                    // to another thread, so this one keeps it.
                    let _stream = stream;
                    loop {
                        std::thread::park();
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            })
            .map_err(|e| e.to_string())?;
        let (rate, ring) = rx.recv().map_err(|e| e.to_string())??;
        Ok(Self::new(ring, rate))
    }

    /// No output device support on this platform.
    #[cfg(not(windows))]
    pub fn open() -> Result<Self, String> {
        Err("live sound is only built for Windows".into())
    }
}

impl Sink for Speaker {
    fn write(&mut self, stereo: &[f32]) -> std::io::Result<()> {
        self.scratch.clear();
        self.resampler.process(stereo, &mut self.scratch);
        lock(&self.ring).push(&self.scratch);
        Ok(())
    }
}

#[cfg(windows)]
mod device {
    use super::{lock, to_channels, Ring};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};

    type Started = (cpal::Stream, u32, Arc<Mutex<Ring>>);

    pub fn start() -> Result<Started, String> {
        let dev = cpal::default_host()
            .default_output_device()
            .ok_or("no default output device")?;
        let supported = dev.default_output_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let rate = config.sample_rate.0;
        let ring = Arc::new(Mutex::new(Ring::new(rate)));
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&dev, &config, ring.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&dev, &config, ring.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&dev, &config, ring.clone()),
            f => return Err(format!("device sample format {f} is not handled")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok((stream, rate, ring))
    }

    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        dev: &cpal::Device,
        config: &cpal::StreamConfig,
        ring: Arc<Mutex<Ring>>,
    ) -> Result<cpal::Stream, String> {
        let channels = config.channels as usize;
        let mut stereo = Vec::new();
        dev.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                stereo.resize(data.len() / channels * 2, 0.0);
                lock(&ring).pop_into(&mut stereo);
                let mut out = data.iter_mut();
                to_channels(&stereo, channels, |v| {
                    if let Some(o) = out.next() {
                        *o = T::from_sample(v);
                    }
                });
            },
            |e| eprintln!("audio: speaker: {e}"),
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
    fn ring_overflow_drops_oldest() {
        let mut r = Ring::new(1000);
        let v: Vec<f32> = (0..250).map(|i| i as f32).collect();
        r.push(&v);
        assert_eq!((r.len(), r.overflow), (200, 25));
        let mut out = [0.0; 4];
        r.pop_into(&mut out);
        assert_eq!(out, [50.0, 51.0, 52.0, 53.0]);
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
    fn resample_same_rate_passes_through() {
        let mut out = Vec::new();
        Resampler::new(48_000, 48_000).process(&[0.1, 0.2, 0.3, 0.4], &mut out);
        assert_eq!(out, [0.1, 0.2, 0.3, 0.4]);
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
        let ring = Arc::new(Mutex::new(Ring::new(44_100)));
        let mut sp = Speaker::new(ring.clone(), 44_100);
        for _ in 0..10 {
            sp.write(&[0.1; 512]).unwrap();
        }
        let frames = lock(&ring).len() / 2;
        assert!(frames.abs_diff(2560 * 441 / 480) <= 1, "{frames}");
        assert_eq!(sp.counts_ms(), (0, 0));
    }
}
