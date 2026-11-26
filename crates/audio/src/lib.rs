//! `xenon-audio`: headless sound for the host.
//!
//! - [`Mixer`] takes the frames render clients submit and sums them into
//!   one stereo stream, one 256-sample frame at a time (the console's audio
//!   tick: 48 kHz, 256 samples, every 5.33 ms).
//! - [`wav::WavWriter`] is a [`Sink`] that writes that stream to a .wav
//!   file.
//! - [`speaker::Speaker`] is a [`Sink`] that plays it on the default
//!   output device (Windows only, through `cpal`).
//! - [`ffmpeg`] decodes the console's compressed audio (XMA) through an
//!   LGPL FFmpeg that is loaded at run time, never linked in.
//! - [`xma`] holds the pure XMA packet facts the decoder path needs.
//! - [`compare::correlation`] checks a written .wav against the original.
//!
//! Everything except [`ffmpeg`] and the device side of [`speaker`] is pure
//! and tests anywhere; [`ffmpeg`] only needs the shared libraries, which
//! tests skip when absent.

pub mod compare;
pub mod ffmpeg;
pub mod speaker;
pub mod wav;
pub mod xma;

use std::collections::VecDeque;

/// Output rate of the console's audio hardware.
pub const SAMPLE_RATE: u32 = 48_000;
/// Samples per channel in one render-client frame.
pub const FRAME_SAMPLES: usize = 256;
/// Channels in a frame a render client submits (5.1: FL, FR, C, LFE, BL, BR).
pub const GUEST_CHANNELS: usize = 6;
/// Bytes in one submitted frame: 6 channels one after another, 256
/// big-endian floats each.
pub const GUEST_FRAME_BYTES: usize = GUEST_CHANNELS * FRAME_SAMPLES * 4;

/// Where mixed stereo goes: interleaved left/right floats in -1..=1.
pub trait Sink {
    fn write(&mut self, stereo: &[f32]) -> std::io::Result<()>;
}

/// Turns one submitted 5.1 frame (see [`GUEST_FRAME_BYTES`]) into 256
/// interleaved stereo samples. Centre and back channels fold in at -3 dB,
/// the low-frequency channel is dropped.
pub fn guest_frame_to_stereo(frame: &[u8]) -> Vec<f32> {
    assert_eq!(
        frame.len(),
        GUEST_FRAME_BYTES,
        "a submitted frame is 6 x 256 floats"
    );
    let ch = |c: usize, i: usize| {
        let o = (c * FRAME_SAMPLES + i) * 4;
        f32::from_be_bytes(frame[o..o + 4].try_into().unwrap())
    };
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut out = Vec::with_capacity(FRAME_SAMPLES * 2);
    for i in 0..FRAME_SAMPLES {
        let (c, bl, br) = (ch(2, i), ch(4, i), ch(5, i));
        out.push(ch(0, i) + H * (c + bl));
        out.push(ch(1, i) + H * (c + br));
    }
    out
}

/// Sums the queued stereo of every source, a frame at a time. The host
/// adds a source per registered render client (or per decoded voice),
/// pushes what it submits, and pulls [`Mixer::mix_frame`] on each tick.
#[derive(Default)]
pub struct Mixer {
    sources: Vec<VecDeque<f32>>,
    /// Samples thrown away because a source held [`MAX_QUEUED`].
    pub dropped: u64,
}

/// Stereo samples a source holds at most (64 frames, 341 ms, 128 KB); older
/// ones are dropped when a source is pushed faster than it is mixed.
pub const MAX_QUEUED: usize = 64 * FRAME_SAMPLES * 2;

impl Mixer {
    pub fn new() -> Self {
        Self::default()
    }

    /// A new, empty source; the returned id is its index.
    pub fn add_source(&mut self) -> usize {
        self.sources.push(VecDeque::new());
        self.sources.len() - 1
    }

    /// Queues interleaved stereo samples on a source.
    pub fn push(&mut self, src: usize, stereo: &[f32]) {
        let q = &mut self.sources[src];
        q.extend(stereo);
        let over = q.len().saturating_sub(MAX_QUEUED);
        q.drain(..over);
        self.dropped += over as u64;
    }

    /// Queues one frame as a render client submits it.
    pub fn push_guest_frame(&mut self, src: usize, frame: &[u8]) {
        self.push(src, &guest_frame_to_stereo(frame));
    }

    /// Stereo samples still queued on a source.
    pub fn queued(&self, src: usize) -> usize {
        self.sources[src].len()
    }

    /// Stereo samples queued on all sources.
    pub fn queued_total(&self) -> usize {
        self.sources.iter().map(|q| q.len()).sum()
    }

    /// One frame of mixed stereo (256 left/right pairs). A source with too
    /// little queued plays silence for the missing part; the sum is clamped.
    pub fn mix_frame(&mut self) -> Vec<f32> {
        let mut out = vec![0.0f32; FRAME_SAMPLES * 2];
        for q in &mut self.sources {
            for (o, s) in out
                .iter_mut()
                .zip(q.drain(..q.len().min(FRAME_SAMPLES * 2)))
            {
                *o += s;
            }
        }
        out.iter_mut().for_each(|s| *s = s.clamp(-1.0, 1.0));
        out
    }

    /// Mixes one frame and writes it to `sink`.
    pub fn mix_into(&mut self, sink: &mut impl Sink) -> std::io::Result<()> {
        sink.write(&self.mix_frame())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(f: impl Fn(usize, usize) -> f32) -> Vec<u8> {
        let mut b = Vec::with_capacity(GUEST_FRAME_BYTES);
        for c in 0..GUEST_CHANNELS {
            for i in 0..FRAME_SAMPLES {
                b.extend(f(c, i).to_be_bytes());
            }
        }
        b
    }

    #[test]
    fn guest_frame_downmix() {
        // FL 0.5, FR -0.25, C 0.1, LFE 1 (dropped), BL 0.2, BR 0
        let v = [0.5, -0.25, 0.1, 1.0, 0.2, 0.0];
        let s = guest_frame_to_stereo(&frame(|c, _| v[c]));
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(s.len(), 512);
        assert!((s[0] - (0.5 + h * 0.3)).abs() < 1e-6);
        assert!((s[1] - (-0.25 + h * 0.1)).abs() < 1e-6);
    }

    #[test]
    fn mixer_sums_pads_and_clamps() {
        let mut m = Mixer::new();
        let (a, b) = (m.add_source(), m.add_source());
        m.push(a, &[0.25; 600]);
        m.push(b, &[0.9; 100]);
        let f = m.mix_frame();
        assert_eq!(f.len(), 512);
        assert_eq!(f[0], 1.0); // 1.15 clamped
        assert_eq!(f[100], 0.25); // b ran out
        assert_eq!(m.queued(a), 88);
        let f = m.mix_frame();
        assert_eq!((f[87], f[88]), (0.25, 0.0));
        assert_eq!(m.queued(a), 0);
    }

    #[test]
    fn mixer_caps_each_source() {
        let mut m = Mixer::new();
        let a = m.add_source();
        m.push(a, &vec![0.1; MAX_QUEUED + 10]);
        assert_eq!((m.queued(a), m.dropped), (MAX_QUEUED, 10));
    }
}
