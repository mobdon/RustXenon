//! Sound tests with synthetic input only: generated tones and hand-built
//! streams go through decode -> mixer -> .wav, and the .wav read back must
//! correlate with what went in.
//!
//! The FFmpeg tests skip when no LGPL FFmpeg can be loaded, unless
//! `XENON_REQUIRE_FFMPEG=1` (CI's FFmpeg job) turns the skip into a failure.

use xenon_audio::compare::correlation;
use xenon_audio::ffmpeg::{Decoder, Ffmpeg};
use xenon_audio::wav::{self, WavWriter};
use xenon_audio::xma;
use xenon_audio::{Mixer, FRAME_SAMPLES, GUEST_CHANNELS, SAMPLE_RATE};

fn ffmpeg() -> Option<Ffmpeg> {
    match Ffmpeg::load() {
        Ok(ff) => Some(ff),
        Err(e) if std::env::var_os("XENON_REQUIRE_FFMPEG").is_none() => {
            eprintln!("skipping FFmpeg test: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// 440 Hz left, 1 kHz right, interleaved stereo.
fn tone(frames: usize) -> Vec<f32> {
    (0..frames * FRAME_SAMPLES)
        .flat_map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let w = std::f32::consts::TAU * t;
            [0.5 * (440.0 * w).sin(), 0.3 * (1000.0 * w).sin()]
        })
        .collect()
}

/// Mixes everything queued into a .wav in a temp dir and reads it back.
fn mix_to_wav(m: &mut Mixer, frames: usize, name: &str) -> Vec<f32> {
    let path = std::env::temp_dir().join(format!("xenon-audio-{}-{name}.wav", std::process::id()));
    let mut w = WavWriter::create(&path, SAMPLE_RATE, 2).unwrap();
    for _ in 0..frames {
        m.mix_into(&mut w).unwrap();
    }
    w.finish().unwrap();
    let back = wav::read(&std::fs::read(&path).unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!((back.rate, back.channels), (SAMPLE_RATE, 2));
    back.samples
}

#[test]
fn guest_frames_through_mixer_to_wav() {
    // A render client submits 5.1 frames: the tone on front left/right,
    // silence elsewhere. Two clients, the second one quieter and inverted.
    let src = tone(40);
    let mut m = Mixer::new();
    let (a, b) = (m.add_source(), m.add_source());
    for f in src.chunks(FRAME_SAMPLES * 2) {
        let frame = |gain: f32| {
            let mut bytes = Vec::new();
            for c in 0..GUEST_CHANNELS {
                for i in 0..FRAME_SAMPLES {
                    let v = if c < 2 { f[i * 2 + c] * gain } else { 0.0 };
                    bytes.extend(v.to_be_bytes());
                }
            }
            bytes
        };
        m.push_guest_frame(a, &frame(1.0));
        m.push_guest_frame(b, &frame(-0.5));
    }
    let out = mix_to_wav(&mut m, 40, "guest");
    assert_eq!(out.len(), src.len());
    let (c, lag) = correlation(&src, &out, 8);
    assert_eq!(lag, 0);
    assert!(c > 0.9999, "correlation {c}");
    let peak = out.iter().fold(0f32, |p, v| p.max(v.abs()));
    assert!(
        (peak - 0.25).abs() < 0.01,
        "0.5 - 0.25 of the 0.5 tone, got {peak}"
    );
}

#[test]
fn ffmpeg_pcm_stream_through_mixer_to_wav() {
    let Some(ff) = ffmpeg() else { return };
    assert!(ff.license().starts_with("LGPL"));
    // A hand-built big-endian 16-bit stream of the tone, fed in odd-sized
    // packets the way a guest buffer would arrive.
    let src = tone(50);
    let stream: Vec<u8> = src
        .iter()
        .flat_map(|v| ((v * 32767.0).round() as i16).to_be_bytes())
        .collect();
    let mut d = Decoder::open(&ff, "pcm_s16be", SAMPLE_RATE, 2, &[]).unwrap();
    let mut pcm = Vec::new();
    for p in stream.chunks(4004) {
        d.decode(p, &mut pcm).unwrap();
    }
    d.finish(&mut pcm).unwrap();
    assert_eq!(pcm.len(), src.len());
    let mut m = Mixer::new();
    let s = m.add_source();
    m.push(s, &pcm);
    let out = mix_to_wav(&mut m, 50, "pcm");
    let (c, lag) = correlation(&src, &out, 8);
    assert_eq!(lag, 0);
    assert!(c > 0.9999, "correlation {c}");
}

#[test]
fn ffmpeg_xma2_hand_built_silence() {
    let Some(ff) = ffmpeg() else { return };
    let mut d = Decoder::xma2(&ff, SAMPLE_RATE, 1).unwrap();
    let mut pcm = Vec::new();
    for _ in 0..8 {
        d.decode(&xma::silent_packet(), &mut pcm).unwrap();
    }
    d.finish(&mut pcm).unwrap();
    // 8 frames of 512, less the decoder's 64-sample start delay, plus at
    // most one frame flushed at the end.
    eprintln!("xma2 silence: {} samples", pcm.len());
    assert!(
        pcm.len() >= 7 * 512 && pcm.len() <= 9 * 512,
        "{} samples",
        pcm.len()
    );
    assert!(pcm.iter().all(|&v| v == 0.0));
    // Junk must give an error or nothing, never a crash.
    let mut d = Decoder::xma2(&ff, SAMPLE_RATE, 2).unwrap();
    let junk: Vec<u8> = (0..xma::PACKET_BYTES as u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let _ = d.decode(&junk, &mut Vec::new());
}

#[test]
fn ffmpeg_channel_mismatch_is_an_error() {
    let Some(ff) = ffmpeg() else { return };
    // Mono stream data, opened as stereo: FFmpeg decodes one channel, and
    // reading a second plane would run off the frame.
    let mono = xenon_audio::xma::xma2_extradata(1);
    let mut d = Decoder::open(&ff, "xma2", SAMPLE_RATE, 2, &mono).unwrap();
    let mut pcm = Vec::new();
    let r = (0..8)
        .try_for_each(|_| d.decode(&xma::silent_packet(), &mut pcm))
        .and_then(|_| d.finish(&mut pcm));
    assert!(r.is_err() && pcm.is_empty(), "{r:?} {}", pcm.len());
    assert!(Decoder::open(&ff, "pcm_s16be", SAMPLE_RATE, 9, &[]).is_err());
}

#[test]
fn ffmpeg_unknown_decoder_is_an_error() {
    let Some(ff) = ffmpeg() else { return };
    assert!(Decoder::open(&ff, "no_such_codec", SAMPLE_RATE, 2, &[]).is_err());
}
