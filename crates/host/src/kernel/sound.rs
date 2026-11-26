//! Sound out of the host. Decoder contexts are decoded through an LGPL
//! FFmpeg loaded at run time (silence when none is found), and the frames
//! render clients submit are mixed to stereo, written to `--audio-wav` and
//! played on the default output device (with `--window` or `--sound`).
//! Pure parts live in `xenon-audio`.

use super::*;
use std::io::BufWriter;
use xenon_audio::ffmpeg::{Decoder, Ffmpeg};
use xenon_audio::speaker::Speaker;
use xenon_audio::wav::WavWriter;
use xenon_audio::xma;
use xenon_audio::{Mixer, Sink, GUEST_FRAME_BYTES, SAMPLE_RATE};

fn ffmpeg() -> Option<&'static Ffmpeg> {
    static FF: OnceLock<Option<Ffmpeg>> = OnceLock::new();
    FF.get_or_init(|| match Ffmpeg::load() {
        Ok(f) => {
            let v = f.version();
            crate::log(&format!(
                "audio: FFmpeg avcodec {}.{}.{} ({}) decodes XMA",
                v.0,
                v.1,
                v.2,
                f.license()
            ));
            Some(f)
        }
        Err(e) => {
            crate::log(&format!("audio: {e}; decoder contexts play silence"));
            None
        }
    })
    .as_ref()
}

/// One context's decoder and what it decoded but could not write yet.
#[derive(Default)]
struct Ctx {
    dec: Option<Decoder<'static>>,
    format: (u32, u16),
    stream: xma::Stream,
    /// The read offset this module last wrote; another value means the
    /// title moved it (a new sound or a seek).
    expect: Option<u32>,
}

static CTXS: Mutex<Option<HashMap<u32, Ctx>>> = Mutex::new(None);
static ERRORS: AtomicU32 = AtomicU32::new(0);
static DECODED: AtomicU64 = AtomicU64::new(0);

/// Forgets a context's decoder (created or released).
pub fn forget(ctx: u32) {
    if let Some(m) = CTXS.lock().unwrap().as_mut() {
        m.remove(&ctx);
    }
}

/// Logs the first few decoder errors.
fn error(ctx: u32, e: &dyn std::fmt::Display) {
    if ERRORS.fetch_add(1, Ordering::Relaxed) < 5 {
        crate::log(&format!("audio: context {ctx:#x}: {e}"));
    }
}

/// One decoder round for the context at `a`: feeds its input packets and
/// fills its output ring as far as the title has read it. Returns false
/// when no FFmpeg is loaded (the caller writes silence instead).
pub fn decode(rt: &Runtime, a: u32) -> bool {
    let Some(ff) = ffmpeg() else {
        return false;
    };
    let c = xma::Context(std::array::from_fn(|i| rt.mem.read_u32(a + 4 * i as u32)));
    let mut all = CTXS.lock().unwrap();
    let s = all.get_or_insert_with(HashMap::new).entry(a).or_default();
    let off = c.read_offset();
    let format = (c.rate(), c.channels());
    if s.expect != Some(off) || s.format != format {
        *s = Ctx {
            format,
            ..Ctx::default()
        };
    }
    let blocks = c.blocks();
    let (mut space, mut wr) = (c.out_space(), c.write_offset());
    let out = super::gpu::gpu_to_cpu(c.output_ptr());
    let (mut cur, mut packet, mut consumed, mut moved) = (c.current(), off >> 14, 0u32, false);
    let mut buf = vec![0u8; xma::PACKET_BYTES];
    while space > 0 {
        if let Some(b) = s.stream.block() {
            rt.mem.write_bytes(out + 256 * wr, &b);
            wr = (wr + 1) % blocks;
            space -= 1;
            continue;
        }
        if !c.input_valid(cur) || consumed >> cur & 1 != 0 {
            // Input ran dry. FFmpeg holds about 8 frames back, so flush
            // them now or the end of a sound never plays.
            // shortcut: a stream refilled late restarts its decoder and
            // loses a frame that spans the gap; keep the decoder across
            // short gaps if that is heard.
            if let Some(mut d) = s.dec.take() {
                let mut o = Vec::new();
                if let Err(e) = d.finish(&mut o) {
                    error(a, &e);
                }
                DECODED.fetch_add(o.len() as u64, Ordering::Relaxed);
                s.stream.push(&o);
                s.stream.pad();
                continue;
            }
            break;
        }
        moved = true;
        if packet >= c.packets(cur) {
            consumed |= 1 << cur;
            cur ^= 1;
            packet = 0;
            continue;
        }
        let src = super::gpu::gpu_to_cpu(c.input_ptr(cur)) + (xma::PACKET_BYTES as u32) * packet;
        rt.mem.read_bytes(src, &mut buf);
        if s.dec.is_none() {
            s.dec = Decoder::xma2(ff, format.0, format.1)
                .map_err(|e| error(a, &e))
                .ok();
        }
        let dec = s.dec.as_mut();
        s.stream.feed(&buf, |p, o| {
            if let Some(d) = dec {
                if let Err(e) = d.decode(p, o) {
                    error(a, &e);
                }
            }
            DECODED.fetch_add(o.len() as u64, Ordering::Relaxed);
        });
        packet += 1;
    }
    // Write back only the fields this round changed: the title sets the
    // other buffer's valid bit and pointer while the decoder runs.
    let new_off = if moved { packet << 14 } else { off };
    s.expect = Some(new_off);
    if new_off != off {
        let w2 = rt.mem.read_u32(a + 8);
        rt.mem.write_u32(a + 8, w2 & !0x03FF_FFFF | new_off);
    }
    if cur != c.current() {
        let w4 = rt.mem.read_u32(a + 16);
        rt.mem.write_u32(a + 16, w4 & !(1 << 31) | cur << 31);
    }
    if consumed != 0 || wr != c.write_offset() {
        let w0 = rt.mem.read_u32(a);
        rt.mem
            .write_u32(a, (w0 & !(consumed << 20) & !(31 << 27)) | wr << 27);
    }
    true
}

#[derive(Default)]
struct Out {
    mixer: Mixer,
    /// Mixer source of each render client, by driver handle.
    sources: HashMap<u32, usize>,
    wav: Option<WavWriter<BufWriter<std::fs::File>>>,
    /// Never blocks the tick: it only copies into the device's ring.
    speaker: Option<Speaker>,
    frames: u64,
}

static OUT: Mutex<Option<Out>> = Mutex::new(None);

/// Starts writing the mix to `path` (`--audio-wav`).
pub fn start_wav(path: &std::path::Path) -> std::io::Result<()> {
    let wav = WavWriter::create(path, SAMPLE_RATE, 2)?;
    OUT.lock().unwrap().get_or_insert_with(Out::default).wav = Some(wav);
    Ok(())
}

/// Starts playing the mix on the default output device. Without one, or
/// once it is lost, the speaker thread keeps trying the default device and
/// [`tick`] logs each change; only a platform without live sound gives up.
pub fn start_speaker() {
    match Speaker::open() {
        Ok(s) => OUT.lock().unwrap().get_or_insert_with(Out::default).speaker = Some(s),
        Err(e) => crate::log(&format!("audio: no speaker ({e}); sound is not played")),
    }
}

/// Stops playing on the speaker (the run has stopped). try_lock: this
/// runs in the panic hook, maybe on a thread that holds the lock.
pub fn silence() {
    if let Ok(mut o) = OUT.try_lock() {
        if let Some(o) = o.as_mut() {
            o.speaker = None;
        }
    }
}

/// (driver, samples): one 5.1 frame of 256 big-endian floats per channel.
pub fn submit(rt: &Runtime, driver: u32, samples: u32) {
    let mut o = OUT.lock().unwrap();
    let Some(o) = o.as_mut() else {
        return;
    };
    if o.sources.is_empty() {
        crate::log(&format!(
            "audio: first frame from {driver:#x} at {samples:#010x}"
        ));
    }
    let mut frame = vec![0u8; GUEST_FRAME_BYTES];
    rt.mem.read_bytes(samples, &mut frame);
    let src = *o
        .sources
        .entry(driver)
        .or_insert_with(|| o.mixer.add_source());
    o.mixer.push_guest_frame(src, &frame);
}

/// One audio tick (5.33 ms): mixes a frame and writes it.
pub fn tick() {
    let mut o = OUT.lock().unwrap();
    let Some(o) = o.as_mut() else {
        return;
    };
    let mix = o.mixer.mix_frame();
    if let Some(w) = o.wav.as_mut() {
        if let Err(e) = w.write(&mix) {
            crate::log(&format!("audio: wav stopped: {e}"));
            if let Some(w) = o.wav.take() {
                let _ = w.finish();
            }
        }
    }
    if let Some(s) = o.speaker.as_mut() {
        if let Some(line) = s.poll() {
            crate::log(&format!("audio: {line}"));
        }
        // Never fails: with no device open the mix is dropped.
        let _ = s.write(&mix);
    }
    o.frames += 1;
}

/// Mixes what is still queued, then finishes the .wav so its header holds
/// the right length.
pub fn finish() {
    while OUT
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|o| o.wav.is_some() && o.mixer.queued_total() > 0)
    {
        tick();
    }
    if let Some(o) = OUT.lock().unwrap().as_mut() {
        if let Some(w) = o.wav.take() {
            let secs = o.frames as f64 * 256.0 / SAMPLE_RATE as f64;
            match w.finish() {
                Ok(_) => crate::log(&format!("audio: wrote {secs:.1} s of sound")),
                Err(e) => crate::log(&format!("audio: wav not finished: {e}")),
            }
        }
    }
}

/// Sizes for the once-a-second line: decoded bytes waiting on contexts
/// (cap 64 KB each), mixer samples queued (cap 128 KB a client), seconds
/// written and samples dropped at either cap; with a speaker, the ms of
/// silence it played (underflow) and of sound it dropped (overflow).
pub fn line() -> String {
    let (n, q, d) = CTXS.lock().unwrap().as_ref().map_or((0, 0, 0), |m| {
        m.values().fold((0, 0, 0), |(n, q, d), c| {
            (n + 1, q + c.stream.queued(), d + c.stream.dropped)
        })
    });
    let (mq, md, secs, sp) = OUT.lock().unwrap().as_ref().map_or((0, 0, 0, None), |o| {
        (
            o.mixer.queued_total() * 4,
            o.mixer.dropped,
            o.frames * 256 / SAMPLE_RATE as u64,
            o.speaker.as_ref().and_then(Speaker::counts_ms),
        )
    });
    let sp = sp.map_or(String::new(), |(u, o)| {
        format!(", speaker underflow {u} ms overflow {o} ms")
    });
    format!(
        "audio {n} ctx {} KB, decoded {}k samples, mix {} KB, wav {secs} s, dropped {}{sp}",
        q >> 10,
        DECODED.load(Ordering::Relaxed) / 1000,
        mq >> 10,
        d + md
    )
}
