//! Audio calls, headless: render clients are asked for frames at the
//! console's rate, and nothing is played.

use super::*;

/// (callback record, driver out). The record holds the callback and its
/// argument. The callback asks the title for the next 256-sample frame;
/// it runs on a thread of its own at the console's rate (48 kHz / 256, one
/// call every 5.33 ms) and the frames it submits are dropped.
pub fn x_audio_register_render_driver_client(rt: &Runtime, c: &mut Context) {
    let (rec, out) = (arg(rt, c, 0), arg(rt, c, 1));
    let (f, user) = (rt.mem.read_u32(rec), rt.mem.read_u32(rec + 4));
    let n = CLIENTS.fetch_add(1, Ordering::Relaxed);
    if out != 0 {
        rt.mem.write_u32(out, 0x4155_0000 | n);
    }
    if f != 0 {
        let rt = crate::runtime();
        std::thread::Builder::new()
            .name(format!("audio-{n}"))
            .spawn(move || {
                let (base, _) = super::threads::new_context(&rt, 64 << 10);
                let period = std::time::Duration::from_nanos(5_333_333);
                let mut next = std::time::Instant::now();
                loop {
                    next += period;
                    std::thread::sleep(next.saturating_duration_since(std::time::Instant::now()));
                    let mut c = base.clone();
                    c.r[3] = user as u64;
                    rt.call(&mut c, f);
                }
            })
            .expect("cannot start a host thread");
    }
    ret(c, STATUS_SUCCESS);
}

static CLIENTS: AtomicU32 = AtomicU32::new(0);

pub fn x_audio_unregister_render_driver_client(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
pub fn x_audio_submit_render_driver_frame(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
pub fn x_audio_get_voice_category_volume_change_mask(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 1), 0);
    ret(c, STATUS_SUCCESS);
}
pub fn x_audio_get_voice_category_volume(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 1), 1f32.to_bits());
    ret(c, STATUS_SUCCESS);
}
/// Stereo speakers.
pub fn x_audio_get_speaker_config(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 0), 0x0001_0001);
    ret(c, STATUS_SUCCESS);
}

// ---- audio decoder contexts ------------------------------------------------

/// Decoder contexts: 64-byte records in physical memory that the title
/// fills (input buffers, output ring) and the decoder hardware drains.
const CONTEXTS: u32 = 320;
static XMA: Mutex<Option<(u32, Vec<bool>)>> = Mutex::new(None);

/// Context words (each a big-endian word; bit numbers from the low end):
/// word 0: input buffer 0 valid (bit 20) and 1 valid (21), output block
/// count (22..27) and write offset (27..32); word 1: output valid (31);
/// word 2: input read offset (0..26); word 4: current input buffer (31);
/// word 7: output buffer; word 9: output read offset (0..5).
///
/// shortcut: no audio is decoded. Each round consumes the current input
/// buffer at once and fills the output ring with silence, so streams move
/// on as if played; upgrade when the decoder lands (step 6).
fn drain(rt: &Runtime, ctx: u32) {
    let w0 = rt.mem.read_u32(ctx);
    let w1 = rt.mem.read_u32(ctx + 4);
    if std::env::var_os("XENON_TRACE_XMA").is_some() {
        static LAST: Mutex<Vec<(u32, [u32; 10])>> = Mutex::new(Vec::new());
        let words: [u32; 10] = std::array::from_fn(|i| rt.mem.read_u32(ctx + 4 * i as u32));
        let mut l = LAST.lock().unwrap();
        let prev = l.iter_mut().find(|x| x.0 == ctx);
        let changed = prev.as_ref().map_or(true, |p| p.1 != words);
        if changed {
            crate::log(&format!(
                "[xma] {ctx:#x}: {}",
                words
                    .iter()
                    .map(|w| format!("{w:08x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
            match prev {
                Some(p) => p.1 = words,
                None => l.push((ctx, words)),
            }
        }
    }
    if w1 >> 31 == 0 {
        return;
    }
    let mut w0n = w0;
    let w4 = rt.mem.read_u32(ctx + 16);
    let cur = w4 >> 31;
    let consumed = w0 >> (20 + cur) & 1 != 0;
    if consumed {
        w0n &= !(1 << (20 + cur));
        rt.mem.write_u32(ctx + 16, w4 ^ 1 << 31);
        rt.mem
            .write_u32(ctx + 8, rt.mem.read_u32(ctx + 8) & !0x03FF_FFFF);
    }
    // Output only comes from input: with no input left the stream ends,
    // and the title sees its output ring stop filling.
    let blocks = w0 >> 22 & 31;
    if consumed && blocks > 0 {
        let read = rt.mem.read_u32(ctx + 36) & 31;
        let full = (read + blocks - 1) % blocks;
        let out = super::gpu::gpu_to_cpu(rt.mem.read_u32(ctx + 28));
        let mut wr = w0 >> 27 & 31;
        while wr != full {
            fill(rt, out + 256 * wr, 256, 0);
            wr = (wr + 1) % blocks;
        }
        w0n = (w0n & !(31 << 27)) | wr << 27;
    }
    if w0n != w0 {
        rt.mem.write_u32(ctx, w0n);
    }
}

fn start_decoder(rt: std::sync::Arc<Runtime>, base: u32) {
    std::thread::Builder::new()
        .name("audio-decoder".into())
        .spawn(move || loop {
            let used: Vec<u32> = match &*XMA.lock().unwrap() {
                Some((_, u)) => u
                    .iter()
                    .enumerate()
                    .filter(|x| *x.1)
                    .map(|x| base + 64 * x.0 as u32)
                    .collect(),
                None => vec![],
            };
            for c in used {
                drain(&rt, c);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        })
        .expect("cannot start a host thread");
}

/// (context out): a free context, zeroed.
pub fn xma_create_context(rt: &Runtime, c: &mut Context) {
    let out = arg(rt, c, 0);
    let mut x = XMA.lock().unwrap();
    if x.is_none() {
        let base = super::memory::physical(rt, CONTEXTS * 64, 256);
        *x = Some((base, vec![false; CONTEXTS as usize]));
        start_decoder(crate::runtime(), base);
    }
    let (base, used) = x.as_mut().unwrap();
    let Some(i) = used.iter().position(|u| !u) else {
        return ret(c, STATUS_NO_MEMORY);
    };
    used[i] = true;
    let a = *base + 64 * i as u32;
    fill(rt, a, 64, 0);
    rt.mem.write_u32(out, a);
    ret(c, STATUS_SUCCESS);
}

pub fn xma_release_context(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    if let Some((base, used)) = XMA.lock().unwrap().as_mut() {
        if let Some(u) = a
            .checked_sub(*base)
            .map(|o| o / 64)
            .and_then(|i| used.get_mut(i as usize))
        {
            *u = false;
        }
    }
    ret(c, STATUS_SUCCESS);
}
