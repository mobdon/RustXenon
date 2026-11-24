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
pub fn xma_create_context(rt: &Runtime, c: &mut Context) {
    let a = k().pool_alloc(rt, 64);
    rt.mem.write_u32(arg(rt, c, 0), a);
    ret(c, STATUS_SUCCESS);
}
pub fn xma_release_context(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
