//! Audio calls, headless: a render client is registered but never asked
//! for frames, and nothing is played.

use super::*;

pub fn x_audio_register_render_driver_client(rt: &Runtime, c: &mut Context) {
    let out = arg(rt, c, 1);
    if out != 0 {
        rt.mem.write_u32(out, 0x4155_0000);
    }
    ret(c, STATUS_SUCCESS);
}
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
