//! System library calls: controllers (none connected), users (nobody
//! signed in), content packages on the local data folder, notifications,
//! tasks, system interface screens (closed at once) and voice (no headset).

use super::*;
use std::path::PathBuf;

/// Completes an asynchronous request: result, length, extended error, and
/// its event.
pub fn complete(rt: &Runtime, c: &Context, ov: u32, result: u32, length: u32) {
    if ov == 0 {
        return;
    }
    rt.mem.write_u32(ov, result);
    rt.mem.write_u32(ov + 4, length);
    rt.mem.write_u32(ov + 0x18, result);
    let ev = rt.mem.read_u32(ov + 0xC);
    if ev != 0 {
        if let Some(a) = super::objects::object_of(rt, c, ev) {
            if let super::objects::KObj::Event(e) = &*super::objects::get(rt, a) {
                e.set();
            }
        }
    }
}

/// With a request block the call completes it and answers "pending";
/// without one it answers the result directly.
fn finish(rt: &Runtime, c: &mut Context, ov: u32, result: u32) {
    if ov != 0 {
        complete(rt, c, ov, result, 0);
        ret(c, ERROR_IO_PENDING);
    } else {
        ret(c, result);
    }
}

// ---- input -----------------------------------------------------------------

pub fn xam_input_get_capabilities(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_input_get_state(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_input_set_state(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_input_get_keystroke_ex(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_enable_inactivity_processing(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn xam_reset_inactivity(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

// ---- users -----------------------------------------------------------------

pub fn xam_user_get_signin_state(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn xam_user_get_xuid(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_NO_SUCH_USER);
}
pub fn xam_user_get_name(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_NO_SUCH_USER);
}
pub fn xam_user_get_signin_info(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 2);
    if p != 0 {
        fill(rt, p, 0x28, 0);
    }
    ret(c, ERROR_NO_SUCH_USER);
}
pub fn xam_user_check_privilege(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_NO_SUCH_USER);
}
pub fn xam_user_are_users_friends(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 4);
    finish(rt, c, ov, ERROR_NOT_LOGGED_ON);
}
pub fn xam_user_get_membership_tier_from_xuid(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 2);
    finish(rt, c, ov, ERROR_NO_SUCH_USER);
}
pub fn xam_user_get_online_country_from_xuid(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 2);
    finish(rt, c, ov, ERROR_NO_SUCH_USER);
}
pub fn xam_user_read_profile_settings(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 8);
    finish(rt, c, ov, ERROR_NOT_LOGGED_ON);
}
pub fn xam_user_write_profile_settings(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 4);
    finish(rt, c, ov, ERROR_NOT_LOGGED_ON);
}
pub fn xam_user_create_achievement_enumerator(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_NO_SUCH_USER);
}
pub fn xam_user_create_stats_enumerator(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_NO_SUCH_USER);
}

// ---- enumerators -------------------------------------------------------------

pub fn xam_create_enumerator_handle(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_FUNCTION_FAILED);
}
pub fn xam_get_private_enum_structure_from_handle(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_INVALID_HANDLE);
}
/// Every enumerator is empty.
pub fn xam_enumerate(rt: &Runtime, c: &mut Context) {
    let (items, ov) = (arg(rt, c, 4), arg(rt, c, 5));
    if items != 0 {
        rt.mem.write_u32(items, 0);
    }
    finish(rt, c, ov, ERROR_NO_MORE_FILES);
}

// ---- content -----------------------------------------------------------------

/// Content data: device id, content type, display name (128 UTF-16
/// characters), file name (42 bytes) at +0x108.
fn content_dir(rt: &Runtime, data: u32) -> PathBuf {
    let mut name = cstr(rt, data + 0x108);
    name.retain(|ch| ch.is_ascii_alphanumeric() || "._-".contains(ch));
    if name.is_empty() {
        name = "content".into();
    }
    k().data_dir.join("content").join(name)
}

pub fn xam_content_create_ex(rt: &Runtime, c: &mut Context) {
    let (root, data, disp, lic, ov) = (
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 8),
    );
    let dir = content_dir(rt, data);
    let existed = dir.exists();
    super::files::link_root(&cstr(rt, root), dir);
    if disp != 0 {
        rt.mem.write_u32(disp, if existed { 2 } else { 1 });
    }
    if lic != 0 {
        rt.mem.write_u32(lic, 0);
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}

pub fn xam_content_close(rt: &Runtime, c: &mut Context) {
    let (root, ov) = (arg(rt, c, 0), arg(rt, c, 1));
    super::files::unlink_root(&cstr(rt, root));
    finish(rt, c, ov, ERROR_SUCCESS);
}

pub fn xam_content_create_enumerator(rt: &Runtime, c: &mut Context) {
    let (size, out) = (arg(rt, c, 5), arg(rt, c, 6));
    if size != 0 {
        rt.mem.write_u32(size, 0x134);
    }
    let h = k().add_handle(Handle::Other(0));
    rt.mem.write_u32(out, h);
    ret(c, ERROR_SUCCESS);
}

/// One hard disk: id, type, total and free bytes, name.
pub fn xam_content_get_device_data(rt: &Runtime, c: &mut Context) {
    let (id, p) = (arg(rt, c, 0), arg(rt, c, 1));
    if id != 1 {
        return ret(c, ERROR_DEVICE_NOT_CONNECTED);
    }
    fill(rt, p, 0x50, 0);
    rt.mem.write_u32(p, 1);
    rt.mem.write_u32(p + 4, 1);
    rt.mem.write_u64(p + 8, 20 << 30);
    rt.mem.write_u64(p + 16, 10 << 30);
    for (i, ch) in "Hard Drive".encode_utf16().enumerate() {
        rt.mem.write_u16(p + 24 + 2 * i as u32, ch);
    }
    ret(c, ERROR_SUCCESS);
}

pub fn xam_content_set_thumbnail(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 4);
    finish(rt, c, ov, ERROR_SUCCESS);
}

pub fn xam_content_get_creator(rt: &Runtime, c: &mut Context) {
    let (is, xuid, ov) = (arg(rt, c, 2), arg(rt, c, 3), arg(rt, c, 4));
    if is != 0 {
        rt.mem.write_u32(is, 1);
    }
    if xuid != 0 {
        rt.mem.write_u64(xuid, 0);
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}

// ---- system ----------------------------------------------------------------

pub fn xam_get_execution_id(rt: &Runtime, c: &mut Context) {
    let p = super::rtl::header_field(rt, k().xex_header, 0x0004_0006);
    rt.mem.write_u32(arg(rt, c, 0), p);
    ret(
        c,
        if p != 0 {
            STATUS_SUCCESS
        } else {
            STATUS_UNSUCCESSFUL
        },
    );
}

pub fn xam_get_system_version(_: &Runtime, c: &mut Context) {
    ret(c, 0x2000_4497);
}

pub fn xam_alloc(rt: &Runtime, c: &mut Context) {
    let (size, out) = (arg(rt, c, 1), arg(rt, c, 2));
    let a = k().pool_alloc(rt, size);
    rt.mem.write_u32(out, a);
    ret(c, ERROR_SUCCESS);
}

pub fn xam_free(rt: &Runtime, c: &mut Context) {
    k().pool.lock().unwrap().free(arg(rt, c, 0));
    ret(c, ERROR_SUCCESS);
}

pub fn x_get_av_pack(_: &Runtime, c: &mut Context) {
    ret(c, 8); // HDMI
}
pub fn x_get_game_region(_: &Runtime, c: &mut Context) {
    ret(c, 0x00FF);
}
pub fn x_get_language(_: &Runtime, c: &mut Context) {
    ret(c, 1); // English
}
pub fn x_get_video_mode(rt: &Runtime, c: &mut Context) {
    super::video::vd_query_video_mode(rt, c);
}

pub fn xam_loader_launch_title(rt: &Runtime, c: &mut Context) {
    panic!(
        "guest asked to launch another title ({})",
        cstr(rt, arg(rt, c, 0))
    );
}
pub fn xam_loader_terminate_title(_: &Runtime, _: &mut Context) {
    panic!("guest asked to end the title");
}

/// Runs the task callback on a new thread with the message as argument.
pub fn xam_task_schedule(rt: &Runtime, c: &mut Context) {
    let (f, msg, out) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 3));
    let (h, _) = super::threads::create(rt, 0, f, msg, 0, false);
    if out != 0 {
        rt.mem.write_u32(out, h);
    }
    ret(c, STATUS_SUCCESS);
}
pub fn xam_task_close_handle(rt: &Runtime, c: &mut Context) {
    k().close(arg(rt, c, 0));
    ret(c, STATUS_SUCCESS);
}
pub fn xam_task_should_exit(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

// ---- messages and notifications ----------------------------------------------

pub fn x_msg_in_process_call(rt: &Runtime, c: &mut Context) {
    crate::log(&format!(
        "[xam] message {:#x} to app {:#x} refused",
        arg(rt, c, 1),
        arg(rt, c, 0)
    ));
    ret(c, 0x8007_0490); // ERROR_NOT_FOUND as HRESULT
}
pub fn x_msg_start_io_request(rt: &Runtime, c: &mut Context) {
    crate::log(&format!(
        "[xam] request {:#x} to app {:#x} refused",
        arg(rt, c, 1),
        arg(rt, c, 0)
    ));
    let ov = arg(rt, c, 2);
    finish(rt, c, ov, ERROR_FUNCTION_FAILED);
}
pub fn x_msg_start_io_request_ex(rt: &Runtime, c: &mut Context) {
    x_msg_start_io_request(rt, c);
}
pub fn x_msg_cancel_io_request(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

pub fn xam_notify_create_listener(_: &Runtime, c: &mut Context) {
    let h = k().add_handle(Handle::Other(1));
    ret(c, h);
}
/// No notifications.
pub fn x_notify_get_next(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn x_notify_position_ui(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

// ---- system screens: closed at once ------------------------------------------

pub fn xam_show_signin_ui(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_SUCCESS);
}
pub fn xam_show_player_review_ui(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_SUCCESS);
}
pub fn xam_show_marketplace_ui(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_SUCCESS);
}
pub fn xam_show_marketplace_download_items_ui(rt: &Runtime, c: &mut Context) {
    let ov = arg(rt, c, 5);
    finish(rt, c, ov, ERROR_SUCCESS);
}
pub fn xam_show_gamer_card_ui_for_xuid(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_SUCCESS);
}
pub fn xam_show_dirty_disc_error_ui(_: &Runtime, _: &mut Context) {
    panic!("guest reported a disc read error");
}
/// Picks the first button.
pub fn xam_show_message_box_ui_ex(rt: &Runtime, c: &mut Context) {
    let (result, ov) = (arg(rt, c, 7), arg(rt, c, 8));
    if result != 0 {
        rt.mem.write_u32(result, 0);
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}
/// Picks the hard disk (device 1).
pub fn xam_show_device_selector_ui(rt: &Runtime, c: &mut Context) {
    let (dev, ov) = (arg(rt, c, 4), arg(rt, c, 5));
    if dev != 0 {
        rt.mem.write_u32(dev, 1);
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}

// ---- voice and sessions -------------------------------------------------------

pub fn xam_voice_create(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_voice_headset_present(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn xam_voice_submit_packet(_: &Runtime, c: &mut Context) {
    ret(c, ERROR_DEVICE_NOT_CONNECTED);
}
pub fn xam_voice_close(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn xam_session_create_handle(rt: &Runtime, c: &mut Context) {
    let h = k().add_handle(Handle::Other(2));
    rt.mem.write_u32(arg(rt, c, 0), h);
    ret(c, ERROR_SUCCESS);
}
pub fn xam_session_ref_obj_by_handle(rt: &Runtime, c: &mut Context) {
    let a = k().pool_alloc(rt, 0x40);
    rt.mem.write_u32(arg(rt, c, 1), a);
    ret(c, ERROR_SUCCESS);
}
