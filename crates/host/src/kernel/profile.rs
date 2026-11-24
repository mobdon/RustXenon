//! One local profile, signed in on controller port 0: sign-in state and
//! information, its user id and name, privileges, and profile settings.
//! Settings the title writes are kept in the data folder (`profile/`) and
//! read back on later runs.
//!
//! A setting id packs its type in bits 28..31 (1 int32, 2 int64, 3 double,
//! 4 string, 5 float, 6 binary, 7 date) and its size in bits 16..27. A
//! setting record is 40 bytes: source (0 unset, 1 profile, 2 title), user
//! index or id at +8, setting id at +0x10, then the value: type byte, and
//! at +8 the number, or size and pointer for strings and binary data.

use super::xam::complete;
use super::*;

/// An offline profile id (the 0xE0 prefix marks a local profile).
pub const XUID: u64 = 0xE000_0000_C0DE_0001;
pub const NAME: &str = "Player";
/// Signed in locally (not to the online service).
const SIGNED_IN_LOCALLY: u32 = 1;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const ERROR_INVALID_PARAMETER: u32 = 87;

/// Defaults a console gives a new profile, for the integer settings that
/// are not zero (control options, colours, region).
const DEFAULTS: [(u32, u32); 7] = [
    (0x1004_0003, 3),
    (0x1004_0006, 0xFA),
    (0x1004_000E, 0x64),
    (0x1004_0012, 1),
    (0x1004_001D, 0xFFFF_0000),
    (0x1004_001E, 0xFF00_FF00),
    (0x1004_0022, 1),
];

fn signed_in(user: u32) -> bool {
    user == 0
}

pub fn xam_user_get_signin_state(rt: &Runtime, c: &mut Context) {
    let s = if signed_in(arg(rt, c, 0)) {
        SIGNED_IN_LOCALLY
    } else {
        0
    };
    ret(c, s);
}

pub fn xam_user_get_xuid(rt: &Runtime, c: &mut Context) {
    let (user, out) = (arg(rt, c, 0), arg(rt, c, 2));
    if out == 0 {
        return ret(c, 0x8007_0057); // E_INVALIDARG
    }
    let ok = signed_in(user);
    rt.mem.write_u64(out, if ok { XUID } else { 0 });
    ret(c, if ok { 0 } else { 0x8007_0525 }); // HRESULT for "no such user"
}

pub fn xam_user_get_name(rt: &Runtime, c: &mut Context) {
    let (user, buf, len) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    if !signed_in(user) {
        return ret(c, ERROR_NO_SUCH_USER);
    }
    let n = (len.min(16) as usize).saturating_sub(1).min(NAME.len());
    write_cstr(rt, buf, &NAME[..n]);
    ret(c, ERROR_SUCCESS);
}

/// Sign-in information: id, flags, state, two reserved words, name (16).
pub fn xam_user_get_signin_info(rt: &Runtime, c: &mut Context) {
    let (user, p) = (arg(rt, c, 0), arg(rt, c, 2));
    if p == 0 {
        return ret(c, 0x8007_0057);
    }
    fill(rt, p, 0x28, 0);
    if !signed_in(user) {
        return ret(c, 0x8007_0525);
    }
    rt.mem.write_u64(p, XUID);
    rt.mem.write_u32(p + 0xC, SIGNED_IN_LOCALLY);
    write_cstr(rt, p + 0x18, NAME);
    ret(c, 0);
}

/// Online privileges (multiplayer, voice, content from others) are not
/// granted to a local profile.
pub fn xam_user_check_privilege(rt: &Runtime, c: &mut Context) {
    let (user, out) = (arg(rt, c, 0), arg(rt, c, 2));
    if !signed_in(user) {
        return ret(c, ERROR_NO_SUCH_USER);
    }
    if out != 0 {
        rt.mem.write_u32(out, 0);
    }
    ret(c, ERROR_SUCCESS);
}

fn stored(id: u32) -> std::path::PathBuf {
    k().data_dir.join("profile").join(format!("{id:08x}.bin"))
}

fn finish(rt: &Runtime, c: &mut Context, ov: u32, result: u32) {
    if ov != 0 {
        complete(rt, c, ov, result, 0);
        ret(c, ERROR_IO_PENDING);
    } else {
        ret(c, result);
    }
}

/// (title id, user, id count, ids, setting count, setting ids, buffer size
/// pointer, buffer, request block). The buffer gets a count and a pointer
/// to the records, the records, then string and binary data.
pub fn xam_user_read_profile_settings(rt: &Runtime, c: &mut Context) {
    let (user, xuids_n, n, ids, size_p, buf, ov) = (
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
        arg(rt, c, 7),
        arg(rt, c, 8),
    );
    if !(1..=32).contains(&n) || size_p == 0 {
        return ret(c, ERROR_INVALID_PARAMETER);
    }
    let ids: Vec<u32> = (0..n).map(|i| rt.mem.read_u32(ids + 4 * i)).collect();
    let copies = xuids_n.max(1);
    let data: u32 = ids
        .iter()
        .filter(|&&id| matches!(id >> 28, 4 | 6))
        .map(|&id| id >> 16 & 0xFFF)
        .sum();
    let need = 8 + copies * (40 * n + data);
    let have = rt.mem.read_u32(size_p);
    if buf == 0 || have < need {
        rt.mem.write_u32(size_p, need);
        return ret(c, ERROR_INSUFFICIENT_BUFFER);
    }
    if xuids_n == 0 && !signed_in(user) {
        return finish(rt, c, ov, ERROR_NO_SUCH_USER);
    }
    fill(rt, buf, need, 0);
    rt.mem.write_u32(buf, n * copies);
    rt.mem.write_u32(buf + 4, buf + 8);
    let mut rec = buf + 8;
    let mut extra = buf + 8 + copies * 40 * n;
    for _ in 0..copies {
        for &id in &ids {
            if xuids_n > 0 {
                rt.mem.write_u64(rec + 8, XUID);
            } else {
                rt.mem.write_u32(rec + 8, user);
            }
            rt.mem.write_u32(rec + 0x10, id);
            let ty = id >> 28;
            let v = rec + 0x18;
            rt.mem.write_u8(v, ty as u8);
            match ty {
                // Strings and binary data: the title's own, if it wrote them.
                4 | 6 => {
                    if let Ok(b) = std::fs::read(stored(id)) {
                        let b = &b[..b.len().min((id >> 16 & 0xFFF) as usize)];
                        rt.mem.write_bytes(extra, b);
                        rt.mem.write_u32(rec, 2);
                        rt.mem.write_u32(v + 8, b.len() as u32);
                        rt.mem.write_u32(v + 12, extra);
                    }
                    extra += id >> 16 & 0xFFF;
                }
                _ => {
                    let d = DEFAULTS.iter().find(|x| x.0 == id).map_or(0, |x| x.1);
                    rt.mem.write_u32(rec, 1);
                    match ty {
                        2 | 3 | 7 => rt.mem.write_u64(v + 8, d as u64),
                        _ => rt.mem.write_u32(v + 8, d),
                    }
                }
            }
            rec += 40;
        }
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}

/// (title id, user, count, records, request block). Strings and binary
/// settings are stored; numbers keep their defaults.
pub fn xam_user_write_profile_settings(rt: &Runtime, c: &mut Context) {
    let (user, n, recs, ov) = (arg(rt, c, 1), arg(rt, c, 2), arg(rt, c, 3), arg(rt, c, 4));
    if n == 0 || recs == 0 {
        return ret(c, ERROR_INVALID_PARAMETER);
    }
    if !signed_in(user) {
        return finish(rt, c, ov, ERROR_NO_SUCH_USER);
    }
    for i in 0..n {
        let r = recs + 40 * i;
        let id = rt.mem.read_u32(r + 0x10);
        if matches!(id >> 28, 4 | 6) {
            let (len, p) = (rt.mem.read_u32(r + 0x20), rt.mem.read_u32(r + 0x24));
            let mut b = vec![0; len.min(0x1000) as usize];
            rt.mem.read_bytes(p, &mut b);
            let path = stored(id);
            let _ = std::fs::create_dir_all(path.parent().unwrap());
            if let Err(e) = std::fs::write(&path, &b) {
                crate::log(&format!("[profile] cannot store setting {id:#x}: {e}"));
            }
        }
    }
    finish(rt, c, ov, ERROR_SUCCESS);
}
