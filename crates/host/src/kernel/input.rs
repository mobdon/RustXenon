//! A scripted virtual controller on port 0; ports 1 to 3 stay unplugged.
//!
//! Headless runs need a way past "press start": the script presses buttons
//! at fixed times since start. The default presses Start, then A a second
//! and a half later, every four seconds from the fifth second. `--press`
//! replaces it with a list of `<ms>:<button>[+<button>...]` presses (each
//! held 150 ms), and `--no-controller` unplugs the pad. `--window` plugs in
//! the window's live pad instead: an XInput controller and the keyboard.

use super::*;
use std::sync::{Arc, OnceLock};
use xenon_input::Pad;

pub use xenon_input::BUTTONS;

/// Virtual key codes for keystroke events, in the order of [`BUTTONS`].
const KEYS: [u16; 14] = [
    0x5810, 0x5811, 0x5812, 0x5813, 0x5814, 0x5815, 0x5816, 0x5817, 0x5805, 0x5804, 0x5800, 0x5801,
    0x5802, 0x5803,
];

const HOLD_MS: u64 = 150;
/// Script-only bits for the triggers (`lt`, `rt`), which XInput reports as
/// bytes, not buttons: two button bits XInput leaves unused.
const SCRIPT_LT: u16 = 0x0400;
const SCRIPT_RT: u16 = 0x0800;
const ERROR_EMPTY: u32 = 0x10D2;

pub enum Script {
    Unplugged,
    /// (start ms, buttons), sorted.
    Presses(Vec<(u64, u16)>),
    /// Start, then A, every four seconds from five seconds.
    Default,
    /// The window's controller and keyboard.
    Live,
}

/// The game window, when `--window` opened one.
pub static WINDOW: OnceLock<Arc<xenon_window::Link>> = OnceLock::new();

static SCRIPT: OnceLock<Script> = OnceLock::new();
/// Buttons last reported as keystrokes.
static LAST_KEYS: AtomicU32 = AtomicU32::new(0);
/// The last state reported and its packet number.
static LAST_STATE: Mutex<(Pad, u32)> = Mutex::new((
    Pad {
        buttons: 0,
        lt: 0,
        rt: 0,
        lx: 0,
        ly: 0,
        rx: 0,
        ry: 0,
    },
    0,
));

/// Parses `--press` text: `5000:start,6500:a+b`. `lt` and `rt` pull a
/// trigger fully.
pub fn parse(text: &str) -> Result<Script, String> {
    let mut v = Vec::new();
    for item in text.split(',').filter(|s| !s.trim().is_empty()) {
        let (t, b) = item
            .split_once(':')
            .ok_or_else(|| format!("press {item:?}: need <ms>:<button>"))?;
        let t: u64 = t
            .trim()
            .parse()
            .map_err(|_| format!("press {item:?}: bad time"))?;
        let mut bits = 0;
        for name in b.split('+') {
            let name = name.trim().to_ascii_lowercase();
            bits |= match name.as_str() {
                "lt" => SCRIPT_LT,
                "rt" => SCRIPT_RT,
                _ => {
                    BUTTONS
                        .iter()
                        .find(|x| x.0 == name)
                        .ok_or_else(|| format!("unknown button {name:?}"))?
                        .1
                }
            };
        }
        v.push((t, bits));
    }
    v.sort();
    Ok(Script::Presses(v))
}

pub fn set_script(s: Script) {
    let _ = SCRIPT.set(s);
}

fn script() -> &'static Script {
    SCRIPT.get_or_init(|| Script::Default)
}

/// Buttons held at `ms` since start, or None when unplugged.
pub fn buttons_at(s: &Script, ms: u64) -> Option<u16> {
    match s {
        Script::Unplugged => None,
        Script::Live => Some(WINDOW.get().map_or(0, |w| w.pad().buttons)),
        Script::Presses(v) => Some(
            v.iter()
                .filter(|(t, _)| ms >= *t && ms < t + HOLD_MS)
                .fold(0, |a, x| a | x.1),
        ),
        Script::Default => {
            if ms < 5000 {
                return Some(0);
            }
            let p = (ms - 5000) % 4000;
            Some(if p < HOLD_MS {
                0x0010
            } else if (1500..1500 + HOLD_MS).contains(&p) {
                0x1000
            } else {
                0
            })
        }
    }
}

fn now() -> Option<Pad> {
    let s = script();
    if let (Script::Live, Some(w)) = (s, WINDOW.get()) {
        return Some(w.pad());
    }
    buttons_at(s, crate::ms() as u64).map(scripted_pad)
}

/// The pad a script's bits stand for: its buttons, and a trigger fully
/// pulled for each trigger bit.
fn scripted_pad(bits: u16) -> Pad {
    let pulled = |bit| if bits & bit != 0 { 255 } else { 0 };
    Pad {
        buttons: bits & !(SCRIPT_LT | SCRIPT_RT),
        lt: pulled(SCRIPT_LT),
        rt: pulled(SCRIPT_RT),
        ..Pad::default()
    }
}

fn connected(user: u32) -> bool {
    (user == 0 || user == 0xFF) && now().is_some()
}

/// Capabilities: type gamepad, subtype gamepad, flags, then the gamepad's
/// supported buttons, triggers and sticks, then vibration motors.
pub fn xam_input_get_capabilities(rt: &Runtime, c: &mut Context) {
    let (user, caps) = (arg(rt, c, 0), arg(rt, c, 2));
    if !connected(user) {
        return ret(c, ERROR_DEVICE_NOT_CONNECTED);
    }
    fill(rt, caps, 20, 0);
    rt.mem.write_u8(caps, 1);
    rt.mem.write_u8(caps + 1, 1);
    rt.mem.write_u16(caps + 4, 0xFFFF);
    rt.mem.write_u8(caps + 6, 0xFF);
    rt.mem.write_u8(caps + 7, 0xFF);
    for i in 0..4 {
        rt.mem.write_u16(caps + 8 + 2 * i, 0xFFC0);
    }
    rt.mem.write_u16(caps + 16, 0xFFFF);
    rt.mem.write_u16(caps + 18, 0xFFFF);
    ret(c, ERROR_SUCCESS);
}

/// State: packet number, buttons, two triggers, four stick axes.
pub fn xam_input_get_state(rt: &Runtime, c: &mut Context) {
    let (user, state) = (arg(rt, c, 0), arg(rt, c, 2));
    let Some(p) = now().filter(|_| connected(user)) else {
        return ret(c, ERROR_DEVICE_NOT_CONNECTED);
    };
    rt.mem.write_u32(state, packet(p));
    rt.mem.write_u16(state + 4, p.buttons);
    rt.mem.write_u8(state + 6, p.lt);
    rt.mem.write_u8(state + 7, p.rt);
    for (i, v) in [p.lx, p.ly, p.rx, p.ry].into_iter().enumerate() {
        rt.mem.write_u16(state + 8 + 2 * i as u32, v as u16);
    }
    ret(c, ERROR_SUCCESS);
}

/// The packet number changes exactly when the state does.
fn packet(p: Pad) -> u32 {
    let mut last = LAST_STATE.lock().unwrap();
    if last.0 != p {
        *last = (p, last.1.wrapping_add(1));
    }
    last.1
}

pub fn xam_input_set_state(rt: &Runtime, c: &mut Context) {
    let s = if connected(arg(rt, c, 0)) {
        ERROR_SUCCESS
    } else {
        ERROR_DEVICE_NOT_CONNECTED
    };
    ret(c, s);
}

/// One keystroke per call for each button that went down or up since the
/// last keystroke reported: virtual key, character, flags (1 down, 2 up),
/// user, HID code.
pub fn xam_input_get_keystroke_ex(rt: &Runtime, c: &mut Context) {
    let (user_p, ks) = (arg(rt, c, 0), arg(rt, c, 2));
    let user = if user_p != 0 {
        rt.mem.read_u32(user_p)
    } else {
        0xFF
    };
    let Some(b) = now().filter(|_| connected(user)).map(|p| p.buttons) else {
        return ret(c, ERROR_DEVICE_NOT_CONNECTED);
    };
    let last = LAST_KEYS.load(Ordering::Relaxed) as u16;
    let changed = b ^ last;
    let Some(i) = BUTTONS.iter().position(|x| changed & x.1 != 0) else {
        return ret(c, ERROR_EMPTY);
    };
    let bit = BUTTONS[i].1;
    let down = b & bit != 0;
    LAST_KEYS.store((last ^ bit) as u32, Ordering::Relaxed);
    fill(rt, ks, 8, 0);
    rt.mem.write_u16(ks, KEYS[i]);
    rt.mem.write_u16(ks + 4, if down { 1 } else { 2 });
    if user_p != 0 {
        rt.mem.write_u32(user_p, 0);
    }
    ret(c, ERROR_SUCCESS);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts() {
        let s = parse("1000:start, 2000:a+b").unwrap();
        assert_eq!(buttons_at(&s, 999), Some(0));
        assert_eq!(buttons_at(&s, 1000), Some(0x10));
        assert_eq!(buttons_at(&s, 1149), Some(0x10));
        assert_eq!(buttons_at(&s, 1150), Some(0));
        assert_eq!(buttons_at(&s, 2010), Some(0x3000));
        assert!(parse("10:jump").is_err());
        // Triggers are bytes in the pad, not button bits.
        let t = parse("0:rt+a,500:lt").unwrap();
        let pad = scripted_pad(buttons_at(&t, 10).unwrap());
        assert_eq!((pad.buttons, pad.lt, pad.rt), (0x1000, 0, 255));
        let pad = scripted_pad(buttons_at(&t, 510).unwrap());
        assert_eq!((pad.buttons, pad.lt, pad.rt), (0, 255, 0));
        assert!(parse("start").is_err());
        assert_eq!(buttons_at(&Script::Unplugged, 0), None);
        assert_eq!(buttons_at(&Script::Default, 5000), Some(0x10));
        assert_eq!(buttons_at(&Script::Default, 6600), Some(0x1000));
        assert_eq!(buttons_at(&Script::Default, 9000), Some(0x10));
        // No window yet: a live pad is plugged in and centred.
        assert_eq!(buttons_at(&Script::Live, 0), Some(0));
    }

    #[test]
    fn packet_follows_state() {
        let a = packet(Pad::default());
        assert_eq!(packet(Pad::default()), a);
        let moved = Pad {
            lx: 100,
            ..Pad::default()
        };
        let b = packet(moved);
        assert_ne!(a, b);
        assert_eq!(packet(moved), b);
    }
}
