//! A scripted virtual controller on port 0; ports 1 to 3 stay unplugged.
//!
//! Headless runs need a way past "press start": the script presses buttons
//! at fixed times since start. The default presses Start, then A a second
//! and a half later, every four seconds from the fifth second. `--press`
//! replaces it with a list of `<ms>:<button>[+<button>...]` presses (each
//! held 150 ms), and `--no-controller` unplugs the pad.

use super::*;
use std::sync::OnceLock;

/// Button bits of the gamepad state.
pub const BUTTONS: [(&str, u16); 14] = [
    ("up", 0x0001),
    ("down", 0x0002),
    ("left", 0x0004),
    ("right", 0x0008),
    ("start", 0x0010),
    ("back", 0x0020),
    ("lthumb", 0x0040),
    ("rthumb", 0x0080),
    ("lb", 0x0100),
    ("rb", 0x0200),
    ("a", 0x1000),
    ("b", 0x2000),
    ("x", 0x4000),
    ("y", 0x8000),
];

/// Virtual key codes for keystroke events, in the order of [`BUTTONS`].
const KEYS: [u16; 14] = [
    0x5810, 0x5811, 0x5812, 0x5813, 0x5814, 0x5815, 0x5816, 0x5817, 0x5805, 0x5804, 0x5800, 0x5801,
    0x5802, 0x5803,
];

const HOLD_MS: u64 = 150;
const ERROR_EMPTY: u32 = 0x10D2;

pub enum Script {
    Unplugged,
    /// (start ms, buttons), sorted.
    Presses(Vec<(u64, u16)>),
    /// Start, then A, every four seconds from five seconds.
    Default,
}

static SCRIPT: OnceLock<Script> = OnceLock::new();
/// Buttons last reported as keystrokes, and the state packet counter.
static LAST_KEYS: AtomicU32 = AtomicU32::new(0);

/// Parses `--press` text: `5000:start,6500:a+b`.
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
            bits |= BUTTONS
                .iter()
                .find(|x| x.0 == name)
                .ok_or_else(|| format!("unknown button {name:?}"))?
                .1;
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

fn now() -> Option<u16> {
    buttons_at(script(), crate::ms() as u64)
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

/// State: packet number, buttons, two triggers, four stick axes (centred).
pub fn xam_input_get_state(rt: &Runtime, c: &mut Context) {
    let (user, state) = (arg(rt, c, 0), arg(rt, c, 2));
    let Some(b) = now().filter(|_| connected(user)) else {
        return ret(c, ERROR_DEVICE_NOT_CONNECTED);
    };
    fill(rt, state, 16, 0);
    // The packet number changes exactly when the state does.
    rt.mem.write_u32(
        state,
        (crate::ms() as u64 / HOLD_MS) as u32 * 2 + (b != 0) as u32,
    );
    rt.mem.write_u16(state + 4, b);
    ret(c, ERROR_SUCCESS);
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
    let Some(b) = now().filter(|_| connected(user)) else {
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
        assert!(parse("start").is_err());
        assert_eq!(buttons_at(&Script::Unplugged, 0), None);
        assert_eq!(buttons_at(&Script::Default, 5000), Some(0x10));
        assert_eq!(buttons_at(&Script::Default, 6600), Some(0x1000));
        assert_eq!(buttons_at(&Script::Default, 9000), Some(0x10));
    }
}
