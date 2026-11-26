//! Android gamepad events to a [`Pad`](crate::Pad) reading.
//!
//! Pure logic, built and tested on every platform. The Android window layer
//! feeds it the `KeyEvent`s and joystick `MotionEvent`s it gets from
//! `android-activity`. Code and axis values are those of
//! `android.view.KeyEvent` and `android.view.MotionEvent`.
//!
//! | Android | Pad |
//! |---|---|
//! | `KEYCODE_BUTTON_A/B/X/Y` (96, 97, 99, 100) | A, B, X, Y |
//! | `KEYCODE_BUTTON_L1/R1` (102, 103) | LB, RB |
//! | `KEYCODE_BUTTON_THUMBL/THUMBR` (106, 107) | left, right stick click |
//! | `KEYCODE_BUTTON_START` (108) | Start |
//! | `KEYCODE_BUTTON_SELECT` (109), `KEYCODE_BACK` (4) | Back |
//! | `KEYCODE_BUTTON_MODE` (110) | consumed, ignored (`Pad` has no Guide bit) |
//! | `KEYCODE_DPAD_UP/DOWN/LEFT/RIGHT` (19–22) | D-pad |
//! | `AXIS_HAT_X/Y` (15, 16) | D-pad, past 0.5, merged with the D-pad keys |
//! | `KEYCODE_BUTTON_L2/R2` (104, 105) | LT, RT fully pressed |
//! | `AXIS_X/Y` (0, 1) | left stick, Y negated |
//! | `AXIS_Z/RZ` (11, 14) | right stick, RZ negated |
//! | max(`AXIS_LTRIGGER`, `AXIS_BRAKE`) (17, 23) | LT, 0.0–1.0 to 0–255 |
//! | max(`AXIS_RTRIGGER`, `AXIS_GAS`) (18, 22) | RT, 0.0–1.0 to 0–255 |

use crate::Pad;

/// Key codes, values from `android.view.KeyEvent`.
pub const KEYCODE_BACK: i32 = 4;
pub const KEYCODE_DPAD_UP: i32 = 19;
pub const KEYCODE_DPAD_DOWN: i32 = 20;
pub const KEYCODE_DPAD_LEFT: i32 = 21;
pub const KEYCODE_DPAD_RIGHT: i32 = 22;
pub const KEYCODE_BUTTON_A: i32 = 96;
pub const KEYCODE_BUTTON_B: i32 = 97;
pub const KEYCODE_BUTTON_X: i32 = 99;
pub const KEYCODE_BUTTON_Y: i32 = 100;
pub const KEYCODE_BUTTON_L1: i32 = 102;
pub const KEYCODE_BUTTON_R1: i32 = 103;
pub const KEYCODE_BUTTON_L2: i32 = 104;
pub const KEYCODE_BUTTON_R2: i32 = 105;
pub const KEYCODE_BUTTON_THUMBL: i32 = 106;
pub const KEYCODE_BUTTON_THUMBR: i32 = 107;
pub const KEYCODE_BUTTON_START: i32 = 108;
pub const KEYCODE_BUTTON_SELECT: i32 = 109;
pub const KEYCODE_BUTTON_MODE: i32 = 110;

/// Axis ids, values from `android.view.MotionEvent`.
pub const AXIS_X: i32 = 0;
pub const AXIS_Y: i32 = 1;
pub const AXIS_Z: i32 = 11;
pub const AXIS_RZ: i32 = 14;
pub const AXIS_HAT_X: i32 = 15;
pub const AXIS_HAT_Y: i32 = 16;
pub const AXIS_LTRIGGER: i32 = 17;
pub const AXIS_RTRIGGER: i32 = 18;
pub const AXIS_GAS: i32 = 22;
pub const AXIS_BRAKE: i32 = 23;

const UP: u16 = 0x0001;
const DOWN: u16 = 0x0002;
const LEFT: u16 = 0x0004;
const RIGHT: u16 = 0x0008;

/// Key codes and the button bits they set (D-pad bits included).
const KEY_BUTTONS: [(i32, u16); 15] = [
    (KEYCODE_DPAD_UP, UP),
    (KEYCODE_DPAD_DOWN, DOWN),
    (KEYCODE_DPAD_LEFT, LEFT),
    (KEYCODE_DPAD_RIGHT, RIGHT),
    (KEYCODE_BUTTON_START, 0x0010),
    (KEYCODE_BUTTON_SELECT, 0x0020),
    (KEYCODE_BACK, 0x0020),
    (KEYCODE_BUTTON_THUMBL, 0x0040),
    (KEYCODE_BUTTON_THUMBR, 0x0080),
    (KEYCODE_BUTTON_L1, 0x0100),
    (KEYCODE_BUTTON_R1, 0x0200),
    (KEYCODE_BUTTON_A, 0x1000),
    (KEYCODE_BUTTON_B, 0x2000),
    (KEYCODE_BUTTON_X, 0x4000),
    (KEYCODE_BUTTON_Y, 0x8000),
];

/// One Android controller: held keys and the last joystick reading.
#[derive(Clone, Debug, Default)]
pub struct Gamepad {
    /// Held keys, one bit per [`KEY_BUTTONS`] entry, so SELECT and BACK
    /// release independently.
    keys: u16,
    l2: bool,
    r2: bool,
    hat: u16,
    lx: i16,
    ly: i16,
    rx: i16,
    ry: i16,
    lt: u8,
    rt: u8,
}

/// -1.0..=1.0 to an XInput stick axis: 1.0 is 32767, -1.0 is -32768.
fn stick(v: f32) -> i16 {
    let v = v.clamp(-1.0, 1.0);
    // NaN casts to 0.
    (if v < 0.0 { v * 32768.0 } else { v * 32767.0 }).round() as i16
}

/// 0.0..=1.0 to an XInput trigger.
fn trigger(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl Gamepad {
    /// A key event whose source is a gamepad, joystick or D-pad. Returns true when the code
    /// is one the pad uses, so the caller marks the event handled (stops B turning into BACK).
    pub fn key(&mut self, keycode: i32, down: bool) -> bool {
        let flag = match keycode {
            KEYCODE_BUTTON_L2 => &mut self.l2,
            KEYCODE_BUTTON_R2 => &mut self.r2,
            KEYCODE_BUTTON_MODE => return true,
            _ => {
                let Some(i) = KEY_BUTTONS.iter().position(|k| k.0 == keycode) else {
                    return false;
                };
                if down {
                    self.keys |= 1 << i;
                } else {
                    self.keys &= !(1 << i);
                }
                return true;
            }
        };
        *flag = down;
        true
    }

    /// A joystick MotionEvent: `axis(id)` is the event's value for Android axis `id`.
    pub fn motion(&mut self, axis: impl Fn(i32) -> f32) {
        self.lx = stick(axis(AXIS_X));
        self.ly = stick(-axis(AXIS_Y));
        self.rx = stick(axis(AXIS_Z));
        self.ry = stick(-axis(AXIS_RZ));
        self.lt = trigger(axis(AXIS_LTRIGGER).max(axis(AXIS_BRAKE)));
        self.rt = trigger(axis(AXIS_RTRIGGER).max(axis(AXIS_GAS)));
        let (hx, hy) = (axis(AXIS_HAT_X), axis(AXIS_HAT_Y));
        self.hat = 0;
        for (on, bit) in [
            (hy <= -0.5, UP),
            (hy >= 0.5, DOWN),
            (hx <= -0.5, LEFT),
            (hx >= 0.5, RIGHT),
        ] {
            if on {
                self.hat |= bit;
            }
        }
    }

    /// The pad now, before dead zones (callers apply xenon_input::filtered).
    pub fn pad(&self) -> Pad {
        let mut buttons = self.hat;
        for (i, (_, bit)) in KEY_BUTTONS.iter().enumerate() {
            if self.keys & (1 << i) != 0 {
                buttons |= bit;
            }
        }
        Pad {
            buttons,
            lt: if self.l2 { 255 } else { self.lt },
            rt: if self.r2 { 255 } else { self.rt },
            lx: self.lx,
            ly: self.ly,
            rx: self.rx,
            ry: self.ry,
        }
    }

    /// Everything released (focus lost, controller gone).
    pub fn release_all(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BUTTONS;

    fn bit(name: &str) -> u16 {
        BUTTONS.iter().find(|b| b.0 == name).unwrap().1
    }

    /// A motion event with the given axes set and every other axis 0.
    fn motion(g: &mut Gamepad, set: &[(i32, f32)]) {
        g.motion(|id| set.iter().find(|a| a.0 == id).map_or(0.0, |a| a.1));
    }

    #[test]
    fn every_button() {
        let cases = [
            (KEYCODE_BUTTON_A, "a"),
            (KEYCODE_BUTTON_B, "b"),
            (KEYCODE_BUTTON_X, "x"),
            (KEYCODE_BUTTON_Y, "y"),
            (KEYCODE_BUTTON_L1, "lb"),
            (KEYCODE_BUTTON_R1, "rb"),
            (KEYCODE_BUTTON_THUMBL, "lthumb"),
            (KEYCODE_BUTTON_THUMBR, "rthumb"),
            (KEYCODE_BUTTON_START, "start"),
            (KEYCODE_BUTTON_SELECT, "back"),
            (KEYCODE_BACK, "back"),
            (KEYCODE_DPAD_UP, "up"),
            (KEYCODE_DPAD_DOWN, "down"),
            (KEYCODE_DPAD_LEFT, "left"),
            (KEYCODE_DPAD_RIGHT, "right"),
        ];
        let mut all = 0;
        for (code, name) in cases {
            let mut g = Gamepad::default();
            assert!(g.key(code, true), "{name}");
            assert_eq!(
                g.pad(),
                Pad {
                    buttons: bit(name),
                    ..Pad::default()
                },
                "{name}"
            );
            assert!(g.key(code, false));
            assert_eq!(g.pad(), Pad::default(), "{name}");
            all |= bit(name);
        }
        // Every pad button is reachable.
        assert_eq!(all, BUTTONS.iter().fold(0, |a, b| a | b.1));
    }

    #[test]
    fn buttons_combine_and_back_keys_are_independent() {
        let mut g = Gamepad::default();
        g.key(KEYCODE_BUTTON_A, true);
        g.key(KEYCODE_BUTTON_R1, true);
        assert_eq!(g.pad().buttons, bit("a") | bit("rb"));
        g.key(KEYCODE_BUTTON_SELECT, true);
        g.key(KEYCODE_BACK, true);
        g.key(KEYCODE_BACK, false);
        assert_eq!(g.pad().buttons & bit("back"), bit("back"));
        g.key(KEYCODE_BUTTON_SELECT, false);
        assert_eq!(g.pad().buttons, bit("a") | bit("rb"));
    }

    #[test]
    fn mode_is_consumed_and_ignored() {
        let mut g = Gamepad::default();
        assert!(g.key(KEYCODE_BUTTON_MODE, true));
        assert_eq!(g.pad(), Pad::default());
        assert!(g.key(KEYCODE_BUTTON_MODE, false));
        assert_eq!(g.pad(), Pad::default());
    }

    #[test]
    fn unknown_codes_return_false_and_change_nothing() {
        let mut g = Gamepad::default();
        g.key(KEYCODE_BUTTON_X, true);
        motion(&mut g, &[(AXIS_X, 0.5)]);
        let before = g.pad();
        // 0 UNKNOWN, 23 DPAD_CENTER, 62 SPACE, 98 BUTTON_C, 101 BUTTON_Z, 188 BUTTON_1.
        for code in [0, 23, 62, 98, 101, 188, -1, i32::MAX] {
            assert!(!g.key(code, true), "{code}");
            assert!(!g.key(code, false), "{code}");
            assert_eq!(g.pad(), before, "{code}");
        }
    }

    #[test]
    fn digital_triggers() {
        let mut g = Gamepad::default();
        assert!(g.key(KEYCODE_BUTTON_L2, true));
        assert_eq!(
            g.pad(),
            Pad {
                lt: 255,
                ..Pad::default()
            }
        );
        assert!(g.key(KEYCODE_BUTTON_R2, true));
        assert_eq!((g.pad().lt, g.pad().rt), (255, 255));
        // The key wins over a partly pressed analog trigger.
        motion(&mut g, &[(AXIS_LTRIGGER, 0.2), (AXIS_RTRIGGER, 0.2)]);
        assert_eq!((g.pad().lt, g.pad().rt), (255, 255));
        g.key(KEYCODE_BUTTON_L2, false);
        g.key(KEYCODE_BUTTON_R2, false);
        assert_eq!((g.pad().lt, g.pad().rt), (51, 51));
        assert_eq!(g.pad().buttons, 0);
    }

    #[test]
    fn sticks_sign_and_range() {
        let mut g = Gamepad::default();
        // Android Y points down; pad Y points up.
        motion(
            &mut g,
            &[
                (AXIS_X, 1.0),
                (AXIS_Y, -1.0),
                (AXIS_Z, -1.0),
                (AXIS_RZ, 1.0),
            ],
        );
        let p = g.pad();
        assert_eq!((p.lx, p.ly, p.rx, p.ry), (32767, 32767, -32768, -32768));
        motion(
            &mut g,
            &[
                (AXIS_X, -1.0),
                (AXIS_Y, 1.0),
                (AXIS_Z, 1.0),
                (AXIS_RZ, -1.0),
            ],
        );
        let p = g.pad();
        assert_eq!((p.lx, p.ly, p.rx, p.ry), (-32768, -32768, 32767, 32767));
        // Out of range clamps; half way rounds; NaN is centred.
        motion(
            &mut g,
            &[
                (AXIS_X, 3.0),
                (AXIS_Y, 3.0),
                (AXIS_Z, 0.5),
                (AXIS_RZ, f32::NAN),
            ],
        );
        let p = g.pad();
        assert_eq!((p.lx, p.ly, p.rx, p.ry), (32767, -32768, 16384, 0));
        motion(&mut g, &[]);
        assert_eq!(g.pad(), Pad::default());
        // Sticks set no buttons and leave triggers alone.
        motion(&mut g, &[(AXIS_X, 1.0), (AXIS_Y, 1.0)]);
        assert_eq!((g.pad().buttons, g.pad().lt, g.pad().rt), (0, 0, 0));
    }

    #[test]
    fn triggers_take_the_larger_axis() {
        let mut g = Gamepad::default();
        motion(
            &mut g,
            &[
                (AXIS_LTRIGGER, 0.25),
                (AXIS_BRAKE, 1.0),
                (AXIS_RTRIGGER, 0.5),
                (AXIS_GAS, 0.1),
            ],
        );
        assert_eq!((g.pad().lt, g.pad().rt), (255, 128));
        motion(&mut g, &[(AXIS_BRAKE, 0.5), (AXIS_GAS, 1.0)]);
        assert_eq!((g.pad().lt, g.pad().rt), (128, 255));
        motion(&mut g, &[(AXIS_LTRIGGER, -1.0), (AXIS_RTRIGGER, 2.0)]);
        assert_eq!((g.pad().lt, g.pad().rt), (0, 255));
        motion(
            &mut g,
            &[(AXIS_LTRIGGER, f32::NAN), (AXIS_RTRIGGER, f32::NAN)],
        );
        assert_eq!((g.pad().lt, g.pad().rt), (0, 0));
        // Triggers set no buttons and leave sticks alone.
        motion(&mut g, &[(AXIS_LTRIGGER, 1.0), (AXIS_RTRIGGER, 1.0)]);
        assert_eq!(
            Pad {
                lt: 0,
                rt: 0,
                ..g.pad()
            },
            Pad::default()
        );
    }

    #[test]
    fn hat_is_the_dpad() {
        let mut g = Gamepad::default();
        let cases = [
            (0.0, -1.0, "up"),
            (0.0, 1.0, "down"),
            (-1.0, 0.0, "left"),
            (1.0, 0.0, "right"),
            (0.0, -0.5, "up"),
        ];
        for (x, y, name) in cases {
            motion(&mut g, &[(AXIS_HAT_X, x), (AXIS_HAT_Y, y)]);
            assert_eq!(
                g.pad(),
                Pad {
                    buttons: bit(name),
                    ..Pad::default()
                },
                "{name}"
            );
        }
        motion(&mut g, &[(AXIS_HAT_X, 1.0), (AXIS_HAT_Y, -1.0)]);
        assert_eq!(g.pad().buttons, bit("up") | bit("right"));
        // Under the threshold is centred.
        motion(&mut g, &[(AXIS_HAT_X, 0.49), (AXIS_HAT_Y, -0.49)]);
        assert_eq!(g.pad().buttons, 0);
    }

    #[test]
    fn hat_and_dpad_keys_merge() {
        let mut g = Gamepad::default();
        g.key(KEYCODE_DPAD_UP, true);
        motion(&mut g, &[(AXIS_HAT_X, -1.0)]);
        assert_eq!(g.pad().buttons, bit("up") | bit("left"));
        // Hat says up too; the key going up keeps it down.
        motion(&mut g, &[(AXIS_HAT_Y, -1.0)]);
        g.key(KEYCODE_DPAD_UP, false);
        assert_eq!(g.pad().buttons, bit("up"));
        // The key holds it when the hat centres.
        g.key(KEYCODE_DPAD_UP, true);
        motion(&mut g, &[]);
        assert_eq!(g.pad().buttons, bit("up"));
        g.key(KEYCODE_DPAD_UP, false);
        assert_eq!(g.pad().buttons, 0);
    }

    #[test]
    fn release_all() {
        let mut g = Gamepad::default();
        for code in KEY_BUTTONS
            .iter()
            .map(|k| k.0)
            .chain([KEYCODE_BUTTON_L2, KEYCODE_BUTTON_R2])
        {
            g.key(code, true);
        }
        let all = [
            (AXIS_X, 1.0),
            (AXIS_Y, 1.0),
            (AXIS_Z, 1.0),
            (AXIS_RZ, 1.0),
            (AXIS_HAT_X, 1.0),
            (AXIS_HAT_Y, 1.0),
            (AXIS_LTRIGGER, 1.0),
            (AXIS_RTRIGGER, 1.0),
        ];
        motion(&mut g, &all);
        assert_ne!(g.pad(), Pad::default());
        g.release_all();
        assert_eq!(g.pad(), Pad::default());
    }

    #[test]
    fn filtered_reuses_dead_zones() {
        let mut g = Gamepad::default();
        motion(&mut g, &[(AXIS_X, 0.1), (AXIS_RZ, -1.0)]);
        let p = crate::filtered(g.pad());
        assert_eq!((p.lx, p.ry), (0, 32767));
    }
}
