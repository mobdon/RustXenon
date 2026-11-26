//! Android controller events (key codes and joystick axes) to a [`Pad`].
//!
//! Stub: a simple version of the API so the window crate compiles. The real
//! mapping with its tests replaces this file.

use crate::Pad;

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

const KEYS: [(i32, u16); 15] = [
    (KEYCODE_DPAD_UP, 0x0001),
    (KEYCODE_DPAD_DOWN, 0x0002),
    (KEYCODE_DPAD_LEFT, 0x0004),
    (KEYCODE_DPAD_RIGHT, 0x0008),
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

#[derive(Clone, Debug, Default)]
pub struct Gamepad {
    keys: u16,
    l2: bool,
    r2: bool,
    axes: Pad,
}

impl Gamepad {
    /// A key event whose source is a gamepad, joystick or D-pad. Returns
    /// true when the code is one the pad uses.
    pub fn key(&mut self, keycode: i32, down: bool) -> bool {
        match keycode {
            KEYCODE_BUTTON_L2 => self.l2 = down,
            KEYCODE_BUTTON_R2 => self.r2 = down,
            KEYCODE_BUTTON_MODE => {}
            _ => match KEYS.iter().find(|k| k.0 == keycode) {
                Some(&(_, bit)) if down => self.keys |= bit,
                Some(&(_, bit)) => self.keys &= !bit,
                None => return false,
            },
        }
        true
    }

    /// A joystick MotionEvent: `axis(id)` is the event's value for axis `id`.
    pub fn motion(&mut self, axis: impl Fn(i32) -> f32) {
        let stick = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let trig = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let (hx, hy) = (axis(AXIS_HAT_X), axis(AXIS_HAT_Y));
        let bit = |on: bool, b: u16| if on { b } else { 0 };
        self.axes = Pad {
            buttons: bit(hy < -0.5, 1) | bit(hy > 0.5, 2) | bit(hx < -0.5, 4) | bit(hx > 0.5, 8),
            lt: trig(axis(AXIS_LTRIGGER).max(axis(AXIS_BRAKE))),
            rt: trig(axis(AXIS_RTRIGGER).max(axis(AXIS_GAS))),
            lx: stick(axis(AXIS_X)),
            ly: stick(-axis(AXIS_Y)),
            rx: stick(axis(AXIS_Z)),
            ry: stick(-axis(AXIS_RZ)),
        };
    }

    /// The pad now, before dead zones.
    pub fn pad(&self) -> Pad {
        let mut p = self.axes;
        p.buttons |= self.keys;
        if self.l2 {
            p.lt = 255;
        }
        if self.r2 {
            p.rt = 255;
        }
        p
    }

    /// Everything released (focus lost, controller gone).
    pub fn release_all(&mut self) {
        *self = Gamepad::default();
    }
}
