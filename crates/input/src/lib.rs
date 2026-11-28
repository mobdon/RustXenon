//! `xenon-input`: the controller state the host reports to the game.
//!
//! Pure logic only: the default keyboard layout (keys named by their
//! physical-key code, as winit's `KeyCode` prints them, such as `KeyW` or
//! `ArrowUp`), radial stick dead zones, and merging a real pad with the
//! keyboard. The window crate feeds it key events and XInput readings.

use std::collections::HashSet;

/// Button bits of the gamepad state, with the names `--press` uses.
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

/// One controller reading, in XInput's units: triggers 0..=255, stick axes
/// -32768..=32767 with up positive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub buttons: u16,
    pub lt: u8,
    pub rt: u8,
    pub lx: i16,
    pub ly: i16,
    pub rx: i16,
    pub ry: i16,
}

/// What a key does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Button(u16),
    LeftTrigger,
    RightTrigger,
    /// Left stick: x and y direction, each -1, 0 or 1.
    Left(i8, i8),
    Right(i8, i8),
}

/// The default keyboard layout, PC action-game style: WASD to move,
/// mouse to look (see [`Mouse`]), Space to sprint, Shift to jump, F to get
/// in or out of a vehicle. [`VEHICLE_KEYS`] override some keys while in one.
pub const DEFAULT_KEYS: [(&str, Bind); 27] = [
    ("KeyW", Bind::Left(0, 1)),
    ("KeyS", Bind::Left(0, -1)),
    ("KeyA", Bind::Left(-1, 0)),
    ("KeyD", Bind::Left(1, 0)),
    ("Space", Bind::Button(0x1000)),     // A: sprint
    ("ShiftLeft", Bind::Button(0x4000)), // X: jump
    ("KeyF", Bind::Button(0x8000)),      // Y: enter or leave a vehicle
    ("KeyR", Bind::Button(0x2000)),      // B: reload, melee
    ("KeyQ", Bind::Button(0x0200)),      // RB: take cover
    ("KeyE", Bind::Button(0x0100)),      // LB: action, pick up
    ("KeyC", Bind::Button(0x0040)),      // left stick click: crouch
    ("KeyX", Bind::Button(0x0080)),      // right stick click: look behind
    ("KeyV", Bind::Button(0x0020)),      // Back: camera
    ("Tab", Bind::Button(0x0020)),
    ("ArrowUp", Bind::Button(0x0001)), // phone
    ("ArrowDown", Bind::Button(0x0002)),
    ("ArrowLeft", Bind::Button(0x0004)),
    ("ArrowRight", Bind::Button(0x0008)),
    ("Enter", Bind::Button(0x0010)),
    ("Escape", Bind::Button(0x0010)),
    ("Backspace", Bind::Button(0x0020)),
    ("KeyI", Bind::Right(0, 1)),
    ("KeyK", Bind::Right(0, -1)),
    ("KeyJ", Bind::Right(-1, 0)),
    ("KeyL", Bind::Right(1, 0)),
    ("KeyZ", Bind::LeftTrigger),
    ("KeyB", Bind::RightTrigger),
];

/// Keys that do something else in a vehicle: W and S drive (the triggers),
/// Space is the handbrake, H the horn, and the numpad pitches and rolls
/// aircraft (left stick).
pub const VEHICLE_KEYS: [(&str, Bind); 8] = [
    ("KeyW", Bind::RightTrigger),
    ("KeyS", Bind::LeftTrigger),
    ("Space", Bind::Button(0x0200)), // RB: handbrake
    ("KeyH", Bind::Button(0x0040)),  // left stick click: horn
    ("Numpad8", Bind::Left(0, 1)),
    ("Numpad5", Bind::Left(0, -1)),
    ("Numpad4", Bind::Left(-1, 0)),
    ("Numpad6", Bind::Left(1, 0)),
];

/// The key that gets in or out of a vehicle, switching [`VEHICLE_KEYS`].
pub const VEHICLE_TOGGLE: &str = "KeyF";

/// XInput's recommended dead zones.
pub const LEFT_DEAD_ZONE: i16 = 7849;
pub const RIGHT_DEAD_ZONE: i16 = 8689;
pub const TRIGGER_THRESHOLD: u8 = 30;

/// Seconds a held movement key takes to push the left stick from centre
/// to full, and to let it back: a key that snapped the stick to full made
/// steering jerk from straight to full lock.
const RAMP_UP: f32 = 0.18;
const RAMP_DOWN: f32 = 0.08;

/// Mouse speed, in counts per second, that pushes the right stick fully.
pub const MOUSE_FULL: f32 = 1500.0;

/// Keys held down, turned into a pad reading with [`DEFAULT_KEYS`].
#[derive(Default)]
pub struct Keyboard {
    held: HashSet<String>,
    /// The left stick as it ramps toward the held keys, -1..=1.
    left: (f32, f32),
    /// In a vehicle: [`VEHICLE_KEYS`] apply.
    vehicle: bool,
}

/// Moves `v` toward `target` by `dt` seconds of ramping.
fn approach(v: f32, target: f32, dt: f32) -> f32 {
    let pushing =
        target != 0.0 && (v == 0.0 || v.signum() == target.signum()) && v.abs() < target.abs();
    let step = dt / if pushing { RAMP_UP } else { RAMP_DOWN };
    if (target - v).abs() <= step {
        target
    } else {
        v + step * (target - v).signum()
    }
}

impl Keyboard {
    /// A key went down or up. Unbound keys are ignored.
    pub fn key(&mut self, code: &str, down: bool) {
        let bound = |t: &[(&str, Bind)]| t.iter().any(|k| k.0 == code);
        if !bound(&DEFAULT_KEYS) && !bound(&VEHICLE_KEYS) {
            return;
        }
        // shortcut: the vehicle keys follow F presses, not the game's state, so
        // an F that gets in nothing flips them wrongly; F again sets them back.
        if down && code == VEHICLE_TOGGLE && !self.held.contains(code) {
            self.vehicle = !self.vehicle;
        }
        if down {
            self.held.insert(code.to_string());
        } else {
            self.held.remove(code);
        }
    }

    /// Whether the vehicle keys apply.
    pub fn in_vehicle(&self) -> bool {
        self.vehicle
    }

    /// Lets go of every key (the window lost focus).
    pub fn release_all(&mut self) {
        self.held.clear();
        self.left = (0.0, 0.0);
    }

    /// Ramps the left stick `dt` seconds toward the held keys.
    pub fn step(&mut self, dt: f32) {
        let t = self.target();
        let unit = |v: i16| v as f32 / 32767.0;
        self.left = (
            approach(self.left.0, unit(t.lx), dt),
            approach(self.left.1, unit(t.ly), dt),
        );
    }

    /// The reading, with the left stick where [`Keyboard::step`] ramped it.
    pub fn pad(&self) -> Pad {
        let axis = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        Pad {
            lx: axis(self.left.0),
            ly: axis(self.left.1),
            ..self.target()
        }
    }

    /// The reading the held keys ask for, sticks at once.
    fn target(&self) -> Pad {
        let mut p = Pad::default();
        let (mut l, mut r) = ((0i32, 0i32), (0i32, 0i32));
        let find = |t: &[(&str, Bind)], code: &str| t.iter().find(|k| k.0 == code).map(|k| k.1);
        for code in &self.held {
            let car = self.vehicle.then(|| find(&VEHICLE_KEYS, code)).flatten();
            let Some(bind) = car.or_else(|| find(&DEFAULT_KEYS, code)) else {
                continue;
            };
            match bind {
                Bind::Button(b) => p.buttons |= b,
                Bind::LeftTrigger => p.lt = 255,
                Bind::RightTrigger => p.rt = 255,
                Bind::Left(x, y) => l = (l.0 + x as i32, l.1 + y as i32),
                Bind::Right(x, y) => r = (r.0 + x as i32, r.1 + y as i32),
            }
        }
        // Opposite keys cancel; a diagonal is full on both axes like a
        // square-gated stick.
        let axis = |v: i32| (v.clamp(-1, 1) * 32767) as i16;
        (p.lx, p.ly, p.rx, p.ry) = (axis(l.0), axis(l.1), axis(r.0), axis(r.1));
        p
    }
}

/// Mouse motion as the right stick (the camera) and its buttons as the
/// triggers: left fires (right trigger), right aims (left trigger).
#[derive(Default)]
pub struct Mouse {
    moved: (f64, f64),
    stick: (f32, f32),
    pub left: bool,
    pub right: bool,
}

impl Mouse {
    /// Raw motion in counts, right and down positive.
    pub fn motion(&mut self, dx: f64, dy: f64) {
        self.moved.0 += dx;
        self.moved.1 += dy;
    }

    /// Lets go of the buttons and stops the camera (focus lost).
    pub fn release(&mut self) {
        *self = Mouse::default();
    }

    /// Turns the motion since the last call, `dt` seconds ago, into a pad
    /// reading: speed over [`MOUSE_FULL`] is a full stick, up is up. In a
    /// vehicle the left button shoots from it (LB) and the triggers stay
    /// with the keys that drive.
    pub fn pad_in(&mut self, dt: f32, vehicle: bool) -> Pad {
        let mut p = self.pad(dt);
        if vehicle {
            if p.rt != 0 {
                p.buttons |= 0x0100;
            }
            (p.lt, p.rt) = (0, 0);
        }
        p
    }

    /// [`Mouse::pad_in`] on foot.
    pub fn pad(&mut self, dt: f32) -> Pad {
        let dt = dt.max(0.001);
        let v = (
            self.moved.0 as f32 / dt / MOUSE_FULL,
            -self.moved.1 as f32 / dt / MOUSE_FULL,
        );
        self.moved = (0.0, 0.0);
        // Polls every few ms see a count or two: half of the last reading
        // keeps the camera from stuttering between polls.
        self.stick = ((self.stick.0 + v.0) / 2.0, (self.stick.1 + v.1) / 2.0);
        let axis = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        Pad {
            rx: axis(self.stick.0),
            ry: axis(self.stick.1),
            lt: if self.right { 255 } else { 0 },
            rt: if self.left { 255 } else { 0 },
            ..Pad::default()
        }
    }
}

/// Radial dead zone: inside `dz` the stick is centred; outside, the
/// distance is rescaled so it still runs from 0 to full.
pub fn dead_zone(x: i16, y: i16, dz: i16) -> (i16, i16) {
    let (fx, fy) = (x as f32, y as f32);
    let m = (fx * fx + fy * fy).sqrt();
    if m <= dz as f32 {
        return (0, 0);
    }
    let scaled = ((m - dz as f32) / (32767.0 - dz as f32)).min(1.0) * 32767.0;
    let k = scaled / m;
    let c = |v: f32| (v * k).round().clamp(-32768.0, 32767.0) as i16;
    (c(fx), c(fy))
}

/// A raw XInput reading with the standard dead zones and trigger threshold.
pub fn filtered(p: Pad) -> Pad {
    let (lx, ly) = dead_zone(p.lx, p.ly, LEFT_DEAD_ZONE);
    let (rx, ry) = dead_zone(p.rx, p.ry, RIGHT_DEAD_ZONE);
    let t = |v: u8| if v < TRIGGER_THRESHOLD { 0 } else { v };
    Pad {
        lt: t(p.lt),
        rt: t(p.rt),
        lx,
        ly,
        rx,
        ry,
        ..p
    }
}

/// Pad and keyboard together: buttons from either, the further-pressed
/// trigger, and each stick from whichever source pushes it further.
pub fn merge(a: Pad, b: Pad) -> Pad {
    let far = |a: (i16, i16), b: (i16, i16)| {
        let m = |s: (i16, i16)| (s.0 as i64).pow(2) + (s.1 as i64).pow(2);
        if m(b) > m(a) {
            b
        } else {
            a
        }
    };
    let (lx, ly) = far((a.lx, a.ly), (b.lx, b.ly));
    let (rx, ry) = far((a.rx, a.ry), (b.rx, b.ry));
    Pad {
        buttons: a.buttons | b.buttons,
        lt: a.lt.max(b.lt),
        rt: a.rt.max(b.rt),
        lx,
        ly,
        rx,
        ry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_layout() {
        let mut k = Keyboard::default();
        assert_eq!(k.pad(), Pad::default());
        k.key("KeyW", true);
        k.key("KeyD", true);
        k.key("Space", true);
        k.key("KeyB", true);
        k.key("KeyP", true); // unbound
        k.step(1.0);
        let p = k.pad();
        assert_eq!((p.lx, p.ly), (32767, 32767));
        assert_eq!(p.buttons, 0x1000);
        assert_eq!((p.lt, p.rt), (0, 255));
        k.key("KeyA", true); // opposite of D cancels
        k.step(1.0);
        assert_eq!(k.pad().lx, 0);
        k.key("KeyW", false);
        k.key("KeyK", true);
        k.step(1.0);
        let p = k.pad();
        assert_eq!((p.ly, p.ry), (0, -32767));
        k.key("Enter", true);
        k.key("Escape", true);
        k.key("Enter", false);
        assert_eq!(k.pad().buttons & 0x0010, 0x0010); // Escape still holds Start
        k.release_all();
        k.step(1.0);
        assert_eq!(k.pad(), Pad::default());
    }

    #[test]
    fn vehicle_keys_follow_f() {
        let mut k = Keyboard::default();
        k.key("KeyF", true);
        assert!(k.in_vehicle());
        assert_eq!(k.pad().buttons, 0x8000); // F is still Y
        k.key("KeyF", true); // key repeat does not flip it back
        k.key("KeyF", false);
        k.key("KeyW", true);
        k.key("Space", true);
        k.key("KeyD", true);
        k.step(1.0);
        let p = k.pad();
        assert_eq!((p.rt, p.lt, p.ly, p.lx), (255, 0, 0, 32767));
        assert_eq!(p.buttons, 0x0200); // handbrake
        k.key("KeyH", true);
        assert_eq!(k.pad().buttons & 0x0040, 0x0040);
        k.key("KeyF", true);
        k.key("KeyF", false);
        assert!(!k.in_vehicle());
        k.step(1.0);
        let p = k.pad();
        assert_eq!((p.rt, p.ly), (0, 32767)); // W walks again
        assert_eq!(p.buttons & 0x0040, 0); // H does nothing on foot
        let mut m = Mouse {
            left: true,
            ..Mouse::default()
        };
        assert_eq!(
            (m.pad_in(0.01, true).buttons, m.pad_in(0.01, true).rt),
            (0x0100, 0)
        );
    }

    #[test]
    fn steering_ramps_in_and_out() {
        let mut k = Keyboard::default();
        k.key("KeyD", true);
        k.step(RAMP_UP / 2.0);
        let half = k.pad().lx;
        assert!((15000..18000).contains(&half), "{half}");
        k.step(RAMP_UP);
        assert_eq!(k.pad().lx, 32767);
        k.key("KeyD", false);
        k.step(RAMP_DOWN);
        assert_eq!(k.pad().lx, 0);
        // Straight from right to left goes through the centre at the
        // release rate, then ramps in.
        k.key("KeyD", true);
        k.step(1.0);
        k.key("KeyD", false);
        k.key("KeyA", true);
        k.step(RAMP_DOWN);
        assert_eq!(k.pad().lx, 0);
        k.step(1.0);
        assert_eq!(k.pad().lx, -32767);
    }

    #[test]
    fn mouse_turns_the_camera_and_holds_triggers() {
        let mut m = Mouse::default();
        m.motion(MOUSE_FULL as f64 * 0.01, 0.0);
        let p = m.pad(0.01);
        assert_eq!(p.rx, 16384); // half: half of the last reading is kept
        let p = m.pad(0.01);
        assert_eq!((p.rx, p.ry), (8192, 0));
        m.motion(0.0, -(MOUSE_FULL as f64) * 0.1);
        m.left = true;
        let p = m.pad(0.01);
        assert_eq!(p.ry, 32767); // mouse up looks up, clamped
        assert_eq!((p.lt, p.rt), (0, 255));
        m.release();
        assert_eq!(m.pad(0.01), Pad::default());
    }

    #[test]
    fn every_button_has_a_key() {
        for (name, bit) in BUTTONS {
            assert!(
                DEFAULT_KEYS.iter().any(|k| k.1 == Bind::Button(bit)),
                "{name} has no key"
            );
        }
        let mut car: Vec<_> = VEHICLE_KEYS.iter().map(|k| k.0).collect();
        car.sort();
        car.dedup();
        assert_eq!(
            car.len(),
            VEHICLE_KEYS.len(),
            "a vehicle key is bound twice"
        );
        let mut codes: Vec<_> = DEFAULT_KEYS.iter().map(|k| k.0).collect();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), DEFAULT_KEYS.len(), "a key is bound twice");
    }

    #[test]
    fn dead_zones() {
        assert_eq!(dead_zone(7000, 3000, LEFT_DEAD_ZONE), (0, 0));
        assert_eq!(dead_zone(32767, 0, LEFT_DEAD_ZONE), (32767, 0));
        assert_eq!(dead_zone(-32768, 0, LEFT_DEAD_ZONE), (-32767, 0));
        // Just outside the edge starts near zero, direction kept.
        let (x, y) = dead_zone(0, -(LEFT_DEAD_ZONE + 100), LEFT_DEAD_ZONE);
        assert_eq!(x, 0);
        assert!((-200..0).contains(&y), "{y}");
        // Halfway along the live range is half deflection.
        let mid = LEFT_DEAD_ZONE as i32 + (32767 - LEFT_DEAD_ZONE as i32) / 2;
        let (x, _) = dead_zone(mid as i16, 0, LEFT_DEAD_ZONE);
        assert!((x - 16384).abs() < 3, "{x}");
        // Corners of a square-ish stick clamp, never wrap.
        let (x, y) = dead_zone(32767, 32767, RIGHT_DEAD_ZONE);
        assert!(x > 20000 && y > 20000 && x == y);
        let f = filtered(Pad {
            lt: 29,
            rt: 30,
            lx: 100,
            ..Pad::default()
        });
        assert_eq!((f.lt, f.rt, f.lx), (0, 30, 0));
    }

    #[test]
    fn merging() {
        let pad = Pad {
            buttons: 0x1000,
            lt: 100,
            lx: 20000,
            ry: -5000,
            ..Pad::default()
        };
        let keys = Pad {
            buttons: 0x0010,
            lt: 255,
            ly: 32767,
            ..Pad::default()
        };
        let m = merge(pad, keys);
        assert_eq!(m.buttons, 0x1010);
        assert_eq!((m.lt, m.rt), (255, 0));
        assert_eq!((m.lx, m.ly), (0, 32767)); // keyboard pushes further
        assert_eq!((m.rx, m.ry), (0, -5000)); // pad's stick kept
        assert_eq!(merge(pad, Pad::default()), pad);
    }
}
