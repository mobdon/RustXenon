//! `xenon-input`: the controller state the host reports to the game.
//!
//! Pure logic only: the default keyboard layout (keys named by their
//! physical-key code, as winit's `KeyCode` prints them, such as `KeyW` or
//! `ArrowUp`), radial stick dead zones, and merging a real pad with the
//! keyboard. The window crate feeds it key events and XInput readings.

pub mod android;

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

/// The default keyboard layout (documented in the README).
pub const DEFAULT_KEYS: [(&str, Bind); 26] = [
    ("KeyW", Bind::Left(0, 1)),
    ("KeyS", Bind::Left(0, -1)),
    ("KeyA", Bind::Left(-1, 0)),
    ("KeyD", Bind::Left(1, 0)),
    ("ShiftLeft", Bind::Button(0x0040)),
    ("KeyI", Bind::Right(0, 1)),
    ("KeyK", Bind::Right(0, -1)),
    ("KeyJ", Bind::Right(-1, 0)),
    ("KeyL", Bind::Right(1, 0)),
    ("ShiftRight", Bind::Button(0x0080)),
    ("ArrowUp", Bind::Button(0x0001)),
    ("ArrowDown", Bind::Button(0x0002)),
    ("ArrowLeft", Bind::Button(0x0004)),
    ("ArrowRight", Bind::Button(0x0008)),
    ("Space", Bind::Button(0x1000)),
    ("KeyC", Bind::Button(0x2000)),
    ("KeyF", Bind::Button(0x4000)),
    ("KeyR", Bind::Button(0x8000)),
    ("KeyQ", Bind::Button(0x0100)),
    ("KeyE", Bind::Button(0x0200)),
    ("KeyZ", Bind::LeftTrigger),
    ("KeyX", Bind::RightTrigger),
    ("Enter", Bind::Button(0x0010)),
    ("Escape", Bind::Button(0x0010)),
    ("Backspace", Bind::Button(0x0020)),
    ("Tab", Bind::Button(0x0020)),
];

/// XInput's recommended dead zones.
pub const LEFT_DEAD_ZONE: i16 = 7849;
pub const RIGHT_DEAD_ZONE: i16 = 8689;
pub const TRIGGER_THRESHOLD: u8 = 30;

/// Keys held down, turned into a pad reading with [`DEFAULT_KEYS`].
#[derive(Default)]
pub struct Keyboard {
    held: HashSet<String>,
}

impl Keyboard {
    /// A key went down or up. Unbound keys are ignored.
    pub fn key(&mut self, code: &str, down: bool) {
        if !DEFAULT_KEYS.iter().any(|k| k.0 == code) {
            return;
        }
        if down {
            self.held.insert(code.to_string());
        } else {
            self.held.remove(code);
        }
    }

    /// Lets go of every key (the window lost focus).
    pub fn release_all(&mut self) {
        self.held.clear();
    }

    pub fn pad(&self) -> Pad {
        let mut p = Pad::default();
        let (mut l, mut r) = ((0i32, 0i32), (0i32, 0i32));
        for (code, bind) in DEFAULT_KEYS {
            if !self.held.contains(code) {
                continue;
            }
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
        k.key("KeyX", true);
        k.key("KeyP", true); // unbound
        let p = k.pad();
        assert_eq!((p.lx, p.ly), (32767, 32767));
        assert_eq!(p.buttons, 0x1000);
        assert_eq!((p.lt, p.rt), (0, 255));
        k.key("KeyA", true); // opposite of D cancels
        assert_eq!(k.pad().lx, 0);
        k.key("KeyW", false);
        k.key("KeyK", true);
        let p = k.pad();
        assert_eq!((p.ly, p.ry), (0, -32767));
        k.key("Enter", true);
        k.key("Escape", true);
        k.key("Enter", false);
        assert_eq!(k.pad().buttons & 0x0010, 0x0010); // Escape still holds Start
        k.release_all();
        assert_eq!(k.pad(), Pad::default());
    }

    #[test]
    fn every_button_has_a_key() {
        for (name, bit) in BUTTONS {
            assert!(
                DEFAULT_KEYS.iter().any(|k| k.1 == Bind::Button(bit)),
                "{name} has no key"
            );
        }
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
