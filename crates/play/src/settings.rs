//! The player's defaults in `<home>/settings.toml`: the `[window]` and
//! `[draw]` keys a game's `run.toml` can override. Keys the launcher does not
//! know are kept when it saves (comments are not).

use std::path::Path;
use toml::{Table, Value};

pub enum Kind {
    Bool(bool),
    /// The values offered, smallest first, and the default.
    Int(&'static [i64], i64),
}

pub struct Field {
    pub table: &'static str,
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    pub kind: Kind,
}

pub const FIELDS: [Field; 7] = [
    Field {
        table: "window",
        key: "width",
        label: "Window width",
        help: "Width of the game window in pixels when it is not full screen.",
        kind: Kind::Int(&[960, 1280, 1600, 1920, 2560, 3840], 1280),
    },
    Field {
        table: "window",
        key: "height",
        label: "Window height",
        help: "Height of the game window in pixels when it is not full screen.",
        kind: Kind::Int(&[540, 720, 900, 1080, 1440, 2160], 720),
    },
    Field {
        table: "window",
        key: "fullscreen",
        label: "Full screen",
        help: "Start games full screen.",
        kind: Kind::Bool(false),
    },
    Field {
        table: "window",
        key: "refresh",
        label: "Refresh rate",
        help: "Frames a second the game window aims for.",
        kind: Kind::Int(&[30, 50, 60, 120, 144, 165, 240], 60),
    },
    Field {
        table: "draw",
        key: "aniso",
        label: "Anisotropic filter",
        help: "Texture sharpness at steep angles. 1 is off, 16 is sharpest.",
        kind: Kind::Int(&[1, 2, 4, 8, 16], 16),
    },
    Field {
        table: "draw",
        key: "force_linear",
        label: "Force linear filter",
        help: "Smooth every texture, even ones the game draws with sharp pixels.",
        kind: Kind::Bool(false),
    },
    Field {
        table: "draw",
        key: "scale",
        label: "Render scale",
        help: "Draw at this many times the console's resolution. Higher is sharper and slower.",
        kind: Kind::Int(&[1, 2, 3, 4], 1),
    },
];

#[derive(Default)]
pub struct Settings(pub Table);

impl Settings {
    /// A missing file is all defaults.
    pub fn load(path: &Path) -> Result<Settings, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => text.parse::<Table>().map(Settings).map_err(|e| {
                format!(
                    "{}: {}",
                    path.display(),
                    e.to_string().lines().next().unwrap_or("")
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = toml::to_string(&self.0).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn get(&self, f: &Field) -> Option<&Value> {
        self.0.get(f.table)?.as_table()?.get(f.key)
    }

    pub fn int(&self, i: usize) -> i64 {
        let f = &FIELDS[i];
        let d = match f.kind {
            Kind::Int(_, d) => d,
            Kind::Bool(b) => b as i64,
        };
        match self.get(f) {
            Some(Value::Integer(v)) => *v,
            Some(Value::Boolean(b)) => *b as i64,
            _ => d,
        }
    }

    /// The value as the Settings page shows it.
    pub fn show(&self, i: usize) -> String {
        match FIELDS[i].kind {
            Kind::Bool(_) => if self.int(i) != 0 { "On" } else { "Off" }.into(),
            Kind::Int(..) => self.int(i).to_string(),
        }
    }

    /// Moves field `i` one value up (`dir` > 0) or down; a bool flips.
    pub fn step(&mut self, i: usize, dir: i32) {
        let f = &FIELDS[i];
        let v = match f.kind {
            Kind::Bool(_) => Value::Boolean(self.int(i) == 0),
            Kind::Int(choices, _) => {
                let cur = self.int(i);
                let next = if dir > 0 {
                    choices.iter().copied().find(|&c| c > cur)
                } else {
                    choices.iter().rev().copied().find(|&c| c < cur)
                };
                Value::Integer(next.unwrap_or(cur))
            }
        };
        let t = self
            .0
            .entry(f.table)
            .or_insert_with(|| Value::Table(Table::new()));
        if !t.is_table() {
            *t = Value::Table(Table::new());
        }
        if let Value::Table(t) = t {
            t.insert(f.key.into(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_steps_and_unknown_keys() {
        let mut s = Settings::default();
        assert_eq!(s.show(0), "1280");
        assert_eq!(s.show(2), "Off");
        s.step(0, 1);
        assert_eq!(s.int(0), 1600);
        s.step(0, -1);
        s.step(0, -1);
        s.step(0, -1);
        assert_eq!(s.int(0), 960, "stops at the smallest");
        s.step(2, 1);
        assert_eq!(s.show(2), "On");

        // A value not offered steps to its neighbour; other keys survive.
        let dir = crate::tests::temp("settings");
        let path = dir.join("settings.toml");
        std::fs::write(&path, "[draw]\naniso = 3\nmine = 'kept'\n[other]\nx = 1\n").unwrap();
        let mut s = Settings::load(&path).unwrap();
        s.step(4, 1);
        assert_eq!(s.int(4), 4);
        s.save(&path).unwrap();
        let back = Settings::load(&path).unwrap();
        assert_eq!(back.int(4), 4);
        assert_eq!(back.0["draw"]["mine"].as_str(), Some("kept"));
        assert_eq!(back.0["other"]["x"].as_integer(), Some(1));
        assert!(Settings::load(&dir.join("missing.toml"))
            .unwrap()
            .0
            .is_empty());
        std::fs::write(&path, "[[[").unwrap();
        assert!(Settings::load(&path).is_err());
    }
}
