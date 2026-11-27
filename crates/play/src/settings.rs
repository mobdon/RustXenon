//! The player's defaults in `<home>/settings.toml`: the `[window]` and
//! `[draw]` keys a game's `run.toml` can override. A save changes only the
//! key that changed: other keys, comments and layout are kept.

use std::path::Path;
use toml_edit::{DocumentMut, Item};

pub enum Kind {
    /// The default.
    Bool(bool),
    /// The values offered, smallest first, the default, and whether the game
    /// takes only these values (it turns anything else off).
    Int(&'static [i64], i64, bool),
}

pub struct Field {
    pub table: &'static str,
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    pub kind: Kind,
}

/// Each default is what the game uses when the key is missing.
pub const FIELDS: [Field; 7] = [
    Field {
        table: "window",
        key: "width",
        label: "Window width",
        help: "Width of the game window in pixels when it is not full screen.",
        kind: Kind::Int(&[960, 1280, 1600, 1920, 2560, 3840], 1280, false),
    },
    Field {
        table: "window",
        key: "height",
        label: "Window height",
        help: "Height of the game window in pixels when it is not full screen.",
        kind: Kind::Int(&[540, 720, 900, 1080, 1440, 2160], 720, false),
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
        kind: Kind::Int(&[30, 50, 60, 120, 144, 165, 240], 60, false),
    },
    Field {
        table: "draw",
        key: "aniso",
        label: "Anisotropic filter",
        help: "Texture sharpness at steep angles. 1 is off, 16 is sharpest.",
        kind: Kind::Int(&[1, 2, 4, 8, 16], 1, true),
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
        // The drawer takes 1 to 4 and draws at 1 for anything else.
        kind: Kind::Int(&[1, 2, 3, 4], 1, true),
    },
];

/// The first line of a TOML error and its reason, without the source snippet.
pub(crate) fn toml_error(e: &impl std::fmt::Display) -> String {
    let text = e.to_string();
    let mut parts = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.contains('|'));
    let first = parts.next().unwrap_or("");
    match parts.next() {
        Some(why) => format!("{first}: {why}"),
        None => first.into(),
    }
}

#[derive(Clone, Debug, Default)]
pub struct Settings(pub DocumentMut);

impl Settings {
    /// A missing file is all defaults. Errors say what is wrong, not where
    /// the file is (the Settings page shows that).
    pub fn load(path: &Path) -> Result<Settings, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => text
                .parse::<DocumentMut>()
                .map(Settings)
                .map_err(|e| toml_error(&e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Writes a new file, flushes it to the disk and renames it over the old
    /// one, so a crash or power cut part way leaves the old file whole. A
    /// symlink is followed (the file it names gets the new text), and the
    /// old file's line endings, byte order mark and hidden attribute are
    /// kept. shortcut: a hard link to settings.toml is split off and the
    /// file gets the folder's default permissions; write in place if a
    /// player ever needs either.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let path = &std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let old = std::fs::metadata(path).ok();
        if old.as_ref().is_some_and(|m| m.permissions().readonly()) {
            return Err("the file is read-only".into());
        }
        let was = std::fs::read(path).unwrap_or_default();
        let mut text = self.0.to_string();
        if was.windows(2).any(|w| w == b"\r\n") {
            text = text.replace("\r\n", "\n").replace('\n', "\r\n");
        }
        if was.starts_with("\u{feff}".as_bytes()) && !text.starts_with('\u{feff}') {
            text.insert(0, '\u{feff}');
        }
        let tmp = path.with_extension("toml.new");
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(windows)]
        if let Some(m) = &old {
            use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
            // Hidden, system, archive and not-indexed.
            o.attributes(m.file_attributes() & 0x2026);
        }
        let r = o
            .open(&tmp)
            .and_then(|mut f| {
                std::io::Write::write_all(&mut f, text.as_bytes())?;
                f.sync_all()
            })
            .and_then(|()| std::fs::rename(&tmp, path));
        if r.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        r.map_err(|e| e.to_string())
    }

    fn get(&self, f: &Field) -> Option<&Item> {
        self.0.get(f.table)?.as_table_like()?.get(f.key)
    }

    /// False when the field's table or key is there but is not a table and
    /// a value (`[[window]]`, `window = 5`, `[window.width]`,
    /// `width = { x = 1 }`, `width = [1]`): the game ignores it, and a step
    /// would replace what the player wrote.
    fn plain(&self, f: &Field) -> bool {
        let scalar = |v: &Item| {
            v.as_value()
                .is_some_and(|v| !v.is_inline_table() && !v.is_array())
        };
        match self.0.get(f.table) {
            None => true,
            Some(t) => t
                .as_table_like()
                .is_some_and(|t| t.get(f.key).map_or(true, scalar)),
        }
    }

    /// The value the game uses, by the host's rules (a flag is true or false,
    /// a number is a whole number above 0), and whether the file's value, if
    /// it has one, is taken. A value the game does not take gives the default.
    fn value(&self, i: usize) -> (i64, bool) {
        let f = &FIELDS[i];
        if !self.plain(f) {
            let d = match f.kind {
                Kind::Bool(d) => d as i64,
                Kind::Int(_, d, _) => d,
            };
            return (d, false);
        }
        let (d, got) = match f.kind {
            Kind::Bool(d) => (d as i64, self.get(f).map(|v| v.as_bool().map(i64::from))),
            Kind::Int(choices, d, only) => {
                let ok =
                    |v: &i64| (1..=u32::MAX as i64).contains(v) && (!only || choices.contains(v));
                (d, self.get(f).map(|v| v.as_integer().filter(ok)))
            }
        };
        match got {
            None => (d, true),
            Some(Some(v)) => (v, true),
            Some(None) => (d, false),
        }
    }

    pub fn int(&self, i: usize) -> i64 {
        self.value(i).0
    }

    /// False when settings.toml has a value for field `i` that the game does
    /// not take (it uses the default instead).
    pub fn valid(&self, i: usize) -> bool {
        self.value(i).1
    }

    /// The value as the Settings page shows it.
    pub fn show(&self, i: usize) -> String {
        match FIELDS[i].kind {
            Kind::Bool(_) => if self.int(i) != 0 { "On" } else { "Off" }.into(),
            Kind::Int(..) => self.int(i).to_string(),
        }
    }

    /// Moves field `i` one value up (`dir` > 0) or down, wrapping round at
    /// either end so a single button reaches every value; a bool flips.
    /// False, and nothing changed, when its table or key is not a plain
    /// table and value: what the player wrote there is kept.
    pub fn step(&mut self, i: usize, dir: i32) -> bool {
        let f = &FIELDS[i];
        if !self.plain(f) {
            return false;
        }
        let v: toml_edit::Value = match f.kind {
            Kind::Bool(_) => (self.int(i) == 0).into(),
            Kind::Int(choices, ..) => {
                let cur = self.int(i);
                let next = if dir > 0 {
                    let up = choices.iter().copied().find(|&c| c > cur);
                    up.unwrap_or(choices[0])
                } else {
                    let down = choices.iter().rev().copied().find(|&c| c < cur);
                    down.unwrap_or(choices[choices.len() - 1])
                };
                next.into()
            }
        };
        if !self.0.get(f.table).is_some_and(Item::is_table_like) {
            let mut t = toml_edit::Table::new();
            // Comments in a file with no keys yet are the document's trailing
            // text, written after every table: keep them on top.
            let lead = self.0.trailing().as_str().unwrap_or("").to_string();
            if self.0.as_table().is_empty() && !lead.is_empty() {
                let nl = if lead.ends_with('\n') { "" } else { "\n" };
                t.decor_mut().set_prefix(lead + nl);
                self.0.set_trailing("");
            }
            self.0.as_table_mut().insert(f.table, Item::Table(t));
        }
        let root = self.0.as_table_mut();
        if let Some(it) = root.get_mut(f.table).and_then(Item::as_inline_table_mut) {
            if !it.contains_key(f.key) {
                // The last value's space before `}` goes after the new one,
                // not before its comma.
                let mut v = v;
                if let Some((_, last)) = it.iter_mut().last() {
                    let end = last.decor().suffix().and_then(|s| s.as_str());
                    v.decor_mut().set_suffix(end.unwrap_or("").to_string());
                    last.decor_mut().set_suffix("");
                }
                it.insert(f.key, v);
                return true;
            }
        }
        let Some(t) = root.get_mut(f.table).and_then(Item::as_table_like_mut) else {
            return false;
        };
        // Keep a comment after the old value.
        match t.get_mut(f.key).and_then(Item::as_value_mut) {
            Some(old) => {
                let decor = old.decor().clone();
                *old = v;
                *old.decor_mut() = decor;
            }
            None => {
                t.insert(f.key, Item::Value(v));
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_reaches_every_value_both_ways() {
        for (i, f) in FIELDS.iter().enumerate() {
            let n = match f.kind {
                Kind::Bool(_) => 2,
                Kind::Int(c, ..) => c.len(),
            };
            for dir in [1, -1] {
                let mut s = Settings::default();
                let mut seen = std::collections::BTreeSet::new();
                for _ in 0..n {
                    s.step(i, dir);
                    seen.insert(s.show(i));
                }
                assert_eq!(seen.len(), n, "{} dir {dir}", f.key);
                // n steps in one direction come back to the start.
                assert_eq!(s.show(i), Settings::default().show(i), "{}", f.key);
            }
        }
    }

    #[test]
    fn defaults_steps_and_unknown_keys() {
        let mut s = Settings::default();
        assert_eq!(s.show(0), "1280");
        assert_eq!(s.show(2), "Off");
        // The game draws with anisotropy off when the key is missing.
        assert_eq!(s.show(4), "1");
        s.step(0, 1);
        assert_eq!(s.int(0), 1600);
        s.step(0, -1);
        s.step(0, -1);
        assert_eq!(s.int(0), 960);
        s.step(0, -1);
        assert_eq!(s.int(0), 3840, "wraps from the smallest to the largest");
        s.step(0, 1);
        assert_eq!(s.int(0), 960, "and back");
        s.step(2, 1);
        assert_eq!(s.show(2), "On");
        s.step(2, -1);
        assert_eq!(s.show(2), "Off");

        // A value not offered steps to its neighbour; other keys survive.
        let dir = crate::tests::temp("settings");
        let path = dir.join("settings.toml");
        std::fs::write(
            &path,
            "[window]\nwidth = 1000\nmine = 'kept'\n[other]\nx = 1\n",
        )
        .unwrap();
        let mut s = Settings::load(&path).unwrap();
        s.step(0, 1);
        assert_eq!(s.int(0), 1280);
        s.save(&path).unwrap();
        let back = Settings::load(&path).unwrap();
        assert_eq!(back.int(0), 1280);
        assert_eq!(back.0["window"]["mine"].as_str(), Some("kept"));
        assert_eq!(back.0["other"]["x"].as_integer(), Some(1));
        assert!(Settings::load(&dir.join("missing.toml"))
            .unwrap()
            .0
            .is_empty());
    }

    /// A save keeps the player's comments, key order and layout.
    #[test]
    fn save_keeps_comments_and_order() {
        let dir = crate::tests::temp("settings-keep");
        let path = dir.join("settings.toml");
        let text = "# my notes\ntop = 1\n[window]\nwidth = 1920 # wide\n\
                    fullscreen = false\n[draw]\nscale = 2\n[zzz]\nq = 'x'\n";
        std::fs::write(&path, text).unwrap();
        let mut s = Settings::load(&path).unwrap();
        s.step(2, 1);
        s.step(0, 1);
        s.save(&path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            text.replace("1920", "2560")
                .replace("fullscreen = false", "fullscreen = true")
        );
        assert!(!dir.join("settings.toml.new").exists());
        // A new key goes into its table.
        s.step(3, 1);
        assert!(s
            .0
            .to_string()
            .contains("[window]\nwidth = 2560 # wide\nfullscreen = true\nrefresh = 120\n"));
    }

    /// The page shows what the game uses: a value the host does not take
    /// shows the default and is marked.
    #[test]
    fn values_the_game_does_not_take() {
        let load = |text: &str| Settings(text.parse().unwrap());
        let s = load(
            "[window]\nfullscreen = 1\nwidth = 'wide'\nheight = 0\nrefresh = 75\n\
             [draw]\naniso = 3\nforce_linear = 'yes'\nscale = 2.5\n",
        );
        for (i, shown) in [
            (0, "1280"),
            (1, "720"),
            (2, "Off"),
            (4, "1"),
            (5, "Off"),
            (6, "1"),
        ] {
            assert_eq!((s.show(i), s.valid(i)), (shown.into(), false), "{i}");
        }
        // 75 is not offered, but the game takes it.
        assert_eq!((s.show(3), s.valid(3)), ("75".into(), true));
        let s = load("[window]\nfullscreen = true\n[draw]\naniso = 8\n");
        assert_eq!((s.show(2), s.valid(2)), ("On".into(), true));
        assert_eq!((s.show(4), s.valid(4)), ("8".into(), true));
        assert!(Settings::default().valid(0));
        // The drawer takes scale 1 to 4 only.
        let s = load("[draw]\nscale = 8\n");
        assert_eq!((s.show(6), s.valid(6)), ("1".into(), false));
    }

    /// A table that is not a plain table, or a key that is a table, is not
    /// what the game reads: marked, and a step keeps what the player wrote.
    #[test]
    fn tables_that_are_not_tables() {
        for text in [
            "[[window]]\nwidth = 1920\nnote = 'mine'\n",
            "window = 5\n",
            "[window.width]\nx = 1\n",
            "[window]\nwidth = { mine = 1, note = 'keep' }\n",
            "[window]\nwidth = [1920, 1080]\n",
        ] {
            let mut s = Settings(text.parse().unwrap());
            assert_eq!((s.show(0), s.valid(0)), ("1280".into(), false), "{text}");
            assert!(!s.step(0, 1), "{text}");
            assert_eq!(s.0.to_string(), text);
        }
        // An inline table is a table to the game.
        let mut s = Settings("window = { width = 1920 }\n".parse().unwrap());
        assert!(s.valid(0) && s.step(0, 1));
        assert_eq!(s.int(0), 2560);
        // A new key in it goes in its layout.
        assert!(s.step(3, 1));
        assert_eq!(
            s.0.to_string(),
            "window = { width = 2560, refresh = 120 }\n"
        );
    }

    /// A file of comments only keeps them on top when tables are added.
    #[test]
    fn comments_only_stay_on_top() {
        for text in ["# only a comment", "# only a comment\n\n"] {
            let mut s = Settings(text.parse().unwrap());
            s.step(3, 1);
            s.step(4, 1);
            let out = s.0.to_string();
            assert!(out.starts_with("# only a comment\n"), "{out:?}");
            assert_eq!(out.matches("# only").count(), 1, "{out:?}");
            assert!(out.contains("[window]\nrefresh = 120\n"), "{out:?}");
        }
    }

    /// CRLF line endings, a byte order mark and the hidden attribute stay.
    #[test]
    fn save_keeps_file_form() {
        let dir = crate::tests::temp("settings-form");
        let path = dir.join("settings.toml");
        let text = "\u{feff}# my settings\r\n[window]\r\nwidth = 1920 # wide\r\n";
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true);
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::attributes(&mut o, 2);
        std::io::Write::write_all(&mut o.open(&path).unwrap(), text.as_bytes()).unwrap();
        let mut s = Settings::load(&path).unwrap();
        s.step(0, 1);
        s.step(3, 1);
        s.save(&path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            text.replace("1920", "2560") + "refresh = 120\r\n"
        );
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            let a = std::fs::metadata(&path).unwrap().file_attributes();
            assert_ne!(a & 2, 0, "still hidden");
        }
    }

    #[test]
    fn errors_say_what_is_wrong() {
        let dir = crate::tests::temp("settings-err");
        let path = dir.join("settings.toml");
        std::fs::write(&path, "[window]\nwidth = 1920\nheight = \n").unwrap();
        let e = Settings::load(&path).unwrap_err();
        assert!(e.starts_with("TOML parse error at line 3"), "{e}");
        assert!(!e.contains(&*dir.to_string_lossy()), "{e}");
        assert!(!e.contains('\n'), "{e}");

        // A read-only file is not changed, and says why.
        std::fs::write(&path, "[window]\nwidth = 1280\n").unwrap();
        let mut p = std::fs::metadata(&path).unwrap().permissions();
        p.set_readonly(true);
        std::fs::set_permissions(&path, p.clone()).unwrap();
        let mut s = Settings::load(&path).unwrap();
        s.step(0, 1);
        assert_eq!(s.save(&path).unwrap_err(), "the file is read-only");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "[window]\nwidth = 1280\n"
        );
        #[allow(clippy::permissions_set_readonly_false)]
        p.set_readonly(false);
        std::fs::set_permissions(&path, p).unwrap();
    }
}
