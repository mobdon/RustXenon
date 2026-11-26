//! The player's settings files.
//!
//! `<home>\settings.toml` holds the player's defaults; each built game's
//! `run.toml` (beside its `game.exe`, normally `<home>\games\<sha256>\`)
//! holds its paths and overrides:
//!
//! ```toml
//! title = "shown in the launcher"
//! exe = 'C:\path\to\patched executable'
//! game = 'C:\path\to\unpacked game folder'
//! update = '...'
//! data = '...'
//! [window]
//! width = 1280
//! height = 720
//! fullscreen = false
//! refresh = 60
//! [draw]
//! aniso = 16
//! force_linear = false
//! scale = 1
//! ```
//!
//! Every key is optional. `run.toml` wins over `settings.toml`, which wins
//! over the defaults; for `[draw]`, set environment variables
//! (`XENON_ANISO`, `XENON_FORCE_LINEAR`, `XENON_SCALE`) win over both. A bad
//! value or unknown key is a warning and that key is left unset: a settings
//! file never stops the game.

use std::path::{Path, PathBuf};
use toml::{Table, Value};

/// The window and drawer values one file sets; `None` where it says nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layer {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fullscreen: Option<bool>,
    pub refresh: Option<u32>,
    pub aniso: Option<u32>,
    pub force_linear: Option<bool>,
    pub scale: Option<u32>,
}

impl Layer {
    /// This layer's values, and `base`'s where this one has none.
    pub fn over(self, base: Layer) -> Layer {
        Layer {
            width: self.width.or(base.width),
            height: self.height.or(base.height),
            fullscreen: self.fullscreen.or(base.fullscreen),
            refresh: self.refresh.or(base.refresh),
            aniso: self.aniso.or(base.aniso),
            force_linear: self.force_linear.or(base.force_linear),
            scale: self.scale.or(base.scale),
        }
    }

    /// The window to open: these values over the defaults.
    pub fn window(&self) -> xenon_window::Config {
        let d = xenon_window::Config::default();
        xenon_window::Config {
            width: self.width.unwrap_or(d.width),
            height: self.height.unwrap_or(d.height),
            fullscreen: self.fullscreen.unwrap_or(d.fullscreen),
            refresh: self.refresh.unwrap_or(d.refresh),
        }
    }

    /// The drawer's environment variables to set from these values: only
    /// those `is_set` says the player has not set themselves.
    pub fn draw_env(&self, is_set: impl Fn(&str) -> bool) -> Vec<(&'static str, String)> {
        let flag = |b: bool| if b { "1" } else { "0" }.to_string();
        [
            ("XENON_ANISO", self.aniso.map(|v| v.to_string())),
            ("XENON_FORCE_LINEAR", self.force_linear.map(flag)),
            ("XENON_SCALE", self.scale.map(|v| v.to_string())),
        ]
        .into_iter()
        .filter_map(|(k, v)| Some((k, v?)))
        .filter(|(k, _)| !is_set(k))
        .collect()
    }
}

/// One file's contents.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Run {
    pub title: Option<String>,
    pub exe: Option<String>,
    pub game: Option<String>,
    pub update: Option<String>,
    pub data: Option<String>,
    pub layer: Layer,
}

/// Parses one settings file; each problem is pushed to `warn` (prefixed
/// with `name`) and its key left unset.
pub fn parse(name: &str, text: &str, warn: &mut Vec<String>) -> Run {
    let mut out = Run::default();
    let table: Table = match text.parse() {
        Ok(t) => t,
        Err(e) => {
            warn.push(format!("{name}: {e}; using none of it"));
            return out;
        }
    };
    let mut bad = |key: &str, want: &str| warn.push(format!("{name}: {key} is not {want}"));
    for (k, v) in &table {
        let path = match k.as_str() {
            "title" => &mut out.title,
            "exe" => &mut out.exe,
            "game" => &mut out.game,
            "update" => &mut out.update,
            "data" => &mut out.data,
            "window" | "draw" => {
                let Some(t) = v.as_table() else {
                    bad(k, "a table");
                    continue;
                };
                for (sk, sv) in t {
                    let key = format!("{k}.{sk}");
                    let l = &mut out.layer;
                    let (num, flag) = match key.as_str() {
                        "window.width" => (Some(&mut l.width), None),
                        "window.height" => (Some(&mut l.height), None),
                        "window.refresh" => (Some(&mut l.refresh), None),
                        "draw.aniso" => (Some(&mut l.aniso), None),
                        "draw.scale" => (Some(&mut l.scale), None),
                        "window.fullscreen" => (None, Some(&mut l.fullscreen)),
                        "draw.force_linear" => (None, Some(&mut l.force_linear)),
                        _ => {
                            bad(&key, "a known setting");
                            continue;
                        }
                    };
                    if let Some(slot) = num {
                        match sv.as_integer().and_then(|i| u32::try_from(i).ok()) {
                            Some(n) if n > 0 => *slot = Some(n),
                            _ => bad(&key, "a whole number above 0"),
                        }
                    } else if let Some(slot) = flag {
                        match sv.as_bool() {
                            Some(b) => *slot = Some(b),
                            None => bad(&key, "true or false"),
                        }
                    }
                }
                continue;
            }
            _ => {
                bad(k, "a known setting");
                continue;
            }
        };
        match v {
            Value::String(s) => *path = Some(s.clone()),
            _ => bad(k, "a string"),
        }
    }
    out
}

/// The player's folder: `--home` when given; else the folder above
/// `games\<sha256>\` when the program sits there as `game.exe`; else the
/// program's own folder.
pub fn home(arg: Option<&str>, exe: &Path) -> PathBuf {
    if let Some(h) = arg {
        return h.into();
    }
    let dir = exe.parent().unwrap_or(Path::new("."));
    let named = |p: &Path, s: &str| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(s));
    match dir.parent() {
        Some(games) if named(exe, "game.exe") && named(games, "games") => {
            games.parent().unwrap_or(Path::new(".")).into()
        }
        _ => dir.into(),
    }
}

/// Reads `<home>\settings.toml` and the game's `run.toml` (beside the
/// program, else `<home>\games\<sha256>\run.toml`) and merges them. Missing
/// files are fine; problems are pushed to `warn`.
pub fn load(home: &Path, exe_dir: &Path, sha256: &str, warn: &mut Vec<String>) -> Run {
    let mut read = |p: PathBuf| match std::fs::read_to_string(&p) {
        Ok(t) => parse(&p.display().to_string(), &t, warn),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Run::default(),
        Err(e) => {
            warn.push(format!("{}: {e}", p.display()));
            Run::default()
        }
    };
    let settings = home.join("settings.toml");
    let defaults = read(settings.clone());
    let beside = exe_dir.join("run.toml");
    let run_toml = if beside.exists() {
        beside
    } else {
        home.join("games").join(sha256).join("run.toml")
    };
    let mut run = read(run_toml);
    let d = &defaults;
    for (key, v) in [
        ("title", &d.title),
        ("exe", &d.exe),
        ("game", &d.game),
        ("update", &d.update),
        ("data", &d.data),
    ] {
        if v.is_some() {
            warn.push(format!(
                "{}: {key} belongs in a game's run.toml; ignored",
                settings.display()
            ));
        }
    }
    run.layer = run.layer.over(defaults.layer);
    run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_key() {
        let mut w = Vec::new();
        let r = parse(
            "run.toml",
            r#"
            title = "Mine"
            exe = 'C:\x\default.xex'
            game = 'C:\x\game'
            update = 'C:\x\update'
            data = 'C:\x\data'
            [window]
            width = 1920
            height = 1080
            fullscreen = true
            refresh = 144
            [draw]
            aniso = 16
            force_linear = false
            scale = 2
            "#,
            &mut w,
        );
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(r.title.as_deref(), Some("Mine"));
        assert_eq!(r.exe.as_deref(), Some(r"C:\x\default.xex"));
        assert_eq!(r.game.as_deref(), Some(r"C:\x\game"));
        assert_eq!(r.update.as_deref(), Some(r"C:\x\update"));
        assert_eq!(r.data.as_deref(), Some(r"C:\x\data"));
        assert_eq!(
            r.layer,
            Layer {
                width: Some(1920),
                height: Some(1080),
                fullscreen: Some(true),
                refresh: Some(144),
                aniso: Some(16),
                force_linear: Some(false),
                scale: Some(2),
            }
        );
    }

    #[test]
    fn bad_values_warn_and_stay_unset() {
        let mut w = Vec::new();
        let r = parse(
            "settings.toml",
            "exe = 3\nspeed = 2\n[window]\nwidth = -5\nheight = 0\nfullscreen = 'yes'\n\
             refresh = 99999999999\ncolour = 1\n[draw]\naniso = 2.5\nscale = 3\n",
            &mut w,
        );
        assert_eq!(w.len(), 8, "{w:?}");
        assert!(w.iter().all(|m| m.starts_with("settings.toml: ")));
        assert_eq!(r.exe, None);
        assert_eq!(
            r.layer,
            Layer {
                scale: Some(3),
                ..Layer::default()
            }
        );
        let mut w = Vec::new();
        assert_eq!(parse("x", "window = 1", &mut w), Run::default());
        assert_eq!(w.len(), 1);
        // Not TOML at all: nothing used, one warning, no panic.
        let mut w = Vec::new();
        assert_eq!(parse("x", "[window\nwidth = ", &mut w), Run::default());
        assert_eq!(w.len(), 1);
        assert_eq!(parse("x", "", &mut w), Run::default());
    }

    #[test]
    fn run_over_settings_over_defaults_and_env_over_all() {
        let mut w = Vec::new();
        let settings = parse(
            "s",
            "[window]\nwidth = 800\nheight = 600\n[draw]\naniso = 4\nscale = 2\n",
            &mut w,
        );
        let run = parse("r", "[window]\nwidth = 1024\n[draw]\naniso = 8\n", &mut w);
        assert!(w.is_empty());
        let l = run.layer.over(settings.layer);
        let d = xenon_window::Config::default();
        assert_eq!(
            l.window(),
            xenon_window::Config {
                width: 1024,
                height: 600,
                ..d
            }
        );
        assert_eq!(Layer::default().window(), d);
        // XENON_SCALE set by the player: left alone; force_linear unset in
        // both files: not set.
        let env = l.draw_env(|k| k == "XENON_SCALE");
        assert_eq!(env, [("XENON_ANISO", "8".to_string())]);
        let env = Layer {
            force_linear: Some(true),
            scale: Some(2),
            ..Layer::default()
        }
        .draw_env(|_| false);
        assert_eq!(
            env,
            [
                ("XENON_FORCE_LINEAR", "1".to_string()),
                ("XENON_SCALE", "2".to_string())
            ]
        );
    }

    #[test]
    fn home_from_argument_or_program_place() {
        let p = |s: &str| PathBuf::from(s);
        let sha = "a".repeat(64);
        let built = p("h").join("games").join(&sha).join("game.exe");
        assert_eq!(home(Some("elsewhere"), &built), p("elsewhere"));
        assert_eq!(home(None, &built), p("h"));
        assert_eq!(
            home(None, &p("dev").join("target").join("game.exe")),
            p("dev").join("target")
        );
        // Not named game.exe, or not under games: its own folder.
        let other = p("h").join("games").join(&sha).join("other.exe");
        assert_eq!(home(None, &other), p("h").join("games").join(&sha));
        assert_eq!(home(None, &p("game.exe")), p(""));
    }

    #[test]
    fn load_merges_files_and_finds_run_toml() {
        let root = std::env::temp_dir().join(format!("xenon-settings-{}", std::process::id()));
        let sha = "b".repeat(64);
        let built = root.join("games").join(&sha);
        std::fs::create_dir_all(&built).unwrap();
        let mut w = Vec::new();
        // No files: all unset, no warnings.
        assert_eq!(load(&root, &root, &sha, &mut w), Run::default());
        assert!(w.is_empty());
        std::fs::write(
            root.join("settings.toml"),
            "[window]\nwidth = 800\nheight = 600\n",
        )
        .unwrap();
        std::fs::write(built.join("run.toml"), "exe = 'e'\n[window]\nwidth = 640\n").unwrap();
        // The program elsewhere: run.toml found under games\<sha256>.
        let r = load(&root, &root, &sha, &mut w);
        assert_eq!(r.exe.as_deref(), Some("e"));
        assert_eq!((r.layer.width, r.layer.height), (Some(640), Some(600)));
        // A run.toml beside the program wins.
        let dropped = root.join("dropped");
        std::fs::create_dir_all(&dropped).unwrap();
        std::fs::write(dropped.join("run.toml"), "game = 'g'\n").unwrap();
        let r = load(&root, &dropped, &sha, &mut w);
        assert_eq!((r.exe, r.game.as_deref()), (None, Some("g")));
        assert_eq!(r.layer.width, Some(800));
        assert!(w.is_empty(), "{w:?}");
        // A game's paths in settings.toml: a warning, and not used.
        std::fs::write(root.join("settings.toml"), "exe = 'x'\n").unwrap();
        let r = load(&root, &dropped, &sha, &mut w);
        assert_eq!(r.exe, None);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("exe belongs in a game's run.toml"), "{w:?}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
