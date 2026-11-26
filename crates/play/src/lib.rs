//! `xenon-play`: the launcher's headless part. Finds built game folders,
//! keeps the recent list, builds the game's command line, edits the player's
//! default settings, and draws the launcher screen into a pixel buffer.
//!
//! On-disk layout, all under the launcher's home folder:
//!
//! ```text
//! rustxenon-play.exe        the launcher
//! settings.toml             player defaults ([window] and [draw])
//! recent.txt                one built game folder per line, newest first
//! games/<sha256>/game.exe   a built game program
//! games/<sha256>/run.toml   its run settings (title, exe, game, update, data)
//! ```

pub mod draw;
mod font;
pub mod settings;

use serde::Deserialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Most folders kept in `recent.txt`.
pub const RECENT_MAX: usize = 20;

/// One built game folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub dir: PathBuf,
    pub title: String,
    pub exe: String,
    pub game: String,
    pub update: Option<String>,
    pub data: Option<String>,
    /// Listed in `recent.txt`.
    pub recent: bool,
}

/// The `run.toml` keys the launcher reads; the rest are the game's.
#[derive(Deserialize)]
struct RunToml {
    title: Option<String>,
    exe: String,
    game: String,
    update: Option<String>,
    data: Option<String>,
}

/// Reads a built game folder, or says why it is not one.
pub fn load_game(dir: &Path) -> Result<Game, String> {
    if !dir.is_dir() {
        return Err(format!("not a folder: {}", dir.display()));
    }
    if !dir.join("game.exe").is_file() {
        return Err(format!("no game.exe in {}", dir.display()));
    }
    let run = dir.join("run.toml");
    let text =
        std::fs::read_to_string(&run).map_err(|_| format!("no run.toml in {}", dir.display()))?;
    let r: RunToml = toml::from_str(&text).map_err(|e| {
        let first = e.to_string();
        let first = first.lines().next().unwrap_or("").to_string();
        format!("{}: {first}", run.display())
    })?;
    let title = r.title.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| {
        // No title chosen: the unpacked game folder's name.
        // Split by hand: run.toml holds Windows paths on any system.
        let seps = ['\\', '/'];
        let name = r.game.trim_end_matches(seps).rsplit(seps).next();
        match name.filter(|n| !n.is_empty() && !n.ends_with(':')) {
            Some(n) => n.to_string(),
            None => dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        }
    });
    Ok(Game {
        dir: dir.to_path_buf(),
        title,
        exe: r.exe,
        game: r.game,
        update: r.update,
        data: r.data,
        recent: false,
    })
}

/// Every built game folder under `<home>/games`.
pub fn discover(home: &Path) -> Vec<Game> {
    let Ok(rd) = std::fs::read_dir(home.join("games")) else {
        return Vec::new();
    };
    rd.flatten()
        .filter_map(|e| load_game(&e.path()).ok())
        .collect()
}

/// A path compared loosely: separators, a trailing separator and letter case
/// ignored. shortcut: case is ignored on every system (Windows is the only
/// one played on), compare exactly if another system ever is.
fn key(p: &Path) -> String {
    p.to_string_lossy()
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_lowercase()
}

/// `recent.txt`, keeping only lines that are still built game folders.
pub fn load_recent(home: &Path) -> Vec<PathBuf> {
    let text = std::fs::read_to_string(home.join("recent.txt")).unwrap_or_default();
    let list: Vec<PathBuf> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        // A trailing separator would show in the list.
        .map(|l| PathBuf::from(l.trim_end_matches(['\\', '/'])))
        .filter(|p| load_game(p).is_ok())
        .collect();
    dedupe(list)
}

fn dedupe(list: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<PathBuf> = list.into_iter().filter(|p| seen.insert(key(p))).collect();
    out.truncate(RECENT_MAX);
    out
}

/// `dir` moved to the front of `list`, once.
pub fn add_recent(list: &[PathBuf], dir: &Path) -> Vec<PathBuf> {
    dedupe(
        std::iter::once(dir.to_path_buf())
            .chain(list.iter().cloned())
            .collect(),
    )
}

pub fn save_recent(home: &Path, list: &[PathBuf]) -> std::io::Result<()> {
    let mut text = String::new();
    for p in list {
        text.push_str(&p.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(home.join("recent.txt"), text)
}

/// The games to list: recent ones first (newest first), then every other
/// folder under `<home>/games` by title.
pub fn merged(home: &Path) -> Vec<Game> {
    let recent = load_recent(home);
    let mut out: Vec<Game> = recent
        .iter()
        .filter_map(|p| load_game(p).ok())
        .map(|g| Game { recent: true, ..g })
        .collect();
    let keys: std::collections::HashSet<String> = recent.iter().map(|p| key(p)).collect();
    let mut rest: Vec<Game> = discover(home)
        .into_iter()
        .filter(|g| !keys.contains(&key(&g.dir)))
        .collect();
    rest.sort_by_cached_key(|g| (g.title.to_lowercase(), key(&g.dir)));
    out.extend(rest);
    out
}

/// The folder a dropped path means: the folder itself, or the folder of a
/// dropped `game.exe` or `run.toml`.
pub fn dropped_folder(p: &Path) -> PathBuf {
    let is_part = p.is_file()
        && p.file_name().is_some_and(|n| {
            let n = n.to_string_lossy().to_lowercase();
            n == "game.exe" || n == "run.toml"
        });
    match p.parent() {
        Some(parent) if is_part => parent.to_path_buf(),
        _ => p.to_path_buf(),
    }
}

/// The game program's arguments: `--window --home <home> --exe <exe>
/// --game <game> [--update <u>] [--data <d>]`.
pub fn args(home: &Path, g: &Game) -> Vec<OsString> {
    let mut a: Vec<OsString> = vec![
        "--window".into(),
        "--home".into(),
        home.into(),
        "--exe".into(),
        (&g.exe).into(),
        "--game".into(),
        (&g.game).into(),
    ];
    for (flag, v) in [("--update", &g.update), ("--data", &g.data)] {
        if let Some(v) = v {
            a.push(flag.into());
            a.push(v.into());
        }
    }
    a
}

/// The game program started from its own folder.
pub fn command(home: &Path, g: &Game) -> std::process::Command {
    let mut c = std::process::Command::new(g.dir.join("game.exe"));
    c.args(args(home, g)).current_dir(&g.dir);
    #[cfg(windows)]
    {
        // No console window flashing up if the game is a console program.
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A fresh empty folder under the system temp folder.
    pub fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xenon-play-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub fn make_game(dir: &Path, run: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("game.exe"), b"").unwrap();
        std::fs::write(dir.join("run.toml"), run).unwrap();
    }

    #[test]
    fn reads_run_toml() {
        let home = temp("read");
        let d = home.join("games").join("aa");
        make_game(
            &d,
            "title = \"Test Game One\"\nexe = 'C:\\x\\a.xex'\ngame = 'C:\\x\\Unpacked'\n\
             data = 'D'\n[window]\nwidth = 1280\n",
        );
        let g = load_game(&d).unwrap();
        assert_eq!(g.title, "Test Game One");
        assert_eq!(g.exe, "C:\\x\\a.xex");
        assert_eq!((g.update, g.data.as_deref()), (None, Some("D")));
        // No title: the game folder's name.
        make_game(&d, "exe = 'a'\ngame = 'C:\\y\\Folder Name'\n");
        assert_eq!(load_game(&d).unwrap().title, "Folder Name");
        // Missing keys and missing files say why.
        make_game(&d, "title = 'x'\n");
        assert!(load_game(&d).unwrap_err().contains("run.toml"));
        std::fs::remove_file(d.join("game.exe")).unwrap();
        assert!(load_game(&d).unwrap_err().contains("game.exe"));
        assert!(load_game(&home.join("nope"))
            .unwrap_err()
            .contains("not a folder"));
    }

    #[test]
    fn recent_and_merged() {
        let home = temp("merge");
        let g = home.join("games");
        make_game(&g.join("1"), "title='Zeta'\nexe='e'\ngame='g'\n");
        make_game(&g.join("2"), "title='alpha'\nexe='e'\ngame='g'\n");
        make_game(&g.join("3"), "title='Mid'\nexe='e'\ngame='g'\n");
        let elsewhere = home.join("dropped");
        make_game(&elsewhere, "title='Dropped'\nexe='e'\ngame='g'\n");
        std::fs::create_dir_all(g.join("not-a-game")).unwrap();

        // Add to front, dedupe (loosely: case and trailing separator).
        let r = add_recent(&[], &g.join("1"));
        let r = add_recent(&r, &elsewhere);
        // Case differs only where the file system ignores it.
        let one = g.join("1").to_string_lossy().into_owned();
        let one = if cfg!(windows) {
            one.to_uppercase()
        } else {
            one
        };
        let same = PathBuf::from(format!("{one}{}", std::path::MAIN_SEPARATOR));
        let r = add_recent(&r, &same);
        assert_eq!(r, [same.clone(), elsewhere.clone()]);
        // A line that is no longer a game is dropped on load.
        let mut lines = r.clone();
        lines.push(home.join("gone"));
        save_recent(&home, &lines).unwrap();
        assert_eq!(load_recent(&home), r);

        let titles: Vec<_> = merged(&home)
            .into_iter()
            .map(|g| (g.title, g.recent))
            .collect();
        assert_eq!(
            titles,
            [
                ("Zeta".into(), true),
                ("Dropped".into(), true),
                ("alpha".into(), false),
                ("Mid".into(), false)
            ]
        );
        // The cap.
        let mut r = Vec::new();
        for i in 0..RECENT_MAX + 5 {
            r = add_recent(&r, Path::new(&format!("C:\\{i}")));
        }
        assert_eq!(r.len(), RECENT_MAX);
        assert_eq!(r[0], Path::new(&format!("C:\\{}", RECENT_MAX + 4)));
    }

    #[test]
    fn command_line() {
        let mut g = Game {
            dir: PathBuf::from("F"),
            title: "t".into(),
            exe: "E x".into(),
            game: "G".into(),
            update: None,
            data: None,
            recent: false,
        };
        let s = |a: Vec<OsString>| {
            a.into_iter()
                .map(|a| a.into_string().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            s(args(Path::new("H"), &g)),
            ["--window", "--home", "H", "--exe", "E x", "--game", "G"]
        );
        g.update = Some("U".into());
        g.data = Some("D".into());
        assert_eq!(
            s(args(Path::new("H"), &g))[7..],
            ["--update", "U", "--data", "D"]
        );
        let c = command(Path::new("H"), &g);
        assert_eq!(c.get_program(), Path::new("F").join("game.exe"));
    }

    #[test]
    fn dropped_paths() {
        let home = temp("drop");
        make_game(&home, "exe='e'\ngame='g'\n");
        assert_eq!(dropped_folder(&home.join("game.exe")), home);
        assert_eq!(dropped_folder(&home.join("run.toml")), home);
        assert_eq!(dropped_folder(&home), home);
    }
}
