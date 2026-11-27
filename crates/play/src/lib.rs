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

/// Reads a built game folder, or says why it is not one (without the path:
/// the caller names the folder as it fits).
pub fn load_game(dir: &Path) -> Result<Game, String> {
    if !dir.is_dir() {
        return Err("no such folder".into());
    }
    if !dir.join("game.exe").is_file() {
        return Err("no game.exe in it".into());
    }
    let run = dir.join("run.toml");
    let text = read_text(&run).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => "no run.toml in it".into(),
        _ if run.is_dir() => "run.toml: it is a folder, not a file".into(),
        _ => format!("run.toml: {e}"),
    })?;
    let r: RunToml =
        toml::from_str(&text).map_err(|e| format!("run.toml: {}", settings::toml_error(&e)))?;
    let title = r.title.map(|t| one_line(&t)).filter(|t| !t.is_empty());
    let title = title.unwrap_or_else(|| {
        // No title chosen: the unpacked game folder's name.
        // Split by hand: run.toml holds Windows paths on any system.
        let seps = ['\\', '/'];
        let name = r.game.trim_end_matches(seps).rsplit(seps).next();
        match name.filter(|n| !n.is_empty() && !n.ends_with(':')) {
            Some(n) => one_line(n),
            None => one_line(&dir.file_name().unwrap_or_default().to_string_lossy()),
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

/// A text file a player may have saved from Notepad: UTF-8 with or without
/// a byte order mark, or UTF-16 with one. Anything else is an
/// `InvalidData` error that says how to fix it.
fn read_text(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let utf16 = |b: &[u8], from: fn([u8; 2]) -> u16| {
        let units: Vec<u16> = b.chunks_exact(2).map(|c| from([c[0], c[1]])).collect();
        (b.len() % 2 == 0)
            .then(|| String::from_utf16(&units).ok())
            .flatten()
    };
    let text = match bytes.as_slice() {
        [0xff, 0xfe, rest @ ..] => utf16(rest, u16::from_le_bytes),
        [0xfe, 0xff, rest @ ..] => utf16(rest, u16::from_be_bytes),
        b => std::str::from_utf8(b)
            .ok()
            .map(|s| s.trim_start_matches('\u{feff}').into()),
    };
    text.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "it is not UTF-8 text; save it as UTF-8",
        )
    })
}

/// A title on one line: line breaks, tabs and other control characters
/// become spaces, invisible format marks (zero-width space, direction marks)
/// go, and a run of spaces is one.
fn one_line(s: &str) -> String {
    let invisible = |c: char| matches!(c, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{2069}' | '\u{feff}');
    let s: String = s.chars().filter(|&c| !invisible(c)).collect();
    s.split(|c: char| c.is_whitespace() || c.is_control())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a load error means the folder is a game whose run.toml is broken
/// (the player is told), not a folder that is gone or not a game.
fn broken(e: &str) -> bool {
    e.starts_with("run.toml:")
}

/// Every built game folder under `<home>/games`, and each folder there whose
/// run.toml is broken with why.
pub fn discover(home: &Path) -> (Vec<Game>, Vec<(PathBuf, String)>) {
    let (mut games, mut bad) = (Vec::new(), Vec::new());
    let Ok(rd) = std::fs::read_dir(home.join("games")) else {
        return (games, bad);
    };
    for e in rd.flatten() {
        match load_game(&e.path()) {
            Ok(g) => games.push(g),
            Err(why) if broken(&why) => bad.push((e.path(), why)),
            Err(_) => {}
        }
    }
    (games, bad)
}

/// "folder: why", by the folder's name: the whole path would not fit.
fn bad_line(dir: &Path, why: &str) -> String {
    let name = dir.file_name().unwrap_or(dir.as_os_str());
    format!("{}: {why}", name.to_string_lossy())
}

/// A path compared loosely: separators, a trailing separator, letter case,
/// a `\\?\` prefix and `.` and `..` parts ignored. shortcut: case is ignored
/// on every system (Windows is the only one played on), compare exactly if
/// another system ever is.
fn key(p: &Path) -> String {
    let s = p.to_string_lossy().replace('/', "\\").to_lowercase();
    let s = match s.strip_prefix(r"\\?\") {
        Some(rest) => match rest.strip_prefix(r"unc\") {
            Some(unc) => format!(r"\\{unc}"),
            None => rest.to_string(),
        },
        None => s,
    };
    let lead = s.len() - s.trim_start_matches('\\').len();
    let mut parts: Vec<&str> = Vec::new();
    for part in s.split('\\') {
        match part {
            "" | "." => {}
            // Never above the drive, or a UNC path's server and share.
            ".." if parts.len() > if lead >= 2 { 2 } else { 1 } => {
                parts.pop();
            }
            ".." => {}
            p => parts.push(p),
        }
    }
    format!("{}{}", "\\".repeat(lead), parts.join("\\"))
}

/// `recent.txt`. Lines that are not built game folders now are kept (the
/// drive may be unplugged): [`merged`] leaves them out. A relative line is
/// under `home`. A missing file is an empty list; a file that cannot be
/// read says why, and must not be overwritten (its lines would be lost).
pub fn load_recent(home: &Path) -> Result<Vec<PathBuf>, String> {
    let text = match read_text(&home.join("recent.txt")) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.to_string()),
    };
    let list: Vec<PathBuf> = text
        .lines()
        // Explorer's "Copy as path" puts the path in quotes.
        .map(|l| l.trim().trim_matches('"').trim())
        .filter(|l| !l.is_empty())
        // A trailing separator would show in the list.
        .map(|l| home.join(l.trim_end_matches(['\\', '/'])))
        .collect();
    Ok(dedupe(list))
}

/// The launcher window's first size in logical pixels: 1280x720, or 16:9
/// and smaller so it fits a `screen` of that logical size with room for the
/// title bar and the taskbar. Under [`MIN_WINDOW`] only when the screen is
/// too small for that.
pub fn window_size(screen: (f64, f64)) -> (f64, f64) {
    let w = (screen.0 * 0.9).min((screen.1 - 120.0) * 16.0 / 9.0);
    let w = w.clamp(320.0, 1280.0).round();
    (w, (w * 9.0 / 16.0).round())
}

/// The smallest launcher window, in logical pixels: below this the small
/// text is not readable. A screen too small for it gets a smaller window
/// ([`window_size`]), and then that is the smallest.
pub const MIN_WINDOW: (f64, f64) = (768.0, 432.0);

fn dedupe(list: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    list.into_iter().filter(|p| seen.insert(key(p))).collect()
}

/// `dir` moved to the front of `list`, once, and the list cut to
/// [`RECENT_MAX`]. The oldest lines not in `present` (the games listed now)
/// are cut first, so kept offline lines never push out a game that is there.
pub fn add_recent(list: &[PathBuf], dir: &Path, present: &[Game]) -> Vec<PathBuf> {
    let out = dedupe(
        std::iter::once(dir.to_path_buf())
            .chain(list.iter().cloned())
            .collect(),
    );
    let here: std::collections::HashSet<String> = present.iter().map(|g| key(&g.dir)).collect();
    let mut extra = out.len().saturating_sub(RECENT_MAX);
    let mut keep = vec![true; out.len()];
    for i in (1..out.len()).rev() {
        if extra == 0 {
            break;
        }
        if !here.contains(&key(&out[i])) {
            keep[i] = false;
            extra -= 1;
        }
    }
    let mut out: Vec<PathBuf> = out
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect();
    out.truncate(RECENT_MAX);
    out
}

pub fn save_recent(home: &Path, list: &[PathBuf]) -> std::io::Result<()> {
    let mut text = String::new();
    for p in list {
        text.push_str(&p.to_string_lossy());
        text.push('\n');
    }
    std::fs::write(home.join("recent.txt"), text)
}

/// How long the recent folders get to answer, all together. A folder on a
/// network share that does not answer (an unplugged NAS) holds Windows for
/// about 30 s; it is left out like an unplugged drive until the next refresh.
const RECENT_WAIT: std::time::Duration = std::time::Duration::from_millis(1500);

/// Each folder read on its own thread, so a dead one costs at most
/// [`RECENT_WAIT`]. `None` for a folder that did not answer in time.
fn load_all(list: &[PathBuf]) -> Vec<Option<Result<Game, String>>> {
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, p) in list.iter().enumerate() {
        let (tx, p) = (tx.clone(), p.clone());
        // A thread that hangs is left to finish on its own.
        let _ = std::thread::Builder::new()
            .stack_size(256 << 10)
            .spawn(move || {
                let _ = tx.send((i, load_game(&p)));
            });
    }
    drop(tx);
    let mut out = vec![None; list.len()];
    let end = std::time::Instant::now() + RECENT_WAIT;
    let left = || end.saturating_duration_since(std::time::Instant::now());
    while let Ok((i, r)) = rx.recv_timeout(left()) {
        out[i] = Some(r);
    }
    out
}

/// The games to list: recent ones first (newest first), then every other
/// folder under `<home>/games` by title. Also a "folder: why" line for each
/// game folder left out because its run.toml is broken. `recent` is
/// [`load_recent`]'s list.
pub fn merged(home: &Path, recent: &[PathBuf]) -> (Vec<Game>, Vec<String>) {
    let mut out = Vec::new();
    let mut bad = Vec::new();
    // Only the lines that answered: a slow folder under <home>/games (a disk
    // spinning up) is still listed below, once it is waited for.
    let mut keys = std::collections::HashSet::new();
    for (p, r) in recent.iter().zip(load_all(recent)) {
        match r {
            Some(Ok(g)) => out.push(Game { recent: true, ..g }),
            Some(Err(why)) if broken(&why) => bad.push(bad_line(p, &why)),
            Some(Err(_)) => {}
            None => continue,
        }
        keys.insert(key(p));
    }
    let (found, here_bad) = discover(home);
    let mut rest: Vec<Game> = found
        .into_iter()
        .filter(|g| !keys.contains(&key(&g.dir)))
        .collect();
    rest.sort_by_cached_key(|g| (g.title.to_lowercase(), key(&g.dir)));
    out.extend(rest);
    let here_bad = here_bad.iter().filter(|(d, _)| !keys.contains(&key(d)));
    bad.extend(here_bad.map(|(d, why)| bad_line(d, why)));
    (out, bad)
}

/// Whether the game in `dir` is running now, started by any launcher (one
/// closed since, or another on the same home): Windows does not let a
/// running program's file be opened for writing.
pub fn is_running(dir: &Path) -> bool {
    #[cfg(windows)]
    {
        // Opening for writing without writing changes nothing.
        let exe = dir.join("game.exe");
        match std::fs::OpenOptions::new().write(true).open(&exe) {
            // ERROR_SHARING_VIOLATION.
            Err(e) if e.raw_os_error() == Some(32) => true,
            // ERROR_ACCESS_DENIED: a read-only file, or one the player may
            // not write, is refused before the running check.
            Err(e) if e.raw_os_error() == Some(5) => image_running(&exe),
            _ => false,
        }
    }
    #[cfg(not(windows))]
    {
        // shortcut: no check off Windows, add one if the launcher runs there.
        let _ = dir;
        false
    }
}

/// Whether a running process's program file is `exe`, by its path.
/// shortcut: another path to the same file (a hard link, a `subst` drive, an
/// 8.3 name) is not seen; compare file ids if a player ever starts games so.
#[cfg(windows)]
fn image_running(exe: &Path) -> bool {
    use std::os::windows::ffi::OsStringExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn K32EnumProcesses(ids: *mut u32, size: u32, needed: *mut u32) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> isize;
        fn QueryFullProcessImageNameW(h: isize, flags: u32, name: *mut u16, size: *mut u32) -> i32;
        fn CloseHandle(h: isize) -> i32;
    }
    let mut ids = vec![0u32; 1 << 14];
    let mut needed = 0;
    // SAFETY: `ids` is a buffer of the size passed, `needed` a valid u32.
    if unsafe { K32EnumProcesses(ids.as_mut_ptr(), (ids.len() * 4) as u32, &mut needed) } == 0 {
        return false;
    }
    ids.truncate(needed as usize / 4);
    let want = key(exe);
    ids.into_iter().any(|pid| {
        // PROCESS_QUERY_LIMITED_INFORMATION.
        // SAFETY: a plain call; the handle is closed below.
        let h = unsafe { OpenProcess(0x1000, 0, pid) };
        if h == 0 {
            return false;
        }
        let mut name = [0u16; 1024];
        let mut n = name.len() as u32;
        // SAFETY: `name` holds `n` units, as passed.
        let ok = unsafe { QueryFullProcessImageNameW(h, 0, name.as_mut_ptr(), &mut n) } != 0;
        // SAFETY: `h` is an open handle from OpenProcess.
        unsafe { CloseHandle(h) };
        let name = OsString::from_wide(&name[..(n as usize).min(name.len())]);
        ok && key(Path::new(&name)) == want
    })
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
            .contains("no such folder"));
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
        let r = add_recent(&[], &g.join("1"), &[]);
        let r = add_recent(&r, &elsewhere, &[]);
        // Case differs only where the file system ignores it.
        let one = g.join("1").to_string_lossy().into_owned();
        let one = if cfg!(windows) {
            one.to_uppercase()
        } else {
            one
        };
        let same = PathBuf::from(format!("{one}{}", std::path::MAIN_SEPARATOR));
        let r = add_recent(&r, &same, &[]);
        assert_eq!(r, [same.clone(), elsewhere.clone()]);
        // A \\?\ prefix and . and .. parts name the same folder.
        let one = g.join("1").to_string_lossy().into_owned();
        for alias in [format!(r"\\?\{one}"), format!(r"{one}\..\1\."), one.clone()] {
            assert_eq!(key(Path::new(&alias)), key(&g.join("1")), "{alias}");
        }
        assert_eq!(
            key(Path::new(r"\\?\UNC\nas\g\x")),
            key(Path::new(r"\\nas\g\x"))
        );
        assert_eq!(key(Path::new(r"C:\..\x")), key(Path::new(r"C:\x")));
        // A line that is not a game now is kept (its drive may be
        // unplugged), but not listed.
        let mut lines = r.clone();
        lines.push(home.join("gone"));
        save_recent(&home, &lines).unwrap();
        assert_eq!(load_recent(&home).unwrap(), lines);

        let titles: Vec<_> = merged(&home, &load_recent(&home).unwrap())
            .0
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
        // A byte order mark from a text editor, and a line under home.
        std::fs::write(
            home.join("recent.txt"),
            format!("\u{feff}{}\r\ngames/2\r\n", elsewhere.display()),
        )
        .unwrap();
        assert_eq!(
            load_recent(&home).unwrap(),
            [elsewhere.clone(), g.join("2")]
        );
        // A line in quotes, as Explorer's "Copy as path" gives it.
        std::fs::write(
            home.join("recent.txt"),
            format!("\"{}\"\n", elsewhere.display()),
        )
        .unwrap();
        assert_eq!(
            load_recent(&home).unwrap(),
            std::slice::from_ref(&elsewhere)
        );
        // Notepad's "Unicode" (UTF-16 with a byte order mark).
        let utf16 = |s: &str| {
            let mut b = vec![0xff, 0xfe];
            b.extend(s.encode_utf16().flat_map(u16::to_le_bytes));
            b
        };
        std::fs::write(home.join("recent.txt"), utf16("games/3\r\ngames/1\r\n")).unwrap();
        assert_eq!(load_recent(&home).unwrap(), [g.join("3"), g.join("1")]);
        // Anything else (Notepad's "ANSI") is not read, so it is not
        // overwritten either.
        std::fs::write(home.join("recent.txt"), b"games\\1\r\nE:\\Sp\xe8le\r\n").unwrap();
        let e = load_recent(&home).unwrap_err();
        assert!(e.contains("not UTF-8"), "{e}");
        // `..` stays inside a UNC share.
        assert_ne!(
            key(Path::new(r"\\nas\share\..\other\g")),
            key(Path::new(r"\\nas\other\g"))
        );
        assert_eq!(
            key(Path::new(r"\\nas\share\..\..\g")),
            key(Path::new(r"\\nas\share\g"))
        );
        // The cap.
        let mut r = Vec::new();
        for i in 0..RECENT_MAX + 5 {
            r = add_recent(&r, Path::new(&format!("C:\\{i}")), &[]);
        }
        assert_eq!(r.len(), RECENT_MAX);
        assert_eq!(r[0], Path::new(&format!("C:\\{}", RECENT_MAX + 4)));
    }

    /// Many offline lines before a game that is there: it is still recent,
    /// and starting another game cuts the offline lines, not it.
    #[test]
    fn offline_lines_do_not_crowd_out() {
        let home = temp("crowd");
        let (here, other) = (home.join("here"), home.join("other"));
        make_game(&here, "title='Here'\nexe='e'\ngame='g'\n");
        make_game(&other, "title='Other'\nexe='e'\ngame='g'\n");
        let mut lines: Vec<PathBuf> = (0..300).map(|i| home.join(format!("gone{i}"))).collect();
        lines.push(here.clone());
        save_recent(&home, &lines).unwrap();
        let (games, bad) = merged(&home, &load_recent(&home).unwrap());
        assert_eq!(games.len(), 1);
        assert!(games[0].recent && bad.is_empty());
        let r = add_recent(&load_recent(&home).unwrap(), &other, &games);
        assert_eq!(r.len(), RECENT_MAX);
        assert_eq!((&r[0], r.last()), (&other, Some(&here)));
        assert_eq!(r[1], home.join("gone0"), "the newest offline lines stay");
    }

    /// A broken run.toml is named with its reason, a folder that is not a
    /// game at all is not.
    #[test]
    fn broken_run_toml_is_named() {
        let home = temp("broken");
        let g = home.join("games");
        make_game(&g.join("ok"), "title='Fine'\nexe='e'\ngame='g'\n");
        make_game(&g.join("num"), "title = 42\nexe='e'\ngame='g'\n");
        std::fs::create_dir_all(g.join("empty")).unwrap();
        save_recent(&home, &[g.join("num")]).unwrap();
        let (games, bad) = merged(&home, &load_recent(&home).unwrap());
        assert_eq!(games.len(), 1);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].starts_with("num: run.toml: TOML parse error at line 1"));
        assert!(bad[0].contains("invalid type: integer"), "{}", bad[0]);

        // A run.toml that is not UTF-8, or is a folder, is named too; UTF-16
        // with a byte order mark is read.
        let g = temp("broken-form").join("games");
        make_game(&g.join("ansi"), "");
        std::fs::write(
            g.join("ansi/run.toml"),
            b"title='Caf\xe9'\nexe='e'\ngame='g'\n",
        )
        .unwrap();
        make_game(&g.join("utf16"), "");
        let mut b = vec![0xff, 0xfe];
        b.extend(
            "title='Caf\u{e9}'\nexe='e'\ngame='g'\n"
                .encode_utf16()
                .flat_map(u16::to_le_bytes),
        );
        std::fs::write(g.join("utf16/run.toml"), b).unwrap();
        make_game(&g.join("dir"), "");
        std::fs::remove_file(g.join("dir/run.toml")).unwrap();
        std::fs::create_dir(g.join("dir/run.toml")).unwrap();
        let (games, mut bad) = discover(g.parent().unwrap());
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].title, "Caf\u{e9}");
        bad.sort();
        let why: Vec<_> = bad.iter().map(|(_, why)| why.as_str()).collect();
        assert_eq!(
            why,
            [
                "run.toml: it is not UTF-8 text; save it as UTF-8",
                "run.toml: it is a folder, not a file"
            ]
        );
    }

    #[test]
    fn titles_on_one_line() {
        assert_eq!(
            one_line("First Line\nSecond\r\nThird\tx\u{200b}y\u{202e} "),
            "First Line Second Third xy"
        );
        let home = temp("title");
        make_game(&home, "title = '\u{200b}\t'\nexe='e'\ngame='C:\\a\\Name'\n");
        assert_eq!(load_game(&home).unwrap().title, "Name");
    }

    /// A program that runs is seen as running; a plain file is not.
    #[test]
    fn running_game_is_seen() {
        let home = temp("running");
        make_game(&home, "exe='e'\ngame='g'\n");
        assert!(!is_running(&home));
        // This test program is running: a game.exe that is another name for it
        // (a hard link needs the same volume; the temp folder usually is).
        let dir = temp("running-me");
        let me = std::env::current_exe().unwrap();
        if cfg!(windows) && std::fs::hard_link(me, dir.join("game.exe")).is_ok() {
            assert!(is_running(&dir));
        }
    }

    /// Stands in for a game that runs, for the read-only test below.
    #[test]
    #[ignore]
    fn sleeper() {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }

    /// A game.exe the player may not write (read-only, or under a folder
    /// they do not own) is still seen as running.
    #[cfg(windows)]
    #[test]
    fn running_read_only_game_is_seen() {
        let dir = temp("running-ro");
        let exe = dir.join("game.exe");
        std::fs::copy(std::env::current_exe().unwrap(), &exe).unwrap();
        let mut p = std::fs::metadata(&exe).unwrap().permissions();
        p.set_readonly(true);
        std::fs::set_permissions(&exe, p.clone()).unwrap();
        assert!(!is_running(&dir));
        let mut child = std::process::Command::new(&exe)
            .args(["--ignored", "--exact", "tests::sleeper"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let seen = is_running(&dir);
        let _ = child.kill();
        let _ = child.wait();
        #[allow(clippy::permissions_set_readonly_false)]
        p.set_readonly(false);
        let _ = std::fs::set_permissions(&exe, p);
        assert!(seen);
    }

    #[test]
    fn window_fits_the_screen() {
        assert_eq!(window_size((2560.0, 1440.0)), (1280.0, 720.0));
        // 100% on 1366x768, 150% on 1920x1080: room for the bars.
        for screen in [(1366.0, 768.0), (1280.0, 720.0)] {
            let (w, h) = window_size(screen);
            assert!(h + 120.0 <= screen.1 && w <= screen.0, "{screen:?}");
            assert_eq!((w * 9.0 / 16.0).round(), h);
        }
        // Screens too small for the smallest window: 1280x800 and 1280x720
        // at 150%, 1920x1080 at 200%. The window shrinks to fit.
        for screen in [(853.3, 533.3), (853.3, 480.0), (960.0, 540.0)] {
            let (w, h) = window_size(screen);
            assert!(h + 120.0 <= screen.1 && w <= screen.0, "{screen:?}");
        }
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
