//! Bakes the source commit into the host as `XENON_BUILD`, so every run log
//! (and the window title) says which build produced it, and the per-game
//! hook files (`games/<sha256>/hooks.toml`) into `games.rs`, so a built game
//! finds its run-time hooks without the repository.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let id = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(c) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            format!("{c}{}", if dirty { "-dirty" } else { "" })
        }
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=XENON_BUILD={id}");
    hook_files();
    // Rebuild the id when HEAD moves or tracked sources change (edits
    // before `git add` touch no git file). --git-path also resolves linked
    // worktrees; a missing path would make cargo rerun every build.
    for f in ["HEAD", "index", "packed-refs"] {
        rerun(git(&["rev-parse", "--git-path", f]));
    }
    if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
        rerun(git(&["rev-parse", "--git-path", &r]));
    }
    if let Some(top) = git(&["rev-parse", "--show-toplevel"]) {
        for d in ["crates", "tools"] {
            rerun(Some(format!("{top}/{d}")));
        }
    }
}

fn rerun(path: Option<String>) {
    if let Some(p) = path.filter(|p| std::path::Path::new(p).exists()) {
        let p = std::fs::canonicalize(&p).map_or(p, |c| c.display().to_string());
        println!("cargo:rerun-if-changed={p}");
    }
}

/// Writes `$OUT_DIR/games.rs`: `GAMES`, each hook file's SHA-256 (its folder
/// name) and text.
fn hook_files() {
    let games = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../games");
    println!("cargo:rerun-if-changed={}", games.display());
    let mut found: Vec<(String, String)> = std::fs::read_dir(&games)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let f = e.path().join("hooks.toml");
            println!("cargo:rerun-if-changed={}", f.display());
            let text = std::fs::read_to_string(&f).ok()?;
            Some((e.file_name().to_string_lossy().into_owned(), text))
        })
        .collect();
    found.sort();
    let rows: String = found
        .iter()
        .map(|(sha, text)| format!("    ({sha:?}, {text:?}),\n"))
        .collect();
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("games.rs");
    let text = format!("pub static GAMES: &[(&str, &str)] = &[\n{rows}];\n");
    std::fs::write(out, text).unwrap();
}
