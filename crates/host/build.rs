//! Bakes the source commit into the host as `XENON_BUILD`, so every run log
//! (and the window title) says which build produced it.

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
    // Rebuild the id when HEAD moves or tracked files change.
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        for f in ["HEAD", "index", "packed-refs"] {
            println!("cargo:rerun-if-changed={dir}/{f}");
        }
        if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={dir}/{r}");
        }
    }
}
