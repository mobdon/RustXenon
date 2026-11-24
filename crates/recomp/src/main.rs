//! `xenon-recomp`: reads an Xbox 360 executable and writes Rust that links
//! against `xenon-rt`.
//!
//! Stages, in order:
//! 1. 64-bit PowerPC and vector instruction translation, tested per instruction
//!    against a CPU emulator with no game bytes (done: see `ppc.rs` and the
//!    `xenon-cpu-tests` crate).
//! 2. Executable loader, title updates, function and jump-table analysis
//!    (done: `xex.rs`, `patch.rs`, `analysis.rs`; disc and package unpackers
//!    still to come).
//! 3. Code generation by fixed address range, written only when bytes change,
//!    with oversized functions split and capped parallel compiles.
//!
//! Input always comes from the player's own files. Output is never committed.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use xenon_recomp::{analysis, hooks::Hooks, patch, xex};

const USAGE: &str = "\
xenon-recomp: Xbox 360 executable to Rust recompiler

USAGE:
    xenon-recomp scan <executable> [--patch <title update>] [--key <32 hex digits>]
                      [--hooks <hooks.toml>] [--games <dir>] [--list <file>]

scan    Loads the executable (applying the title update in memory), finds
        functions and jump tables, and prints a summary. --list writes every
        function and table to a local file; it is derived from the game, so
        never commit it.

The console key for encrypted executables comes from --key or the
XENON_CONSOLE_KEY environment variable; it is never stored in this repository.
Without --hooks, games/<sha256>/hooks.toml under --games (default ./games) is
used when it exists. The SHA-256 is of the executable as loaded (patched).";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("scan") || args.len() < 2 {
        println!("{USAGE}");
        return if args.iter().any(|a| a == "-h" || a == "--help") {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    match scan(&args[1], &args[2..]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xenon-recomp: {e}");
            ExitCode::FAILURE
        }
    }
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn parse_key(hex: &str) -> Result<[u8; 16], String> {
    let hex = hex.trim();
    if hex.len() != 32 {
        return Err("console key must be 32 hex digits".into());
    }
    let mut k = [0u8; 16];
    for (i, b) in k.iter_mut().enumerate() {
        *b =
            u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| "console key is not hex")?;
    }
    Ok(k)
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

fn scan(exe: &str, args: &[String]) -> Result<(), String> {
    let env_key = std::env::var("XENON_CONSOLE_KEY").ok();
    let keys = match opt(args, "--key").or(env_key.as_deref()) {
        Some(k) => vec![parse_key(k)?],
        None => vec![],
    };
    let mut file = read(exe)?;
    if let Some(p) = opt(args, "--patch") {
        file = patch::apply(&file, &read(p)?, &keys).map_err(|e| e.to_string())?;
    }
    let sha: String = Sha256::digest(&file)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let img = xex::load(&file, &keys).map_err(|e| e.to_string())?;

    let hooks_path = match opt(args, "--hooks") {
        Some(p) => Some(PathBuf::from(p)),
        None => Some(Hooks::path(
            Path::new(opt(args, "--games").unwrap_or("games")),
            &sha,
        ))
        .filter(|p| p.exists()),
    };
    let hooks = match &hooks_path {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            Hooks::parse(&text, &sha).map_err(|e| e.to_string())?
        }
        None => Hooks::default(),
    };
    let a = analysis::analyze(&img, &hooks);

    println!("sha256      {sha}");
    println!(
        "hooks       {}",
        hooks_path.map_or("none".into(), |p| p.display().to_string())
    );
    println!(
        "base        {:#010x}  entry {:#010x}  size {:#x}",
        img.base,
        img.entry,
        img.data.len()
    );
    if let (Some(t), Some(v)) = (img.title_id, img.version) {
        println!("title id    {t:#010x}  version {v:#010x}");
    }
    for s in &img.sections {
        let kind = if s.code { "code" } else { "data" };
        println!(
            "section     {:<8} {:#010x} {:#9x} {kind}",
            s.name, s.start, s.size
        );
    }
    let mut libs = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    for i in &img.imports {
        let e = libs.entry(&i.library).or_default();
        if i.thunk {
            e.0 += 1
        } else {
            e.1 += 1
        }
    }
    for (l, (f, v)) in libs {
        println!("imports     {l}: {f} functions, {v} variables");
    }
    println!("exceptions  {} entries", img.exception_table.len());
    let mut by = std::collections::BTreeMap::new();
    for f in a.functions.values() {
        *by.entry(format!("{:?}", f.source)).or_insert(0) += 1;
    }
    println!("functions   {} {by:?}", a.functions.len());
    println!("jump tables {}", a.jump_tables.len());
    println!("problems    {}", a.problems.len());
    for p in a.problems.iter().take(20) {
        println!("  {p}");
    }

    if let Some(out) = opt(args, "--list") {
        use std::fmt::Write;
        let mut s = String::new();
        for f in a.functions.values() {
            let _ = writeln!(
                s,
                "fn {:#010x} {:#x} {:?} {}",
                f.start,
                f.size,
                f.source,
                f.name.as_deref().unwrap_or("")
            );
        }
        for t in a.jump_tables.values() {
            let targets: Vec<String> = t.targets.iter().map(|x| format!("{x:#x}")).collect();
            let _ = writeln!(
                s,
                "table {:#010x} {:?} {}",
                t.bctr,
                t.kind,
                targets.join(" ")
            );
        }
        for p in &a.problems {
            let _ = writeln!(s, "problem {p}");
        }
        std::fs::write(out, s).map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(())
}
