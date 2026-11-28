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
use xenon_recomp::{analysis, codegen, disc, hooks::Hooks, patch, stfs, xex};

const USAGE: &str = "\
xenon-recomp: Xbox 360 executable to Rust recompiler

USAGE:
    xenon-recomp scan <executable> [--patch <title update>] [--key <32 hex digits>]
                      [--hooks <hooks.toml>] [--games <dir>] [--list <file>]
    xenon-recomp build <executable> --out <dir> [scan options]
                      [--range <bytes>] [--max-part <instructions>] [--mem-gb <n>]
    xenon-recomp unpack <disc image or package> <dir>

scan    Loads the executable (applying the title update in memory), finds
        functions and jump tables, and prints a summary. --list writes every
        function and table to a local file; it is derived from the game, so
        never commit it.

build   Writes the executable as a Cargo workspace under --out (one crate
        per --range bytes of guest code, default 0x10000; functions longer
        than --max-part instructions, default 4000, are split; parallel
        jobs capped at --mem-gb / 3, default from 16). Files are rewritten
        only when their text changes. Then build with `cargo build` in --out
        (`cargo build --features game/player` for the double-click player:
        no console window; the log goes to rustxenon-play.log).
        The output is derived from the game: keep it local, never commit it.

unpack  Copies every file of the player's own disc image into <dir> (the
        game folder the host reads), or of a content package such as a title
        update (into its own folder, which the host reads as update:).
        Files already there with the right size are skipped.

The console key for encrypted executables comes from --key or the
XENON_CONSOLE_KEY environment variable; it is never stored in this repository.
Without --hooks, games/<sha256>/hooks.toml under --games (default ./games) is
used when it exists. The SHA-256 is of the executable as loaded (patched).";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    if !matches!(cmd, Some("scan" | "build" | "unpack")) || args.len() < 2 {
        println!("{USAGE}");
        return if args.iter().any(|a| a == "-h" || a == "--help") {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let r = match cmd {
        Some("scan") => scan(&args[1], &args[2..]),
        Some("build") => build(&args[1], &args[2..]),
        _ => unpack(&args[1], args.get(2).map_or("", String::as_str)),
    };
    match r {
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
    // Checked before slicing: a multibyte character would split a byte slice and panic.
    if !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("console key is not hex".into());
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

fn unpack(image: &str, out: &str) -> Result<(), String> {
    if out.is_empty() {
        return Err("unpack needs <disc image> <dir>".into());
    }
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(
        &mut std::fs::File::open(image).map_err(|e| format!("{image}: {e}"))?,
        &mut magic,
    )
    .map_err(|e| format!("{image}: {e}"))?;
    if stfs::is_package(&magic) {
        let p = read(image)?;
        let entries = stfs::entries(&p).map_err(|e| e.to_string())?;
        for e in &entries {
            let path = Path::new(out).join(&e.path);
            if e.dir {
                std::fs::create_dir_all(&path).map_err(|x| format!("{}: {x}", path.display()))?;
            } else {
                if let Some(d) = path.parent() {
                    std::fs::create_dir_all(d).map_err(|x| format!("{}: {x}", d.display()))?;
                }
                std::fs::write(&path, &e.data).map_err(|x| format!("{}: {x}", path.display()))?;
            }
        }
        println!("unpacked    {} package entries", entries.len());
        return Ok(());
    }
    let f = std::fs::File::open(image).map_err(|e| format!("{image}: {e}"))?;
    let mut d = disc::Disc::open(std::io::BufReader::new(f)).map_err(|e| e.to_string())?;
    let entries = d.entries().map_err(|e| e.to_string())?;
    let (mut copied, mut skipped, mut bytes) = (0, 0, 0u64);
    for e in &entries {
        let p = Path::new(out).join(&e.path);
        if e.dir {
            std::fs::create_dir_all(&p).map_err(|x| format!("{}: {x}", p.display()))?;
            continue;
        }
        if std::fs::metadata(&p).is_ok_and(|m| m.len() == e.size) {
            skipped += 1;
            continue;
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|x| format!("{}: {x}", parent.display()))?;
        }
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&p).map_err(|x| format!("{}: {x}", p.display()))?,
        );
        d.copy(e, &mut w).map_err(|x| x.to_string())?;
        copied += 1;
        bytes += e.size;
    }
    println!(
        "unpacked    {copied} files ({} MB), {skipped} already there",
        bytes >> 20
    );
    Ok(())
}

fn num(args: &[String], name: &str, default: u64) -> Result<u64, String> {
    match opt(args, name) {
        None => Ok(default),
        Some(v) => match v.strip_prefix("0x") {
            Some(h) => u64::from_str_radix(h, 16),
            None => v.parse(),
        }
        .map_err(|_| format!("{name}: not a number: {v}")),
    }
}

fn build(exe: &str, args: &[String]) -> Result<(), String> {
    let out = opt(args, "--out").ok_or("build needs --out <dir>")?;
    let (sha, img, _, a) = load(exe, args)?;
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let range = num(args, "--range", codegen::DEFAULT_RANGE as u64)? as u32;
    if range == 0 || range % 4 != 0 {
        return Err("--range must be a positive multiple of 4".into());
    }
    let o = codegen::Options {
        range,
        max_part: num(args, "--max-part", 4000)?.max(1) as usize,
        jobs: codegen::jobs_for(num(args, "--mem-gb", 16)? as usize, cpus),
        crates: Path::new(env!("CARGO_MANIFEST_DIR")).join(".."),
        sha256: sha.clone(),
    };
    let st = codegen::write(&img, &a, Path::new(out), &o).map_err(|e| format!("{out}: {e}"))?;
    println!("sha256      {sha}");
    println!(
        "generated   {} crates, {} functions ({} split), {} instructions",
        st.crates, st.functions, st.split, st.instructions
    );
    println!(
        "files       {} written, {} unchanged; {} jobs",
        st.written, st.unchanged, o.jobs
    );
    Ok(())
}

fn load(
    exe: &str,
    args: &[String],
) -> Result<(String, xex::Image, Option<PathBuf>, analysis::Analysis), String> {
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
    Ok((sha, img, hooks_path, a))
}

fn scan(exe: &str, args: &[String]) -> Result<(), String> {
    let (sha, img, hooks_path, a) = load(exe, args)?;
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
        for i in &img.imports {
            let kind = if i.thunk { "thunk" } else { "slot" };
            let _ = writeln!(
                s,
                "import {} {} {:#010x} {kind}",
                i.library, i.ordinal, i.address
            );
        }
        for p in &a.problems {
            let _ = writeln!(s, "problem {p}");
        }
        std::fs::write(out, s).map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_key;

    #[test]
    fn parse_key_rejects_bad_input_without_panicking() {
        // Ten 3-byte characters plus two ASCII digits: 32 bytes, not on char boundaries.
        let multibyte = format!("{}aa", "\u{20ac}".repeat(10));
        assert_eq!(multibyte.len(), 32);
        assert!(parse_key(&multibyte).is_err());
        assert!(parse_key(&format!("+f{}", "0".repeat(30))).is_err());
        assert!(parse_key("00").is_err());
        assert_eq!(parse_key(&"0f".repeat(16)), Ok([0x0f; 16]));
    }
}
