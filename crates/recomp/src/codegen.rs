//! Writes a whole executable as a local Cargo workspace of generated Rust.
//!
//! - One crate per fixed guest address range, so a change in one place
//!   rebuilds one crate and Cargo compiles the ranges in parallel.
//! - A file is written only when its text changes, so an unchanged range
//!   keeps its timestamp and Cargo does not rebuild it.
//! - Functions longer than `max_part` instructions are split (see
//!   [`crate::emit::block_function`]).
//! - Parallel compile jobs are capped by a memory budget (`.cargo/config.toml`),
//!   which also builds for the building machine's CPU (`target-cpu=native`):
//!   a built game is for that machine.
//!
//! The output is derived from the player's game and never goes in git.

use crate::analysis::{Analysis, Source};
use crate::emit;
use crate::xex::Image;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

pub struct Options {
    /// Guest bytes per generated crate.
    pub range: u32,
    /// Largest function part, in instructions.
    pub max_part: usize,
    /// Parallel rustc jobs.
    pub jobs: usize,
    /// Path of the repository's `crates` folder.
    pub crates: PathBuf,
    /// Executable SHA-256, checked by the host at start.
    pub sha256: String,
}

/// Default guest bytes per generated crate. Measured on a synthetic
/// 300k-instruction executable, 4 jobs: 0x40000 took 41 s from nothing and
/// 23 s after a one-function change; 0x10000 took 30 s and 6 s; 0x8000 took
/// 37 s and 3 s.
pub const DEFAULT_RANGE: u32 = 0x10000;

/// Estimated peak memory of one rustc job on a generated crate, in GB.
pub const GB_PER_JOB: usize = 3;

/// Jobs that fit in `mem_gb` of memory, at most `cpus`.
pub fn jobs_for(mem_gb: usize, cpus: usize) -> usize {
    (mem_gb / GB_PER_JOB).clamp(1, cpus.max(1))
}

#[derive(Debug, Default)]
pub struct Stats {
    pub crates: usize,
    pub functions: usize,
    pub split: usize,
    pub instructions: usize,
    pub written: usize,
    pub unchanged: usize,
}

/// Writes `path` only when `text` differs from what is there.
fn put(path: &Path, text: &str, st: &mut Stats) -> io::Result<()> {
    if std::fs::read(path).ok().as_deref() == Some(text.as_bytes()) {
        st.unchanged += 1;
        return Ok(());
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    st.written += 1;
    std::fs::write(path, text)
}

fn path_str(p: &Path) -> String {
    p.display().to_string().replace('\\', "/")
}

pub fn write(img: &Image, a: &Analysis, out: &Path, o: &Options) -> io::Result<Stats> {
    let mut st = Stats::default();
    // Block starts that only jump tables show, by function start.
    let mut extra: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for t in a.jump_tables.values() {
        if let Some(f) = a.function_at(t.bctr) {
            let e = extra.entry(f.start).or_default();
            e.extend(&t.targets);
            e.extend(t.default);
        }
    }
    let mut chunks: BTreeMap<u32, (String, Vec<u32>)> = BTreeMap::new();
    for f in a.functions.values() {
        // Imports are the host's; a function outside the image cannot be built.
        if f.source == Source::Import || f.size == 0 {
            continue;
        }
        let Some(bytes) = img.bytes(f.start, f.size as usize) else {
            continue;
        };
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|w| u32::from_be_bytes(w.try_into().unwrap()))
            .collect();
        let k = (f.start - img.base) / o.range;
        let (src, fns) = chunks.entry(k).or_default();
        let ex = extra.get(&f.start).map_or(&[][..], Vec::as_slice);
        let code = emit::block_function(
            &format!("f_{:08x}", f.start),
            f.start,
            &words,
            ex,
            o.max_part,
        );
        if code.contains("_p1(") {
            st.split += 1;
        }
        src.push_str(&code);
        fns.push(f.start);
        st.functions += 1;
        st.instructions += words.len();
    }

    let rt = path_str(&o.crates.join("rt"));
    let host = path_str(&o.crates.join("host"));
    let mut members = Vec::new();
    for (k, (src, fns)) in &chunks {
        let name = format!("g{k:03}");
        let dir = out.join("gen").join(&name);
        put(
            &dir.join("Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n[dependencies]\nxenon-rt = {{ path = \"{rt}\" }}\n"
            ),
            &mut st,
        )?;
        let mut lib =
            String::from("//! Generated from the player's own executable. Never commit.\n");
        lib.push_str(emit::PRELUDE);
        lib.push_str(src);
        lib.push_str("\npub fn register(rt: &mut Runtime) {\n");
        for f in fns {
            let _ = writeln!(lib, "    rt.register({f:#010x}, f_{f:08x});");
        }
        lib.push_str("}\n");
        put(&dir.join("src/lib.rs"), &lib, &mut st)?;
        members.push(name);
    }
    st.crates = members.len();

    let mut deps = String::new();
    let mut regs = String::new();
    for m in &members {
        let _ = writeln!(deps, "{m} = {{ path = \"../gen/{m}\" }}");
        let _ = write!(regs, "{m}::register, ");
    }
    put(
        &out.join("game/Cargo.toml"),
        &format!(
            "[package]\nname = \"game\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
             # `cargo build --features game/player`: no console window.\n\
             [features]\nplayer = []\n\n\
             [dependencies]\nxenon-host = {{ path = \"{host}\" }}\nxenon-rt = {{ path = \"{rt}\" }}\n{deps}"
        ),
        &mut st,
    )?;
    put(
        &out.join("game/src/main.rs"),
        &format!(
            "//! Generated. Never commit.\n#![cfg_attr(feature = \"player\", windows_subsystem = \"windows\")]\nfn main() -> std::process::ExitCode {{\n    xenon_host::run(\"{}\", &[{regs}])\n}}\n",
            o.sha256
        ),
        &mut st,
    )?;
    let list: Vec<String> = members.iter().map(|m| format!("\"gen/{m}\"")).collect();
    put(
        &out.join("Cargo.toml"),
        &format!(
            "[workspace]\nresolver = \"2\"\nmembers = [\"game\", {}]\n\n\
             # Generated code: no debug info or overflow checks, no incremental\n\
             # state (it costs more memory than it saves on crates this size).\n\
             [profile.dev]\nopt-level = 1\ndebug = false\noverflow-checks = false\ndebug-assertions = false\nincremental = false\n\n\
             # The runtime, host, drawer and wgpu are not generated: fully\n\
             # optimised (guest memory copies run about 3x faster than at 1).\n\
             [profile.dev.package.\"*\"]\nopt-level = 3\n",
            list.join(", ")
        ),
        &mut st,
    )?;
    // A formatter run over generated code would rewrite every file and
    // rebuild everything.
    put(
        &out.join("rustfmt.toml"),
        "disable_all_formatting = true\n",
        &mut st,
    )?;
    put(
        &out.join(".cargo/config.toml"),
        &format!(
            "# Memory guard: about {GB_PER_JOB} GB per job.\n[build]\njobs = {}\n\n\
             # The build runs on the machine that built it: code for its CPU\n\
             # (fused multiply-add and wide vectors instead of library calls\n\
             # took about a sixth off a game's render thread).\n\
             rustflags = [\"-C\", \"target-cpu=native\"]\n",
            o.jobs
        ),
        &mut st,
    )?;
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_fit_the_memory_budget() {
        assert_eq!(jobs_for(32, 24), 10);
        assert_eq!(jobs_for(2, 24), 1);
        assert_eq!(jobs_for(512, 8), 8);
    }

    /// Writes a synthetic executable's workspace to `$XENON_SYNTH_OUT` for
    /// measuring build times (no game bytes): `XENON_SYNTH_FUNCS` functions
    /// of 100 instructions each (default 3000), `XENON_SYNTH_RANGE` bytes per
    /// crate (default [`DEFAULT_RANGE`]); `XENON_SYNTH_EDIT=<n>`
    /// changes one instruction of function n, for a one-function rebuild.
    #[test]
    #[ignore]
    fn synthetic_workspace() {
        use crate::fixture::{Packing, XexSpec};
        use crate::xex::RuntimeFunction;
        let Ok(out) = std::env::var("XENON_SYNTH_OUT") else {
            return;
        };
        let env = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
        let n = env("XENON_SYNTH_FUNCS", 3000);
        let edit = env("XENON_SYNTH_EDIT", usize::MAX);
        let mut spec = XexSpec::sample();
        let start = spec.base + spec.code_rva();
        spec.code.clear();
        spec.pdata.clear();
        // lwz, addi, stw, add, cmpwi, beq +8, mullw
        let body = [
            0x8083_0000u32,
            0x3884_0001,
            0x9083_0000,
            0x7CA4_1A14,
            0x2C05_0000,
            0x4182_0008,
            0x7CC5_21D6,
        ];
        for f in 0..n {
            let a = start + spec.code.len() as u32 * 4;
            for _ in 0..14 {
                spec.code.extend(body);
            }
            if f == edit {
                let i = spec.code.len() - 6;
                spec.code[i] = 0x3884_0002;
            }
            // bl to the next function, then blr.
            spec.code.push(0x4800_0001 | 8);
            spec.code.push(0x4E80_0020);
            // The exception table has room for 512 entries; the rest are
            // found through the calls.
            if f < 500 {
                spec.pdata.push(RuntimeFunction {
                    start: a,
                    size: 100 * 4,
                    prolog_words: 0,
                });
            }
        }
        let file = spec.build(&spec.image(), Packing::None, None);
        let img = crate::xex::load(&file, &[]).unwrap();
        let a = crate::analysis::analyze(&img, &crate::hooks::Hooks::default());
        let o = Options {
            range: env("XENON_SYNTH_RANGE", DEFAULT_RANGE as usize) as u32,
            max_part: 4000,
            jobs: 4,
            crates: Path::new(env!("CARGO_MANIFEST_DIR")).join(".."),
            sha256: "0".repeat(64),
        };
        let st = write(&img, &a, Path::new(&out), &o).unwrap();
        eprintln!("{st:?}");
    }

    #[test]
    fn unchanged_files_are_not_rewritten() {
        let dir = std::env::temp_dir().join(format!("xenon-codegen-{}", std::process::id()));
        let p = dir.join("a.rs");
        let mut st = Stats::default();
        put(&p, "x", &mut st).unwrap();
        let t = std::fs::metadata(&p).unwrap().modified().unwrap();
        put(&p, "x", &mut st).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().modified().unwrap(), t);
        put(&p, "y", &mut st).unwrap();
        assert_eq!((st.written, st.unchanged), (2, 1));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
