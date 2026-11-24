//! Writes a whole executable as a local Cargo workspace of generated Rust.
//!
//! - One crate per fixed guest address range, so a change in one place
//!   rebuilds one crate and Cargo compiles the ranges in parallel.
//! - A file is written only when its text changes, so an unchanged range
//!   keeps its timestamp and Cargo does not rebuild it.
//! - Functions longer than `max_part` instructions are split (see
//!   [`crate::emit::block_function`]).
//! - Parallel compile jobs are capped by a memory budget (`.cargo/config.toml`).
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
            "[package]\nname = \"game\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n[dependencies]\nxenon-host = {{ path = \"{host}\" }}\nxenon-rt = {{ path = \"{rt}\" }}\n{deps}"
        ),
        &mut st,
    )?;
    put(
        &out.join("game/src/main.rs"),
        &format!(
            "//! Generated. Never commit.\nfn main() -> std::process::ExitCode {{\n    xenon_host::run(\"{}\", &[{regs}])\n}}\n",
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
             [profile.dev]\nopt-level = 1\ndebug = false\noverflow-checks = false\ndebug-assertions = false\nincremental = false\n",
            list.join(", ")
        ),
        &mut st,
    )?;
    put(
        &out.join(".cargo/config.toml"),
        &format!(
            "# Memory guard: about {GB_PER_JOB} GB per job.\n[build]\njobs = {}\n",
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
