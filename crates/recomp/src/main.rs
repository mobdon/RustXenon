//! `xenon-recomp`: reads an Xbox 360 executable and writes Rust that links
//! against `xenon-rt`.
//!
//! Stages, in order:
//! 1. 64-bit PowerPC and vector instruction translation, tested per instruction
//!    against a CPU emulator with no game bytes (done: see `ppc.rs` and the
//!    `xenon-cpu-tests` crate).
//! 2. Executable loader, package unpackers, function and jump-table analysis.
//! 3. Code generation by fixed address range, written only when bytes change,
//!    with oversized functions split and capped parallel compiles.
//!
//! Input always comes from the player's own files. Output is never committed.

use std::process::ExitCode;

const USAGE: &str = "\
xenon-recomp: Xbox 360 executable to Rust recompiler (scaffold)

USAGE:
    xenon-recomp <executable> --out <dir> [--hooks <hooks.toml>]

Nothing is implemented yet; see the README for the plan.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    eprintln!("xenon-recomp: not implemented yet (step 0 scaffold)");
    ExitCode::FAILURE
}
