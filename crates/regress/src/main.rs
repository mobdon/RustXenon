//! `xenon-regress`: runs game-level regression tests headless.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;
use xenon_regress::{find_tests, report, Runner};

const USAGE: &str = "\
usage: xenon-regress [--program PATH] [--out DIR] [--bless] <test or folder>... [-- program args]
  Runs every test folder (one holding test.toml) at or below each path.
  --program  the program to test (default: $XENON_REGRESS_PROGRAM)
  --out      where actual frames and pictures go (default: runs/regress)
  --bless    save the actual frame as expected where none exists yet
  Arguments after -- go to the program before each test's own (for example
  the local game folder). Exits 0 when every test passes, 1 otherwise.";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut program = std::env::var_os("XENON_REGRESS_PROGRAM").map(PathBuf::from);
    let mut out = PathBuf::from("runs/regress");
    let mut bless = false;
    let mut paths = Vec::new();
    let mut program_args = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--program" => program = args.next().map(PathBuf::from),
            "--out" => match args.next() {
                Some(o) => out = PathBuf::from(o),
                None => return usage("--out needs a folder"),
            },
            "--bless" => bless = true,
            "--" => program_args.extend(args.by_ref()),
            "-h" | "--help" => return usage(""),
            _ if a.starts_with("--") => return usage(&format!("unknown option {a}")),
            _ => paths.push(PathBuf::from(a)),
        }
    }
    let Some(program) = program else {
        return usage("no program: pass --program or set XENON_REGRESS_PROGRAM");
    };
    let tests = find_tests(&paths);
    if tests.is_empty() {
        return usage("no test.toml found under the given paths");
    }
    let mut names = HashSet::new();
    for t in &tests {
        if !names.insert(t.file_name()) {
            eprintln!(
                "xenon-regress: two tests are named {:?}; results would overwrite each other",
                t.file_name().unwrap_or_default()
            );
            return ExitCode::from(2);
        }
    }
    let runner = Runner {
        program,
        program_args,
        out,
        bless,
    };
    let mut failed = 0;
    for t in &tests {
        let o = runner.run(t);
        print!("{}", report(&o));
        if !o.passed() {
            failed += 1;
        }
    }
    println!("{} of {} tests passed", tests.len() - failed, tests.len());
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn usage(msg: &str) -> ExitCode {
    if !msg.is_empty() {
        eprintln!("xenon-regress: {msg}\n");
    }
    eprintln!("{USAGE}");
    ExitCode::from(2)
}
