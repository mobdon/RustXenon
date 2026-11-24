//! `xenon-regress`: game-level regression tests.
//!
//! A test is a folder holding `test.toml` (the definition, safe for git) and,
//! locally only, `expected/frame-<N>.png` (game pixels, never in git). The
//! runner starts a program headless, lets it capture the listed frames, and
//! compares each with the expected frame within the test's tolerance.
//!
//! # Program contract
//!
//! The runner starts
//!
//! ```text
//! <program> <args after --> <test args> [--press <input>] --capture-frames <N,M,...> --capture-dir <dir>
//! ```
//!
//! and expects it to write `<dir>/frame-<N>.png` for every listed frame
//! (frame N = the Nth presented frame, counting from 1), then exit with
//! status 0. Anything else fails the test. Its output goes to
//! `<out>/<test>/program.log`.
//!
//! Pure logic and child processes only: no window, sound or GPU here.

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use xenon_compare::{diff, load_png, save_blink, save_png, side_by_side, Tolerance};

/// File that marks a folder as a test.
pub const TEST_FILE: &str = "test.toml";

/// `test.toml`. Unknown keys are refused so nothing else creeps in.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TestDef {
    /// Controller script, passed as `--press` (`<ms>:<button>,...`).
    #[serde(default)]
    pub input: Option<String>,
    /// Presented frame numbers to capture and compare.
    pub frames: Vec<u64>,
    /// Extra program arguments for this test.
    #[serde(default)]
    pub args: Vec<String>,
    /// The run fails if the program has not exited by then.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Required: multi-thread runs are not exactly repeatable.
    pub tolerance: TolDef,
}

fn default_timeout() -> u64 {
    300
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TolDef {
    /// Per-channel difference (0 to 255) a pixel may have and still match.
    pub channel: u8,
    /// Fraction (0 to 1) of pixels allowed to exceed `channel`.
    pub max_bad_fraction: f64,
}

impl From<TolDef> for Tolerance {
    fn from(t: TolDef) -> Self {
        Tolerance {
            channel: t.channel,
            max_bad_fraction: t.max_bad_fraction,
        }
    }
}

impl TestDef {
    pub fn parse(text: &str) -> Result<Self, String> {
        let def: TestDef = toml::from_str(text).map_err(|e| e.to_string())?;
        if def.frames.is_empty() || def.frames.contains(&0) {
            return Err("frames must list at least one frame, counting from 1".into());
        }
        if !(0.0..=1.0).contains(&def.tolerance.max_bad_fraction) {
            return Err("tolerance.max_bad_fraction must be between 0 and 1".into());
        }
        Ok(def)
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let p = dir.join(TEST_FILE);
        let text = fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", p.display()))
    }
}

/// Test folders at or below each path, sorted.
pub fn find_tests(paths: &[PathBuf]) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if dir.join(TEST_FILE).is_file() {
            out.push(dir.to_path_buf());
            return;
        }
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                if e.file_type().is_ok_and(|t| t.is_dir()) {
                    walk(&e.path(), out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for p in paths {
        walk(p, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

/// How one frame came out.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Pass,
    Fail,
    /// No expected frame; `--bless` was given, so the actual one was saved.
    Blessed,
    /// Could not compare (missing file, size mismatch).
    Error(String),
}

#[derive(Clone, Debug)]
pub struct FrameResult {
    pub frame: u64,
    pub verdict: Verdict,
    pub bad_fraction: f64,
    pub max_channel_delta: u8,
    /// The test's `max_bad_fraction`.
    pub limit: f64,
    /// Expected | actual | diff, when both frames were readable.
    pub sheet: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub name: String,
    /// Set when the run itself failed (bad definition, crash, timeout).
    pub error: Option<String>,
    pub frames: Vec<FrameResult>,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.error.is_none()
            && self
                .frames
                .iter()
                .all(|f| matches!(f.verdict, Verdict::Pass | Verdict::Blessed))
    }
}

/// What to run and where to put results.
pub struct Runner {
    pub program: PathBuf,
    /// Arguments passed before each test's own (game folder and the like).
    pub program_args: Vec<String>,
    /// Results go to `<out>/<test name>/`.
    pub out: PathBuf,
    /// Save the actual frame as expected where none exists yet.
    pub bless: bool,
}

pub fn frame_file(frame: u64) -> String {
    format!("frame-{frame}.png")
}

impl Runner {
    pub fn run(&self, dir: &Path) -> Outcome {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());
        let mut outcome = Outcome {
            name,
            error: None,
            frames: Vec::new(),
        };
        if let Err(e) = self.run_inner(dir, &mut outcome) {
            outcome.error = Some(e);
        }
        outcome
    }

    fn run_inner(&self, dir: &Path, outcome: &mut Outcome) -> Result<(), String> {
        let def = TestDef::load(dir)?;
        let out = self.out.join(&outcome.name);
        let actual = out.join("actual");
        // The results folder is ours; start each run clean so old frames never pass.
        if out.exists() {
            fs::remove_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        }
        fs::create_dir_all(&actual).map_err(|e| format!("{}: {e}", actual.display()))?;

        self.start_program(&def, &actual, &out.join("program.log"))?;

        let tol = Tolerance::from(def.tolerance);
        let expected_dir = dir.join("expected");
        for &frame in &def.frames {
            outcome.frames.push(compare_frame(
                frame,
                &expected_dir,
                &actual,
                &out,
                &tol,
                self.bless,
            ));
        }
        Ok(())
    }

    fn start_program(&self, def: &TestDef, capture: &Path, log: &Path) -> Result<(), String> {
        let log_file = fs::File::create(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.program_args).args(&def.args);
        if let Some(input) = &def.input {
            cmd.arg("--press").arg(input);
        }
        let frames: Vec<String> = def.frames.iter().map(u64::to_string).collect();
        cmd.arg("--capture-frames")
            .arg(frames.join(","))
            .arg("--capture-dir")
            .arg(capture)
            .stdin(Stdio::null())
            .stdout(log_file.try_clone().map_err(|e| e.to_string())?)
            .stderr(log_file);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", self.program.display()))?;
        let deadline = Instant::now() + Duration::from_secs(def.timeout_secs);
        loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(status) if status.success() => return Ok(()),
                Some(status) => {
                    return Err(format!(
                        "program exited with {status}; see {}",
                        log.display()
                    ))
                }
                None if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "program still running after {} s; stopped it; see {}",
                        def.timeout_secs,
                        log.display()
                    ));
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
}

fn compare_frame(
    frame: u64,
    expected_dir: &Path,
    actual_dir: &Path,
    out: &Path,
    tol: &Tolerance,
    bless: bool,
) -> FrameResult {
    let mut r = FrameResult {
        frame,
        verdict: Verdict::Fail,
        bad_fraction: 0.0,
        max_channel_delta: 0,
        limit: tol.max_bad_fraction,
        sheet: None,
    };
    let actual_path = actual_dir.join(frame_file(frame));
    let actual = match load_png(&actual_path) {
        Ok(a) => a,
        Err(e) => {
            r.verdict = Verdict::Error(format!("no actual frame: {e}"));
            return r;
        }
    };
    let expected_path = expected_dir.join(frame_file(frame));
    if !expected_path.exists() {
        r.verdict = if bless {
            match fs::create_dir_all(expected_dir)
                .and_then(|_| fs::copy(&actual_path, &expected_path))
            {
                Ok(_) => Verdict::Blessed,
                Err(e) => Verdict::Error(format!("cannot save {}: {e}", expected_path.display())),
            }
        } else {
            Verdict::Error(format!(
                "no expected frame {}; run with --bless to accept the actual one",
                expected_path.display()
            ))
        };
        return r;
    }
    let result = load_png(&expected_path).and_then(|expected| {
        let d = diff(&expected, &actual, tol.channel)?;
        let sheet = out.join(format!("frame-{frame}.sheet.png"));
        save_png(&sheet, &side_by_side(&[&expected, &actual, &d.image]))?;
        save_blink(
            &out.join(format!("frame-{frame}.blink.png")),
            &expected,
            &actual,
            500,
        )?;
        Ok((d, sheet))
    });
    match result {
        Ok((d, sheet)) => {
            r.verdict = if d.passes(tol) {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            r.bad_fraction = d.bad_fraction();
            r.max_channel_delta = d.max_channel_delta;
            r.sheet = Some(sheet);
        }
        Err(e) => r.verdict = Verdict::Error(e),
    }
    r
}

/// One line per frame, for the terminal.
pub fn report(o: &Outcome) -> String {
    let mut s = String::new();
    if let Some(e) = &o.error {
        s += &format!("FAIL {}: {e}\n", o.name);
    }
    for f in &o.frames {
        let what = match &f.verdict {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Blessed => "BLESSED",
            Verdict::Error(_) => "FAIL",
        };
        s += &format!("{what} {} frame {}", o.name, f.frame);
        match &f.verdict {
            Verdict::Error(e) => s += &format!(": {e}"),
            Verdict::Blessed => s += ": saved as expected",
            _ => {
                s += &format!(
                    ": {:.4}% of pixels differ (limit {:.4}%), largest difference {}",
                    f.bad_fraction * 100.0,
                    f.limit * 100.0,
                    f.max_channel_delta
                );
            }
        }
        if let Some(p) = &f.sheet {
            s += &format!(", picture {}", p.display());
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_definition() {
        let d = TestDef::parse(
            r#"
            input = "5000:start,9000:a"
            frames = [300, 600]
            args = ["--start-in-world"]
            timeout_secs = 60
            [tolerance]
            channel = 8
            max_bad_fraction = 0.001
            "#,
        )
        .unwrap();
        assert_eq!(d.frames, [300, 600]);
        assert_eq!(d.input.as_deref(), Some("5000:start,9000:a"));
        assert_eq!(d.timeout_secs, 60);
        assert_eq!(d.tolerance.channel, 8);
    }

    #[test]
    fn tolerance_is_required() {
        assert!(TestDef::parse("frames = [1]").is_err());
    }

    #[test]
    fn unknown_keys_and_bad_values_are_refused() {
        let tol = "[tolerance]\nchannel = 8\nmax_bad_fraction = 0.01\n";
        assert!(TestDef::parse(&format!("frames = [1]\npixels = []\n{tol}")).is_err());
        assert!(TestDef::parse(&format!("frames = []\n{tol}")).is_err());
        assert!(TestDef::parse(&format!("frames = [0]\n{tol}")).is_err());
        assert!(
            TestDef::parse("frames = [1]\n[tolerance]\nchannel = 8\nmax_bad_fraction = 2.0")
                .is_err()
        );
        assert!(TestDef::parse(&format!("frames = [1]\n{tol}")).is_ok());
    }
}
