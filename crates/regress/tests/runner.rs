//! The runner against the fake program: synthetic frames only, no game.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use xenon_regress::{Outcome, Runner, Verdict};

const FAKE: &str = env!("CARGO_BIN_EXE_xenon-regress-fake");

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("xenon-regress-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn write_test(root: &Path, name: &str, toml: &str) -> PathBuf {
    let dir = root.join("tests").join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("test.toml"), toml).unwrap();
    dir
}

fn runner(root: &Path, bless: bool) -> Runner {
    Runner {
        program: FAKE.into(),
        program_args: vec![],
        out: root.join("out"),
        bless,
    }
}

fn verdicts(o: &Outcome) -> Vec<Verdict> {
    o.frames.iter().map(|f| f.verdict.clone()).collect()
}

const TOL: &str = "[tolerance]\nchannel = 8\nmax_bad_fraction = 0.01\n";

#[test]
fn bless_then_pass_then_tolerance() {
    let root = scratch("bless");
    let t = write_test(
        &root,
        "title",
        &format!("input = \"100:start\"\nframes = [3, 7]\n{TOL}"),
    );

    // No expected frames yet.
    let o = runner(&root, false).run(&t);
    assert!(!o.passed());
    assert!(matches!(verdicts(&o)[0], Verdict::Error(_)));

    let o = runner(&root, true).run(&t);
    assert_eq!(verdicts(&o), [Verdict::Blessed, Verdict::Blessed]);
    assert!(t.join("expected/frame-7.png").is_file());

    let o = runner(&root, false).run(&t);
    assert!(o.passed(), "{o:?}");
    assert!(root.join("out/title/frame-3.sheet.png").is_file());
    assert!(root.join("out/title/frame-3.blink.png").is_file());
    let log = fs::read_to_string(root.join("out/title/program.log")).unwrap();
    assert!(log.contains("input 100:start"));

    // 20 of 2304 pixels (0.87%) off: inside a 1% limit, outside 0.1%.
    let noisy = |limit: &str| {
        fs::write(
            t.join("test.toml"),
            format!("frames = [3, 7]\nargs = [\"--noise\", \"20\"]\n[tolerance]\nchannel = 8\nmax_bad_fraction = {limit}\n"),
        )
        .unwrap();
        runner(&root, false).run(&t)
    };
    assert!(noisy("0.01").passed());
    let o = noisy("0.001");
    assert_eq!(verdicts(&o), [Verdict::Fail, Verdict::Fail]);
    assert!((o.frames[0].bad_fraction - 20.0 / 2304.0).abs() < 1e-9);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn wrong_frame_fails_with_pictures() {
    let root = scratch("broken");
    let t = write_test(&root, "menu", &format!("frames = [1]\n{TOL}"));
    assert!(runner(&root, true).run(&t).passed());
    fs::write(
        t.join("test.toml"),
        format!("frames = [1]\nargs = [\"--broken\"]\n{TOL}"),
    )
    .unwrap();
    let o = runner(&root, false).run(&t);
    assert_eq!(verdicts(&o), [Verdict::Fail]);
    let sheet = xenon_compare::load_png(o.frames[0].sheet.as_ref().unwrap()).unwrap();
    assert_eq!(
        (sheet.width, sheet.height),
        (64 * 3 + 2 * xenon_compare::GAP, 36)
    );
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn crash_hang_and_missing_frame_fail() {
    let root = scratch("bad");
    let crash = write_test(
        &root,
        "crash",
        &format!("frames = [1]\nargs = [\"--crash\"]\n{TOL}"),
    );
    let o = runner(&root, true).run(&crash);
    assert!(o.error.as_deref().unwrap().contains("exited"), "{o:?}");

    let hang = write_test(
        &root,
        "hang",
        &format!("frames = [1]\ntimeout_secs = 1\nargs = [\"--hang\"]\n{TOL}"),
    );
    let o = runner(&root, true).run(&hang);
    assert!(
        o.error.as_deref().unwrap().contains("still running"),
        "{o:?}"
    );

    let skip = write_test(
        &root,
        "skip",
        &format!("frames = [1, 2]\nargs = [\"--skip\", \"2\"]\n{TOL}"),
    );
    let o = runner(&root, true).run(&skip);
    assert!(!o.passed());
    assert_eq!(o.frames[0].verdict, Verdict::Blessed);
    assert!(matches!(&o.frames[1].verdict, Verdict::Error(e) if e.contains("no actual frame")));
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn command_line_finds_tests_and_sets_exit_code() {
    let root = scratch("cli");
    write_test(&root, "a", &format!("frames = [2]\n{TOL}"));
    write_test(&root, "b", &format!("frames = [200]\n{TOL}"));
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xenon-regress"))
            .arg("--program")
            .arg(FAKE)
            .arg("--out")
            .arg(root.join("out"))
            .args(extra)
            .arg(root.join("tests"))
            .output()
            .unwrap()
    };
    let o = run(&["--bless"]);
    assert!(o.status.success());
    let o = run(&[]);
    let text = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{text}");
    assert!(
        text.contains("PASS a frame 2") && text.contains("2 of 2 tests passed"),
        "{text}"
    );

    // A changed frame makes the whole run fail.
    fs::copy(
        root.join("tests/a/expected/frame-2.png"),
        root.join("tests/b/expected/frame-200.png"),
    )
    .unwrap();
    let o = run(&[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stdout).contains("FAIL b frame 200"));
    fs::remove_dir_all(&root).unwrap();
}
