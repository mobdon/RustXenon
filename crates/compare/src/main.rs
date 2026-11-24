//! `xenon-compare`: diff and blink two frames from the command line.

use std::path::PathBuf;
use std::process::ExitCode;
use xenon_compare::{diff, load_png, save_blink, save_png, side_by_side, Tolerance};

const USAGE: &str = "\
usage:
  xenon-compare diff <expected.png> <actual.png> [--out sheet.png] [--channel N] [--max-bad F]
      Prints how many pixels differ by more than N (default 8) in any of R, G, B.
      Exits 1 when more than fraction F (default 0.001) of pixels differ.
      --out writes expected | actual | diff side by side (bad pixels in red).
  xenon-compare blink <a.png> <b.png> --out blink.png [--ms N]
      Writes an animated PNG flipping between a and b every N ms (default 500).";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("xenon-compare: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<String>) -> Result<ExitCode, String> {
    let mut files = Vec::new();
    let mut out = None;
    let mut tol = Tolerance::default();
    let mut ms = 500u16;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val()?)),
            "--channel" => tol.channel = val()?.parse().map_err(|_| "--channel takes 0 to 255")?,
            "--max-bad" => {
                tol.max_bad_fraction = val()?
                    .parse()
                    .ok()
                    .filter(|f| (0.0..=1.0).contains(f))
                    .ok_or("--max-bad takes a fraction from 0 to 1")?
            }
            "--ms" => ms = val()?.parse().map_err(|_| "--ms takes a number")?,
            _ if a.starts_with("--") => return Err(format!("unknown option {a}")),
            _ => files.push(PathBuf::from(a)),
        }
    }
    let [a, b] = files.as_slice() else {
        return Err("need exactly two images".into());
    };
    let (a_img, b_img) = (load_png(a)?, load_png(b)?);
    match args.first().map(String::as_str) {
        Some("diff") => {
            let d = diff(&a_img, &b_img, tol.channel)?;
            let pass = d.passes(&tol);
            println!(
                "{}: {} of {} pixels differ by more than {} ({:.4}%, limit {:.4}%), largest difference {}",
                if pass { "PASS" } else { "FAIL" },
                d.bad_pixels,
                d.total_pixels,
                tol.channel,
                d.bad_fraction() * 100.0,
                tol.max_bad_fraction * 100.0,
                d.max_channel_delta
            );
            if let Some(out) = out {
                save_png(&out, &side_by_side(&[&a_img, &b_img, &d.image]))?;
            }
            Ok(if pass {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Some("blink") => {
            save_blink(&out.ok_or("blink needs --out")?, &a_img, &b_img, ms)?;
            Ok(ExitCode::SUCCESS)
        }
        _ => Err("first argument must be diff or blink".into()),
    }
}
