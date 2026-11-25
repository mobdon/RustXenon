//! Replays a frame bundle (`--capture-bundle`) offline and writes the last
//! frame as a PNG, with draws `lo..=hi` of that frame left out. Halving
//! the skipped range finds the draw behind a wrong colour.
//!
//! cargo run --release -p xenon-draw --example replay_frame -- <bundle dir> <out.png> [lo hi]
//!
//! Draws are counted from 1, as in the capture's draw list.

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 && args.len() != 4 {
        eprintln!("usage: replay_frame <bundle dir> <out.png> [lo hi]");
        std::process::exit(2);
    }
    let num = |i: usize| -> u32 {
        args.get(i).map_or(0, |a| {
            a.parse().unwrap_or_else(|_| {
                eprintln!("error: {a} is not a draw number");
                std::process::exit(2);
            })
        })
    };
    // With no range, 1..=0 skips nothing.
    let (lo, hi) = if args.len() == 4 {
        (num(2), num(3))
    } else {
        (1, 0)
    };
    let (cmds, mut mem) = match xenon_gpu::bundle::read(Path::new(&args[0])) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}: {e}", args[0]);
            std::process::exit(1);
        }
    };
    let (cmds, n) = xenon_gpu::bundle::skip_draws(&cmds, lo, hi);
    println!("last frame: {n} draws; skipping {lo}..={hi}");
    let mut d = match xenon_draw::Drawer::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let report = match d.replay(&cmds, &mut mem) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: replay: {e}");
            std::process::exit(1);
        }
    };
    for w in report.warnings.iter().chain(&report.unsupported) {
        println!("note: {w}");
    }
    let Some(f) = report.frames.last() else {
        eprintln!("error: the bundle drew no frame");
        std::process::exit(1);
    };
    if let Err(e) = f.write_png(&args[1]) {
        eprintln!("error: {}: {e}", args[1]);
        std::process::exit(1);
    }
    println!("{} ({}x{})", args[1], f.width, f.height);
}
