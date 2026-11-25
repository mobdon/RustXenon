//! Replays a frame bundle (`--capture-bundle`) offline.
//!
//! cargo run --release -p xenon-draw --example replay_frame -- <bundle dir> <out.png> [lo hi]
//!
//! writes the last frame as a PNG, with draws `lo..=hi` of that frame left
//! out. Halving the skipped range finds the draw behind a wrong colour.
//!
//! cargo run --release -p xenon-draw --example replay_frame -- <bundle dir> --targets-at <k> <dir>
//!
//! stops after draw `k` of the last frame and writes every colour render
//! target as `<dir>/rt_<base>_<pitch>_<format>.png`, to find the first draw
//! that puts a wrong colour in a target. Float targets are clamped to 0..1.
//!
//! Draws are counted from 1, as in the capture's draw list.

use std::path::Path;
use xenon_draw::{Drawer, TargetFormat};

fn fail(why: String) -> ! {
    eprintln!("error: {why}");
    std::process::exit(1);
}

fn num(a: &str) -> u32 {
    a.parse().unwrap_or_else(|_| {
        eprintln!("error: {a} is not a draw number");
        std::process::exit(2);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let targets = args.len() == 4 && args[1] == "--targets-at";
    if !targets && args.len() != 2 && args.len() != 4 {
        eprintln!(
            "usage: replay_frame <bundle dir> <out.png> [lo hi]\n       replay_frame <bundle dir> --targets-at <k> <dir>"
        );
        std::process::exit(2);
    }
    let (cmds, mut mem) = xenon_gpu::bundle::read(Path::new(&args[0]))
        .unwrap_or_else(|e| fail(format!("{}: {e}", args[0])));
    let (_, n) = xenon_gpu::bundle::skip_draws(&cmds, 1, 0);
    let cmds = if targets {
        let k = num(&args[2]);
        println!("last frame: {n} draws; stopping after draw {k}");
        xenon_gpu::bundle::cut_after_draw(&cmds, k)
    } else {
        // With no range, 1..=0 skips nothing.
        let (lo, hi) = match args.len() {
            4 => (num(&args[2]), num(&args[3])),
            _ => (1, 0),
        };
        println!("last frame: {n} draws; skipping {lo}..={hi}");
        xenon_gpu::bundle::skip_draws(&cmds, lo, hi).0
    };
    let mut d = Drawer::new().unwrap_or_else(|e| fail(e.to_string()));
    let report = d
        .replay(&cmds, &mut mem)
        .unwrap_or_else(|e| fail(format!("replay: {e}")));
    for w in report.warnings.iter().chain(&report.unsupported) {
        println!("note: {w}");
    }
    if targets {
        let dir = Path::new(&args[3]);
        std::fs::create_dir_all(dir).unwrap_or_else(|e| fail(format!("{}: {e}", dir.display())));
        let mut keys = d.render_targets();
        keys.sort_by_key(|k| (k.base_tiles, k.pitch));
        for key in keys {
            match target_png(&d, key) {
                Some(f) => {
                    let path = dir.join(format!(
                        "rt_{}_{}_{:?}.png",
                        key.base_tiles, key.pitch, key.format
                    ));
                    f.write_png(&path)
                        .unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
                    println!("{} ({}x{})", path.display(), f.width, f.height);
                }
                None => println!("skipped {key:?}"),
            }
        }
        return;
    }
    let Some(f) = report.frames.last() else {
        fail("the bundle drew no frame".into());
    };
    f.write_png(&args[1])
        .unwrap_or_else(|e| fail(format!("{}: {e}", args[1])));
    println!("{} ({}x{})", args[1], f.width, f.height);
}

/// Render target `key` as RGBA8, for the formats a colour can be read from.
fn target_png(d: &Drawer, key: xenon_draw::RenderTargetKey) -> Option<xenon_draw::Frame> {
    let (w, h) = d.render_target_size(key)?;
    let raw = d.read_render_target_raw(key, 0, 0, w, h).ok()?;
    let unit = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let rgba8: Vec<u8> = match key.format {
        TargetFormat::Rgba8 => raw,
        TargetFormat::Rgb10a2 => raw
            .chunks_exact(4)
            .flat_map(|c| {
                let v = u32::from_le_bytes(c.try_into().unwrap());
                let ch = |s: u32| unit((v >> s & 1023) as f32 / 1023.0);
                [ch(0), ch(10), ch(20), unit((v >> 30) as f32 / 3.0)]
            })
            .collect(),
        TargetFormat::Float7e3 | TargetFormat::Rgba16Float => raw
            .chunks_exact(8)
            .flat_map(|c| std::array::from_fn::<u8, 4, _>(|i| unit(half(c, i))))
            .collect(),
        TargetFormat::Rg16Float => raw
            .chunks_exact(4)
            .flat_map(|c| [unit(half(c, 0)), unit(half(c, 1)), 0, 255])
            .collect(),
        TargetFormat::Float32 => raw
            .chunks_exact(4)
            .flat_map(|c| {
                let v = unit(f32::from_le_bytes(c.try_into().unwrap()));
                [v, v, v, 255]
            })
            .collect(),
        _ => return None,
    };
    Some(xenon_draw::Frame {
        width: w,
        height: h,
        rgba8,
    })
}

/// Half-float `i` of `c`, little-endian.
fn half(c: &[u8], i: usize) -> f32 {
    let h = u16::from_le_bytes([c[2 * i], c[2 * i + 1]]);
    let (e, m) = (i32::from(h >> 10 & 31), f32::from(h & 1023));
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}
