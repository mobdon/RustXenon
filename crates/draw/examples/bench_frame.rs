//! Replays a frame bundle (`--capture-bundle`) `n` times on one drawer and
//! prints each replay's timings, to measure the drawer on a real frame.
//!
//! cargo run --release -p xenon-draw --example bench_frame -- <bundle dir> [n] [out.png]
//!
//! writes the first replay's last frame to `out.png` (and the last
//! replay's to `out.png.last.png`). `XENON_GPU_TIMESTAMPS=1` adds GPU time
//! per pass kind; it waits for the GPU at every flush, so the wall time
//! is then longer.

use std::path::Path;
use std::time::Instant;
use xenon_draw::Drawer;

fn fail(why: String) -> ! {
    eprintln!("error: {why}");
    std::process::exit(1);
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        eprintln!("usage: bench_frame <bundle dir> [n] [out.png]");
        std::process::exit(2);
    }
    let n: u32 = args.get(1).map_or(5, |a| a.parse().unwrap_or(5)).max(1);
    let (cmds, mut mem) = xenon_gpu::bundle::read(Path::new(&args[0]))
        .unwrap_or_else(|e| fail(format!("{}: {e}", args[0])));
    let mut d = Drawer::new().unwrap_or_else(|e| fail(e.to_string()));
    println!("{}", d.memory_line());
    let mut best = f64::MAX;
    for i in 0..n {
        let t0 = Instant::now();
        let r = d
            .replay(&cmds, &mut mem)
            .unwrap_or_else(|e| fail(format!("replay: {e}")));
        let wall = t0.elapsed();
        if i > 0 {
            best = best.min(ms(wall));
        }
        let t = &r.timings;
        println!(
            "replay {i}: wall {:.1} ms | textures {:.1} gpu_wait {:.1} readback {:.1} resolve_cpu {:.1} submit {:.1} finish_thread {:.1} submit_thread {:.1} transfer {:.1} clears {:.1} swap {:.1} pipeline {:.1}",
            ms(wall),
            ms(t.textures),
            ms(t.gpu_wait),
            ms(t.readback),
            ms(t.resolve_cpu),
            ms(t.submit),
            ms(t.finish_thread),
            ms(t.submit_thread),
            ms(t.transfer),
            ms(t.clears),
            ms(t.swap),
            ms(t.pipeline),
        );
        println!(
            "  render passes {} transfers {} transfer passes {} submits {} {:?} readback {} KiB writebacks {} | attachments loaded {} cleared {} stored {} discarded {}",
            t.render_passes,
            t.transfers,
            t.transfer_passes,
            t.submits,
            t.submits_by_site,
            t.readback_bytes >> 10,
            t.writebacks,
            t.attachment_ops[0],
            t.attachment_ops[1],
            t.attachment_ops[2],
            t.attachment_ops[3],
        );
        if t.passes_timed > 0 {
            let g = t.gpu_passes;
            println!(
                "  GPU {:.1} ms over {} passes: draw {:.1} clear {:.1} resolve {:.1} transfer {:.1}",
                ms(g.iter().sum()),
                t.passes_timed,
                ms(g[0]),
                ms(g[1]),
                ms(g[2]),
                ms(g[3]),
            );
        }
        for w in r.warnings.iter().chain(&r.unsupported).filter(|_| i == 0) {
            println!("  note: {w}");
        }
        if let (Some(out), Some(f)) = (args.get(2), r.frames.last()) {
            if i == 0 || i == n - 1 {
                let path = if i == 0 {
                    out.clone()
                } else {
                    format!("{out}.last.png")
                };
                f.write_png(&path)
                    .unwrap_or_else(|e| fail(format!("{path}: {e}")));
            }
        }
    }
    if n > 1 {
        println!("best warm replay: {best:.1} ms");
    }
}
