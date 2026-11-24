//! Draws a synthetic stream (clear, a coloured triangle, resolve, swap)
//! headlessly and writes the presented frame as a PNG.
//!
//! cargo run -p xenon-draw --example triangle -- [out.png]
//! (default: target/xenon-draw-triangle.png)

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/xenon-draw-triangle.png".into());
    let mut drawer = match xenon_draw::Drawer::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (32.0, 4.0, 0.0, [255, 0, 0, 255]),
            (60.0, 58.0, 0.0, [0, 255, 0, 255]),
            (4.0, 58.0, 0.0, [0, 0, 255, 255]),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba([16, 16, 32, 255]), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(4, 3)
        .resolve(&mut mem, W, H, true, None)
        .swap(W, H);
    let report = run(&mut drawer, &s, &mut mem);
    let Some(frame) = report.frames.first() else {
        eprintln!("error: no frame: {report:?}");
        std::process::exit(1);
    };
    if let Err(e) = frame.write_png(&out) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    println!(
        "{} draws, {} resolves on {}; wrote {out}",
        report.draws, report.resolves, drawer.adapter
    );
}
