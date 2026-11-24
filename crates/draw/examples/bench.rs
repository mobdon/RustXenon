//! Times a synthetic frame shaped like a real game's: a 1280x720 7e3 render
//! target cleared with a rectangle list, `DRAWS` textured quads, a Convert
//! resolve to the front buffer and a swap. Prints where the time goes.
//!
//! cargo run --release -p xenon-draw --example bench -- [frames] [draws]

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use std::time::{Duration, Instant};
use xenon_gpu::regs::*;
use xenon_texture::{endian, tile};

const FW: u32 = 1280;
const FH: u32 = 720;
const FRONT: u32 = 0x20_0000;
const TEX: u32 = 256;

fn main() {
    let mut args = std::env::args().skip(1);
    // One cold frame plus at least one warm frame to average.
    let frames: u32 = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(10)
        .max(2);
    let draws: u32 = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);
    let mut d = match xenon_draw::Drawer::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("adapter: {}", d.adapter);
    let mut mem = xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; 0x60_0000],
    };
    let linear: Vec<u8> = (0..TEX * TEX)
        .flat_map(|i| [(i % TEX) as u8, (i / TEX) as u8, 128, 255])
        .collect();
    let mut tiled = tile::tile_2d(&linear, TEX, TEX, TEX, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
    let c = [0, 0, 0, 255];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, c),
            (FW as f32, 0.0, 0.0, c),
            (0.0, FH as f32, 0.0, c),
        ]),
    );
    // Quads of 128x128 pixels tiled over the screen.
    let quads: Vec<u32> = (0..draws)
        .flat_map(|i| {
            let x = (i * 128 % FW) as f32;
            let y = ((i * 128 / FW) * 128 % FH) as f32;
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .into_iter()
                .flat_map(move |(u, v)| [f(x + u * 128.0), f(y + v * 128.0), f(0.0), f(u), f(v)])
        })
        .collect();
    let quad_base = 0x5_0000;
    mem.write_words(quad_base, &quads);
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((TEX >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | TEXTURE,
        (TEX - 1) | ((TEX - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
        0,
        1 << 9,
    ];
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[FW, 0, DEPTH_BASE])
        .regs(RB_COLOR_INFO, &[12 << 16])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, FW | (FH << 16)])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12)
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[f(0.25), f(0.5), f(0.75), f(1.0)],
        )
        .draw_auto(8, 3)
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch);
    for i in 0..draws {
        s.vertex_fetch(0, quad_base + i * 80, 20).draw_auto(13, 4);
    }
    s.resolve_with(
        &mut mem,
        (FW, FH),
        COPY_CONVERT,
        (FRONT, FW | (FH << 16), dest_info(6, 0, false)),
        [0; 3],
    )
    .swap_at(FRONT, FW, FH);

    let mut total = xenon_draw::Timings::default();
    let mut wall = Duration::ZERO;
    for i in 0..frames {
        let t = Instant::now();
        let r = run(&mut d, &s, &mut mem);
        let dt = t.elapsed();
        assert_eq!(r.frames.len(), 1, "{r:?}");
        if i == 0 {
            println!("frame 0 (cold caches): {dt:?}, {:?}", r.timings);
            if !r.unsupported.is_empty() {
                println!("unsupported: {:?}", r.unsupported);
            }
            continue;
        }
        wall += dt;
        let t = &r.timings;
        for (a, b) in [
            (&mut total.shaders, t.shaders),
            (&mut total.guest_data, t.guest_data),
            (&mut total.textures, t.textures),
            (&mut total.submit, t.submit),
            (&mut total.gpu_wait, t.gpu_wait),
            (&mut total.readback, t.readback),
            (&mut total.resolve_cpu, t.resolve_cpu),
            (&mut total.clears, t.clears),
            (&mut total.swap, t.swap),
        ] {
            *a += b;
        }
        total.textures_uploaded += t.textures_uploaded;
        total.pipelines_created += t.pipelines_created;
    }
    let n = frames - 1;
    println!(
        "warm frames: {} x {draws} draws, mean {:?} per frame",
        n,
        wall / n
    );
    let t = &total;
    for (name, v) in [
        ("shaders+pipelines", t.shaders),
        ("guest data upload", t.guest_data),
        ("texture decode+upload", t.textures),
        ("encode+submit", t.submit),
        ("GPU wait (rendering)", t.gpu_wait),
        ("resolve readback", t.readback),
        ("resolve CPU convert+tile", t.resolve_cpu),
        ("clears", t.clears),
        ("swap read+detile", t.swap),
    ] {
        println!("  {name:26} {:>10.2?} per frame", v / n);
    }
    println!(
        "  textures uploaded {} / pipelines created {} per frame",
        t.textures_uploaded / n,
        t.pipelines_created / n
    );
}
