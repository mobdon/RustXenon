//! Replays one synthetic 1280x720 frame over and over, shaped to churn the
//! drawer's resources the way a long capture does, and prints the process
//! RSS (from /proc/self/statm) as it goes. Each frame:
//!
//! - draws a full-screen colour rectangle into a target at the depth
//!   target's EDRAM tiles (a depth restore drawn as colour), then writes
//!   all depth rows, so that colour target is dropped at the end of the
//!   replay and made again in the next one;
//! - draws `textures` quads, each sampling its own texture (64 to 512
//!   texels square, by slot) whose bytes change every frame, with a
//!   texture cache of `cache MB`, too small to keep them, so every texture
//!   is uploaded and older ones are evicted;
//! - draws `draws` more quads over those textures, with `resolves - 1`
//!   scratch resolves spread among them;
//! - resolves to a front buffer and swaps.
//!
//! The drawer's own buckets (memory_line) stay flat; RSS should too. On
//! lavapipe GPU memory is host memory, so RSS includes it.
//!
//! cargo run --release -p xenon-draw --example churn -- [frames] [every] [textures] [draws] [resolves] [cache MB]

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use xenon_gpu::regs::*;

const FW: u32 = 1280;
const FH: u32 = 720;
/// 720 tiles of colour at tile 0, then depth.
const DEPTH_AT: u32 = 1024;
const FRONT: u32 = 0x20_0000;
const TEXTURES: u32 = 0x100_0000;
/// Texture slots are 1 MB apart; slot `i` is `tex(i)` texels square.
const SLOT: u32 = 0x10_0000;

fn tex(i: u32) -> u32 {
    64 << (i % 4)
}

// Past the command stream, which grows with `draws`.
const FULL: u32 = 0x90_0000;
const QUADS: u32 = 0xA0_0000;
const SCRATCH: u32 = 0x60_0000;

fn rss_mb() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    (pages * 4096) >> 20
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut arg = |default: u32| args.next().and_then(|a| a.parse().ok()).unwrap_or(default);
    let (frames, every, textures) = (arg(1000), arg(100).max(1), arg(8).max(1));
    let (draws, resolves, cache_mb) = (arg(0), arg(1).max(1), arg(1));
    let mut d = match xenon_draw::Drawer::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("adapter: {}", d.adapter);
    d.texture_cache_limit = u64::from(cache_mb) << 20;
    let mut mem = xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; (TEXTURES + textures * SLOT) as usize],
    };
    let full = |z: f32| {
        color_verts(&[
            (0.0, 0.0, z, [0; 4]),
            (FW as f32, 0.0, z, [0; 4]),
            (0.0, FH as f32, z, [0; 4]),
        ])
    };
    mem.write_words(FULL, &full(0.5));
    let quads: Vec<u32> = (0..textures.max(draws))
        .flat_map(|i| {
            let x = (i * 128 % FW) as f32;
            let y = ((i * 128 / FW) * 128 % FH) as f32;
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .into_iter()
                .flat_map(move |(u, v)| [f(x + u * 128.0), f(y + v * 128.0), f(0.0), f(u), f(v)])
        })
        .collect();
    mem.write_words(QUADS, &quads);

    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[FW, DEPTH_AT, DEPTH_AT])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, FW | (FH << 16)])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[f(0.25), f(0.5), f(0.75), f(1.0)],
        )
        .vertex_fetch(0, FULL, 12)
        .draw_auto(8, 3)
        // Depth only, z ALWAYS with z write, over every row.
        .regs(RB_COLOR_INFO, &[0])
        .regs(RB_COLOR_MASK, &[0])
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .draw_auto(8, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_COLOR_MASK, &[0xF])
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1));
    for i in 0..textures + draws {
        if draws > 0 && i > textures && (i - textures) % draws.div_ceil(resolves).max(1) == 0 {
            s.resolve_with(
                &mut mem,
                (FW, FH),
                COPY_CONVERT,
                (SCRATCH, FW | (FH << 16), dest_info(6, 0, false)),
                [0; 3],
            );
        }
        let t = tex(i % textures);
        let fetch = [
            2 | (2 << 10) | (2 << 13) | ((t >> 5) << 22) | (1 << 31),
            6 | (2 << 6) | (TEXTURES + i % textures * SLOT),
            (t - 1) | ((t - 1) << 13),
            ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
            0,
            1 << 9,
        ];
        s.regs(SHADER_CONSTANT_FETCH + 6, &fetch)
            .vertex_fetch(0, QUADS + i % textures.max(draws) * 80, 20)
            .draw_auto(13, 4);
    }
    s.resolve_with(
        &mut mem,
        (FW, FH),
        COPY_CONVERT,
        (FRONT, FW | (FH << 16), dest_info(6, 0, false)),
        [0; 3],
    )
    .swap_at(FRONT, FW, FH);

    let start = rss_mb();
    println!("frame 0: rss {start} MB");
    for frame in 1..=frames {
        // New bytes in every texture, so none is a cache hit.
        for i in 0..textures {
            let at = (TEXTURES + i * SLOT) as usize;
            mem.data[at..at + 4].copy_from_slice(&frame.to_le_bytes());
        }
        let r = run(&mut d, &s, &mut mem);
        assert_eq!(r.frames.len(), 1, "{r:?}");
        if frame % every == 0 {
            println!(
                "frame {frame}: rss {} MB (+{}), {}",
                rss_mb(),
                rss_mb().saturating_sub(start),
                d.memory_line()
            );
        }
    }
}
