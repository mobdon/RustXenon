//! Draw batching: draws are recorded into one command buffer and submitted
//! at a resolve or the end of the replay, without changing what they draw.

mod common;

use common::*;
use xenon_draw::RenderTargetKey;
use xenon_gpu::regs::*;
use xenon_texture::{endian, tile};

const PRIM_RECT_LIST: u32 = 8;
const PRIM_QUAD_LIST: u32 = 13;
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

/// Writes a `vs_color` rectangle list rectangle (0,0)-(w,h) in `c` at `addr`.
fn rect(mem: &mut xenon_gpu::SliceMemory, addr: u32, w: f32, h: f32, c: [u8; 4]) {
    mem.write_words(
        addr,
        &color_verts(&[(0.0, 0.0, 0.0, c), (w, 0.0, 0.0, c), (0.0, h, 0.0, c)]),
    );
}

/// Draws to one target, another, then the first again: one submission for
/// the three draws (two render passes), in order.
#[test]
fn draws_until_the_end_share_one_submit() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, wf, wf, GREEN);
    rect(&mut mem, VERTS + 96, 32.0, 32.0, BLUE);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_COLOR_INFO, &[256])
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_COLOR_INFO, &[0])
        .vertex_fetch(0, VERTS + 96, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert_eq!(report.timings.submits, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), RED);
    let other = RenderTargetKey {
        base_tiles: 256,
        ..COLOR_KEY
    };
    let img = d.read_render_target(other, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 40, 40), GREEN);
}

/// A resolve submits the draws before it: its copy sees them and its clear
/// lands under the draws after it.
#[test]
fn resolve_sees_batched_draws() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, BLUE);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .resolve(&mut mem, W, H, true, Some((rgba(GREEN), 0)))
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    let copy = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&copy, W, 40, 40), RED);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
}

/// A texture read again after a resolve rewrote its guest memory shows the
/// new texels: the per-replay texture lookup memo is dropped at resolves.
#[test]
fn resolve_into_a_texture_is_sampled_by_later_draws() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let linear: Vec<u8> = (0..W * H).flat_map(|_| GREEN).collect();
    let mut tiled = tile::tile_2d(&linear, W, H, W, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    // Quad (32,32)-(64,64) showing the texture's top-left quarter.
    let quad: Vec<u32> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        .iter()
        .flat_map(|&(u, v): &(f32, f32)| {
            [
                f(32.0 + u * 32.0),
                f(32.0 + v * 32.0),
                f(0.0),
                f(u / 2.0),
                f(v / 2.0),
            ]
        })
        .collect();
    mem.write_words(VERTS + 0x100, &quad);
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((W >> 5) << 22) | (1 << 31), // clamp, tiled
        6 | (2 << 6) | TEXTURE,                                   // 8_8_8_8, k8in32
        (W - 1) | ((H - 1) << 13),
        swizzle_rgba << 1, // point filters
        0,
        1 << 9, // 2D
    ];
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4);
    // The target (red, the quad green) becomes the texture.
    s.resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (TEXTURE, W | (H << 16), dest_info(6, 0, false)),
        [0; 3],
    )
    .vertex_fetch(0, VERTS + 0x100, 20)
    .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (3, 1), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), RED);
    assert_eq!(px(&img, W, 40, 40), RED);
}

/// More draw data than the 16 MiB upload arena holds submits the batch
/// part-way and reuses the arena; the draws still land in order.
#[test]
fn a_full_upload_arena_submits_and_continues() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, wf, wf, GREEN);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp());
    // About 9 KiB of constants per draw: 2001 draws need two arena fills.
    for i in 0..2001 {
        s.vertex_fetch(0, VERTS + 48 * (i % 2), 12)
            .draw_auto(PRIM_RECT_LIST, 3);
    }
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2001, "{report:?}");
    assert_eq!(report.timings.submits, 2, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 40, 40), RED);
}
