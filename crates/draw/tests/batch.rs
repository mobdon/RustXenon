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

/// Fills the 64x64 8_8_8_8 texture at TEXTURE with green and writes, at
/// VERTS + 0x100, a `vs_uv` quad list quad (32,32)-(64,64) showing its
/// top-left quarter. Returns the texture fetch constant.
fn green_texture_quad(mem: &mut xenon_gpu::SliceMemory) -> [u32; 6] {
    let linear: Vec<u8> = (0..W * H).flat_map(|_| GREEN).collect();
    let mut tiled = tile::tile_2d(&linear, W, H, W, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
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
    [
        2 | (2 << 10) | (2 << 13) | ((W >> 5) << 22) | (1 << 31), // clamp, tiled
        6 | (2 << 6) | TEXTURE,                                   // 8_8_8_8, k8in32
        (W - 1) | ((H - 1) << 13),
        swizzle_rgba << 1, // point filters
        0,
        1 << 9, // 2D
    ]
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
    let t = &report.timings;
    assert_eq!(t.submits_by_site.iter().sum::<u32>(), t.submits, "{t:?}");
    // The copy and clears go into the batch with the draws.
    assert_eq!(t.submits_by_site[..3], [0, 0, 0], "{t:?}");
    // The resolve's flush ends the first draw's pass.
    assert_eq!(t.render_passes, 2, "{t:?}");
    assert!(t.queue <= t.submit, "{t:?}");
    let copy = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&copy, W, 40, 40), RED);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
}

/// A texture read again after a resolve rewrote its guest memory shows the
/// new texels: the per-replay texture lookup memo is dropped at resolves
/// that write its bytes.
#[test]
fn resolve_into_a_texture_is_sampled_by_later_draws() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    let fetch = green_texture_quad(&mut mem);
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

/// A resolve to memory that no texture of the replay reads keeps their
/// lookups: the texture is read and hashed once for the draws on both
/// sides of it, and both draw it.
#[test]
fn resolve_elsewhere_keeps_the_texture_lookup() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let fetch = green_texture_quad(&mut mem);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .resolve(&mut mem, W, H, true, Some((rgba(BLUE), 0)))
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    assert_eq!(
        report.timings.hashed_bytes,
        u64::from(W * H * 4),
        "{report:?}"
    );
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    // The resolve cleared to blue; the second draw drew green over it.
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
    let resolved = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&resolved, W, 40, 40), GREEN);
}

/// A MEM_WRITE over a texture's guest memory between two draws with the
/// same fetch constant shows the new texels: the memo is dropped there too.
#[test]
fn mem_write_into_a_texture_is_sampled_by_later_draws() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let fetch = green_texture_quad(&mut mem);
    // The same texture in red, as the words MEM_WRITE (endian 0) stores.
    let linear: Vec<u8> = (0..W * H).flat_map(|_| RED).collect();
    let mut tiled = tile::tile_2d(&linear, W, H, W, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    let mut packet = vec![TEXTURE];
    packet.extend(
        tiled
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
    );
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4);
    s.0.extend(xenon_gpu::pm4::build::type3(
        xenon_gpu::pm4::PM4_MEM_WRITE,
        &packet,
    ));
    s.draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
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

/// Draws whose vertex window lies in one staged in the same batch bind
/// that one (from its base) instead of staging the bytes again.
#[test]
fn vertex_window_in_a_staged_one_is_reused() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, GREEN);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 24)
        // Both rectangles, so the window holds the second one's vertices.
        .draw_auto(PRIM_RECT_LIST, 6)
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert_eq!(report.timings.windows_reused, 2, "{report:?}");
    assert_eq!(report.timings.reused_bytes, 96, "{report:?}");
    // Only the first window is read from guest memory.
    assert_eq!(report.timings.guest_bytes, 96, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), GREEN);
    assert_eq!(px(&img, W, 40, 40), RED);
}

/// A MEM_WRITE into a staged window between two draws makes the second
/// read the new vertices.
#[test]
fn mem_write_into_a_staged_window_reads_it_again() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, VERTS, W as f32, H as f32, RED);
    let green = color_verts(&[
        (0.0, 0.0, 0.0, GREEN),
        (32.0, 0.0, 0.0, GREEN),
        (0.0, 32.0, 0.0, GREEN),
    ]);
    let mut write = vec![VERTS | 2]; // k8in32
    write.extend(green);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    s.0.extend(xenon_gpu::pm4::build::type3(
        xenon_gpu::pm4::PM4_MEM_WRITE,
        &write,
    ));
    s.draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert_eq!(report.timings.windows_reused, 0, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), GREEN);
    assert_eq!(px(&img, W, 40, 40), RED);
}

/// A resolve into a staged window between two draws makes the second draw
/// stage it again and read the resolved bytes: all its vertices are the
/// clear colour's word, so it draws nothing over the cleared target.
#[test]
fn resolve_into_a_staged_window_reads_it_again() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, DEST, W as f32, H as f32, RED);
    let clear = [10, 20, 30, 255];
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, DEST, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .resolve(&mut mem, W, H, false, Some((rgba(clear), 0)))
        .resolve(&mut mem, W, H, true, None)
        .vertex_fetch(0, DEST, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert_eq!(report.timings.windows_reused, 0, "{report:?}");
    assert!(report.timings.submits >= 2, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), clear);
    assert_eq!(px(&img, W, 40, 40), clear);
}

/// A window staged before a resolve elsewhere is bound again after it: the
/// resolve submits the batch, but the arena keeps the window.
#[test]
fn vertex_window_is_reused_across_a_resolve() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, VERTS, W as f32, H as f32, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, GREEN);
    let clear = [10, 20, 30, 255];
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 24)
        // Both rectangles, so the window holds the second draw's vertices.
        .draw_auto(PRIM_RECT_LIST, 6)
        .resolve(&mut mem, W, H, true, Some((rgba(clear), 0)))
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    assert_eq!(report.timings.windows_reused, 1, "{report:?}");
    assert_eq!(report.timings.guest_bytes, 96, "{report:?}");
    let copy = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&copy, W, 40, 40), RED);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), GREEN);
    assert_eq!(px(&img, W, 40, 40), clear);
}

/// A staged window that starts before a resolve destination and runs into
/// it is read again after the resolve: the draw from the destination sees
/// the resolved words (one point, drawing nothing), not the old red
/// rectangle there.
#[test]
fn vertex_window_over_a_resolve_destination_reads_it_again() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, DEST - 48, 32.0, 32.0, GREEN);
    rect(&mut mem, DEST, W as f32, H as f32, RED);
    let clear = [10, 20, 30, 255];
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, DEST - 48, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .resolve(&mut mem, W, H, false, Some((rgba(clear), 0)))
        .resolve(&mut mem, W, H, true, None)
        .vertex_fetch(0, DEST, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 2), "{report:?}");
    assert_eq!(report.timings.windows_reused, 0, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), clear);
    assert_eq!(px(&img, W, 40, 40), clear);
}

/// A draw with the same constants as the one before it in the batch binds
/// that block again; one whose pixel constant changed stages its own, and
/// each draw shows its own colour.
#[test]
fn equal_constants_are_staged_once_per_batch() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, VERTS, W as f32, H as f32, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, RED);
    let c = |v: [u8; 4]| v.map(|x| f(f32::from(x) / 255.0));
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 24)
        .regs(SHADER_CONSTANT_FLOAT + 256 * 4, &c(GREEN))
        .draw_auto(PRIM_RECT_LIST, 3)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(SHADER_CONSTANT_FLOAT + 256 * 4, &c(BLUE))
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert_eq!(report.timings.consts_reused, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
}

/// Draws in one batch that sample the same texture share one texture bind
/// group; one that samples another texture gets its own.
#[test]
fn texture_bind_groups_follow_the_textures() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let green = green_texture_quad(&mut mem);
    // The same texture in red, one texture further on.
    let red_base = TEXTURE + W * H * 4;
    let linear: Vec<u8> = (0..W * H).flat_map(|_| RED).collect();
    let mut tiled = tile::tile_2d(&linear, W, H, W, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[red_base as usize..][..tiled.len()].copy_from_slice(&tiled);
    let mut red = green;
    red[1] = 6 | (2 << 6) | red_base;
    // Quads (0,0)-(32,32) and (0,32)-(32,64), after the one at (32,32).
    for (i, y) in [0.0, 32.0].into_iter().enumerate() {
        let quad: Vec<u32> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .flat_map(|&(u, v): &(f32, f32)| {
                [f(u * 32.0), f(y + v * 32.0), f(0.0), f(u / 2.0), f(v / 2.0)]
            })
            .collect();
        mem.write_words(VERTS + 0x200 + 0x100 * i as u32, &quad);
    }
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &green)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .regs(SHADER_CONSTANT_FETCH + 6, &red)
        .vertex_fetch(0, VERTS + 0x200, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .regs(SHADER_CONSTANT_FETCH + 6, &green)
        .vertex_fetch(0, VERTS + 0x300, 20)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert_eq!(report.timings.texture_groups_reused, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 40, 40), GREEN);
    assert_eq!(px(&img, W, 8, 8), RED);
    assert_eq!(px(&img, W, 8, 40), GREEN);
}

/// Draws whose arena slices have the same sizes share one group 0 bind
/// group at their own offsets, and none outlives the arena: after a draw
/// grows it, a draw of the old sizes binds the new one.
#[test]
fn arena_bind_groups_bind_each_draws_data() {
    let Some(mut d) = drawer() else { return };
    let mut mem = xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; 20 << 20],
    };
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, GREEN);
    rect(&mut mem, VERTS + 96, 8.0, 8.0, RED);
    rect(&mut mem, VERTS + 144, 16.0, 16.0, BLUE);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        // A 17 MiB window: more than the 16 MiB arena, which is replaced.
        .vertex_fetch(0, VERTS + 144, (17 << 20) / 4)
        .draw_auto(PRIM_RECT_LIST, 3)
        // Not in the large window, so not bound from it.
        .vertex_fetch(0, VERTS + 96, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 4, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 4), RED);
    assert_eq!(px(&img, W, 12, 12), BLUE);
    assert_eq!(px(&img, W, 24, 24), GREEN);
    assert_eq!(px(&img, W, 40, 40), RED);
}

/// A draw whose vertex window is not in guest memory is skipped, with the
/// window in the report, and the draws staged around it still land.
#[test]
fn unreadable_vertex_window_skips_only_its_draw() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    rect(&mut mem, VERTS, W as f32, H as f32, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, GREEN);
    let outside = mem.data.len() as u32;
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .vertex_fetch(0, outside, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    let why = format!(
        "vertex data at 0x{outside:08X}..0x{:08X} not readable",
        outside + 48
    );
    assert!(
        report.unsupported.iter().any(|u| u.contains(&why)),
        "{report:?}"
    );
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), GREEN);
    assert_eq!(px(&img, W, 40, 40), RED);
}

/// Resolve clears are recorded into the next batch instead of submitted on
/// their own: after the copy that reads the target, before the draws after
/// them, and seen by the next resolve's copy.
#[test]
fn resolve_clears_join_the_next_batch() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    rect(&mut mem, VERTS + 48, 32.0, 32.0, BLUE);
    let dest2 = DEST + 0x4000;
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        // Copies red, then clears colour to green and depth.
        .resolve(&mut mem, W, H, true, Some((rgba(GREEN), 0)))
        // Copies the cleared target, clears nothing.
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_RAW,
            (dest2, W | (H << 16), dest_info(6, 0, false)),
            [0; 3],
        )
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 2), "{report:?}");
    let copy = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&copy, W, 40, 40), RED);
    let copy = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!(px(&copy, W, 8, 8), GREEN);
    assert_eq!(px(&copy, W, 40, 40), GREEN);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
    // The copies and clears are recorded in the batch: 4 flushes and
    // write-backs, where a submit per copy and a flush before each made 7.
    let t = &report.timings;
    assert_eq!(t.submits_by_site, [0, 0, 0, 4], "{t:?}");
    assert_eq!(t.submits, 4, "{t:?}");
}

/// A resolve copy taken on the CPU (16_16_FLOAT destination) counts its
/// readback submit as a resolve submit.
#[test]
fn cpu_resolve_readback_counts_as_resolve_submit() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let wf = W as f32;
    rect(&mut mem, VERTS, wf, wf, RED);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_RAW,
            (DEST, W | (H << 16), dest_info(31, 0, false)),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 1), "{report:?}");
    let t = &report.timings;
    assert_eq!(t.submits_by_site.iter().sum::<u32>(), t.submits, "{t:?}");
    assert_eq!(t.submits_by_site[0], 1, "{t:?}");
}

/// A texture bind group outlives the batch: draws on either side of a
/// resolve that sample the same texture share it.
#[test]
fn texture_bind_group_is_reused_across_a_resolve() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let green = green_texture_quad(&mut mem);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &green)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .resolve(&mut mem, W, H, true, None)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert_eq!(report.timings.texture_groups_reused, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 40, 40), GREEN);
}

#[test]
fn gpu_timestamps_time_each_batch_pass() {
    // Not requested unless asked for.
    {
        let Some(mut plain) = drawer() else { return };
        assert!(!plain.enable_gpu_timestamps());
    }
    let Some(mut d) = drawer_with_gpu_timestamps() else {
        return;
    };
    if !d.enable_gpu_timestamps() {
        eprintln!("skipped: no timestamp queries on this device");
        return;
    }
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
    let t = &report.timings;
    eprintln!("{:?} over {} passes", t.gpu_passes, t.passes_timed);
    // Two draw passes, the resolve copy and its clear at least.
    assert!(t.passes_timed >= 4, "{t:?}");
    assert!(t.gpu_passes[0] > std::time::Duration::ZERO, "{t:?}");
    // Timing changes nothing drawn.
    let copy = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&copy, W, 40, 40), RED);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), BLUE);
    assert_eq!(px(&img, W, 40, 40), GREEN);
    // A second replay reads only its own passes.
    let again = run(&mut d, &s, &mut mem).timings;
    assert_eq!(again.passes_timed, t.passes_timed, "{again:?}");
}
