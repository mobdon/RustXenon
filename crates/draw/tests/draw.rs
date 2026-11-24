//! End-to-end drawer tests on synthetic streams. They need a wgpu adapter
//! (lavapipe works); without one they print a skip line, unless
//! XENON_REQUIRE_GPU=1 makes them fail.

mod common;

use common::*;
use xenon_draw::{RenderTargetKey, TargetFormat};
use xenon_gpu::regs::*;
use xenon_texture::{endian, tile};

const CLEAR: [u8; 4] = [10, 20, 30, 255];
const PRIM_TRI_LIST: u32 = 4;
const PRIM_TRI_STRIP: u32 = 6;
const PRIM_RECT_LIST: u32 = 8;
const PRIM_LINE_LOOP: u32 = 12;
const PRIM_QUAD_LIST: u32 = 13;

#[test]
fn resolve_clear_fills_render_target() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0xFFFF_FF00)));
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert!(img.chunks_exact(4).all(|p| p == CLEAR));
}

/// Clears, then draws a triangle with per-vertex colours through the
/// Direct3D-style viewport transform (clip space, VTE scale/offset on,
/// pixel centres at .0).
#[test]
fn triangle_interpolates_vertex_colours() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (-1.0, 1.0, 0.0, [255, 0, 0, 255]),
            (1.0, 1.0, 0.0, [0, 255, 0, 255]),
            (-1.0, -1.0, 0.0, [0, 0, 255, 255]),
        ]),
    );
    let half = (W / 2) as f32;
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .regs(PA_CL_VTE_CNTL, &[0x3F | (1 << 10)])
        .regs(PA_SU_VTX_CNTL, &[0])
        .regs(
            PA_CL_VPORT_XSCALE,
            &[f(half), f(half), f(-half), f(half), f(1.0), f(0.0)],
        )
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    // Pixel (8, 8) is clip (-0.75, 0.75): weights 0.75 red, 0.125 green,
    // 0.125 blue.
    let p = px(&img, W, 8, 8);
    assert!(near(p, [191, 32, 32, 255], 2), "{p:?}");
    // Below the diagonal: untouched.
    assert_eq!(px(&img, W, 60, 40), CLEAR);
    assert_eq!(px(&img, W, 33, 63), CLEAR);
}

/// A triangle strip from a byte-swapped 16-bit index buffer covers the
/// 32x32 square at the origin.
#[test]
fn indexed_strip_with_dma_16_bit_indices() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let white = [255, 255, 255, 255];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, white),
            (32.0, 0.0, 0.0, white),
            (0.0, 32.0, 0.0, white),
            (32.0, 32.0, 0.0, white),
        ]),
    );
    // Big-endian 16-bit indices 3 1 2 0 (k8in16 swaps them back).
    mem.write_words(INDICES, &[0x0003_0001, 0x0002_0000]);
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 16)
        .draw_dma16(PRIM_TRI_STRIP, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    for (x, y) in [(0, 0), (31, 0), (0, 31), (31, 31), (16, 16)] {
        assert_eq!(px(&img, W, x, y), white, "({x}, {y})");
    }
    for (x, y) in [(32, 0), (0, 32), (32, 32), (63, 63)] {
        assert_eq!(px(&img, W, x, y), CLEAR, "({x}, {y})");
    }
}

/// A quad list with a tiled 8_8_8_8 texture, nearest sampling: every
/// pixel of the quad is exactly its texel.
#[test]
fn textured_quad_samples_tiled_texture() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let n = 32u32;
    let texel = |x: u32, y: u32| [(x * 8) as u8, (y * 8) as u8, 128, 255];
    let linear: Vec<u8> = (0..n * n).flat_map(|i| texel(i % n, i / n)).collect();
    let mut tiled = tile::tile_2d(&linear, n, n, n, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
    let q = n as f32;
    let verts: Vec<u32> = [
        (0.0, 0.0, 0.0, 0.0),
        (q, 0.0, 1.0, 0.0),
        (q, q, 1.0, 1.0),
        (0.0, q, 0.0, 1.0),
    ]
    .iter()
    .flat_map(|&(x, y, u, v)| [f(x), f(y), f(0.0), f(u), f(v)])
    .collect();
    mem.write_words(VERTS, &verts);
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31), // clamp, tiled
        6 | (2 << 6) | TEXTURE,                                   // 8_8_8_8, k8in32
        (n - 1) | ((n - 1) << 13),
        swizzle_rgba << 1, // point filters
        0,
        1 << 9, // 2D
    ];
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 20)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    for y in 0..n {
        for x in 0..n {
            assert_eq!(px(&img, W, x, y), texel(x, y), "({x}, {y})");
        }
    }
    assert_eq!(px(&img, W, n, n), CLEAR);
    assert_eq!(report.timings.textures_uploaded, 1);

    // Unchanged guest bytes come from the texture cache.
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 0, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 5, 7), texel(5, 7));

    // Rewritten guest bytes under the same fetch constant are decoded again.
    let texel2 = |x: u32, y: u32| [255 - (x * 8) as u8, (y * 8) as u8, 64, 255];
    let linear: Vec<u8> = (0..n * n).flat_map(|i| texel2(i % n, i / n)).collect();
    let mut tiled = tile::tile_2d(&linear, n, n, n, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    for y in 0..n {
        for x in 0..n {
            assert_eq!(px(&img, W, x, y), texel2(x, y), "({x}, {y})");
        }
    }
}

/// The pixel shader's colour comes from its float constant c0.
#[test]
fn pixel_shader_float_constant() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let black = [0, 0, 0, 255];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, black),
            (64.0, 0.0, 0.0, black),
            (0.0, 64.0, 0.0, black),
        ]),
    );
    let c = |v: u32| f(v as f32 / 255.0);
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12)
        // Pixel c0 is float constant 256.
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[c(51), c(102), c(204), c(255)],
        )
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 4), [51, 102, 204, 255]);
    assert_eq!(px(&img, W, 60, 60), CLEAR);
}

/// Depth test hides a farther triangle drawn later; an alpha-blended
/// triangle without depth test mixes with what is below.
#[test]
fn depth_test_and_blending() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    let blue = [0, 0, 255, 128];
    let a = color_verts(&[
        (0.0, 0.0, 0.5, red),
        (40.0, 0.0, 0.5, red),
        (0.0, 40.0, 0.5, red),
    ]);
    let b = color_verts(&[
        (0.0, 0.0, 0.7, green),
        (64.0, 0.0, 0.7, green),
        (0.0, 64.0, 0.7, green),
    ]);
    let c = color_verts(&[
        (0.0, 0.0, 0.1, blue),
        (64.0, 0.0, 0.1, blue),
        (64.0, 64.0, 0.1, blue),
    ]);
    mem.write_words(VERTS, &[a, b, c].concat());
    let depth_less_write = 2 | 4 | (1 << 4);
    // Colour and alpha: src SrcAlpha (6), add, dst OneMinusSrcAlpha (7).
    let alpha_blend = 6 | (7 << 8) | (6 << 16) | (7 << 24);
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0xFFFF_FF00)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[depth_less_write])
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_BLENDCONTROL0, &[alpha_blend])
        .vertex_fetch(0, VERTS + 96, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 10), red, "nearer red wins");
    assert_eq!(px(&img, W, 5, 45), green, "green outside red");
    // Blue at alpha 128/255 over red.
    let p = px(&img, W, 20, 10);
    assert!(near(p, [127, 0, 128, 192], 2), "{p:?}");
    assert_eq!(px(&img, W, 62, 63), CLEAR);
}

fn draw_pattern(mem: &mut xenon_gpu::SliceMemory, s: &mut Stream) {
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, [255, 0, 0, 255]),
            (64.0, 0.0, 0.0, [0, 255, 0, 255]),
            (0.0, 64.0, 0.0, [0, 0, 255, 255]),
        ]),
    );
    s.setup()
        .resolve(mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .resolve(mem, W, H, true, None);
}

/// A resolve writes the render target to guest memory as a tiled k8in32
/// 8_8_8_8 surface.
#[test]
fn resolve_copy_matches_render_target() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 2), "{report:?}");
    let rt = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let extent = tile::tiled_extent_2d(W, H, W, 4).unwrap() as usize;
    let mut raw = mem.data[DEST as usize..][..extent].to_vec();
    endian::swap_in_place(&mut raw, xenon_texture::Endian::K8in32);
    assert_eq!(tile::detile_2d(&raw, W, H, W, 4).unwrap(), rt);
    assert_ne!(px(&rt, W, 2, 2), CLEAR);
}

/// Swap reads the resolved front buffer into a frame; its PNG decodes back
/// to the same pixels.
#[test]
fn swap_frame_png_round_trip() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    s.swap(W, H);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.frames.len(), 1, "{report:?}");
    let frame = &report.frames[0];
    let rt = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!((frame.width, frame.height), (W, H));
    assert_eq!(frame.rgba8, rt);
    let bytes = frame.to_png_bytes().unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!((info.width, info.height), (W, H));
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(&buf[..info.buffer_size()], &frame.rgba8[..]);
}

/// Unsupported render target formats and primitive types are reported and
/// skipped, and replay goes on.
#[test]
fn unsupported_format_and_primitive_are_reported() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let white = [255, 255, 255, 255];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, white),
            (64.0, 0.0, 0.0, white),
            (0.0, 64.0, 0.0, white),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_LINE_LOOP, 3)
        // k_16_16 colour target.
        .regs(RB_COLOR_INFO, &[4 << 16])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_COLOR_INFO, &[0])
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert_eq!(
        report.unsupported,
        vec![
            "primitive type LineLoop".to_string(),
            "colour render target format 4".to_string()
        ]
    );
    let missing = RenderTargetKey {
        base_tiles: 1,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    assert!(d.read_render_target(missing, 0, 0, 1, 1).is_err());
}

const WHITE: [u8; 4] = [255, 255, 255, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// A triangle covering the whole 64x64 target in window pixels.
fn full_triangle(c: [u8; 4], z: f32) -> Vec<u32> {
    color_verts(&[(0.0, 0.0, z, c), (128.0, 0.0, z, c), (0.0, 128.0, z, c)])
}

/// Two triangles in one list: `a` is counter-clockwise on screen (window
/// y down) in the top-left corner, `b` clockwise in the bottom-right one.
fn winding_pair() -> Vec<u32> {
    color_verts(&[
        (0.0, 0.0, 0.0, RED),
        (0.0, 32.0, 0.0, RED),
        (32.0, 0.0, 0.0, RED),
        (32.0, 32.0, 0.0, GREEN),
        (64.0, 32.0, 0.0, GREEN),
        (32.0, 64.0, 0.0, GREEN),
    ])
}

/// PA_SU_SC_MODE_CNTL face bit 2 (0: front is counter-clockwise, as
/// Direct3D 9 sees it on screen) and cull_front / cull_back.
#[test]
fn cull_mode_and_front_face() {
    let Some(mut d) = drawer() else { return };
    // (mode, top-left red drawn, bottom-right green drawn)
    for (mode, a, b) in [
        (2, true, false),     // front CCW, cull back: CW culled
        (1, false, true),     // front CCW, cull front: CCW culled
        (2 | 4, false, true), // front CW, cull back: CCW culled
        (1 | 4, true, false), // front CW, cull front: CW culled
        (0, true, true),
    ] {
        let mut mem = memory();
        mem.write_words(VERTS, &winding_pair());
        let mut s = Stream::default();
        s.setup()
            .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
            .shader(false, &vs_color())
            .shader(true, &ps_interp())
            .vertex_fetch(0, VERTS, 24)
            .regs(PA_SU_SC_MODE_CNTL, &[mode])
            .draw_auto(PRIM_TRI_LIST, 6);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.draws, 1, "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        let want = |on, c| if on { c } else { CLEAR };
        assert_eq!(px(&img, W, 4, 4), want(a, RED), "mode {mode}");
        assert_eq!(px(&img, W, 36, 36), want(b, GREEN), "mode {mode}");
    }
}

/// RB_COLOR_MASK keeps the masked channels of the render target.
#[test]
fn colour_write_mask() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba([10, 20, 30, 40]), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(RB_COLOR_MASK, &[0b0101]) // red and blue
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), [255, 20, 255, 40]);
}

/// z_write_enable without z_enable writes no depth (Xenia ignores it), so
/// a later depth-tested draw behind the first one still passes.
#[test]
fn depth_write_needs_depth_test() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &[full_triangle(RED, 0.5), full_triangle(GREEN, 0.7)].concat(),
    );
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0xFFFF_FF00)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[4]) // write only
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (1 << 4)]) // less, write
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), GREEN);
}

/// The window scissor (offset disabled) limits a full-target draw.
#[test]
fn window_scissor_limits_draw() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(
            PA_SC_WINDOW_SCISSOR_TL,
            &[(1 << 31) | 8 | (4 << 16), 24 | (20 << 16)],
        )
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    for (x, y, inside) in [
        (8, 4, true),
        (23, 19, true),
        (7, 4, false),
        (8, 3, false),
        (24, 19, false),
        (23, 20, false),
    ] {
        let want = if inside { WHITE } else { CLEAR };
        assert_eq!(px(&img, W, x, y), want, "({x}, {y})");
    }
}

/// Binding contract: VGT_INDX_OFFSET is `vertex_base` (the last fields of
/// the Consts uniform), and a vertex fetch constant other than 0 (number 5:
/// dwords 10 and 11, `fetch[2].zw`) through a cached (0xA...) address.
#[test]
fn index_offset_and_vertex_fetch_constant_packing() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    // Three vertices that would cover the top-left pixel, then the real
    // triangle over the bottom-right half.
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, RED),
            (16.0, 0.0, 0.0, RED),
            (0.0, 16.0, 0.0, RED),
            (64.0, 0.0, 0.0, GREEN),
            (64.0, 64.0, 0.0, GREEN),
            (0.0, 64.0, 0.0, GREEN),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color_from(5))
        .shader(true, &ps_interp())
        .vertex_fetch(5, 0xA000_0000 | VERTS, 24)
        .regs(VGT_INDX_OFFSET, &[3])
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 60, 60), GREEN);
    assert_eq!(px(&img, W, 1, 1), CLEAR);
}

/// Binding contract: bool constant 37 is bit 5 of flattened word 1
/// (`bool_consts[0][1]`), and pixel c1 is float constant 257.
#[test]
fn bool_constant_bit_order() {
    let Some(mut d) = drawer() else { return };
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let cexec = Exec::new(1, 1, true).with_cond(ExecCond::Bool {
        index: 37,
        value: true,
        predicate_clean: false,
    });
    // oC0 = c1, then oC0 = c0 if b37.
    let ps = assemble(
        &[exec(0, 1, false), CfInstr::Exec(cexec)],
        &[
            export(0, AluSrc::constant(1, SWZ_XYZW)),
            export(0, AluSrc::constant(0, SWZ_XYZW)),
        ],
    );
    for (bools, want) in [([0, 1 << 5], RED), ([1 << 5, 0], GREEN)] {
        let mut mem = memory();
        mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
        let mut s = Stream::default();
        s.setup()
            .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
            .shader(false, &vs_color())
            .shader(true, &ps)
            .vertex_fetch(0, VERTS, 12)
            .regs(
                SHADER_CONSTANT_FLOAT + 256 * 4,
                &[
                    f(1.0),
                    f(0.0),
                    f(0.0),
                    f(1.0),
                    f(0.0),
                    f(1.0),
                    f(0.0),
                    f(1.0),
                ],
            )
            .regs(SHADER_CONSTANT_BOOL, &bools)
            .draw_auto(PRIM_TRI_LIST, 3);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.draws, 1, "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        assert_eq!(px(&img, W, 10, 10), want, "bools {bools:?}");
    }
}

/// MEM_WRITE stores each value byte-swapped by the address's endian field.
#[test]
fn mem_write_applies_endian() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.0.extend(xenon_gpu::pm4::build::type3(
        xenon_gpu::pm4::PM4_MEM_WRITE,
        &[DEST | 2, 0x1122_3344, 0x5566_7788],
    ));
    run(&mut d, &s, &mut mem);
    assert_eq!(
        &mem.data[DEST as usize..][..8],
        &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    );
}

/// A depth resolve with colour clear enabled clears depth only (Xenia
/// IsClearingColor is false for depth copies).
#[test]
fn depth_resolve_does_not_clear_colour() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        // Clear-only resolve from the depth source (copy_src_select 4) with
        // both clears enabled.
        .regs(RB_MODECONTROL, &[6])
        .regs(RB_COPY_CONTROL, &[4 | (1 << 8) | (1 << 9) | (3 << 20)])
        .regs(RB_DEPTH_CLEAR, &[0, rgba(WHITE)])
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_MODECONTROL, &[4]);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 2, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), CLEAR);
}

/// A 32-bit (k8in32) triangle strip with primitive reset: 0xFFFFFFFF
/// splits it into two separate triangles, and nothing is drawn between
/// them.
#[test]
fn strip_primitive_reset_32_bit() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, WHITE),
            (16.0, 0.0, 0.0, WHITE),
            (0.0, 16.0, 0.0, WHITE),
            (48.0, 48.0, 0.0, WHITE),
            (64.0, 48.0, 0.0, WHITE),
            (48.0, 64.0, 0.0, WHITE),
        ]),
    );
    // SliceMemory::write_words stores big-endian words; k8in32 swaps back.
    mem.write_words(INDICES, &[0, 1, 2, u32::MAX, 3, 4, 5]);
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 24)
        .regs(PA_SU_SC_MODE_CNTL, &[1 << 21])
        .regs(VGT_MULTI_PRIM_IB_RESET_INDX, &[u32::MAX]);
    // DRAW_INDX: DMA source, 32-bit indices (initiator bit 11), k8in32.
    s.0.extend(xenon_gpu::pm4::build::type3(
        xenon_gpu::pm4::PM4_DRAW_INDX,
        &[
            0,
            PRIM_TRI_STRIP | (1 << 11) | (7 << 16),
            INDICES,
            7 | (2 << 30),
        ],
    ));
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 4), WHITE);
    assert_eq!(px(&img, W, 52, 52), WHITE);
    for (x, y) in [(20, 22), (30, 30), (40, 44), (20, 10)] {
        assert_eq!(px(&img, W, x, y), CLEAR, "({x}, {y})");
    }
}

/// Depth resolve of `format` (RB_DEPTH_INFO bit 16): clear depth and
/// stencil, draw a triangle at z = 0.25 with depth always passing, resolve
/// the depth surface raw to k_8_8_8_8 and return the words at (2, 2) (in
/// the triangle) and (40, 40) (cleared).
fn depth_resolve_words(depth_info: u32, clear: u32) -> (u32, u32) {
    let Some(mut d) = drawer() else {
        return (0, 0);
    };
    let mut mem = memory();
    let z = 0.25;
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, z, WHITE),
            (16.0, 0.0, z, WHITE),
            (0.0, 16.0, z, WHITE),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .regs(RB_DEPTH_INFO, &[depth_info])
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_NULL | CLEAR_DEPTH,
            (DEST, W | (H << 16), 0),
            [clear, 0, 0],
        )
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        // z test always, z write.
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .resolve_with(
            &mut mem,
            (W, H),
            4 | COPY_RAW,
            (DEST, W | (H << 16), dest_info(6, 0, false)),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 2), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = detiled(&mem, DEST, W, H, W, 4);
    let word = |x: u32, y: u32| u32::from_le_bytes(px(&img, W, x, y));
    (word(2, 2), word(40, 40))
}

/// A D24S8 depth resolve writes the EDRAM word: 24-bit unorm depth in
/// bits 8..31, stencil in 0..7, as Xenia's fast resolve copies it.
#[test]
fn depth_resolve_d24s8_words() {
    if drawer().is_none() {
        return;
    }
    assert_eq!(
        depth_resolve_words(DEPTH_BASE, 0x1234_5678),
        ((0x40_0000 << 8) | 0x78, 0x1234_5678)
    );
}

/// A D24FS8 depth resolve writes 20e4 depth in bits 8..31; the host
/// surface has no stencil, so bits 0..7 are 0.
#[test]
fn depth_resolve_d24fs8_words() {
    if drawer().is_none() {
        return;
    }
    // 0.25 is 20e4 exponent 13, 0.5 exponent 14.
    assert_eq!(
        depth_resolve_words(DEPTH_BASE | (1 << 16), (14 << 28) | 0x78),
        (13 << 28, 14 << 28)
    );
}
