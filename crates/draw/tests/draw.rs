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
        .draw_auto(PRIM_RECT_LIST, 3)
        // k_16_16_16_16_FLOAT colour target.
        .regs(RB_COLOR_INFO, &[7 << 16])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_COLOR_INFO, &[0])
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert_eq!(
        report.unsupported,
        vec![
            "primitive type RectangleList".to_string(),
            "colour render target format 7".to_string()
        ]
    );
    let missing = RenderTargetKey {
        base_tiles: 1,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    assert!(d.read_render_target(missing, 0, 0, 1, 1).is_err());
}
