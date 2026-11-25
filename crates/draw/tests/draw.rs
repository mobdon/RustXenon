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
    let m = d.memory_line();
    assert!(m.contains("targets 1 MB in 2,"), "{m}");
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

    // Other clamp, filter and LOD fields sample the same decode.
    let mut sampled = fetch;
    sampled[0] &= !(0x1FF << 10);
    sampled[3] |= 1 << 19 | 1 << 21;
    sampled[4] = 3 << 12;
    let mut s2 = Stream(s.0.clone());
    s2.regs(SHADER_CONSTANT_FETCH + 6, &sampled)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s2, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 0, "{report:?}");

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

/// Draws a 32x32 DXT `format` (k8in16, nearest sampling, tiled or with a
/// pitch of 32 blocks) of pseudo-random blocks of `bpb` bytes over the
/// target's corner with BC passthrough on or off; returns the quad's pixels
/// and the bytes uploaded.
fn draw_dxt(bc: bool, format: u32, bpb: u32, tiled: bool) -> Option<(Vec<[u8; 4]>, u64)> {
    let mut d = drawer()?;
    if bc && !d.bc_textures {
        println!("skip: the adapter has no BC textures");
        return None;
    }
    d.bc_textures = bc;
    let mut mem = memory();
    let n = 32u32;
    let mut seed = 0x1234_5678u32;
    let linear: Vec<u8> = (0..n * n / 16 * bpb)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        })
        .collect();
    let mut guest = if tiled {
        tile::tile_2d(&linear, n / 4, n / 4, n / 4, bpb).unwrap()
    } else {
        let row = (n / 4 * bpb) as usize;
        linear
            .chunks(row)
            .flat_map(|r| [r, &vec![0; row * 3]].concat())
            .collect()
    };
    endian::swap_in_place(&mut guest, xenon_texture::Endian::K8in16);
    mem.data[TEXTURE as usize..][..guest.len()].copy_from_slice(&guest);
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
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (u32::from(tiled) << 31), // clamp
        format | (1 << 6) | TEXTURE,                                             // k8in16
        (n - 1) | ((n - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1, // swizzle RGBA, point filters
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
    assert!(report.warnings.is_empty(), "{report:?}");
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let pixels = (0..n * n).map(|i| px(&img, W, i % n, i / n)).collect();
    Some((pixels, report.timings.texture_bytes))
}

/// DXT1 and DXT5 uploaded as BC blocks sample the texels the CPU decoder
/// makes, in an eighth or a quarter of the bytes. Interpolated texels may
/// be 1 off: the host rounds the 1/3, 2/3 and 1/7.. mixes its own way
/// (lavapipe rounds where the CPU decoder truncates), as the BC spec allows.
#[test]
fn dxt_passthrough_matches_cpu_decode() {
    for (format, bpb, tiled) in [(18, 8, true), (20, 16, true), (18, 8, false)] {
        let Some((bc, bc_bytes)) = draw_dxt(true, format, bpb, tiled) else {
            return;
        };
        let (cpu, cpu_bytes) = draw_dxt(false, format, bpb, tiled).unwrap();
        assert_eq!(cpu_bytes, 32 * 32 * 4);
        assert_eq!(bc_bytes, 64 * u64::from(bpb), "format {format}");
        for (i, (b, c)) in bc.iter().zip(&cpu).enumerate() {
            let near = b.iter().zip(c).all(|(&b, &c)| b.abs_diff(c) <= 1);
            assert!(
                near,
                "format {format} ({}, {}): {b:?} {c:?}",
                i % 32,
                i / 32
            );
        }
    }
}

/// The texture cache evicts the least recently used textures when full,
/// keeping one drawn since, and counts the re-upload of an evicted one.
#[test]
fn texture_cache_evicts_least_recently_used() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let n = 32u32; // 4 KiB per decoded texture
    d.texture_cache_limit = 13 << 10;
    let verts: Vec<u32> = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0), f(0.5), f(0.5)])
        .collect();
    mem.write_words(VERTS, &verts);
    let draw = |d: &mut xenon_draw::Drawer, mem: &mut xenon_gpu::SliceMemory, k: u32| {
        let fetch = [
            2 | ((n >> 5) << 22) | (1 << 31),
            6 | (2 << 6) | (TEXTURE + k * 0x1000),
            (n - 1) | ((n - 1) << 13),
            ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
            0,
            1 << 9,
        ];
        let mut s = Stream::default();
        s.setup()
            .shader(false, &vs_uv())
            .shader(true, &ps_texture(1))
            .vertex_fetch(0, VERTS, 20)
            .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
            .draw_auto(PRIM_QUAD_LIST, 4);
        let t = run(d, &s, mem).timings;
        (
            t.textures_uploaded,
            t.textures_reuploaded,
            t.textures_evicted,
        )
    };
    for k in 0..3 {
        assert_eq!(draw(&mut d, &mut mem, k), (1, 0, 0), "texture {k}");
    }
    assert_eq!(draw(&mut d, &mut mem, 0), (0, 0, 0));
    // Full: evicts 1 and 2, the least recently drawn, and keeps 0.
    assert_eq!(draw(&mut d, &mut mem, 3), (1, 0, 2));
    assert_eq!(draw(&mut d, &mut mem, 0), (0, 0, 0));
    assert_eq!(draw(&mut d, &mut mem, 1), (1, 1, 0));
}

/// A texture slot without a texture fetch constant, or with a format that
/// cannot be decoded, samples zeros (Xenia's null image) and the draw still
/// happens, with one warning per kind.
#[test]
fn unusable_texture_slots_sample_zeros() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let q = 32.0;
    let verts: Vec<u32> = [(0.0, 0.0), (q, 0.0), (q, q), (0.0, q)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0), f(0.5), f(0.5)])
        .collect();
    mem.write_words(VERTS, &verts);
    const CTX1: u32 = 60;
    let fetch = [2 | (1 << 22), CTX1 | TEXTURE, 0, 0, 0, 1 << 9];
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert_eq!(
        report.warnings,
        [
            "texture slot without a texture fetch constant drawn with zeros",
            "unsupported texture format Ctx1; drawn with zeros",
        ],
        "{report:?}"
    );
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 4), [0; 4]);
    assert_eq!(px(&img, W, 40, 40), CLEAR);
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
        // Colour target format 8, which Xenos does not define.
        .regs(RB_COLOR_INFO, &[8 << 16])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_COLOR_INFO, &[0])
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert_eq!(
        report.unsupported,
        vec![
            "primitive type LineLoop".to_string(),
            "colour render target format 8".to_string()
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

/// States that differ only where the pipeline does not look share one:
/// the winding without culling, an alpha blend with alpha masked off, and
/// a stencil test that always passes and keeps next to a depth test that
/// always passes. Five draws, two pipelines, the same pixels.
#[test]
fn pipeline_keys_drop_what_the_pipeline_ignores() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(RB_COLOR_MASK, &[0b0111])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(PA_SU_SC_MODE_CNTL, &[4])
        .draw_auto(PRIM_TRI_LIST, 3)
        // Alpha: src alpha, one minus src alpha.
        .regs(RB_BLENDCONTROL0, &[1 | (6 << 16) | (7 << 24)])
        .draw_auto(PRIM_TRI_LIST, 3)
        // Stencil on, ALWAYS, all ops KEEP.
        .regs(RB_DEPTHCONTROL, &[1 | (7 << 8)])
        .draw_auto(PRIM_TRI_LIST, 3)
        // Depth test ALWAYS without write.
        .regs(RB_DEPTHCONTROL, &[2 | (7 << 4)])
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 5, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert_eq!(report.timings.pipelines_created, 2, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), [255, 255, 255, 255]);
}

/// Past the pipeline cap the least recently used pipeline goes, with the
/// pixel shader only it used; drawing with it again makes it again and
/// draws the same pixels as a drawer that kept it.
#[test]
fn dropped_pipeline_is_made_again_and_draws_the_same() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
    mem.write_words(VERTS + 48, &winding_pair());
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS + 48, 24)
        .draw_auto(PRIM_TRI_LIST, 6)
        .shader(true, &ps_const())
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[0.5, 0.25, 0.0, 1.0].map(f),
        )
        .regs(RB_BLENDCONTROL0, &[1 | (1 << 8)])
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .shader(true, &ps_interp())
        .regs(RB_BLENDCONTROL0, &[0x0001_0001])
        .vertex_fetch(0, VERTS + 48, 24)
        .draw_auto(PRIM_TRI_LIST, 6);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.pipelines_created, 2, "{report:?}");
    let kept = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let m = d.memory_line();
    assert!(
        m.contains("(2 vs/ps pairs, 0 dropped), shaders 3, translated 3 ("),
        "{m}"
    );

    drop(d);
    let Some(mut d) = drawer() else { return };
    d.set_pipeline_cap(1);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert_eq!(report.timings.pipelines_created, 3, "{report:?}");
    let m = d.memory_line();
    assert!(
        m.contains("(1 vs/ps pairs, 2 dropped), shaders 2, translated 2 ("),
        "{m}"
    );
    assert!(!m.contains("(0 KB WGSL)"), "{m}");
    assert!(kept == d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap());
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
fn depth_resolve_words(d: &mut xenon_draw::Drawer, depth_info: u32, clear: u32) -> (u32, u32) {
    depth_resolve_words_at(d, depth_info, clear, 0.25, 6)
}

/// [`depth_resolve_words`] with the triangle at depth `z` and destination
/// format `dest_format` (ignored for depth copies).
fn depth_resolve_words_at(
    d: &mut xenon_draw::Drawer,
    depth_info: u32,
    clear: u32,
    z: f32,
    dest_format: u32,
) -> (u32, u32) {
    let mut mem = memory();
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
            (DEST, W | (H << 16), dest_info(dest_format, 0, false)),
            [0; 3],
        );
    let report = run(d, &s, &mut mem);
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
    let Some(mut d) = drawer() else { return };
    assert_eq!(
        depth_resolve_words(&mut d, DEPTH_BASE, 0x1234_5678),
        ((0x40_0000 << 8) | 0x78, 0x1234_5678)
    );
}

/// A D24FS8 depth resolve writes 20e4 depth in bits 8..31 and the
/// stencil in 0..7 (0 where the host has no float depth with stencil).
#[test]
fn depth_resolve_d24fs8_words() {
    let Some(mut d) = drawer() else { return };
    let stencil = if d.float_stencil() { 0x78 } else { 0 };
    // 0.25 is 20e4 exponent 13, 0.5 exponent 14.
    assert_eq!(
        depth_resolve_words(&mut d, DEPTH_BASE | (1 << 16), (14 << 28) | 0x78),
        ((13 << 28) | stencil, (14 << 28) | stencil)
    );
}

/// Stencil on depth surface `depth_info`: a corner triangle writes
/// reference 5 (always, replace) with colour off, then a full-window
/// triangle drawn only where the stencil equals 5 colours the corner red
/// and leaves the rest at the clear colour.
fn stencil_masks_draw(depth_info: u32) {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, WHITE),
            (16.0, 0.0, 0.0, WHITE),
            (0.0, 16.0, 0.0, WHITE),
            (0.0, 0.0, 0.0, RED),
            (128.0, 0.0, 0.0, RED),
            (0.0, 128.0, 0.0, RED),
        ]),
    );
    let always_replace = 1 | (7 << 8) | (2 << 11) | (2 << 14) | (2 << 17);
    let equal_keep = 1 | (2 << 8);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_DEPTH_INFO, &[depth_info])
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 24)
        // Reference 5, read mask 0xFF, write mask 0xFF.
        .regs(RB_STENCILREFMASK, &[5 | (0xFF << 8) | (0xFF << 16)])
        .regs(RB_DEPTHCONTROL, &[always_replace])
        .regs(RB_COLOR_MASK, &[0])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_COLOR_MASK, &[0xF])
        .regs(RB_DEPTHCONTROL, &[equal_keep])
        .vertex_fetch(0, VERTS + 48, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    if !d.float_stencil() && depth_info & (1 << 16) != 0 {
        assert!(
            report.warnings.iter().any(|w| w.contains("stencil")),
            "{report:?}"
        );
        return;
    }
    assert_eq!(report.draws, 2, "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 2, 2), RED);
    assert_eq!(px(&img, W, 30, 30), CLEAR);
}

#[test]
fn stencil_masks_draw_d24s8() {
    stencil_masks_draw(DEPTH_BASE);
}

#[test]
fn stencil_masks_draw_d24fs8() {
    stencil_masks_draw(DEPTH_BASE | (1 << 16));
}

/// 24-bit depth survives clear and resolve exactly for odd values from
/// 2^23 up (where `x + 0.5` rounding would tie upwards), and a depth copy
/// ignores the destination format (k_16_16 here) as Xenia does.
#[test]
fn depth_resolve_d24s8_exact_and_any_dest_format() {
    let Some(mut d) = drawer() else { return };
    // shortcut: values within a few steps of 1.0 (0xFFFFFD) can come back
    // one step high where the host reads 24-bit depth through a reciprocal
    // multiply (lavapipe does); exact readback needs an integer depth copy.
    for clear in [0x8000_0178, 0xABCD_EF78, 0xFFFF_FF78] {
        assert_eq!(depth_resolve_words(&mut d, DEPTH_BASE, clear).1, clear);
    }
    // 0.75 * 0xFFFFFF = 12582911.25.
    assert_eq!(
        depth_resolve_words_at(&mut d, DEPTH_BASE, 0x1234_5678, 0.75, 25),
        ((12_582_911 << 8) | 0x78, 0x1234_5678)
    );
}

/// 20e4 depth is truncated like Xenia's default dump: f32 0x3E99999E
/// truncates to 0xD33333 (rounding would give 0xD33334).
#[test]
fn depth_resolve_d24fs8_truncates() {
    let Some(mut d) = drawer() else { return };
    let stencil = if d.float_stencil() { 0x78 } else { 0 };
    let z = f32::from_bits(0x3E99_999E);
    assert_eq!(
        depth_resolve_words_at(&mut d, DEPTH_BASE | (1 << 16), 0x78, z, 6).0,
        (0xD3_3333 << 8) | stencil
    );
}

/// A pixel shader with getGradients goes through the drawer's own WGSL
/// validation (the derivatives sit inside the pc-switch dispatch) and
/// draws the colour's screen-space derivatives: red rises 255 over 64
/// pixels along x, so d(red)/dx is 1/64 (about 4 of 255) and d(red)/dy 0.
#[test]
fn get_gradients_pixel_shader_draws() {
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, [0, 0, 0, 255]),
            (64.0, 0.0, 0.0, [255, 0, 0, 255]),
            (0.0, 64.0, 0.0, [0, 0, 0, 255]),
        ]),
    );
    // getGradients r1, r0.xy (swizzle x, y); oC0 = (ddx r, ddy r, ddx g, 1).
    let mut tf = TextureFetch::fetch_2d(1, 0, 0);
    tf.op = TextureOp::GetGradients;
    tf.src_swizzle = 0b01_00;
    tf.dest_swizzle = fetch_dest_swz("xyz1");
    let ps = assemble(
        &[exec(0, 2, true)],
        &[
            Instr::Fetch(FetchInstr::Texture(tf)),
            export(0, AluSrc::temp(1, SWZ_XYZW)),
        ],
    );
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps)
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert!(
        near(px(&img, W, 8, 8), [4, 0, 0, 255], 1),
        "{:?}",
        px(&img, W, 8, 8)
    );
}

/// Pixel shader: oC0 = r1 * c0, with c0 = (1/64, 1/64, 0, 0) set by
/// `param_gen_stream`, so red and green show r1.xy / 64 (0 when negative).
fn ps_param_gen_scaled() -> Vec<u32> {
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let mul = AluInstr::vector(
        VectorOp::Mul,
        0,
        mask("xyzw"),
        &[AluSrc::temp(1, SWZ_XYZW), AluSrc::constant(0, SWZ_XYZW)],
    )
    .exported(0);
    assemble(&[exec(0, 1, true)], &[Instr::Alu(mul)])
}

/// A stream with SQ_PROGRAM_CNTL `program_cntl` (bit 18 is param_gen) and
/// SQ_CONTEXT_MISC param_gen_pos 1, drawing `count` vertices of `prim` from VERTS.
fn param_gen_stream(
    mem: &mut xenon_gpu::SliceMemory,
    mode: u32,
    program_cntl: u32,
    prim: u32,
    count: u32,
) -> Stream {
    let mut s = Stream::default();
    s.setup()
        .resolve(mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_param_gen_scaled())
        .vertex_fetch(0, VERTS, 4 * count)
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[f(1.0 / 64.0), f(1.0 / 64.0), f(0.0), f(0.0)],
        )
        .regs(SQ_PROGRAM_CNTL, &[program_cntl])
        .regs(SQ_CONTEXT_MISC, &[1 << 8])
        .regs(PA_SU_SC_MODE_CNTL, &[mode])
        .draw_auto(prim, count);
    s
}

/// SQ_PROGRAM_CNTL.param_gen: the pixel shader's r1 is the pixel position
/// rounded down, with the sign of x set on back faces (Xenia's PsParamGen).
#[test]
fn pixel_param_gen_position_and_face() {
    let Some(mut d) = drawer() else { return };
    // Unorm of v / 64.
    let u = |v: u32| (v as f32 / 64.0 * 255.0).round() as u8;
    // PA_SU_SC_MODE_CNTL face bit 2: front CCW (top-left `a` front, `b`
    // back), then front CW.
    for (mode, a_front) in [(0, true), (4, false)] {
        let mut mem = memory();
        mem.write_words(VERTS, &winding_pair());
        let s = param_gen_stream(&mut mem, mode, 1 << 18, PRIM_TRI_LIST, 6);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.draws, 1, "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        let want = |front: bool, x, y| [if front { u(x) } else { 0 }, u(y), 0, 0];
        for (x, y, front) in [
            (4, 4, a_front),
            (20, 8, a_front),
            (36, 36, !a_front),
            (40, 50, !a_front),
        ] {
            let p = px(&img, W, x, y);
            assert!(
                near(p, want(front, x, y), 1),
                "mode {mode} ({x}, {y}): {p:?}"
            );
        }
    }
}

/// A point sets the sign of y (and never the facing sign of x). The same
/// shaders first draw without param_gen (r1 is then a zero interpolator),
/// so the cached pipeline must not be reused.
#[test]
fn pixel_param_gen_point_flag() {
    let Some(mut d) = drawer() else { return };
    for (program_cntl, want) in [(0, [0, 0, 0, 0]), (1 << 18, [40, 0, 0, 0])] {
        let mut mem = memory();
        mem.write_words(VERTS, &color_verts(&[(10.5, 20.5, 0.0, RED)]));
        let s = param_gen_stream(&mut mem, 0, program_cntl, 1, 1);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.draws, 1, "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        let p = px(&img, W, 10, 20);
        assert!(near(p, want, 1), "SQ_PROGRAM_CNTL {program_cntl:#x}: {p:?}");
    }
}
