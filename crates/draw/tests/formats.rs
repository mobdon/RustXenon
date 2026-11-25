//! Render target formats beyond 8_8_8_8, their clears, blending and
//! resolves, on synthetic streams. They need a wgpu adapter (lavapipe
//! works); without one they print a skip line, unless XENON_REQUIRE_GPU=1
//! makes them fail.

mod common;

use common::*;
use xenon_draw::{RenderTargetKey, TargetFormat};
use xenon_gpu::regs::*;
use xenon_shader::asm::*;
use xenon_shader::ucode::*;

const PRIM_TRI_LIST: u32 = 4;
const PRIM_RECT_LIST: u32 = 8;

/// `xenos::ColorRenderTargetFormat` values.
const RT_2_10_10_10: u32 = 2;
const RT_16_16: u32 = 4;
const RT_16_16_16_16: u32 = 5;
const RT_16_16_FLOAT: u32 = 6;
const RT_16_16_16_16_FLOAT: u32 = 7;
const RT_2_10_10_10_FLOAT_AS_16_16_16_16: u32 = 12;
const RT_32_FLOAT: u32 = 14;
const RT_32_32_FLOAT: u32 = 15;

/// `xenos::ColorFormat` destination values.
const DEST_8_8_8_8: u32 = 6;
const DEST_2_10_10_10: u32 = 7;
const DEST_16_16_16_16_FLOAT: u32 = 32;

fn key(format: TargetFormat) -> RenderTargetKey {
    RenderTargetKey {
        base_tiles: COLOR_BASE,
        pitch: W,
        format,
    }
}

/// Texel (x, y) of `bytes`-byte texels in rows of `w`.
fn texel(img: &[u8], w: u32, bytes: u32, x: u32, y: u32) -> Vec<u8> {
    let i = ((y * w + x) * bytes) as usize;
    img[i..i + bytes as usize].to_vec()
}

/// The four halves of a 16-bit float texel.
fn halves(t: &[u8]) -> [u16; 4] {
    [0, 1, 2, 3].map(|i| u16::from_le_bytes([t[2 * i], t[2 * i + 1]]))
}

/// A triangle in window pixels covering the corner (0,0)-(n,n) of the
/// target and more (vs_color layout, colour unused by ps_const).
fn corner_triangle(n: f32) -> Vec<u32> {
    let c = [0, 0, 0, 255];
    color_verts(&[(0.0, 0.0, 0.0, c), (n, 0.0, 0.0, c), (0.0, n, 0.0, c)])
}

/// Pixel constant c0.
fn c0(s: &mut Stream, v: [f32; 4]) -> &mut Stream {
    s.regs(SHADER_CONSTANT_FLOAT + 256 * 4, &v.map(f))
}

/// Draws into a k_2_10_10_10_FLOAT_AS_16_16_16_16 target and resolves it
/// with Convert to tiled 8_8_8_8: out-of-range outputs are clamped when the
/// target is written (to [0, 31.875], alpha to [0, 1]) and saturated by the
/// resolve, 0.5 packs to 128 (x * 255 + 0.5, truncated), the 2-bit alpha of
/// the EDRAM format turns 0.5 into 2/3 (170), and the exponent bias and the
/// red/blue swap apply.
#[test]
fn float7e3_target_draw_and_resolve() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &[corner_triangle(128.0), corner_triangle(16.0)].concat(),
    );
    let dest2 = DEST + 0x1_0000;
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_2_10_10_10_FLOAT_AS_16_16_16_16 << 16])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 24);
    c0(&mut s, [40.0, -1.0, 0.5, 1.0]).draw_auto(PRIM_TRI_LIST, 3);
    s.vertex_fetch(0, VERTS + 48, 12);
    c0(&mut s, [0.75, 0.25, 0.125, 0.5]).draw_auto(PRIM_TRI_LIST, 3);
    let pitch_height = W | (H << 16);
    s.resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (DEST, pitch_height, dest_info(DEST_8_8_8_8, 0, false)),
        [0; 3],
    )
    .resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (dest2, pitch_height, dest_info(DEST_8_8_8_8, -1, true)),
        [0; 3],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 2), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");

    let raw = d
        .read_render_target_raw(key(TargetFormat::Float7e3), 0, 0, W, H)
        .unwrap();
    // 31.875, 0, 0.5, 1 and 0.75, 0.25, 0.125, 0.5 as halves.
    assert_eq!(
        halves(&texel(&raw, W, 8, 40, 40)),
        [0x4FF8, 0, 0x3800, 0x3C00]
    );
    assert_eq!(
        halves(&texel(&raw, W, 8, 2, 2)),
        [0x3A00, 0x3400, 0x3000, 0x3800]
    );
    assert!(d
        .read_render_target(key(TargetFormat::Float7e3), 0, 0, 1, 1)
        .is_err());

    let img = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&img, W, 40, 40), [255, 0, 128, 255]);
    assert_eq!(px(&img, W, 63, 63), [255, 0, 128, 255]);
    assert_eq!(px(&img, W, 2, 2), [191, 64, 32, 170]);
    // Bias -1 halves (31.875 still saturates), then red and blue swap.
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!(px(&img, W, 40, 40), [64, 0, 255, 128]);
    assert_eq!(px(&img, W, 2, 2), [16, 32, 96, 85]);
}

/// A resolve clear of a 7e3 target decodes RB_COLOR_CLEAR as three 7e3
/// values and a 2-bit alpha; a resolve then converts them.
#[test]
fn float7e3_target_clear_and_resolve() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    // 2.0, 0.375, 0.125 (a 7e3 denormal), alpha 1 of 3.
    let clear = 0x200 | (0x0C0 << 10) | (0x040 << 20) | (1 << 30);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_2_10_10_10_FLOAT_AS_16_16_16_16 << 16])
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_NULL | CLEAR_COLOR,
            (DEST, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
            [0, clear, 0xFFFF_FFFF],
        )
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (DEST, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 2, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Float7e3), 0, 0, W, H)
        .unwrap();
    // 1/3 as a half is 0x3555.
    assert!(raw
        .chunks_exact(8)
        .all(|t| halves(t) == [0x4000, 0x3600, 0x3000, 0x3555]));
    let img = detiled(&mem, DEST, W, H, W, 4);
    assert!(img.chunks_exact(4).all(|p| p == [255, 96, 32, 85]));
}

/// Blending on a 7e3 target works in float: an additive blend goes above
/// 1.0 and a reverse subtract brings it back, which a unorm target could
/// not do.
#[test]
fn float7e3_target_blending() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(128.0));
    // 0.75, 0.25, 0, alpha 3 of 3.
    let clear = 0x140 | (0x080 << 10) | (3 << 30);
    // Colour and alpha: One, add / reverse subtract (4), One.
    let add = 1 | (1 << 8) | (1 << 16) | (1 << 24);
    let rev_sub = add | (4 << 5) | (4 << 21);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_2_10_10_10_FLOAT_AS_16_16_16_16 << 16])
        .resolve(&mut mem, W, H, false, Some((clear, 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12)
        .regs(RB_BLENDCONTROL0, &[add]);
    // 1.5, 0.5, 0.5, 1 after the add; 0.5, 0.25, 0.25, 1 after the
    // reverse subtract.
    c0(&mut s, [0.75, 0.25, 0.5, 0.0])
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(RB_BLENDCONTROL0, &[rev_sub]);
    c0(&mut s, [1.0, 0.25, 0.25, 0.0])
        .draw_auto(PRIM_TRI_LIST, 3)
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (DEST, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Float7e3), 0, 0, W, H)
        .unwrap();
    assert_eq!(
        halves(&texel(&raw, W, 8, 30, 30)),
        [0x3800, 0x3400, 0x3400, 0x3C00]
    );
    let img = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&img, W, 30, 30), [128, 64, 64, 255]);
}

/// k_2_10_10_10: a clear writes the bits as they are, a draw writes unorm,
/// a resolve to k_2_10_10_10 copies the bits and one to k_8_8_8_8 converts.
#[test]
fn rgb10a2_target_round_trip() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(16.0));
    let clear: u32 = 683 | (341 << 10) | (1023 << 20) | (1 << 30);
    let dest2 = DEST + 0x1_0000;
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_2_10_10_10 << 16])
        .resolve(&mut mem, W, H, false, Some((clear, 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12);
    c0(&mut s, [1.0, 0.0, 0.0, 1.0])
        .draw_auto(PRIM_TRI_LIST, 3)
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_RAW,
            (DEST, W | (H << 16), dest_info(DEST_2_10_10_10, 0, false)),
            [0; 3],
        )
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (dest2, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgb10a2), 0, 0, W, H)
        .unwrap();
    let red: u32 = 1023 | (3 << 30);
    let word = |t: Vec<u8>| u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
    assert_eq!(word(texel(&raw, W, 4, 2, 2)), red);
    assert_eq!(word(texel(&raw, W, 4, 40, 40)), clear);
    assert_eq!(detiled(&mem, DEST, W, H, W, 4), raw);
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!(px(&img, W, 2, 2), [255, 0, 0, 255]);
    // 683/1023 * 255 = 170.25, 341/1023 = 1/3 -> 85.
    assert_eq!(px(&img, W, 40, 40), [170, 85, 255, 85]);
}

/// k_16_16_16_16_FLOAT: the 64-bit clear value is RB_COLOR_CLEAR (red and
/// green halves) | RB_COLOR_CLEAR_LO << 32 (blue and alpha); a resolve to
/// k_16_16_16_16_FLOAT copies the halves, one to k_8_8_8_8 converts.
#[test]
fn rgba16f_target_clear_and_resolves() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let dest2 = DEST + 0x2_0000;
    let dest3 = DEST + 0x4_0000;
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_16_16_16_16_FLOAT << 16])
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_NULL | CLEAR_COLOR,
            (DEST, W | (H << 16), 0),
            // 1.0, 0.5 | -1.0, 2.0
            [0, 0x3800_3C00, 0x4000_BC00],
        )
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (
                DEST,
                W | (H << 16),
                dest_info(DEST_16_16_16_16_FLOAT, 0, false),
            ),
            [0; 3],
        )
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (dest2, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
            [0; 3],
        )
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (
                dest3,
                W | (H << 16),
                // Endian 4, 8in64.
                (dest_info(DEST_16_16_16_16_FLOAT, 0, false) & !7) | 4,
            ),
            [0; 3],
        );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 4, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgba16Float), 0, 0, W, H)
        .unwrap();
    assert!(raw
        .chunks_exact(8)
        .all(|t| halves(t) == [0x3C00, 0x3800, 0xBC00, 0x4000]));
    assert_eq!(detiled(&mem, DEST, W, H, W, 8), raw);
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert!(img.chunks_exact(4).all(|p| p == [255, 128, 0, 255]));
    // 8in64 reverses all 8 bytes; `detiled` undoes the 8in32 part, which
    // leaves the two 32-bit words swapped.
    let swapped: Vec<u8> = raw
        .chunks_exact(8)
        .flat_map(|t| [&t[4..], &t[..4]].concat())
        .collect();
    assert_eq!(detiled(&mem, dest3, W, H, W, 8), swapped);
}

/// A frame shaped like the one a game draws: a draw before the first
/// render target setup (surface pitch 0, skipped with a warning), a
/// rectangle-list clear quad into a 1280x720 k_2_10_10_10_FLOAT_AS_16_16_16_16
/// target at EDRAM base 0, a triangle, a Convert resolve to k_8_8_8_8 with
/// pitch 1280 and height 720, and a swap of that address.
/// The real sequence: 7e3 target, rect-list clear, triangle, Convert resolve
/// with `endian`, swap. The frame must read back in the resolve's byte order.
fn synthetic_frame(endian: u32, every_pixel: bool) {
    let Some(mut d) = drawer() else { return };
    let (fw, fh) = (1280u32, 720u32);
    let front = 0x20_0000;
    let mut mem = xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; 0x60_0000],
    };
    let c = [0, 0, 0, 255];
    let quad = color_verts(&[
        (0.0, 0.0, 0.0, c),
        (fw as f32, 0.0, 0.0, c),
        (0.0, fh as f32, 0.0, c),
    ]);
    let tri = color_verts(&[
        (100.0, 100.0, 0.0, c),
        (300.0, 100.0, 0.0, c),
        (100.0, 300.0, 0.0, c),
    ]);
    mem.write_words(VERTS, &[quad, tri].concat());
    let mut s = Stream::default();
    s.setup()
        .regs(
            RB_SURFACE_INFO,
            &[0, RT_2_10_10_10_FLOAT_AS_16_16_16_16 << 16, DEPTH_BASE],
        )
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, fw | (fh << 16)])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12);
    c0(&mut s, [0.25, 0.5, 0.75, 1.0])
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_SURFACE_INFO, &[fw])
        .draw_auto(PRIM_RECT_LIST, 3)
        .vertex_fetch(0, VERTS + 48, 12);
    c0(&mut s, [1.5, 0.0, 0.125, 1.0])
        .draw_auto(PRIM_TRI_LIST, 3)
        .resolve_with(
            &mut mem,
            (fw, fh),
            COPY_CONVERT,
            (
                front,
                fw | (fh << 16),
                (dest_info(DEST_8_8_8_8, 0, false) & !3) | endian,
            ),
            [0; 3],
        )
        .swap_at(front, fw, fh);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("surface pitch 0")),
        "{report:?}"
    );
    assert_eq!(report.frames.len(), 1);
    let frame = &report.frames[0];
    assert_eq!((frame.width, frame.height), (fw, fh));
    let back = [64, 128, 191, 255];
    let red = [255, 0, 32, 255];
    for (x, y) in [
        (0, 0),
        (fw - 1, 0),
        (0, fh - 1),
        (fw - 1, fh - 1),
        (300, 300),
        (99, 150),
    ] {
        assert_eq!(px(&frame.rgba8, fw, x, y), back, "({x}, {y})");
    }
    for (x, y) in [(100, 100), (150, 150), (298, 100), (100, 298)] {
        assert_eq!(px(&frame.rgba8, fw, x, y), red, "({x}, {y})");
    }
    if !every_pixel {
        return;
    }
    // Every pixel: the clear quad leaves no gap (its diagonal included),
    // the triangle covers its pixel centres (centres on its hypotenuse,
    // x + y = 399, are left out).
    for y in 0..fh {
        for x in 0..fw {
            if x + y == 399 {
                continue;
            }
            let inside = x >= 100 && y >= 100 && x + y < 399;
            let want = if inside { red } else { back };
            assert_eq!(px(&frame.rgba8, fw, x, y), want, "({x}, {y})");
        }
    }
}

#[test]
fn synthetic_frame_float7e3_to_front_buffer() {
    synthetic_frame(2, true);
}

#[test]
fn front_buffer_follows_resolve_endian() {
    // A game may resolve its front buffer without a swap (endian 0) or with
    // any other mode; each must give the same frame, not reversed bytes.
    for endian in [0, 1, 3] {
        synthetic_frame(endian, false);
    }
}

/// The GPU resolve of 32bpp destinations matches the CPU reference
/// conversion ([`xenon_draw::formats::edram_rgba`] and
/// [`xenon_draw::formats::resolve_pixel`], then
/// [`xenon_draw::resolve::write_tiled_texels`]) byte for byte, over a
/// gradient of every source format, every 32bpp destination format, exponent
/// biases, the red/blue swap and every endian.
#[test]
fn gpu_resolve_matches_cpu_reference() {
    use xenon_draw::{formats, resolve};
    let Some(mut d) = drawer() else { return };
    let (w2, h2) = ((2 * W) as f32, (2 * H) as f32);
    let verts = color_verts(&[
        (0.0, 0.0, 0.0, [0, 0, 0, 0]),
        (w2, 0.0, 0.0, [255, 128, 3, 255]),
        (0.0, h2, 0.0, [7, 255, 255, 64]),
    ]);
    let cases = [
        (DEST_8_8_8_8, 0, false, 0),
        (DEST_8_8_8_8, 3, true, 1),
        (DEST_8_8_8_8, -2, false, 2),
        (DEST_2_10_10_10, 1, false, 3),
        (DEST_2_10_10_10, -5, true, 2),
        (formats::DEST_16_16, 0, true, 1),
    ];
    let snorm16 = d.snorm16();
    for (rt, format) in [
        (0, TargetFormat::Rgba8),
        (RT_2_10_10_10, TargetFormat::Rgb10a2),
        (RT_2_10_10_10_FLOAT_AS_16_16_16_16, TargetFormat::Float7e3),
        (RT_16_16_16_16_FLOAT, TargetFormat::Rgba16Float),
        (RT_16_16, TargetFormat::Rg16Fixed),
        (RT_16_16_16_16, TargetFormat::Rgba16Fixed),
        (RT_16_16_FLOAT, TargetFormat::Rg16Float),
        (RT_32_32_FLOAT, TargetFormat::Rg32Float),
    ]
    .into_iter()
    .filter(|(rt, _)| snorm16 || !matches!(*rt, RT_16_16 | RT_16_16_16_16))
    {
        let mut mem = memory();
        mem.write_words(VERTS, &verts);
        let mut s = Stream::default();
        s.setup()
            .regs(RB_COLOR_INFO, &[rt << 16])
            .shader(false, &vs_color())
            .shader(true, &ps_interp())
            .vertex_fetch(0, VERTS, 12)
            .draw_auto(PRIM_TRI_LIST, 3);
        let dest = |i: usize| DEST + 0x1_0000 * i as u32;
        for (i, &(dest_format, bias, swap, endian)) in cases.iter().enumerate() {
            s.resolve_with(
                &mut mem,
                (W, H),
                COPY_CONVERT,
                (
                    dest(i),
                    W | (H << 16),
                    (dest_info(dest_format, bias, swap) & !3) | endian,
                ),
                [0; 3],
            );
        }
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.resolves, cases.len() as u32, "{report:?}");
        assert!(report.unsupported.is_empty(), "{report:?}");
        let raw = d.read_render_target_raw(key(format), 0, 0, W, H).unwrap();
        let src_bytes = raw.len() / (W * H) as usize;
        let mut expected = memory();
        for (i, &(dest_format, bias, swap, endian)) in cases.iter().enumerate() {
            let mut out = vec![0; (W * H * 4) as usize];
            for (src, dst) in raw.chunks_exact(src_bytes).zip(out.chunks_exact_mut(4)) {
                let rgba = formats::edram_rgba(format, src);
                formats::resolve_pixel(rgba, bias, swap, dest_format, dst);
            }
            let rect = resolve::Rect {
                x0: 0,
                y0: 0,
                x1: W,
                y1: H,
            };
            resolve::write_tiled_texels(&mut expected, dest(i), W, rect, &out, 2, endian).unwrap();
            let range = dest(i) as usize..(dest(i) + W * H * 4) as usize;
            let distinct: std::collections::HashSet<_> =
                mem.data[range.clone()].chunks(4).collect();
            assert!(
                distinct.len() > 200,
                "{format:?} case {i}: {}",
                distinct.len()
            );
            assert!(
                mem.data[range.clone()] == expected.data[range],
                "{format:?} case {i}"
            );
        }
    }
}

/// Draws a quad over the whole k_16_16_16_16_FLOAT target sampling a 1x1
/// linear texture of `format` (endian none, swizzle RGBA, point filter)
/// whose guest bytes are `texel_bytes`, and returns the halves at (8, 8).
fn sample_into_rgba16f(d: &mut xenon_draw::Drawer, format: u32, texel_bytes: &[u8]) -> [u16; 4] {
    let t = sample_into(d, RT_16_16_16_16_FLOAT, format, texel_bytes);
    halves(&t)
}

/// As `sample_into_rgba16f`, into a target of `rt_format` (16_16_16_16_FLOAT
/// or 32_FLOAT), returning the raw texel at (8, 8).
fn sample_into(
    d: &mut xenon_draw::Drawer,
    rt_format: u32,
    format: u32,
    texel_bytes: &[u8],
) -> Vec<u8> {
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let fetch = [
        2 | (2 << 10) | (2 << 13) | (1 << 22), // clamp, pitch 32, linear
        format | TEXTURE,
        0, // 1x1
        swizzle_rgba << 1,
        0,
        1 << 9, // 2D
    ];
    sample_fetch(d, rt_format, fetch, texel_bytes)
}

/// Draws a quad sampling `fetch` (whose base address is TEXTURE) at uv
/// (0.5, 0.5) into a target of `rt_format`, returning the raw texel at
/// (8, 8).
fn sample_fetch(
    d: &mut xenon_draw::Drawer,
    rt_format: u32,
    fetch: [u32; 6],
    texel_bytes: &[u8],
) -> Vec<u8> {
    sample_fetch_report(d, rt_format, fetch, texel_bytes).0
}

/// As `sample_fetch`, also returning the draw's report.
fn sample_fetch_report(
    d: &mut xenon_draw::Drawer,
    rt_format: u32,
    fetch: [u32; 6],
    texel_bytes: &[u8],
) -> (Vec<u8>, xenon_draw::Report) {
    let mut mem = memory();
    mem.data[TEXTURE as usize..][..texel_bytes.len()].copy_from_slice(texel_bytes);
    let q = W as f32;
    let verts: Vec<u32> = [(0.0, 0.0), (q, 0.0), (q, q), (0.0, q)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0), f(0.5), f(0.5)])
        .collect();
    mem.write_words(VERTS, &verts);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[rt_format << 16])
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 20)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let (target, bytes) = if rt_format == RT_32_FLOAT {
        (TargetFormat::Float32, 4)
    } else {
        (TargetFormat::Rgba16Float, 8)
    };
    let raw = d.read_render_target_raw(key(target), 0, 0, W, H).unwrap();
    (texel(&raw, W, bytes, 8, 8), report)
}

const PRIM_QUAD_LIST: u32 = 13;

/// Depth textures sample their depth (high 24 bits) in every channel at
/// float precision: 20e4 0.75 and unorm 0x800000 (0.5 + 2^-24, a half's
/// 0.5) would read 191/255 and 128/255 through 8-bit unorm.
#[test]
fn depth_textures_sample_depth() {
    let Some(mut d) = drawer() else { return };
    const K24_8: u32 = 22;
    const K24_8_FLOAT: u32 = 23;
    let word = |depth: u32| ((depth << 8) | 0x5A).to_le_bytes();
    assert_eq!(
        sample_into_rgba16f(&mut d, K24_8_FLOAT, &word((14 << 20) | (1 << 19))),
        [0x3A00; 4]
    );
    assert_eq!(
        sample_into_rgba16f(&mut d, K24_8, &word(0x80_0000)),
        [0x3800; 4]
    );
    // A k_32_FLOAT target keeps the 2^-24 a half drops, when depth
    // textures upload as 32-bit floats.
    if d.float32_textures() {
        let red = sample_into(&mut d, RT_32_FLOAT, K24_8, &word(0x80_0000));
        assert_eq!(red, (0.5 + 1.0 / (1 << 24) as f32).to_le_bytes());
        let red = sample_into(&mut d, RT_32_FLOAT, K24_8_FLOAT, &word(1));
        assert_eq!(red, f32::powi(2.0, -34).to_le_bytes());
    } else {
        println!("skip: no filterable Rgba32Float, depth textures are halves");
    }
}

/// Half and single float textures keep values outside [0, 1].
#[test]
fn hdr_textures_sample_unclamped() {
    let Some(mut d) = drawer() else { return };
    const K16_16_16_16_FLOAT: u32 = 32;
    const K32_32_32_32_FLOAT: u32 = 38;
    // 4.0, -2.0, 0.5, 1.0 as halves.
    let h: Vec<u8> = [0x4400u16, 0xC000, 0x3800, 0x3C00]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        sample_into_rgba16f(&mut d, K16_16_16_16_FLOAT, &h),
        [0x4400, 0xC000, 0x3800, 0x3C00]
    );
    let s: Vec<u8> = [3.0f32, 0.25, 100.0, -1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        sample_into_rgba16f(&mut d, K32_32_32_32_FLOAT, &s),
        [0x4200, 0x3400, 0x5640, 0xBC00]
    );
}

/// One draw into three render targets of different formats: oC<i> goes
/// to RB_COLOR_INFO<i>, and a target whose RB_COLOR_MASK bits are all off
/// is not drawn (nor created).
#[test]
fn multiple_render_targets() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(128.0));
    let ps = assemble(
        &[exec(0, 4, true)],
        &(0..4)
            .map(|i| export(i, AluSrc::constant(i, SWZ_XYZW)))
            .collect::<Vec<_>>(),
    );
    let (b1, b2, b3) = (100, 200, 300);
    let mut s = Stream::default();
    s.setup()
        .regs(
            RB_COLOR_INFO_RT[1],
            &[b1 | (RT_2_10_10_10_FLOAT_AS_16_16_16_16 << 16)],
        )
        .regs(RB_COLOR_INFO_RT[2], &[b2 | (RT_32_FLOAT << 16)])
        .regs(RB_COLOR_INFO_RT[3], &[b3])
        // Blending off (one, zero, add) on every target.
        .regs(RB_BLENDCONTROL1, &[0x0001_0001; 3])
        // RT0 red and alpha, RT1 and RT2 all channels, RT3 none.
        .regs(RB_COLOR_MASK, &[0x9 | (0xF << 4) | (0xF << 8)])
        .shader(false, &vs_color())
        .shader(true, &ps)
        .vertex_fetch(0, VERTS, 12)
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[
                [0.5, 0.25, 0.75, 1.0],
                [2.5, 0.5, 40.0, 0.5],
                [-3.25, 9.0, 9.0, 9.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
            .concat()
            .into_iter()
            .map(f)
            .collect::<Vec<_>>(),
        )
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let at = |base, format| RenderTargetKey {
        base_tiles: base,
        pitch: W,
        format,
    };
    let rt0 = d
        .read_render_target(key(TargetFormat::Rgba8), 0, 0, W, H)
        .unwrap();
    // Green and blue masked off: they keep the zero the target starts with.
    assert_eq!(px(&rt0, W, 4, 4), [128, 0, 0, 255]);
    let rt1 = d
        .read_render_target_raw(at(b1, TargetFormat::Float7e3), 0, 0, W, H)
        .unwrap();
    // 7e3 output clamp (blending off): 40 to 31.875.
    assert_eq!(
        halves(&texel(&rt1, W, 8, 4, 4)),
        [0x4100, 0x3800, 0x4FF8, 0x3800]
    );
    let rt2 = d
        .read_render_target_raw(at(b2, TargetFormat::Float32), 0, 0, W, H)
        .unwrap();
    assert_eq!(texel(&rt2, W, 4, 4, 4), (-3.25f32).to_le_bytes());
    assert!(!d.render_targets().iter().any(|k| k.base_tiles == b3));
}

/// MSAA surfaces are drawn and resolved with one sample per pixel instead
/// of skipping the draw.
#[test]
fn msaa_surface_draws_and_resolves_single_sampled() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(128.0));
    let mut s = Stream::default();
    s.setup()
        // 2x MSAA.
        .regs(RB_SURFACE_INFO, &[W | (1 << 16)])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12);
    c0(&mut s, [0.0, 1.0, 0.0, 1.0]).draw_auto(PRIM_TRI_LIST, 3);
    s.resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (DEST, W | (H << 16), dest_info(DEST_8_8_8_8, 0, false)),
        [0; 3],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 1), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert!(
        report.warnings.iter().any(|w| w.contains("MSAA")),
        "{report:?}"
    );
    let img = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!(px(&img, W, 10, 10), [0, 255, 0, 255]);
}

/// A k_32_FLOAT render target resolved to k_32_FLOAT keeps the red
/// channel's float bits, scaled by the exponent bias and not saturated,
/// on the GPU and the CPU reference alike; a clear writes the clear word's
/// float.
#[test]
fn float32_target_clear_draw_and_resolve() {
    use xenon_draw::formats;
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(16.0));
    let dest2 = DEST + 0x1_0000;
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_32_FLOAT << 16])
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_NULL | CLEAR_COLOR,
            (DEST, W | (H << 16), 0),
            [0, 0.125f32.to_bits(), 0],
        )
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12);
    c0(&mut s, [6.5, 2.0, 3.0, 4.0]).draw_auto(PRIM_TRI_LIST, 3);
    s.resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (
            DEST,
            W | (H << 16),
            dest_info(formats::DEST_32_FLOAT, 0, false),
        ),
        [0; 3],
    )
    .resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT,
        (
            dest2,
            W | (H << 16),
            dest_info(formats::DEST_32_FLOAT, -1, false),
        ),
        [0; 3],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let word =
        |img: &[u8], x: u32, y: u32| f32::from_le_bytes(texel(img, W, 4, x, y).try_into().unwrap());
    let img = detiled(&mem, DEST, W, H, W, 4);
    assert_eq!((word(&img, 2, 2), word(&img, 40, 40)), (6.5, 0.125));
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!((word(&img, 2, 2), word(&img, 40, 40)), (3.25, 0.0625));
    let mut out = [0; 4];
    formats::resolve_pixel(
        formats::edram_rgba(TargetFormat::Float32, &6.5f32.to_le_bytes()),
        -1,
        false,
        formats::DEST_32_FLOAT,
        &mut out,
    );
    assert_eq!(out, 3.25f32.to_le_bytes());
}

/// The 32-bit words of a `w`-wide image of `bytes`-byte texels at (x, y).
fn words(img: &[u8], bytes: u32, x: u32, y: u32) -> Vec<u32> {
    texel(img, W, bytes, x, y)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The 16-bit fields of a texel, low first.
fn shorts(t: &[u8]) -> Vec<u16> {
    t.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// A colour clear of the whole target, then `c0` drawn into the corner
/// (0,0)-(16,16), then one Convert resolve per (dest address, format,
/// exponent bias). `blend` turns additive blending on.
fn clear_draw_resolve(
    d: &mut xenon_draw::Drawer,
    rt: u32,
    clear: (u32, u32),
    c: [f32; 4],
    blend: bool,
    resolves: &[(u32, u32, i32)],
) -> (xenon_gpu::SliceMemory, xenon_draw::Report) {
    let mut mem = memory();
    mem.write_words(VERTS, &corner_triangle(16.0));
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[rt << 16])
        .resolve_with(
            &mut mem,
            (W, H),
            COPY_NULL | CLEAR_COLOR,
            (DEST, W | (H << 16), 0),
            [0, clear.0, clear.1],
        )
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 12);
    if blend {
        // ONE + ONE for colour and alpha.
        s.regs(RB_BLENDCONTROL0, &[0x0101_0101]);
    }
    c0(&mut s, c).draw_auto(PRIM_TRI_LIST, 3);
    for &(dest, format, bias) in resolves {
        s.resolve_with(
            &mut mem,
            (W, H),
            COPY_CONVERT,
            (dest, W | (H << 16), dest_info(format, bias, false)),
            [0; 3],
        );
    }
    let report = run(d, &s, &mut mem);
    (mem, report)
}

/// k_16_16 holds [-32, 32] as 16-bit fixed point; the snorm16 host target
/// holds the shader output / 32. A clear writes the raw 16-bit values
/// (0x7FFF is 32, 0xC000 is -16.0005), a resolve unpacks x * 32 / 32767
/// floored at -1 (Xenia XeUnpackR16G16Edram) and packs it with the
/// exponent bias: to 8_8_8_8 and 16_16 (unorm) and 32_FLOAT on the GPU,
/// to 16_16_FLOAT on the CPU.
#[test]
fn fixed16_target_clear_draw_and_resolve() {
    use xenon_draw::formats::*;
    let Some(mut d) = drawer() else { return };
    if !d.snorm16() {
        eprintln!("SKIP: the host cannot render into 16-bit snorm");
        return;
    }
    let dest = |i: u32| DEST + 0x1_0000 * i;
    let (mem, report) = clear_draw_resolve(
        &mut d,
        RT_16_16,
        (0xC000_7FFF, 0),
        [3.0, -0.5, 7.0, 8.0],
        false,
        &[
            (dest(0), DEST_8_8_8_8, -5),
            (dest(1), DEST_16_16, -5),
            (dest(2), DEST_32_FLOAT, 0),
            (dest(3), DEST_16_16_FLOAT, 0),
        ],
    );
    assert_eq!((report.draws, report.resolves), (1, 5), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rg16Fixed), 0, 0, W, H)
        .unwrap();
    // 3 / 32 * 32767 = 3071.9 and -0.5 / 32 * 32767 = -511.98, rounded.
    assert_eq!(shorts(&texel(&raw, W, 4, 2, 2)), [0x0C00, 0xFE00]);
    assert_eq!(shorts(&texel(&raw, W, 4, 40, 40)), [0x7FFF, 0xC000]);
    // 3072 * 32 / 32767 = 3.0001, / 32 * 255 + 0.5 = 24.4; -16 floors at -1.
    let img = detiled(&mem, dest(0), W, H, W, 4);
    assert_eq!(px(&img, W, 2, 2), [24, 0, 0, 0]);
    assert_eq!(px(&img, W, 40, 40), [255, 0, 0, 0]);
    let img = detiled(&mem, dest(1), W, H, W, 4);
    assert_eq!(shorts(&texel(&img, W, 4, 2, 2)), [6144, 0]);
    assert_eq!(shorts(&texel(&img, W, 4, 40, 40)), [65535, 0]);
    let img = detiled(&mem, dest(2), W, H, W, 4);
    let r = 3072.0f32 * (32.0 / 32767.0);
    assert_eq!(words(&img, 4, 2, 2), [r.to_bits()]);
    assert_eq!(words(&img, 4, 40, 40), [32f32.to_bits()]);
    // 3.0001 and -0.50002 round to the halves 3 and -0.5.
    let img = detiled(&mem, dest(3), W, H, W, 4);
    assert_eq!(shorts(&texel(&img, W, 4, 2, 2)), [0x4200, 0xB800]);
    assert_eq!(shorts(&texel(&img, W, 4, 40, 40)), [0x5000, 0xBC00]);
}

/// k_16_16_16_16: as k_16_16 with four channels; -40 clamps to -32
/// (0x8001), the 64-bit clear value fills blue and alpha from
/// RB_COLOR_CLEAR_LO, and a resolve to 16_16_16_16 (unorm, CPU) and
/// 8_8_8_8 (GPU) converts all four.
#[test]
fn fixed16x4_target_clear_draw_and_resolve() {
    use xenon_draw::formats::*;
    let Some(mut d) = drawer() else { return };
    if !d.snorm16() {
        eprintln!("SKIP: the host cannot render into 16-bit snorm");
        return;
    }
    let dest = |i: u32| DEST + 0x2_0000 * i;
    let (mem, report) = clear_draw_resolve(
        &mut d,
        RT_16_16_16_16,
        (0x0000_7FFF, 0x7FFF_0400),
        [3.0, -0.5, 1.0, -40.0],
        false,
        &[(dest(0), DEST_16_16_16_16, -5), (dest(1), DEST_8_8_8_8, -5)],
    );
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgba16Fixed), 0, 0, W, H)
        .unwrap();
    assert_eq!(
        shorts(&texel(&raw, W, 8, 2, 2)),
        [0x0C00, 0xFE00, 0x0400, 0x8001]
    );
    assert_eq!(
        shorts(&texel(&raw, W, 8, 40, 40)),
        [0x7FFF, 0, 0x0400, 0x7FFF]
    );
    // 1024 * 32 / 32767 / 32 * 65535 + 0.5 = 2048.5, truncated.
    let img = detiled(&mem, dest(0), W, H, W, 8);
    assert_eq!(shorts(&texel(&img, W, 8, 2, 2)), [6144, 0, 2048, 0]);
    assert_eq!(shorts(&texel(&img, W, 8, 40, 40)), [65535, 0, 2048, 65535]);
    let img = detiled(&mem, dest(1), W, H, W, 4);
    assert_eq!(px(&img, W, 2, 2), [24, 0, 8, 0]);
    assert_eq!(px(&img, W, 40, 40), [255, 0, 8, 255]);
}

/// Blending into k_16_16_16_16 works on the value / 32 (ONE + ONE adds
/// the snorm texels) where the host blends snorm16, and is dropped with a
/// warning, not the whole draw, where it cannot (lavapipe).
#[test]
fn fixed16_target_blends_or_draws_without_blending() {
    let Some(mut d) = drawer() else { return };
    if !d.snorm16() {
        eprintln!("SKIP: the host cannot render into 16-bit snorm");
        return;
    }
    let (_, report) = clear_draw_resolve(
        &mut d,
        RT_16_16_16_16,
        (0x0400_0400, 0x0400_0400),
        [1.0, -1.0, 8.0, 0.5],
        true,
        &[],
    );
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let dropped = report.warnings.iter().any(|w| w.contains("16-bit snorm"));
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgba16Fixed), 0, 0, W, H)
        .unwrap();
    let expected = if dropped {
        [0x0400, 0xFC00, 0x2000, 0x0200]
    } else {
        [0x0800, 0x0000, 0x2400, 0x0600]
    };
    assert_eq!(shorts(&texel(&raw, W, 8, 2, 2)), expected, "{report:?}");
}

/// k_16_16_FLOAT as Rg16Float: a clear writes the two halves of
/// RB_COLOR_CLEAR, a resolve to 16_16_FLOAT copies them, one to 8_8_8_8
/// (GPU) gets blue and alpha 0 (Xenia unpacks only red and green).
#[test]
fn rg16f_target_clear_draw_and_resolve() {
    use xenon_draw::formats::*;
    let Some(mut d) = drawer() else { return };
    let dest2 = DEST + 0x1_0000;
    let (mem, report) = clear_draw_resolve(
        &mut d,
        RT_16_16_FLOAT,
        (0x3800_3C00, 0),
        [2.0, -1.5, 9.0, 9.0],
        false,
        &[(DEST, DEST_16_16_FLOAT, 0), (dest2, DEST_8_8_8_8, 0)],
    );
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rg16Float), 0, 0, W, H)
        .unwrap();
    assert_eq!(shorts(&texel(&raw, W, 4, 2, 2)), [0x4000, 0xBE00]);
    assert_eq!(shorts(&texel(&raw, W, 4, 40, 40)), [0x3C00, 0x3800]);
    assert_eq!(detiled(&mem, DEST, W, H, W, 4), raw);
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!(px(&img, W, 2, 2), [255, 0, 0, 0]);
    assert_eq!(px(&img, W, 40, 40), [255, 128, 0, 0]);
}

/// k_32_32_FLOAT as Rg32Float: the 64-bit clear value is red | green <<
/// 32, blending is dropped with a warning (no FLOAT32_BLENDABLE), a resolve
/// to 32_32_FLOAT copies both floats and one to 32_FLOAT (GPU) keeps red
/// with the exponent bias.
#[test]
fn rg32f_target_clear_draw_and_resolve() {
    use xenon_draw::formats::*;
    let Some(mut d) = drawer() else { return };
    let dest2 = DEST + 0x2_0000;
    let (mem, report) = clear_draw_resolve(
        &mut d,
        RT_32_32_FLOAT,
        (0.25f32.to_bits(), (-3.0f32).to_bits()),
        [6.5, 1e10, 1.0, 1.0],
        true,
        &[(DEST, DEST_32_32_FLOAT, 0), (dest2, DEST_32_FLOAT, -1)],
    );
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert!(
        report.warnings.iter().any(|w| w.contains("32-bit float")),
        "{report:?}"
    );
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rg32Float), 0, 0, W, H)
        .unwrap();
    let fw = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(words(&raw, 8, 2, 2), fw(&[6.5, 1e10]));
    assert_eq!(words(&raw, 8, 40, 40), fw(&[0.25, -3.0]));
    assert_eq!(detiled(&mem, DEST, W, H, W, 8), raw);
    let img = detiled(&mem, dest2, W, H, W, 4);
    assert_eq!(words(&img, 4, 2, 2), fw(&[3.25]));
    assert_eq!(words(&img, 4, 40, 40), fw(&[0.125]));
}

/// A resolve whose copy is unsupported (destination format 0) still
/// clears, as Xenia does, and reports the copy.
#[test]
fn resolve_clears_when_the_copy_is_unsupported() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_COLOR_INFO, &[0]).resolve_with(
        &mut mem,
        (W, H),
        COPY_CONVERT | CLEAR_COLOR,
        (DEST, W | (H << 16), dest_info(0, 0, false)),
        [0, rgba([10, 20, 30, 40]), 0],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(
        report.unsupported,
        ["resolve copy to destination format 0"],
        "{report:?}"
    );
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert!(img.chunks_exact(4).all(|p| p == [10, 20, 30, 40]));
}

/// RB_COLOR_INFO's colour exponent bias (bits 20..25, signed) scales every
/// component of the pixel shader's colour by 2^bias before it is written:
/// -2 for the first draw, +1 for the second.
#[test]
fn color_exp_bias_scales_pixel_output() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &[corner_triangle(128.0), corner_triangle(16.0)].concat(),
    );
    let info = |bias: i32| (RT_16_16_16_16_FLOAT << 16) | (((bias as u32) & 0x3F) << 20);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[info(-2)])
        .shader(false, &vs_color())
        .shader(true, &ps_const())
        .vertex_fetch(0, VERTS, 24);
    c0(&mut s, [1.0, -0.5, 3.0, 0.25]).draw_auto(PRIM_TRI_LIST, 3);
    s.regs(RB_COLOR_INFO, &[info(1)])
        .vertex_fetch(0, VERTS + 48, 12);
    c0(&mut s, [0.25, -0.5, 3.0, 0.125]).draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgba16Float), 0, 0, W, H)
        .unwrap();
    // 0.25, -0.125, 0.75, 0.0625.
    assert_eq!(
        halves(&texel(&raw, W, 8, 40, 4)),
        [0x3400, 0xB000, 0x3A00, 0x2C00]
    );
    // 0.5, -1.0, 6.0, 0.25.
    assert_eq!(
        halves(&texel(&raw, W, 8, 2, 2)),
        [0x3800, 0xBC00, 0x4600, 0x3400]
    );
}

/// A texture fetch applies the fetch constant's sign per component and its
/// exponent adjust: an 8_8_8_8 texel (0x80, 12, 200, 255) with signs
/// signed, unsigned biased, gamma, signed, swizzle YXZ1 and exp_adjust 1
/// reads as 2 * (12/255 * 2 - 1, -1 (clamped from -128/127), PWL gamma
/// 580/1023, 1): each sign follows its source component through the
/// swizzle, and the constant 1 takes no sign.
#[test]
fn texture_fetch_sign_gamma_and_exp_adjust() {
    use xenon_texture::{endian, tile};
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let n = 32u32;
    let linear: Vec<u8> = (0..n * n).flat_map(|_| [0x80, 12, 200, 255]).collect();
    let mut tiled = tile::tile_2d(&linear, n, n, n, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
    let q = n as f32;
    let verts: Vec<u32> = [(0.0, 0.0, 0.0, 0.0), (q, 0.0, 1.0, 0.0), (0.0, q, 0.0, 1.0)]
        .iter()
        .flat_map(|&(x, y, u, v)| [f(x), f(y), f(0.0), f(u), f(v)])
        .collect();
    mem.write_words(VERTS, &verts);
    let swizzle_yxz1 = 1 | (2 << 6) | (5 << 9);
    let signs = (1 << 2) | (2 << 4) | (3 << 6) | (1 << 8);
    let fetch = [
        2 | signs | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | TEXTURE, // 8_8_8_8, k8in32
        (n - 1) | ((n - 1) << 13),
        (swizzle_yxz1 << 1) | (1 << 13), // exp_adjust 1, point filters
        0,
        1 << 9, // 2D
    ];
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_16_16_16_16_FLOAT << 16])
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 15)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let raw = d
        .read_render_target_raw(key(TargetFormat::Rgba16Float), 0, 0, W, H)
        .unwrap();
    // -1.8115 (rounded down), -2.0, 1.1338, 2.0.
    assert_eq!(
        halves(&texel(&raw, W, 8, 4, 4)),
        [0xBF3F, 0xC000, 0x3C89, 0x4000]
    );
}

/// The texture log reports each sampled fetch constant once with its
/// decoded fields, only for the pixel shader it is filtered to.
#[test]
fn texture_log_reports_sampled_fetch_fields() {
    let Some(mut d) = drawer() else { return };
    let swizzle_zyxw = 2 | (1 << 3) | (3 << 9);
    let signs = (2 << 2) | (3 << 4) | (1 << 8);
    let fetch = [
        2 | signs | (2 << 10) | (2 << 13) | (1 << 22),
        6 | (2 << 6) | TEXTURE,
        0,
        (swizzle_zyxw << 1) | (0x3F << 13), // exp_adjust -1
        0,
        1 << 9,
    ];
    let line = |d: &mut xenon_draw::Drawer| {
        let mut mem = memory();
        let mut s = Stream::default();
        s.setup()
            .regs(RB_COLOR_INFO, &[RT_16_16_16_16_FLOAT << 16])
            .shader(false, &vs_uv())
            .shader(true, &ps_texture(1))
            .vertex_fetch(0, VERTS, 15)
            .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
            .draw_auto(PRIM_TRI_LIST, 3)
            .draw_auto(PRIM_TRI_LIST, 3);
        let r = run(d, &s, &mut mem);
        r.warnings
            .into_iter()
            .filter(|w| w.starts_with("texture log"))
            .collect::<Vec<_>>()
    };
    d.texture_log = Some(Some(0));
    assert!(line(&mut d).is_empty());
    d.texture_log = Some(None);
    let lines = line(&mut d);
    assert_eq!(lines.len(), 1, "{lines:?}");
    let l = &lines[0];
    for part in [
        " t1 type 2 K8_8_8_8 1x1 ",
        " swizzle zyxw sign 2301 exp_adjust -1 endian K8in32 ",
        " mips 0-0 at 00000000 lod_bias 0 instr_bias 0 computed lod |",
    ] {
        assert!(l.contains(part), "{l}");
    }
    // Filtered to the draw's own pixel shader hash, the line is kept.
    let ps = &l["texture log: ps ".len()..][..16];
    d.texture_log = Some(Some(u64::from_str_radix(ps, 16).unwrap()));
    assert_eq!(line(&mut d), lines);
}

/// Signed components convert before filtering: a 2x1 signed 8_8_8_8 texture
/// of 0x7F (1.0) next to 0x80 (-1.0), sampled with linear filtering exactly
/// between the two texel centres, reads 0 in every component. Converting the
/// filtered unsigned value instead would read 127.5/255 as about -1 or +1.
#[test]
fn signed_texels_filter_after_conversion() {
    let Some(mut d) = drawer() else { return };
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let fetch = [
        2 | (0x55 << 2) | (2 << 10) | (2 << 13) | (1 << 22), // all signed
        6 | TEXTURE,                                         // 8_8_8_8
        1,                                                   // 2x1
        (swizzle_rgba << 1) | (1 << 19) | (1 << 21),         // linear mag/min
        0,
        1 << 9,
    ];
    let bytes = [[0x7F; 4], [0x80; 4]].concat();
    let raw = sample_fetch(&mut d, RT_16_16_16_16_FLOAT, fetch, &bytes);
    // Every component within 2^-8 of 0 (either sign).
    let h = halves(&raw);
    assert!(h.iter().all(|&h| h & 0x7FFF <= 0x1C00), "{h:x?}");
}

/// A texture signed in every component uploads as halves: a signed 8_8_8_8
/// texel 0x40 reads 64/127 rounded to the nearest half, 1032/2048 (within
/// 2^-12 of it), and 0x81 reads -1 exactly.
#[test]
fn all_signed_8_8_8_8_texel_rounds_to_a_half() {
    let Some(mut d) = drawer() else { return };
    let fetch = [
        2 | (0x55 << 2) | (2 << 10) | (2 << 13) | (1 << 22),
        6 | TEXTURE, // 8_8_8_8
        0,
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1, // swizzle XYZW
        0,
        1 << 9,
    ];
    let bytes = [0x40, 0x81, 0, 0];
    let red = |raw: Vec<u8>| f32::from_le_bytes(raw.try_into().unwrap());
    let (raw, report) = sample_fetch_report(&mut d, RT_32_FLOAT, fetch, &bytes);
    assert_eq!(red(raw), 1032.0 / 2048.0);
    // Bucket 2 is Rgba16Float: one 1x1 texel of four halves.
    assert_eq!(report.timings.uploads_by_format[2], (1, 8), "{report:?}");
    let mut fetch = fetch;
    fetch[3] = (1 | (2 << 6) | (3 << 9)) << 1; // swizzle YXZW
    assert_eq!(red(sample_fetch(&mut d, RT_32_FLOAT, fetch, &bytes)), -1.0);
}

/// A signed 16_16 texel reads its fields exactly as two's complement:
/// 0x8001 is -1 and 0x4000 is 16384/32767, both exact in a 32-bit float
/// target.
#[test]
fn signed_16_16_texel_is_exact() {
    let Some(mut d) = drawer() else { return };
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let fetch = [
        2 | (0x55 << 2) | (2 << 10) | (2 << 13) | (1 << 22),
        25 | TEXTURE, // 16_16
        0,
        swizzle_rgba << 1,
        0,
        1 << 9,
    ];
    let mut bytes = 0x8001u16.to_le_bytes().to_vec();
    bytes.extend_from_slice(&0x4000u16.to_le_bytes());
    let red = |raw: Vec<u8>| f32::from_le_bytes(raw.try_into().unwrap());
    assert_eq!(red(sample_fetch(&mut d, RT_32_FLOAT, fetch, &bytes)), -1.0);
    // Swizzle YXZW puts Y in red.
    let mut fetch = fetch;
    fetch[3] = (1 | (2 << 6) | (3 << 9)) << 1;
    let y = red(sample_fetch(&mut d, RT_32_FLOAT, fetch, &bytes));
    assert_eq!(y, 16384.0 / 32767.0);
}
