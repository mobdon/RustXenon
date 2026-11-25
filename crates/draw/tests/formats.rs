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
const RT_16_16_16_16_FLOAT: u32 = 7;
const RT_2_10_10_10_FLOAT_AS_16_16_16_16: u32 = 12;
const RT_32_FLOAT: u32 = 14;

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
/// gradient of every source format, both destination formats, exponent
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
    ];
    for (rt, format) in [
        (0, TargetFormat::Rgba8),
        (RT_2_10_10_10, TargetFormat::Rgb10a2),
        (RT_2_10_10_10_FLOAT_AS_16_16_16_16, TargetFormat::Float7e3),
        (RT_16_16_16_16_FLOAT, TargetFormat::Rgba16Float),
    ] {
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
/// signed, unsigned biased, gamma, unsigned and exp_adjust 1 reads as
/// 2 * (-1 (clamped from -128/127), 12/255 * 2 - 1, PWL gamma 580/1023, 1).
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
    let swizzle_rgba = (1 << 3) | (2 << 6) | (3 << 9);
    let signs = (1 << 2) | (2 << 4) | (3 << 6);
    let fetch = [
        2 | signs | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | TEXTURE, // 8_8_8_8, k8in32
        (n - 1) | ((n - 1) << 13),
        (swizzle_rgba << 1) | (1 << 13), // exp_adjust 1, point filters
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
    // -2.0, -1.8115 (rounded down), 1.1338, 2.0.
    assert_eq!(
        halves(&texel(&raw, W, 8, 4, 4)),
        [0xC000, 0xBF3F, 0x3C89, 0x4000]
    );
}
