//! Rectangle lists: the host makes each rectangle's fourth corner from the
//! three guest vertices after the vertex shader. They need a wgpu adapter
//! (lavapipe works); without one they print a skip line, unless
//! XENON_REQUIRE_GPU=1 makes them fail.

mod common;

use common::*;
use xenon_gpu::regs::*;

const PRIM_RECT_LIST: u32 = 8;
const CLEAR: [u8; 4] = [10, 20, 30, 255];

const TL: (f32, f32) = (16.0, 16.0);
const TR: (f32, f32) = (48.0, 16.0);
const BL: (f32, f32) = (16.0, 48.0);
const BLACK: [u8; 4] = [0, 0, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

fn vert(p: (f32, f32), c: [u8; 4]) -> (f32, f32, f32, [u8; 4]) {
    (p.0, p.1, 0.0, c)
}

/// Draws `verts` (vs_color layout) as a rectangle list, with auto indices or
/// 16-bit DMA `indices`, and returns the 64x64 render target.
fn draw_rects(
    verts: &[(f32, f32, f32, [u8; 4])],
    indices: Option<&[u16]>,
    cull_both: bool,
) -> Vec<u8> {
    let Some(mut d) = drawer() else {
        return Vec::new();
    };
    let mut mem = memory();
    mem.write_words(VERTS, &color_verts(verts));
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 4 * verts.len() as u32)
        .regs(PA_SU_SC_MODE_CNTL, &[if cull_both { 3 } else { 0 }]);
    match indices {
        Some(idx) => {
            // Big-endian 16-bit indices, read back with k8in16.
            let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_be_bytes()).collect();
            mem.data[INDICES as usize..][..bytes.len()].copy_from_slice(&bytes);
            s.draw_dma16(PRIM_RECT_LIST, idx.len() as u32);
        }
        None => {
            s.draw_auto(PRIM_RECT_LIST, verts.len() as u32);
        }
    }
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap()
}

/// The rectangle (16,16)-(48,48) with black at the top left, red at the top
/// right and green at the bottom left: red grows with x, green with y, so
/// the bottom right (the made corner) is yellow. Coverage is exact; the
/// interpolated colours get a tolerance of 1 for the rasterizer's
/// interpolation precision.
fn check_rect(img: &[u8]) {
    if img.is_empty() {
        return; // no adapter
    }
    // Pixel centres at +0.5: x = 16 -> 0.5 / 32 of the way, 47 -> 31.5 / 32.
    let lo = 4; // 0.015625 * 255 = 3.98
    let hi = 251; // 0.984375 * 255 = 251.02
    for (x, y, c) in [
        (16, 16, [lo, lo, 0, 255]),
        (47, 16, [hi, lo, 0, 255]),
        (16, 47, [lo, hi, 0, 255]),
        (47, 47, [hi, hi, 0, 255]),
        (40, 24, [195, 68, 0, 255]), // 24.5 / 32 and 8.5 / 32
    ] {
        let p = px(img, W, x, y);
        assert!(near(p, c, 1), "({x}, {y}): {p:?}, want {c:?}");
    }
    // Every pixel of both triangles, the shared diagonal included (a crack
    // there would leave the clear colour).
    let unorm = |v: u32| (((v as f32 + 0.5 - 16.0) / 32.0) * 255.0).round() as u8;
    for y in 16..48 {
        for x in 16..48 {
            let (p, c) = (px(img, W, x, y), [unorm(x), unorm(y), 0, 255]);
            assert!(near(p, c, 1), "({x}, {y}): {p:?}, want {c:?}");
        }
    }
    for (x, y) in [
        (15, 16),
        (48, 16),
        (16, 15),
        (16, 48),
        (48, 48),
        (47, 48),
        (48, 47),
    ] {
        assert_eq!(px(img, W, x, y), CLEAR, "({x}, {y})");
    }
}

/// Auto indices, the right angle at vertex 0.
#[test]
fn rect_list_auto_indices() {
    let img = draw_rects(
        &[vert(TL, BLACK), vert(TR, RED), vert(BL, GREEN)],
        None,
        false,
    );
    check_rect(&img);
}

/// Auto indices, the right angle at vertex 1; culling both faces does not
/// apply to rectangle lists (not polygonal for Xenia).
#[test]
fn rect_list_corner_at_vertex_1_ignores_culling() {
    let img = draw_rects(
        &[vert(TR, RED), vert(TL, BLACK), vert(BL, GREEN)],
        None,
        true,
    );
    check_rect(&img);
}

/// DMA indices: two rectangles, the first with its right angle at its
/// third index (vertex 2 of the rectangle), the second a white 8x8 square
/// at the origin.
#[test]
fn rect_list_dma_indices_two_rects() {
    let white = [255, 255, 255, 255];
    let img = draw_rects(
        &[
            vert(BL, GREEN),
            vert(TL, BLACK),
            vert(TR, RED),
            vert((0.0, 8.0), white),
            vert((8.0, 0.0), white),
            vert((0.0, 0.0), white),
        ],
        Some(&[2, 0, 1, 5, 4, 3]),
        false,
    );
    check_rect(&img);
    if img.is_empty() {
        return;
    }
    for (x, y) in [(0, 0), (7, 0), (0, 7), (7, 7)] {
        assert_eq!(px(&img, W, x, y), white, "({x}, {y})");
    }
    for (x, y) in [(8, 0), (0, 8), (8, 8)] {
        assert_eq!(px(&img, W, x, y), CLEAR, "({x}, {y})");
    }
}
