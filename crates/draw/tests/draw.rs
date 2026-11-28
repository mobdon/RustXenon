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
    // 1.5 MB of targets at scale 1, scale^2 times that above (XENON_SCALE).
    let area = d.scale() * d.scale();
    let mb = 3 * area / 2;
    assert!(m.contains(&format!("targets {mb} MB in 2,")), "{m}");
    // wgpu's own counters are live: the targets above are counted.
    let live = area;
    assert!(
        m.contains(&format!("; wgpu live: textures {live} MB,")),
        "{m}"
    );
    assert!(m.contains(" MB reserved in "), "{m}");
    // Small allocator blocks: a 1 MB target does not reserve 64 MB or more.
    let reserved: u64 = m
        .split(" MB reserved")
        .next()
        .and_then(|s| s.rsplit(' ').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(u64::MAX);
    assert!(reserved < 64, "{m}");
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

/// With pipelines made on worker threads, a draw whose shaders or pipeline
/// are not made yet is skipped and counted, and the same draw in a replay
/// after they are made is drawn.
#[test]
fn draw_waits_for_its_pipeline_without_stalling() {
    let Some(mut d) = drawer() else { return };
    d.async_pipelines = true;
    let mut mem = memory();
    let white = [255, 255, 255, 255];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (-1.0, 1.0, 0.0, white),
            (1.0, 1.0, 0.0, white),
            (-1.0, -1.0, 0.0, white),
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
    let mut waited = 0;
    for _ in 0..3 {
        let report = run(&mut d, &s, &mut mem);
        assert!(report.unsupported.is_empty(), "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        if report.draws == 0 {
            // Skipped first for the translations, then for the pipeline.
            assert_eq!(report.timings.draws_waiting, 1, "{report:?}");
            assert_eq!(px(&img, W, 8, 8), CLEAR);
            waited += 1;
            d.wait_pipelines();
            continue;
        }
        assert_eq!(report.timings.draws_waiting, 0, "{report:?}");
        assert_eq!(px(&img, W, 8, 8), white);
        assert_eq!(px(&img, W, 60, 40), CLEAR);
        assert_eq!(waited, 2, "translations, then the pipeline");
        return;
    }
    panic!("never drawn");
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

/// Two draws of one guest index buffer with different vertices: the second
/// binds the indices the first staged, and both land where their vertices
/// put them.
#[test]
fn draws_of_one_index_buffer_share_its_staged_indices() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    let square = |x: f32, c| {
        [
            (x, 0.0, 0.0, c),
            (x + 16.0, 0.0, 0.0, c),
            (x, 16.0, 0.0, c),
            (x + 16.0, 16.0, 0.0, c),
        ]
    };
    mem.write_words(VERTS, &color_verts(&square(0.0, red)));
    mem.write_words(VERTS + 0x100, &color_verts(&square(32.0, green)));
    // Big-endian 16-bit indices 3 1 2 0 (k8in16 swaps them back).
    mem.write_words(INDICES, &[0x0003_0001, 0x0002_0000]);
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 16)
        .draw_dma16(PRIM_TRI_STRIP, 4)
        .vertex_fetch(0, VERTS + 0x100, 16)
        .draw_dma16(PRIM_TRI_STRIP, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert_eq!(report.timings.indices_reused, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 8), red);
    assert_eq!(px(&img, W, 40, 8), green);
    assert_eq!(px(&img, W, 24, 8), CLEAR);
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
    let rgba8 = (1, report.timings.texture_bytes);
    assert_eq!(report.timings.uploads_by_format[0], rgba8, "{report:?}");

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

    // Unnormalized coordinates (in texels, over the fetch constant's size)
    // sample the same texels.
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let verts: Vec<u32> = [(0.0, 0.0), (q, 0.0), (q, q), (0.0, q)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0), f(x), f(y)])
        .collect();
    mem.write_words(VERTS, &verts);
    let mut tf = TextureFetch::fetch_2d(1, 0, 1);
    tf.unnormalized = true;
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
        .shader(false, &vs_uv())
        .shader(true, &ps)
        .vertex_fetch(0, VERTS, 20)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    for y in 0..n {
        for x in 0..n {
            assert_eq!(px(&img, W, x, y), texel2(x, y), "({x}, {y})");
        }
    }
}

/// A tiled 8_8_8_8 texture with a mip page: a 32x32 red base and a 16x16
/// green level 1. Drawn texel for texel it samples the base; drawn at 8x8
/// (LOD 2, clamped to the last level) it samples level 1. Both levels are
/// uploaded and hashed, so new mip bytes alone decode the chain again.
#[test]
fn guest_mip_level_is_sampled() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let (red, green, blue) = ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]);
    solid_level(&mut mem, TEXTURE, 32, red);
    solid_level(&mut mem, TEXTURE + 0x1000, 16, green);
    let s = mip_stream(&mut mem, TEXTURE, 0, TEXTURE + 0x1000);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 2, "{report:?}");
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    assert_eq!(report.timings.texture_bytes, (32 * 32 + 16 * 16) * 4);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 5, 5), red);
    assert_eq!(px(&img, W, 44, 4), green);

    // New level 1 bytes, same base: decoded again.
    solid_level(&mut mem, TEXTURE + 0x1000, 16, blue);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 5, 5), red);
    assert_eq!(px(&img, W, 44, 4), blue);
}

/// A fetch sampling from level 1 whose base page is valid but whose mip
/// page cannot be read draws its base, as without mips: neither the
/// skipped base nor the unread mip page turns it to zeros.
#[test]
fn guest_mip_chain_falls_back_to_base() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let red = [255, 0, 0, 255];
    solid_level(&mut mem, TEXTURE, 32, red);
    let s = mip_stream(&mut mem, TEXTURE, 1, 0x40_0000);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    assert_eq!(report.timings.texture_bytes, 32 * 32 * 4);
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 5, 5), red);
    assert_eq!(px(&img, W, 44, 4), red);
}

/// A fetch with base page 0 and a mip page: the base is neither read,
/// hashed nor uploaded, every LOD samples level 1, and new level 1 bytes
/// decode it again.
#[test]
fn guest_mip_chain_without_base_page() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let (green, blue) = ([0, 255, 0, 255], [0, 0, 255, 255]);
    solid_level(&mut mem, TEXTURE + 0x1000, 16, green);
    let s = mip_stream(&mut mem, 0, 0, TEXTURE + 0x1000);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    assert_eq!(report.timings.texture_bytes, 16 * 16 * 4);
    assert_eq!(report.timings.hashed_bytes, 0x1000, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 5, 5), green);
    assert_eq!(px(&img, W, 44, 4), green);

    solid_level(&mut mem, TEXTURE + 0x1000, 16, blue);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.timings.textures_uploaded, 1, "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 44, 4), blue);
}

/// Writes a tiled k8in32 8_8_8_8 `n`x`n` level of colour `c` at `at`.
fn solid_level(mem: &mut xenon_gpu::SliceMemory, at: u32, n: u32, c: [u8; 4]) {
    let linear: Vec<u8> = (0..n * n).flat_map(|_| c).collect();
    let mut tiled = tile::tile_2d(&linear, n, n, 32, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    mem.data[at as usize..][..tiled.len()].copy_from_slice(&tiled);
}

/// Two draws of a tiled 32x32 8_8_8_8 texture at `base` with mip levels up
/// to 1 from `min` at `mip`: texel for texel at (0, 0), and at 8x8 (LOD 2,
/// clamped to the last level) at (40, 0).
fn mip_stream(mem: &mut xenon_gpu::SliceMemory, base: u32, min: u32, mip: u32) -> Stream {
    let n = 32u32;
    let quad = |x0: f32, q: f32| -> Vec<u32> {
        [
            (0.0, 0.0, 0.0, 0.0),
            (q, 0.0, 1.0, 0.0),
            (q, q, 1.0, 1.0),
            (0.0, q, 0.0, 1.0),
        ]
        .iter()
        .flat_map(|&(x, y, u, v)| [f(x0 + x), f(y), f(0.0), f(u), f(v)])
        .collect()
    };
    mem.write_words(VERTS, &quad(0.0, n as f32));
    mem.write_words(VERTS + 0x100, &quad(40.0, 8.0));
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31), // clamp, tiled
        6 | (2 << 6) | base,                                      // 8_8_8_8, k8in32
        (n - 1) | ((n - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1, // swizzle RGBA, point filters
        (min << 2) | (1 << 6),                 // mip_min_level, mip_max_level 1
        mip | (1 << 9),                        // mip page, 2D
    ];
    let mut s = Stream::default();
    s.setup()
        .resolve(mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .vertex_fetch(0, VERTS, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .vertex_fetch(0, VERTS + 0x100, 20)
        .draw_auto(PRIM_QUAD_LIST, 4);
    s
}

/// Draws a `w`x`h` DXT `format` (k8in16, nearest sampling, tiled or linear
/// with a pitch field of `pitch`) of pseudo-random blocks of `bpb` bytes over
/// the target's corner, at most `W` wide, with BC passthrough on or off;
/// returns the quad's pixels and the bytes uploaded.
fn draw_dxt(
    bc: bool,
    format: u32,
    bpb: u32,
    tiled: bool,
    (w, h, pitch): (u32, u32, u32),
) -> Option<(Vec<[u8; 4]>, u64)> {
    let mut d = drawer()?;
    if bc && !d.bc_textures {
        println!("skip: the adapter has no BC textures");
        return None;
    }
    d.bc_textures = bc;
    let mut mem = memory();
    let mut seed = 0x1234_5678u32;
    let linear: Vec<u8> = (0..w * h / 16 * bpb)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        })
        .collect();
    let mut guest = if tiled {
        tile::tile_2d(&linear, w / 4, h / 4, w / 4, bpb).unwrap()
    } else {
        let row = (w / 4 * bpb) as usize;
        let stride = (tile::align_to_tile(pitch * 8) * bpb) as usize;
        linear
            .chunks(row)
            .flat_map(|r| [r, &vec![0; stride.saturating_sub(row)]].concat())
            .collect()
    };
    endian::swap_in_place(&mut guest, xenon_texture::Endian::K8in16);
    mem.data[TEXTURE as usize..][..guest.len()].copy_from_slice(&guest);
    let (qw, qh) = (w.min(W), h);
    let (x, y) = (qw as f32, qh as f32);
    let verts: Vec<u32> = [
        (0.0, 0.0, 0.0, 0.0),
        (x, 0.0, 1.0, 0.0),
        (x, y, 1.0, 1.0),
        (0.0, y, 0.0, 1.0),
    ]
    .iter()
    .flat_map(|&(x, y, u, v)| [f(x), f(y), f(0.0), f(u), f(v)])
    .collect();
    mem.write_words(VERTS, &verts);
    let fetch = [
        2 | (2 << 10) | (2 << 13) | (pitch << 22) | (u32::from(tiled) << 31), // clamp
        format | (1 << 6) | TEXTURE,                                          // k8in16
        (w - 1) | ((h - 1) << 13),
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
    // Bucket 1 is BC blocks, bucket 0 the CPU decode's Rgba8Unorm.
    let by = report.timings.uploads_by_format[usize::from(bc)];
    assert_eq!(by, (1, report.timings.texture_bytes), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let pixels = (0..qw * qh).map(|i| px(&img, W, i % qw, i / qw)).collect();
    Some((pixels, report.timings.texture_bytes))
}

/// DXT1 and DXT5 uploaded as BC blocks sample the texels the CPU decoder
/// makes, in an eighth or a quarter of the bytes. Interpolated texels may
/// be 1 off: the host rounds the 1/3, 2/3 and 1/7.. mixes its own way
/// (lavapipe rounds where the CPU decoder truncates), as the BC spec allows.
#[test]
fn dxt_passthrough_matches_cpu_decode() {
    let square = (32, 32, 1);
    // One linear block row may have a pitch below its width.
    let strip = (256, 4, 0);
    for (format, bpb, tiled, size) in [
        (18, 8, true, square),
        (19, 16, true, square),
        (20, 16, true, square),
        (18, 8, false, square),
        (20, 16, false, strip),
    ] {
        let Some((bc, bc_bytes)) = draw_dxt(true, format, bpb, tiled, size) else {
            return;
        };
        let (cpu, cpu_bytes) = draw_dxt(false, format, bpb, tiled, size).unwrap();
        let texels = u64::from(size.0 * size.1);
        assert_eq!(cpu_bytes, texels * 4);
        assert_eq!(bc_bytes, texels / 16 * u64::from(bpb), "format {format}");
        for (i, (b, c)) in bc.iter().zip(&cpu).enumerate() {
            let near = b.iter().zip(c).all(|(&b, &c)| b.abs_diff(c) <= 1);
            assert!(
                near,
                "format {format} ({}, {}): {b:?} {c:?}",
                i as u32 % size.0.min(W),
                i as u32 / size.0.min(W)
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

/// The texture cache also evicts past its texture count, however few bytes
/// the textures hold: at 4 a fifth texture evicts the least recently drawn.
#[test]
fn texture_cache_evicts_past_its_count() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let n = 32u32;
    d.texture_cache_count = 4;
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
        (t.textures_uploaded, t.textures_evicted)
    };
    for k in 0..4 {
        assert_eq!(draw(&mut d, &mut mem, k), (1, 0), "texture {k}");
    }
    assert_eq!(draw(&mut d, &mut mem, 0), (0, 0));
    // Full: evicts 1, the least recently drawn, down to 3 of 4.
    assert_eq!(draw(&mut d, &mut mem, 4), (1, 1));
    assert_eq!(draw(&mut d, &mut mem, 0), (0, 0));
    assert_eq!(draw(&mut d, &mut mem, 2), (0, 0));
}

/// With guest memory listing its writes, an eviction drops only the memo
/// entries of evicted textures: the textures that stay are still found
/// without hashing their bytes, and an evicted one is read and uploaded again.
#[test]
fn eviction_keeps_memo_entries_of_kept_textures() {
    let Some(mut d) = drawer() else { return };
    let n = 32u32;
    d.texture_cache_count = 4;
    let mut w = Watched {
        mem: memory(),
        written: Some(vec![]),
    };
    let verts: Vec<u32> = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0), f(0.5), f(0.5)])
        .collect();
    w.mem.write_words(VERTS, &verts);
    let draw = |d: &mut xenon_draw::Drawer, w: &mut Watched, k: u32| {
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
        w.mem.write_words(CMD, &s.0);
        let mut rec = xenon_gpu::Recorder::new();
        rec.record_buffer(&w.mem, CMD, s.0.len() as u32).unwrap();
        let r = d
            .replay(&rec.commands, w as &mut dyn xenon_draw::GuestMemory)
            .unwrap();
        let t = r.timings;
        (t.textures_uploaded, t.textures_evicted, t.hashed_bytes)
    };
    let bytes = u64::from(n * n * 4);
    for k in 0..4 {
        assert_eq!(draw(&mut d, &mut w, k), (1, 0, bytes), "texture {k}");
    }
    assert_eq!(draw(&mut d, &mut w, 0), (0, 0, 0));
    // Full: evicts 1, the least recently drawn.
    assert_eq!(draw(&mut d, &mut w, 4), (1, 1, bytes));
    // Kept textures: still in the memo, nothing hashed.
    for k in [0, 2, 3, 4] {
        assert_eq!(draw(&mut d, &mut w, k), (0, 0, 0), "texture {k}");
    }
    // The evicted one is gone from the memo: read and uploaded again.
    assert_eq!(draw(&mut d, &mut w, 1), (1, 1, bytes));
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

/// A stage is bound only the span of float constants it reads (c7 alone
/// here, not c0): each draw gets the values at its time, and a draw after
/// no write to them binds the span staged before.
#[test]
fn float_constant_span_follows_each_draw() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let black = [0, 0, 0, 255];
    let tri = |x: f32, y: f32| {
        color_verts(&[
            (x, y, 0.0, black),
            (x + 32.0, y, 0.0, black),
            (x, y + 32.0, 0.0, black),
        ])
    };
    mem.write_words(VERTS, &tri(0.0, 0.0));
    mem.write_words(VERTS + 0x100, &tri(32.0, 32.0));
    mem.write_words(VERTS + 0x200, &tri(0.0, 32.0));
    let c = |v: u32| f(v as f32 / 255.0);
    // Pixel c7 is float constant 256 + 7.
    let c7 = SHADER_CONSTANT_FLOAT + (256 + 7) * 4;
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_const_n(7))
        .regs(c7, &[c(51), c(102), c(204), c(255)])
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(c7, &[c(255), c(153), c(0), c(255)])
        .vertex_fetch(0, VERTS + 0x100, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .vertex_fetch(0, VERTS + 0x200, 12)
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 4, 4), [51, 102, 204, 255]);
    assert_eq!(px(&img, W, 40, 36), [255, 153, 0, 255]);
    assert_eq!(px(&img, W, 4, 36), [255, 153, 0, 255]);
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

/// The front buffer is shown through fetch constant 0, as Xenia does: a
/// zyxw swizzle there undoes the red/blue swap of a BGRA front buffer.
#[test]
fn swap_applies_fetch_constant_0_swizzle() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    let zyxw = 2 | (1 << 3) | (3 << 9);
    s.regs(SHADER_CONSTANT_FETCH, &[2, DEST, 0, zyxw << 1, 0, 0])
        .swap(W, H);
    let report = run(&mut d, &s, &mut mem);
    let rt = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let swapped: Vec<u8> = rt
        .chunks_exact(4)
        .flat_map(|p| [p[2], p[1], p[0], p[3]])
        .collect();
    assert_ne!(rt, swapped);
    assert_eq!(report.frames[0].rgba8, swapped);

    // Fetch constant 0 describing another address leaves the frame as is.
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    s.regs(
        SHADER_CONSTANT_FETCH,
        &[2, DEST + 0x1000, 0, zyxw << 1, 0, 0],
    )
    .swap(W, H);
    assert_eq!(run(&mut d, &s, &mut mem).frames[0].rgba8, rt);
}

/// With late front buffers a swap gives the frame of the swap before it,
/// copied out on the GPU, and the first swap none.
#[test]
fn late_front_buffer_is_one_swap_behind() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    s.swap(W, H);
    let sync = run(&mut d, &s, &mut mem).frames.remove(0);
    let cleared = |mem: &mut xenon_gpu::SliceMemory| {
        let mut s = Stream::default();
        s.setup()
            .resolve(mem, W, H, true, Some((rgba(CLEAR), 0)))
            .swap(W, H);
        s
    };

    d.late_front_buffer = true;
    assert!(run(&mut d, &s, &mut mem).frames.is_empty());
    let next = cleared(&mut mem);
    let r = run(&mut d, &next, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    assert_eq!(r.frames[0].rgba8, sync.rgba8);
    // The cleared copy (a resolve copies before it clears) shows two later.
    run(&mut d, &next, &mut mem);
    let r = run(&mut d, &next, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    assert!(r.frames[0].rgba8.chunks_exact(4).all(|p| p == CLEAR));

    // A swap of memory no resolve kept is read now and drops the queued
    // copy, so the next late swap gives no older frame out of order.
    let mut plain = Stream::default();
    plain.swap_at(DEST + 0x4_0000, W, H);
    assert_eq!(run(&mut d, &plain, &mut mem).frames.len(), 1);
    assert!(run(&mut d, &next, &mut mem).frames.is_empty());
}

/// The late copy is shown through fetch constant 0's swizzle too. The host
/// turns late front buffers on, so this is the path a game's frames take.
#[test]
fn late_front_buffer_applies_fetch_constant_0_swizzle() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    let zyxw = 2 | (1 << 3) | (3 << 9);
    s.regs(SHADER_CONSTANT_FETCH, &[2, DEST, 0, zyxw << 1, 0, 0])
        .swap(W, H);
    let sync = run(&mut d, &s, &mut mem).frames.remove(0);
    let rt = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_ne!(sync.rgba8, rt);

    d.late_front_buffer = true;
    assert!(run(&mut d, &s, &mut mem).frames.is_empty());
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    assert_eq!(r.frames[0].rgba8, sync.rgba8);
}

/// With a window presenting on the GPU, a swap hands over a texture holding
/// the frame the read-back path gives (swizzle and all) once its batch is
/// submitted, and reports the frame with no pixels.
#[test]
fn present_hands_over_the_frame_on_the_gpu() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    let zyxw = 2 | (1 << 3) | (3 << 9);
    s.regs(SHADER_CONSTANT_FETCH, &[2, DEST, 0, zyxw << 1, 0, 0])
        .swap(W, H);
    let sync = run(&mut d, &s, &mut mem).frames.remove(0);

    let (tx, rx) = std::sync::mpsc::channel();
    let tx = std::sync::Mutex::new(tx);
    d.present = Some(std::sync::Arc::new(move |t| {
        let _ = tx.lock().unwrap().send(t);
    }));
    for _ in 0..2 {
        let r = run(&mut d, &s, &mut mem);
        assert_eq!(r.frames.len(), 1, "{r:?}");
        let f = &r.frames[0];
        assert_eq!((f.width, f.height, f.rgba8.len()), (W, H, 0));
        let tex: wgpu::Texture = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert_eq!((tex.width(), tex.height()), (W, H));
        let (_, _, device, queue) = d.gpu();
        let row = (4 * W).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * H),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(H),
                },
            },
            tex.size(),
        );
        queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let got: Vec<u8> = (buf.slice(..).get_mapped_range().unwrap())
            .chunks(row as usize)
            .flat_map(|r| r[..4 * W as usize].to_vec())
            .collect();
        assert_eq!(got, sync.rgba8);
    }
}

/// A frame bundle of two frames, saved and read back, replays to the same
/// last frame; with that frame's draw skipped, only the clear is left.
#[test]
fn bundle_replays_with_a_draw_skipped() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    draw_pattern(&mut mem, &mut s);
    s.swap(W, H);
    draw_pattern(&mut mem, &mut s);
    s.swap(W, H);
    mem.write_words(CMD, &s.0);
    let mut rec = xenon_gpu::Recorder::new();
    rec.record_buffer(&mem, CMD, s.0.len() as u32).unwrap();
    let dir = std::env::temp_dir().join(format!("xenon-draw-bundle-{}", std::process::id()));
    xenon_gpu::bundle::write_commands(&dir, &rec.commands).unwrap();
    std::fs::write(dir.join(xenon_gpu::bundle::MEMORY), &mem.data).unwrap();
    let (cmds, mut back) = xenon_gpu::bundle::read(&dir).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();

    let (all, n) = xenon_gpu::bundle::skip_draws(&cmds, 1, 0);
    assert_eq!(n, 1);
    let whole = d.replay(&all, &mut back).unwrap();
    let frame = &whole.frames.last().unwrap().rgba8;
    assert_ne!(px(frame, W, 2, 2), CLEAR);

    let (skipped, _) = xenon_gpu::bundle::skip_draws(&cmds, 1, 1);
    let r = d.replay(&skipped, &mut back).unwrap();
    assert_eq!((r.frames.len(), r.draws), (2, 1), "{r:?}");
    let frame = &r.frames[1].rgba8;
    assert!(frame.chunks_exact(4).all(|p| p == CLEAR));

    // Stopped after the last frame's draw, its target holds the pattern.
    let cut = xenon_gpu::bundle::cut_after_draw(&cmds, 1);
    d.replay(&cut, &mut back).unwrap();
    let (w, h) = d.render_target_size(COLOR_KEY).unwrap();
    assert!(w >= W && h >= H, "{w}x{h}");
    let rt = d.read_render_target(COLOR_KEY, 0, 0, w, h).unwrap();
    assert_ne!(px(&rt, w, 2, 2), CLEAR);
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

/// RB_COLORCONTROL's alpha test discards the pixels whose oC0 alpha fails
/// against RB_ALPHA_REF, with blending off (alpha-tested cut-out cards draw
/// opaque black around their shapes without it), and alpha to mask covers
/// from alpha 0.5.
#[test]
fn alpha_test_discards_failing_pixels() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let faint = [255, 0, 0, 64];
    let solid = [0, 255, 0, 200];
    mem.write_words(
        VERTS,
        &[
            color_verts(&[
                (0.0, 0.0, 0.0, faint),
                (32.0, 0.0, 0.0, faint),
                (0.0, 32.0, 0.0, faint),
            ]),
            color_verts(&[
                (32.0, 32.0, 0.0, solid),
                (64.0, 32.0, 0.0, solid),
                (32.0, 64.0, 0.0, solid),
            ]),
        ]
        .concat(),
    );
    let draw_both = |s: &mut Stream, colorcontrol: u32| {
        s.regs(RB_COLORCONTROL, &[colorcontrol])
            .vertex_fetch(0, VERTS, 12)
            .draw_auto(PRIM_TRI_LIST, 3)
            .vertex_fetch(0, VERTS + 48, 12)
            .draw_auto(PRIM_TRI_LIST, 3);
    };
    for (colorcontrol, faint_drawn) in [
        // Off, then always: both drawn.
        (0x04, true),
        (0x0F, true),
        // Greater than 0.5: only the alpha 200 one.
        (0x0C, false),
        // Less than 0.5: only the alpha 64 one.
        (0x09, true),
        // Alpha to mask.
        (0x10, false),
    ] {
        let mut s = Stream::default();
        s.setup()
            .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
            .shader(false, &vs_color())
            .shader(true, &ps_interp())
            .regs(RB_ALPHA_REF, &[f(0.5)]);
        draw_both(&mut s, colorcontrol);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.draws, 2, "{report:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        let want_faint = if faint_drawn { faint } else { CLEAR };
        let solid_drawn = colorcontrol & 0x0F != 0x09;
        let want_solid = if solid_drawn { solid } else { CLEAR };
        assert_eq!(
            px(&img, W, 4, 4),
            want_faint,
            "colorcontrol {colorcontrol:#x}"
        );
        assert_eq!(
            px(&img, W, 36, 36),
            want_solid,
            "colorcontrol {colorcontrol:#x}"
        );
    }
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
    depth_resolve_words_at(d, depth_info, clear, 0.25, 6, &ps_interp())
}

/// [`depth_resolve_words`] with the triangle at depth `z`, destination
/// format `dest_format` (ignored for depth copies) and pixel shader `ps`.
fn depth_resolve_words_at(
    d: &mut xenon_draw::Drawer,
    depth_info: u32,
    clear: u32,
    z: f32,
    dest_format: u32,
    ps: &[u32],
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
        .shader(true, ps)
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
        depth_resolve_words_at(&mut d, DEPTH_BASE, 0x1234_5678, 0.75, 25, &ps_interp()),
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
        depth_resolve_words_at(&mut d, DEPTH_BASE | (1 << 16), 0x78, z, 6, &ps_interp()).0,
        (0xD3_3333 << 8) | stencil
    );
}

/// A pixel shader's oDepth replaces the triangle's depth (here oDepth.x is
/// the white colour's red, 1.0, over the triangle's 0.25), and without a
/// depth target the same shader still draws its colour.
#[test]
fn depth_export_writes_depth() {
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let Some(mut d) = drawer() else { return };
    let r0 = AluSrc::temp(0, SWZ_XYZW);
    let ps = assemble(&[exec(0, 2, true)], &[export(0, r0), export(61, r0)]);
    assert_eq!(
        depth_resolve_words_at(&mut d, DEPTH_BASE, 0x1234_5678, 0.25, 6, &ps).0,
        (0xFF_FFFF << 8) | 0x78
    );

    let mut mem = memory();
    mem.write_words(VERTS, &full_triangle(WHITE, 0.0));
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
    assert_eq!(px(&img, W, 8, 8), WHITE);
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
    // shortcut: derivatives are per host pixel, so above scale 1 they are
    // 1 / scale of the guest's; multiply them by xe_scale in the shader if
    // a game reads them.
    let want = 4 / d.scale() as u8;
    assert!(
        near(px(&img, W, 8, 8), [want, 0, 0, 255], 1),
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
        // shortcut: points are one host pixel, so above scale 1 the guest
        // pixel averages the point with the clear; size them by the scale
        // with PA_SU_POINT_SIZE support.
        if d.scale() > 1 {
            continue;
        }
        assert!(near(p, want, 1), "SQ_PROGRAM_CNTL {program_cntl:#x}: {p:?}");
    }
}

/// Draws the triangle of `triangle_interpolates_vertex_colours` and
/// returns the report and the render target.
fn draw_triangle(d: &mut xenon_draw::Drawer) -> (xenon_draw::Report, Vec<u8>) {
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
    let report = run(d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert!(near(px(&img, W, 8, 8), [191, 32, 32, 255], 2));
    (report, img)
}

/// The pipeline cache file is written when the drawer drops (or on a save
/// after new pipelines), read by the
/// next drawer, and replaced when it is garbage or from another adapter;
/// the drawer draws the same with and without it.
#[test]
fn pipeline_cache_persists_across_drawers() {
    let dir = std::env::temp_dir().join(format!("xenon-pipeline-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let reference = {
        let Some(mut d) = drawer() else { return };
        // No cache set: nothing to save.
        d.save_pipeline_cache().unwrap();
        let (report, img) = draw_triangle(&mut d);
        assert!(report.timings.pipelines_created >= 1, "{report:?}");
        assert!(report.timings.pipeline > std::time::Duration::ZERO);
        img
    };
    let file = {
        let mut d = drawer().unwrap();
        let Some(file) = d.set_pipeline_cache_dir(&dir).unwrap() else {
            eprintln!("SKIP: no pipeline cache on {}", d.adapter);
            return;
        };
        assert!(!file.exists());
        assert_eq!(draw_triangle(&mut d).1, reference);
        file
    };
    let saved = std::fs::read(&file).unwrap();
    assert!(saved.starts_with(b"WGPUPLCH"), "{} bytes", saved.len());
    // No temporary file left behind.
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    // Reloaded: same drawing, and saved again (the driver's data may
    // grow but keeps wgpu's header).
    {
        let mut d = drawer().unwrap();
        assert_eq!(d.set_pipeline_cache_dir(&dir).unwrap(), Some(file.clone()));
        assert_eq!(draw_triangle(&mut d).1, reference);
    }
    assert_eq!(std::fs::read(&file).unwrap()[..32], saved[..32]);
    // Garbage, then a header for another adapter (wgpu refuses that one
    // even with `fallback`): an empty cache instead, and a valid file after.
    let mut other = saved.clone();
    other[20] ^= 0xFF;
    for bad in [b"not a cache".to_vec(), other] {
        std::fs::write(&file, bad).unwrap();
        {
            let mut d = drawer().unwrap();
            assert_eq!(d.set_pipeline_cache_dir(&dir).unwrap(), Some(file.clone()));
            assert_eq!(draw_triangle(&mut d).1, reference);
        }
        assert_eq!(std::fs::read(&file).unwrap()[..32], saved[..32]);
    }
    // A save writes only when pipelines were made since the last one.
    std::fs::remove_file(&file).unwrap();
    let mut d = drawer().unwrap();
    d.set_pipeline_cache_dir(&dir).unwrap();
    d.save_pipeline_cache().unwrap();
    assert!(!file.exists());
    draw_triangle(&mut d);
    d.save_pipeline_cache().unwrap();
    // The write runs on its own thread: the next save waits for it.
    d.save_pipeline_cache().unwrap();
    assert!(file.exists());
    std::fs::remove_file(&file).unwrap();
    d.save_pipeline_cache().unwrap();
    assert!(!file.exists());
    drop(d);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Guest memory that lists the ranges written to it, as watched memory
/// does: whatever the test says was written since the last replay.
struct Watched {
    mem: xenon_gpu::SliceMemory,
    written: Option<Vec<(u64, u64)>>,
}

impl xenon_gpu::GuestRead for Watched {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        self.mem.read_u32(addr)
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        self.mem.read_bytes(addr, len)
    }
}

impl xenon_draw::GuestMemory for Watched {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        self.mem.write_bytes(addr, bytes)
    }
    fn take_written(&mut self) -> Option<Vec<(u64, u64)>> {
        self.written.replace(Vec::new())
    }
}

/// With guest memory listing its writes, a texture looked up in one replay
/// is used in the next without reading its bytes again, and vertices stay
/// staged, until their bytes are written.
#[test]
fn listed_writes_keep_textures_across_replays() {
    let Some(mut d) = drawer() else { return };
    let n = 32u32;
    let put_texture = |mem: &mut xenon_gpu::SliceMemory, shade: u8| {
        let linear: Vec<u8> = (0..n * n).flat_map(|_| [shade, 0, 0, 255]).collect();
        let mut tiled = tile::tile_2d(&linear, n, n, n, 4).unwrap();
        endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
        mem.data[TEXTURE as usize..][..tiled.len()].copy_from_slice(&tiled);
        tiled.len() as u64
    };
    let mut mem = memory();
    let len = put_texture(&mut mem, 200);
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
        2 | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | TEXTURE,
        (n - 1) | ((n - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
        0,
        1 << 9,
    ];
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 20)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    mem.write_words(CMD, &s.0);
    let mut rec = xenon_gpu::Recorder::new();
    rec.record_buffer(&mem, CMD, s.0.len() as u32).unwrap();
    let cmds = rec.commands;
    let mut w = Watched {
        mem,
        written: Some(vec![]),
    };
    let replay = |d: &mut xenon_draw::Drawer, w: &mut Watched| {
        let r = d
            .replay(&cmds, w as &mut dyn xenon_draw::GuestMemory)
            .unwrap();
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        let t = &r.timings;
        (t.hashed_bytes, t.guest_bytes > 0, px(&img, W, 3, 3)[0])
    };
    assert_eq!(replay(&mut d, &mut w), (len, true, 200));
    // Nothing written: no guest bytes read, vertices stay staged.
    assert_eq!(replay(&mut d, &mut w), (0, false, 200));
    // Writes elsewhere change nothing.
    let far = u64::from(TEXTURE) + 0x10_0000;
    w.written = Some(vec![(far, far + 4096)]);
    assert_eq!(replay(&mut d, &mut w), (0, false, 200));
    // A write to the vertices stages them again.
    w.written = Some(vec![(u64::from(VERTS), u64::from(VERTS) + 4)]);
    assert_eq!(replay(&mut d, &mut w), (0, true, 200));
    // A write to the texture: read and drawn again.
    put_texture(&mut w.mem, 90);
    w.written = Some(vec![(
        u64::from(TEXTURE) + len - 1,
        u64::from(TEXTURE) + len + 4096,
    )]);
    assert_eq!(replay(&mut d, &mut w), (len, false, 90));
    // Writes not listed: everything is read again.
    w.written = None;
    assert_eq!(replay(&mut d, &mut w), (len, true, 90));
}

/// Records `s` from `mem` without replaying it.
fn record(mem: &mut xenon_gpu::SliceMemory, s: &Stream) -> Vec<xenon_gpu::Command> {
    mem.write_words(CMD, &s.0);
    let mut rec = xenon_gpu::Recorder::new();
    rec.record_buffer(mem, CMD, s.0.len() as u32).unwrap();
    rec.commands
}

/// Replays `cmds` over `w`: the report and the colour target.
fn replay_watched(
    d: &mut xenon_draw::Drawer,
    cmds: &[xenon_gpu::Command],
    w: &mut Watched,
) -> (xenon_draw::Report, Vec<u8>) {
    let r = d
        .replay(cmds, w as &mut dyn xenon_draw::GuestMemory)
        .unwrap();
    (r, d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap())
}

/// A listed write between a texture's base and its mip page, far apart,
/// keeps the texture: nothing is read or hashed again. A write in the mip
/// level still decodes it again.
#[test]
fn write_between_levels_keeps_texture() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let (red, green, blue) = ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]);
    let mip = TEXTURE + 0x8000;
    solid_level(&mut mem, TEXTURE, 32, red);
    solid_level(&mut mem, mip, 16, green);
    let s = mip_stream(&mut mem, TEXTURE, 0, mip);
    let cmds = record(&mut mem, &s);
    let mut w = Watched {
        mem,
        written: Some(vec![]),
    };
    let (r, img) = replay_watched(&mut d, &cmds, &mut w);
    assert_eq!(r.timings.textures_uploaded, 1, "{r:?}");
    assert_eq!((px(&img, W, 5, 5), px(&img, W, 44, 4)), (red, green));

    let gap = u64::from(TEXTURE) + 0x3000;
    w.written = Some(vec![(gap, gap + 0x1000)]);
    let (r, img) = replay_watched(&mut d, &cmds, &mut w);
    let t = &r.timings;
    assert_eq!(
        (t.hashed_bytes, t.textures_uploaded, t.gap_writes),
        (0, 0, 1),
        "{r:?}"
    );
    assert_eq!((px(&img, W, 5, 5), px(&img, W, 44, 4)), (red, green));

    solid_level(&mut w.mem, mip, 16, blue);
    w.written = Some(vec![(u64::from(mip), u64::from(mip) + 0x1000)]);
    let (r, img) = replay_watched(&mut d, &cmds, &mut w);
    assert_eq!(r.timings.textures_uploaded, 1, "{r:?}");
    assert_eq!((px(&img, W, 5, 5), px(&img, W, 44, 4)), (red, blue));
}

/// A listed write over one page of a four-page texture that leaves its
/// bytes as they were hashes only that page and draws the texture it had,
/// which is what a fresh decode draws; a write that changes the page
/// decodes the texture again, as a fresh drawer does.
#[test]
fn unchanged_written_page_is_checked_alone() {
    let Some(mut d) = drawer() else { return };
    let n = 64u32;
    let mut mem = memory();
    let linear: Vec<u8> = (0..n * n)
        .flat_map(|i| [(i % 251) as u8, (i / 64) as u8, 7, 255])
        .collect();
    let mut tiled = tile::tile_2d(&linear, n, n, n, 4).unwrap();
    endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
    let len = tiled.len() as u64;
    assert_eq!(len, 0x4000);
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
    let fetch = [
        2 | (2 << 10) | (2 << 13) | ((n >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | TEXTURE,
        (n - 1) | ((n - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
        0,
        1 << 9,
    ];
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .vertex_fetch(0, VERTS, 20)
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .draw_auto(PRIM_QUAD_LIST, 4);
    let cmds = record(&mut mem, &s);
    let mut w = Watched {
        mem,
        written: Some(vec![]),
    };
    let (r, fresh) = replay_watched(&mut d, &cmds, &mut w);
    assert_eq!(
        (r.timings.hashed_bytes, r.timings.textures_uploaded),
        (len, 1)
    );

    // Page 1 written with the bytes it held.
    let page = u64::from(TEXTURE) + 0x1000;
    w.written = Some(vec![(page + 8, page + 12)]);
    let (r, img) = replay_watched(&mut d, &cmds, &mut w);
    let t = &r.timings;
    assert_eq!(
        (t.hashed_bytes, t.textures_uploaded, t.textures_restored),
        (0x1000, 0, 1),
        "{r:?}"
    );
    assert!(img == fresh, "restored texture draws as the fresh decode");

    // Page 1 changed: hashed alone, then the whole texture, and decoded.
    w.mem.data[page as usize..][..0x1000].fill(0x80);
    w.written = Some(vec![(page, page + 0x1000)]);
    let (r, img) = replay_watched(&mut d, &cmds, &mut w);
    let t = &r.timings;
    assert_eq!(
        (t.hashed_bytes, t.textures_uploaded, t.textures_restored),
        (0x1000 + len, 1, 0),
        "{r:?}"
    );
    assert!(img != fresh);
    drop(d);
    let Some(mut d) = drawer() else { return };
    let mut mem = w.mem;
    let _ = run(&mut d, &s, &mut mem);
    let again = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert!(
        img == again,
        "changed texture draws as a fresh drawer's decode"
    );
}

/// With guest memory listing its writes, a draw's index buffer is read once
/// and its host indices drawn again in later replays, until a listed write
/// over it: vertices 0..3 are red and 3..6 green, and the indices move from
/// the first three to the last.
#[test]
fn listed_writes_keep_indices_across_replays() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let (wf, hf) = (W as f32, H as f32);
    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    let corners = |c| [(0.0, 0.0, 0.0, c), (wf, 0.0, 0.0, c), (0.0, hf, 0.0, c)];
    let verts = [corners(red), corners(green)].concat();
    mem.write_words(VERTS, &color_verts(&verts));
    // Big-endian 16-bit indices (k8in16 swaps them back).
    let put = |mem: &mut xenon_gpu::SliceMemory, i: u32| {
        mem.write_words(INDICES, &[(i << 16) | (i + 1), (i + 2) << 16]);
    };
    put(&mut mem, 0);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 24)
        .draw_dma16(PRIM_RECT_LIST, 3);
    mem.write_words(CMD, &s.0);
    let mut rec = xenon_gpu::Recorder::new();
    rec.record_buffer(&mem, CMD, s.0.len() as u32).unwrap();
    let cmds = rec.commands;
    let mut w = Watched {
        mem,
        written: Some(vec![]),
    };
    let mut replay = |w: &mut Watched| {
        let r = d
            .replay(&cmds, w as &mut dyn xenon_draw::GuestMemory)
            .unwrap();
        assert_eq!(r.draws, 1, "{r:?}");
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        px(&img, W, 10, 10)
    };
    assert_eq!(replay(&mut w), red);
    // Written but not listed: the kept indices are drawn.
    put(&mut w.mem, 3);
    assert_eq!(replay(&mut w), red);
    // Listed: read again.
    let at = u64::from(INDICES);
    w.written = Some(vec![(at, at + 8)]);
    assert_eq!(replay(&mut w), green);
    // Writes unknown: read again.
    put(&mut w.mem, 0);
    w.written = None;
    assert_eq!(replay(&mut w), red);
}

/// Drawing at twice the guest resolution keeps flat edges where they were:
/// with Direct3D pixel centres a rectangle whose edges sit at half guest
/// pixels covers whole guest pixels, so at scale 2 the target and its
/// resolve to guest memory decimate to the same pixels as at scale 1.
#[test]
fn half_pixel_edges_decimate_to_the_same_pixels_at_scale_2() {
    let Some(mut d) = drawer() else { return };
    let mut out = Vec::new();
    for scale in [1, 2] {
        if d.set_scale(scale) != scale {
            eprintln!("SKIP: the device cannot draw at scale {scale}");
            return;
        }
        let mut mem = memory();
        mem.write_words(
            VERTS,
            &color_verts(&[
                (10.5, 20.5, 0.0, RED),
                (40.5, 20.5, 0.0, RED),
                (10.5, 50.5, 0.0, RED),
            ]),
        );
        let mut s = Stream::default();
        s.setup()
            .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
            .shader(false, &vs_color())
            .shader(true, &ps_interp())
            .vertex_fetch(0, VERTS, 12)
            // Direct3D pixel centres for the draw only.
            .regs(PA_SU_VTX_CNTL, &[0])
            .draw_auto(PRIM_RECT_LIST, 3)
            .regs(PA_SU_VTX_CNTL, &[1])
            .resolve(&mut mem, W, H, true, None);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!((report.draws, report.resolves), (1, 2), "{report:?}");
        assert!(report.unsupported.is_empty(), "{report:?}");
        assert_eq!(d.render_target_size(COLOR_KEY).map(|s| s.0), Some(W));
        let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
        out.push((img, detiled(&mem, DEST, W, H, W, 4)));
    }
    let (img, resolved) = &out[0];
    for y in 0..H {
        for x in 0..W {
            let inside = (11..41).contains(&x) && (21..51).contains(&y);
            let want = if inside { RED } else { CLEAR };
            assert_eq!(px(img, W, x, y), want, "({x}, {y})");
        }
    }
    assert!(*img == out[1].0, "scale 2 target differs from scale 1");
    assert!(
        *resolved == out[1].1,
        "scale 2 resolve differs from scale 1"
    );
    assert_eq!(*resolved, *img);
}

/// Each guest pixel of `img` (`w` x `h` guest pixels at `s` host pixels
/// each way) as the mean of its host pixels, halves up, as a resolve
/// decimates 8-bit channels.
fn box_mean(img: &[u8], w: u32, h: u32, s: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let mut sum = 0;
                for j in 0..s {
                    for i in 0..s {
                        sum += u32::from(px(img, w * s, x * s + i, y * s + j)[c]);
                    }
                }
                out.push(((sum + s * s / 2) / (s * s)) as u8);
            }
        }
    }
    out
}

/// With sharp front buffers on, guest memory after a resolve and a swap
/// is the same at scales 1, 2 and 3 (half-pixel edges decimate to the same
/// pixels), and from the second swap on the frame is the scale times the
/// front buffer on each axis.
#[test]
fn sharp_front_buffer_leaves_guest_memory_as_at_scale_1() {
    let Some(mut d) = drawer() else { return };
    d.sharp_front = true;
    let mut out = Vec::new();
    for scale in [1, 2, 3] {
        if d.set_scale(scale) != scale {
            eprintln!("SKIP: the device cannot draw at scale {scale}");
            break;
        }
        let mut mem = memory();
        mem.write_words(
            VERTS,
            &color_verts(&[
                (10.5, 20.5, 0.0, RED),
                (40.5, 20.5, 0.0, RED),
                (10.5, 50.5, 0.0, RED),
            ]),
        );
        let mut s = Stream::default();
        s.setup()
            .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
            .shader(false, &vs_color())
            .shader(true, &ps_interp())
            .vertex_fetch(0, VERTS, 12)
            .regs(PA_SU_VTX_CNTL, &[0])
            .draw_auto(PRIM_RECT_LIST, 3)
            .regs(PA_SU_VTX_CNTL, &[1])
            .resolve(&mut mem, W, H, true, None)
            .swap(W, H);
        let first = run(&mut d, &s, &mut mem);
        assert!(first.unsupported.is_empty(), "{first:?}");
        // Late from scale 2 on: the first swap gives no frame.
        assert_eq!(first.frames.len(), usize::from(scale == 1), "{first:?}");
        let r = run(&mut d, &s, &mut mem);
        assert_eq!(r.frames.len(), 1, "{r:?}");
        let f = &r.frames[0];
        assert_eq!((f.width, f.height), (W * scale, H * scale));
        assert_eq!(
            box_mean(&f.rgba8, W, H, scale),
            detiled(&mem, DEST, W, H, W, 4)
        );
        out.push(mem.data);
    }
    for (k, m) in out.iter().enumerate().skip(1) {
        assert!(*m == out[0], "guest memory at scale {} differs", k + 1);
    }
}

/// At scale 2 a 1280x720 front buffer is shown as 2560x1440 drawn pixels:
/// along a diagonal edge they are not doubled guest pixels, and their mean
/// is the resolve in guest memory. A CPU write to the front buffer drops
/// the scaled copy, so the next swap shows guest memory at 1280x720.
#[test]
fn sharp_front_buffer_at_scale_2_has_the_drawn_detail() {
    const FW: u32 = 1280;
    const FH: u32 = 720;
    let Some(mut d) = drawer() else { return };
    if d.set_scale(2) != 2 {
        eprintln!("SKIP: the device cannot draw at scale 2");
        return;
    }
    d.sharp_front = true;
    let mut mem = xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; 0x50_0000],
    };
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, RED),
            (FW as f32, 0.0, 0.0, RED),
            (0.0, FH as f32, 0.0, RED),
        ]),
    );
    let dest = (DEST, FW | (FH << 16), dest_info(6, 0, false));
    let mut s = Stream::default();
    s.setup()
        // Colour from tile 0 (720 tiles), depth past it.
        .regs(RB_SURFACE_INFO, &[FW, COLOR_BASE, 1024])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, FW | (FH << 16)])
        .resolve_with(
            &mut mem,
            (FW, FH),
            COPY_NULL | CLEAR_COLOR,
            dest,
            [0, rgba(CLEAR), 0],
        )
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_TRI_LIST, 3)
        .resolve_with(&mut mem, (FW, FH), 0, dest, [0; 3])
        .swap_at(DEST, FW, FH);
    assert!(run(&mut d, &s, &mut mem).frames.is_empty());
    let r = run(&mut d, &s, &mut mem);
    assert!(r.unsupported.is_empty(), "{r:?}");
    assert_eq!(r.frames.len(), 1, "{r:?}");
    let f = &r.frames[0];
    assert_eq!((f.width, f.height), (2 * FW, 2 * FH));
    let guest = detiled(&mem, DEST, FW, FH, FW, 4);
    assert_eq!(box_mean(&f.rgba8, FW, FH, 2), guest);
    // Guest pixels on the edge are a mix; some of their host pixels are
    // all red and some all clear.
    let mixed = (0..FH).find_map(|y| {
        (0..FW)
            .find(|&x| {
                let g = px(&guest, FW, x, y);
                g != RED && g != CLEAR
            })
            .map(|x| (x, y))
    });
    let (x, y) = mixed.expect("no guest pixel mixes red and clear");
    let host: Vec<[u8; 4]> = [(0, 0), (1, 0), (0, 1), (1, 1)]
        .map(|(i, j)| px(&f.rgba8, 2 * FW, 2 * x + i, 2 * y + j))
        .to_vec();
    assert!(host.contains(&RED) && host.contains(&CLEAR), "{host:?}");
    assert!(
        d.memory_line().contains("frames 2 scaled,"),
        "{}",
        d.memory_line()
    );

    // The CPU writes the front buffer: the swap reads guest memory now.
    mem.data[DEST as usize] ^= 0xFF;
    let mut swap = Stream::default();
    swap.swap_at(DEST, FW, FH);
    let r = run(&mut d, &swap, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    let f = &r.frames[0];
    assert_eq!((f.width, f.height), (FW, FH));
    assert_eq!(f.rgba8, detiled(&mem, DEST, FW, FH, FW, 4));
    let m = d.memory_line();
    assert!(m.contains("1 written by the CPU"), "{m}");
}

/// A kept resolve written back at the end of a replay, then written by the
/// CPU, merges into the kept resolve of a resolve next to it: the merged
/// buffer is read from guest memory as the CPU left it, so the old scaled
/// copy does not go over and the front buffer is shown at guest size from
/// guest memory.
#[test]
fn sharp_front_buffer_merge_drops_a_copy_the_cpu_wrote_over() {
    let Some(mut d) = drawer() else { return };
    if d.set_scale(2) != 2 {
        eprintln!("SKIP: the device cannot draw at scale 2");
        return;
    }
    d.sharp_front = true;
    let mut mem = memory();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (10.5, 20.5, 0.0, RED),
            (40.5, 20.5, 0.0, RED),
            (10.5, 50.5, 0.0, RED),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(PA_SU_VTX_CNTL, &[0])
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(PA_SU_VTX_CNTL, &[1])
        .resolve(&mut mem, W, H, true, None)
        .swap(W, H);
    run(&mut d, &s, &mut mem);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    assert_eq!((r.frames[0].width, r.frames[0].height), (2 * W, 2 * H));

    // The CPU writes the front buffer, then a resolve just past it merges
    // its kept resolve.
    mem.data[DEST as usize] ^= 0xFF;
    let next = DEST + xenon_texture::tile::tiled_extent_2d(W, H, W, 4).unwrap();
    let mut s = Stream::default();
    s.setup()
        .resolve_with(
            &mut mem,
            (W, H),
            0,
            (next, W | (H << 16), dest_info(6, 0, false)),
            [0; 3],
        )
        .swap(W, H);
    run(&mut d, &s, &mut mem);
    let m = d.memory_line();
    assert!(
        m.contains("frames 2 scaled,") && m.contains("1 no scaled copy"),
        "{m}"
    );
    // The late frame of that swap.
    let mut swap = Stream::default();
    swap.swap(W, H);
    let r = run(&mut d, &swap, &mut mem);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    let f = &r.frames[0];
    assert_eq!((f.width, f.height), (W, H));
    assert_eq!(f.rgba8, detiled(&mem, DEST, W, H, W, 4));
}

/// The other Direct3D convention: corners on guest pixel centres (integer
/// window coordinates), where the scale 1 tie rule takes the whole first
/// column and row and leaves the last. At scale 2 the same guest pixels are
/// drawn whole, not half: a rectangle inside the target, one over all of
/// it, and one in a 1x1 window scissor (the last pass of a luminance chain).
#[test]
fn centre_edges_decimate_to_the_same_pixels_at_scale_2() {
    let Some(mut d) = drawer() else { return };
    let cases = [
        ((10, 20), (40, 50), (W, H)),
        ((0, 0), (W, H), (W, H)),
        ((0, 0), (1, 1), (1, 1)),
    ];
    for ((x0, y0), (x1, y1), (sx, sy)) in cases {
        let mut out = Vec::new();
        for scale in [1, 2] {
            if d.set_scale(scale) != scale {
                eprintln!("SKIP: the device cannot draw at scale {scale}");
                return;
            }
            let mut mem = memory();
            let (fx0, fy0, fx1, fy1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
            mem.write_words(
                VERTS,
                &color_verts(&[
                    (fx0, fy0, 0.0, RED),
                    (fx1, fy0, 0.0, RED),
                    (fx0, fy1, 0.0, RED),
                ]),
            );
            let mut s = Stream::default();
            s.setup()
                .resolve(&mut mem, W, H, false, Some((rgba(CLEAR), 0)))
                .shader(false, &vs_color())
                .shader(true, &ps_interp())
                .vertex_fetch(0, VERTS, 12)
                .regs(PA_SU_VTX_CNTL, &[0])
                .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, sx | (sy << 16)])
                .draw_auto(PRIM_RECT_LIST, 3)
                .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, W | (H << 16)])
                .regs(PA_SU_VTX_CNTL, &[1])
                .resolve(&mut mem, W, H, true, None);
            let report = run(&mut d, &s, &mut mem);
            assert_eq!((report.draws, report.resolves), (1, 2), "{report:?}");
            let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
            out.push((img, detiled(&mem, DEST, W, H, W, 4)));
        }
        let (img, resolved) = &out[0];
        for y in 0..H {
            for x in 0..W {
                let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
                let want = if inside { RED } else { CLEAR };
                assert_eq!(px(img, W, x, y), want, "({x}, {y}) of {x0},{y0}-{x1},{y1}");
            }
        }
        assert!(
            *img == out[1].0,
            "scale 2 target differs, {x0},{y0}-{x1},{y1}"
        );
        assert!(
            *resolved == out[1].1,
            "scale 2 resolve differs, {x0},{y0}-{x1},{y1}"
        );
    }
}
