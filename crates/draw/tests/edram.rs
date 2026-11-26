//! EDRAM ownership transfers between k_8_8_8_8 colour and depth targets
//! on the same EDRAM tiles (one base and pitch, or another pitch and MSAA
//! sample count with rows as wide), on synthetic streams: what one writes
//! the other reads as the same 32-bit EDRAM word. They need a wgpu
//! adapter (lavapipe works); without one they print a skip line, unless
//! XENON_REQUIRE_GPU=1 makes them fail.

mod common;

use common::*;
use xenon_draw::{RenderTargetKey, TargetFormat};
use xenon_gpu::regs::*;
use xenon_gpu::SliceMemory;

const PRIM_TRI_LIST: u32 = 4;
const PRIM_RECT_LIST: u32 = 8;
const RT_32_FLOAT: u32 = 14;
/// RB_DEPTH_INFO for a depth target at tile 0, where the colour target is.
const D24S8_AT_0: u32 = 0;
const D24FS8_AT_0: u32 = 1 << 16;
/// RB_DEPTHCONTROL: z test LESS, no z write.
const DEPTH_LESS: u32 = 2 | (1 << 4);
/// A second resolve destination after DEST.
const DEST2: u32 = DEST + 0x2_0000;
const DEST3: u32 = DEST + 0x4_0000;

/// The 8_8_8_8 colour whose EDRAM word is `word` (red in the low byte).
fn word_color(word: u32) -> [f32; 4] {
    word.to_le_bytes().map(|b| f32::from(b) / 255.0)
}

/// Clear-only resolve of the depth target to `word` (depth << 8 | stencil).
fn clear_depth(s: &mut Stream, mem: &mut SliceMemory, word: u32) {
    s.resolve_with(
        mem,
        (W, H),
        COPY_NULL | CLEAR_DEPTH,
        (DEST, W | (H << 16), 0),
        [word, 0, 0],
    );
}

/// Raw resolve to `dest` of the colour target (`depth` false) or the depth
/// target: the EDRAM words either way.
fn resolve_words(s: &mut Stream, mem: &mut SliceMemory, depth: bool, dest: u32) {
    s.resolve_with(
        mem,
        (W, H),
        if depth { 4 } else { 0 } | COPY_RAW,
        (dest, W | (H << 16), dest_info(6, 0, false)),
        [0; 3],
    );
}

/// The resolved word at (x, y) of `dest`.
fn word_at(mem: &SliceMemory, dest: u32, x: u32, y: u32) -> u32 {
    u32::from_le_bytes(px(&detiled(mem, dest, W, H, W, 4), W, x, y))
}

/// Draws the rectangle (0, 0)-(`x1`, H) into the colour target with pixel
/// constant c0 = the colour of `word`, depth off.
fn draw_word(s: &mut Stream, mem: &mut SliceMemory, x1: f32, word: u32) {
    let c = [0; 4];
    mem.write_words(
        VERTS,
        &color_verts(&[(0.0, 0.0, 0.0, c), (x1, 0.0, 0.0, c), (0.0, 64.0, 0.0, c)]),
    );
    s.shader(false, &vs_color())
        .shader(true, &ps_const())
        .regs(SHADER_CONSTANT_FLOAT + 256 * 4, &word_color(word).map(f))
        .regs(RB_DEPTHCONTROL, &[0])
        .vertex_fetch(0, VERTS, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
}

/// The restore of depth through colour: the depth target at tile 0 is
/// cleared to `stale`, the colour target at tile 0 gets a full-screen
/// rectangle of `word` as colour, then, with colour moved to tile 100,
/// a red triangle at `z_near` (top left) and a green one at `z_far`
/// (bottom right) are depth tested LESS against the depth target, and the
/// depth target is resolved. Returns the colour at (4, 4) and (60, 60) and
/// the resolved depth word at (10, 10).
fn restore_depth(
    depth_info: u32,
    stale: u32,
    word: u32,
    (z_near, z_far): (f32, f32),
) -> Option<([u8; 4], [u8; 4], u32, bool)> {
    let mut d = drawer()?;
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[depth_info]);
    clear_depth(&mut s, &mut mem, stale);
    draw_word(&mut s, &mut mem, 64.0, word);
    mem.write_words(
        VERTS + 48,
        &color_verts(&[
            (0.0, 0.0, z_near, RED),
            (32.0, 0.0, z_near, RED),
            (0.0, 32.0, z_near, RED),
            (64.0, 64.0, z_far, GREEN),
            (32.0, 64.0, z_far, GREEN),
            (64.0, 32.0, z_far, GREEN),
        ]),
    );
    s.regs(RB_COLOR_INFO, &[100])
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[DEPTH_LESS])
        .vertex_fetch(0, VERTS + 48, 24)
        .draw_auto(PRIM_TRI_LIST, 6)
        .regs(RB_DEPTHCONTROL, &[0]);
    resolve_words(&mut s, &mut mem, true, DEST2);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 2), "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    // The transfer to depth (its depth and stencil-bit draws) is drawn in
    // the colour-and-depth draw's pass, not in passes of its own.
    assert!(report.timings.passes_merged >= 1, "{:?}", report.timings);
    let key = RenderTargetKey {
        base_tiles: 100,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let img = d.read_render_target(key, 0, 0, W, H).unwrap();
    Some((
        px(&img, W, 4, 4),
        px(&img, W, 60, 60),
        word_at(&mem, DEST2, 10, 10),
        d.float_stencil(),
    ))
}

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// kD24S8 restored through k_8_8_8_8 at its EDRAM base: word 0x40000078 is
/// depth 0x400000 / 0xFFFFFF (just over 0.25) and stencil 0x78, so a
/// triangle at 0.2499 passes LESS, one at 0.2501 fails, and the depth
/// resolve gives the word back exactly.
#[test]
fn colour_restores_d24s8_depth() {
    let Some((near, far, resolved, _)) =
        restore_depth(D24S8_AT_0, 0xFFFF_FF00, 0x4000_0078, (0.2499, 0.2501))
    else {
        return;
    };
    assert_eq!(near, RED);
    assert_eq!(far, [0; 4]);
    assert_eq!(resolved, 0x4000_0078);
}

/// kD24FS8 restored through k_8_8_8_8: word 0xD3333378 is 20e4 0xD33333
/// (about 0.3) and stencil 0x78 (0 where kD24FS8 has no host stencil).
#[test]
fn colour_restores_d24fs8_depth() {
    let Some((near, far, resolved, float_stencil)) =
        restore_depth(D24FS8_AT_0, 0xF000_0000, 0xD333_3378, (0.2999, 0.3001))
    else {
        return;
    };
    assert_eq!(near, RED);
    assert_eq!(far, [0; 4]);
    let stencil = if float_stencil { 0x78 } else { 0 };
    assert_eq!(resolved, 0xD333_3300 | stencil);
}

/// Colour to kD24S8 and back through the depth resolve is exact for any
/// 24-bit depth, odd ones from 2^23 up included (n / 0xFFFFFF must be the
/// nearest f32, as the depth clear makes it), and for every stencil bit.
#[test]
fn colour_to_d24s8_words_exact() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    // shortcut: depths from 0.75 up come back one or two steps high on
    // lavapipe (its 24-bit depth store and load), for clears too; extend
    // the list once the depth copy reads integer depth.
    for word in [
        0x0000_01FF,
        0x0012_3401,
        0x8000_01A5,
        0xABCD_EF5A,
        0xBFFF_FF3C,
    ] {
        let mut s = Stream::default();
        s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
        draw_word(&mut s, &mut mem, 64.0, word);
        resolve_words(&mut s, &mut mem, true, DEST2);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!(report.resolves, 1, "{report:?}");
        assert_eq!(word_at(&mem, DEST2, 7, 7), word, "{word:08X}");
    }
}

/// A depth clear at the colour target's EDRAM base reaches the colour
/// target: its resolve reads the depth word as 8_8_8_8 bytes.
fn depth_to_colour(depth_info: u32, word: u32) -> Option<(u32, bool)> {
    let mut d = drawer()?;
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[depth_info]);
    clear_depth(&mut s, &mut mem, word);
    resolve_words(&mut s, &mut mem, false, DEST2);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 2, "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    Some((word_at(&mem, DEST2, 5, 5), d.float_stencil()))
}

#[test]
fn d24s8_clear_reaches_colour() {
    let Some((w, _)) = depth_to_colour(D24S8_AT_0, 0x1234_5678) else {
        return;
    };
    assert_eq!(w, 0x1234_5678);
}

#[test]
fn d24fs8_clear_reaches_colour() {
    let Some((w, float_stencil)) = depth_to_colour(D24FS8_AT_0, 0xD333_3378) else {
        return;
    };
    assert_eq!(w, 0xD333_3300 | if float_stencil { 0x78 } else { 0 });
}

/// Only a newer target is transferred: depth cleared to 0x12345678, then
/// colour drawn on the left half (after taking the depth word everywhere).
/// The colour resolve must not take the older depth again (left keeps the
/// drawn word), and the depth resolve then takes the newer colour, in the
/// other half of each tile.
#[test]
fn older_target_is_not_transferred() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    clear_depth(&mut s, &mut mem, 0x1234_5678);
    draw_word(&mut s, &mut mem, 32.0, 0x4000_0078);
    resolve_words(&mut s, &mut mem, false, DEST);
    resolve_words(&mut s, &mut mem, false, DEST2);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 3), "{report:?}");
    for dest in [DEST, DEST2] {
        assert_eq!(word_at(&mem, dest, 5, 5), 0x4000_0078);
        assert_eq!(word_at(&mem, dest, 50, 5), 0x1234_5678);
    }
    // The recorder starts from zeroed registers: set the targets again.
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    resolve_words(&mut s, &mut mem, true, DEST3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 1, "{report:?}");
    assert_eq!(word_at(&mem, DEST3, 50, 5), 0x4000_0078);
    assert_eq!(word_at(&mem, DEST3, 5, 5), 0x1234_5678);
}

/// Aliasing that is not transferred is reported: a depth target at tile 2
/// inside the colour target's tiles (another base), and a k_32_FLOAT
/// colour target at the depth target's base.
#[test]
fn untransferred_aliasing_is_reported() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup();
    draw_word(&mut s, &mut mem, 64.0, 0x4000_0078);
    s.regs(RB_DEPTH_INFO, &[2]);
    clear_depth(&mut s, &mut mem, 0);
    let report = run(&mut d, &s, &mut mem);
    assert!(
        report.warnings.iter().any(|w| w.contains("overlaps")),
        "{report:?}"
    );
    let mut s = Stream::default();
    s.setup().regs(RB_COLOR_INFO, &[RT_32_FLOAT << 16]);
    draw_word(&mut s, &mut mem, 64.0, 0);
    s.regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    clear_depth(&mut s, &mut mem, 0);
    let report = run(&mut d, &s, &mut mem);
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.contains("D24S8 render target aliases a newer Float32")),
        "{report:?}"
    );
    // The snorm16 fixed-point and the 16/32-bit float formats are not
    // transferred either (snorm16 only where the host renders it).
    for (format, name) in [
        (5, "Rgba16Fixed"),
        (4, "Rg16Fixed"),
        (6, "Rg16Float"),
        (15, "Rg32Float"),
    ] {
        let mut s = Stream::default();
        s.setup().regs(RB_COLOR_INFO, &[(format << 16) | 200]);
        draw_word(&mut s, &mut mem, 64.0, 0);
        s.regs(RB_DEPTH_INFO, &[200]);
        clear_depth(&mut s, &mut mem, 0);
        let report = run(&mut d, &s, &mut mem);
        if name.ends_with("Fixed") && !d.snorm16() {
            assert!(!report.unsupported.is_empty(), "{report:?}");
            continue;
        }
        let want = format!("D24S8 render target aliases a newer {name}");
        assert!(
            report.warnings.iter().any(|w| w.contains(&want)),
            "{report:?}"
        );
    }
}

/// A transfer makes both targets current: reading the depth target's
/// words through the colour target does not make the colour target newer,
/// so the depth target is not transferred back. Back through kD24FS8's
/// 20e4 word, a depth of 0.3 written by a draw would become a little less
/// and fail a coplanar EQUAL test after the colour resolve.
#[test]
fn transfer_does_not_bounce_back() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.3, RED),
            (64.0, 0.0, 0.3, RED),
            (0.0, 64.0, 0.3, RED),
        ]),
    );
    // Depth only (colour mask 0): z ALWAYS with z write.
    s.setup()
        .regs(RB_DEPTH_INFO, &[D24FS8_AT_0])
        .regs(RB_COLOR_MASK, &[0])
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .vertex_fetch(0, VERTS, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_COLOR_MASK, &[0xF]);
    resolve_words(&mut s, &mut mem, false, DEST2);
    // The same rectangle into colour at tile 100, z EQUAL.
    s.regs(RB_COLOR_INFO, &[100])
        .regs(RB_DEPTHCONTROL, &[2 | (2 << 4)])
        .vertex_fetch(0, VERTS, 24)
        .draw_auto(PRIM_RECT_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let key = RenderTargetKey {
        base_tiles: 100,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let img = d.read_render_target(key, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), RED);
}

/// Batched colour draws come before the transfer they feed: three
/// rectangles of narrowing width (depth words 0x2.., 0x6.. and 0xA..,
/// just over 0.125, 0.375 and 0.625) are recorded into one render pass on
/// the colour target at tile 0, then a depth-tested draw makes the depth
/// target at tile 0 current. The transfer must see all three: it is
/// recorded in the batch after them, before the depth-tested draws, all
/// in one submit. Depth column x takes
/// colour column x + 40 (or x - 40), the other half of the tile.
#[test]
fn batched_colour_draws_reach_the_transfer() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .regs(RB_DEPTH_INFO, &[D24S8_AT_0])
        .regs(RB_DEPTHCONTROL, &[0])
        .shader(false, &vs_color())
        .shader(true, &ps_const());
    let c = [0; 4];
    for (i, (x1, word)) in [
        (64.0, 0x2000_0000),
        (32.0, 0x6000_0000),
        (16.0, 0xA000_0000),
    ]
    .into_iter()
    .enumerate()
    {
        let at = VERTS + 48 * i as u32;
        mem.write_words(
            at,
            &color_verts(&[(0.0, 0.0, 0.0, c), (x1, 0.0, 0.0, c), (0.0, 64.0, 0.0, c)]),
        );
        s.regs(SHADER_CONSTANT_FLOAT + 256 * 4, &word_color(word).map(f))
            .vertex_fetch(0, at, 12)
            .draw_auto(PRIM_RECT_LIST, 3);
    }
    // Into colour at tile 100, z LESS without z write: red at 0.25 passes
    // over the last two rectangles, green at 0.5 over the last one only.
    s.regs(RB_COLOR_INFO, &[100])
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[DEPTH_LESS]);
    for (i, (z, colour)) in [(0.25, RED), (0.5, GREEN)].into_iter().enumerate() {
        let at = VERTS + 48 * (3 + i as u32);
        mem.write_words(
            at,
            &color_verts(&[
                (0.0, 0.0, z, colour),
                (64.0, 0.0, z, colour),
                (0.0, 64.0, z, colour),
            ]),
        );
        s.vertex_fetch(0, at, 24).draw_auto(PRIM_RECT_LIST, 3);
    }
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 5, "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    assert_eq!(report.timings.submits, 1, "{report:?}");
    let key = RenderTargetKey {
        base_tiles: 100,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let img = d.read_render_target(key, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 48, 30), GREEN);
    assert_eq!(px(&img, W, 60, 30), RED);
    assert_eq!(px(&img, W, 8, 30), [0; 4]);
}

/// One tile of a tiled (predicated) frame: the tile's rows of a 2x MSAA
/// surface at pitch W are drawn through the window offset into EDRAM rows
/// 0..32, as a G-buffer target at tile 100 over a kD24FS8 depth target at
/// tile 0. The tile starts with a clear drawn as 4x MSAA at half the pitch
/// (the same EDRAM tiles, a quarter of the pixels): blue colour, depth 0.
/// The G-buffer draw then covers x < 48 of the tile's rows, z 0.25 GEQUAL,
/// and the tile is resolved to `dest` with a colour and depth clear.
fn tile(s: &mut Stream, mem: &mut SliceMemory, t: u32, colour: [u8; 4], dest: u32) {
    const HALF: u32 = H / 2;
    let mut rect = |at: u32, x1: f32, z: f32, c: [u8; 4]| {
        let h = H as f32;
        mem.write_words(
            at,
            &color_verts(&[(0.0, 0.0, z, c), (x1, 0.0, z, c), (0.0, h, z, c)]),
        );
    };
    // Clear of draw_word's vertices at VERTS.
    let at = VERTS + 0x1000 + 0x100 * t;
    rect(at, (W / 2) as f32, 0.0, BLUE);
    rect(at + 48, 48.0, 0.25, colour);
    s.regs(RB_SURFACE_INFO, &[(W / 2) | (2 << 16)])
        .regs(RB_COLOR_INFO, &[100])
        .regs(RB_DEPTH_INFO, &[D24FS8_AT_0])
        .regs(PA_SU_SC_MODE_CNTL, &[0])
        .regs(PA_SC_WINDOW_OFFSET, &[0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, W | (H << 16)])
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .vertex_fetch(0, at, 24)
        .draw_auto(PRIM_RECT_LIST, 3);
    // The tile: window offset -32 rows per tile, scissor in window rows.
    let off = (HALF * t).wrapping_neg() & 0x7FFF;
    s.regs(RB_SURFACE_INFO, &[W | (1 << 16)])
        .regs(PA_SU_SC_MODE_CNTL, &[1 << 16])
        .regs(PA_SC_WINDOW_OFFSET, &[off << 16])
        .regs(
            PA_SC_WINDOW_SCISSOR_TL,
            &[(HALF * t) << 16, W | ((HALF * (t + 1)) << 16)],
        )
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (6 << 4)])
        .vertex_fetch(0, at + 48, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(PA_SU_SC_MODE_CNTL, &[0])
        .regs(PA_SC_WINDOW_OFFSET, &[0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, W | (H << 16)]);
    s.resolve_with(
        mem,
        (W, HALF),
        COPY_RAW | CLEAR_COLOR | CLEAR_DEPTH,
        (dest, W | (HALF << 16), dest_info(6, 0, false)),
        [0; 3],
    );
}

/// The frame after the tiles: single-sampled at pitch W, depth is
/// restored by drawing its words as k_8_8_8_8 colour at the depth base
/// (20e4 0.5 everywhere), then a depth-tested draw (colour at tile 300)
/// uses the restored depth.
fn after_tiles(s: &mut Stream, mem: &mut SliceMemory) {
    s.regs(RB_SURFACE_INFO, &[W])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, W | (H << 16)])
        .regs(RB_COLOR_INFO, &[0]);
    draw_word(s, mem, 64.0, 0xE000_0000);
    mem.write_words(
        VERTS + 0x400,
        &color_verts(&[
            (0.0, 0.0, 0.25, RED),
            (64.0, 0.0, 0.25, RED),
            (0.0, 64.0, 0.25, RED),
        ]),
    );
    s.regs(RB_COLOR_INFO, &[300])
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[DEPTH_LESS])
        .vertex_fetch(0, VERTS + 0x400, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0]);
}

const BLUE: [u8; 4] = [0, 0, 255, 255];

/// Two frames of two tiles each. The second frame's first tile starts
/// with the depth the first frame restored (0.5, which the G-buffer draw
/// at 0.25 GEQUAL fails); only its 4x half-pitch clear makes it 0. That
/// clear, at another pitch and sample count but the same EDRAM tiles, is
/// the newest data there, so the 2x target must take it, colour (blue
/// where the G-buffer draw does not reach) and depth alike. The second
/// tile's depth is also cleared by the first tile's resolve, so it worked
/// either way.
#[test]
fn tiled_frame_takes_the_half_pitch_msaa_clear() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup();
    for frame in 0..2 {
        for (t, colour) in [(0, RED), (1, GREEN)] {
            let dest = DEST + 0x2_0000 * (2 * frame + t);
            tile(&mut s, &mut mem, t, colour, dest);
        }
        after_tiles(&mut s, &mut mem);
    }
    let report = run(&mut d, &s, &mut mem);
    assert!(report.unsupported.is_empty(), "{report:?}");
    assert_eq!((report.draws, report.resolves), (12, 4), "{report:?}");
    assert!(
        !report.warnings.iter().any(|w| w.contains("aliases")),
        "{report:?}"
    );
    for frame in 0..2 {
        for (t, colour) in [(0, RED), (1, GREEN)] {
            let dest = DEST + 0x2_0000 * (2 * frame + t);
            let got = |x, y| word_at(&mem, dest, x, y).to_le_bytes();
            assert_eq!(got(10, 5), colour, "frame {frame} tile {t}");
            assert_eq!(got(10, 30), colour, "frame {frame} tile {t}");
        }
    }
    // Where the G-buffer draw does not reach, the tile keeps the clear.
    for dest in (0..4).map(|i| DEST + 0x2_0000 * i) {
        assert_eq!(word_at(&mem, dest, 56, 20).to_le_bytes(), BLUE);
    }
    // The second frame's restored depth (0.5) passes the draw at 0.25
    // LESS; the tiles' resolve clear (0) would fail it.
    let key = RenderTargetKey {
        base_tiles: 300,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let img = d.read_render_target(key, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 10, 10), RED);
}

/// kD24S8 and kD24FS8 at one EDRAM base and pitch share the 32-bit word:
/// a kD24S8 clear read as kD24FS8 is the same word (24-bit depth bits
/// taken as 20e4), and back.
#[test]
fn d24s8_and_d24fs8_reinterpret_the_word() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    clear_depth(&mut s, &mut mem, 0x4000_0078);
    s.regs(RB_DEPTH_INFO, &[D24FS8_AT_0]);
    resolve_words(&mut s, &mut mem, true, DEST2);
    clear_depth(&mut s, &mut mem, 0x2345_6700);
    s.regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    resolve_words(&mut s, &mut mem, true, DEST3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 4, "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let stencil = if d.float_stencil() { 0x78 } else { 0 };
    assert_eq!(word_at(&mem, DEST2, 5, 5), 0x4000_0000 | stencil);
    assert_eq!(word_at(&mem, DEST3, 5, 5), 0x2345_6700);
}

/// The depth restore of a tiled frame, as drawn after the tiles: the
/// k_8_8_8_8 words are drawn at the depth base, a 4x MSAA draw at half the
/// pitch z tests against them without writing (its target takes a
/// resampled copy, as new as the words), then the depth target is used
/// single-sampled. It must take the words, not the copy, which holds one
/// pixel of every two columns: colour column 12 is word B, 13 word A, so
/// depth columns 52 and 53 (the other half of the tile) are B and A.
#[test]
fn depth_restore_takes_the_words_not_a_resampled_copy() {
    const A: u32 = 0xD333_3300;
    const B: u32 = 0x4000_0000;
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24FS8_AT_0]);
    clear_depth(&mut s, &mut mem, 0);
    draw_word(&mut s, &mut mem, 64.0, A);
    let at = VERTS + 0x400;
    let c = [0; 4];
    mem.write_words(
        at,
        &color_verts(&[(0.0, 0.0, 0.0, c), (13.0, 0.0, 0.0, c), (0.0, 64.0, 0.0, c)]),
    );
    s.regs(SHADER_CONSTANT_FLOAT + 256 * 4, &word_color(B).map(f))
        .vertex_fetch(0, at, 12)
        .draw_auto(PRIM_RECT_LIST, 3);
    // z ALWAYS without z write, colour at tile 200.
    s.regs(RB_SURFACE_INFO, &[(W / 2) | (2 << 16)])
        .regs(RB_COLOR_INFO, &[200])
        .regs(RB_DEPTHCONTROL, &[2 | (7 << 4)])
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_SURFACE_INFO, &[W]);
    resolve_words(&mut s, &mut mem, true, DEST2);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (3, 2), "{report:?}");
    assert!(
        !report.warnings.iter().any(|w| w.contains("aliases")),
        "{report:?}"
    );
    for y in [4, 33] {
        assert_eq!(word_at(&mem, DEST2, 52, y), B, "row {y}");
        assert_eq!(word_at(&mem, DEST2, 53, y), A, "row {y}");
    }
}

/// A depth restore drawn as colour at the depth target's base is dropped
/// once a depth draw has written all its rows (the depth target holds its
/// words), and the next frame, which restores into a new colour target,
/// resolves the same depth: the restored word on the right, the drawn
/// depth on the left.
#[test]
fn restore_target_is_dropped_after_the_depth_draw() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let s = restore_then_draw_depth(&mut mem);
    let colour = RenderTargetKey {
        base_tiles: 0,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let mut frames = Vec::new();
    for frame in 0..2 {
        mem.data[DEST2 as usize..][..(W * H * 4) as usize].fill(0);
        let report = run(&mut d, &s, &mut mem);
        assert_eq!((report.draws, report.resolves), (2, 1), "{report:?}");
        assert!(report.warnings.is_empty(), "{report:?}");
        let m = d.memory_line();
        let drops = format!("targets 1 MB in 1, {} dropped,", frame + 1);
        assert!(m.contains(&drops), "{m}");
        // The next frame's colour target takes the dropped one's texture.
        let reuse = format!(" 2 made, {frame} reused, 1 spare (0 MB),");
        assert!(m.contains(&reuse), "{m}");
        assert!(d.read_render_target(colour, 0, 0, 1, 1).is_err());
        assert_eq!(word_at(&mem, DEST2, 50, 10), 0x4000_0078);
        assert_ne!(word_at(&mem, DEST2, 10, 10), 0x4000_0078);
        frames.push(detiled(&mem, DEST2, W, H, W, 4));
    }
    assert!(frames[0] == frames[1]);
}

/// A target dropped at the end of every replay and recreated in the next
/// leaves nothing behind in wgpu: its allocations and live textures, views,
/// bind groups and buffers stay flat replay after replay.
#[test]
fn dropped_and_recreated_targets_do_not_leak() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let s = restore_then_draw_depth(&mut mem);
    // Two warm-up replays: the upload belt keeps a second chunk while the
    // first is still in flight, then reuses the pair.
    for _ in 0..2 {
        run(&mut d, &s, &mut mem);
    }
    let live = d.gpu_live();
    for frame in 2..30 {
        run(&mut d, &s, &mut mem);
        assert_eq!(d.gpu_live(), live, "replay {frame}");
    }
    let m = d.memory_line();
    assert!(m.contains("targets 1 MB in 1, 30 dropped,"), "{m}");
}

/// A colour restore of the depth words at tile 0, a depth-only draw over
/// its left half and a raw depth resolve to DEST2: the colour target is
/// dropped at the end of the replay and recreated by the next.
fn restore_then_draw_depth(mem: &mut SliceMemory) -> Stream {
    let c = [0; 4];
    mem.write_words(
        VERTS + 48,
        &color_verts(&[(0.0, 0.0, 0.3, c), (32.0, 0.0, 0.3, c), (0.0, 64.0, 0.3, c)]),
    );
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    draw_word(&mut s, mem, 64.0, 0x4000_0078);
    // Depth only (colour mask 0): z ALWAYS with z write.
    s.regs(RB_COLOR_MASK, &[0])
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .vertex_fetch(0, VERTS + 48, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_COLOR_MASK, &[0xF]);
    resolve_words(&mut s, mem, true, DEST2);
    s
}

/// Two tiles across: a pitch of 160 pixels at 1x.
const WIDE: u32 = 160;

/// Whether column `x` of one target holds what column `x` of the other
/// target on its tiles had below `x1`, colour against depth: the 40-sample
/// halves of each 80-sample tile are swapped.
fn swapped_below(x: u32, x1: u32) -> bool {
    x - x % 80 + (x % 80 + 40) % 80 < x1
}

/// The words of row `y` of a `WIDE`-pitch raw resolve at `dest`.
fn wide_row(mem: &SliceMemory, dest: u32, y: u32) -> Vec<u32> {
    let img = detiled(mem, dest, WIDE, H, WIDE, 4);
    (0..WIDE)
        .map(|x| u32::from_le_bytes(px(&img, WIDE, x, y)))
        .collect()
}

fn wide_resolve(s: &mut Stream, mem: &mut SliceMemory, control: u32, word: u32) {
    s.resolve_with(
        mem,
        (WIDE, H),
        control,
        (DEST2, WIDE | (H << 16), dest_info(6, 0, false)),
        [word, 0, 0],
    );
}

/// Colour columns 0..100 restored as depth land in the other half of each
/// tile: depth column x takes colour column x +- 40 within its tile, so
/// columns 40..80 of depth come from colour 0..40 and 120..140 from 80..100.
#[test]
fn colour_to_depth_swaps_tile_halves() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[WIDE, 0, D24S8_AT_0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, WIDE | (H << 16)]);
    draw_word(&mut s, &mut mem, 100.0, 0x4000_0078);
    wide_resolve(&mut s, &mut mem, 4 | COPY_RAW, 0);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 1), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let row = wide_row(&mem, DEST2, 9);
    for (x, &w) in (0..).zip(&row) {
        let want = if swapped_below(x, 100) {
            0x4000_0078
        } else {
            0
        };
        assert_eq!(w, want, "depth column {x}: {row:08X?}");
    }
}

/// The other way: a depth clear of columns 0..96 (resolve rectangles are
/// 8-pixel aligned) read through colour.
#[test]
fn depth_to_colour_swaps_tile_halves() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[WIDE, 0, D24S8_AT_0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, WIDE | (H << 16)]);
    s.resolve_with(
        &mut mem,
        (96, H),
        COPY_NULL | CLEAR_DEPTH,
        (DEST, 96 | (H << 16), 0),
        [0x1234_5678, 0, 0],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 1, "{report:?}");
    // Its own replay: the resolve rectangles are read from RECT then.
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[WIDE, 0, D24S8_AT_0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, WIDE | (H << 16)]);
    wide_resolve(&mut s, &mut mem, COPY_RAW, 0);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 1, "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let row = wide_row(&mem, DEST2, 9);
    for (x, &w) in (0..).zip(&row) {
        let want = if swapped_below(x, 96) { 0x1234_5678 } else { 0 };
        assert_eq!(w, want, "colour column {x}: {row:08X?}");
    }
}

/// A pitch of 120 at 1x is a tile and a half: depth columns 80..120 would
/// read colour 120..160, past the colour row, and keep their own depth and
/// stencil, while 0..80 take the swapped colour.
#[test]
fn swapped_samples_past_the_row_keep_the_destination() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let pitch = 120;
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[pitch, 0, D24S8_AT_0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, pitch | (H << 16)]);
    s.resolve_with(
        &mut mem,
        (pitch, H),
        COPY_NULL | CLEAR_DEPTH,
        (DEST, pitch | (H << 16), 0),
        [0x2000_0033, 0, 0],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.resolves, 1, "{report:?}");
    let mut s = Stream::default();
    s.setup()
        .regs(RB_SURFACE_INFO, &[pitch, 0, D24S8_AT_0])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, pitch | (H << 16)]);
    draw_word(&mut s, &mut mem, pitch as f32, 0x4000_0078);
    s.resolve_with(
        &mut mem,
        (pitch, H),
        4 | COPY_RAW,
        (DEST2, pitch | (H << 16), dest_info(6, 0, false)),
        [0; 3],
    );
    let report = run(&mut d, &s, &mut mem);
    assert_eq!((report.draws, report.resolves), (1, 1), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
    let img = detiled(&mem, DEST2, pitch, H, pitch, 4);
    let row: Vec<u32> = (0..pitch)
        .map(|x| u32::from_le_bytes(px(&img, pitch, x, 9)))
        .collect();
    for (x, &w) in (0..).zip(&row) {
        let want = if x < 80 { 0x4000_0078 } else { 0x2000_0033 };
        assert_eq!(w, want, "depth column {x}: {row:08X?}");
    }
}

/// A colour draw over the depth target's tiles, then a draw that only
/// tests depth: each replay transfers the colour words to depth, neither
/// target is dropped, and the transfer's setup is made once and kept.
#[test]
fn repeated_transfers_reuse_their_setup() {
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let c = [0; 4];
    mem.write_words(
        VERTS + 48,
        &color_verts(&[(0.0, 0.0, 0.3, c), (64.0, 0.0, 0.3, c), (0.0, 64.0, 0.3, c)]),
    );
    let mut s = Stream::default();
    s.setup().regs(RB_DEPTH_INFO, &[D24S8_AT_0]);
    draw_word(&mut s, &mut mem, 64.0, 0x4000_0078);
    // Colour at tile 100, z LESS without z write against tile 0.
    s.regs(RB_COLOR_INFO, &[100])
        .shader(true, &ps_interp())
        .regs(RB_DEPTHCONTROL, &[2 | (1 << 4)])
        .vertex_fetch(0, VERTS + 48, 24)
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_COLOR_INFO, &[0]);
    let mut counts = Vec::new();
    for _ in 0..3 {
        let r = run(&mut d, &s, &mut mem);
        assert!(r.warnings.is_empty(), "{r:?}");
        counts.push((r.timings.transfers, r.timings.transfer_setups));
    }
    assert_eq!(counts, [(1, 1), (1, 0), (1, 0)]);
}
