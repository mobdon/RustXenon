//! EDRAM ownership transfers between a k_8_8_8_8 colour target and a depth
//! target at the same EDRAM base and pitch, on synthetic streams: what one
//! writes the other reads as the same 32-bit EDRAM word. They need a wgpu
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
/// drawn word), and the depth resolve then takes the newer colour.
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
    assert_eq!(word_at(&mem, DEST3, 5, 5), 0x4000_0078);
    assert_eq!(word_at(&mem, DEST3, 50, 5), 0x1234_5678);
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
/// target at tile 0 current. The transfer must see all three, so the batch
/// is submitted first: one submit for the colour draws, one for the
/// transfer and one for the depth-tested draws.
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
    assert_eq!(report.timings.submits, 3, "{report:?}");
    let key = RenderTargetKey {
        base_tiles: 100,
        pitch: W,
        format: TargetFormat::Rgba8,
    };
    let img = d.read_render_target(key, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 8, 30), GREEN);
    assert_eq!(px(&img, W, 24, 30), RED);
    assert_eq!(px(&img, W, 48, 30), [0; 4]);
}
