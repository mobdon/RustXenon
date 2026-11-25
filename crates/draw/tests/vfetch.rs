//! Vertex fetch formats end to end: known bit patterns in a vertex buffer,
//! fetched by a vertex shader built with the shader assembler, passed to a
//! k_32_32_FLOAT render target and read back as floats. The expected values
//! follow the Xenos vertex fetch rules (as Xenia models them): packed
//! components extracted from bit 0 up, unsigned normalized as x / (2^n - 1),
//! signed normalized as max(x / (2^(n-1) - 1), -1) or, without a zero code,
//! (2x + 1) / (2^n - 1), integers converted as is, exp_adjust scaling by
//! 2^e, and components the format lacks reading 0.

mod common;

use common::*;
use xenon_draw::{RenderTargetKey, TargetFormat};
use xenon_gpu::regs::*;
use xenon_shader::asm::*;
use xenon_shader::ucode::*;

const RT_32_32_FLOAT: u32 = 15;
const PRIM_QUAD_LIST: u32 = 13;
/// Where the fetched vertex data lives (fetch constant 1).
const VALS: u32 = 0x2_8000;
/// Word at offset 0 of each fetched vertex, which offset 1 skips.
const GARBAGE: u32 = 0xDEAD_BEEF;

const E_NONE: u32 = 0;
const E_8IN16: u32 = 1;
const E_8IN32: u32 = 2;
const E_16IN32: u32 = 3;

fn swap(e: u32, w: u32) -> u32 {
    match e {
        E_8IN16 => ((w & 0x00FF_00FF) << 8) | ((w >> 8) & 0x00FF_00FF),
        E_8IN32 => w.swap_bytes(),
        E_16IN32 => w.rotate_left(16),
        _ => w,
    }
}

/// A full fetch of fetch constant 1 into r2 from r0.x, with the vertex
/// laid out as one garbage word then `words` (stride `words + 1`, offset 1).
fn vf(format: VertexFormat, signed: bool, normalized: bool, words: u32) -> VertexFetch {
    let mut v = VertexFetch::full(2, 0, 1, format, words + 1, 1);
    v.signed = signed;
    v.normalized = normalized;
    v
}

/// Draws a quad over the whole k_32_32_FLOAT target with a vertex shader
/// that fetches the position through fetch constant 0, runs `fetches` and
/// exports r2 to interpolator 0; returns the texel at (8, 8). Every vertex
/// at VALS holds `raw` (guest memory bytes).
fn draw_xy(
    d: &mut xenon_draw::Drawer,
    fetches: &[VertexFetch],
    raw: &[u8],
    endian: u32,
) -> [f32; 2] {
    draw_buf(d, fetches, &raw.repeat(4), endian, [0.0; 4])
}

/// `draw_xy` with `vals` (guest memory bytes) at VALS as is, and r3 set to
/// `r3` before `fetches` run, for fetches that take their index from r3
/// instead of the vertex index in r0.
fn draw_buf(
    d: &mut xenon_draw::Drawer,
    fetches: &[VertexFetch],
    vals: &[u8],
    endian: u32,
    r3: [f32; 4],
) -> [f32; 2] {
    let mut mem = memory();
    let q = W as f32;
    let verts: Vec<u32> = [(0.0, 0.0), (q, 0.0), (q, q), (0.0, q)]
        .iter()
        .flat_map(|&(x, y)| [f(x), f(y), f(0.0)])
        .collect();
    mem.write_words(VERTS, &verts);
    mem.data[VALS as usize..][..vals.len()].copy_from_slice(vals);
    let mut pos = VertexFetch::full(1, 0, 0, VertexFormat::F32_32_32_FLOAT, 3, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let c0 = AluSrc::constant(0, SWZ_XYZW);
    let mut instrs = vec![
        Instr::Fetch(FetchInstr::Vertex(pos)),
        Instr::Alu(AluInstr::vector(VectorOp::Max, 3, mask("xyzw"), &[c0, c0])),
    ];
    instrs.extend(fetches.iter().map(|&v| Instr::Fetch(FetchInstr::Vertex(v))));
    let r = |n| AluSrc::temp(n, SWZ_XYZW);
    instrs.push(export(62, r(1)));
    instrs.push(export(0, r(2)));
    let vs = assemble(&[exec(0, instrs.len() as u32, true)], &instrs);
    let mut s = Stream::default();
    s.setup()
        .regs(RB_COLOR_INFO, &[RT_32_32_FLOAT << 16])
        .shader(false, &vs)
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(
            SHADER_CONSTANT_FETCH + 2,
            &[VALS | 3, ((vals.len() as u32) << 2) | endian],
        )
        .regs(SHADER_CONSTANT_FLOAT, &r3.map(f))
        .draw_auto(PRIM_QUAD_LIST, 4);
    let report = run(d, &s, &mut mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let key = RenderTargetKey {
        base_tiles: COLOR_BASE,
        pitch: W,
        format: TargetFormat::Rg32Float,
    };
    let img = d.read_render_target_raw(key, 0, 0, W, H).unwrap();
    let i = ((8 * W + 8) * 8) as usize;
    let c = |o: usize| f32::from_le_bytes(img[i + o..i + o + 4].try_into().unwrap());
    [c(0), c(4)]
}

/// All four components of `v` (its destination swizzle replaced), fetched
/// from vertices holding a garbage word then `words` (dword values after
/// the fetch constant's `endian` swap).
fn fetch4(d: &mut xenon_draw::Drawer, v: VertexFetch, words: &[u32], endian: u32) -> [f32; 4] {
    let raw: Vec<u8> = std::iter::once(GARBAGE)
        .chain(words.iter().copied())
        .flat_map(|w| swap(endian, w).to_le_bytes())
        .collect();
    let mut out = [0.0; 4];
    for (half, swz) in ["xyzw", "zw01"].iter().enumerate() {
        let mut v = v;
        v.dest_swizzle = fetch_dest_swz(swz);
        let xy = draw_xy(d, &[v], &raw, endian);
        out[2 * half..][..2].copy_from_slice(&xy);
    }
    out
}

#[track_caller]
fn check(got: [f32; 4], want: [f32; 4]) {
    let ok = got.iter().zip(want).all(|(&g, w)| {
        (g - w).abs() <= w.abs() * 2e-7
            && (w == 0.0 || g.is_sign_negative() == w.is_sign_negative())
    });
    assert!(ok, "got {got:?}, want {want:?}");
}

/// One format, all four signed/normalized combinations (signed normalized
/// in both repeating-fraction modes) of one bit pattern.
#[track_caller]
#[allow(clippy::too_many_arguments)]
fn check_modes(
    d: &mut xenon_draw::Drawer,
    format: VertexFormat,
    words: &[u32],
    unorm: [f32; 4],
    uint: [f32; 4],
    snorm: [f32; 4],
    snorm_no_zero: [f32; 4],
    sint: [f32; 4],
) {
    let n = words.len() as u32;
    let e = E_8IN32;
    check(fetch4(d, vf(format, false, true, n), words, e), unorm);
    check(fetch4(d, vf(format, false, false, n), words, e), uint);
    check(fetch4(d, vf(format, true, true, n), words, e), snorm);
    let mut v = vf(format, true, true, n);
    v.signed_rf_no_zero = true;
    check(fetch4(d, v, words, e), snorm_no_zero);
    check(fetch4(d, vf(format, true, false, n), words, e), sint);
}

#[test]
fn vfetch_8_8_8_8() {
    let Some(mut d) = drawer() else { return };
    // x 0xFF, y 0x80, z 0x81, w 0x7F.
    let w = 0x7F81_80FF;
    check_modes(
        &mut d,
        VertexFormat::F8_8_8_8,
        &[w],
        [1.0, 128.0 / 255.0, 129.0 / 255.0, 127.0 / 255.0],
        [255.0, 128.0, 129.0, 127.0],
        [-1.0 / 127.0, -1.0, -1.0, 1.0],
        [-1.0 / 255.0, -1.0, -253.0 / 255.0, 1.0],
        [-1.0, -128.0, -127.0, 127.0],
    );
    // exp_adjust scales by a power of two after the conversion.
    let mut v = vf(VertexFormat::F8_8_8_8, false, false, 1);
    v.exp_adjust = -1;
    check(fetch4(&mut d, v, &[w], E_8IN32), [127.5, 64.0, 64.5, 63.5]);
    v.exp_adjust = 3;
    check(
        fetch4(&mut d, v, &[w], E_8IN32),
        [2040.0, 1024.0, 1032.0, 1016.0],
    );
}

#[test]
fn vfetch_2_10_10_10() {
    let Some(mut d) = drawer() else { return };
    // x 0x200, y 0x1FF, z 0x001, w 0b10.
    let w = 0x200 | (0x1FF << 10) | (1 << 20) | (2 << 30);
    check_modes(
        &mut d,
        VertexFormat::F2_10_10_10,
        &[w],
        [512.0 / 1023.0, 511.0 / 1023.0, 1.0 / 1023.0, 2.0 / 3.0],
        [512.0, 511.0, 1.0, 2.0],
        [-1.0, 1.0, 1.0 / 511.0, -1.0],
        [-1.0, 1.0, 3.0 / 1023.0, -1.0],
        [-512.0, 511.0, 1.0, -2.0],
    );
}

#[test]
fn vfetch_10_11_11() {
    let Some(mut d) = drawer() else { return };
    // x 11 bits at 0 (0x400), y 11 bits at 11 (0x3FF), z 10 bits at 22
    // (0x201); w is missing and reads 0.
    let w = 0x400 | (0x3FF << 11) | (0x201 << 22);
    check_modes(
        &mut d,
        VertexFormat::F10_11_11,
        &[w],
        [1024.0 / 2047.0, 1023.0 / 2047.0, 513.0 / 1023.0, 0.0],
        [1024.0, 1023.0, 513.0, 0.0],
        [-1.0, 1.0, -1.0, 0.0],
        [-1.0, 1.0, -1021.0 / 1023.0, 0.0],
        [-1024.0, 1023.0, -511.0, 0.0],
    );
}

#[test]
fn vfetch_11_11_10() {
    let Some(mut d) = drawer() else { return };
    // x 10 bits at 0 (0x1FF), y 11 bits at 10 (0x400), z 11 bits at 21
    // (0x001).
    let w = 0x1FF | (0x400 << 10) | (1 << 21);
    check_modes(
        &mut d,
        VertexFormat::F11_11_10,
        &[w],
        [511.0 / 1023.0, 1024.0 / 2047.0, 1.0 / 2047.0, 0.0],
        [511.0, 1024.0, 1.0, 0.0],
        [1.0, -1.0, 1.0 / 1023.0, 0.0],
        [1.0, -1.0, 3.0 / 2047.0, 0.0],
        [511.0, -1024.0, 1.0, 0.0],
    );
}

#[test]
fn vfetch_16_16_and_16_16_16_16() {
    let Some(mut d) = drawer() else { return };
    // x in the low half.
    check_modes(
        &mut d,
        VertexFormat::F16_16,
        &[0x7FFF_8000],
        [32768.0 / 65535.0, 32767.0 / 65535.0, 0.0, 0.0],
        [32768.0, 32767.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-32768.0, 32767.0, 0.0, 0.0],
    );
    // x 0xFFFF, y 0x0001 | z 0x8001, w 0x4000.
    check_modes(
        &mut d,
        VertexFormat::F16_16_16_16,
        &[0x0001_FFFF, 0x4000_8001],
        [1.0, 1.0 / 65535.0, 32769.0 / 65535.0, 16384.0 / 65535.0],
        [65535.0, 1.0, 32769.0, 16384.0],
        [-1.0 / 32767.0, 1.0 / 32767.0, -1.0, 16384.0 / 32767.0],
        [
            -1.0 / 65535.0,
            3.0 / 65535.0,
            -65533.0 / 65535.0,
            32769.0 / 65535.0,
        ],
        [-1.0, 1.0, -32767.0, 16384.0],
    );
}

#[test]
fn vfetch_16_float() {
    let Some(mut d) = drawer() else { return };
    let v = |format, n| vf(format, false, true, n);
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F16_16_FLOAT, 1),
            &[0xC000_3C00],
            E_8IN32,
        ),
        [1.0, -2.0, 0.0, 0.0],
    );
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F16_16_16_16_FLOAT, 2),
            &[0x7BFF_3800, 0x8000_0001],
            E_8IN32,
        ),
        // The sign of -0.0 does not survive interpolation, so it is not checked.
        [0.5, 65504.0, 2f32.powi(-24), -0.0],
    );
    // Sign and normalization do not apply to floats; exp_adjust does.
    let mut s = vf(VertexFormat::F16_16_FLOAT, true, false, 1);
    s.exp_adjust = 2;
    check(
        fetch4(&mut d, s, &[0xC000_3C00], E_8IN32),
        [4.0, -8.0, 0.0, 0.0],
    );
    // Console halves have no infinity or NaN: exponent 31 is a number (Xenia
    // decodes it as IEEE, a known divergence it notes).
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F16_16_FLOAT, 1),
            &[0xFFFF_7C00],
            E_8IN32,
        ),
        [65536.0, -131008.0, 0.0, 0.0],
    );
}

#[test]
fn vfetch_32_integer() {
    let Some(mut d) = drawer() else { return };
    let m = 2147483648.0;
    check_modes(
        &mut d,
        VertexFormat::F32,
        &[0xFFFF_FFFB],
        [1.0, 0.0, 0.0, 0.0],
        [4294967291.0, 0.0, 0.0, 0.0],
        [-5.0 / 2147483647.0, 0.0, 0.0, 0.0],
        [-4.5 / 2147483647.5, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0, 0.0],
    );
    check_modes(
        &mut d,
        VertexFormat::F32_32,
        &[0x8000_0000, 0x7FFF_FFFF],
        [0.5, 0.5, 0.0, 0.0],
        [m, m, 0.0, 0.0],
        // No -1 clamp for 32 bits: -2^31 / (2^31 - 1) is -1 in f32.
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [-m, m, 0.0, 0.0],
    );
    check_modes(
        &mut d,
        VertexFormat::F32_32_32_32,
        &[1, 0xFFFF_FFFF, 7, 0],
        [1.0 / 4294967295.0, 1.0, 7.0 / 4294967295.0, 0.0],
        [1.0, 4294967295.0, 7.0, 0.0],
        [
            1.0 / 2147483647.0,
            -1.0 / 2147483647.0,
            7.0 / 2147483647.0,
            0.0,
        ],
        [
            1.5 / 2147483647.5,
            -0.5 / 2147483647.5,
            7.5 / 2147483647.5,
            0.5 / 2147483647.5,
        ],
        [1.0, -1.0, 7.0, 0.0],
    );
}

#[test]
fn vfetch_32_float() {
    let Some(mut d) = drawer() else { return };
    let v = |format, n| vf(format, false, true, n);
    check(
        fetch4(&mut d, v(VertexFormat::F32_FLOAT, 1), &[f(1.5)], E_8IN32),
        [1.5, 0.0, 0.0, 0.0],
    );
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F32_32_FLOAT, 2),
            &[f(-0.25), f(1e30)],
            E_8IN32,
        ),
        [-0.25, 1e30, 0.0, 0.0],
    );
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F32_32_32_FLOAT, 3),
            &[f(1.0), f(-2.0), f(3.5)],
            E_8IN32,
        ),
        [1.0, -2.0, 3.5, 0.0],
    );
    let words = [f(-1.0), f(0.125), f(-1e-30), f(7.0)];
    check(
        fetch4(
            &mut d,
            v(VertexFormat::F32_32_32_32_FLOAT, 4),
            &words,
            E_8IN32,
        ),
        [-1.0, 0.125, -1e-30, 7.0],
    );
    let mut s = vf(VertexFormat::F32_32_32_32_FLOAT, true, false, 4);
    s.exp_adjust = -3;
    check(
        fetch4(&mut d, s, &words, E_8IN32),
        [-0.125, 0.015625, -1.25e-31, 0.875],
    );
}

/// The fetch constant's endian field, on guest bytes 11 22 33 44 55 66 77
/// 88: each mode swaps the dwords read before the components are taken
/// from bit 0 up.
#[test]
fn vfetch_endian_modes_on_16_bit_data() {
    let Some(mut d) = drawer() else { return };
    let mut raw = GARBAGE.to_le_bytes().to_vec();
    raw.extend([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    let cases = [
        (E_NONE, [0x2211, 0x4433, 0x6655, 0x8877]),
        (E_8IN16, [0x1122, 0x3344, 0x5566, 0x7788]),
        (E_8IN32, [0x3344, 0x1122, 0x7788, 0x5566]),
        (E_16IN32, [0x4433, 0x2211, 0x8877, 0x6655]),
    ];
    for (e, want) in cases {
        let mut v = vf(VertexFormat::F16_16_16_16, false, false, 2);
        let mut got = [0.0; 4];
        for (half, swz) in ["xyzw", "zw01"].iter().enumerate() {
            v.dest_swizzle = fetch_dest_swz(swz);
            got[2 * half..][..2].copy_from_slice(&draw_xy(&mut d, &[v], &raw, e));
        }
        check(got, want.map(|x| x as f32));
        // 16_16_FLOAT takes its halves the same way.
        let h = vf(VertexFormat::F16_16_FLOAT, false, true, 1);
        let mut raw = GARBAGE.to_le_bytes().to_vec();
        raw.extend(swap(e, 0xC000_3C00).to_le_bytes());
        check_xy(draw_xy(&mut d, &[h], &raw, e), [1.0, -2.0]);
    }
}

/// Guest bytes (8in32 swapped) of vertices laid out as one garbage word
/// then the vertex's words, one after another.
fn vertices(words: &[&[u32]]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|v| std::iter::once(&GARBAGE).chain(v.iter()))
        .flat_map(|w| w.to_be_bytes())
        .collect()
}

/// The full fetch address is the fetch constant's base plus the index
/// from the source register's selected component, rounded down or (index
/// rounding) to nearest, times the stride, plus the offset.
#[test]
fn vfetch_index_stride_and_rounding() {
    let Some(mut d) = drawer() else { return };
    // Stride 3 words: garbage, value 10 + i, padding 99.
    let verts: Vec<[u32; 2]> = (0..5).map(|i| [f(10.0 + i as f32), f(99.0)]).collect();
    let vals = vertices(&verts.iter().map(|v| &v[..]).collect::<Vec<_>>());
    let mut v = VertexFetch::full(2, 3, 1, VertexFormat::F32_FLOAT, 3, 1);
    v.src_swizzle = 1;
    v.dest_swizzle = fetch_dest_swz("x0__");
    for (index, rounded, want) in [
        (2.0, false, 12.0),
        (2.75, false, 12.0),
        (2.75, true, 13.0),
        (2.25, true, 12.0),
        (3.5, true, 14.0),
    ] {
        v.index_rounded = rounded;
        let got = draw_buf(&mut d, &[v], &vals, E_8IN32, [0.0, index, 0.0, 0.0]);
        check_xy(got, [want, 0.0]);
    }
}

/// vfetch_mini reuses the preceding full fetch's constant, index and
/// stride, ignoring its own, and applies its own format, offset and
/// conversion; constant 0 and 1 selects work in the destination swizzle.
#[test]
fn vfetch_mini_reuses_full_fetch_address() {
    let Some(mut d) = drawer() else { return };
    // The full fetch reads vertex 2 (index from r3.x), stride 4 words; the
    // other vertices hold zeros, so a mini fetch that used its own index
    // register (r0.x, the vertex index), stride or constant reads wrong data.
    let words = [f(1.5), 0x7FFF_8000, 0xC000_3C00];
    let zero = [0; 3];
    let vals = vertices(&[&zero, &zero, &words, &zero]);
    let r3 = [2.0, 0.0, 0.0, 0.0];
    // The full fetch writes nothing (every component kept).
    let mut full = vf(VertexFormat::F32_FLOAT, false, true, 3);
    full.src = 3;
    full.dest = 3;
    full.dest_swizzle = fetch_dest_swz("____");
    let mut mini = vf(VertexFormat::F16_16, true, true, 3);
    mini.mini = true;
    mini.src = 0;
    mini.fetch_const = 0;
    mini.stride = 0;
    mini.offset = 2;
    mini.dest_swizzle = fetch_dest_swz("yx01");
    let draw = |d: &mut _, mini| draw_buf(d, &[full, mini], &vals, E_8IN32, r3);
    check_xy(draw(&mut d, mini), [1.0, -1.0]);
    mini.dest_swizzle = fetch_dest_swz("01xy");
    check_xy(draw(&mut d, mini), [0.0, 1.0]);
    let mut half = mini;
    half.format = VertexFormat::F16_16_FLOAT;
    half.offset = 3;
    half.dest_swizzle = fetch_dest_swz("xyzw");
    check_xy(draw(&mut d, half), [1.0, -2.0]);
}

#[track_caller]
fn check_xy(got: [f32; 2], want: [f32; 2]) {
    check([got[0], got[1], 0.0, 0.0], [want[0], want[1], 0.0, 0.0]);
}
