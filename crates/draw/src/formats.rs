// The 7e3 conversions follow Xenia src/xenia/gpu/xenos.cc (Float7e3To32) and
// src/xenia/gpu/spirv_shader_translator_rb.cc (PreClampedFloat32To7e3,
// UnclampedFloat32To7e3). Host render target formats follow
// GetColorVulkanFormat in src/xenia/gpu/vulkan/vulkan_render_target_cache.cc
// and the D3D12 format table in d3d12/d3d12_render_target_cache.cc; colour
// clear values follow PerformTransfersAndResolveClears in
// vulkan/vulkan_render_target_cache.cc (the 64-bit clear value is
// RB_COLOR_CLEAR | RB_COLOR_CLEAR_LO << 32); the EDRAM packing of a 7e3
// host render target follows its render target dump shader (7e3 RGB, alpha
// saturated to 2 bits); the resolve conversion follows
// src/xenia/gpu/shaders/resolve.xesli (exponent bias factor, red/blue swap)
// and pixel_formats.xesli (XeUnpackR10G10B10A2Float, XePackR8G8B8A8UNorm,
// XePackR10G10B10A2UNorm, the 16_16_16_16_FLOAT packing); the output clamp
// of 7e3 targets follows GetPSIColorFormatInfo and AddPSIColorFormatFlags in
// src/xenia/gpu/render_target_cache.cc / .h; k_16_16 / k_16_16_16_16 follow
// the full-range snorm16 path of Xenia's host render targets (shader output
// divided by 32 in d3d12/d3d12_command_processor.cc, R16G16[B16A16]_SNORM in
// d3d12/d3d12_render_target_cache.cc, clears as raw 16-bit values in
// vulkan/vulkan_render_target_cache.cc) and XeUnpackR16G16Edram /
// XeUnpackR16G16B16A16Edram, XePackR16G16UNorm, XePackR16G16B16A16UNorm and
// the 16_16_FLOAT / 32_32_FLOAT packing in pixel_formats.xesli (BSD 3-Clause,
// see CREDITS.md).

//! Colour formats without the GPU: clear values, the EDRAM view of host
//! render target texels and the resolve packing into guest formats.

use crate::TargetFormat;

/// `xenos::ColorFormat` values a resolve can write.
pub const DEST_8_8_8_8: u32 = 6;
pub const DEST_2_10_10_10: u32 = 7;
pub const DEST_16_16: u32 = 25;
pub const DEST_16_16_16_16: u32 = 26;
pub const DEST_16_16_FLOAT: u32 = 31;
pub const DEST_16_16_16_16_FLOAT: u32 = 32;
pub const DEST_32_FLOAT: u32 = 36;
pub const DEST_32_32_FLOAT: u32 = 37;

/// k_16_16 and k_16_16_16_16 hold [-32, 32]; the snorm16 host target
/// holds the value / 32, as Xenia's full-range host render targets do.
pub const FIXED16_SCALE: f32 = 1.0 / 32.0;

/// A k_16_16 EDRAM value (signed 16 bits) as Xenia's resolve unpacks it:
/// x * 32 / 32767, floored at -1 (XeUnpackR16G16Edram).
pub fn fixed16_to_f32(bits: u16) -> f32 {
    (f32::from(bits as i16) * (32.0 / 32767.0)).max(-1.0)
}

/// Largest 7e3 value (exponent 7, mantissa 127).
pub const FLOAT7E3_MAX: f32 = 31.875;

/// Xenos 7e3 float in bits 0..10 to f32, not clamping (Xenia
/// Float7e3To32). The mantissa goes to f32 bits 16..23 as in Xenia's
/// shader version (XeUnpackR10G10B10A2Float); the C++ version in xenos.cc
/// shifts it by 3, which looks like a slip copied from the 20e4 code.
pub fn float7e3_to_f32(f10: u32) -> f32 {
    let f10 = f10 & 0x3FF;
    if f10 == 0 {
        return 0.0;
    }
    let mut mantissa = f10 & 0x7F;
    let mut exponent = f10 >> 7;
    if exponent == 0 {
        // Normalise the denormal: lzcnt of the mantissa within 8 bits.
        let lz = mantissa.leading_zeros() - (32 - 8);
        exponent = (1 - lz as i32) as u32;
        mantissa = (mantissa << lz) & 0x7F;
    }
    f32::from_bits((exponent.wrapping_add(124) << 23) | (mantissa << 16))
}

/// f32 to 7e3, clamped to [0, 31.875] (NaN and -0 to 0), rounded to the
/// nearest even (Xenia UnclampedFloat32To7e3).
pub fn f32_to_float7e3(f: f32) -> u32 {
    let f = if f > 0.0 { f.min(FLOAT7E3_MAX) } else { 0.0 };
    let u = f.to_bits();
    let biased = if u < 0x3E80_0000 {
        // Too small for a normal 7e3 number.
        ((u & 0x7F_FFFF) | 0x80_0000) >> (125 - (u >> 23)).min(24)
    } else {
        u - (124 << 23)
    };
    ((biased + 0x7FFF + ((biased >> 16) & 1)) >> 16) & 0x3FF
}

/// IEEE half to f32 (exact).
pub fn half_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let man = h & 0x3FF;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Denormal: man * 2^-24, exact in f32.
            let v = man as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, _) => sign | 0x7FC0_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half, round to nearest even, overflow to infinity.
pub fn f32_to_half(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xFF) as i32;
    let man = u & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    // Shift the 24-bit significand so the half's last kept bit is bit 0.
    let (sig, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        (man, 13)
    };
    let half_ulp = 1u32 << (shift - 1);
    let rest = sig & ((1 << shift) - 1);
    let mut h = sig >> shift;
    if rest > half_ulp || (rest == half_ulp && h & 1 != 0) {
        h += 1;
    }
    // For normals, a mantissa carry correctly bumps the exponent.
    let h = if e <= 0 { h } else { ((e as u32) << 10) + h };
    sign | h as u16
}

/// The 64-bit colour clear value of a resolve, as Xenia builds it.
pub fn clear_value(color_clear: u32, color_clear_lo: u32) -> u64 {
    u64::from(color_clear) | (u64::from(color_clear_lo) << 32)
}

/// The RGBA a colour clear writes into a host render target of `format`
/// (Xenia PerformTransfersAndResolveClears).
pub fn clear_color(format: TargetFormat, value: u64) -> [f32; 4] {
    let field = |shift: u32, bits: u32| ((value >> shift) as u32) & ((1 << bits) - 1);
    match format {
        TargetFormat::Rgba8 => [0, 8, 16, 24].map(|s| field(s, 8) as f32 * (1.0 / 255.0)),
        TargetFormat::Rgb10a2 => [
            field(0, 10) as f32 * (1.0 / 1023.0),
            field(10, 10) as f32 * (1.0 / 1023.0),
            field(20, 10) as f32 * (1.0 / 1023.0),
            field(30, 2) as f32 * (1.0 / 3.0),
        ],
        TargetFormat::Float7e3 => [
            float7e3_to_f32(field(0, 10)),
            float7e3_to_f32(field(10, 10)),
            float7e3_to_f32(field(20, 10)),
            field(30, 2) as f32 * (1.0 / 3.0),
        ],
        // Xenia clears these through a uint view with the half bits; the
        // half -> f32 -> half round trip of the clear shader is exact.
        TargetFormat::Rgba16Float => [0, 16, 32, 48].map(|s| half_to_f32(field(s, 16) as u16)),
        TargetFormat::Float32 => [f32::from_bits(value as u32), 0.0, 0.0, 0.0],
        // Xenia clears these with the raw 16-bit values: snorm x / 32767.
        // shortcut: 0x8000 clears as 0x8001 (snorm has two -1s), clear
        // through a uint view if a game clears to it.
        TargetFormat::Rg16Fixed | TargetFormat::Rgba16Fixed => {
            [0, 16, 32, 48].map(|s| (f32::from(field(s, 16) as u16 as i16) / 32767.0).max(-1.0))
        }
        TargetFormat::Rg16Float => [
            half_to_f32(field(0, 16) as u16),
            half_to_f32(field(16, 16) as u16),
            0.0,
            0.0,
        ],
        TargetFormat::Rg32Float => [
            f32::from_bits(value as u32),
            f32::from_bits((value >> 32) as u32),
            0.0,
            0.0,
        ],
        TargetFormat::D24S8 | TargetFormat::D24FS8 => [0.0; 4],
    }
}

/// Output clamp and scale of a pixel shader drawing into `format`:
/// `(rgb_high, alpha_high, scale)` with low bounds 0 (an infinite bound
/// does not clamp), or None. Xenia's EDRAM emulation clamps 7e3 outputs to
/// [0, 31.875] / [0, 1] when blending is off; with blending only the alpha
/// (fixed point in this format) is clamped, the colour is blended as float.
/// k_16_16 outputs are divided by 32 into their snorm16 host target, whose
/// conversion clamps them to [-32, 32] (and blends them / 32, which Xenia
/// accepts too).
pub fn output_clamp(format: TargetFormat, blending: bool) -> Option<(f32, f32, f32)> {
    match (format, blending) {
        (TargetFormat::Float7e3, false) => Some((FLOAT7E3_MAX, 1.0, 1.0)),
        (TargetFormat::Float7e3, true) => Some((f32::INFINITY, 1.0, 1.0)),
        (TargetFormat::Rg16Fixed | TargetFormat::Rgba16Fixed, _) => {
            Some((f32::INFINITY, f32::INFINITY, FIXED16_SCALE))
        }
        _ => None,
    }
}

/// A host texel of `format` as the EDRAM holds it, in RGBA f32. A 7e3
/// target is hosted as half floats; the EDRAM keeps 7e3 RGB and 2-bit
/// alpha, so the value is quantised like Xenia's dump of a host render
/// target into EDRAM.
pub fn edram_rgba(format: TargetFormat, texel: &[u8]) -> [f32; 4] {
    let word = |i: usize| u32::from_le_bytes([texel[i], texel[i + 1], texel[i + 2], texel[i + 3]]);
    let half = |i: usize| half_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
    match format {
        TargetFormat::Rgba8 => [0, 1, 2, 3].map(|i| f32::from(texel[i]) * (1.0 / 255.0)),
        TargetFormat::Rgb10a2 => {
            let p = word(0);
            [
                (p & 0x3FF) as f32 * (1.0 / 1023.0),
                ((p >> 10) & 0x3FF) as f32 * (1.0 / 1023.0),
                ((p >> 20) & 0x3FF) as f32 * (1.0 / 1023.0),
                (p >> 30) as f32 * (1.0 / 3.0),
            ]
        }
        TargetFormat::Float7e3 => {
            let q = |v: f32| float7e3_to_f32(f32_to_float7e3(v));
            let a = half(3);
            let a = if a > 0.0 { a.min(1.0) } else { 0.0 };
            let a2 = (a * 3.0 + 0.5) as u32;
            [q(half(0)), q(half(1)), q(half(2)), a2 as f32 * (1.0 / 3.0)]
        }
        TargetFormat::Rgba16Float => [half(0), half(1), half(2), half(3)],
        // XeResolveUnpack32bpp: red only.
        TargetFormat::Float32 => [f32::from_bits(word(0)), 0.0, 0.0, 0.0],
        // The snorm16 host bits are the EDRAM bits.
        TargetFormat::Rg16Fixed | TargetFormat::Rgba16Fixed => {
            let c = |i: usize| fixed16_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            if format == TargetFormat::Rg16Fixed {
                [c(0), c(1), 0.0, 0.0]
            } else {
                [c(0), c(1), c(2), c(3)]
            }
        }
        TargetFormat::Rg16Float => [half(0), half(1), 0.0, 0.0],
        TargetFormat::Rg32Float => [f32::from_bits(word(0)), f32::from_bits(word(4)), 0.0, 0.0],
        TargetFormat::D24S8 | TargetFormat::D24FS8 => [0.0; 4],
    }
}

/// Bytes per texel of a resolve destination format, or None when the
/// drawer cannot write it.
pub fn dest_bytes(dest_format: u32) -> Option<usize> {
    match dest_format {
        DEST_8_8_8_8 | DEST_2_10_10_10 | DEST_16_16 | DEST_16_16_FLOAT | DEST_32_FLOAT => Some(4),
        DEST_16_16_16_16 | DEST_16_16_16_16_FLOAT | DEST_32_32_FLOAT => Some(8),
        _ => None,
    }
}

/// Resolve conversion of one pixel (Xenia's resolve shader): the
/// `2^exp_bias` factor, the red/blue swap, then the destination packing,
/// as little-endian words before the destination endian swap.
pub fn resolve_pixel(rgba: [f32; 4], exp_bias: i32, swap: bool, dest_format: u32, out: &mut [u8]) {
    let k = 2f32.powi(exp_bias);
    let mut p = rgba.map(|c| c * k);
    if swap {
        p.swap(0, 2);
    }
    // saturate then +0.5 and truncate; NaN saturates to 0.
    let unorm = |c: f32, max: f32| {
        let c = if c > 0.0 { c.min(1.0) } else { 0.0 };
        (c * max + 0.5) as u32
    };
    match dest_format {
        DEST_8_8_8_8 => {
            let n = p.map(|c| unorm(c, 255.0));
            out.copy_from_slice(&(n[0] | (n[1] << 8) | (n[2] << 16) | (n[3] << 24)).to_le_bytes());
        }
        DEST_2_10_10_10 => {
            let w = unorm(p[0], 1023.0)
                | (unorm(p[1], 1023.0) << 10)
                | (unorm(p[2], 1023.0) << 20)
                | (unorm(p[3], 3.0) << 30);
            out.copy_from_slice(&w.to_le_bytes());
        }
        // XePack32bpp: the red channel's bits, not saturated.
        DEST_32_FLOAT => out.copy_from_slice(&p[0].to_bits().to_le_bytes()),
        DEST_16_16 | DEST_16_16_16_16 => {
            for (o, c) in out.chunks_exact_mut(2).zip(p) {
                o.copy_from_slice(&(unorm(c, 65535.0) as u16).to_le_bytes());
            }
        }
        DEST_16_16_FLOAT | DEST_16_16_16_16_FLOAT => {
            for (o, c) in out.chunks_exact_mut(2).zip(p) {
                o.copy_from_slice(&f32_to_half(c).to_le_bytes());
            }
        }
        DEST_32_32_FLOAT => {
            out[..4].copy_from_slice(&p[0].to_bits().to_le_bytes());
            out[4..].copy_from_slice(&p[1].to_bits().to_le_bytes());
        }
        _ => unreachable!("checked by dest_bytes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float7e3_known_values_and_round_trip() {
        assert_eq!(float7e3_to_f32(0), 0.0);
        assert_eq!(float7e3_to_f32(0x80), 0.25);
        assert_eq!(float7e3_to_f32(0x3FF), 31.875);
        assert_eq!(float7e3_to_f32(0x40), 0.125); // denormal
        assert_eq!(float7e3_to_f32(1), 0.25 / 128.0);
        for f10 in 0..0x400 {
            assert_eq!(f32_to_float7e3(float7e3_to_f32(f10)), f10, "{f10:#x}");
        }
        assert_eq!(f32_to_float7e3(-1.0), 0);
        assert_eq!(f32_to_float7e3(-0.0), 0);
        assert_eq!(f32_to_float7e3(f32::NAN), 0);
        assert_eq!(f32_to_float7e3(1000.0), 0x3FF);
        // Half way between 0x80 and 0x81 rounds to even, above it rounds up.
        let mid = 0.25 + 0.25 / 256.0;
        assert_eq!(f32_to_float7e3(mid), 0x80);
        assert_eq!(f32_to_float7e3(f32::from_bits(mid.to_bits() + 1)), 0x81);
    }

    #[test]
    fn half_round_trips_and_rounds_to_even() {
        for h in 0..=u16::MAX {
            let f = half_to_f32(h);
            if f.is_nan() {
                assert!(half_to_f32(f32_to_half(f)).is_nan());
            } else {
                assert_eq!(f32_to_half(f), h, "{h:#x}");
            }
        }
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f32_to_half(65520.0), 0x7C00);
        assert_eq!(f32_to_half(65519.0), 0x7BFF);
        // 1 + 2^-11 is half way between 1 and the next half: to even (1).
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_half(2f32.powi(-26)), 0);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-26)), 1);
    }

    #[test]
    fn clear_colors_per_format() {
        let v = clear_value(0x0403_0201, 0x0807_0605);
        assert_eq!(
            clear_color(TargetFormat::Rgba8, v),
            [1.0, 2.0, 3.0, 4.0].map(|x: f32| x * (1.0 / 255.0))
        );
        // 2.0, 0.375, 0.125 (denormal) and alpha 1 of 3.
        let c = 0x200 | (0x0C0 << 10) | (0x040 << 20) | (1 << 30);
        assert_eq!(
            clear_color(TargetFormat::Float7e3, clear_value(c, 0xFFFF_FFFF)),
            [2.0, 0.375, 0.125, 1.0 / 3.0]
        );
        // Half bits: R G from RB_COLOR_CLEAR, B A from RB_COLOR_CLEAR_LO.
        let h = clear_value(0x3800_3C00, 0x0000_4000);
        assert_eq!(
            clear_color(TargetFormat::Rgba16Float, h),
            [1.0, 0.5, 2.0, 0.0]
        );
        let u: u32 = 1023 | (3 << 30);
        assert_eq!(
            clear_color(TargetFormat::Rgb10a2, u64::from(u)),
            [1023.0 * (1.0 / 1023.0), 0.0, 0.0, 3.0 * (1.0 / 3.0)]
        );
    }

    #[test]
    fn resolve_packing_saturates_and_rounds_like_xenia() {
        let mut out = [0u8; 4];
        resolve_pixel([2.0, -1.0, 0.5, 1.0], 0, false, DEST_8_8_8_8, &mut out);
        assert_eq!(out, [255, 0, 128, 255]);
        resolve_pixel([2.0, -1.0, 0.5, 1.0], 0, true, DEST_8_8_8_8, &mut out);
        assert_eq!(out, [128, 0, 255, 255]);
        // Exponent bias -1 halves before packing.
        resolve_pixel([1.0, 0.5, 0.0, 1.0], -1, false, DEST_8_8_8_8, &mut out);
        assert_eq!(out, [128, 64, 0, 128]);
        for x in 0..=255u8 {
            let rgba = edram_rgba(TargetFormat::Rgba8, &[x, x, x, x]);
            resolve_pixel(rgba, 0, false, DEST_8_8_8_8, &mut out);
            assert_eq!(out, [x; 4]);
        }
        for x in 0..1024u32 {
            let w = x | (x << 10) | (x << 20) | ((x & 3) << 30);
            let rgba = edram_rgba(TargetFormat::Rgb10a2, &w.to_le_bytes());
            resolve_pixel(rgba, 0, false, DEST_2_10_10_10, &mut out);
            assert_eq!(u32::from_le_bytes(out), w);
        }
        let mut out8 = [0u8; 8];
        let texel = [0x00, 0x3C, 0x00, 0x38, 0x00, 0xC0, 0x00, 0x00];
        let rgba = edram_rgba(TargetFormat::Rgba16Float, &texel);
        resolve_pixel(rgba, 0, false, DEST_16_16_16_16_FLOAT, &mut out8);
        assert_eq!(out8, texel);
    }

    #[test]
    fn float7e3_edram_view_quantises() {
        let h = |v: f32| f32_to_half(v).to_le_bytes();
        let texel = |r: f32, g: f32, b: f32, a: f32| [h(r), h(g), h(b), h(a)].concat();
        // Clamped to [0, 31.875]; alpha to 2 bits (0.5 -> 2 of 3).
        assert_eq!(
            edram_rgba(TargetFormat::Float7e3, &texel(40.0, -1.0, 0.5, 0.5)),
            [31.875, 0.0, 0.5, 2.0 * (1.0 / 3.0)]
        );
        let mut out = [0u8; 4];
        resolve_pixel(
            [0.0, 0.0, 0.0, 2.0 * (1.0 / 3.0)],
            0,
            false,
            DEST_8_8_8_8,
            &mut out,
        );
        assert_eq!(out[3], 170);
        assert_eq!(output_clamp(TargetFormat::Rgba8, false), None);
    }

    #[test]
    fn fixed16_and_two_channel_formats() {
        assert_eq!(fixed16_to_f32(0x7FFF), 32.0);
        assert_eq!(fixed16_to_f32(0), 0.0);
        // Floored at -1, as Xenia's resolve unpacks it.
        assert_eq!(fixed16_to_f32(0x8001), -1.0);
        assert_eq!(fixed16_to_f32(0xFC00), -1.0);
        assert_eq!(fixed16_to_f32(0xFF00), -256.0 * (32.0 / 32767.0));
        let v = clear_value(0xC000_7FFF, 0x8001_0000);
        assert_eq!(
            clear_color(TargetFormat::Rgba16Fixed, v),
            [1.0, -16384.0 / 32767.0, 0.0, -1.0]
        );
        assert_eq!(
            clear_color(
                TargetFormat::Rg32Float,
                clear_value(0x3F80_0000, 0x4000_0000)
            ),
            [1.0, 2.0, 0.0, 0.0]
        );
        let texel = [0x00, 0x3C, 0x00, 0xB8];
        assert_eq!(
            edram_rgba(TargetFormat::Rg16Float, &texel),
            [1.0, -0.5, 0.0, 0.0]
        );
        let mut out = [0u8; 8];
        resolve_pixel([1.0, 0.5, 2.0, -1.0], 0, true, DEST_16_16_16_16, &mut out);
        assert_eq!(out, [0xFF, 0xFF, 0x00, 0x80, 0xFF, 0xFF, 0x00, 0x00]);
        resolve_pixel([1.5, -2.0, 0.0, 0.0], 0, false, DEST_32_32_FLOAT, &mut out);
        assert_eq!(
            out,
            [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()[..]
        );
    }
}
