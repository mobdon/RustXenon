// Channel bit layouts, component replication and DXT/DXN decoding rules ported
// from Xenia src/xenia/gpu/shaders/pixel_formats.xesli, the texture_load_*.xesl
// shaders and the host format table in
// src/xenia/gpu/d3d12/d3d12_texture_cache.cc; swizzle encoding from
// src/xenia/gpu/xenos.h (XE_GPU_TEXTURE_SWIZZLE) (BSD 3-Clause, see
// CREDITS.md).

//! Format conversion of linear, endian-swapped texture data.
//!
//! Input is rows of blocks, tightly packed, after the fetch constant's endian
//! swap: every element then reads little-endian with component X in the
//! lowest bits. For example `k_8_8_8_8` is bytes X Y Z W in memory order;
//! a guest big-endian `0xAARRGGBB` word with `k8in32` therefore gives
//! X = B, Y = G, Z = R, W = A, and the fetch constant's swizzle (ZYXW for
//! that case) puts the channels in RGBA order.
//!
//! Output is the four components X Y Z W **before** the swizzle. As on the
//! console, formats with fewer components repeat the last one into the
//! missing ones (`k_8` gives XXXX, `k_8_8` XYYY, `k_5_6_5` XYZZ).

use crate::format::TextureFormat as F;
use crate::{Error, TextureFormat};

fn unorm(v: u64, bits: u32) -> f32 {
    v as f32 / ((1u64 << bits) - 1) as f32
}

fn field(v: u64, shift: u32, bits: u32) -> u64 {
    (v >> shift) & ((1 << bits) - 1)
}

/// IEEE 754 half to single.
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as f32;
    match exp {
        0 => sign * mant * f32::powi(2.0, -24),
        31 if mant == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mant / 1024.0) * f32::powi(2.0, exp - 15),
    }
}

/// 24-bit unorm depth to float (Xenia xenos.h UNorm24To32): exact, unlike a
/// division by 0xFFFFFF. Bits above 24 are ignored.
pub fn unorm24_to_f32(n24: u32) -> f32 {
    let n24 = n24 & 0xFF_FFFF;
    (n24 + (n24 >> 23)) as f32 / (1 << 24) as f32
}

/// Xenos 20e4 depth to float (Xenia xenos.cc Float20e4To32). Bits above 24
/// are ignored.
pub fn float20e4_to_f32(f24: u32) -> f32 {
    let f24 = f24 & 0xFF_FFFF;
    if f24 == 0 {
        return 0.0;
    }
    let mut mantissa = f24 & 0xF_FFFF;
    let mut exponent = f24 >> 20;
    if exponent == 0 {
        let lz = mantissa.leading_zeros() - (32 - 21);
        exponent = (1 - lz as i32) as u32;
        mantissa = (mantissa << lz) & 0xF_FFFF;
    }
    f32::from_bits((exponent.wrapping_add(112) << 23) | (mantissa << 3))
}

fn read_le(data: &[u8]) -> u64 {
    data.iter().rev().fold(0, |acc, &b| (acc << 8) | b as u64)
}

/// Bit position and width of each XYZW component of a fixed-point format,
/// with missing components repeating the last one; `None` for depth and
/// float formats.
fn fixed_point_fields(format: TextureFormat) -> Option<[(u32, u32); 4]> {
    Some(match format {
        F::K8 => [(0, 8); 4],
        F::K8_8 => [(0, 8), (8, 8), (8, 8), (8, 8)],
        F::K8_8_8_8 => [(0, 8), (8, 8), (16, 8), (24, 8)],
        F::K5_6_5 => [(0, 5), (5, 6), (11, 5), (11, 5)],
        // Named 6_5_5 but laid out 5:5:6 from the low bits (Xenia).
        F::K6_5_5 => [(0, 5), (5, 5), (10, 6), (10, 6)],
        F::K1_5_5_5 => [(0, 5), (5, 5), (10, 5), (15, 1)],
        F::K4_4_4_4 => [(0, 4), (4, 4), (8, 4), (12, 4)],
        F::K2_10_10_10 => [(0, 10), (10, 10), (20, 10), (30, 2)],
        // Xenia loads k_10_11_11 as R11G11B10 and k_11_11_10 as R10G11B11.
        F::K10_11_11 => [(0, 11), (11, 11), (22, 10), (22, 10)],
        F::K11_11_10 => [(0, 10), (10, 11), (21, 11), (21, 11)],
        F::K16 => [(0, 16); 4],
        F::K16_16 => [(0, 16), (16, 16), (16, 16), (16, 16)],
        F::K16_16_16_16 => [(0, 16), (16, 16), (32, 16), (48, 16)],
        _ => return None,
    })
}

/// An n-bit field read as two's complement and scaled so that the largest
/// positive value is 1; the most negative value (one step past -1) clamps
/// to -1. A 1-bit field holds only 0 and -1.
fn snorm(v: u64, bits: u32) -> f32 {
    if bits == 1 {
        return -(v as f32);
    }
    let half = 1i64 << (bits - 1);
    let s = v as i64 - if v as i64 >= half { 2 * half } else { 0 };
    (s as f32 / (half - 1) as f32).max(-1.0)
}

/// One uncompressed texel to XYZW. `signed` components of fixed-point
/// formats read as two's complement ([`snorm`]); depth and float formats
/// ignore it, as they have no unsigned encoding to reinterpret.
fn texel(format: TextureFormat, d: &[u8], signed: [bool; 4]) -> [f32; 4] {
    let v = read_le(&d[..d.len().min(8)]);
    if let Some(fields) = fixed_point_fields(format) {
        return std::array::from_fn(|i| {
            let (shift, bits) = fields[i];
            let raw = field(v, shift, bits);
            if signed[i] {
                snorm(raw, bits)
            } else {
                unorm(raw, bits)
            }
        });
    }
    let f16 = |i: usize| half_to_f32(u16::from_le_bytes([d[2 * i], d[2 * i + 1]]));
    let f32_at =
        |i: usize| f32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]);
    match format {
        // Depth in the high 24 bits, stencil in the low 8 (Xenia's
        // texture_load_depth_* shaders); the host swizzle is RRRR.
        F::K24_8 => [unorm24_to_f32((v >> 8) as u32); 4],
        F::K24_8Float => [float20e4_to_f32((v >> 8) as u32); 4],
        F::K16Float => [f16(0); 4],
        F::K16_16Float => [f16(0), f16(1), f16(1), f16(1)],
        F::K16_16_16_16Float => [f16(0), f16(1), f16(2), f16(3)],
        F::K32Float => [f32_at(0); 4],
        F::K32_32Float => [f32_at(0), f32_at(1), f32_at(1), f32_at(1)],
        F::K32_32_32Float => [f32_at(0), f32_at(1), f32_at(2), f32_at(2)],
        F::K32_32_32_32Float => [f32_at(0), f32_at(1), f32_at(2), f32_at(3)],
        _ => unreachable!("not an uncompressed convertible format"),
    }
}

fn expand_565(c: u16) -> [u32; 3] {
    let (r, g, b) = ((c >> 11) as u32, ((c >> 5) & 63) as u32, (c & 31) as u32);
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// The 8-byte DXT colour half-block. `four_color` forces the opaque
/// four-colour mode, as DXT2/3 and DXT4/5 always use it.
fn dxt_color(block: &[u8], four_color: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let codes = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let (e0, e1) = (expand_565(c0), expand_565(c1));
    let mix = |w0: u32, w1: u32, div: u32| -> [u8; 4] {
        let c = |i: usize| ((e0[i] * w0 + e1[i] * w1) / div) as u8;
        [c(0), c(1), c(2), 255]
    };
    let palette = if four_color || c0 > c1 {
        [mix(1, 0, 1), mix(0, 1, 1), mix(2, 1, 3), mix(1, 2, 3)]
    } else {
        [mix(1, 0, 1), mix(0, 1, 1), mix(1, 1, 2), [0, 0, 0, 0]]
    };
    std::array::from_fn(|i| palette[((codes >> (2 * i)) & 3) as usize])
}

/// The 8-byte DXT5 alpha half-block (also DXT5A and each DXN channel).
fn dxt5_alpha(block: &[u8]) -> [u8; 16] {
    let (a0, a1) = (block[0] as u32, block[1] as u32);
    let codes = read_le(&block[2..8]);
    std::array::from_fn(|i| {
        let code = ((codes >> (3 * i)) & 7) as u32;
        let a = match code {
            0 => a0,
            1 => a1,
            _ if a0 > a1 => ((8 - code) * a0 + (code - 1) * a1) / 7,
            6 => 0,
            7 => 255,
            _ => ((6 - code) * a0 + (code - 1) * a1) / 5,
        };
        a as u8
    })
}

/// The 8-byte DXT3 explicit 4-bit alpha half-block (also DXT3A).
fn dxt3_alpha(block: &[u8]) -> [u8; 16] {
    let bits = read_le(&block[..8]);
    std::array::from_fn(|i| (((bits >> (4 * i)) & 0xF) * 17) as u8)
}

/// One 4x4 block to XYZW bytes, row-major.
fn block4x4(format: TextureFormat, b: &[u8]) -> [[u8; 4]; 16] {
    match format {
        F::Dxt1 => dxt_color(b, false),
        F::Dxt2_3 => {
            let (alpha, mut texels) = (dxt3_alpha(&b[..8]), dxt_color(&b[8..], true));
            texels.iter_mut().zip(alpha).for_each(|(t, a)| t[3] = a);
            texels
        }
        F::Dxt4_5 => {
            let (alpha, mut texels) = (dxt5_alpha(&b[..8]), dxt_color(&b[8..], true));
            texels.iter_mut().zip(alpha).for_each(|(t, a)| t[3] = a);
            texels
        }
        F::Dxn => {
            let (x, y) = (dxt5_alpha(&b[..8]), dxt5_alpha(&b[8..]));
            std::array::from_fn(|i| [x[i], y[i], y[i], y[i]])
        }
        F::Dxt5A => dxt5_alpha(b).map(|a| [a; 4]),
        F::Dxt3A => dxt3_alpha(b).map(|a| [a; 4]),
        _ => unreachable!("not a 4x4 block format"),
    }
}

/// Checks the format is decodable and `data` holds every block; returns the
/// output texel count.
/// Checked arithmetic: sizes come from guest registers or callers and must
/// not overflow into a small allocation.
fn check_len(format: TextureFormat, data: &[u8], width: u32, height: u32) -> Result<usize, Error> {
    if !format.can_convert() {
        return Err(Error::UnsupportedFormat(format));
    }
    let info = format.info();
    let too_big = Error::BadLayout("texture size overflows");
    let needed = (width.div_ceil(info.block_width) as usize)
        .checked_mul(height.div_ceil(info.block_height) as usize)
        .and_then(|n| n.checked_mul(info.bytes_per_block() as usize))
        .ok_or(too_big.clone())?;
    let texels = (width as usize)
        .checked_mul(height as usize)
        .filter(|n| n.checked_mul(4).is_some())
        .ok_or(too_big)?;
    if data.len() < needed {
        return Err(Error::TooShort {
            needed,
            got: data.len(),
        });
    }
    Ok(texels)
}

/// A decoded texel: block-compressed formats decode to bytes, which stay
/// exact; the rest decode to floats.
enum Texel {
    U8([u8; 4]),
    F32([f32; 4]),
}

fn decode(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
    signed: [bool; 4],
    mut put: impl FnMut(usize, Texel),
) -> Result<(), Error> {
    check_len(format, data, width, height)?;
    let format = format.data_format();
    let info = format.info();
    let bpb = info.bytes_per_block() as usize;
    let (w, h) = (width as usize, height as usize);
    if info.block_width == 1 {
        for (i, d) in data.chunks_exact(bpb).take(w * h).enumerate() {
            put(i, Texel::F32(texel(format, d, signed)));
        }
        return Ok(());
    }
    let blocks_x = w.div_ceil(4);
    for (bi, b) in data
        .chunks_exact(bpb)
        .take(blocks_x * h.div_ceil(4))
        .enumerate()
    {
        let (bx, by) = (bi % blocks_x * 4, bi / blocks_x * 4);
        for (i, t) in block4x4(format, b).into_iter().enumerate() {
            let (x, y) = (bx + i % 4, by + i / 4);
            if x < w && y < h {
                put(y * w + x, Texel::U8(t));
            }
        }
    }
    Ok(())
}

/// Decodes `width` x `height` texels to XYZW floats (`width * height * 4`).
/// Normalized formats give 0..1; float formats are passed through unclamped.
pub fn to_rgba_f32(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Error> {
    to_rgba_f32_signed(format, data, width, height, [false; 4])
}

/// [`to_rgba_f32`] with the X Y Z W components marked in `signed` (fetch
/// sign 1, before the swizzle) decoded as signed normalized: the unsigned
/// n-bit field is read as two's complement and divided by 2^(n-1) - 1, with
/// the most negative value clamped to -1, so the result is in -1..1.
///
/// Fixed-point formats convert from the raw field, exactly. Depth and float
/// formats are unaffected. Block-compressed formats convert their decoded
/// 8-bit values.
pub fn to_rgba_f32_signed(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
    signed: [bool; 4],
) -> Result<Vec<f32>, Error> {
    let mut out = vec![0.0; check_len(format, data, width, height)? * 4];
    decode(format, data, width, height, signed, |i, t| {
        let t = match t {
            // shortcut: signed block formats reinterpret the byte after
            // unsigned palette interpolation, not interpolating signed
            // endpoints; exact for endpoint texels only. Upgrade if signed
            // DXT/DXN content shows up.
            Texel::U8(t) => std::array::from_fn(|c| {
                if signed[c] {
                    snorm(t[c] as u64, 8)
                } else {
                    t[c] as f32 / 255.0
                }
            }),
            Texel::F32(t) => t,
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&t);
    })?;
    Ok(out)
}

/// [`to_rgba_f32`] of a 16-bit float format (`k_16_FLOAT`, `k_16_16_FLOAT`,
/// `k_16_16_16_16_FLOAT` and the expand formats that read as them) kept as
/// halves, bit for bit what [`to_rgba_f32`] then a round to half gives,
/// without the float round trip: every NaN becomes the quiet NaN 0x7E00, as
/// through f32.
pub fn to_rgba_f16(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u16>, Error> {
    let texels = check_len(format, data, width, height)?;
    let n = match format.data_format() {
        F::K16Float => 1,
        F::K16_16Float => 2,
        F::K16_16_16_16Float => 4,
        _ => return Err(Error::UnsupportedFormat(format)),
    };
    let mut out = Vec::with_capacity(texels * 4);
    for d in data.chunks_exact(2 * n).take(texels) {
        let h = |i: usize| match u16::from_le_bytes([d[2 * i], d[2 * i + 1]]) {
            h if h & 0x7FFF > 0x7C00 => 0x7E00,
            h => h,
        };
        out.extend_from_slice(&match n {
            1 => [h(0); 4],
            2 => [h(0), h(1), h(1), h(1)],
            _ => [h(0), h(1), h(2), h(3)],
        });
    }
    Ok(out)
}

/// Decodes `width` x `height` texels to XYZW bytes (`width * height * 4`).
/// Values are clamped to 0..1 and rounded to the nearest 8-bit step; NaN
/// becomes 0.
pub fn to_rgba8(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Error> {
    let texels = check_len(format, data, width, height)?;
    // k_8_8_8_8 bytes are already XYZW unorm8: the float round trip below
    // gives each byte back unchanged, so copy them.
    if format.data_format() == F::K8_8_8_8 {
        return Ok(data[..texels * 4].to_vec());
    }
    let mut out = vec![0; texels * 4];
    decode(format, data, width, height, [false; 4], |i, t| {
        let t = match t {
            Texel::U8(t) => t,
            Texel::F32(t) => t.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8),
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&t);
    })?;
    Ok(out)
}

/// Applies the fetch constant's 12-bit destination swizzle to XYZW bytes in
/// place. Per output channel (3 bits, X first): 0-3 select X Y Z W, 4 is
/// constant 0, 5 constant 1. 6 and 7 are undefined; like Xenia's
/// `TextureCache::GuestToHostSwizzle` (texture_cache.cc) they become 0 and 1.
pub fn apply_swizzle(rgba8: &mut [u8], swizzle: u16) {
    swizzle_texels(rgba8, swizzle, 0, 255);
}

/// [`apply_swizzle`] for XYZW floats (constant 1 is 1.0).
pub fn apply_swizzle_f32(rgba: &mut [f32], swizzle: u16) {
    swizzle_texels(rgba, swizzle, 0.0, 1.0);
}

/// [`apply_swizzle_f32`] on halves: constant 1 is 0x3C00.
pub fn apply_swizzle_f16(rgba: &mut [u16], swizzle: u16) {
    swizzle_texels(rgba, swizzle, 0, 0x3C00);
}

fn swizzle_texels<T: Copy>(texels: &mut [T], swizzle: u16, zero: T, one: T) {
    let sel: [u16; 4] = std::array::from_fn(|i| (swizzle >> (3 * i)) & 7);
    if sel == [0, 1, 2, 3] {
        return;
    }
    // No constants: a plain reorder, which the compiler makes a shuffle
    // (a 1280x720 front buffer goes through here every frame).
    if sel.iter().all(|&s| s < 4) {
        let i = sel.map(usize::from);
        for t in texels.chunks_exact_mut(4) {
            let src = [t[0], t[1], t[2], t[3]];
            t.copy_from_slice(&[src[i[0]], src[i[1]], src[i[2]], src[i[3]]]);
        }
        return;
    }
    for t in texels.chunks_exact_mut(4) {
        let src = [t[0], t[1], t[2], t[3]];
        for (dst, &s) in t.iter_mut().zip(&sel) {
            *dst = match s {
                0..=3 => src[s as usize],
                _ if s & 1 == 0 => zero,
                _ => one,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swz(sel: [u16; 4]) -> u16 {
        sel[0] | (sel[1] << 3) | (sel[2] << 6) | (sel[3] << 9)
    }

    fn one(format: TextureFormat, bytes: &[u8]) -> [u8; 4] {
        let v = to_rgba8(format, bytes, 1, 1).unwrap();
        [v[0], v[1], v[2], v[3]]
    }

    #[test]
    fn k8_and_8_8_replicate() {
        assert_eq!(one(F::K8, &[0x5A]), [0x5A; 4]);
        assert_eq!(one(F::K8A, &[0x11]), [0x11; 4]);
        assert_eq!(one(F::K8_8, &[0x12, 0x34]), [0x12, 0x34, 0x34, 0x34]);
    }

    #[test]
    fn k8_8_8_8_and_variants() {
        let px = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(one(F::K8_8_8_8, &px), px);
        assert_eq!(one(F::K8_8_8_8A, &px), px);
        assert_eq!(one(F::K8_8_8_8As16_16_16_16, &px), px);
    }

    /// The k_8_8_8_8 copy path gives what the float conversion gives, for
    /// every byte value in every channel.
    #[test]
    fn k8_8_8_8_copy_matches_the_float_conversion() {
        let bytes: Vec<u8> = (0..=255u8)
            .flat_map(|b| [b, 255 - b, b ^ 0x5A, b / 3])
            .collect();
        let floats = to_rgba_f32(F::K8_8_8_8, &bytes, 16, 16).unwrap();
        let rounded: Vec<u8> = floats.iter().map(|c| (c * 255.0).round() as u8).collect();
        assert_eq!(to_rgba8(F::K8_8_8_8, &bytes, 16, 16).unwrap(), rounded);
        assert_eq!(rounded, bytes);
    }

    #[test]
    fn k5_6_5() {
        // X = 31, Y = 0, Z = 16: 0b10000_000000_11111 = 0x801F.
        assert_eq!(one(F::K5_6_5, &0x801Fu16.to_le_bytes()), [255, 0, 132, 132]);
        // Y = 63 only: 0x07E0.
        assert_eq!(one(F::K5_6_5, &0x07E0u16.to_le_bytes()), [0, 255, 0, 0]);
    }

    #[test]
    fn k6_5_5_is_5_5_6_from_low_bits() {
        // X = 0, Y = 31, Z = 0: 0b11111 << 5 = 0x03E0.
        assert_eq!(one(F::K6_5_5, &0x03E0u16.to_le_bytes()), [0, 255, 0, 0]);
        // Z = 63: 0xFC00.
        assert_eq!(one(F::K6_5_5, &0xFC00u16.to_le_bytes()), [0, 0, 255, 255]);
    }

    #[test]
    fn k1_5_5_5_and_4_4_4_4() {
        // A = 1, Z = 31, others 0: 0xFC00.
        assert_eq!(one(F::K1_5_5_5, &0xFC00u16.to_le_bytes()), [0, 0, 255, 255]);
        // X = 1 only: alpha 0.
        assert_eq!(one(F::K1_5_5_5, &0x0001u16.to_le_bytes()), [8, 0, 0, 0]);
        // 4_4_4_4: X=0xF, Y=0x0, Z=0x5, W=0xA -> 0xA50F.
        assert_eq!(
            one(F::K4_4_4_4, &0xA50Fu16.to_le_bytes()),
            [255, 0, 85, 170]
        );
    }

    #[test]
    fn k2_10_10_10() {
        // X = 1023, Y = 0, Z = 512, W = 2.
        let v: u32 = 1023 | (512 << 20) | (2 << 30);
        assert_eq!(one(F::K2_10_10_10, &v.to_le_bytes()), [255, 0, 128, 170]);
    }

    #[test]
    fn k10_11_11_and_11_11_10() {
        // k_10_11_11: X 11 bits, Y 11 bits, Z 10 bits from the low end.
        let v: u32 = 2047 | (1023 << 22);
        assert_eq!(one(F::K10_11_11, &v.to_le_bytes()), [255, 0, 255, 255]);
        // k_11_11_10: X 10 bits, Y 11, Z 11.
        let v: u32 = (1023 << 10) | (2047 << 21);
        // 1023 / 2047 * 255 = 127.44.
        assert_eq!(one(F::K11_11_10, &v.to_le_bytes()), [0, 127, 255, 255]);
    }

    #[test]
    fn k16_unorm() {
        assert_eq!(one(F::K16, &0xFFFFu16.to_le_bytes()), [255; 4]);
        let v: u32 = 0x8000 << 16;
        assert_eq!(one(F::K16_16, &v.to_le_bytes()), [0, 128, 128, 128]);
        let v: u64 = 0xFFFF | (0x0101 << 32) | (0xFFFF << 48);
        assert_eq!(one(F::K16_16_16_16, &v.to_le_bytes()), [255, 0, 1, 255]);
    }

    #[test]
    fn half_floats() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x0001), f32::powi(2.0, -24));
        assert_eq!(half_to_f32(0x8001), -f32::powi(2.0, -24));
        assert_eq!(half_to_f32(0x7C00), f32::INFINITY);
        // 16_16_16_16_FLOAT: 1.0, 0.5, -2.0 (clamped to 0), 2.0 (clamped).
        let mut d = Vec::new();
        for h in [0x3C00u16, 0x3800, 0xC000, 0x4000] {
            d.extend_from_slice(&h.to_le_bytes());
        }
        assert_eq!(one(F::K16_16_16_16Float, &d), [255, 128, 0, 255]);
        let f = to_rgba_f32(F::K16_16_16_16Float, &d, 1, 1).unwrap();
        assert_eq!(f, [1.0, 0.5, -2.0, 2.0]);
        assert_eq!(
            one(F::K16_16Float, &[0x00, 0x38, 0x00, 0x3C]),
            [128, 255, 255, 255]
        );
        assert_eq!(one(F::K16Expand, &[0x00, 0x3C]), [255; 4]);
    }

    #[test]
    fn single_floats() {
        let mut d = Vec::new();
        for f in [0.25f32, 1.5, -1.0, 1.0] {
            d.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(one(F::K32_32_32_32Float, &d), [64, 255, 0, 255]);
        assert_eq!(one(F::K32_32_32Float, &d[..12]), [64, 255, 0, 0]);
        assert_eq!(one(F::K32_32Float, &d[..8]), [64, 255, 255, 255]);
        assert_eq!(one(F::K32Float, &d[..4]), [64; 4]);
        assert_eq!(to_rgba_f32(F::K32Float, &d[4..8], 1, 1).unwrap(), [1.5; 4]);
    }

    #[test]
    fn depth_24_8_unorm_and_float() {
        // Depth in bits 8..32, stencil (0x5A) in bits 0..8 and ignored.
        let word = |d: u32| ((d << 8) | 0x5A).to_le_bytes();
        let f = |format, d| to_rgba_f32(format, &word(d), 1, 1).unwrap();
        assert_eq!(f(F::K24_8, 0xFF_FFFF), [1.0; 4]);
        assert_eq!(f(F::K24_8, 0x80_0000), [0.5 + 1.0 / (1 << 24) as f32; 4]);
        assert_eq!(f(F::K24_8, 0), [0.0; 4]);
        // 20e4: exponent 15 is 1.0; 14 with the top mantissa bit is 0.75.
        assert_eq!(f(F::K24_8Float, 15 << 20), [1.0; 4]);
        assert_eq!(f(F::K24_8Float, (14 << 20) | (1 << 19)), [0.75; 4]);
        // Smallest denormal: 2^-20 * 2^-14.
        assert_eq!(f(F::K24_8Float, 1), [f32::powi(2.0, -34); 4]);
        assert_eq!(f(F::K24_8Float, 0), [0.0; 4]);
        assert_eq!(one(F::K24_8, &word(0x80_0000)), [128; 4]);
    }

    #[test]
    fn unorm24_end_points() {
        assert_eq!(unorm24_to_f32(0), 0.0);
        assert_eq!(unorm24_to_f32(0xFF_FFFF), 1.0);
        assert_eq!(unorm24_to_f32(0x80_0000), 0.5 + 1.0 / (1 << 24) as f32);
    }

    #[test]
    fn float20e4_known_values() {
        assert_eq!(float20e4_to_f32(0), 0.0);
        // Exponent 15 (bias 15), mantissa 0: 1.0.
        assert_eq!(float20e4_to_f32(15 << 20), 1.0);
        assert_eq!(float20e4_to_f32((15 << 20) | (1 << 19)), 1.5);
    }

    #[test]
    fn snorm_fields() {
        assert_eq!(snorm(0x7F, 8), 1.0);
        assert_eq!(snorm(0x80, 8), -1.0);
        assert_eq!(snorm(0x81, 8), -1.0);
        assert_eq!(snorm(0xFF, 8), -1.0 / 127.0);
        assert_eq!(snorm(0x8001, 16), -1.0);
        assert_eq!(snorm(0x200, 10), -1.0);
        assert_eq!(snorm(0x3FF, 11), 1.0);
        assert_eq!([snorm(0, 1), snorm(1, 1)], [0.0, -1.0]);
        assert_eq!([snorm(1, 2), snorm(2, 2), snorm(3, 2)], [1.0, -1.0, -1.0]);
        // k_2_10_10_10 with only W signed: X stays unorm.
        let v = 0x3FFu32 | (3 << 30);
        let t = to_rgba_f32_signed(
            F::K2_10_10_10,
            &v.to_le_bytes(),
            1,
            1,
            [false, false, false, true],
        );
        assert_eq!(t.unwrap(), [1.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn swizzle_f32_constants() {
        let mut t = [0.25, 2.0, -3.0, 4.0];
        apply_swizzle_f32(&mut t, swz([3, 0, 4, 5]));
        assert_eq!(t, [4.0, 0.25, 0.0, 1.0]);
    }

    /// Builds a DXT1 block: color0, color1, 2-bit codes row-major.
    fn dxt1_block(c0: u16, c1: u16, codes: [u32; 16]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&c0.to_le_bytes());
        b.extend_from_slice(&c1.to_le_bytes());
        let bits = codes
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &c)| acc | (c << (2 * i)));
        b.extend_from_slice(&bits.to_le_bytes());
        b
    }

    #[test]
    fn dxt1_four_color() {
        // c0 = pure red 0xF800, c1 = pure blue 0x001F (c0 > c1). Codes 0..3
        // across the first row.
        let mut codes = [0; 16];
        codes[..4].copy_from_slice(&[0, 1, 2, 3]);
        let v = to_rgba8(F::Dxt1, &dxt1_block(0xF800, 0x001F, codes), 4, 4).unwrap();
        assert_eq!(&v[0..4], &[255, 0, 0, 255]);
        assert_eq!(&v[4..8], &[0, 0, 255, 255]);
        // (2*255 + 0) / 3 = 170, (0 + 255) / 3 = 85.
        assert_eq!(&v[8..12], &[170, 0, 85, 255]);
        assert_eq!(&v[12..16], &[85, 0, 170, 255]);
        assert_eq!(&v[60..64], &[255, 0, 0, 255]);
    }

    #[test]
    fn dxt1_punch_through() {
        // c0 = 0x001F (blue) <= c1 = 0x07E0 (green): code 2 is the average,
        // code 3 transparent black.
        let mut codes = [3; 16];
        codes[0] = 2;
        let v = to_rgba8(F::Dxt1, &dxt1_block(0x001F, 0x07E0, codes), 4, 4).unwrap();
        assert_eq!(&v[0..4], &[0, 127, 127, 255]);
        assert_eq!(&v[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn dxt1_partial_block_size() {
        // 2x3 texture from one block: only the covered texels are written.
        let codes = std::array::from_fn(|i| (i % 4) as u32);
        let v = to_rgba8(F::Dxt1, &dxt1_block(0xFFFF, 0x0000, codes), 2, 3).unwrap();
        assert_eq!(v.len(), 2 * 3 * 4);
        assert_eq!(&v[0..8], &[255, 255, 255, 255, 0, 0, 0, 255]);
        assert_eq!(&v[16..20], &[255, 255, 255, 255]); // (0, 2): code 0
    }

    #[test]
    fn dxt2_3_explicit_alpha_and_forced_four_color() {
        // Alpha nibbles 0..15 row-major; color block with c0 < c1 still in
        // four-colour mode (code 3 is not transparent).
        let alpha: u64 = (0..16).fold(0, |acc, i| acc | ((i as u64) << (4 * i)));
        let mut b = alpha.to_le_bytes().to_vec();
        b.extend(dxt1_block(0x0000, 0xFFFF, [3; 16]));
        let v = to_rgba8(F::Dxt2_3, &b, 4, 4).unwrap();
        // Code 3 = (c0 + 2*c1)/3 = 170.
        assert_eq!(&v[0..4], &[170, 170, 170, 0]);
        assert_eq!(&v[4..8], &[170, 170, 170, 17]);
        assert_eq!(&v[60..64], &[170, 170, 170, 255]);
    }

    /// Builds a DXT5-style alpha block: two endpoints, 3-bit codes.
    fn alpha_block(a0: u8, a1: u8, codes: [u64; 16]) -> Vec<u8> {
        let bits = codes
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &c)| acc | (c << (3 * i)));
        let mut b = vec![a0, a1];
        b.extend_from_slice(&bits.to_le_bytes()[..6]);
        b
    }

    #[test]
    fn dxt4_5_interpolated_alpha() {
        // a0 = 255 > a1 = 0: 8-step. Code 2 = 6*255/7 = 218, code 7 = 255/7 = 36.
        let mut codes = [0; 16];
        codes[1] = 1;
        codes[2] = 2;
        codes[3] = 7;
        let mut b = alpha_block(255, 0, codes);
        b.extend(dxt1_block(0xF800, 0xF800, [0; 16]));
        let v = to_rgba8(F::Dxt4_5, &b, 4, 4).unwrap();
        let alphas: Vec<u8> = v.chunks(4).take(4).map(|t| t[3]).collect();
        assert_eq!(alphas, [255, 0, 218, 36]);
        assert_eq!(&v[0..3], &[255, 0, 0]);
    }

    #[test]
    fn dxt5a_six_step_and_constants() {
        // a0 = 0 <= a1 = 250: 6-step. Code 2 = (4*0 + 250)/5 = 50, code 5 =
        // (0 + 4*250)/5 = 200, 6 = 0, 7 = 255.
        let mut codes = [0; 16];
        codes[..6].copy_from_slice(&[1, 2, 5, 6, 7, 0]);
        let v = to_rgba8(F::Dxt5A, &alpha_block(0, 250, codes), 4, 4).unwrap();
        let x: Vec<u8> = v.chunks(4).take(6).map(|t| t[0]).collect();
        assert_eq!(x, [250, 50, 200, 0, 255, 0]);
        assert_eq!(&v[4..8], &[50; 4]);
    }

    #[test]
    fn dxn_two_channels() {
        // X block: all code 0 (a0 = 10). Y block: all code 1 (a1 = 200).
        let mut b = alpha_block(10, 20, [0; 16]);
        b.extend(alpha_block(100, 200, [1; 16]));
        let v = to_rgba8(F::Dxn, &b, 4, 4).unwrap();
        assert!(v.chunks(4).all(|t| t == [10, 200, 200, 200]));
    }

    #[test]
    fn dxt3a_nibbles() {
        let bits: u64 = 0xF | (0x8 << 4);
        let v = to_rgba8(F::Dxt3A, &bits.to_le_bytes(), 4, 4).unwrap();
        assert_eq!(&v[0..8], &[255, 255, 255, 255, 136, 136, 136, 136]);
        assert_eq!(&v[8..12], &[0; 4]);
    }

    #[test]
    fn errors() {
        assert_eq!(
            to_rgba8(F::Ctx1, &[0; 8], 4, 4),
            Err(Error::UnsupportedFormat(F::Ctx1))
        );
        assert_eq!(
            to_rgba8(F::K8_8_8_8, &[0; 7], 2, 1),
            Err(Error::TooShort { needed: 8, got: 7 })
        );
        assert_eq!(
            to_rgba8(F::Dxt1, &[0; 8], 5, 4),
            Err(Error::TooShort { needed: 16, got: 8 })
        );
    }

    #[test]
    fn swizzle_selects_and_constants() {
        let mut px = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // W Z Y X.
        apply_swizzle(&mut px, swz([3, 2, 1, 0]));
        assert_eq!(px, [4, 3, 2, 1, 8, 7, 6, 5]);
        // 1 1 1 W, as used for alpha-only text textures.
        let mut px = vec![9, 9, 9, 42];
        apply_swizzle(&mut px, swz([5, 5, 5, 3]));
        assert_eq!(px, [255, 255, 255, 42]);
        // X 0 1 X.
        let mut px = vec![7, 8, 9, 10];
        apply_swizzle(&mut px, swz([0, 4, 5, 0]));
        assert_eq!(px, [7, 0, 255, 7]);
    }

    #[test]
    fn hostile_sizes_are_errors_not_panics() {
        // u32::MAX squared overflows usize; must be an error before any
        // allocation.
        assert!(to_rgba8(F::K8, &[0; 4], u32::MAX, u32::MAX).is_err());
        assert!(to_rgba_f32(F::Dxt1, &[0; 8], u32::MAX, u32::MAX).is_err());
        // Unsupported format wins over a short buffer.
        assert_eq!(
            to_rgba8(F::Ctx1, &[], 4, 4),
            Err(Error::UnsupportedFormat(F::Ctx1))
        );
    }

    #[test]
    fn swizzle_6_and_7_are_constants() {
        // Xenia texture_cache.cc GuestToHostSwizzle: 6 -> 0, 7 -> 1.
        let mut px = vec![9, 9, 9, 9];
        apply_swizzle(&mut px, swz([6, 7, 6, 7]));
        assert_eq!(px, [0, 255, 0, 255]);
    }
}
