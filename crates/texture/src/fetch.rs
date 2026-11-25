// Field layout ported from Xenia src/xenia/gpu/xenos.h (xe_gpu_texture_fetch_t,
// DataDimension) and size selection from src/xenia/gpu/texture_util.cc
// GetSubresourcesFromFetchConstant (BSD 3-Clause, see CREDITS.md).

//! The six-word texture fetch constant.

use crate::endian::Endian;
use crate::format::TextureFormat;

/// Xenos `DataDimension` (fetch constant, not the fetch instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    D1 = 0,
    /// 2D, or a stack of 2D slices when `stacked` is set.
    D2OrStacked = 1,
    D3 = 2,
    Cube = 3,
}

/// Every field of `xe_gpu_texture_fetch_t`. Enum-like fields the decoder
/// does not need are kept as their raw bit values (see Xenia's `xenos.h` for
/// their meaning).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFetch {
    // dword 0
    /// `FetchConstantType`: 2 = texture.
    pub fetch_type: u8,
    /// `TextureSign` of X, Y, Z, W: 0 unsigned, 1 signed, 2 biased, 3 gamma.
    pub sign: [u8; 4],
    /// `ClampMode` of X, Y, Z.
    pub clamp: [u8; 3],
    /// Base row pitch in texels >> 5 (use [`TextureFetch::pitch_texels`]).
    pub pitch: u32,
    pub tiled: bool,
    // dword 1
    pub format: TextureFormat,
    pub endian: Endian,
    pub request_size: u8,
    pub stacked: bool,
    /// Direct3D (0) or OpenGL (1) nearest clamp policy.
    pub nearest_clamp_policy: bool,
    /// Guest byte address of the base level (field << 12).
    pub base_address: u32,
    // dword 2
    /// Raw size word; sizes depend on `dimension`, see [`TextureFetch::width`].
    pub size: u32,
    // dword 3
    /// 0 = fraction, 1 = integer.
    pub num_format: u8,
    /// Destination swizzle, 3 bits per X Y Z W: 0-3 pick a component,
    /// 4 = constant 0, 5 = constant 1.
    pub swizzle: u16,
    pub exp_adjust: i8,
    /// `TextureFilter`: 0 point, 1 linear, 2 base map, 3 use fetch constant.
    pub mag_filter: u8,
    pub min_filter: u8,
    pub mip_filter: u8,
    /// `AnisoFilter`.
    pub aniso_filter: u8,
    /// `ArbitraryFilter`.
    pub arbitrary_filter: u8,
    pub border_size: bool,
    // dword 4
    pub vol_mag_filter: bool,
    pub vol_min_filter: bool,
    pub mip_min_level: u8,
    pub mip_max_level: u8,
    pub mag_aniso_walk: bool,
    pub min_aniso_walk: bool,
    /// LOD bias with 5 fractional bits.
    pub lod_bias: i16,
    pub grad_exp_adjust_h: i8,
    pub grad_exp_adjust_v: i8,
    // dword 5
    /// `BorderColor`.
    pub border_color: u8,
    pub force_bc_w_to_max: bool,
    pub tri_clamp: u8,
    pub aniso_bias: i8,
    pub dimension: Dimension,
    pub packed_mips: bool,
    /// Guest byte address of mip level 1 (field << 12); 0 when no mips.
    pub mip_address: u32,
}

fn bits(word: u32, shift: u32, count: u32) -> u32 {
    (word >> shift) & ((1 << count) - 1)
}

fn signed_bits(word: u32, shift: u32, count: u32) -> i32 {
    ((word << (32 - shift - count)) as i32) >> (32 - count)
}

impl TextureFetch {
    /// Decodes the six words as held in the GPU registers (host values, not
    /// guest-memory bytes).
    pub fn decode(w: [u32; 6]) -> Self {
        TextureFetch {
            fetch_type: bits(w[0], 0, 2) as u8,
            sign: [2, 4, 6, 8].map(|s| bits(w[0], s, 2) as u8),
            clamp: [10, 13, 16].map(|s| bits(w[0], s, 3) as u8),
            pitch: bits(w[0], 22, 9),
            tiled: bits(w[0], 31, 1) != 0,

            format: TextureFormat::from_bits(w[1]),
            endian: Endian::from_bits(w[1] >> 6),
            request_size: bits(w[1], 8, 2) as u8,
            stacked: bits(w[1], 10, 1) != 0,
            nearest_clamp_policy: bits(w[1], 11, 1) != 0,
            base_address: w[1] & 0xFFFF_F000,

            size: w[2],

            num_format: bits(w[3], 0, 1) as u8,
            swizzle: bits(w[3], 1, 12) as u16,
            exp_adjust: signed_bits(w[3], 13, 6) as i8,
            mag_filter: bits(w[3], 19, 2) as u8,
            min_filter: bits(w[3], 21, 2) as u8,
            mip_filter: bits(w[3], 23, 2) as u8,
            aniso_filter: bits(w[3], 25, 3) as u8,
            arbitrary_filter: bits(w[3], 28, 3) as u8,
            border_size: bits(w[3], 31, 1) != 0,

            vol_mag_filter: bits(w[4], 0, 1) != 0,
            vol_min_filter: bits(w[4], 1, 1) != 0,
            mip_min_level: bits(w[4], 2, 4) as u8,
            mip_max_level: bits(w[4], 6, 4) as u8,
            mag_aniso_walk: bits(w[4], 10, 1) != 0,
            min_aniso_walk: bits(w[4], 11, 1) != 0,
            lod_bias: signed_bits(w[4], 12, 10) as i16,
            grad_exp_adjust_h: signed_bits(w[4], 22, 5) as i8,
            grad_exp_adjust_v: signed_bits(w[4], 27, 5) as i8,

            border_color: bits(w[5], 0, 2) as u8,
            force_bc_w_to_max: bits(w[5], 2, 1) != 0,
            tri_clamp: bits(w[5], 3, 2) as u8,
            aniso_bias: signed_bits(w[5], 5, 4) as i8,
            dimension: match bits(w[5], 9, 2) {
                0 => Dimension::D1,
                1 => Dimension::D2OrStacked,
                2 => Dimension::D3,
                _ => Dimension::Cube,
            },
            packed_mips: bits(w[5], 11, 1) != 0,
            mip_address: w[5] & 0xFFFF_F000,
        }
    }

    /// Base level width in texels.
    pub fn width(&self) -> u32 {
        1 + match self.dimension {
            Dimension::D1 => bits(self.size, 0, 24),
            Dimension::D2OrStacked | Dimension::Cube => bits(self.size, 0, 13),
            Dimension::D3 => bits(self.size, 0, 11),
        }
    }

    /// Base level height in texels (1 for 1D).
    pub fn height(&self) -> u32 {
        match self.dimension {
            Dimension::D1 => 1,
            Dimension::D2OrStacked | Dimension::Cube => 1 + bits(self.size, 13, 13),
            Dimension::D3 => 1 + bits(self.size, 11, 11),
        }
    }

    /// Depth (3D), slice count (stacked), 6 (cube) or 1.
    pub fn depth(&self) -> u32 {
        match self.dimension {
            Dimension::D1 => 1,
            Dimension::D2OrStacked if self.stacked => 1 + bits(self.size, 26, 6),
            Dimension::D2OrStacked => 1,
            Dimension::D3 => 1 + bits(self.size, 22, 10),
            Dimension::Cube => 6,
        }
    }

    /// The mip levels the fetch samples, (first, last), as Xenia's
    /// GetSubresourcesFromFetchConstant: (0, 0) without a mip page, both
    /// clamped to the levels the size has, the last at least the first, and
    /// the first at least 1 when there are mips but no base page. Pages are
    /// physical: the address bits under 512 MB.
    pub fn mip_range(&self) -> (u32, u32) {
        let page = |address: u32| address & 0x1FFF_F000;
        if page(self.mip_address) == 0 {
            return (0, 0);
        }
        let longest = self.width().max(self.height());
        let longest = if self.dimension == Dimension::D3 {
            longest.max(self.depth())
        } else {
            longest
        };
        let top = longest.ilog2();
        let mut min = u32::from(self.mip_min_level).min(top);
        let max = u32::from(self.mip_max_level).min(top).max(min);
        if max != 0 && page(self.base_address) == 0 {
            min = min.max(1);
        }
        (min, max)
    }

    /// Base row pitch in texels (not blocks).
    pub fn pitch_texels(&self) -> u32 {
        self.pitch << 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_every_dword() {
        let w0 = 2
            | (1 << 2)
            | (2 << 4)
            | (3 << 6)
            | (1 << 10)
            | (2 << 13)
            | (7 << 16)
            | (40 << 22)
            | (1 << 31);
        let w1 = 18 | (1 << 6) | (2 << 8) | (1 << 11) | (0xABCDE << 12);
        let w2 = 1279 | (719 << 13);
        let w3 = 1
            | (0x688 << 1)
            | (0x3F << 13)
            | (1 << 19)
            | (2 << 21)
            | (3 << 23)
            | (4 << 25)
            | (5 << 28)
            | (1 << 31);
        let w4 = 1 | (3 << 2) | (9 << 6) | (1 << 11) | (0x3E0 << 12) | (0x1F << 22) | (1 << 27);
        let w5 = 1 | (1 << 2) | (2 << 3) | (0xF << 5) | (1 << 9) | (1 << 11) | (0x12345 << 12);
        let f = TextureFetch::decode([w0, w1, w2, w3, w4, w5]);
        assert_eq!(f.fetch_type, 2);
        assert_eq!(f.sign, [1, 2, 3, 0]);
        assert_eq!(f.clamp, [1, 2, 7]);
        assert_eq!(f.pitch_texels(), 1280);
        assert!(f.tiled);
        assert_eq!(f.format, TextureFormat::Dxt1);
        assert_eq!(f.endian, Endian::K8in16);
        assert_eq!(f.request_size, 2);
        assert!(!f.stacked);
        assert!(f.nearest_clamp_policy);
        assert_eq!(f.base_address, 0xABCD_E000);
        assert_eq!((f.width(), f.height(), f.depth()), (1280, 720, 1));
        assert_eq!(f.num_format, 1);
        assert_eq!(f.swizzle, 0x688);
        assert_eq!(f.exp_adjust, -1);
        assert_eq!((f.mag_filter, f.min_filter, f.mip_filter), (1, 2, 3));
        assert_eq!((f.aniso_filter, f.arbitrary_filter), (4, 5));
        assert!(f.border_size);
        assert!(f.vol_mag_filter && !f.vol_min_filter);
        assert_eq!((f.mip_min_level, f.mip_max_level), (3, 9));
        assert!(!f.mag_aniso_walk && f.min_aniso_walk);
        assert_eq!(f.lod_bias, -32);
        assert_eq!((f.grad_exp_adjust_h, f.grad_exp_adjust_v), (-1, 1));
        assert_eq!(f.border_color, 1);
        assert!(f.force_bc_w_to_max);
        assert_eq!(f.tri_clamp, 2);
        assert_eq!(f.aniso_bias, -1);
        assert_eq!(f.dimension, Dimension::D2OrStacked);
        assert!(f.packed_mips);
        assert_eq!(f.mip_address, 0x1234_5000);
    }

    #[test]
    fn sizes_per_dimension() {
        let size = 0x1234_5678;
        let with = |dim: u32, stacked: bool| {
            TextureFetch::decode([0, (stacked as u32) << 10, size, 0, 0, dim << 9])
        };
        let d1 = with(0, false);
        assert_eq!(
            (d1.width(), d1.height(), d1.depth()),
            ((size & 0xFF_FFFF) + 1, 1, 1)
        );
        let d2 = with(1, true);
        assert_eq!(
            (d2.width(), d2.height(), d2.depth()),
            (
                (size & 0x1FFF) + 1,
                ((size >> 13) & 0x1FFF) + 1,
                (size >> 26) + 1
            )
        );
        assert_eq!(with(1, false).depth(), 1);
        let d3 = with(2, false);
        assert_eq!(
            (d3.width(), d3.height(), d3.depth()),
            (
                (size & 0x7FF) + 1,
                ((size >> 11) & 0x7FF) + 1,
                (size >> 22) + 1
            )
        );
        assert_eq!(with(3, false).depth(), 6);
    }
}
