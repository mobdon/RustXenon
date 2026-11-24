// Format list ported from Xenia src/xenia/gpu/xenos.h (TextureFormat) and the
// block size table from src/xenia/gpu/texture_info_formats.cc (BSD 3-Clause,
// see CREDITS.md).

//! The Xenos texture formats and their block sizes.

/// Xenos `TextureFormat` (6 bits in the fetch constant). Names follow the
/// hardware's component bit counts, lowest component (X) first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum TextureFormat {
    K1Reverse = 0,
    K1 = 1,
    K8 = 2,
    K1_5_5_5 = 3,
    K5_6_5 = 4,
    K6_5_5 = 5,
    K8_8_8_8 = 6,
    K2_10_10_10 = 7,
    K8A = 8,
    K8B = 9,
    K8_8 = 10,
    CrY1CbY0Rep = 11,
    Y1CrY0CbRep = 12,
    K16_16Edram = 13,
    K8_8_8_8A = 14,
    K4_4_4_4 = 15,
    K10_11_11 = 16,
    K11_11_10 = 17,
    Dxt1 = 18,
    Dxt2_3 = 19,
    Dxt4_5 = 20,
    K16_16_16_16Edram = 21,
    K24_8 = 22,
    K24_8Float = 23,
    K16 = 24,
    K16_16 = 25,
    K16_16_16_16 = 26,
    K16Expand = 27,
    K16_16Expand = 28,
    K16_16_16_16Expand = 29,
    K16Float = 30,
    K16_16Float = 31,
    K16_16_16_16Float = 32,
    K32 = 33,
    K32_32 = 34,
    K32_32_32_32 = 35,
    K32Float = 36,
    K32_32Float = 37,
    K32_32_32_32Float = 38,
    K32As8 = 39,
    K32As8_8 = 40,
    K16Mpeg = 41,
    K16_16Mpeg = 42,
    K8Interlaced = 43,
    K32As8Interlaced = 44,
    K32As8_8Interlaced = 45,
    K16Interlaced = 46,
    K16MpegInterlaced = 47,
    K16_16MpegInterlaced = 48,
    Dxn = 49,
    K8_8_8_8As16_16_16_16 = 50,
    Dxt1As16_16_16_16 = 51,
    Dxt2_3As16_16_16_16 = 52,
    Dxt4_5As16_16_16_16 = 53,
    K2_10_10_10As16_16_16_16 = 54,
    K10_11_11As16_16_16_16 = 55,
    K11_11_10As16_16_16_16 = 56,
    K32_32_32Float = 57,
    Dxt3A = 58,
    Dxt5A = 59,
    Ctx1 = 60,
    Dxt3AAs1_1_1_1 = 61,
    K8_8_8_8GammaEdram = 62,
    K2_10_10_10FloatEdram = 63,
}

/// Block geometry of a format. Uncompressed formats have 1x1 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatInfo {
    pub block_width: u32,
    pub block_height: u32,
    pub bits_per_pixel: u32,
}

impl FormatInfo {
    pub fn bytes_per_block(self) -> u32 {
        self.block_width * self.block_height * self.bits_per_pixel / 8
    }
}

use TextureFormat as F;

/// Indexed by format number: (format, block width, block height, bits per pixel).
const TABLE: [(TextureFormat, u32, u32, u32); 64] = [
    (F::K1Reverse, 1, 1, 1),
    (F::K1, 1, 1, 1),
    (F::K8, 1, 1, 8),
    (F::K1_5_5_5, 1, 1, 16),
    (F::K5_6_5, 1, 1, 16),
    (F::K6_5_5, 1, 1, 16),
    (F::K8_8_8_8, 1, 1, 32),
    (F::K2_10_10_10, 1, 1, 32),
    (F::K8A, 1, 1, 8),
    (F::K8B, 1, 1, 8),
    (F::K8_8, 1, 1, 16),
    (F::CrY1CbY0Rep, 2, 1, 16),
    (F::Y1CrY0CbRep, 2, 1, 16),
    (F::K16_16Edram, 1, 1, 32),
    (F::K8_8_8_8A, 1, 1, 32),
    (F::K4_4_4_4, 1, 1, 16),
    (F::K10_11_11, 1, 1, 32),
    (F::K11_11_10, 1, 1, 32),
    (F::Dxt1, 4, 4, 4),
    (F::Dxt2_3, 4, 4, 8),
    (F::Dxt4_5, 4, 4, 8),
    (F::K16_16_16_16Edram, 1, 1, 64),
    (F::K24_8, 1, 1, 32),
    (F::K24_8Float, 1, 1, 32),
    (F::K16, 1, 1, 16),
    (F::K16_16, 1, 1, 32),
    (F::K16_16_16_16, 1, 1, 64),
    (F::K16Expand, 1, 1, 16),
    (F::K16_16Expand, 1, 1, 32),
    (F::K16_16_16_16Expand, 1, 1, 64),
    (F::K16Float, 1, 1, 16),
    (F::K16_16Float, 1, 1, 32),
    (F::K16_16_16_16Float, 1, 1, 64),
    (F::K32, 1, 1, 32),
    (F::K32_32, 1, 1, 64),
    (F::K32_32_32_32, 1, 1, 128),
    (F::K32Float, 1, 1, 32),
    (F::K32_32Float, 1, 1, 64),
    (F::K32_32_32_32Float, 1, 1, 128),
    (F::K32As8, 4, 1, 8),
    (F::K32As8_8, 2, 1, 16),
    (F::K16Mpeg, 1, 1, 16),
    (F::K16_16Mpeg, 1, 1, 32),
    (F::K8Interlaced, 1, 1, 8),
    (F::K32As8Interlaced, 4, 1, 8),
    (F::K32As8_8Interlaced, 1, 1, 16),
    (F::K16Interlaced, 1, 1, 16),
    (F::K16MpegInterlaced, 1, 1, 16),
    (F::K16_16MpegInterlaced, 1, 1, 32),
    (F::Dxn, 4, 4, 8),
    (F::K8_8_8_8As16_16_16_16, 1, 1, 32),
    (F::Dxt1As16_16_16_16, 4, 4, 4),
    (F::Dxt2_3As16_16_16_16, 4, 4, 8),
    (F::Dxt4_5As16_16_16_16, 4, 4, 8),
    (F::K2_10_10_10As16_16_16_16, 1, 1, 32),
    (F::K10_11_11As16_16_16_16, 1, 1, 32),
    (F::K11_11_10As16_16_16_16, 1, 1, 32),
    (F::K32_32_32Float, 1, 1, 96),
    (F::Dxt3A, 4, 4, 4),
    (F::Dxt5A, 4, 4, 4),
    (F::Ctx1, 4, 4, 4),
    (F::Dxt3AAs1_1_1_1, 4, 4, 4),
    (F::K8_8_8_8GammaEdram, 1, 1, 32),
    (F::K2_10_10_10FloatEdram, 1, 1, 32),
];

impl TextureFormat {
    /// The format for the low 6 bits of `bits`; every value is a format.
    pub fn from_bits(bits: u32) -> Self {
        TABLE[(bits & 63) as usize].0
    }

    pub fn info(self) -> FormatInfo {
        let (_, block_width, block_height, bits_per_pixel) = TABLE[self as usize];
        FormatInfo {
            block_width,
            block_height,
            bits_per_pixel,
        }
    }

    /// The format whose data layout this one shares. The `_AS_16_16_16_16`
    /// variants only change filtering precision; `k_8_A` samples like `k_8`
    /// and `k_8_8_8_8_A` is assumed to be `k_8_8_8_8` (both per Xenia);
    /// `k_16_EXPAND` variants are loaded as 16-bit floats by Xenia.
    pub fn data_format(self) -> Self {
        match self {
            F::K8A => F::K8,
            F::K8_8_8_8A | F::K8_8_8_8As16_16_16_16 => F::K8_8_8_8,
            F::Dxt1As16_16_16_16 => F::Dxt1,
            F::Dxt2_3As16_16_16_16 => F::Dxt2_3,
            F::Dxt4_5As16_16_16_16 => F::Dxt4_5,
            F::K2_10_10_10As16_16_16_16 => F::K2_10_10_10,
            F::K10_11_11As16_16_16_16 => F::K10_11_11,
            F::K11_11_10As16_16_16_16 => F::K11_11_10,
            F::K16Expand => F::K16Float,
            F::K16_16Expand => F::K16_16Float,
            F::K16_16_16_16Expand => F::K16_16_16_16Float,
            other => other,
        }
    }

    /// Whether [`crate::convert`] can decode this format.
    pub fn can_convert(self) -> bool {
        matches!(
            self.data_format(),
            F::K8
                | F::K1_5_5_5
                | F::K5_6_5
                | F::K6_5_5
                | F::K8_8_8_8
                | F::K2_10_10_10
                | F::K8_8
                | F::K4_4_4_4
                | F::K10_11_11
                | F::K11_11_10
                | F::Dxt1
                | F::Dxt2_3
                | F::Dxt4_5
                | F::K24_8
                | F::K24_8Float
                | F::K16
                | F::K16_16
                | F::K16_16_16_16
                | F::K16Float
                | F::K16_16Float
                | F::K16_16_16_16Float
                | F::K32Float
                | F::K32_32Float
                | F::K32_32_32_32Float
                | F::K32_32_32Float
                | F::Dxn
                | F::Dxt3A
                | F::Dxt5A
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_indexed_by_format_number() {
        for i in 0..64u32 {
            assert_eq!(TextureFormat::from_bits(i) as u32, i);
        }
        assert_eq!(TextureFormat::from_bits(64 + 18), TextureFormat::Dxt1);
    }

    #[test]
    fn block_sizes() {
        assert_eq!(F::K8_8_8_8.info().bytes_per_block(), 4);
        assert_eq!(F::Dxt1.info().bytes_per_block(), 8);
        assert_eq!(F::Dxt4_5.info().bytes_per_block(), 16);
        assert_eq!(F::Dxn.info().bytes_per_block(), 16);
        assert_eq!(F::Dxt5A.info().bytes_per_block(), 8);
        assert_eq!(F::K32_32_32Float.info().bytes_per_block(), 12);
        assert_eq!(
            F::K32As8.info(),
            FormatInfo {
                block_width: 4,
                block_height: 1,
                bits_per_pixel: 8
            }
        );
        assert!(F::Dxt4_5As16_16_16_16.can_convert());
        assert!(!F::Ctx1.can_convert());
        assert!(F::K24_8.can_convert());
        assert!(F::K24_8Float.can_convert());
    }
}
