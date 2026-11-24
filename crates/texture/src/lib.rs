// Base-level extent math ported from Xenia src/xenia/gpu/texture_util.cc
// (GetGuestTextureLayout) (BSD 3-Clause, see CREDITS.md).

//! `xenon-texture`: turns the console GPU's textures, as they sit in guest
//! memory, into linear RGBA that a host renderer can upload.
//!
//! Pure functions on byte slices: callers read the guest bytes themselves
//! (see [`bytes_needed`]) and pass them in. Pipeline for one base level:
//!
//! 1. [`TextureFetch::decode`] the six fetch-constant words,
//! 2. [`endian::swap_in_place`] per the fetch constant's endianness,
//! 3. [`tile::detile_2d`] (tiled) or a pitch-linear row copy,
//! 4. [`convert::to_rgba8`] per [`TextureFormat`],
//! 5. [`convert::apply_swizzle`] with the fetch constant's swizzle.
//!
//! [`decode_base_level`] does all five. [`tile::tile_2d`] is the inverse
//! layout step, for writing resolved render targets back to guest memory.

pub mod convert;
pub mod endian;
pub mod fetch;
pub mod format;
pub mod tile;

use std::fmt;

pub use convert::{apply_swizzle, to_rgba8, to_rgba_f32};
pub use endian::Endian;
pub use fetch::{Dimension, TextureFetch};
pub use format::{FormatInfo, TextureFormat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No decoder for this format (see [`TextureFormat::can_convert`]).
    UnsupportedFormat(TextureFormat),
    /// Only 1D and 2D (including slice 0 of stacked and cube) are decoded.
    UnsupportedDimension(Dimension),
    /// The input slice is shorter than the layout requires.
    TooShort { needed: usize, got: usize },
    /// The sizes or pitch do not describe a valid layout.
    BadLayout(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedFormat(format) => write!(f, "unsupported texture format {format:?}"),
            Error::UnsupportedDimension(dim) => write!(f, "unsupported texture dimension {dim:?}"),
            Error::TooShort { needed, got } => {
                write!(f, "texture data too short: need {needed} bytes, got {got}")
            }
            Error::BadLayout(why) => write!(f, "bad texture layout: {why}"),
        }
    }
}

impl std::error::Error for Error {}

/// A decoded base level: `rgba8` is `width * height * 4` bytes, rows top to
/// bottom, R G B A per texel, after the fetch constant's swizzle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Base-level geometry in blocks, shared by [`bytes_needed`] and
/// [`decode_base_level`].
struct BaseLayout {
    width_blocks: u32,
    height_blocks: u32,
    pitch_blocks: u32,
    bytes_per_block: u32,
    /// Bytes of array slice 0 that the level touches.
    slice_extent: usize,
    /// Bytes of all array slices (stacked / cube faces) that the level touches.
    total_extent: usize,
}

// Extent math ported from Xenia src/xenia/gpu/texture_util.cc
// GetGuestTextureLayout (BSD 3-Clause, see CREDITS.md), base level only.
fn base_layout(fetch: &TextureFetch) -> Result<BaseLayout, Error> {
    let (width, height, array_size) = match fetch.dimension {
        Dimension::D1 => (fetch.width(), 1, 1),
        Dimension::D2OrStacked | Dimension::Cube => (fetch.width(), fetch.height(), fetch.depth()),
        Dimension::D3 => return Err(Error::UnsupportedDimension(Dimension::D3)),
    };
    let info = fetch.format.info();
    let bytes_per_block = info.bytes_per_block();
    let width_blocks = width.div_ceil(info.block_width);
    let height_blocks = height.div_ceil(info.block_height);
    let pitch_blocks = tile::align_to_tile(fetch.pitch_texels().div_ceil(info.block_width));
    if pitch_blocks < width_blocks {
        return Err(Error::BadLayout("pitch smaller than width"));
    }
    let row_pitch_bytes = pitch_blocks as usize * bytes_per_block as usize;
    let slice_extent = if fetch.tiled {
        tile::tiled_extent_2d(width_blocks, height_blocks, pitch_blocks, bytes_per_block)? as usize
    } else {
        row_pitch_bytes * (height_blocks as usize - 1)
            + bytes_per_block as usize * width_blocks as usize
    };
    let slice_rows = match fetch.dimension {
        Dimension::D1 => 1,
        _ => tile::align_to_tile(height_blocks) as usize,
    };
    let slice_stride = (row_pitch_bytes * slice_rows).next_multiple_of(4096);
    Ok(BaseLayout {
        width_blocks,
        height_blocks,
        pitch_blocks,
        bytes_per_block,
        slice_extent,
        total_extent: slice_stride * (array_size as usize - 1) + slice_extent,
    })
}

/// Guest bytes of the base level, starting at [`TextureFetch::base_address`],
/// covering every array slice / cube face. Not including mips.
pub fn bytes_needed(fetch: &TextureFetch) -> Result<usize, Error> {
    Ok(base_layout(fetch)?.total_extent)
}

/// Decodes array slice 0 of the base level: endian swap, detile or
/// pitch-linear copy, format conversion, swizzle. `guest_bytes` start at the
/// base address; only slice 0's extent must be present.
///
/// Signedness, `exp_adjust` and gamma are ignored: texels decode as unsigned
/// normalized (or float) values.
pub fn decode_base_level(fetch: &TextureFetch, guest_bytes: &[u8]) -> Result<Image, Error> {
    let layout = base_layout(fetch)?;
    if !fetch.format.can_convert() {
        return Err(Error::UnsupportedFormat(fetch.format));
    }
    if guest_bytes.len() < layout.slice_extent {
        return Err(Error::TooShort {
            needed: layout.slice_extent,
            got: guest_bytes.len(),
        });
    }
    let mut raw = guest_bytes[..layout.slice_extent].to_vec();
    endian::swap_in_place(&mut raw, fetch.endian);
    let row_bytes = (layout.width_blocks * layout.bytes_per_block) as usize;
    let linear = if fetch.tiled {
        tile::detile_2d(
            &raw,
            layout.width_blocks,
            layout.height_blocks,
            layout.pitch_blocks,
            layout.bytes_per_block,
        )?
    } else {
        let pitch_bytes = (layout.pitch_blocks * layout.bytes_per_block) as usize;
        let mut out = Vec::with_capacity(row_bytes * layout.height_blocks as usize);
        for row in raw.chunks(pitch_bytes).take(layout.height_blocks as usize) {
            out.extend_from_slice(&row[..row_bytes]);
        }
        out
    };
    let (width, height) = (fetch.width(), fetch.height());
    let mut rgba8 = to_rgba8(fetch.format, &linear, width, height)?;
    apply_swizzle(&mut rgba8, fetch.swizzle);
    Ok(Image {
        width,
        height,
        rgba8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds fetch words for a 2D texture.
    fn fetch_words(
        format: TextureFormat,
        endian: Endian,
        tiled: bool,
        width: u32,
        height: u32,
        pitch_texels: u32,
        swizzle: u16,
    ) -> [u32; 6] {
        [
            2 | ((pitch_texels >> 5) << 22) | ((tiled as u32) << 31),
            format as u32 | ((endian as u32) << 6) | (0x1234 << 12),
            (width - 1) | ((height - 1) << 13),
            (swizzle as u32) << 1,
            0,
            1 << 9,
        ]
    }

    const RGBA: u16 = (1 << 3) | (2 << 6) | (3 << 9);

    #[test]
    fn linear_8888_with_endian_and_swizzle() {
        // 2x2 texture, pitch 32 texels -> 128-byte rows. Guest words are
        // big-endian 0xAARRGGBB with k8in32; swizzle ZYXW makes it RGBA.
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::K8in32,
            false,
            2,
            2,
            32,
            2 | (1 << 3) | (3 << 9), // Z Y X W
        ));
        assert_eq!(fetch.base_address, 0x0123_4000);
        let needed = bytes_needed(&fetch).unwrap();
        assert_eq!(needed, 128 + 8);
        let mut guest = vec![0u8; needed];
        let words = [0xFF10_2030u32, 0x8040_5060, 0x0170_8090, 0x02A0_B0C0];
        for (i, w) in words.iter().enumerate() {
            let off = (i / 2) * 128 + (i % 2) * 4;
            guest[off..off + 4].copy_from_slice(&w.to_be_bytes());
        }
        let img = decode_base_level(&fetch, &guest).unwrap();
        assert_eq!((img.width, img.height), (2, 2));
        assert_eq!(
            img.rgba8,
            [
                0x10, 0x20, 0x30, 0xFF, 0x40, 0x50, 0x60, 0x80, //
                0x70, 0x80, 0x90, 0x01, 0xA0, 0xB0, 0xC0, 0x02
            ]
        );
    }

    #[test]
    fn tiled_565_round_trip_through_tile_2d() {
        // 40x3 texels of k_5_6_5, pitch 64. Build the guest bytes by tiling
        // big-endian 16-bit texels, then decode with k8in16.
        let (w, h) = (40u32, 3u32);
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::K5_6_5,
            Endian::K8in16,
            true,
            w,
            h,
            64,
            RGBA,
        ));
        let mut linear = Vec::new();
        let mut expect = Vec::new();
        for y in 0..h {
            for x in 0..w {
                // Red = x & 31, green = y, blue = 31 - (x & 31).
                let (r, g, b) = (x & 31, y, 31 - (x & 31));
                let v = (r | (g << 5) | (b << 11)) as u16;
                linear.extend_from_slice(&v.to_be_bytes());
                let e = |c: u32, bits: u32| {
                    ((c as f32 / ((1 << bits) - 1) as f32) * 255.0).round() as u8
                };
                let bb = e(b, 5);
                expect.extend_from_slice(&[e(r, 5), e(g, 6), bb, bb]);
            }
        }
        let guest = tile::tile_2d(&linear, w, h, 64, 2).unwrap();
        assert_eq!(guest.len(), bytes_needed(&fetch).unwrap());
        let img = decode_base_level(&fetch, &guest).unwrap();
        assert_eq!(img.rgba8, expect);
    }

    #[test]
    fn tiled_dxt1_decodes() {
        // 8x4 texels = 2x1 DXT1 blocks. Block 0 solid white, block 1 solid
        // pure red (color0 = 0xF800, all indices 0). Guest blocks stored with
        // 16-bit words big-endian (k8in16).
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::Dxt1,
            Endian::K8in16,
            true,
            8,
            4,
            128,
            RGBA,
        ));
        let blocks: [[u16; 4]; 2] = [[0xFFFF, 0xFFFF, 0, 0], [0xF800, 0x0000, 0, 0]];
        let mut linear = Vec::new();
        for b in blocks {
            for v in b {
                linear.extend_from_slice(&v.to_be_bytes());
            }
        }
        let guest = tile::tile_2d(&linear, 2, 1, 32, 8).unwrap();
        let img = decode_base_level(&fetch, &guest).unwrap();
        assert_eq!(&img.rgba8[0..4], &[255, 255, 255, 255]);
        assert_eq!(&img.rgba8[4 * 4..4 * 4 + 4], &[255, 0, 0, 255]);
        assert_eq!(&img.rgba8[(3 * 8 + 7) * 4..], &[255, 0, 0, 255]);
    }

    #[test]
    fn short_input_and_3d_are_errors() {
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::K8,
            Endian::None,
            false,
            4,
            4,
            32,
            RGBA,
        ));
        assert_eq!(
            decode_base_level(&fetch, &[0; 10]),
            Err(Error::TooShort {
                needed: 3 * 32 + 4,
                got: 10
            })
        );
        let mut words = fetch_words(TextureFormat::K8, Endian::None, false, 4, 4, 32, RGBA);
        words[5] = 2 << 9;
        let fetch3d = TextureFetch::decode(words);
        assert_eq!(
            bytes_needed(&fetch3d),
            Err(Error::UnsupportedDimension(Dimension::D3))
        );
    }

    #[test]
    fn stacked_bytes_needed_counts_slices() {
        // 16x16 k_8_8_8_8 linear, pitch 32, 3 slices: slice stride is
        // 32 blocks * 4 bytes * 32 rows = 4096; slice extent 15*128 + 64.
        let mut words = fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::None,
            false,
            16,
            16,
            32,
            RGBA,
        );
        words[1] |= 1 << 10; // stacked
        words[2] |= 2 << 26; // stack_depth - 1
        let fetch = TextureFetch::decode(words);
        assert_eq!(fetch.depth(), 3);
        assert_eq!(bytes_needed(&fetch).unwrap(), 2 * 4096 + 15 * 128 + 64);
    }
}
