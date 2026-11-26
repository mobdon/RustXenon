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

pub use convert::{apply_swizzle, apply_swizzle_f32, to_rgba8, to_rgba_f32, to_rgba_f32_signed};
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

/// A decoded base level as floats: `rgba` is `width * height * 4` values,
/// rows top to bottom, R G B A per texel, after the fetch constant's swizzle.
/// Float formats are unclamped and depth keeps its 24 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageF32 {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f32>,
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
    // A single linear row never steps by the pitch (Xenia's extent ignores it
    // then too), so only multi-row or tiled layouts need a pitch that fits.
    if pitch_blocks < width_blocks && (fetch.tiled || height_blocks > 1) {
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
/// Signedness, `exp_adjust` and gamma are not applied here: texels decode as
/// unsigned normalized (or float) values. Signed components need
/// [`decode_base_level_f32`]; the translated shader applies the biased sign,
/// gamma and `exp_adjust` to the sampled value (see `xenon-shader`).
pub fn decode_base_level(fetch: &TextureFetch, guest_bytes: &[u8]) -> Result<Image, Error> {
    let linear = linear_base_level(fetch, guest_bytes)?;
    let (width, height) = (fetch.width(), fetch.height());
    let mut rgba8 = to_rgba8(fetch.format, &linear, width, height)?;
    apply_swizzle(&mut rgba8, fetch.swizzle);
    Ok(Image {
        width,
        height,
        rgba8,
    })
}

/// [`decode_base_level`] to floats, for formats whose values do not fit 8-bit
/// unorm (depth, 16- and 32-bit, float) and for fetches with signed
/// components. Components whose fetch sign is 1 decode as signed normalized
/// ([`to_rgba_f32_signed`]) before the swizzle, so each output component
/// takes the sign of the source component it selects and constant 0 / 1
/// selects stay unsigned. Decoding before sampling lets the host filter the
/// signed values.
pub fn decode_base_level_f32(fetch: &TextureFetch, guest_bytes: &[u8]) -> Result<ImageF32, Error> {
    let linear = linear_base_level(fetch, guest_bytes)?;
    let (width, height) = (fetch.width(), fetch.height());
    let signed = fetch.sign.map(|s| s == 1);
    let mut rgba = to_rgba_f32_signed(fetch.format, &linear, width, height, signed)?;
    apply_swizzle_f32(&mut rgba, fetch.swizzle);
    Ok(ImageF32 {
        width,
        height,
        rgba,
    })
}

/// Mip level `level` (1 and up) of a 2D mip chain, below its packed tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLevel {
    /// Size in texels: the base size halved `level` times, at least 1.
    pub width: u32,
    pub height: u32,
    /// Row pitch in blocks (a multiple of 32 when tiled).
    pub pitch_blocks: u32,
    /// Byte offset of the level from [`TextureFetch::mip_address`].
    pub offset: usize,
    /// Bytes from `offset` that the level touches.
    pub extent: usize,
}

/// Layout of mip `level` of `fetch` (level >= 1) per Xenia's
/// GetGuestTextureLayout: each mip's strides come from the base size
/// rounded up to a power of two, halved, and aligned to 32x32 blocks
/// (linear rows to 256 bytes); each level starts on a 4 KB boundary after
/// the one before. Only single-slice 2D chains; with packed mips, only the
/// levels before `packed_mip_level` (the tail's levels are an error).
pub fn mip_level(fetch: &TextureFetch, level: u32) -> Result<MipLevel, Error> {
    if fetch.dimension != Dimension::D2OrStacked || fetch.stacked {
        return Err(Error::UnsupportedDimension(fetch.dimension));
    }
    if level == 0 || level > 13 {
        return Err(Error::BadLayout("mip level out of range"));
    }
    if fetch.packed_mips && level >= packed_mip_level(fetch.width(), fetch.height()) {
        return Err(Error::BadLayout("packed mip tail"));
    }
    let info = fetch.format.info();
    let bpb = info.bytes_per_block() as usize;
    let (w, h) = (fetch.width(), fetch.height());
    let (pw, ph) = (w.next_power_of_two(), h.next_power_of_two());
    let geometry = |l: u32| {
        let pitch_blocks = tile::align_to_tile((pw >> l).max(1).div_ceil(info.block_width));
        let mut row_bytes = pitch_blocks as usize * bpb;
        if !fetch.tiled {
            row_bytes = row_bytes.next_multiple_of(256);
        }
        let rows = tile::align_to_tile((ph >> l).max(1).div_ceil(info.block_height)) as usize;
        (
            pitch_blocks,
            row_bytes,
            (row_bytes * rows).next_multiple_of(4096),
        )
    };
    let offset = (1..level).map(|l| geometry(l).2).sum();
    let (pitch_blocks, row_bytes, _) = geometry(level);
    let (width, height) = ((w >> level).max(1), (h >> level).max(1));
    let (wb, hb) = (
        width.div_ceil(info.block_width),
        height.div_ceil(info.block_height),
    );
    let extent = if fetch.tiled {
        tile::tiled_extent_2d(wb, hb, pitch_blocks, bpb as u32)? as usize
    } else {
        row_bytes * (hb as usize - 1) + bpb * wb as usize
    };
    Ok(MipLevel {
        width,
        height,
        pitch_blocks: if fetch.tiled {
            pitch_blocks
        } else {
            (row_bytes / bpb) as u32
        },
        offset,
        extent,
    })
}

/// Mip `level` of `fetch` as a fetch whose base level it is: its size,
/// pitch and address (`mip_address` plus the level's offset), no mips. So
/// [`bytes_needed`] and the base level decoders read the level.
pub fn mip_fetch(fetch: &TextureFetch, level: u32) -> Result<TextureFetch, Error> {
    let m = mip_level(fetch, level)?;
    Ok(TextureFetch {
        // Mip pitches are whole tiles of blocks, so whole 32-texel units.
        pitch: (m.pitch_blocks * fetch.format.info().block_width) >> 5,
        base_address: fetch.mip_address.wrapping_add(m.offset as u32),
        size: (m.width - 1) | ((m.height - 1) << 13),
        mip_min_level: 0,
        mip_max_level: 0,
        mip_address: 0,
        ..*fetch
    })
}

/// [`mip_fetch`] of each level after the base that `fetch` samples, in
/// order: 1 up to the last of [`TextureFetch::mip_range`], stopping before
/// the first level [`mip_level`] does not lay out (the packed mip tail).
pub fn mip_fetches(fetch: &TextureFetch) -> impl Iterator<Item = TextureFetch> + '_ {
    (1..=fetch.mip_range().1).map_while(|l| mip_fetch(fetch, l).ok())
}

/// First level of a `width` x `height` chain stored in the packed mip tail,
/// as Xenia's GetPackedMipLevel: the levels whose shorter side is 16 or
/// less. Levels before it keep the unpacked layout.
fn packed_mip_level(width: u32, height: u32) -> u32 {
    width
        .min(height)
        .next_power_of_two()
        .ilog2()
        .saturating_sub(4)
}

/// Endian swap and detile (or pitch-linear copy) of slice 0's base level to
/// tightly packed rows of blocks.
fn linear_base_level(fetch: &TextureFetch, guest_bytes: &[u8]) -> Result<Vec<u8>, Error> {
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
    if fetch.tiled {
        return tile::detile_2d(
            &raw,
            layout.width_blocks,
            layout.height_blocks,
            layout.pitch_blocks,
            layout.bytes_per_block,
        );
    }
    let pitch_bytes = (layout.pitch_blocks * layout.bytes_per_block) as usize;
    let mut out = Vec::with_capacity(row_bytes * layout.height_blocks as usize);
    for row in 0..layout.height_blocks as usize {
        out.extend_from_slice(&raw[row * pitch_bytes..][..row_bytes]);
    }
    Ok(out)
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
    fn unpacked_mip_chain_layout() {
        // 256x256 tiled k_16_16_16_16_EXPAND, mips 0-3 right after the base.
        let mut words = fetch_words(
            TextureFormat::K16_16_16_16Expand,
            Endian::K8in16,
            true,
            256,
            256,
            256,
            RGBA,
        );
        words[4] = 3 << 6;
        words[5] |= 0x0008_0000;
        let fetch = TextureFetch::decode(words);
        assert_eq!(fetch.mip_range(), (0, 3));
        let levels: Vec<_> = (1..=3).map(|l| mip_level(&fetch, l).unwrap()).collect();
        let at = |l: &MipLevel| (l.width, l.height, l.pitch_blocks, l.offset);
        assert_eq!(at(&levels[0]), (128, 128, 128, 0));
        assert_eq!(at(&levels[1]), (64, 64, 64, 0x20000));
        assert_eq!(at(&levels[2]), (32, 32, 32, 0x28000));
        assert_eq!(levels[0].extent, 128 * 128 * 8);
        assert_eq!(levels[2].extent, 32 * 32 * 8);
        // mip_min_level 1, mip_max_level 1.
        let one = TextureFetch::decode([
            words[0],
            words[1],
            words[2],
            words[3],
            (1 << 2) | (1 << 6),
            words[5],
        ]);
        assert_eq!(one.mip_range(), (1, 1));
        // No mip address: base only, whatever the levels say.
        let none = TextureFetch::decode([words[0], words[1], words[2], words[3], 3 << 6, 1 << 9]);
        assert_eq!(none.mip_range(), (0, 0));
        // Only the physical page counts: a mip address in the 512 MB mirror
        // of page 0 is none either.
        let mirror = [
            words[0],
            words[1],
            words[2],
            words[3],
            3 << 6,
            0x2000_0000 | (1 << 9),
        ];
        assert_eq!(TextureFetch::decode(mirror).mip_range(), (0, 0));
        // No base page: sampling starts at level 1, unless there are no mips.
        let no_base = |w4| {
            TextureFetch::decode([words[0], words[1] & 0xFFF, words[2], words[3], w4, words[5]])
        };
        assert_eq!(no_base(3 << 6).mip_range(), (1, 3));
        assert_eq!(no_base(0).mip_range(), (0, 0));
        // A non-power-of-two base: strides from 512x256, sizes from 300x200;
        // linear rows of 256 bytes at least.
        let mut w = fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::None,
            false,
            300,
            200,
            320,
            RGBA,
        );
        w[5] |= 0x1000;
        let l1 = mip_level(&TextureFetch::decode(w), 1).unwrap();
        assert_eq!(
            (l1.width, l1.height, l1.pitch_blocks, l1.offset),
            (150, 100, 256, 0)
        );
        let l2 = mip_level(&TextureFetch::decode(w), 2).unwrap();
        assert_eq!(
            (l2.width, l2.pitch_blocks, l2.offset),
            (75, 128, 1024 * 128)
        );
        // Packed mips: 200 rounds up to 256, so levels 1-3 keep the layout
        // above and level 4 (18x12) is in the tail.
        w[5] |= 1 << 11;
        let packed = TextureFetch::decode(w);
        assert_eq!(mip_level(&packed, 1).unwrap(), l1);
        assert_eq!(mip_level(&packed, 2).unwrap(), l2);
        assert!(mip_level(&packed, 3).is_ok());
        assert!(mip_level(&packed, 4).is_err());
        assert_eq!(
            [(256, 256), (64, 64), (17, 300), (16, 16), (1, 1)]
                .map(|(w, h)| packed_mip_level(w, h)),
            [4, 2, 1, 0, 0]
        );
    }

    #[test]
    fn mip_fetches_decode_each_level() {
        // 64x64 tiled k_8_8_8_8 with mip_max_level 3, unpacked: levels 1-3
        // (32x32 to 8x8), each a pattern of its own, decode from the mip page.
        let texel = |l: u32, x: u32, y: u32| [l as u8 * 60, x as u8, y as u8, 255];
        let mut words = fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::None,
            true,
            64,
            64,
            64,
            RGBA,
        );
        words[4] = 3 << 6;
        words[5] |= 0x0010_0000;
        let fetch = TextureFetch::decode(words);
        let chain: Vec<_> = mip_fetches(&fetch).collect();
        assert_eq!(chain.len(), 3);
        let mut mip_page = Vec::new();
        for (l, m) in (1..).zip(&chain) {
            assert_eq!((m.width(), m.height()), (64 >> l, 64 >> l));
            assert_eq!(m.mip_range(), (0, 0));
            let n = 64 >> l;
            let linear: Vec<u8> = (0..n * n).flat_map(|i| texel(l, i % n, i / n)).collect();
            let tiled = tile::tile_2d(&linear, n, n, m.pitch_texels(), 4).unwrap();
            let at = (m.base_address - fetch.mip_address) as usize;
            assert_eq!(at, mip_level(&fetch, l).unwrap().offset);
            mip_page.resize(at, 0);
            mip_page.extend_from_slice(&tiled);
        }
        for (l, m) in (1..).zip(&chain) {
            let at = (m.base_address - fetch.mip_address) as usize;
            let img = decode_base_level(m, &mip_page[at..]).unwrap();
            let n = 64 >> l;
            let expect: Vec<u8> = (0..n * n).flat_map(|i| texel(l, i % n, i / n)).collect();
            assert_eq!(img.rgba8, expect, "level {l}");
        }
        // Packed mips: 64 packs from level 2, so only level 1 is listed.
        words[5] |= 1 << 11;
        assert_eq!(mip_fetches(&TextureFetch::decode(words)).count(), 1);
        // No mip page: none.
        words[5] &= 0xFFF;
        assert_eq!(mip_fetches(&TextureFetch::decode(words)).count(), 0);
    }

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

    #[test]
    fn linear_1d_ignores_pitch() {
        // A 1D texture is one row, so its pitch field is never stepped by;
        // pitch 0 must still decode (Xenia's extent for one row ignores it).
        let words = [
            2,
            TextureFormat::K8 as u32,
            47, // width 48
            (RGBA as u32) << 1,
            0,
            0, // dimension 1D
        ];
        let fetch = TextureFetch::decode(words);
        assert_eq!(fetch.dimension, Dimension::D1);
        assert_eq!(bytes_needed(&fetch).unwrap(), 48);
        let guest: Vec<u8> = (0..48).collect();
        let img = decode_base_level(&fetch, &guest).unwrap();
        assert_eq!((img.width, img.height), (48, 1));
        assert_eq!(&img.rgba8[47 * 4..], &[47; 4]);
    }

    #[test]
    fn f32_decode_keeps_depth_and_hdr() {
        // 2x1 k_24_8_FLOAT, guest words big-endian (k8in32), swizzle RRRR
        // after the decode's XXXX gives the same depth in every channel.
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::K24_8Float,
            Endian::K8in32,
            false,
            2,
            1,
            32,
            RGBA,
        ));
        let mut guest = Vec::new();
        for d in [(14u32 << 20) | (1 << 19), 15 << 20] {
            guest.extend_from_slice(&((d << 8) | 0xFF).to_be_bytes());
        }
        let img = decode_base_level_f32(&fetch, &guest).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgba, [0.75, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0]);

        // k_16_16_16_16_FLOAT above 1 stays unclamped; swizzle WZYX.
        let fetch = TextureFetch::decode(fetch_words(
            TextureFormat::K16_16_16_16Float,
            Endian::None,
            false,
            1,
            1,
            32,
            3 | (2 << 3) | (1 << 6),
        ));
        // 4.0, -2.0, 0.5, 1.0
        let guest: Vec<u8> = [0x4400u16, 0xC000, 0x3800, 0x3C00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let img = decode_base_level_f32(&fetch, &guest).unwrap();
        assert_eq!(img.rgba, [1.0, 0.5, -2.0, 4.0]);
    }

    #[test]
    fn f32_decode_signs_components_before_swizzle() {
        // 1x1 k_8_8_8_8, bytes X Y Z W = 0x80 0x7F 0x01 0xFF. X and Y are
        // signed, Z and W unsigned. Swizzle (Y, X, 1, Z): outputs 0 and 1
        // take the signs of sources Y and X, the constant stays 1, output 3
        // is the unsigned Z.
        let mut w = fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::None,
            false,
            1,
            1,
            32,
            1 | (5 << 6) | (2 << 9),
        );
        w[0] |= (1 << 2) | (1 << 4);
        let fetch = TextureFetch::decode(w);
        let img = decode_base_level_f32(&fetch, &[0x80, 0x7F, 0x01, 0xFF]).unwrap();
        assert_eq!(img.rgba, [1.0, -1.0, 1.0, 1.0 / 255.0]);

        // Constant 0 with a signed source keeps 0; signed 0xFF is -1/127.
        let mut w = fetch_words(
            TextureFormat::K8_8_8_8,
            Endian::None,
            false,
            1,
            1,
            32,
            4 | (3 << 3) | (4 << 6) | (4 << 9),
        );
        w[0] |= 0x55 << 2;
        let img = decode_base_level_f32(&TextureFetch::decode(w), &[0, 0, 0, 0xFF]).unwrap();
        assert_eq!(img.rgba, [0.0, -1.0 / 127.0, 0.0, 0.0]);

        // k_16_16 signed reads the 16-bit fields exactly; Y repeats into Z, W.
        let mut w = fetch_words(TextureFormat::K16_16, Endian::None, false, 1, 1, 32, RGBA);
        w[0] |= 0x55 << 2;
        let mut guest = 0x8001u16.to_le_bytes().to_vec();
        guest.extend_from_slice(&0x4000u16.to_le_bytes());
        let img = decode_base_level_f32(&TextureFetch::decode(w), &guest).unwrap();
        let y = 16384.0 / 32767.0;
        assert_eq!(img.rgba, [-1.0, y, y, y]);
    }

    #[test]
    fn random_fetch_words_do_not_panic() {
        // xorshift32; half the cases get small sizes so decoding really runs.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let guest = vec![0xA5u8; 1 << 16];
        for i in 0..20_000 {
            let mut words = [next(), next(), next(), next(), next(), next()];
            if i % 2 == 0 {
                words[2] &= 0x000F_E01F; // width, height up to 32, depth up to 1
                words[0] &= !(0x1FF << 22);
                words[0] |= (next() & 3) << 22; // pitch 0..96
            }
            let fetch = TextureFetch::decode(words);
            let _ = bytes_needed(&fetch);
            let _ = decode_base_level(&fetch, &guest);
        }
    }
}
