// Tiled address formulas ported from Xenia src/xenia/gpu/texture_address.h
// (Tiled2D, Tiled3D, TiledCombine) and the extent bound from
// src/xenia/gpu/texture_util.cc GetTiledAddressUpperBound2D (BSD 3-Clause, see
// CREDITS.md).

//! The GPU's tiled texture layout.
//!
//! Tiled textures are stored in 32x32-block macro tiles (2D), with bank and
//! pipe bits interleaved into the address. All coordinates and pitches here
//! are in blocks (texels for uncompressed formats, 4x4 texel blocks for DXT).

use crate::Error;

/// Pitch and height of tiled storage are multiples of this many blocks.
pub const TILE_BLOCKS: u32 = 32;

/// Rounds a pitch or height in blocks up to the 32-block tile size.
pub fn align_to_tile(blocks: u32) -> u32 {
    blocks.next_multiple_of(TILE_BLOCKS)
}

fn tiled_combine(outer_inner_bytes: u64, bank: u64, pipe: u64, y_lsb: u64) -> u64 {
    (y_lsb << 4)
        | (pipe << 6)
        | (bank << 11)
        | (outer_inner_bytes & 0b1111)
        | (((outer_inner_bytes >> 4) & 0b1) << 5)
        | (((outer_inner_bytes >> 5) & 0b111) << 8)
        | ((outer_inner_bytes >> 8) << 12)
}

/// Byte offset of block (x, y) in a tiled 2D surface. `pitch_aligned` is in
/// blocks and must be a multiple of 32.
pub fn tiled_offset_2d(x: u32, y: u32, pitch_aligned: u32, bytes_per_block_log2: u32) -> u32 {
    debug_assert_eq!(pitch_aligned % TILE_BLOCKS, 0);
    let (x, y) = (x as u64, y as u64);
    let outer_blocks = (((y >> 5) * (pitch_aligned as u64 >> 5)) + (x >> 5)) << 6;
    let inner_blocks = (((y >> 1) & 0b111) << 3) | (x & 0b111);
    let outer_inner_bytes = (outer_blocks | inner_blocks) << bytes_per_block_log2;
    let bank = (y >> 4) & 0b1;
    let pipe = ((x >> 3) & 0b11) ^ (((y >> 3) & 0b1) << 1);
    tiled_combine(outer_inner_bytes, bank, pipe, y & 1) as u32
}

/// Byte offset of block (x, y, z) in a tiled 3D surface. `pitch_aligned` and
/// `height_aligned` are in blocks and must be multiples of 32.
pub fn tiled_offset_3d(
    x: u32,
    y: u32,
    z: u32,
    pitch_aligned: u32,
    height_aligned: u32,
    bytes_per_block_log2: u32,
) -> u64 {
    debug_assert_eq!(pitch_aligned % TILE_BLOCKS, 0);
    debug_assert_eq!(height_aligned % TILE_BLOCKS, 0);
    let (x, y, z) = (x as u64, y as u64, z as u64);
    let outer_blocks = ((((z >> 2) * (height_aligned as u64 >> 4)) + (y >> 4))
        * (pitch_aligned as u64 >> 5)
        + (x >> 5))
        << 7;
    let inner_blocks = ((z & 0b11) << 5) | (((y >> 1) & 0b11) << 3) | (x & 0b111);
    let outer_inner_bytes = (outer_blocks | inner_blocks) << bytes_per_block_log2;
    let bank = ((y >> 3) ^ (z >> 2)) & 0b1;
    let pipe = ((x >> 3) & 0b11) ^ (bank << 1);
    tiled_combine(outer_inner_bytes, bank, pipe, y & 1)
}

fn block_log2(bytes_per_block: u32) -> Result<u32, Error> {
    if bytes_per_block.is_power_of_two() && bytes_per_block <= 16 {
        Ok(bytes_per_block.trailing_zeros())
    } else {
        Err(Error::BadLayout(
            "tiled bytes per block must be 1, 2, 4, 8 or 16",
        ))
    }
}

/// Bytes a tiled 2D surface of `width` x `height` blocks touches from its
/// start (Xenia's `GetTiledAddressUpperBound2D`). `pitch_blocks` is rounded
/// up to 32.
pub fn tiled_extent_2d(
    width: u32,
    height: u32,
    pitch_blocks: u32,
    bytes_per_block: u32,
) -> Result<u32, Error> {
    let log2 = block_log2(bytes_per_block)?;
    if width == 0 || height == 0 {
        return Ok(0);
    }
    let last_tile = |v: u32| (v - 1) & !(TILE_BLOCKS - 1);
    let origin = tiled_offset_2d(
        last_tile(width),
        last_tile(height),
        align_to_tile(pitch_blocks),
        log2,
    );
    // Within the last 32x32 tile, addressing repeats every 128x128 (1 byte
    // blocks) or 64x64 (2 byte blocks), so the tile does not end at 32x32.
    Ok(origin
        + match log2 {
            0 => 0xA00,
            1 => 0xC00,
            _ => 0x400 << log2,
        })
}

fn check_pitch(width: u32, pitch_blocks: u32) -> Result<u32, Error> {
    let pitch = align_to_tile(pitch_blocks);
    if pitch < width {
        return Err(Error::BadLayout("pitch smaller than width"));
    }
    Ok(pitch)
}

/// Reads a tiled 2D surface into tightly packed rows of `width` blocks.
/// `src` starts at the surface's base; `pitch_blocks` is rounded up to 32.
pub fn detile_2d(
    src: &[u8],
    width: u32,
    height: u32,
    pitch_blocks: u32,
    bytes_per_block: u32,
) -> Result<Vec<u8>, Error> {
    let needed = tiled_extent_2d(width, height, pitch_blocks, bytes_per_block)? as usize;
    if src.len() < needed {
        return Err(Error::TooShort {
            needed,
            got: src.len(),
        });
    }
    let pitch = check_pitch(width, pitch_blocks)?;
    let log2 = bytes_per_block.trailing_zeros();
    let bpb = bytes_per_block as usize;
    let mut out = Vec::with_capacity(width as usize * height as usize * bpb);
    for y in 0..height {
        for x in 0..width {
            let at = tiled_offset_2d(x, y, pitch, log2) as usize;
            out.extend_from_slice(&src[at..at + bpb]);
        }
    }
    Ok(out)
}

/// Writes tightly packed rows of `width` blocks into the tiled surface `dst`,
/// leaving bytes no block maps to untouched (the inverse of [`detile_2d`],
/// used when the GPU resolves a render target into guest memory).
pub fn tile_2d_into(
    dst: &mut [u8],
    linear: &[u8],
    width: u32,
    height: u32,
    pitch_blocks: u32,
    bytes_per_block: u32,
) -> Result<(), Error> {
    let needed = tiled_extent_2d(width, height, pitch_blocks, bytes_per_block)? as usize;
    if dst.len() < needed {
        return Err(Error::TooShort {
            needed,
            got: dst.len(),
        });
    }
    let bpb = bytes_per_block as usize;
    let linear_needed = width as usize * height as usize * bpb;
    if linear.len() < linear_needed {
        return Err(Error::TooShort {
            needed: linear_needed,
            got: linear.len(),
        });
    }
    let pitch = check_pitch(width, pitch_blocks)?;
    let log2 = bytes_per_block.trailing_zeros();
    for (i, block) in linear[..linear_needed].chunks_exact(bpb).enumerate() {
        let (x, y) = (i as u32 % width, i as u32 / width);
        let at = tiled_offset_2d(x, y, pitch, log2) as usize;
        dst[at..at + bpb].copy_from_slice(block);
    }
    Ok(())
}

/// Like [`tile_2d_into`] into a new zero-filled buffer of exactly
/// [`tiled_extent_2d`] bytes.
pub fn tile_2d(
    linear: &[u8],
    width: u32,
    height: u32,
    pitch_blocks: u32,
    bytes_per_block: u32,
) -> Result<Vec<u8>, Error> {
    let mut dst = vec![0; tiled_extent_2d(width, height, pitch_blocks, bytes_per_block)? as usize];
    tile_2d_into(
        &mut dst,
        linear,
        width,
        height,
        pitch_blocks,
        bytes_per_block,
    )?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed with Tiled2D/TiledCombine from texture_address.h:
    ///   outer = ((y >> 5) * (pitch >> 5) + (x >> 5)) << 6
    ///   inner = (((y >> 1) & 7) << 3) | (x & 7)
    ///   oib   = (outer | inner) << log2
    ///   bank  = (y >> 4) & 1, pipe = ((x >> 3) & 3) ^ (((y >> 3) & 1) << 1)
    ///   addr  = (y&1)<<4 | pipe<<6 | bank<<11 | oib&15 | ((oib>>4)&1)<<5
    ///           | ((oib>>5)&7)<<8 | (oib>>8)<<12
    #[test]
    fn hand_computed_2d_offsets() {
        // 4-byte blocks (log2 = 2), pitch 64.
        // (1,0): inner 1, oib 4 -> 4.
        assert_eq!(tiled_offset_2d(1, 0, 64, 2), 4);
        // (0,1): inner 0, y_lsb 1 -> 1<<4 = 16.
        assert_eq!(tiled_offset_2d(0, 1, 64, 2), 16);
        // (0,2): inner 1<<3 = 8, oib 32 -> ((32>>5)&7)<<8 = 256.
        assert_eq!(tiled_offset_2d(0, 2, 64, 2), 256);
        // (8,0): inner 0, pipe 1 -> 1<<6 = 64.
        assert_eq!(tiled_offset_2d(8, 0, 64, 2), 64);
        // (0,8): inner 4<<3 = 32, oib 128 -> 4<<8 = 1024; pipe 0^(1<<1) = 2
        // -> 128; total 1152.
        assert_eq!(tiled_offset_2d(0, 8, 64, 2), 1152);
        // (0,16): inner 0 ((8&7)<<3), bank 1 -> 2048; pipe 0^((2&1)<<1) = 0.
        assert_eq!(tiled_offset_2d(0, 16, 64, 2), 2048);
        // (32,0): outer 1<<6 = 64, oib 256 -> (256>>8)<<12 = 4096; pipe
        // (4&3) = 0.
        assert_eq!(tiled_offset_2d(32, 0, 64, 2), 4096);
        // (0,32) with pitch 64: outer (1*2)<<6 = 128, oib 512 -> 2<<12 = 8192.
        assert_eq!(tiled_offset_2d(0, 32, 64, 2), 8192);
        // (3,5) with 1-byte blocks: inner (2<<3)|3 = 19, oib 19 -> 3 |
        // (1<<5) = 35, y_lsb 1 -> +16 = 51.
        assert_eq!(tiled_offset_2d(3, 5, 32, 0), 51);
        // (9,3) with 8-byte blocks: inner (1<<3)|1 = 9, oib 72 -> 8 |
        // ((72>>4)&1 = 0) | ((72>>5)&7 = 2)<<8 = 512 -> 520; pipe 1 -> +64;
        // y_lsb 1 -> +16; total 600.
        assert_eq!(tiled_offset_2d(9, 3, 32, 3), 600);
    }

    /// Hand-computed with Tiled3D: outer = (((z>>2)*(h>>4) + (y>>4)) *
    /// (pitch>>5) + (x>>5)) << 7, inner = ((z&3)<<5) | (((y>>1)&3)<<3) | (x&7),
    /// bank = ((y>>3) ^ (z>>2)) & 1, pipe = ((x>>3)&3) ^ (bank<<1).
    #[test]
    fn hand_computed_3d_offsets() {
        // (0,0,1), 4-byte blocks: inner 32, oib 128 -> 4<<8 = 1024.
        assert_eq!(tiled_offset_3d(0, 0, 1, 32, 32, 2), 1024);
        // (0,8,0): bank 1 -> 2048, pipe 0^2 = 2 -> 128; inner 0; total 2176.
        assert_eq!(tiled_offset_3d(0, 8, 0, 32, 32, 2), 2176);
        // (0,0,4): bank (0^1)&1 = 1, pipe 2; outer (1*2)<<7 = 256, oib 1024
        // -> 4<<12 = 16384; total 16384 + 2048 + 128 = 18560.
        assert_eq!(tiled_offset_3d(0, 0, 4, 32, 32, 2), 18560);
    }

    #[test]
    fn extent_bounds() {
        // One tile of 4-byte blocks: 0x400 << 2 = 4096.
        assert_eq!(tiled_extent_2d(32, 32, 32, 4).unwrap(), 4096);
        assert_eq!(tiled_extent_2d(1, 1, 32, 1).unwrap(), 0xA00);
        assert_eq!(tiled_extent_2d(1, 1, 32, 2).unwrap(), 0xC00);
        // Second tile row with pitch 64 starts at 8192.
        assert_eq!(
            tiled_extent_2d(64, 33, 64, 4).unwrap(),
            8192 + 0x400 * 4 + 4096
        );
        assert_eq!(tiled_extent_2d(0, 5, 32, 4).unwrap(), 0);
        assert!(tiled_extent_2d(1, 1, 32, 12).is_err());
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn round_trip_various_sizes() {
        for &(w, h, pitch, bpb) in &[
            (32, 32, 32, 4),
            (37, 5, 64, 4),
            (100, 70, 128, 1),
            (65, 33, 96, 2),
            (3, 50, 7, 8),
            (20, 9, 32, 16),
        ] {
            let linear = pattern((w * h * bpb) as usize);
            let tiled = tile_2d(&linear, w, h, pitch, bpb).unwrap();
            assert_eq!(
                detile_2d(&tiled, w, h, pitch, bpb).unwrap(),
                linear,
                "{w}x{h} bpb {bpb}"
            );
        }
    }

    #[test]
    fn every_block_lands_in_a_distinct_place() {
        // 4-byte blocks over a full 64x64 surface: offsets are a permutation
        // of all block slots.
        let mut seen = vec![false; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                let at = tiled_offset_2d(x, y, 64, 2) as usize;
                assert_eq!(at % 4, 0);
                assert!(!seen[at / 4], "({x},{y}) collides");
                seen[at / 4] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn tile_into_keeps_other_bytes_and_checks_sizes() {
        let mut dst = vec![0xEEu8; 4096];
        tile_2d_into(&mut dst, &[1, 2, 3, 4], 1, 1, 32, 4).unwrap();
        assert_eq!(&dst[..5], &[1, 2, 3, 4, 0xEE]);
        assert!(matches!(
            detile_2d(&[0; 10], 1, 1, 32, 4),
            Err(Error::TooShort { .. })
        ));
        assert!(matches!(
            tile_2d(&[0; 3], 1, 1, 32, 4),
            Err(Error::TooShort { .. })
        ));
        assert_eq!(
            detile_2d(&[0; 8192], 40, 1, 32, 4),
            Err(Error::BadLayout("pitch smaller than width"))
        );
    }
}
