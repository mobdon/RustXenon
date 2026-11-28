//! Guest textures decoded on the GPU from their raw guest bytes (detiled,
//! endian swapped, converted and swizzled there) upload the same texels,
//! level by level, as the CPU decode (`decode_level`), the reference.

mod common;

use common::*;

const IDENTITY: u16 = (1 << 3) | (2 << 6) | (3 << 9);
/// B, G, R, constant 1.
const BGR1: u16 = 2 | (1 << 3) | (5 << 9);
/// Constant 0, X, constant 1, Y.
const ZX1Y: u16 = 4 | (5 << 6) | (1 << 9);

/// Fetch constant of a 64x64 texture of `format`, tiled or linear, with
/// mip levels `min` to 6 (32x32 down to 1x1) after the base.
fn fetch(format: u32, endian: u32, swizzle: u16, tiled: bool, min: u32) -> [u32; 6] {
    [
        2 | (2 << 22) | (u32::from(tiled) << 31), // pitch 64
        format | (endian << 6) | TEXTURE,
        63 | (63 << 13),
        u32::from(swizzle) << 1,
        (min << 2) | (6 << 6),
        DEST | (1 << 9), // mip address, 2D
    ]
}

/// Pseudo-random guest bytes for the base and every mip level of `c`.
fn guest_bytes(c: [u32; 6]) -> Vec<u8> {
    let f = xenon_texture::TextureFetch::decode(c);
    let len = xenon_texture::bytes_needed(&f).unwrap()
        + xenon_texture::mip_fetches(&f)
            .map(|m| xenon_texture::bytes_needed(&m).unwrap())
            .sum::<usize>();
    let mut seed = 0x9E37_79B9u32 ^ c[1];
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        })
        .collect()
}

/// Decodes `c` with GPU decoding on and off and checks both upload the
/// same texels; returns whether the first decoded on the GPU and its
/// level count.
fn same_texels(c: [u32; 6], float32: bool) -> Option<(bool, usize)> {
    let mut d = drawer()?;
    let bytes = guest_bytes(c);
    d.gpu_texture_decode = true;
    let (host, gpu, on_gpu) = d.decode_guest_texture(c, &bytes, float32).unwrap();
    d.gpu_texture_decode = false;
    let (cpu_host, cpu, on_cpu) = d.decode_guest_texture(c, &bytes, float32).unwrap();
    assert!(!on_cpu);
    assert_eq!(host, cpu_host, "{c:08x?}");
    assert_eq!(gpu.len(), cpu.len(), "{c:08x?}");
    for (level, (g, c2)) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(g.is_some(), c2.is_some(), "level {level} {c:08x?}");
        if let (Some(g), Some(c2)) = (g, c2) {
            let first = g.iter().zip(c2).position(|(a, b)| a != b);
            assert!(
                first.is_none() && g.len() == c2.len(),
                "level {level} of {c:08x?} ({host:?}) differs at byte {first:?}"
            );
        }
    }
    Some((on_gpu, gpu.len()))
}

/// k_8_8_8_8 with each endian and swizzle, its whole mip chain.
#[test]
fn rgba8_decodes_on_the_gpu_as_on_the_cpu() {
    for endian in 0..4 {
        for swizzle in [IDENTITY, BGR1, ZX1Y] {
            let Some((gpu, levels)) = same_texels(fetch(6, endian, swizzle, true, 0), true) else {
                return;
            };
            assert!(gpu, "endian {endian} swizzle {swizzle:x}");
            assert_eq!(levels, 7);
        }
    }
}

/// Levels below the first sampled one stay zeros on both paths.
#[test]
fn levels_below_the_first_sampled_one_are_left_out() {
    let Some((gpu, levels)) = same_texels(fetch(6, 2, IDENTITY, true, 2), true) else {
        return;
    };
    assert!(gpu);
    assert_eq!(levels, 7);
}

/// The float and depth formats, to 32- and 16-bit float hosts.
#[test]
fn floats_decode_on_the_gpu_as_on_the_cpu() {
    // k_32_FLOAT, k_16_16_16_16_FLOAT, k_16_16_FLOAT, k_24_8, k_24_8_FLOAT
    for format in [36, 32, 31, 22, 23] {
        for float32 in [true, false] {
            for (endian, swizzle) in [(2, IDENTITY), (1, BGR1), (0, ZX1Y)] {
                let c = fetch(format, endian, swizzle, true, 0);
                let Some((gpu, _)) = same_texels(c, float32) else {
                    return;
                };
                assert!(gpu, "format {format} float32 {float32}");
            }
        }
    }
}

/// DXT1, DXT2/3 and DXT4/5 as BC blocks, the chain down to 4x4 (the
/// 2x2 and 1x1 levels are not whole blocks, so both paths stop there).
#[test]
fn bc_blocks_detile_on_the_gpu_as_on_the_cpu() {
    let Some(d) = drawer() else { return };
    if !d.bc_textures {
        println!("skip: the adapter has no BC textures");
        return;
    }
    drop(d);
    for format in [18, 19, 20] {
        for endian in [1, 2] {
            let c = fetch(format, endian, IDENTITY, true, 0);
            let (gpu, levels) = same_texels(c, true).unwrap();
            assert!(gpu, "format {format} endian {endian}");
            assert_eq!(levels, 5, "format {format}");
        }
        // A swizzle a BC view cannot do decodes on the CPU.
        let (gpu, _) = same_texels(fetch(format, 1, BGR1, true, 0), true).unwrap();
        assert!(!gpu);
    }
}

/// Linear layouts and other formats stay on the CPU.
#[test]
fn other_layouts_decode_on_the_cpu() {
    // k_8_8_8_8 linear, k_5_6_5 tiled
    for c in [
        fetch(6, 2, IDENTITY, false, 0),
        fetch(5, 1, IDENTITY, true, 0),
    ] {
        let Some((gpu, _)) = same_texels(c, true) else {
            return;
        };
        assert!(!gpu, "{c:08x?}");
    }
}
