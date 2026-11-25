// Resolve of a colour render target to a 32bpp (or 16_16_16_16_FLOAT)
// guest destination: the same conversion as formats::edram_rgba +
// formats::resolve_pixel, then the 32x32 tiling and the destination endian,
// one invocation per pixel.

struct Params {
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    // Destination pitch in texels, a multiple of 32.
    pitch: u32,
    // 0 8_8_8_8, 1 2_10_10_10, 2 7e3 (hosted as half), 3 16_16_16_16 float,
    // 4 32 float, 5 D24S8, 6 D24FS8, 7 16_16 and 8 16_16_16_16 (hosted as
    // snorm / 32), 9 16_16 float, 10 32_32 float.
    src_format: u32,
    // 0 8_8_8_8, 1 2_10_10_10, 2 32_FLOAT, 3 16_16, 4 16_16_16_16_FLOAT
    // (64bpp).
    dest_format: u32,
    swap: u32,
    // xenos::Endian128; 4 (8in64) only for 64bpp.
    endian: u32,
    scale: f32,
    // Word index of the destination's first byte in `dest`.
    offset: u32,
    pad1: u32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dest: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

fn to_7e3(f: f32) -> u32 {
    var v = 0.0;
    if f > 0.0 {
        v = min(f, 31.875);
    }
    let u = bitcast<u32>(v);
    var b: u32;
    if u < 0x3E800000u {
        b = ((u & 0x7FFFFFu) | 0x800000u) >> min(125u - (u >> 23u), 24u);
    } else {
        b = u - (124u << 23u);
    }
    return ((b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u) & 0x3FFu;
}

fn from_7e3(f10: u32) -> f32 {
    if f10 == 0u {
        return 0.0;
    }
    var m = f10 & 0x7Fu;
    var e = f10 >> 7u;
    if e == 0u {
        let lz = countLeadingZeros(m) - 24u;
        e = 1u - lz;
        m = (m << lz) & 0x7Fu;
    }
    return bitcast<f32>(((e + 124u) << 23u) | (m << 16u));
}

fn q7e3(f: f32) -> f32 {
    return from_7e3(to_7e3(f));
}

// Saturate (NaN to 0), scale, +0.5 and truncate.
fn to_unorm(c: f32, max_value: f32) -> u32 {
    var s = 0.0;
    if c > 0.0 {
        s = min(c, 1.0);
    }
    return u32(s * max_value + 0.5);
}

// formats::f32_to_half of the f32 with bits `u`: round to nearest even,
// overflow to infinity, NaN to the quiet NaN of its sign.
fn half_bits(u: u32) -> u32 {
    let sign = (u >> 16u) & 0x8000u;
    let exp = i32((u >> 23u) & 0xFFu);
    let man = u & 0x7FFFFFu;
    if exp == 0xFF {
        return sign | 0x7C00u | select(0u, 0x200u, man != 0u);
    }
    let e = exp - 112;
    if e >= 0x1F {
        return sign | 0x7C00u;
    }
    var sig = man;
    var shift = 13u;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        sig = man | 0x800000u;
        shift = u32(14 - e);
    }
    let half_ulp = 1u << (shift - 1u);
    let rest = sig & ((1u << shift) - 1u);
    var h = sig >> shift;
    if rest > half_ulp || (rest == half_ulp && (h & 1u) != 0u) {
        h += 1u;
    }
    if e > 0 {
        h += u32(e) << 10u;
    }
    return sign | h;
}

fn to_half(f: f32) -> u32 {
    return half_bits(bitcast<u32>(f));
}

// xenon_texture::tile::tiled_offset_2d for 4-byte texels.
fn tiled(x: u32, y: u32, pitch: u32) -> u32 {
    return tiled_b(x, y, pitch, 2u);
}

// xenon_texture::tile::tiled_offset_2d for `1 << log2` byte texels.
fn tiled_b(x: u32, y: u32, pitch: u32, log2: u32) -> u32 {
    let outer = (((y >> 5u) * (pitch >> 5u)) + (x >> 5u)) << 6u;
    let inner = (((y >> 1u) & 7u) << 3u) | (x & 7u);
    let oi = (outer | inner) << log2;
    let bank = (y >> 4u) & 1u;
    let pipe = ((x >> 3u) & 3u) ^ (((y >> 3u) & 1u) << 1u);
    return ((y & 1u) << 4u) | (pipe << 6u) | (bank << 11u) | (oi & 15u)
        | (((oi >> 4u) & 1u) << 5u) | (((oi >> 5u) & 7u) << 8u) | ((oi >> 8u) << 12u);
}

fn swap_endian(w: u32, e: u32) -> u32 {
    switch e {
        case 1u: {
            return ((w & 0x00FF00FFu) << 8u) | ((w >> 8u) & 0x00FF00FFu);
        }
        case 2u: {
            return (w << 24u) | ((w & 0xFF00u) << 8u) | ((w >> 8u) & 0xFF00u) | (w >> 24u);
        }
        case 3u: {
            return (w << 16u) | (w >> 16u);
        }
        default: {
            return w;
        }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.w || id.y >= p.h {
        return;
    }
    let x = p.x0 + id.x;
    let y = p.y0 + id.y;
    var c = textureLoad(src, vec2<u32>(x, y), 0);
    if p.src_format == 2u {
        var a = 0.0;
        if c.a > 0.0 {
            a = min(c.a, 1.0);
        }
        let a2 = u32(a * 3.0 + 0.5);
        c = vec4<f32>(q7e3(c.r), q7e3(c.g), q7e3(c.b), f32(a2) * (1.0 / 3.0));
    } else if p.src_format == 4u {
        c = vec4<f32>(c.r, 0.0, 0.0, 0.0);
    } else if p.src_format == 7u || p.src_format == 8u {
        // The snorm texel back to its 16 bits, then XeUnpackR16G16Edram.
        c = max(vec4<f32>(-1.0), round(c * 32767.0) * (32.0 / 32767.0));
        if p.src_format == 7u {
            c = vec4<f32>(c.rg, 0.0, 0.0);
        }
    } else if p.src_format >= 9u {
        c = vec4<f32>(c.rg, 0.0, 0.0);
    }
    c = c * p.scale;
    if p.swap != 0u {
        c = c.bgra;
    }
    if p.dest_format == 4u {
        let lo = to_half(c.r) | (to_half(c.g) << 16u);
        let hi = to_half(c.b) | (to_half(c.a) << 16u);
        let o = p.offset + (tiled_b(x, y, p.pitch, 3u) >> 2u);
        // 8in64: the two words swapped, then 8in32.
        if p.endian == 4u {
            dest[o] = swap_endian(hi, 2u);
            dest[o + 1u] = swap_endian(lo, 2u);
        } else {
            dest[o] = swap_endian(lo, p.endian);
            dest[o + 1u] = swap_endian(hi, p.endian);
        }
        return;
    }
    var w: u32;
    if p.dest_format == 0u {
        w = to_unorm(c.r, 255.0) | (to_unorm(c.g, 255.0) << 8u) | (to_unorm(c.b, 255.0) << 16u)
            | (to_unorm(c.a, 255.0) << 24u);
    } else if p.dest_format == 2u {
        w = bitcast<u32>(c.r);
    } else if p.dest_format == 3u {
        w = to_unorm(c.r, 65535.0) | (to_unorm(c.g, 65535.0) << 16u);
    } else {
        w = to_unorm(c.r, 1023.0) | (to_unorm(c.g, 1023.0) << 10u) | (to_unorm(c.b, 1023.0) << 20u)
            | (to_unorm(c.a, 3.0) << 30u);
    }
    store(x, y, w);
}

fn store(x: u32, y: u32, w: u32) {
    dest[p.offset + (tiled(x, y, p.pitch) >> 2u)] = swap_endian(w, p.endian);
}

// Depth copies are raw EDRAM words, as in Xenia's fast resolve: depth in
// bits 8..31 (24-bit unorm, or 20e4 for src_format 6), stencil in 0..7;
// no conversion, bias or swap, only the destination endian.
@group(0) @binding(3) var depth_src: texture_depth_2d;
@group(0) @binding(4) var stencil_src: texture_2d<u32>;

// Xenia pixel_formats.xesli XeFloat32To20e4 without rounding (truncating),
// as Xenia's host render target dump does by default (depth_float24_round
// off).
fn to_20e4(f: f32) -> u32 {
    if !(f > 0.0) {
        return 0u;
    }
    var u = bitcast<u32>(f);
    if u >= 0x3FFFFFF8u {
        return 0xFFFFFFu;
    }
    if u < 0x38800000u {
        u = (0x800000u | (u & 0x7FFFFFu)) >> min(113u - (u >> 23u), 24u);
    } else {
        u = u + 0xC8000000u;
    }
    return (u >> 3u) & 0xFFFFFFu;
}

fn depth_word(x: u32, y: u32) -> u32 {
    let d = textureLoad(depth_src, vec2<u32>(x, y), 0);
    if p.src_format == 6u {
        return to_20e4(d) << 8u;
    }
    // Round to nearest even, as Xenia's dump does: `+ 0.5` ties every odd
    // value from 2^23 up to the next even one.
    return u32(round(clamp(d, 0.0, 1.0) * 16777215.0)) << 8u;
}

// D24FS8 is hosted without stencil, so its stencil bits are 0.
@compute @workgroup_size(8, 8)
fn depth_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.w || id.y >= p.h {
        return;
    }
    let x = p.x0 + id.x;
    let y = p.y0 + id.y;
    store(x, y, depth_word(x, y));
}

@compute @workgroup_size(8, 8)
fn depth_stencil_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.w || id.y >= p.h {
        return;
    }
    let x = p.x0 + id.x;
    let y = p.y0 + id.y;
    store(x, y, depth_word(x, y) | (textureLoad(stencil_src, vec2<u32>(x, y), 0).r & 0xFFu));
}

// A tiled k_8_8_8_8 texture decoded from guest bytes kept on the GPU, as
// xenon_texture::decode_base_level does on the CPU: the fetch endian, the
// 32x32 tiling, then the fetch swizzle. One invocation per texel, into
// rows of `row_words` words for a copy to an Rgba8Unorm texture (rows y0
// to y0 + h).
struct DecodeParams {
    w: u32,
    h: u32,
    pitch: u32,
    endian: u32,
    swizzle: u32,
    // Word index of the texture's first byte in `guest`.
    offset: u32,
    row_words: u32,
    // First row decoded; `h` rows from it.
    y0: u32,
    // decode_float only: 0 k_32_FLOAT to Rgba32Float, 1 k_32_FLOAT to
    // Rgba16Float, 2 k_16_16_16_16_FLOAT to Rgba16Float.
    mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(5) var<storage, read> guest: array<u32>;
@group(0) @binding(6) var<storage, read_write> texels: array<u32>;
@group(0) @binding(7) var<uniform> dp: DecodeParams;

@compute @workgroup_size(8, 8)
fn decode_8888(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dp.w || id.y >= dp.h {
        return;
    }
    let y = dp.y0 + id.y;
    let w = swap_endian(guest[dp.offset + (tiled(id.x, y, dp.pitch) >> 2u)], dp.endian);
    var out = 0u;
    for (var i = 0u; i < 4u; i++) {
        let s = (dp.swizzle >> (3u * i)) & 7u;
        var b = 0u;
        if s < 4u {
            b = (w >> (8u * s)) & 0xFFu;
        } else if (s & 1u) != 0u {
            b = 0xFFu;
        }
        out |= b << (8u * i);
    }
    texels[id.y * dp.row_words + id.x] = out;
}

// A tiled k_32_FLOAT or k_16_16_16_16_FLOAT texture decoded from guest bytes
// kept on the GPU, as xenon_texture::decode_base_level_f32 and the host
// conversion in Drawer::guest_texture do on the CPU: the fetch endian, the
// tiling, the texel as RRRR (k_32_FLOAT) or four halves, the fetch swizzle,
// then f32 bits or halves (formats::f32_to_half) per host texel.
@compute @workgroup_size(8, 8)
fn decode_float(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dp.w || id.y >= dp.h {
        return;
    }
    let y = dp.y0 + id.y;
    var v: array<u32, 4>;
    var one = 0x3F800000u;
    if dp.mode == 2u {
        let o = dp.offset + (tiled_b(id.x, y, dp.pitch, 3u) >> 2u);
        let lo = swap_endian(guest[o], dp.endian);
        let hi = swap_endian(guest[o + 1u], dp.endian);
        v = array<u32, 4>(lo & 0xFFFFu, lo >> 16u, hi & 0xFFFFu, hi >> 16u);
        // f32_to_half(half_to_f32(h)) keeps every half but NaNs: the
        // texture crate's half_to_f32 drops their sign, so all of them
        // become the positive quiet NaN 0x7E00.
        for (var i = 0u; i < 4u; i++) {
            if (v[i] & 0x7C00u) == 0x7C00u && (v[i] & 0x3FFu) != 0u {
                v[i] = 0x7E00u;
            }
        }
        one = 0x3C00u;
    } else {
        let w = swap_endian(guest[dp.offset + (tiled(id.x, y, dp.pitch) >> 2u)], dp.endian);
        v = array<u32, 4>(w, w, w, w);
    }
    var out: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let s = (dp.swizzle >> (3u * i)) & 7u;
        if s < 4u {
            out[i] = v[s];
        } else if (s & 1u) != 0u {
            out[i] = one;
        } else {
            out[i] = 0u;
        }
    }
    if dp.mode == 0u {
        for (var i = 0u; i < 4u; i++) {
            texels[id.y * dp.row_words + id.x * 4u + i] = out[i];
        }
        return;
    }
    if dp.mode == 1u {
        for (var i = 0u; i < 4u; i++) {
            out[i] = half_bits(out[i]);
        }
    }
    texels[id.y * dp.row_words + id.x * 2u] = out[0] | (out[1] << 16u);
    texels[id.y * dp.row_words + id.x * 2u + 1u] = out[2] | (out[3] << 16u);
}
