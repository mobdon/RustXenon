// Resolve of a colour render target to a 32bpp guest destination: the same
// conversion as formats::edram_rgba + formats::resolve_pixel, then the
// 32x32 tiling and the destination endian, one invocation per pixel.

struct Params {
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    // Destination pitch in texels, a multiple of 32.
    pitch: u32,
    // 0 8_8_8_8, 1 2_10_10_10, 2 7e3 (hosted as half), 3 16_16_16_16 float,
    // 4 32 float, 5 D24S8, 6 D24FS8.
    src_format: u32,
    // 0 8_8_8_8, 1 2_10_10_10, 2 32_FLOAT.
    dest_format: u32,
    swap: u32,
    endian: u32,
    scale: f32,
    pad0: u32,
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

// xenon_texture::tile::tiled_offset_2d for 4-byte texels.
fn tiled(x: u32, y: u32, pitch: u32) -> u32 {
    let outer = (((y >> 5u) * (pitch >> 5u)) + (x >> 5u)) << 6u;
    let inner = (((y >> 1u) & 7u) << 3u) | (x & 7u);
    let oi = (outer | inner) << 2u;
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
    }
    c = c * p.scale;
    if p.swap != 0u {
        c = c.bgra;
    }
    var w: u32;
    if p.dest_format == 0u {
        w = to_unorm(c.r, 255.0) | (to_unorm(c.g, 255.0) << 8u) | (to_unorm(c.b, 255.0) << 16u)
            | (to_unorm(c.a, 255.0) << 24u);
    } else if p.dest_format == 2u {
        w = bitcast<u32>(c.r);
    } else {
        w = to_unorm(c.r, 1023.0) | (to_unorm(c.g, 1023.0) << 10u) | (to_unorm(c.b, 1023.0) << 20u)
            | (to_unorm(c.a, 3.0) << 30u);
    }
    store(x, y, w);
}

fn store(x: u32, y: u32, w: u32) {
    dest[tiled(x, y, p.pitch) >> 2u] = swap_endian(w, p.endian);
}

// Depth copies are raw EDRAM words, as in Xenia's fast resolve: depth in
// bits 8..31 (24-bit unorm, or 20e4 for src_format 6), stencil in 0..7;
// no conversion, bias or swap, only the destination endian.
@group(0) @binding(3) var depth_src: texture_depth_2d;
@group(0) @binding(4) var stencil_src: texture_2d<u32>;

// Xenia pixel_formats.xesli XeFloat32To20e4, rounding to nearest even.
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
    u = u + 3u + ((u >> 3u) & 1u);
    return (u >> 3u) & 0xFFFFFFu;
}

fn depth_word(x: u32, y: u32) -> u32 {
    let d = textureLoad(depth_src, vec2<u32>(x, y), 0);
    if p.src_format == 6u {
        return to_20e4(d) << 8u;
    }
    return u32(clamp(d, 0.0, 1.0) * 16777215.0 + 0.5) << 8u;
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
