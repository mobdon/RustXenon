
// EDRAM ownership transfers between k_8_8_8_8 colour, kD24S8 and kD24FS8
// targets on the same EDRAM tiles (Xenia render_target_cache.cc, transfer
// shaders of d3d12/vulkan_render_target_cache.cc): the other target's
// pixels are drawn as the 32-bit EDRAM word with a fullscreen triangle.
// Appended to resolve.wgsl, whose `src`, `depth_src`, `stencil_src`, `p`
// and `depth_word` it uses. Entry points are `<source>_to_<destination>`.
//
// The two targets may differ in pitch and sample count while covering the
// same tiles (a 4x MSAA surface at half the pitch of a 2x one): `p.x0` and
// `p.y0` are the destination's samples per pixel across and down, `p.w`
// and `p.h` the source's, `p.pitch` the source's pitch, so a destination
// pixel reads the source pixel holding its first sample.
//
// Depth tiles store the two 40-sample halves of each 80x16-sample tile
// swapped against colour (Xenia xenos.h, edram.xesli; games pre-swap when
// they restore depth through colour), so a transfer between colour and
// depth reads the other half of the tile.

// The depth target is kD24FS8 (20e4), else kD24S8 (24-bit unorm).
override transfer_20e4: bool = false;
// Stencil bit written by `color_to_stencil`.
override transfer_bit: u32 = 0u;

@vertex
fn transfer_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The source pixel under destination pixel `pos`, `swap` between colour
// and depth. A sample past the source's row (rows not a whole number of
// tiles wide) is not hosted: the destination keeps its pixel.
//
// Targets are drawn at p.px_scale host pixels per guest pixel: the guest
// pixel is mapped as at scale 1 (samples and tile halves are guest ones)
// and the host pixel keeps its place inside it, so a transfer between
// targets of one sample count moves every host pixel unchanged.
fn src_at(pos: vec4<f32>, swap: bool) -> vec2<u32> {
    let k = p.px_scale;
    let host = vec2<u32>(pos.xy);
    var s = host / k * vec2<u32>(p.x0, p.y0);
    if swap {
        s.x = s.x - s.x % 80u + (s.x % 80u + 40u) % 80u;
    }
    let at = s / vec2<u32>(p.w, p.h);
    if at.x >= p.pitch {
        discard;
    }
    return at * k + host % k;
}

// The colour texel as its EDRAM word: red in the low byte.
fn color_word(pos: vec4<f32>, swap: bool) -> u32 {
    return pack4x8unorm(textureLoad(src, src_at(pos, swap), 0));
}

// The depth texel as its EDRAM word, stencil 0 (no host stencil).
fn depth_word_at(pos: vec4<f32>, swap: bool) -> u32 {
    let at = src_at(pos, swap);
    return depth_word(at.x, at.y);
}

fn depth_stencil_word(pos: vec4<f32>, swap: bool) -> u32 {
    let at = src_at(pos, swap);
    return depth_word(at.x, at.y) | (textureLoad(stencil_src, at, 0).r & 0xFFu);
}

// from_20e4 is in resolve.wgsl, which comes before this file.

// n / 0xFFFFFF rounded to nearest, as the drawer's depth clear computes it
// in f64. The quotient's binary digits are n's 24 bits repeated, so the
// significand is n shifted to its leading one, and the next digit (that
// leading one again, with nonzero digits after it) always rounds up.
fn from_unorm24(n: u32) -> f32 {
    if n == 0u {
        return 0.0;
    }
    let lz = countLeadingZeros(n) - 8u;
    return bitcast<f32>(((126u - lz) << 23u) + ((n << lz) & 0x7FFFFFu) + 1u);
}

// The depth of 24-bit EDRAM depth `d`, as the depth clear makes it.
fn word_depth(d: u32) -> f32 {
    if transfer_20e4 {
        return from_20e4(d);
    }
    return from_unorm24(d);
}

@fragment
fn color_to_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    return word_depth(color_word(pos, true) >> 8u);
}

@fragment
fn depth_to_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    return word_depth(depth_word_at(pos, false) >> 8u);
}

@fragment
fn depth_stencil_to_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    return word_depth(depth_stencil_word(pos, false) >> 8u);
}

// wgpu has no stencil export: one pass per stencil bit after the stencil
// is cleared, writing only bit `transfer_bit` where the word has it set
// (Xenia's fallback without shader stencil export).
@fragment
fn color_to_stencil(@builtin(position) pos: vec4<f32>) {
    if (color_word(pos, true) & (1u << transfer_bit)) == 0u {
        discard;
    }
}

@fragment
fn depth_stencil_to_stencil(@builtin(position) pos: vec4<f32>) {
    if (depth_stencil_word(pos, false) & (1u << transfer_bit)) == 0u {
        discard;
    }
}

@fragment
fn color_to_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return unpack4x8unorm(color_word(pos, false));
}

@fragment
fn depth_to_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return unpack4x8unorm(depth_word_at(pos, true));
}

@fragment
fn depth_stencil_to_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return unpack4x8unorm(depth_stencil_word(pos, true));
}
