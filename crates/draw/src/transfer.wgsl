
// EDRAM ownership transfers between a k_8_8_8_8 colour target and a depth
// target at the same EDRAM base and pitch (Xenia render_target_cache.cc,
// transfer shaders of d3d12/vulkan_render_target_cache.cc): the other
// target's pixels are drawn as the 32-bit EDRAM word with a fullscreen
// triangle. Appended to resolve.wgsl, whose `src`, `depth_src`,
// `stencil_src`, `p` and `depth_word` it uses.

// The depth target is kD24FS8 (20e4), else kD24S8 (24-bit unorm).
override transfer_20e4: bool = false;
// Stencil bit written by `color_to_stencil`.
override transfer_bit: u32 = 0u;

@vertex
fn transfer_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The colour texel as its EDRAM word: red in the low byte.
fn color_word(pos: vec4<f32>) -> u32 {
    return pack4x8unorm(textureLoad(src, vec2<u32>(pos.xy), 0));
}

// xenon_texture float20e4_to_f32 (Xenia Float20e4To32): exact in f32.
fn from_20e4(f24: u32) -> f32 {
    if f24 == 0u {
        return 0.0;
    }
    var m = f24 & 0xFFFFFu;
    var e = f24 >> 20u;
    if e == 0u {
        let lz = countLeadingZeros(m) - 11u;
        e = 1u - lz;
        m = (m << lz) & 0xFFFFFu;
    }
    return bitcast<f32>(((e + 112u) << 23u) | (m << 3u));
}

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

@fragment
fn color_to_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let d = color_word(pos) >> 8u;
    if transfer_20e4 {
        return from_20e4(d);
    }
    return f32(d) * (1.0 / 16777215.0);
}

// wgpu has no stencil export: one pass per stencil bit after the stencil
// is cleared, writing only bit `transfer_bit` where the word has it set
// (Xenia's fallback without shader stencil export).
@fragment
fn color_to_stencil(@builtin(position) pos: vec4<f32>) {
    if (color_word(pos) & (1u << transfer_bit)) == 0u {
        discard;
    }
}

@fragment
fn depth_to_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return unpack4x8unorm(depth_word(u32(pos.x), u32(pos.y)));
}

@fragment
fn depth_stencil_to_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(pos.x);
    let y = u32(pos.y);
    return unpack4x8unorm(depth_word(x, y) | (textureLoad(stencil_src, vec2<u32>(x, y), 0).r & 0xFFu));
}
