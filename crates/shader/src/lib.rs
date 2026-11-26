//! Translates the console GPU's shader microcode (vertex and pixel programs
//! as loaded by IM_LOAD packets) into WGSL.
//!
//! - [`ucode`] decodes control-flow, ALU and fetch instructions.
//! - [`asm`] encodes them again, so tests can build programs by hand.
//! - [`translate`] turns a program into a WGSL module.
//!
//! # Binding contract with the drawer
//!
//! Every generated module follows this layout.
//!
//! - `@group(0) @binding(0) var<storage, read> guest: array<u32>` is a window
//!   of guest memory, declared only when [`ShaderInfo::uses_vertex_fetch`].
//!   The words are the guest bytes as stored, read little-endian by the host,
//!   so the shader byte-swaps each word by the vertex fetch constant's endian
//!   field (0 none, 1 8in16, 2 8in32, 3 16in32). Fetch addresses are masked
//!   to physical (`& 0x1FFFFFFF`), so `window_base` is a physical address:
//!   guest byte address `A` of vertex fetch constant `k` is word
//!   `(A - consts.window_base) / 4 + vf_rebase[k / 4][k % 4]`; words outside
//!   the window read as 0. The rebase lets the drawer upload far-apart
//!   parts of guest memory next to each other.
//! - `@group(0) @binding(1) var<uniform> consts: Consts` with
//!
//!   ```text
//!   struct Consts {
//!       fetch: array<vec4<u32>, 48>,
//!       bool_consts: array<vec4<u32>, 2>,
//!       loop_consts: array<vec4<u32>, 8>,
//!       window_base: u32,
//!       vertex_base: u32,
//!       _pad0: u32,
//!       _pad1: u32,
//!       vf_rebase: array<vec4<i32>, 24>,
//!   }
//!   ```
//!
//!   `fetch` holds the 32 fetch constants x 6 dwords in register order:
//!   dword `d` of fetch constant `n` is `fetch[(n*6+d)/4][(n*6+d)%4]`.
//!   Vertex fetch constant `k` (0..=95) is the dword pair starting at
//!   `2k`. `bool_consts` holds 256 bits, bit `i` being bit `i % 32` of
//!   flattened word `i / 32`. `loop_consts` holds the 32 loop constants
//!   (count in bits 0..8, aL start in 8..16, signed aL step in 16..24).
//! - `@group(0) @binding(4)` in the vertex stage and `@binding(5)` in the
//!   pixel stage is `var<uniform> fc: array<vec4<f32>, count>`: the float
//!   constants of that stage that [`ShaderInfo::float_consts_bound`] names,
//!   in ascending order, declared only when it is `Some`. A draw changes a
//!   few of the 512 float constants and a stage reads a few dozen, so the
//!   drawer uploads only those.
//!   The vertex shader's c0..c255 are float constant registers 0..256 and
//!   the pixel shader's are 256..512 ([`PIXEL_FLOAT_CONST_BASE`]; SQ_VS_CONST
//!   base 0 and SQ_PS_CONST base 256, as the console's D3D runtime sets them
//!   and Xenia asserts).
//! - Texture fetch slot `n` binds `@group(1) @binding(2n)` as
//!   `texture_2d<f32>` and `@group(1) @binding(2n+1)` as `sampler`, only for
//!   the slots in [`ShaderInfo::textures_used`]. The drawer applies the
//!   fetch constant's swizzle and converts signed components (sign 1) when
//!   it uploads the texture, so the host filters signed values; the texture
//!   the shader samples holds those swizzled, signed-converted texels.
//!   Format and filtering are also the drawer's. The shader reads the sign
//!   (dword 0), size (dwords 2 and 5, for unnormalized coordinates and
//!   offsets, so the host texture may have another resolution), swizzle and
//!   exponent bias (dword 3) and LOD bias (dword 4) of the fetch constant,
//!   and applies only the biased sign (2), gamma (3) and
//!   exponent bias to the sampled value; it reads the swizzle just to find
//!   which source component's sign each output component takes.
//! - Vertex stage: entry `vs_main`, input `@builtin(vertex_index)`. r0.x
//!   starts as the vertex index `(vertex_index + consts.vertex_base) &
//!   0xFFFFFF` (24 bits, as Xenia masks it; the min/max index clamp is not
//!   applied) (the drawer draws indexed with the guest index buffer). Outputs are
//!   `@builtin(position)` (oPos, as written; any viewport or clip-space
//!   adjustment is the drawer's) and `@location(i)` vec4 for each
//!   interpolator in [`ShaderInfo::interpolators_written`].
//! - Pixel stage: entry `ps_main`. Inputs are `@location(i)` vec4 for each
//!   bit of [`ShaderInfo::interpolators_read`], copied to r<i> on entry;
//!   outputs are `@location(0..4)` vec4 for the colour exports in
//!   [`ShaderInfo::colour_targets_written`], and `@builtin(frag_depth)`
//!   (oDepth.x clamped to [0, 1]) when [`ShaderInfo::writes_depth`]. With
//!   [`translate_with`] and a
//!   [`ParamGen`], the pixel parameters (position, facing, primitive flags)
//!   are written to their register on entry from `@builtin(position)` and
//!   `@builtin(front_facing)`, after the interpolators are copied. Such a
//!   module declares `override xe_scale: f32 = 1.0`, the host pixels per
//!   guest pixel, which divides the host position before it is rounded
//!   down; the drawer can set it per pipeline.
//!
//! Memory export fails with [`Error::Unsupported`]; the
//! point size / kill-vertex export is dropped and reported in
//! [`ShaderInfo::point_size_export_ignored`].
//!
//! # Execution model
//!
//! Temporaries are a private `array<vec4<f32>, 64>` (the register fields are
//! 6 bits; relative indices wrap modulo 64 like Xenia's interpreter). The
//! control-flow program runs as `loop { switch pc { ... } }`, one case per
//! control-flow instruction, with a 4-deep loop stack (count and aL) and a
//! 4-deep call stack. The loop stops after [`WATCHDOG_STEPS`] steps so a bad
//! loop cannot hang the GPU.
//!
//! Multiplies (`mul`, `mad`, `dp*`, `dst`, `muls*`) follow the Shader Model 3
//! rule Xenia implements: if either factor is 0 the product is 0, even for
//! infinity or NaN. `max`/`min` return the first operand when the comparison
//! holds and the second otherwise. `rcpc`/`rsqc` clamp infinities to
//! +-FLT_MAX, `logc` clamps only -infinity, and the `*f` variants turn
//! infinities into signed zeros.

pub mod asm;
pub mod ucode;
mod wgsl;

pub use wgsl::{
    translate, translate_with, Error, ParamGen, ParamGenPrimitive, ShaderInfo, ShaderStage,
    Translated, VertexFetchReach, DEPTH_MEMBER, DEPTH_STORE, PIXEL_FLOAT_CONSTS_BINDING,
    PIXEL_FLOAT_CONST_BASE, TEMP_REGISTERS, VERTEX_FLOAT_CONSTS_BINDING, WATCHDOG_STEPS,
};
