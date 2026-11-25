// float20e4 (Xenia src/xenia/gpu/xenos.cc Float20e4To32) lives in
// xenon-texture; the zero
// texture bound for a slot without a usable texture matches the null images
// (cleared to 0, 0, 0, 0) of src/xenia/gpu/vulkan/vulkan_texture_cache.cc; the
// 8_8_8_8 colour clear packing (red in the low byte) follows
// src/xenia/gpu/shaders/pixel_formats.xesli; render target formats and
// RB_*_INFO fields follow src/xenia/gpu/xenos.h and registers.h; resolve
// clears follow IsClearingColor in src/xenia/gpu/draw_util.h; the rectangle
// list fourth corner (longest edge, v3 = (v1 - v0) + v2 in strip order for
// the position and every interpolator) follows the kRectangleList geometry
// shader in src/xenia/gpu/vulkan/vulkan_pipeline_cache.cc, no face culling
// for rectangle lists follows IsPrimitivePolygonal in draw_util.h, and the
// surface pitch 0 skip follows IsRasterizationPotentiallyDone in
// draw_util.cc (BSD 3-Clause, see CREDITS.md).

//! `xenon-draw`: a headless wgpu drawer for recorded Xenos command lists.
//!
//! [`Drawer::replay`] walks a [`xenon_gpu::Command`] list (from
//! [`xenon_gpu::Recorder`]) and draws it into offscreen render targets: it
//! keeps its own register file, translates the loaded shader programs with
//! [`xenon_shader::translate`], decodes textures with [`xenon_texture`],
//! performs EDRAM resolves into guest memory and turns swaps into [`Frame`]s
//! that can be written as PNG files. There is no window and no surface.
//!
//! # EDRAM model
//!
//! Every render target is its own wgpu texture keyed by
//! [`RenderTargetKey`]: EDRAM base tile, surface pitch and format. Aliasing
//! between targets that share EDRAM tiles (another format, another pitch,
//! another base inside the same range) is not modelled: each key has its
//! own pixels. A target is `pitch` pixels wide and [`RENDER_TARGET_HEIGHT`]
//! pixels high (the EDRAM holds 2048 rows of 1280 32-bit pixels).
//!
//! Colour targets ([`TargetFormat`]; formats with one bit layout share a
//! key):
//!
//! - `k_8_8_8_8` and `k_8_8_8_8_GAMMA` as `Rgba8Unorm`. The gamma format is
//!   drawn and resolved as plain 8_8_8_8, without the piecewise-linear gamma
//!   conversion.
//! - `k_2_10_10_10` and `k_2_10_10_10_AS_10_10_10_10` as `Rgb10a2Unorm`.
//! - `k_2_10_10_10_FLOAT` and `k_2_10_10_10_FLOAT_AS_16_16_16_16` (7e3 RGB in
//!   [0, 31.875], 2-bit alpha) as `Rgba16Float`, as Xenia hosts them. The
//!   pixel shader's colour is clamped to [0, 31.875] and alpha to [0, 1]
//!   when blending is off; with blending only alpha is clamped, as Xenia's
//!   EDRAM emulation does. The blend result itself is not clamped and keeps
//!   half precision until a resolve quantises it (see below).
//! - `k_16_16_16_16_FLOAT` as `Rgba16Float`.
//! - `k_32_FLOAT` as `R32Float`, drawn without blending (a warning) and
//!   resolvable to `k_32_FLOAT` with its float bits kept, unsaturated.
//!
//! All four render targets (`RB_COLOR_INFO_RT[0..4]`) are drawn, each with
//! its own `RB_COLOR_MASK` bits, `RB_BLENDCONTROL_RT` blend and output
//! clamp; a target is skipped when the pixel shader does not export it.
//! The pixel shader's colour is scaled by `2^color_exp_bias`
//! (`RB_COLOR_INFO` bits 20..25, signed) before the output clamp, as Xenia
//! does.
//! MSAA surfaces are drawn and resolved with one sample per pixel (a
//! warning).
//! Resolve clears decode the 64-bit value `RB_COLOR_CLEAR |
//! RB_COLOR_CLEAR_LO << 32` per format ([`formats::clear_color`]).
//! Depth targets: `kD24S8` as `Depth24PlusStencil8` and `kD24FS8` as
//! `Depth32FloatStencil8` where the host has it, else `Depth32Float` without
//! stencil (the 20e4 float is stored at f32 precision). Anything
//! else (`k_16_16`, `k_16_16_16_16`, `k_16_16_FLOAT`, `k_32_32_FLOAT`) is
//! reported in [`Report::unsupported`].
//!
//! # Resolves
//!
//! A colour copy reads the rectangle back, takes each texel as the EDRAM
//! holds it (a 7e3 texel is rounded to 7e3 and its alpha to 2 bits, like
//! Xenia's dump of a host render target into EDRAM), applies the exponent
//! bias `2^copy_dest_exp_bias` and the red/blue swap, packs it like Xenia's
//! resolve shader (unorm: saturate, `x * max + 0.5` truncated) into
//! `k_8_8_8_8`, `k_2_10_10_10` or `k_16_16_16_16_FLOAT`, and tiles it into
//! guest memory with the destination endian. Copy commands are described at
//! `Drawer::resolve` (`Raw` and `Convert` alike, `Null` clears only,
//! `ConstantOne` unsupported).
//!
//! A depth copy writes the raw 32-bit EDRAM word whatever the destination
//! format, as Xenia's fast resolve does: depth in bits 8..31 (`kD24S8` as
//! 24-bit unorm, rounded to nearest even; `kD24FS8` as 20e4, truncated like
//! Xenia's default) and stencil in
//! bits 0..7 (0 for a `kD24FS8` hosted without stencil), with the destination
//! endian and no bias or swap.
//!
//! # Draws
//!
//! The vertex shader's position goes through a WGSL epilogue the drawer
//! appends ([`state::EPILOGUE_WGSL`]): PA_CL_VTE_CNTL's W/XY/Z formats, the
//! PA_CL_VPORT scale and offset (scale 1 and offset 0 when disabled, so the
//! position is then in window pixels), the window offset and the Direct3D 9
//! half-pixel centre, then conversion to the host's NDC for a viewport that
//! covers the whole render target. Interpolators the pixel shader reads but
//! the vertex shader does not write are added to the vertex outputs as
//! zero. Quad lists and triangle fans are expanded to triangle lists by
//! index. Rectangle lists draw six host vertices per rectangle through a
//! wrapper around the translated vertex shader that makes the fourth corner
//! from the three guest vertices after the vertex shader, as Xenia's
//! geometry shader does (`rect_list_wgsl`); they are never face culled.
//!
//! Draws while RB_SURFACE_INFO's pitch is 0 rasterize nothing, as in Xenia
//! (`IsRasterizationPotentiallyDone`): Direct3D draws before its first
//! render target setup. They are skipped with a warning.

pub mod formats;
pub mod frame;
pub mod indices;
pub mod resolve;
pub mod state;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xenon_gpu::regs;
use xenon_gpu::{
    Command, CopyCommand, Draw, FetchConstantType, GuestRead, RegisterFile, Resolve, ShaderStage,
    SliceMemory, PHYSICAL_MASK,
};
use xenon_shader::{ShaderInfo, Translated};
use xenon_texture::convert::float20e4_to_f32 as float20e4;

pub use frame::Frame;

/// Height in pixels of every render target texture.
pub const RENDER_TARGET_HEIGHT: u32 = 2048;

/// Default bound on every GPU wait.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// Guest memory the drawer reads (vertices, indices, textures, resolve
/// rectangles, front buffers) and writes (MEM_WRITE packets, resolves).
pub trait GuestMemory: GuestRead {
    /// Writes `bytes` at guest physical `addr`; false (and nothing written)
    /// if any byte is unmapped.
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool;
}

impl GuestMemory for SliceMemory {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        let Some(start) = addr.checked_sub(self.base).map(|s| s as usize) else {
            return false;
        };
        let Some(end) = start.checked_add(bytes.len()) else {
            return false;
        };
        match self.data.get_mut(start..end) {
            Some(dst) => {
                dst.copy_from_slice(bytes);
                true
            }
            None => false,
        }
    }
}

/// Failures that stop a replay or a drawer call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No Vulkan or GL adapter, hardware or software, was found.
    NoAdapter,
    /// The adapter refused a device, or lacks something the drawer needs.
    RequestDevice(String),
    /// A GPU wait or buffer map did not finish within the drawer's timeout.
    Timeout,
    /// wgpu reported an error (validation, out of memory, device lost).
    Gpu(String),
    /// A readback of something this drawer cannot read back.
    Unsupported(String),
    /// [`Drawer::read_render_target`] for a key that was never drawn to.
    NoRenderTarget(RenderTargetKey),
    Png(String),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter => write!(f, "no Vulkan or GL adapter found"),
            Error::RequestDevice(m) => write!(f, "cannot create a device: {m}"),
            Error::Timeout => write!(f, "GPU wait timed out"),
            Error::Gpu(m) => write!(f, "GPU error: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::NoRenderTarget(k) => write!(f, "no render target {k:?}"),
            Error::Png(m) => write!(f, "PNG error: {m}"),
            Error::Io(m) => write!(f, "I/O error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// What a replay did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// One frame per swap, in order.
    pub frames: Vec<Frame>,
    /// Draws submitted to the GPU.
    pub draws: u32,
    /// Resolves performed (copy, clear or both).
    pub resolves: u32,
    /// Why draws or resolves were skipped, each reason once.
    pub unsupported: Vec<String>,
    /// Things done approximately or ignored, each once.
    pub warnings: Vec<String>,
    /// Where this replay's wall-clock time went.
    pub timings: Timings,
}

/// Wall-clock time per replay phase, to find what dominates a frame. With a
/// software driver (lavapipe) the GPU phases are CPU time too.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timings {
    /// Shader translation, WGSL modules and pipeline creation (cache misses
    /// only cost here), plus fixed-function state decode.
    pub shaders: Duration,
    /// Guest vertex window, constants and index buffers: reading guest
    /// memory and uploading it, per draw.
    pub guest_data: Duration,
    /// Texture decode (detile and convert) and upload, per draw.
    pub textures: Duration,
    /// Encoding and submitting draws (the GPU runs them asynchronously).
    pub submit: Duration,
    /// Waiting for queued GPU work before a resolve readback: where the
    /// rendering itself shows up.
    pub gpu_wait: Duration,
    /// Copying resolve sources back from the GPU.
    pub readback: Duration,
    /// Resolve pixel conversion and tiling into guest memory, on the CPU.
    pub resolve_cpu: Duration,
    /// Clears.
    pub clears: Duration,
    /// Reading, swapping and detiling the front buffer at each swap.
    pub swap: Duration,
    /// Counts: pipelines and shader modules created, textures uploaded.
    pub pipelines_created: u32,
    pub textures_uploaded: u32,
    pub texture_bytes: u64,
    pub guest_bytes: u64,
}

impl Report {
    fn unsupported(&mut self, why: String) {
        if !self.unsupported.contains(&why) {
            self.unsupported.push(why);
        }
    }

    fn warn(&mut self, why: impl Into<String>) {
        let why = why.into();
        if !self.warnings.contains(&why) {
            self.warnings.push(why);
        }
    }
}

/// Render target pixel formats this drawer keeps. Guest formats with the
/// same EDRAM bit layout share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetFormat {
    /// `k_8_8_8_8` and `k_8_8_8_8_GAMMA`.
    Rgba8,
    /// `kD24S8`.
    D24S8,
    /// `kD24FS8`.
    D24FS8,
    /// `k_2_10_10_10` and `k_2_10_10_10_AS_10_10_10_10`, hosted as
    /// `Rgb10a2Unorm` (the same bit layout).
    Rgb10a2,
    /// `k_2_10_10_10_FLOAT` and `k_2_10_10_10_FLOAT_AS_16_16_16_16` (7e3
    /// RGB, 2-bit alpha), hosted as `Rgba16Float` like Xenia does.
    Float7e3,
    /// `k_16_16_16_16_FLOAT`, hosted as `Rgba16Float`.
    Rgba16Float,
    /// `k_32_FLOAT`, hosted as `R32Float`.
    Float32,
}

impl TargetFormat {
    fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            TargetFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            TargetFormat::D24S8 => wgpu::TextureFormat::Depth24PlusStencil8,
            TargetFormat::D24FS8 => wgpu::TextureFormat::Depth32Float,
            TargetFormat::Rgb10a2 => wgpu::TextureFormat::Rgb10a2Unorm,
            TargetFormat::Float7e3 | TargetFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            TargetFormat::Float32 => wgpu::TextureFormat::R32Float,
        }
    }

    /// The host format for `xenos::ColorRenderTargetFormat` `f`.
    pub fn color(f: u32) -> Option<TargetFormat> {
        Some(match f {
            0 | 1 => TargetFormat::Rgba8,
            2 | 10 => TargetFormat::Rgb10a2,
            3 | 12 => TargetFormat::Float7e3,
            14 => TargetFormat::Float32,
            7 => TargetFormat::Rgba16Float,
            _ => return None,
        })
    }

    fn is_depth(self) -> bool {
        matches!(self, TargetFormat::D24S8 | TargetFormat::D24FS8)
    }
}

/// Identity of a render target: EDRAM base (tiles, 12 bits), surface pitch
/// (pixels) and format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetKey {
    pub base_tiles: u32,
    pub pitch: u32,
    pub format: TargetFormat,
}

impl RenderTargetKey {
    /// Colour render target `rt` (0..4) as the registers describe it.
    pub fn color(r: &RegisterFile, rt: usize) -> Result<Self, String> {
        let info = r.get(regs::RB_COLOR_INFO_RT[rt]);
        let f = (info >> 16) & 0xF;
        let format =
            TargetFormat::color(f).ok_or_else(|| format!("colour render target format {f}"))?;
        Ok(RenderTargetKey {
            base_tiles: info & 0xFFF,
            pitch: r.get(regs::RB_SURFACE_INFO) & 0x3FFF,
            format,
        })
    }

    /// The depth render target as the registers describe it.
    pub fn depth(r: &RegisterFile) -> Self {
        let info = r.get(regs::RB_DEPTH_INFO);
        RenderTargetKey {
            base_tiles: info & 0xFFF,
            pitch: r.get(regs::RB_SURFACE_INFO) & 0x3FFF,
            format: if info & (1 << 16) != 0 {
                TargetFormat::D24FS8
            } else {
                TargetFormat::D24S8
            },
        }
    }
}

/// Why one draw or resolve was not done.
enum Skip {
    Unsupported(String),
    Fatal(Error),
}

impl From<Error> for Skip {
    fn from(e: Error) -> Self {
        Skip::Fatal(e)
    }
}

fn unsupported<T>(why: impl Into<String>) -> Result<T, Skip> {
    Err(Skip::Unsupported(why.into()))
}

#[derive(Clone)]
struct Program {
    words: Vec<u32>,
    hash: u64,
}

/// How a translated shader is patched beyond the epilogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Variant {
    Plain,
    /// Vertex shader wrapped to make rectangle list corners.
    RectList,
    /// Pixel shader colour i scaled by 2^bias, then clamped to [0, rgb] and
    /// [0, alpha] (f32 bits, infinite for no clamp) where entry i is set.
    Output([Option<(u32, u32, i8)>; 4]),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    vs: u64,
    vs_extra: u16,
    vs_variant: Variant,
    ps: u64,
    ps_variant: Variant,
    topology: wgpu::PrimitiveTopology,
    /// Colour target i: format, blend and write mask.
    color: [Option<(
        wgpu::TextureFormat,
        Option<wgpu::BlendState>,
        wgpu::ColorWrites,
    )>; 4],
    depth: Option<(
        wgpu::TextureFormat,
        state::Depth,
        Option<wgpu::StencilState>,
    )>,
    cull: (wgpu::FrontFace, Option<wgpu::Face>),
    textures: u32,
}

const CLEAR_WGSL: &str = "
struct ClearValue {
    color: vec4<f32>,
    depth: vec4<f32>,
}

@group(0) @binding(0) var<uniform> clear_value: ClearValue;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, clear_value.depth.x, 1.0);
}

@fragment
fn ps_color() -> @location(0) vec4<f32> {
    return clear_value.color;
}

@fragment
fn ps_depth() {
}
";

/// Replays recorded command lists on a headless wgpu device.
pub struct Drawer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Adapter name and backend, for logs.
    pub adapter: String,
    /// Bound on every GPU wait and buffer map.
    pub timeout: Duration,
    regs: RegisterFile,
    vs: Option<Program>,
    ps: Option<Program>,
    translated: HashMap<(u64, bool), Result<Translated, String>>,
    /// Keyed by program hash, stage, extra vertex outputs and variant (the
    /// rectangle list wrapper or the pixel output bias and clamp).
    modules: HashMap<(u64, bool, u16, Variant), wgpu::ShaderModule>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    texture_layouts: HashMap<u32, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    group0: wgpu::BindGroupLayout,
    clear_layout: wgpu::BindGroupLayout,
    clear_module: wgpu::ShaderModule,
    clear_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    targets: HashMap<RenderTargetKey, wgpu::Texture>,
    /// Destination endian of the last 32bpp resolve to each guest address,
    /// so a swap reads its front buffer in the byte order it was written.
    resolved_endian: HashMap<u32, u32>,
    /// Decoded guest textures keyed by fetch constant and a hash of the
    /// guest bytes they read, so an unchanged texture is not decoded again.
    texture_cache: HashMap<([u32; 6], u64), wgpu::TextureView>,
    texture_cache_bytes: u64,
    /// 1x1 texture of zeros bound for a slot without a usable texture.
    zero_texture: wgpu::TextureView,
    zero_sampler: wgpu::Sampler,
    /// Whether `Rgba32Float` textures can be sampled with filtering; without
    /// it wide formats are uploaded as `Rgba16Float`.
    float32_filterable: bool,
    /// Compute pipeline of resolve.wgsl, made on the first 32bpp resolve.
    /// Colour, depth and depth + stencil entry points of resolve.wgsl.
    resolve_pipelines: [Option<wgpu::ComputePipeline>; 3],
    /// kD24FS8 is hosted as `Depth32FloatStencil8` (else `Depth32Float`).
    float_stencil: bool,
    timings: Timings,
    uncaptured: Arc<Mutex<Vec<String>>>,
    max_dim: u32,
    max_storage: u64,
}

/// Time since `t`, restarting `t`.
fn lap(t: &mut Instant) -> Duration {
    let now = Instant::now();
    let d = now - *t;
    *t = now;
    d
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    pollster::block_on(f)
}

fn hash_words(words: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    words.hash(&mut h);
    h.finish()
}

/// Fast non-cryptographic hash of guest bytes for the texture cache. A
/// collision only shows a stale texture.
fn hash_bytes(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut chunks = bytes.chunks_exact(8);
    let mut h = bytes.len() as u64;
    for c in &mut chunks {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(K);
    }
    for &b in chunks.remainder() {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }
    h
}

/// Cached textures are dropped wholesale past this many bytes.
const TEXTURE_CACHE_LIMIT: u64 = 256 << 20;

fn to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Adds the epilogue and the extra outputs to a translated vertex shader.
fn patch_vertex_wgsl(wgsl: &str, extra: u16) -> Result<String, String> {
    const POS_FIELD: &str = "    @builtin(position) pos: vec4<f32>,\n";
    const POS_STORE: &str = "    out.pos = o_pos;\n";
    if !wgsl.contains(POS_FIELD) || !wgsl.contains(POS_STORE) {
        return Err("vertex shader output layout not recognised".into());
    }
    let mut fields = POS_FIELD.to_string();
    let mut stores = "    out.pos = xe_epilogue(o_pos);\n".to_string();
    for i in 0..16 {
        if extra & (1 << i) != 0 {
            fields.push_str(&format!("    @location({i}) o{i}: vec4<f32>,\n"));
            stores.push_str(&format!("    out.o{i} = o_interp[{i}];\n"));
        }
    }
    Ok(wgsl
        .replacen(POS_FIELD, &fields, 1)
        .replacen(POS_STORE, &stores, 1)
        + state::EPILOGUE_WGSL)
}

/// Turns a vertex shader already patched by [`patch_vertex_wgsl`] into a
/// rectangle list shader. Host vertex `6r + k` belongs to rectangle `r`,
/// whose three guest indices are `xe_rect_indices[3r..3r + 3]`
/// (`@group(0) @binding(3)`). Corners 0..2 are the guest vertices in their
/// order (the first triangle). The translated body runs for all three
/// guest vertices at every host vertex; the corner opposite the longest edge
/// (squared clip-space XY lengths, ties as Xenia's geometry shader) is the
/// right angle `a`, its successors are `b` and `c`, and the second triangle
/// is `(c, b, d)` with the fourth vertex `d = (b - a) + c` for the position
/// and every output (Xenia's strip `a b c d`, written as a list so both
/// triangles keep the first one's winding). Positions are combined after
/// the epilogue, in host clip space, as Xenia's geometry shader combines
/// its vertex shader's clip-space output.
///
/// The translated `vs_main` becomes `xe_vs_body`, which first resets every
/// module-scope private variable to its zero value, so each call starts
/// like a fresh invocation.
fn rect_list_wgsl(vs: &str, outputs: u16) -> Result<String, String> {
    const ENTRY: &str =
        "@vertex\nfn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {\n";
    if !vs.contains(ENTRY) {
        return Err("vertex shader entry point not recognised".into());
    }
    let mut body = "fn xe_vs_body(vertex_index: u32) -> VsOut {\n".to_string();
    for line in vs.lines() {
        if let Some(decl) = line.strip_prefix("var<private> ") {
            let Some((name, ty)) = decl.strip_suffix(';').and_then(|d| d.split_once(": ")) else {
                return Err(format!("private variable not recognised: {line}"));
            };
            if ty.contains('=') {
                return Err(format!("initialised private variable: {line}"));
            }
            body.push_str(&format!("    {name} = {ty}();\n"));
        }
    }
    let mut fourth = "    o.pos = (b.pos - a.pos) + c.pos;\n".to_string();
    for i in (0..16).filter(|i| outputs & (1 << i) != 0) {
        fourth.push_str(&format!("    o.o{i} = (b.o{i} - a.o{i}) + c.o{i};\n"));
    }
    Ok(vs.replacen(ENTRY, &body, 1)
        + &format!(
            "
@group(0) @binding(3) var<storage, read> xe_rect_indices: array<u32>;

fn xe_len2(v: vec2<f32>) -> f32 {{
    return v.x * v.x + v.y * v.y;
}}

@vertex
fn vs_main(@builtin(vertex_index) host_index: u32) -> VsOut {{
    let first = (host_index / 6u) * 3u;
    let corner = host_index % 6u;
    // Every host vertex runs the body for all three guest vertices, so the
    // shared diagonal's two corners are bit-identical in both triangles
    // (no crack from the compiler optimising two call sites differently).
    var v = array<VsOut, 3>(
        xe_vs_body(xe_rect_indices[first]),
        xe_vs_body(xe_rect_indices[first + 1u]),
        xe_vs_body(xe_rect_indices[first + 2u]),
    );
    // Squared edge lengths: 12, 20, 01.
    let l12 = xe_len2(v[2].pos.xy - v[1].pos.xy);
    let l20 = xe_len2(v[0].pos.xy - v[2].pos.xy);
    let l01 = xe_len2(v[1].pos.xy - v[0].pos.xy);
    var s = 2u;
    if (l12 > l20 && l12 > l01) {{
        s = 0u;
    }} else if (l20 > l01) {{
        s = 1u;
    }}
    let a = v[s];
    let b = v[(s + 1u) % 3u];
    let c = v[(s + 2u) % 3u];
    if (corner < 3u) {{
        return v[corner];
    }}
    if (corner == 3u) {{
        return c;
    }}
    if (corner == 4u) {{
        return b;
    }}
    var o: VsOut;
{fourth}    return o;
}}
"
        ))
}

/// Scales colour `rt` of a translated pixel shader by 2^`bias`
/// (RB_COLOR_INFO's colour exponent bias, all four components, as Xenia's
/// spirv_shader_translator_rb.cc does before the EDRAM format clamp), then
/// clamps it to [0, rgb] / [0, alpha].
fn output_pixel_wgsl(
    ps: &str,
    rt: usize,
    bias: i8,
    rgb: f32,
    alpha: f32,
) -> Result<String, String> {
    let store = format!("    out.c{rt} = o_col[{rt}];\n");
    if !ps.contains(&store) {
        return Err(format!("pixel shader colour output {rt} not recognised"));
    }
    let c = if bias == 0 {
        format!("o_col[{rt}]")
    } else {
        format!("(o_col[{rt}] * {:?})", 2f32.powi(bias.into()))
    };
    // An infinite bound leaves the component as it is (blending a 7e3
    // target takes the source colour unclamped, as a float).
    let rgb = if rgb.is_finite() {
        format!("clamp({c}.xyz, vec3<f32>(0.0), vec3<f32>({rgb:?}))")
    } else {
        format!("{c}.xyz")
    };
    let a = if alpha.is_finite() {
        format!("clamp({c}.w, 0.0, {alpha:?})")
    } else {
        format!("{c}.w")
    };
    Ok(ps.replacen(
        &store,
        &format!("    out.c{rt} = vec4<f32>({rgb}, {a});\n"),
        1,
    ))
}

fn sampler_address(clamp: u8) -> (wgpu::AddressMode, bool) {
    use wgpu::AddressMode as A;
    match clamp {
        0 => (A::Repeat, true),
        1 => (A::MirrorRepeat, true),
        2 => (A::ClampToEdge, true),
        // Mirror-once, half-way and border modes have no plain wgpu
        // equivalent (border needs an optional feature): edge clamp.
        _ => (A::ClampToEdge, false),
    }
}

fn filter(f: u8) -> wgpu::FilterMode {
    if f == 1 {
        wgpu::FilterMode::Linear
    } else {
        wgpu::FilterMode::Nearest
    }
}

impl Drawer {
    /// Opens a headless device: Vulkan or GL, default power preference,
    /// first without and then with `force_fallback_adapter`.
    pub fn new() -> Result<Drawer, Error> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(desc);
        let options = |force_fallback_adapter| wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets: false,
        };
        let adapter = block_on(instance.request_adapter(&options(false)))
            .or_else(|_| block_on(instance.request_adapter(&options(true))))
            .map_err(|_| Error::NoAdapter)?;
        let caps = adapter.get_downlevel_capabilities();
        if !caps.flags.contains(wgpu::DownlevelFlags::VERTEX_STORAGE) {
            return Err(Error::RequestDevice(
                "the adapter has no storage buffers in vertex shaders".into(),
            ));
        }
        let info = adapter.get_info();
        let limits = adapter.limits();
        // kD24FS8 keeps its stencil only where the host has a float depth
        // format with stencil.
        let float_stencil = adapter
            .features()
            .contains(wgpu::Features::DEPTH32FLOAT_STENCIL8);
        let mut required_features = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
        if float_stencil {
            required_features |= wgpu::Features::DEPTH32FLOAT_STENCIL8;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("xenon-draw"),
            required_limits: limits.clone(),
            required_features,
            ..Default::default()
        }))
        .map_err(|e| Error::RequestDevice(e.to_string()))?;
        let uncaptured = Arc::new(Mutex::new(Vec::new()));
        let sink = uncaptured.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock() {
                v.push(e.to_string());
            }
        }));
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("guest window, constants, epilogue, rectangle indices"),
            entries: &[
                entry(0, vf, storage),
                entry(1, vf, uniform),
                entry(2, wgpu::ShaderStages::VERTEX, uniform),
                entry(3, wgpu::ShaderStages::VERTEX, storage),
            ],
        });
        let clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear"),
            entries: &[entry(0, vf, uniform)],
        });
        let clear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()),
        });
        let zero_sampler = device.create_sampler(&Default::default());
        // wgpu zero-initialises new textures, so nothing is written.
        let zero_texture = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("zero texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Ok(Drawer {
            device,
            queue,
            adapter: format!("{} ({:?})", info.name, info.backend),
            timeout: DEFAULT_TIMEOUT,
            regs: RegisterFile::default(),
            vs: None,
            ps: None,
            translated: HashMap::new(),
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            texture_layouts: HashMap::new(),
            group0,
            clear_layout,
            clear_module,
            clear_pipelines: HashMap::new(),
            targets: HashMap::new(),
            resolved_endian: HashMap::new(),
            texture_cache: HashMap::new(),
            texture_cache_bytes: 0,
            zero_texture,
            zero_sampler,
            float32_filterable: required_features.contains(wgpu::Features::FLOAT32_FILTERABLE),
            resolve_pipelines: [None, None, None],
            float_stencil,
            timings: Timings::default(),
            uncaptured,
            max_dim: limits.max_texture_dimension_2d,
            max_storage: limits.max_storage_buffer_binding_size,
        })
    }

    /// The drawer's register file (the state after the commands replayed).
    pub fn registers(&self) -> &RegisterFile {
        &self.regs
    }

    /// Whether depth, 16-bit unorm and 32-bit float textures upload as
    /// `Rgba32Float` (full precision) rather than `Rgba16Float`.
    pub fn float32_textures(&self) -> bool {
        self.float32_filterable
    }

    /// Keys of every render target created so far.
    pub fn render_targets(&self) -> Vec<RenderTargetKey> {
        self.targets.keys().copied().collect()
    }

    /// Replays `commands` in order. Register writes, shader loads, draws,
    /// resolves, swaps and MEM_WRITE packets are executed; other commands
    /// (waits, events, interrupts, queries) are ignored.
    pub fn replay(
        &mut self,
        commands: &[Command],
        mem: &mut dyn GuestMemory,
    ) -> Result<Report, Error> {
        let mut report = Report::default();
        self.timings = Timings::default();
        for c in commands {
            let done = match c {
                Command::WriteRegisters { base, values } => {
                    for (i, &v) in values.iter().enumerate() {
                        self.regs.set(base.wrapping_add(i as u32), v);
                    }
                    Ok(())
                }
                Command::LoadShader {
                    stage,
                    start,
                    words,
                    ..
                } => {
                    if *start != 0 {
                        report.warn(format!(
                            "shader load at instruction offset {start} treated as 0"
                        ));
                    }
                    let p = Some(Program {
                        hash: hash_words(words),
                        words: words.clone(),
                    });
                    match stage {
                        ShaderStage::Vertex => self.vs = p,
                        ShaderStage::Pixel => self.ps = p,
                    }
                    Ok(())
                }
                Command::Draw(d) => self.draw(d, mem, &mut report).map(|drawn| {
                    report.draws += u32::from(drawn);
                }),
                Command::Resolve(r) => self.resolve(r, mem, &mut report).map(|done| {
                    report.resolves += u32::from(done);
                }),
                Command::Swap {
                    signature_ok,
                    frontbuffer,
                    width,
                    height,
                } => {
                    if !signature_ok {
                        report.warn("swap packet without the SWAP signature");
                    }
                    let t = Instant::now();
                    let frame = self.front_buffer(*frontbuffer, *width, *height, mem);
                    self.timings.swap += t.elapsed();
                    match frame {
                        Ok(f) => report.frames.push(f),
                        Err(e) => report.warn(e),
                    }
                    Ok(())
                }
                Command::MemWrite {
                    addr,
                    endian,
                    values,
                } => {
                    for (i, &v) in values.iter().enumerate() {
                        let a = addr.wrapping_add(4 * i as u32);
                        if !mem.write_bytes(a, &endian.to_guest_bytes(v)) {
                            report.warn(format!("MEM_WRITE to unmapped 0x{a:08X}"));
                        }
                    }
                    Ok(())
                }
                _ => Ok(()),
            };
            match done {
                Ok(()) => {}
                Err(Skip::Unsupported(why)) => report.unsupported(why),
                Err(Skip::Fatal(e)) => return Err(e),
            }
            self.check_uncaptured()?;
        }
        // Counts made where only the report is mutable.
        let t = &report.timings;
        let (uploaded, tex_bytes, guest_bytes) =
            (t.textures_uploaded, t.texture_bytes, t.guest_bytes);
        report.timings = Timings {
            textures_uploaded: uploaded,
            texture_bytes: tex_bytes,
            guest_bytes,
            ..std::mem::take(&mut self.timings)
        };
        Ok(report)
    }

    fn check_uncaptured(&self) -> Result<(), Error> {
        let mut v = self
            .uncaptured
            .lock()
            .map_err(|e| Error::Gpu(e.to_string()))?;
        if v.is_empty() {
            Ok(())
        } else {
            Err(Error::Gpu(std::mem::take(&mut *v).join("; ")))
        }
    }

    /// Reads `w` x `h` pixels at (`x`, `y`) of an 8_8_8_8 colour render
    /// target as RGBA8 rows. Other colour formats:
    /// [`Drawer::read_render_target_raw`].
    pub fn read_render_target(
        &self,
        key: RenderTargetKey,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, Error> {
        if key.format != TargetFormat::Rgba8 {
            return Err(Error::Unsupported(format!(
                "{:?} render target readback as RGBA8",
                key.format
            )));
        }
        self.read_render_target_raw(key, x, y, w, h)
    }

    /// Reads `w` x `h` pixels at (`x`, `y`) of a colour render target as
    /// host texels, rows top to bottom, little-endian: 4 bytes per pixel for
    /// `Rgba8` (R first) and `Rgb10a2` (one word, R in bits 0..10), 8 for
    /// `Float7e3` and `Rgba16Float` (four halves, R first).
    pub fn read_render_target_raw(
        &self,
        key: RenderTargetKey,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, Error> {
        if key.format.is_depth() {
            return Err(Error::Unsupported("depth render target readback".into()));
        }
        let tex = self.targets.get(&key).ok_or(Error::NoRenderTarget(key))?;
        self.read_texture(tex, x, y, w, h)
    }

    /// Waits for all submitted GPU work, bounded by the drawer's timeout.
    fn wait_idle(&self) -> Result<(), Error> {
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(self.timeout),
        }) {
            Ok(_) => Ok(()),
            Err(wgpu::PollError::Timeout) => Err(Error::Timeout),
            Err(e) => Err(Error::Gpu(e.to_string())),
        }
    }

    fn read_texture(
        &self,
        tex: &wgpu::Texture,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, Error> {
        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }
        if x.saturating_add(w) > tex.width() || y.saturating_add(h) > tex.height() {
            return Err(Error::Unsupported(format!(
                "readback {w}x{h} at ({x}, {y}) outside the {}x{} target",
                tex.width(),
                tex.height()
            )));
        }
        let texel = tex.format().block_copy_size(None).unwrap_or(4);
        let row = w * texel;
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let index = self.queue.submit([enc.finish()]);
        let data = self.map_read(&buf, index)?;
        let mut out = Vec::with_capacity((row * h) as usize);
        for r in 0..h as usize {
            out.extend_from_slice(&data[r * padded as usize..][..row as usize]);
        }
        Ok(out)
    }

    /// Waits for submission `index` and copies out the mapped `buf`.
    fn map_read(&self, buf: &wgpu::Buffer, index: wgpu::SubmissionIndex) -> Result<Vec<u8>, Error> {
        let slot = Arc::new(Mutex::new(None));
        let sink = slot.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if let Ok(mut s) = sink.lock() {
                *s = Some(r);
            }
        });
        let deadline = Instant::now() + self.timeout;
        let mut poll = wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(self.timeout),
        };
        loop {
            match self.device.poll(poll) {
                Ok(_) => {}
                Err(wgpu::PollError::Timeout) => return Err(Error::Timeout),
                Err(e) => return Err(Error::Gpu(e.to_string())),
            }
            match slot.lock().map_err(|e| Error::Gpu(e.to_string()))?.take() {
                Some(Ok(())) => break,
                Some(Err(e)) => return Err(Error::Gpu(e.to_string())),
                None if Instant::now() >= deadline => return Err(Error::Timeout),
                None => std::thread::sleep(Duration::from_millis(1)),
            }
            poll = wgpu::PollType::Poll;
        }
        let data = buf
            .slice(..)
            .get_mapped_range()
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let out = data.to_vec();
        drop(data);
        buf.unmap();
        self.check_uncaptured()?;
        Ok(out)
    }

    fn target(&mut self, key: RenderTargetKey) -> Result<wgpu::Texture, Skip> {
        if key.pitch == 0 || key.pitch > self.max_dim {
            return unsupported(format!("surface pitch {}", key.pitch));
        }
        let host_format = self.host_format(key.format);
        let (device, height) = (&self.device, RENDER_TARGET_HEIGHT.min(self.max_dim));
        Ok(self
            .targets
            .entry(key)
            .or_insert_with(|| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("render target"),
                    size: wgpu::Extent3d {
                        width: key.pitch,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: host_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
            .clone())
    }

    /// Whether `kD24FS8` keeps its stencil (the host has
    /// `Depth32FloatStencil8`); without it the stencil test on such a
    /// target is skipped with a warning and resolves write stencil 0.
    pub fn float_stencil(&self) -> bool {
        self.float_stencil
    }

    /// The host texture format of a render target.
    fn host_format(&self, format: TargetFormat) -> wgpu::TextureFormat {
        match format {
            TargetFormat::D24FS8 if self.float_stencil => wgpu::TextureFormat::Depth32FloatStencil8,
            f => f.wgpu(),
        }
    }

    fn translate(&mut self, p: &Program, stage: ShaderStage) -> Result<Translated, Skip> {
        let vertex = stage == ShaderStage::Vertex;
        let t = self.translated.entry((p.hash, vertex)).or_insert_with(|| {
            let s = if vertex {
                xenon_shader::ShaderStage::Vertex
            } else {
                xenon_shader::ShaderStage::Pixel
            };
            xenon_shader::translate(&p.words, s).map_err(|e| e.to_string())
        });
        match t {
            Ok(t) => Ok(t.clone()),
            Err(e) => unsupported(format!(
                "{} shader: {e}",
                if vertex { "vertex" } else { "pixel" }
            )),
        }
    }

    /// Runs `f` inside a validation error scope; a validation error becomes
    /// `Skip::Unsupported(what: error)`.
    fn validated<T>(&self, what: &str, f: impl FnOnce(&wgpu::Device) -> T) -> Result<T, Skip> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let v = f(&self.device);
        match block_on(scope.pop()) {
            None => Ok(v),
            Some(e) => unsupported(format!("{what}: {e}")),
        }
    }

    fn module(
        &mut self,
        hash: u64,
        vertex: bool,
        extra: u16,
        variant: Variant,
        t: &Translated,
    ) -> Result<wgpu::ShaderModule, Skip> {
        let key = (hash, vertex, extra, variant);
        if let Some(m) = self.modules.get(&key) {
            return Ok(m.clone());
        }
        let src = match (vertex, variant) {
            (true, Variant::RectList) => patch_vertex_wgsl(&t.wgsl, extra)
                .and_then(|vs| rect_list_wgsl(&vs, t.info.interpolators_written | extra)),
            (true, _) => patch_vertex_wgsl(&t.wgsl, extra),
            (false, Variant::Output(c)) => {
                c.iter()
                    .enumerate()
                    .try_fold(t.wgsl.clone(), |ps, (rt, c)| match c {
                        Some((rgb, a, bias)) => output_pixel_wgsl(
                            &ps,
                            rt,
                            *bias,
                            f32::from_bits(*rgb),
                            f32::from_bits(*a),
                        ),
                        None => Ok(ps),
                    })
            }
            (false, _) => Ok(t.wgsl.clone()),
        }
        .map_err(Skip::Unsupported)?;
        let m = self.validated("shader module", |d| {
            d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if vertex { "vertex" } else { "pixel" }),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        })?;
        self.modules.insert(key, m.clone());
        Ok(m)
    }

    fn texture_layout(&mut self, mask: u32) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let (device, group0) = (&self.device, &self.group0);
        self.texture_layouts
            .entry(mask)
            .or_insert_with(|| {
                let mut entries = Vec::new();
                for n in (0..32).filter(|n| mask & (1 << n) != 0) {
                    let vis = wgpu::ShaderStages::VERTEX_FRAGMENT;
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: 2 * n,
                        visibility: vis,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: 2 * n + 1,
                        visibility: vis,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    });
                }
                let g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("textures"),
                    entries: &entries,
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("draw"),
                    bind_group_layouts: &[Some(group0), Some(&g1)],
                    immediate_size: 0,
                });
                (g1, layout)
            })
            .clone()
    }

    /// Uploads the guest memory the vertex fetch constants in `mask` point
    /// at, as one window. Returns the buffer and the window's base address.
    fn guest_window(
        &self,
        mask: u128,
        mem: &dyn GuestMemory,
        report: &mut Report,
    ) -> Result<(wgpu::Buffer, u32), Skip> {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for k in (0..96).filter(|k| mask & (1u128 << k) != 0) {
            let vf = self.regs.vertex_fetch(k);
            if vf.fetch_type != FetchConstantType::Vertex {
                report.warn(format!("vertex fetch constant {k} is not a vertex fetch"));
            }
            // Physical, as the translated shader masks fetch addresses.
            let a = u64::from(vf.address & PHYSICAL_MASK);
            lo = lo.min(a);
            hi = hi.max(a + 4 * u64::from(vf.size_dwords));
        }
        report.timings.guest_bytes += hi.saturating_sub(lo);
        let (base, bytes) = if lo >= hi {
            (0, vec![0; 4])
        } else {
            if hi - lo > self.max_storage {
                return unsupported(format!(
                    "vertex data window of {} bytes is larger than the storage buffer limit",
                    hi - lo
                ));
            }
            let Some(b) = mem.read_bytes(lo as u32, (hi - lo) as usize) else {
                return unsupported(format!(
                    "vertex data at 0x{lo:08X}..0x{hi:08X} not readable"
                ));
            };
            (lo as u32, b)
        };
        Ok((
            self.buffer("guest window", &bytes, wgpu::BufferUsages::STORAGE),
            base,
        ))
    }

    fn buffer(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let size = (bytes.len() as u64).max(4).next_multiple_of(4);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut padded = bytes.to_vec();
        padded.resize(size as usize, 0);
        self.queue.write_buffer(&buf, 0, &padded);
        buf
    }

    /// The `Consts` uniform of the shader binding contract.
    fn consts(&self, window_base: u32) -> Vec<u8> {
        let r = &self.regs;
        let block = |base: u32, n: u32| (0..n).map(move |i| r.get(base + i));
        let mut v: Vec<u32> = block(regs::SHADER_CONSTANT_FLOAT, 2048)
            .chain(block(regs::SHADER_CONSTANT_FETCH, 192))
            .chain(block(regs::SHADER_CONSTANT_BOOL, 8))
            .chain(block(regs::SHADER_CONSTANT_LOOP, 32))
            .collect();
        v.extend([window_base, r.get(regs::VGT_INDX_OFFSET), 0, 0]);
        to_bytes(&v)
    }

    /// Texture and sampler for fetch constant slot `n`. A slot without a
    /// texture fetch constant, or whose texture cannot be read or decoded,
    /// gets a 1x1 texture of zeros (as Xenia binds its null image) and a
    /// warning, so the draw still happens.
    fn texture(
        &mut self,
        n: u32,
        mem: &dyn GuestMemory,
        report: &mut Report,
    ) -> Result<(wgpu::TextureView, wgpu::Sampler), Skip> {
        let fetch = xenon_texture::TextureFetch::decode(self.regs.fetch_constant(n));
        if fetch.fetch_type != 2 {
            report.warn("texture slot without a texture fetch constant drawn with zeros");
            return Ok((self.zero_texture.clone(), self.zero_sampler.clone()));
        }
        let sampler = self.sampler(&fetch, report);
        let view = self
            .guest_texture(n, &fetch, mem, report)
            .unwrap_or_else(|why| {
                report.warn(format!("{why}; drawn with zeros"));
                self.zero_texture.clone()
            });
        Ok((view, sampler))
    }

    /// Host texture of fetch constant slot `n`, from the cache or decoded.
    /// Formats whose values do not fit 8-bit unorm upload as floats.
    fn guest_texture(
        &mut self,
        n: u32,
        fetch: &xenon_texture::TextureFetch,
        mem: &dyn GuestMemory,
        report: &mut Report,
    ) -> Result<wgpu::TextureView, String> {
        let addr = fetch.base_address & PHYSICAL_MASK;
        let bytes = xenon_texture::bytes_needed(fetch)
            .and_then(|len| {
                mem.read_bytes(addr, len)
                    .ok_or(xenon_texture::Error::TooShort {
                        needed: len,
                        got: 0,
                    })
            })
            .map_err(|e| e.to_string())?;
        let key = (self.regs.fetch_constant(n), hash_bytes(&bytes));
        if let Some(view) = self.texture_cache.get(&key) {
            return Ok(view.clone());
        }
        let host = host_texture_format(fetch.format, self.float32_filterable);
        let (width, height, data) = if host == wgpu::TextureFormat::Rgba8Unorm {
            let img = xenon_texture::decode_base_level(fetch, &bytes).map_err(|e| e.to_string())?;
            (img.width, img.height, img.rgba8)
        } else {
            let img =
                xenon_texture::decode_base_level_f32(fetch, &bytes).map_err(|e| e.to_string())?;
            let data = if host == wgpu::TextureFormat::Rgba32Float {
                img.rgba.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                img.rgba
                    .iter()
                    .flat_map(|&v| formats::f32_to_half(v).to_le_bytes())
                    .collect()
            };
            (img.width, img.height, data)
        };
        if width > self.max_dim || height > self.max_dim {
            return Err(format!("texture {width}x{height}"));
        }
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        report.timings.textures_uploaded += 1;
        report.timings.texture_bytes += data.len() as u64;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("guest texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: host,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(data.len() as u32 / height),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = tex.create_view(&Default::default());
        let len = data.len() as u64;
        if self.texture_cache_bytes + len > TEXTURE_CACHE_LIMIT {
            self.texture_cache.clear();
            self.texture_cache_bytes = 0;
        }
        self.texture_cache_bytes += len;
        self.texture_cache.insert(key, view.clone());
        Ok(view)
    }

    fn sampler(&self, fetch: &xenon_texture::TextureFetch, report: &mut Report) -> wgpu::Sampler {
        let mut addr_mode = |c: u8| {
            let (m, exact) = sampler_address(c);
            if !exact {
                report.warn(format!("texture clamp mode {c} drawn as clamp to edge"));
            }
            m
        };
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("guest sampler"),
            address_mode_u: addr_mode(fetch.clamp[0]),
            address_mode_v: addr_mode(fetch.clamp[1]),
            address_mode_w: addr_mode(fetch.clamp[2]),
            mag_filter: filter(fetch.mag_filter),
            min_filter: filter(fetch.min_filter),
            ..Default::default()
        })
    }

    /// Draws `d`; Ok(false) when the state makes it draw nothing.
    fn draw(&mut self, d: &Draw, mem: &dyn GuestMemory, report: &mut Report) -> Result<bool, Skip> {
        let color_on = match self.regs.get(regs::RB_MODECONTROL) & 7 {
            4 => true,
            5 => false,
            0 => return Ok(false),
            m => return unsupported(format!("draw with edram_mode {m}")),
        };
        // Xenia does not rasterize with surface pitch 0
        // (IsRasterizationPotentiallyDone in draw_util.cc); Direct3D leaves
        // RB_SURFACE_INFO at 0 until its first render target setup.
        if self.regs.get(regs::RB_SURFACE_INFO) & 0x3FFF == 0 {
            report.warn("draws with surface pitch 0 skipped (no rasterization, as Xenia)");
            return Ok(false);
        }
        if self.regs.get(regs::VGT_OUTPUT_PATH_CNTL) & 3 != 0 {
            return unsupported("tessellated draws");
        }
        let (topology, expand) = indices::topology(d.primitive).map_err(Skip::Unsupported)?;
        let (Some(vs), Some(ps)) = (self.vs.clone(), self.ps.clone()) else {
            return unsupported("draw without both shaders loaded");
        };
        let mut t = Instant::now();
        let vs_t = self.translate(&vs, ShaderStage::Vertex)?;
        let ps_t = self.translate(&ps, ShaderStage::Pixel)?;
        let surface = self.regs.get(regs::RB_SURFACE_INFO);
        // shortcut: MSAA targets are drawn with one sample per pixel (the
        // resolve then copies that sample); add samples if edges matter.
        if (surface >> 16) & 3 != 0 {
            report.warn("MSAA surfaces drawn and resolved single-sampled");
        }
        // Render target i is drawn when the pixel shader writes oC<i> and
        // RB_COLOR_MASK enables any of its channels (Xenia
        // GetNormalizedColorMask).
        let color_mask = self.regs.get(regs::RB_COLOR_MASK);
        let mut color: [Option<RenderTargetKey>; 4] = [None; 4];
        for rt in 0..4 {
            let written = ps_t.info.colour_targets_written & (1 << rt) != 0;
            if !color_on || !written || state::write_mask(color_mask, rt as u32).is_empty() {
                continue;
            }
            let key = RenderTargetKey::color(&self.regs, rt).map_err(Skip::Unsupported)?;
            if color
                .iter()
                .flatten()
                .any(|k| k.base_tiles == key.base_tiles)
            {
                report.warn("two render targets of one draw at the same EDRAM base; the later one is not drawn");
                continue;
            }
            color[rt] = Some(key);
        }
        let depth_control = self.regs.get(regs::RB_DEPTHCONTROL);
        let depth = state::depth(depth_control);
        let depth_key = depth.used().then(|| RenderTargetKey::depth(&self.regs));
        if color.iter().all(Option::is_none) && depth_key.is_none() {
            return Ok(false);
        }
        let rect_list = expand == indices::Expand::Rects;
        // Rectangle lists are not polygonal for Xenia (IsPrimitivePolygonal):
        // no face culling.
        let polygonal = !rect_list
            && matches!(
                topology,
                wgpu::PrimitiveTopology::TriangleList | wgpu::PrimitiveTopology::TriangleStrip
            );
        let cull = match state::cull(self.regs.get(regs::PA_SU_SC_MODE_CNTL)) {
            Some(c) if polygonal => c,
            Some(c) => (c.0, None),
            None if polygonal => return Ok(false),
            None => (wgpu::FrontFace::Ccw, None),
        };
        let mut stencil = state::stencil(
            depth_control,
            self.regs.get(regs::RB_STENCILREFMASK),
            self.regs.get(regs::RB_STENCILREFMASK_BF),
            polygonal,
            cull.1 == Some(wgpu::Face::Front),
        );
        let depth_format = depth_key.map(|k| self.host_format(k.format));
        if stencil.is_some() && !depth_format.is_some_and(|f| f.has_stencil_aspect()) {
            // shortcut: no host float depth format with stencil; needs
            // DEPTH32FLOAT_STENCIL8 or a separate stencil emulation.
            report.warn("stencil test on a 20e4 depth target without host stencil ignored");
            stencil = None;
        }
        let mut blends = [None; 4];
        for rt in (0..4).filter(|&rt| color[rt].is_some()) {
            blends[rt] = state::blend(self.regs.get(regs::RB_BLENDCONTROL_RT[rt]))
                .map_err(Skip::Unsupported)?;
            // shortcut: wgpu blends R32Float only with FLOAT32_BLENDABLE;
            // request it if a game blends into a 32-bit float target.
            if blends[rt].is_some() && color[rt].map(|k| k.format) == Some(TargetFormat::Float32) {
                report.warn("blending into a 32-bit float render target drawn without blending");
                blends[rt] = None;
            }
        }
        let outputs: [Option<(u32, u32, i8)>; 4] = std::array::from_fn(|rt| {
            let k = color[rt]?;
            // RB_COLOR_INFO bits 20..25: signed colour exponent bias.
            let bias = ((self.regs.get(regs::RB_COLOR_INFO_RT[rt]) << 6) as i32 >> 26) as i8;
            let clamp = formats::output_clamp(k.format, blends[rt].is_some());
            if bias == 0 && clamp.is_none() {
                return None;
            }
            let (rgb, a) = clamp.unwrap_or((f32::INFINITY, f32::INFINITY));
            Some((rgb.to_bits(), a.to_bits(), bias))
        });
        let ps_variant = if outputs.iter().any(Option::is_some) {
            Variant::Output(outputs)
        } else {
            Variant::Plain
        };
        let vs_variant = if rect_list {
            Variant::RectList
        } else {
            Variant::Plain
        };
        // One blend constant for the pass: from the first blended target.
        let blend_rt = (0..4).find(|&rt| blends[rt].is_some()).unwrap_or(0);
        let (blend_rgba, exact) = state::blend_constant(
            self.regs.get(regs::RB_BLENDCONTROL_RT[blend_rt]),
            [
                regs::RB_BLEND_RED,
                regs::RB_BLEND_GREEN,
                regs::RB_BLEND_BLUE,
                regs::RB_BLEND_ALPHA,
            ]
            .map(|r| f32::from_bits(self.regs.get(r))),
        );
        if blends[blend_rt].is_some() && !exact {
            report.warn("blend uses the constant colour and its alpha in the colour channel; drawn with the colour");
        }

        let mut idx =
            indices::guest_indices(mem, &d.source, d.index_count).map_err(Skip::Unsupported)?;
        let strip = matches!(
            topology,
            wgpu::PrimitiveTopology::TriangleStrip | wgpu::PrimitiveTopology::LineStrip
        );
        let reset_on = self.regs.get(regs::PA_SU_SC_MODE_CNTL) & (1 << 21) != 0;
        if reset_on
            && matches!(d.source, xenon_gpu::IndexSource::Dma { .. })
            && indices::mark_resets(&mut idx, self.regs.get(regs::VGT_MULTI_PRIM_IB_RESET_INDX))
            && !strip
        {
            return unsupported("primitive reset outside strips");
        }
        let idx = indices::expand(idx, expand);
        if idx.is_empty() {
            return Ok(false);
        }

        let mut color_tex: [Option<wgpu::Texture>; 4] = Default::default();
        for rt in 0..4 {
            color_tex[rt] = color[rt].map(|k| self.target(k)).transpose()?;
        }
        let depth_tex = depth_key.map(|k| self.target(k)).transpose()?;
        let any = color_tex
            .iter()
            .flatten()
            .next()
            .or(depth_tex.as_ref())
            .expect("one target");
        let (w, h) = (any.width(), any.height());
        let (sx, sy, sw, sh) = state::scissor(&self.regs, w, h);
        if sw == 0 || sh == 0 {
            return Ok(false);
        }

        let info = |a: &ShaderInfo, b: &ShaderInfo| {
            (
                a.vertex_fetch_consts | b.vertex_fetch_consts,
                a.textures_used | b.textures_used,
            )
        };
        let (vf_mask, tex_mask) = info(&vs_t.info, &ps_t.info);
        if (vs_t.info.texture_filter_overrides | ps_t.info.texture_filter_overrides) != 0 {
            report.warn("texture fetch filter overrides ignored");
        }
        let extra = ps_t.info.interpolators_read & !vs_t.info.interpolators_written;
        let vs_mod = self.module(vs.hash, true, extra, vs_variant, &vs_t)?;
        let ps_mod = self.module(ps.hash, false, 0, ps_variant, &ps_t)?;
        let (g1_layout, layout) = self.texture_layout(tex_mask);

        let key = PipelineKey {
            vs: vs.hash,
            vs_extra: extra,
            vs_variant,
            ps: ps.hash,
            ps_variant,
            topology,
            color: std::array::from_fn(|rt| {
                color[rt].map(|k| {
                    (
                        k.format.wgpu(),
                        blends[rt],
                        state::write_mask(color_mask, rt as u32),
                    )
                })
            }),
            depth: depth_format.map(|f| (f, depth, stencil.as_ref().map(|s| s.0.clone()))),
            cull,
            textures: tex_mask,
        };
        let pipeline = match self.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let k = key.clone();
                let p = self.validated("render pipeline", |dev| {
                    dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("draw"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &vs_mod,
                            entry_point: Some("vs_main"),
                            compilation_options: Default::default(),
                            buffers: &[],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology,
                            strip_index_format: strip.then_some(wgpu::IndexFormat::Uint32),
                            front_face: k.cull.0,
                            cull_mode: k.cull.1,
                            ..Default::default()
                        },
                        depth_stencil: k.depth.map(|(format, d, st)| wgpu::DepthStencilState {
                            format,
                            depth_write_enabled: Some(d.write),
                            depth_compare: Some(d.compare),
                            stencil: st.unwrap_or_default(),
                            bias: Default::default(),
                        }),
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &ps_mod,
                            entry_point: Some("ps_main"),
                            compilation_options: Default::default(),
                            targets: &k.color.map(|c| {
                                c.map(|(format, blend, write_mask)| wgpu::ColorTargetState {
                                    format,
                                    blend,
                                    write_mask,
                                })
                            }),
                        }),
                        multiview_mask: None,
                        cache: None,
                    })
                })?;
                self.pipelines.insert(key, p.clone());
                self.timings.pipelines_created += 1;
                p
            }
        };
        self.timings.shaders += lap(&mut t);

        let (window, window_base) = self.guest_window(vf_mask, mem, report)?;
        let consts = self.buffer(
            "consts",
            &self.consts(window_base),
            wgpu::BufferUsages::UNIFORM,
        );
        let epi = self.buffer(
            "epilogue",
            &to_bytes(&state::epilogue(&self.regs, w, h)),
            wgpu::BufferUsages::UNIFORM,
        );
        self.timings.guest_data += lap(&mut t);
        let mut textures = Vec::new();
        for n in (0..32).filter(|n| tex_mask & (1 << n) != 0) {
            textures.push((n, self.texture(n, mem, report)?));
        }
        self.timings.textures += lap(&mut t);
        // Rectangle lists read their guest indices in the vertex shader and
        // draw 6 host vertices per rectangle without an index buffer.
        let index_buf = self.buffer("indices", &to_bytes(&idx), wgpu::BufferUsages::INDEX);
        let rect_indices = self.buffer(
            "rectangle indices",
            &if rect_list {
                to_bytes(&idx)
            } else {
                Vec::new()
            },
            wgpu::BufferUsages::STORAGE,
        );

        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let g0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw group 0"),
            layout: &self.group0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: window.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: consts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: epi.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rect_indices.as_entire_binding(),
                },
            ],
        });
        let mut entries = Vec::new();
        for (n, (view, sampler)) in &textures {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 * n,
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 2 * n + 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        let g1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw group 1"),
            layout: &g1_layout,
            entries: &entries,
        });
        let color_views = color_tex
            .each_ref()
            .map(|t| t.as_ref().map(|t| t.create_view(&Default::default())));
        let depth_view = depth_tex
            .as_ref()
            .map(|t| t.create_view(&Default::default()));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &color_views.each_ref().map(|v| {
                    v.as_ref().map(|view| wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                }),
                depth_stencil_attachment: depth_view.as_ref().map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: depth_format
                            .is_some_and(|f| f.has_stencil_aspect())
                            .then_some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                    }
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(sx, sy, sw, sh);
            if let Some((_, reference)) = &stencil {
                pass.set_stencil_reference(*reference);
            }
            pass.set_blend_constant(wgpu::Color {
                r: f64::from(blend_rgba[0]),
                g: f64::from(blend_rgba[1]),
                b: f64::from(blend_rgba[2]),
                a: f64::from(blend_rgba[3]),
            });
            if rect_list {
                pass.draw(0..idx.len() as u32 * 2, 0..1);
            } else {
                pass.draw_indexed(0..idx.len() as u32, 0, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        self.timings.submit += lap(&mut t);
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        Ok(true)
    }

    fn clear_pipeline(
        &mut self,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::RenderPipeline, Skip> {
        if let Some(p) = self.clear_pipelines.get(&format) {
            return Ok(p.clone());
        }
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("clear"),
                bind_group_layouts: &[Some(&self.clear_layout)],
                immediate_size: 0,
            });
        let is_depth = format.is_depth_stencil_format();
        let stencil_face = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Replace,
            depth_fail_op: wgpu::StencilOperation::Replace,
            pass_op: wgpu::StencilOperation::Replace,
        };
        let module = &self.clear_module;
        let color_target = [Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let p = self.validated("clear pipeline", |dev| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("clear"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: is_depth.then(|| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: if format.has_stencil_aspect() {
                        wgpu::StencilState {
                            front: stencil_face,
                            back: stencil_face,
                            read_mask: 0xFF,
                            write_mask: 0xFF,
                        }
                    } else {
                        Default::default()
                    },
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                // Depth clears still get a fragment shader, with no outputs.
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(if is_depth { "ps_depth" } else { "ps_color" }),
                    compilation_options: Default::default(),
                    targets: if is_depth { &[] } else { &color_target },
                }),
                multiview_mask: None,
                cache: None,
            })
        })?;
        self.clear_pipelines.insert(format, p.clone());
        Ok(p)
    }

    /// Clears `rect` of a render target. `color` is RGBA 0..1; for depth
    /// targets `depth` and `stencil` are used.
    fn clear(
        &mut self,
        key: RenderTargetKey,
        rect: resolve::Rect,
        color: [f32; 4],
        depth: f32,
        stencil: u32,
    ) -> Result<(), Skip> {
        let tex = self.target(key)?;
        let rect = resolve::Rect {
            x0: rect.x0.min(tex.width()),
            y0: rect.y0.min(tex.height()),
            x1: rect.x1.min(tex.width()),
            y1: rect.y1.min(tex.height()),
        };
        if rect.is_empty() {
            return Ok(());
        }
        let format = self.host_format(key.format);
        let pipeline = self.clear_pipeline(format)?;
        let mut v: Vec<u32> = color.iter().map(|c| c.to_bits()).collect();
        v.extend([depth.to_bits(), 0, 0, 0]);
        let ub = self.buffer("clear value", &to_bytes(&v), wgpu::BufferUsages::UNIFORM);
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear"),
            layout: &self.clear_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            }],
        });
        let view = tex.create_view(&Default::default());
        let is_depth = format.is_depth_stencil_format();
        let keep = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let keep_color = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: keep_color,
        })];
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: if is_depth { &[] } else { &color_attachment },
                depth_stencil_attachment: is_depth.then(|| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(keep),
                        stencil_ops: format.has_stencil_aspect().then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_scissor_rect(rect.x0, rect.y0, rect.width(), rect.height());
            pass.set_stencil_reference(stencil);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        match block_on(scope.pop()) {
            Some(e) => Err(Skip::Fatal(Error::Gpu(e.to_string()))),
            None => Ok(()),
        }
    }

    /// Performs a resolve: copy, then the colour and depth clears. Ok(false)
    /// when the rectangle is empty.
    ///
    /// Copy commands: `Raw` and `Convert` both convert from the source to
    /// the destination format, as in Xenia, whose resolve shaders do not
    /// read the command (the copy is bit-exact anyway when the formats
    /// share a layout and the exponent bias is 0). `Null` copies nothing
    /// and still clears; Xenia's GetResolveInfo refuses `Null` and
    /// `ConstantOne` and drops the whole resolve, clears included. This
    /// drawer reports `ConstantOne` as unsupported and does nothing for it.
    fn resolve(
        &mut self,
        r: &Resolve,
        mem: &mut dyn GuestMemory,
        report: &mut Report,
    ) -> Result<bool, Skip> {
        if r.msaa_samples != 0 {
            report.warn("MSAA surfaces drawn and resolved single-sampled");
        }
        let height = RENDER_TARGET_HEIGHT.min(self.max_dim);
        let (rect, from_vertices) = resolve::rect(r, &self.regs, &*mem, height);
        if !from_vertices {
            report.warn("resolve rectangle taken from the window scissor (vertex fetch 0 is not a 6-dword rectangle)");
        }
        if rect.is_empty() {
            return Ok(false);
        }
        match r.copy_command {
            CopyCommand::Raw | CopyCommand::Convert => self.resolve_copy(r, rect, mem)?,
            CopyCommand::Null => {}
            CopyCommand::ConstantOne => return unsupported("resolve copy command ConstantOne"),
        }
        // A depth resolve never clears colour (Xenia draw_util.h
        // IsClearingColor).
        if r.color_clear_enable && !r.is_depth {
            let key = RenderTargetKey::color(&self.regs, r.src_select as usize)
                .map_err(|e| Skip::Unsupported(format!("resolve clear of {e}")))?;
            let value = formats::clear_value(r.color_clear, r.color_clear_lo);
            let t = Instant::now();
            self.clear(key, rect, formats::clear_color(key.format, value), 0.0, 0)?;
            self.timings.clears += t.elapsed();
        }
        if r.depth_clear_enable {
            let key = RenderTargetKey::depth(&self.regs);
            let d = r.depth_clear >> 8;
            let depth = match key.format {
                TargetFormat::D24FS8 => float20e4(d),
                // Not Xenia's UNorm24To32: n / 0xFFFFFF is the value the
                // depth resolve's round(d * 0xFFFFFF) gives back exactly for
                // every 24-bit n, as the EDRAM keeps it.
                _ => (f64::from(d) / f64::from(0xFF_FFFFu32)) as f32,
            };
            let t = Instant::now();
            self.clear(key, rect, [0.0; 4], depth, r.depth_clear & 0xFF)?;
            self.timings.clears += t.elapsed();
        }
        Ok(true)
    }

    /// Colour copy: reads the rectangle back, converts each pixel as
    /// Xenia's resolve shader does ([`formats::edram_rgba`],
    /// [`formats::resolve_pixel`]: exponent bias, red/blue swap, saturate
    /// and round for unorm destinations) and tiles it into guest memory with
    /// the destination endian.
    fn resolve_copy(
        &mut self,
        r: &Resolve,
        rect: resolve::Rect,
        mem: &mut dyn GuestMemory,
    ) -> Result<(), Skip> {
        let format = if r.is_depth {
            if r.src_format & 1 != 0 {
                TargetFormat::D24FS8
            } else {
                TargetFormat::D24S8
            }
        } else {
            let Some(format) = TargetFormat::color(r.src_format) else {
                return unsupported(format!("resolve copy from colour format {}", r.src_format));
            };
            format
        };
        // A depth copy always writes 32bpp words whatever the destination
        // format says (Xenia GetResolveInfo replaces it with k_24_8).
        let dest_bytes = if r.is_depth {
            Some(4)
        } else {
            formats::dest_bytes(r.dest_format)
        };
        let Some(dest_bytes) = dest_bytes else {
            return unsupported(format!(
                "resolve copy to destination format {}",
                r.dest_format
            ));
        };
        if r.dest_array {
            return unsupported("resolve copy to a 3D / array destination");
        }
        // 8in64 is the one endian past 16in32 that write_tiled_texels
        // handles, and only for 64bpp texels.
        if r.dest_endian > 3 && !(dest_bytes == 8 && r.dest_endian == 4) {
            return unsupported(format!("resolve destination endian {}", r.dest_endian));
        }
        if r.dest_pitch == 0 {
            return unsupported("resolve destination pitch 0");
        }
        let key = RenderTargetKey {
            base_tiles: r.src_edram_base,
            pitch: r.surface_pitch,
            format,
        };
        let tex = self.target(key)?;
        let pitch = xenon_texture::tile::align_to_tile(r.dest_pitch);
        let rect = resolve::Rect {
            x1: rect.x1.min(pitch).min(tex.width()),
            ..rect
        };
        if rect.is_empty() {
            return Ok(());
        }
        if dest_bytes == 4 {
            self.resolve_gpu(r, rect, format, &tex, mem)?;
            self.resolved_endian
                .insert(r.dest_base & PHYSICAL_MASK, r.dest_endian);
            return Ok(());
        }
        let mut t = Instant::now();
        self.wait_idle()?;
        self.timings.gpu_wait += lap(&mut t);
        let px = self.read_texture(&tex, rect.x0, rect.y0, rect.width(), rect.height())?;
        self.timings.readback += lap(&mut t);
        let src_bytes = px.len() / (rect.width() * rect.height()) as usize;
        let mut out = vec![0; px.len() / src_bytes * dest_bytes];
        for (src, dst) in px
            .chunks_exact(src_bytes)
            .zip(out.chunks_exact_mut(dest_bytes))
        {
            let rgba = formats::edram_rgba(format, src);
            formats::resolve_pixel(rgba, r.dest_exp_bias, r.dest_swap, r.dest_format, dst);
        }
        let done = resolve::write_tiled_texels(
            mem,
            r.dest_base & PHYSICAL_MASK,
            r.dest_pitch,
            rect,
            &out,
            dest_bytes.trailing_zeros(),
            r.dest_endian,
        )
        .map_err(Skip::Unsupported);
        // Only a resolve that wrote its texels sets the byte order a swap reads.
        if done.is_ok() && dest_bytes == 4 {
            self.resolved_endian
                .insert(r.dest_base & PHYSICAL_MASK, r.dest_endian);
        }
        self.timings.resolve_cpu += lap(&mut t);
        done
    }

    /// 32bpp colour copy on the GPU: resolve.wgsl converts, tiles and
    /// swaps each pixel into a copy of the destination's guest bytes, which
    /// is read back once and written to guest memory. Same result as the CPU
    /// path in [`Self::resolve_copy`].
    fn resolve_gpu(
        &mut self,
        r: &Resolve,
        rect: resolve::Rect,
        format: TargetFormat,
        tex: &wgpu::Texture,
        mem: &mut dyn GuestMemory,
    ) -> Result<(), Skip> {
        if rect.y1 > tex.height() {
            return Err(Error::Unsupported(format!(
                "readback {}x{} at ({}, {}) outside the {}x{} target",
                rect.width(),
                rect.height(),
                rect.x0,
                rect.y0,
                tex.width(),
                tex.height()
            ))
            .into());
        }
        let mut t = Instant::now();
        let base = r.dest_base & PHYSICAL_MASK;
        let pitch = xenon_texture::tile::align_to_tile(r.dest_pitch);
        let extent = xenon_texture::tile::tiled_extent_2d(rect.x1, rect.y1, pitch, 4)
            .map_err(|e| Skip::Unsupported(format!("resolve destination: {e}")))?;
        if u64::from(extent) > self.max_storage {
            return unsupported(format!("resolve destination of {extent} bytes"));
        }
        let Some(guest) = mem.read_bytes(base, extent as usize) else {
            return unsupported(format!("resolve destination 0x{base:08X} is not readable"));
        };
        let src_format = match format {
            TargetFormat::Rgba8 => 0,
            TargetFormat::Rgb10a2 => 1,
            TargetFormat::Float7e3 => 2,
            TargetFormat::Rgba16Float => 3,
            TargetFormat::Float32 => 4,
            TargetFormat::D24S8 => 5,
            TargetFormat::D24FS8 => 6,
        };
        let entry = match format {
            _ if !format.is_depth() => 0,
            _ if self.host_format(format).has_stencil_aspect() => 2,
            _ => 1,
        };
        let params = [
            rect.x0,
            rect.y0,
            rect.width(),
            rect.height(),
            pitch,
            src_format,
            match r.dest_format {
                formats::DEST_8_8_8_8 => 0,
                formats::DEST_2_10_10_10 => 1,
                _ => 2,
            },
            u32::from(r.dest_swap),
            r.dest_endian,
            2f32.powi(r.dest_exp_bias).to_bits(),
            0,
            0,
        ];
        let dest = self.buffer(
            "resolve destination",
            &guest,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniform = self.buffer(
            "resolve params",
            &to_bytes(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolve readback"),
            size: dest.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let device = &self.device;
        let pipeline = self.resolve_pipelines[entry].get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("resolve"),
                source: wgpu::ShaderSource::Wgsl(include_str!("resolve.wgsl").into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resolve"),
                layout: None,
                module: &module,
                entry_point: Some(["main", "depth_main", "depth_stencil_main"][entry]),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let aspect = |aspect| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                ..Default::default()
            })
        };
        let views = match entry {
            0 => vec![(0, aspect(wgpu::TextureAspect::All))],
            1 => vec![(3, aspect(wgpu::TextureAspect::DepthOnly))],
            _ => vec![
                (3, aspect(wgpu::TextureAspect::DepthOnly)),
                (4, aspect(wgpu::TextureAspect::StencilOnly)),
            ],
        };
        let mut entries: Vec<_> = views
            .iter()
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.extend([
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ]);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(rect.width().div_ceil(8), rect.height().div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&dest, 0, &readback, 0, dest.size());
        let index = self.queue.submit([enc.finish()]);
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        let mut out = self.map_read(&readback, index)?;
        self.timings.readback += lap(&mut t);
        out.truncate(extent as usize);
        if !mem.write_bytes(base, &out) {
            return unsupported(format!("resolve destination 0x{base:08X} is not writable"));
        }
        self.timings.resolve_cpu += lap(&mut t);
        Ok(())
    }

    /// Reads the front buffer: a 32x32-tiled 8_8_8_8 surface, pitch = width
    /// rounded up to 32, R in the low byte. Its byte order is the destination
    /// endian of the last resolve to that address (a game may resolve with
    /// any of them), else k8in32.
    fn front_buffer(
        &self,
        addr: u32,
        width: u32,
        height: u32,
        mem: &dyn GuestMemory,
    ) -> Result<Frame, String> {
        use xenon_texture::tile;
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(format!("front buffer size {width}x{height}"));
        }
        let pitch = tile::align_to_tile(width);
        let extent = tile::tiled_extent_2d(width, height, pitch, 4).map_err(|e| e.to_string())?;
        let addr = addr & PHYSICAL_MASK;
        let mut raw = mem
            .read_bytes(addr, extent as usize)
            .ok_or_else(|| format!("front buffer at 0x{addr:08X} not readable"))?;
        let endian = match self.resolved_endian.get(&addr) {
            Some(&e) if e <= 3 => xenon_texture::Endian::from_bits(e),
            _ => xenon_texture::Endian::K8in32,
        };
        xenon_texture::endian::swap_in_place(&mut raw, endian);
        let rgba8 = tile::detile_2d(&raw, width, height, pitch, 4).map_err(|e| e.to_string())?;
        Ok(Frame {
            width,
            height,
            rgba8,
        })
    }
}

/// Host format of a guest texture: `Rgba16Float` for half floats,
/// `Rgba32Float` for depth, 16-bit unorm and 32-bit floats (16-bit unorm and
/// 24-bit depth need more than a half's 11 bits), `Rgba8Unorm` for the rest.
/// Without filterable 32-bit floats the wide formats fall back to
/// `Rgba16Float`. That keeps the [-1, 1] range of 16-bit unorm and 24-bit
/// depth but not their precision. 32-bit floats also lose range: values with
/// a magnitude of 65520 or more become infinity and values below about 2^-25
/// flush to zero.
fn host_texture_format(format: xenon_texture::TextureFormat, float32: bool) -> wgpu::TextureFormat {
    use xenon_texture::TextureFormat as F;
    match format.data_format() {
        F::K16Float | F::K16_16Float | F::K16_16_16_16Float => wgpu::TextureFormat::Rgba16Float,
        F::K24_8
        | F::K24_8Float
        | F::K16
        | F::K16_16
        | F::K16_16_16_16
        | F::K32Float
        | F::K32_32Float
        | F::K32_32_32Float
        | F::K32_32_32_32Float => {
            if float32 {
                wgpu::TextureFormat::Rgba32Float
            } else {
                wgpu::TextureFormat::Rgba16Float
            }
        }
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_memory_writes_inside_only() {
        let mut m = SliceMemory {
            base: 0x10,
            data: vec![0; 4],
        };
        assert!(m.write_bytes(0x12, &[1, 2]));
        assert!(!m.write_bytes(0x13, &[1, 2]));
        assert!(!m.write_bytes(0x0F, &[1]));
        assert_eq!(m.data, vec![0, 0, 1, 2]);
    }

    #[test]
    fn host_texture_formats() {
        use wgpu::TextureFormat as H;
        use xenon_texture::TextureFormat as F;
        assert_eq!(host_texture_format(F::K8_8_8_8, true), H::Rgba8Unorm);
        assert_eq!(host_texture_format(F::Dxt1, true), H::Rgba8Unorm);
        assert_eq!(host_texture_format(F::K16Expand, true), H::Rgba16Float);
        assert_eq!(host_texture_format(F::K24_8Float, true), H::Rgba32Float);
        assert_eq!(host_texture_format(F::K24_8Float, false), H::Rgba16Float);
        assert_eq!(host_texture_format(F::K16_16, true), H::Rgba32Float);
    }

    #[test]
    fn vertex_patch_adds_epilogue_and_outputs() {
        let src =
            "struct VsOut {\n    @builtin(position) pos: vec4<f32>,\n}\n    out.pos = o_pos;\n";
        let p = patch_vertex_wgsl(src, 0b100).unwrap();
        assert!(p.contains("@location(2) o2: vec4<f32>,"));
        assert!(p.contains("out.pos = xe_epilogue(o_pos);\n    out.o2 = o_interp[2];"));
        assert!(p.contains("fn xe_epilogue"));
        assert!(patch_vertex_wgsl("nothing", 0).is_err());
    }

    #[test]
    fn rect_list_wrapper_resets_privates_and_makes_the_fourth_corner() {
        let src = "var<private> r: array<vec4<f32>, 64>;\nvar<private> a0: i32;\n\
            @vertex\nfn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {\n";
        let w = rect_list_wgsl(src, 0b101).unwrap();
        assert!(w.contains(
            "fn xe_vs_body(vertex_index: u32) -> VsOut {\n    \
             r = array<vec4<f32>, 64>();\n    a0 = i32();\n"
        ));
        assert!(w.contains("o.o0 = (b.o0 - a.o0) + c.o0;\n    o.o2 = (b.o2 - a.o2) + c.o2;"));
        assert!(!w.contains("o.o1 ="));
        assert!(rect_list_wgsl("var<private> x: i32 = 1;\n", 0).is_err());
        let ps = output_pixel_wgsl("    out.c0 = o_col[0];\n", 0, 0, 31.875, 1.0).unwrap();
        assert!(ps.contains("vec3<f32>(31.875)), clamp(o_col[0].w, 0.0, 1.0)"));
        assert!(output_pixel_wgsl("", 0, 0, 1.0, 1.0).is_err());
        let ps = output_pixel_wgsl("    out.c2 = o_col[2];\n", 2, 0, f32::INFINITY, 1.0).unwrap();
        assert!(ps.contains("out.c2 = vec4<f32>(o_col[2].xyz, clamp(o_col[2].w, 0.0, 1.0));"));
        let inf = f32::INFINITY;
        let ps = output_pixel_wgsl("    out.c1 = o_col[1];\n", 1, -2, inf, inf).unwrap();
        assert!(ps.contains("out.c1 = vec4<f32>((o_col[1] * 0.25).xyz, (o_col[1] * 0.25).w);"));
    }
}
