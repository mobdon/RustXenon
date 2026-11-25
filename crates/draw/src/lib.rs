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
// draw_util.cc; pixel parameter generation setup follows
// Shader::GetInterpolatorInputMask in shader_translator.cc and the pixel
// shader modification in vulkan/vulkan_pipeline_cache.cc; EDRAM ownership
// transfers between colour and depth (and the per-bit stencil passes without
// stencil export) follow src/xenia/gpu/render_target_cache.cc and the
// transfer shaders of d3d12/d3d12_render_target_cache.cc (BSD 3-Clause, see
// CREDITS.md).

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
//! [`RenderTargetKey`]: EDRAM base tile, surface pitch and format. A target
//! is `pitch` pixels wide and [`RENDER_TARGET_HEIGHT`] pixels high (the
//! EDRAM holds 2048 rows of 1280 32-bit pixels).
//!
//! Each target records when it was last written (draw or clear; a transfer
//! gives the target its source's stamp, so neither is transferred back).
//! When a draw, clear or resolve uses a target and a `k_8_8_8_8` colour
//! and a depth target at the same base and pitch were written later by
//! the other one, that one's pixels are first transferred into it, as
//! Xenia's render target cache transfers EDRAM ownership: they are drawn
//! as the shared 32-bit EDRAM word (colour `r | g << 8 | b << 16 | a <<
//! 24`; depth `depth24 << 8 | stencil`, converted like the depth clear and
//! the depth resolve) by the fullscreen passes of `transfer.wgsl`. Colour
//! to depth writes `frag_depth`, then, without shader stencil export, sets
//! the stencil one bit per pass as Xenia does. Other aliasing is not
//! modelled: each key keeps its own pixels. A newer target at the same
//! base and pitch that cannot be transferred (depth to depth, or depth and
//! a colour format other than `k_8_8_8_8`), or a newer target at another
//! base or pitch whose tiles overlap (estimated from the rows written, one
//! sample per pixel), is only reported in [`Report::warnings`]; colour
//! targets of different formats at one base are not reported.
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
//! - `k_32_FLOAT` as `R32Float` and `k_32_32_FLOAT` as `Rg32Float`, drawn
//!   without blending (a warning) and resolvable to `k_32_FLOAT` /
//!   `k_32_32_FLOAT` with their float bits kept, unsaturated.
//! - `k_16_16_FLOAT` as `Rg16Float`.
//! - `k_16_16` and `k_16_16_16_16` (16-bit fixed point in [-32, 32]) as
//!   `Rg16Snorm` / `Rgba16Snorm` holding the value / 32, Xenia's full-range
//!   host path: the pixel shader's colour is divided by 32 (so blending
//!   works on the value / 32, as in Xenia) and a resolve unpacks the 16
//!   bits as `x * 32 / 32767`, floored at -1, like Xenia's resolve shader.
//!   They need a host that renders into 16-bit snorm
//!   ([`Drawer::snorm16`]) and are unsupported without one; where the
//!   host cannot blend snorm16 they are drawn without blending (a warning).
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
//! stencil (the 20e4 float is stored at f32 precision).
//!
//! # Resolves
//!
//! A colour copy reads the rectangle back, takes each texel as the EDRAM
//! holds it (a 7e3 texel is rounded to 7e3 and its alpha to 2 bits, like
//! Xenia's dump of a host render target into EDRAM), applies the exponent
//! bias `2^copy_dest_exp_bias` and the red/blue swap, packs it like Xenia's
//! resolve shader (unorm: saturate, `x * max + 0.5` truncated) into
//! `k_8_8_8_8`, `k_2_10_10_10`, `k_16_16`, `k_16_16_16_16` (unorm),
//! `k_16_16_FLOAT`, `k_16_16_16_16_FLOAT`, `k_32_FLOAT` or `k_32_32_FLOAT`,
//! and tiles it into guest memory with the destination endian. A copy that
//! cannot be done is reported and the resolve's clears still run. Copy
//! commands are described at `Drawer::resolve` (`Raw` and `Convert` alike,
//! `Null` clears only, `ConstantOne` unsupported).
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
//! zero. With SQ_PROGRAM_CNTL.param_gen the pixel shader's register
//! SQ_CONTEXT_MISC.param_gen_pos holds the pixel position and the
//! facing / point / line flags instead ([`xenon_shader::ParamGen`]; points
//! get no sprite coordinates). Quad lists and triangle fans are expanded
//! to triangle lists by index. Rectangle lists draw six host vertices per rectangle through a
//! wrapper around the translated vertex shader that makes the fourth corner
//! from the three guest vertices after the vertex shader, as Xenia's
//! geometry shader does (`rect_list_wgsl`); they are never face culled.
//! Point lists draw one-pixel host points: `PA_SU_POINT_SIZE` is not
//! applied.
//!
//! Draws are recorded into one command buffer, in one render pass while
//! their render targets stay the same, with their buffers as slices of one
//! upload arena; the batch is submitted before each resolve, before each
//! EDRAM ownership transfer and at the end of [`Drawer::replay`].
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
use xenon_shader::{ParamGen, ParamGenPrimitive, ShaderInfo, Translated};
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

/// Guest memory that remembers the ranges written through it, so the
/// textures read from them are looked up again.
struct Tracked<'a> {
    mem: &'a mut dyn GuestMemory,
    /// Physical `[start, end)` ranges written since the last drain.
    written: Vec<(u64, u64)>,
}

impl GuestRead for Tracked<'_> {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        self.mem.read_u32(addr)
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        self.mem.read_bytes(addr, len)
    }
}

impl GuestMemory for Tracked<'_> {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        let start = u64::from(addr & PHYSICAL_MASK);
        self.written.push((start, start + bytes.len() as u64));
        self.mem.write_bytes(addr, bytes)
    }
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
    /// EDRAM ownership transfers between colour and depth targets (also
    /// counted in the phase that needed the target).
    pub transfer: Duration,
    /// Counts: pipelines and shader modules created, textures uploaded.
    pub pipelines_created: u32,
    pub textures_uploaded: u32,
    /// Command buffer submissions for draws, clears, EDRAM transfers and
    /// 32bpp resolves.
    pub submits: u32,
    pub texture_bytes: u64,
    pub guest_bytes: u64,
    /// EDRAM transfers and their render passes.
    pub transfers: u32,
    pub transfer_passes: u32,
    /// Bytes read back from the GPU for resolves.
    pub readback_bytes: u64,
    /// Guest texture bytes read and hashed to look up the texture cache.
    pub hashed_bytes: u64,
}

impl std::ops::AddAssign<&Timings> for Timings {
    /// Sums the phases and counts of several replays.
    fn add_assign(&mut self, t: &Timings) {
        self.shaders += t.shaders;
        self.guest_data += t.guest_data;
        self.textures += t.textures;
        self.submit += t.submit;
        self.gpu_wait += t.gpu_wait;
        self.readback += t.readback;
        self.resolve_cpu += t.resolve_cpu;
        self.clears += t.clears;
        self.swap += t.swap;
        self.transfer += t.transfer;
        self.pipelines_created += t.pipelines_created;
        self.textures_uploaded += t.textures_uploaded;
        self.submits += t.submits;
        self.texture_bytes += t.texture_bytes;
        self.guest_bytes += t.guest_bytes;
        self.transfers += t.transfers;
        self.transfer_passes += t.transfer_passes;
        self.readback_bytes += t.readback_bytes;
        self.hashed_bytes += t.hashed_bytes;
    }
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
    /// `k_16_16`: 16-bit fixed point in [-32, 32], hosted as `Rg16Snorm`
    /// holding the value / 32 (Xenia's full-range snorm16 path).
    Rg16Fixed,
    /// `k_16_16_16_16`, as `Rg16Fixed` with four channels (`Rgba16Snorm`).
    Rgba16Fixed,
    /// `k_16_16_FLOAT`, hosted as `Rg16Float`.
    Rg16Float,
    /// `k_32_32_FLOAT`, hosted as `Rg32Float`.
    Rg32Float,
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
            TargetFormat::Rg16Fixed => wgpu::TextureFormat::Rg16Snorm,
            TargetFormat::Rgba16Fixed => wgpu::TextureFormat::Rgba16Snorm,
            TargetFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
            TargetFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        }
    }

    /// The host format for `xenos::ColorRenderTargetFormat` `f`.
    pub fn color(f: u32) -> Option<TargetFormat> {
        Some(match f {
            0 | 1 => TargetFormat::Rgba8,
            2 | 10 => TargetFormat::Rgb10a2,
            3 | 12 => TargetFormat::Float7e3,
            14 => TargetFormat::Float32,
            4 => TargetFormat::Rg16Fixed,
            5 => TargetFormat::Rgba16Fixed,
            6 => TargetFormat::Rg16Float,
            7 => TargetFormat::Rgba16Float,
            15 => TargetFormat::Rg32Float,
            _ => return None,
        })
    }

    fn is_depth(self) -> bool {
        matches!(self, TargetFormat::D24S8 | TargetFormat::D24FS8)
    }

    /// Hosted as 16-bit snorm, which needs an optional wgpu feature.
    fn is_snorm16(self) -> bool {
        matches!(self, TargetFormat::Rg16Fixed | TargetFormat::Rgba16Fixed)
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
    /// Pixel shader colour i scaled by 2^bias, clamped to [0, rgb] and
    /// [0, alpha], then multiplied by scale (f32 bits; an infinite bound
    /// does not clamp) where entry i is set.
    Output([Option<(u32, u32, i8, u32)>; 4]),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    vs: u64,
    vs_extra: u16,
    vs_variant: Variant,
    ps: u64,
    ps_variant: Variant,
    ps_param_gen: Option<ParamGen>,
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

/// The last write to a render target.
#[derive(Clone, Copy, Default)]
struct Written {
    stamp: u64,
    /// Rows written so far, at least (scissor or rectangle bottom).
    rows: u32,
}

/// EDRAM tiles `[start, end)` that `k` covers down to row `rows`.
// shortcut: one sample per pixel and no wrap past tile 2048; count MSAA
// samples and wrap if a capture's overlap report needs to be exact.
fn edram_range(k: RenderTargetKey, rows: u32) -> (u32, u32) {
    let tile_width = if k.format == TargetFormat::Rgba16Float {
        40
    } else {
        80
    };
    let len = k.pitch.div_ceil(tile_width) * rows.div_ceil(16);
    (k.base_tiles, k.base_tiles + len)
}

/// Whether the EDRAM word of a target of format `a` can be transferred to
/// one of format `b`: k_8_8_8_8 colour to or from depth.
// shortcut: depth <-> depth (D24S8 / D24FS8) and depth <-> colour formats
// other than k_8_8_8_8 (k_32_FLOAT, 2_10_10_10, ...) are not transferred,
// only reported; add their word conversions to transfer.wgsl when a capture
// aliases them.
fn transferable(a: TargetFormat, b: TargetFormat) -> bool {
    (a == TargetFormat::Rgba8 && b.is_depth()) || (a.is_depth() && b == TargetFormat::Rgba8)
}

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
    translated: HashMap<(u64, bool, Option<ParamGen>), Result<Translated, String>>,
    /// Keyed by program hash, stage, extra vertex outputs, variant (the
    /// rectangle list wrapper or the pixel output bias and clamp) and pixel parameter
    /// generation.
    modules: HashMap<(u64, bool, u16, Variant, Option<ParamGen>), wgpu::ShaderModule>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    texture_layouts: HashMap<u32, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    group0: wgpu::BindGroupLayout,
    clear_layout: wgpu::BindGroupLayout,
    clear_module: wgpu::ShaderModule,
    clear_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    targets: HashMap<RenderTargetKey, wgpu::Texture>,
    /// When each render target was last written (draw or clear; a transfer
    /// copies its source's stamp) and how many rows down, for EDRAM
    /// ownership transfers.
    written: HashMap<RenderTargetKey, Written>,
    clock: u64,
    transfer_module: Option<wgpu::ShaderModule>,
    /// Keyed by source and destination host format and pass (stencil bit
    /// 0..8, or 8 for the depth or colour pass).
    transfer_pipelines:
        HashMap<(wgpu::TextureFormat, wgpu::TextureFormat, u32), wgpu::RenderPipeline>,
    /// Destination endian of the last 32bpp resolve to each guest address,
    /// so a swap reads its front buffer in the byte order it was written.
    resolved_endian: HashMap<u32, u32>,
    /// Decoded guest textures keyed by fetch constant and a hash of the
    /// guest bytes they read, so an unchanged texture is not decoded again.
    texture_cache: HashMap<([u32; 6], u64), wgpu::TextureView>,
    texture_cache_bytes: u64,
    /// Textures already looked up this replay by fetch constant, with the
    /// physical `[start, end)` guest range they were read from. Guest
    /// memory only changes at a resolve or MEM_WRITE during a replay, which
    /// drop the entries whose range they write, so later draws skip reading
    /// and hashing the guest bytes of the others.
    texture_memo: HashMap<[u32; 6], (wgpu::TextureView, u64, u64)>,
    samplers: HashMap<([u8; 3], u8, u8), wgpu::Sampler>,
    /// Draws recorded but not yet submitted (see [`Batch`]).
    batch: Option<Batch>,
    /// Draw buffers (guest window, constants, epilogue, indices) of the
    /// batch, as slices of `arena`, written to it once at submit.
    staged: Vec<u8>,
    arena: Option<wgpu::Buffer>,
    arena_align: u64,
    max_buffer: u64,
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
    /// `Rg16Snorm` and `Rgba16Snorm` can be render targets (k_16_16 and
    /// k_16_16_16_16 are refused without them).
    snorm16: bool,
    /// The snorm16 targets blend (else their blending is dropped).
    snorm16_blend: bool,
    timings: Timings,
    uncaptured: Arc<Mutex<Vec<String>>>,
    max_dim: u32,
    max_storage: u64,
}

/// Draws recorded into one command encoder, in one render pass while the
/// render targets stay the same. Submitted by [`Drawer::flush`] before
/// anything that reads or writes the targets outside the batch (resolves,
/// clears and EDRAM transfers) and at the end of a replay.
struct Batch {
    /// Dropped (ended) before `enc` is finished.
    pass: Option<(PassKey, wgpu::RenderPass<'static>)>,
    enc: wgpu::CommandEncoder,
}

/// Colour and depth targets of a render pass.
type PassKey = ([Option<RenderTargetKey>; 4], Option<RenderTargetKey>);

/// Smallest draw upload arena; it grows to fit a larger draw.
const ARENA_MIN: u64 = 16 << 20;

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
/// (RB_COLOR_INFO's colour exponent bias, all four components, before the
/// EDRAM format clamp), clamps it to [0, rgb] / [0, alpha], and multiplies
/// it by `scale` (the host storage scale of the target format).
fn output_pixel_wgsl(
    ps: &str,
    rt: usize,
    bias: i8,
    rgb: f32,
    alpha: f32,
    scale: f32,
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
    let scaled = if scale == 1.0 {
        String::new()
    } else {
        format!(" * {scale:?}")
    };
    Ok(ps.replacen(
        &store,
        &format!("    out.c{rt} = vec4<f32>({rgb}, {a}){scaled};\n"),
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
        // k_16_16 and k_16_16_16_16 render into snorm16, which WebGPU
        // only renders into with these features and the adapter's say-so.
        let snorm16_features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let snorm16_all = |ok: &dyn Fn(wgpu::TextureFormatFeatures) -> bool| {
            adapter.features().contains(snorm16_features)
                && [
                    wgpu::TextureFormat::Rg16Snorm,
                    wgpu::TextureFormat::Rgba16Snorm,
                ]
                .iter()
                .all(|&f| ok(adapter.get_texture_format_features(f)))
        };
        let snorm16 = snorm16_all(&|f| {
            f.allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        });
        let snorm16_blend = snorm16
            && snorm16_all(&|f| f.flags.contains(wgpu::TextureFormatFeatureFlags::BLENDABLE));
        let mut required_features = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
        required_features.set(wgpu::Features::DEPTH32FLOAT_STENCIL8, float_stencil);
        required_features.set(snorm16_features, snorm16);
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
            written: HashMap::new(),
            clock: 0,
            transfer_module: None,
            transfer_pipelines: HashMap::new(),
            resolved_endian: HashMap::new(),
            texture_cache: HashMap::new(),
            texture_cache_bytes: 0,
            texture_memo: HashMap::new(),
            samplers: HashMap::new(),
            batch: None,
            staged: Vec::new(),
            arena: None,
            arena_align: u64::from(
                limits
                    .min_uniform_buffer_offset_alignment
                    .max(limits.min_storage_buffer_offset_alignment),
            ),
            max_buffer: limits.max_buffer_size,
            zero_texture,
            zero_sampler,
            float32_filterable: required_features.contains(wgpu::Features::FLOAT32_FILTERABLE),
            resolve_pipelines: [None, None, None],
            float_stencil,
            snorm16,
            snorm16_blend,
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
        self.texture_memo.clear();
        let mut mem = Tracked {
            mem,
            written: Vec::new(),
        };
        let done = self.replay_commands(commands, &mut mem, &mut report);
        // On an error the batch is dropped unsubmitted.
        let done = done.and_then(|()| {
            self.flush().map_err(|e| match e {
                Skip::Fatal(e) => e,
                Skip::Unsupported(why) => Error::Gpu(why),
            })
        });
        self.batch = None;
        self.staged.clear();
        done?;
        // Counts made where only the report is mutable.
        let t = &report.timings;
        let (uploaded, tex_bytes, guest_bytes, hashed_bytes) = (
            t.textures_uploaded,
            t.texture_bytes,
            t.guest_bytes,
            t.hashed_bytes,
        );
        report.timings = Timings {
            textures_uploaded: uploaded,
            texture_bytes: tex_bytes,
            guest_bytes,
            hashed_bytes,
            ..std::mem::take(&mut self.timings)
        };
        Ok(report)
    }

    fn replay_commands(
        &mut self,
        commands: &[Command],
        tracked: &mut Tracked,
        report: &mut Report,
    ) -> Result<(), Error> {
        for c in commands {
            let mem: &mut dyn GuestMemory = tracked;
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
                Command::Draw(d) => self.draw(d, mem, report).map(|drawn| {
                    report.draws += u32::from(drawn);
                }),
                Command::Resolve(r) => self.resolve(r, mem, report).map(|done| {
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
            for (lo, hi) in tracked.written.drain(..) {
                self.texture_memo
                    .retain(|_, &mut (_, a, b)| b <= lo || hi <= a);
            }
            match done {
                Ok(()) => {}
                Err(Skip::Unsupported(why)) => report.unsupported(why),
                Err(Skip::Fatal(e)) => return Err(e),
            }
            self.check_uncaptured()?;
        }
        Ok(())
    }

    /// Submits the batched draws, with their staged buffers.
    fn flush(&mut self) -> Result<(), Skip> {
        if let (Some(arena), false) = (&self.arena, self.staged.is_empty()) {
            self.queue.write_buffer(arena, 0, &self.staged);
            self.staged.clear();
        }
        let Some(mut batch) = self.batch.take() else {
            return Ok(());
        };
        let t = Instant::now();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        drop(batch.pass.take());
        self.queue.submit([batch.enc.finish()]);
        self.timings.submits += 1;
        self.timings.submit += t.elapsed();
        match block_on(scope.pop()) {
            Some(e) => Err(Skip::Fatal(Error::Gpu(e.to_string()))),
            None => Ok(()),
        }
    }

    /// Copies `parts` into the batch's staging area, each at an offset
    /// aligned for uniform and storage bindings, and returns their
    /// (offset, size) in the arena. Submits the batch first when the arena
    /// is full, and grows the arena when one draw does not fit.
    fn stage(&mut self, parts: &[&[u8]]) -> Result<Vec<(u64, u64)>, Skip> {
        let size = |p: &[u8]| (p.len() as u64).max(4).next_multiple_of(4);
        let align = self.arena_align;
        let need: u64 = parts.iter().map(|p| size(p).next_multiple_of(align)).sum();
        let have = self.arena.as_ref().map_or(0, |b| b.size());
        if self.staged.len() as u64 + need > have {
            self.flush()?;
            if need > have {
                if need > self.max_buffer {
                    return unsupported(format!("{need} bytes of draw data"));
                }
                self.arena = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("draw arena"),
                    size: need.max(ARENA_MIN).next_power_of_two().min(self.max_buffer),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::INDEX
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }
        let mut out = Vec::with_capacity(parts.len());
        for p in parts {
            let off = self.staged.len() as u64;
            self.staged.extend_from_slice(p);
            self.staged.resize((off + size(p)) as usize, 0);
            out.push((off, size(p)));
            let end = self.staged.len() as u64;
            self.staged.resize(end.next_multiple_of(align) as usize, 0);
        }
        Ok(out)
    }

    /// The batch's render pass, begun on `key`'s targets unless the current
    /// pass already draws to them.
    fn pass(
        &mut self,
        key: PassKey,
        color: &[Option<wgpu::Texture>; 4],
        depth: Option<&wgpu::Texture>,
    ) -> &mut wgpu::RenderPass<'static> {
        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            pass: None,
            enc: device.create_command_encoder(&Default::default()),
        });
        if batch.pass.as_ref().map(|p| &p.0) != Some(&key) {
            batch.pass = None;
            fn keep<V>() -> wgpu::Operations<V> {
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }
            }
            let color_views = color
                .each_ref()
                .map(|t| t.as_ref().map(|t| t.create_view(&Default::default())));
            let depth_view = depth.map(|t| (t.format(), t.create_view(&Default::default())));
            let pass = batch
                .enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("draw"),
                    color_attachments: &color_views.each_ref().map(|v| {
                        v.as_ref().map(|view| wgpu::RenderPassColorAttachment {
                            view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: keep(),
                        })
                    }),
                    depth_stencil_attachment: depth_view.as_ref().map(|(format, view)| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: Some(keep()),
                            stencil_ops: format.has_stencil_aspect().then(keep),
                        }
                    }),
                    ..Default::default()
                })
                .forget_lifetime();
            batch.pass = Some((key, pass));
        }
        &mut batch.pass.as_mut().expect("pass begun").1
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
    /// `Float7e3` and `Rgba16Float` (four halves, R first); the other
    /// formats as their wgpu texels (`Float32` 4, `Rg16Fixed` and
    /// `Rg16Float` 4, `Rgba16Fixed` and `Rg32Float` 8, R first).
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

    /// The texture of render target `key`, made current first: when another
    /// target at the same EDRAM base and pitch was written after it, that
    /// target's pixels are transferred into it (the EDRAM holds one set of
    /// bytes for both). `rows` is how far down this use reaches, for the
    /// overlap report.
    fn target(
        &mut self,
        key: RenderTargetKey,
        rows: u32,
        report: &mut Report,
    ) -> Result<wgpu::Texture, Skip> {
        if key.pitch == 0 || key.pitch > self.max_dim {
            return unsupported(format!("surface pitch {}", key.pitch));
        }
        if key.format.is_snorm16() && !self.snorm16 {
            return unsupported(format!(
                "{:?} render target: the host cannot render into 16-bit snorm",
                key.format
            ));
        }
        let host_format = self.host_format(key.format);
        let (device, height) = (&self.device, RENDER_TARGET_HEIGHT.min(self.max_dim));
        let tex = self
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
            .clone();
        let own = self.written.get(&key).copied().unwrap_or_default();
        let range = edram_range(key, rows.max(own.rows));
        let mut newest: Option<(RenderTargetKey, Written)> = None;
        for (&k, &w) in &self.written {
            if w.stamp <= own.stamp || k == key {
                continue;
            }
            if k.base_tiles == key.base_tiles && k.pitch == key.pitch {
                // shortcut: colour targets of different formats at one base
                // keep their own pixels, silently; transfer them too if a
                // capture reinterprets colour in place.
                let with_depth = k.format.is_depth() || key.format.is_depth();
                if with_depth && newest.is_none_or(|(_, n)| w.stamp > n.stamp) {
                    newest = Some((k, w));
                }
            } else {
                let other = edram_range(k, w.rows);
                // shortcut: partial overlaps (another base or pitch) are only
                // reported; transfer the overlapping tiles if a capture
                // needs them.
                if range.0 < other.1 && other.0 < range.1 {
                    report.warn(
                        "render target overlaps a newer one at another EDRAM base or pitch; not transferred",
                    );
                }
            }
        }
        if let Some((src, w)) = newest {
            if transferable(src.format, key.format) {
                let t = Instant::now();
                self.transfer(src, key, &tex, w.rows)?;
                self.timings.transfer += t.elapsed();
                // Both now hold the same words: the source's stamp, so
                // using the source again does not transfer them back.
                let own = self.written.entry(key).or_default();
                own.stamp = w.stamp;
                own.rows = own.rows.max(w.rows);
            } else {
                report.warn(format!(
                    "{:?} render target aliases a newer {:?} one at its EDRAM base; not transferred",
                    key.format, src.format
                ));
            }
        }
        Ok(tex)
    }

    /// Records a write to `key` down to row `rows`.
    fn mark_written(&mut self, key: RenderTargetKey, rows: u32) {
        self.clock += 1;
        let w = self.written.entry(key).or_default();
        w.stamp = self.clock;
        w.rows = w.rows.max(rows);
    }

    /// The transfer pipeline from host format `src` to `dst`; `bit` 0..8 is
    /// a stencil bit pass, 8 the depth (or colour) pass.
    fn transfer_pipeline(
        &mut self,
        src: wgpu::TextureFormat,
        dst: wgpu::TextureFormat,
        bit: u32,
    ) -> Result<wgpu::RenderPipeline, Skip> {
        if let Some(p) = self.transfer_pipelines.get(&(src, dst, bit)) {
            return Ok(p.clone());
        }
        let device = &self.device;
        let module = &self
            .transfer_module
            .get_or_insert_with(|| {
                let wgsl = [include_str!("resolve.wgsl"), include_str!("transfer.wgsl")].concat();
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("transfer"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                })
            })
            .clone();
        let to_depth = dst.is_depth_stencil_format();
        let entry = match (to_depth, bit) {
            (true, 8) => "color_to_depth",
            (true, _) => "color_to_stencil",
            (false, _) if src.has_stencil_aspect() => "depth_stencil_to_color",
            (false, _) => "depth_to_color",
        };
        let float20e4 = dst != wgpu::TextureFormat::Depth24PlusStencil8;
        let constants = [
            ("transfer_20e4", if float20e4 { 1.0 } else { 0.0 }),
            ("transfer_bit", f64::from(bit)),
        ];
        // The depth pass replaces the whole stencil with the reference 0;
        // bit pass i replaces only bit i, with the reference 0xFF.
        let face = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Replace,
        };
        let stencil = wgpu::StencilState {
            front: face,
            back: face,
            read_mask: 0xFF,
            write_mask: if bit == 8 { 0xFF } else { 1 << bit },
        };
        let color_target = [Some(wgpu::ColorTargetState {
            format: dst,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let p = self.validated("transfer pipeline", |dev| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transfer"),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("transfer_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: to_depth.then(|| wgpu::DepthStencilState {
                    format: dst,
                    depth_write_enabled: Some(bit == 8),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: if dst.has_stencil_aspect() {
                        stencil.clone()
                    } else {
                        Default::default()
                    },
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: if to_depth { &constants } else { &[] },
                        ..Default::default()
                    },
                    targets: if to_depth { &[] } else { &color_target },
                }),
                multiview_mask: None,
                cache: None,
            })
        })?;
        self.transfer_pipelines.insert((src, dst, bit), p.clone());
        Ok(p)
    }

    /// EDRAM ownership transfer: draws the first `rows` rows of target
    /// `src` into `dst` (same base and pitch, one k_8_8_8_8 and one depth)
    /// as their shared 32-bit EDRAM word: colour `r | g << 8 | b << 16 |
    /// a << 24`, depth `depth24 << 8 | stencil` (24-bit unorm for kD24S8,
    /// 20e4 for kD24FS8, converted as the depth clear and resolve do).
    fn transfer(
        &mut self,
        src: RenderTargetKey,
        dst: RenderTargetKey,
        dst_tex: &wgpu::Texture,
        rows: u32,
    ) -> Result<(), Skip> {
        let src_tex = self.targets[&src].clone();
        let rows = rows.min(dst_tex.height());
        if rows == 0 {
            return Ok(());
        }
        // The source's batched draws (and any reads of `dst` in the batch)
        // come before the transfer, which is submitted on its own.
        self.flush()?;
        let (sf, df) = (self.host_format(src.format), self.host_format(dst.format));
        let to_depth = df.is_depth_stencil_format();
        let bits = if df.has_stencil_aspect() { 0..8 } else { 0..0 };
        let mut pipelines = vec![self.transfer_pipeline(sf, df, 8)?];
        for bit in bits {
            pipelines.push(self.transfer_pipeline(sf, df, bit)?);
        }
        // resolve.wgsl's depth_word reads only `src_format` of its params.
        let mut params = [0u32; 12];
        params[5] = if src.format == TargetFormat::D24FS8 {
            6
        } else {
            5
        };
        let uniform = self.buffer(
            "transfer params",
            &to_bytes(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let aspect = |aspect| {
            src_tex.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                ..Default::default()
            })
        };
        let views = if to_depth {
            vec![(0, aspect(wgpu::TextureAspect::All))]
        } else if sf.has_stencil_aspect() {
            vec![
                (3, aspect(wgpu::TextureAspect::DepthOnly)),
                (4, aspect(wgpu::TextureAspect::StencilOnly)),
            ]
        } else {
            vec![(3, aspect(wgpu::TextureAspect::DepthOnly))]
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let groups: Vec<_> = pipelines
            .iter()
            .map(|p| {
                let mut entries: Vec<_> = views
                    .iter()
                    .map(|(binding, view)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: wgpu::BindingResource::TextureView(view),
                    })
                    .collect();
                if !to_depth {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    });
                }
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("transfer"),
                    layout: &p.get_bind_group_layout(0),
                    entries: &entries,
                })
            })
            .collect();
        let view = dst_tex.create_view(&Default::default());
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })];
        let keep = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transfer"),
                color_attachments: if to_depth { &[] } else { &color_attachment },
                depth_stencil_attachment: to_depth.then(|| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(keep),
                        stencil_ops: df.has_stencil_aspect().then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                ..Default::default()
            });
            pass.set_scissor_rect(0, 0, dst_tex.width(), rows);
            for (i, (p, g)) in pipelines.iter().zip(&groups).enumerate() {
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                pass.set_stencil_reference(if i == 0 { 0 } else { 0xFF });
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        self.timings.submits += 1;
        self.timings.transfers += 1;
        self.timings.transfer_passes += pipelines.len() as u32;
        match block_on(scope.pop()) {
            Some(e) => Err(Skip::Fatal(Error::Gpu(e.to_string()))),
            None => Ok(()),
        }
    }

    /// Whether `kD24FS8` keeps its stencil (the host has
    /// `Depth32FloatStencil8`); without it the stencil test on such a
    /// target is skipped with a warning and resolves write stencil 0.
    pub fn float_stencil(&self) -> bool {
        self.float_stencil
    }

    /// Whether the host renders into 16-bit snorm, which `k_16_16` and
    /// `k_16_16_16_16` targets need (they are unsupported without it).
    pub fn snorm16(&self) -> bool {
        self.snorm16
    }

    /// The host texture format of a render target.
    fn host_format(&self, format: TargetFormat) -> wgpu::TextureFormat {
        match format {
            TargetFormat::D24FS8 if self.float_stencil => wgpu::TextureFormat::Depth32FloatStencil8,
            f => f.wgpu(),
        }
    }

    fn translate(
        &mut self,
        p: &Program,
        stage: ShaderStage,
        param_gen: Option<ParamGen>,
    ) -> Result<Translated, Skip> {
        let vertex = stage == ShaderStage::Vertex;
        let t = self
            .translated
            .entry((p.hash, vertex, param_gen))
            .or_insert_with(|| {
                let s = if vertex {
                    xenon_shader::ShaderStage::Vertex
                } else {
                    xenon_shader::ShaderStage::Pixel
                };
                xenon_shader::translate_with(&p.words, s, param_gen).map_err(|e| e.to_string())
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
        param_gen: Option<ParamGen>,
        t: &Translated,
    ) -> Result<wgpu::ShaderModule, Skip> {
        let key = (hash, vertex, extra, variant, param_gen);
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
                        Some((rgb, a, bias, k)) => output_pixel_wgsl(
                            &ps,
                            rt,
                            *bias,
                            f32::from_bits(*rgb),
                            f32::from_bits(*a),
                            f32::from_bits(*k),
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
    ) -> Result<(Vec<u8>, u32), Skip> {
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
        Ok((bytes, base))
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
        if let Some((view, ..)) = self.texture_memo.get(&self.regs.fetch_constant(n)) {
            return Ok((view.clone(), sampler));
        }
        let view = self
            .guest_texture(n, &fetch, mem, report)
            .unwrap_or_else(|why| {
                report.warn(format!("{why}; drawn with zeros"));
                self.zero_texture.clone()
            });
        Ok((view, sampler))
    }

    /// Host texture of fetch constant slot `n`, from the cache or decoded.
    /// Formats whose values do not fit 8-bit unorm, and fetches with a signed
    /// component, upload as floats. The cache key holds the whole fetch
    /// constant (sign bits included), so signed and unsigned views of the
    /// same bytes are separate textures.
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
        report.timings.hashed_bytes += bytes.len() as u64;
        let key = (self.regs.fetch_constant(n), hash_bytes(&bytes));
        let range = (u64::from(addr), u64::from(addr) + bytes.len() as u64);
        if let Some(view) = self.texture_cache.get(&key) {
            self.texture_memo
                .insert(key.0, (view.clone(), range.0, range.1));
            return Ok(view.clone());
        }
        let signed = fetch.sign.contains(&1);
        let host = host_texture_format(fetch.format, signed, self.float32_filterable);
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
        self.texture_memo
            .insert(key.0, (view.clone(), range.0, range.1));
        Ok(view)
    }

    fn sampler(
        &mut self,
        fetch: &xenon_texture::TextureFetch,
        report: &mut Report,
    ) -> wgpu::Sampler {
        let mut addr_mode = |c: u8| {
            let (m, exact) = sampler_address(c);
            if !exact {
                report.warn(format!("texture clamp mode {c} drawn as clamp to edge"));
            }
            m
        };
        let modes = fetch.clamp.map(&mut addr_mode);
        let device = &self.device;
        self.samplers
            .entry((fetch.clamp, fetch.mag_filter, fetch.min_filter))
            .or_insert_with(|| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("guest sampler"),
                    address_mode_u: modes[0],
                    address_mode_v: modes[1],
                    address_mode_w: modes[2],
                    mag_filter: filter(fetch.mag_filter),
                    min_filter: filter(fetch.min_filter),
                    ..Default::default()
                })
            })
            .clone()
    }

    /// Pixel parameter generation of the draw: SQ_PROGRAM_CNTL.param_gen
    /// (bit 18) into register SQ_CONTEXT_MISC.param_gen_pos (bits 8..16),
    /// as Xenia's GetInterpolatorInputMask and pipeline caches set it up.
    // shortcut: any register below 16 is written, where Xenia also requires
    // it below the program's register count (static bound or ps_num_reg);
    // differs only for a program that reads registers it never declares.
    fn param_gen(&self, topology: wgpu::PrimitiveTopology, polygonal: bool) -> Option<ParamGen> {
        use wgpu::PrimitiveTopology as T;
        if self.regs.get(regs::SQ_PROGRAM_CNTL) & (1 << 18) == 0 {
            return None;
        }
        let register = (self.regs.get(regs::SQ_CONTEXT_MISC) >> 8) & 0xFF;
        let primitive = match topology {
            _ if polygonal => ParamGenPrimitive::Polygon,
            T::LineList | T::LineStrip => ParamGenPrimitive::Line,
            T::PointList => ParamGenPrimitive::Point,
            _ => ParamGenPrimitive::Other,
        };
        (register < 16).then_some(ParamGen {
            register,
            primitive,
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
        let rect_list = expand == indices::Expand::Rects;
        // Rectangle lists are not polygonal for Xenia (IsPrimitivePolygonal):
        // no face culling.
        let polygonal = !rect_list
            && matches!(
                topology,
                wgpu::PrimitiveTopology::TriangleList | wgpu::PrimitiveTopology::TriangleStrip
            );
        let param_gen = self.param_gen(topology, polygonal);
        let mut t = Instant::now();
        let vs_t = self.translate(&vs, ShaderStage::Vertex, None)?;
        let ps_t = self.translate(&ps, ShaderStage::Pixel, param_gen)?;
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
            if blends[rt].is_some()
                && matches!(
                    color[rt].map(|k| k.format),
                    Some(TargetFormat::Float32 | TargetFormat::Rg32Float)
                )
            {
                report.warn("blending into a 32-bit float render target drawn without blending");
                blends[rt] = None;
            }
            // shortcut: no blending into snorm16 where the host cannot (e.g.
            // lavapipe); emulate it in the shader if a game relies on it.
            if blends[rt].is_some()
                && !self.snorm16_blend
                && color[rt].is_some_and(|k| k.format.is_snorm16())
            {
                report.warn("blending into a 16-bit snorm render target drawn without blending");
                blends[rt] = None;
            }
        }
        let outputs: [Option<(u32, u32, i8, u32)>; 4] = std::array::from_fn(|rt| {
            let k = color[rt]?;
            // RB_COLOR_INFO bits 20..25: signed colour exponent bias.
            let bias = ((self.regs.get(regs::RB_COLOR_INFO_RT[rt]) << 6) as i32 >> 26) as i8;
            let clamp = formats::output_clamp(k.format, blends[rt].is_some());
            if bias == 0 && clamp.is_none() {
                return None;
            }
            let (rgb, a, scale) = clamp.unwrap_or((f32::INFINITY, f32::INFINITY, 1.0));
            Some((rgb.to_bits(), a.to_bits(), bias, scale.to_bits()))
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

        // Every target of a draw has the surface pitch and the same height.
        let (w, h) = (surface & 0x3FFF, RENDER_TARGET_HEIGHT.min(self.max_dim));
        let (sx, sy, sw, sh) = state::scissor(&self.regs, w, h);
        let rows = sy + sh;
        let mut color_tex: [Option<wgpu::Texture>; 4] = Default::default();
        for rt in 0..4 {
            color_tex[rt] = color[rt]
                .map(|k| self.target(k, rows, report))
                .transpose()?;
        }
        let depth_tex = depth_key
            .map(|k| self.target(k, rows, report))
            .transpose()?;
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
        let vs_mod = self.module(vs.hash, true, extra, vs_variant, None, &vs_t)?;
        let ps_mod = self.module(ps.hash, false, 0, ps_variant, param_gen, &ps_t)?;
        let (g1_layout, layout) = self.texture_layout(tex_mask);

        let key = PipelineKey {
            vs: vs.hash,
            vs_extra: extra,
            vs_variant,
            ps: ps.hash,
            ps_variant,
            ps_param_gen: param_gen,
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
        let consts = self.consts(window_base);
        let epi = to_bytes(&state::epilogue(&self.regs, w, h));
        self.timings.guest_data += lap(&mut t);
        let mut textures = Vec::new();
        for n in (0..32).filter(|n| tex_mask & (1 << n) != 0) {
            textures.push((n, self.texture(n, mem, report)?));
        }
        self.timings.textures += lap(&mut t);
        // Rectangle lists read their guest indices in the vertex shader and
        // draw 6 host vertices per rectangle without an index buffer.
        let idx_bytes = to_bytes(&idx);
        let rect_indices: &[u8] = if rect_list { &idx_bytes } else { &[] };
        let slices = self.stage(&[&window, &consts, &epi, rect_indices, &idx_bytes])?;
        let arena = self.arena.clone().expect("staged");
        let binding = |i: usize| {
            let (offset, size) = slices[i];
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &arena,
                offset,
                size: wgpu::BufferSize::new(size),
            })
        };
        let g0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw group 0"),
            layout: &self.group0,
            entries: &[0, 1, 2, 3].map(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: binding(i),
            }),
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
        let pass = self.pass((color, depth_key), &color_tex, depth_tex.as_ref());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &g0, &[]);
        pass.set_bind_group(1, &g1, &[]);
        let (offset, size) = slices[4];
        pass.set_index_buffer(
            arena.slice(offset..offset + size),
            wgpu::IndexFormat::Uint32,
        );
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
        self.timings.submit += lap(&mut t);
        // Recorded, not yet submitted: everything else that touches the
        // targets flushes the batch first, so stamps follow submit order.
        for k in color.into_iter().flatten() {
            self.mark_written(k, rows);
        }
        if let Some(k) = depth_key.filter(|_| depth.write || stencil.is_some()) {
            self.mark_written(k, rows);
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
        report: &mut Report,
    ) -> Result<(), Skip> {
        let tex = self.target(key, rect.y1, report)?;
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
        self.timings.submits += 1;
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        self.mark_written(key, rect.y1);
        Ok(())
    }

    /// Performs a resolve: copy, then the colour and depth clears (also when
    /// the copy is unsupported). Ok(false) when the rectangle is empty.
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
        // The copy and clears see every draw before them.
        self.flush()?;
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
        let copied = match r.copy_command {
            CopyCommand::Raw | CopyCommand::Convert => self.resolve_copy(r, rect, mem, report),
            CopyCommand::Null => Ok(()),
            CopyCommand::ConstantOne => return unsupported("resolve copy command ConstantOne"),
        };
        // A copy this drawer cannot do still clears, as Xenia always would;
        // the copy is reported after the clears.
        if let Err(Skip::Fatal(_)) = copied {
            return copied.map(|()| true);
        }
        // A depth resolve never clears colour (Xenia draw_util.h
        // IsClearingColor).
        if r.color_clear_enable && !r.is_depth {
            let key = RenderTargetKey::color(&self.regs, r.src_select as usize)
                .map_err(|e| Skip::Unsupported(format!("resolve clear of {e}")))?;
            let value = formats::clear_value(r.color_clear, r.color_clear_lo);
            let t = Instant::now();
            self.clear(
                key,
                rect,
                formats::clear_color(key.format, value),
                0.0,
                0,
                report,
            )?;
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
            self.clear(key, rect, [0.0; 4], depth, r.depth_clear & 0xFF, report)?;
            self.timings.clears += t.elapsed();
        }
        copied.map(|()| true)
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
        report: &mut Report,
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
        let tex = self.target(key, rect.y1, report)?;
        let pitch = xenon_texture::tile::align_to_tile(r.dest_pitch);
        let rect = resolve::Rect {
            x1: rect.x1.min(pitch).min(tex.width()),
            ..rect
        };
        if rect.is_empty() {
            return Ok(());
        }
        // shortcut: 16_16_FLOAT colour copies take the CPU path (WGSL's
        // pack2x16float does not promise rounding); port f32_to_half to
        // resolve.wgsl if they show up in profiles.
        if dest_bytes == 4 && (r.is_depth || r.dest_format != formats::DEST_16_16_FLOAT) {
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
        self.timings.readback_bytes += px.len() as u64;
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
        // Waiting here only splits the rendering from the resolve in the
        // timings: the readback waits for it anyway.
        let mut t = Instant::now();
        self.wait_idle()?;
        self.timings.gpu_wait += lap(&mut t);
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
            TargetFormat::Rg16Fixed => 7,
            TargetFormat::Rgba16Fixed => 8,
            TargetFormat::Rg16Float => 9,
            TargetFormat::Rg32Float => 10,
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
                formats::DEST_16_16 => 3,
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
        self.timings.submits += 1;
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        let mut out = self.map_read(&readback, index)?;
        self.timings.readback += lap(&mut t);
        self.timings.readback_bytes += out.len() as u64;
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
///
/// `signed` (a fetch sign of 1 on any component) moves the 8-bit formats to
/// the wide float format too: the decoder writes signed values in -1..1,
/// which the host then filters directly, and the texture's unsigned
/// components keep their 8-bit values exactly (a half would not).
fn host_texture_format(
    format: xenon_texture::TextureFormat,
    signed: bool,
    float32: bool,
) -> wgpu::TextureFormat {
    use xenon_texture::TextureFormat as F;
    let wide = if float32 {
        wgpu::TextureFormat::Rgba32Float
    } else {
        wgpu::TextureFormat::Rgba16Float
    };
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
        | F::K32_32_32_32Float => wide,
        _ if signed => wide,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// transfer.wgsl's `word_depth` makes exactly the depth the depth clear
    /// makes: n / 0xFFFFFF rounded to the nearest f32 for kD24S8 (a host
    /// that keeps Depth24PlusStencil8 as 32-bit float stores it as is;
    /// lavapipe's 24-bit store would hide a mismatch), float20e4 for kD24FS8.
    #[test]
    fn transfer_word_depth_matches_the_clear() {
        let d = match Drawer::new() {
            Ok(d) => d,
            Err(Error::NoAdapter) if std::env::var("XENON_REQUIRE_GPU").as_deref() != Ok("1") => {
                return
            }
            Err(e) => panic!("{e}"),
        };
        // Odd n (the ones a reciprocal multiply gets wrong) across 24 bits.
        const N: u32 = 1 << 20;
        let n = |i: u32| (i << 4) | 15;
        let wgsl = [
            include_str!("resolve.wgsl"),
            include_str!("transfer.wgsl"),
            "@compute @workgroup_size(64)\n\
             fn word_depth_test(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             \x20   dest[id.x] = bitcast<u32>(word_depth((id.x << 4u) | 15u));\n}\n",
        ]
        .concat();
        let module = d.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        for float in [false, true] {
            let pipeline = d
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("word_depth_test"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("transfer_20e4", if float { 1.0 } else { 0.0 })],
                        ..Default::default()
                    },
                    cache: None,
                });
            let buffer = |usage| {
                d.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: u64::from(N) * 4,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let out = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let read = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let group = d.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                }],
            });
            let mut enc = d.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(N / 64, 1, 1);
            }
            enc.copy_buffer_to_buffer(&out, 0, &read, 0, None);
            let data = d.map_read(&read, d.queue.submit([enc.finish()])).unwrap();
            for i in 0..N {
                let got = u32::from_le_bytes(data[i as usize * 4..][..4].try_into().unwrap());
                let want = if float {
                    float20e4(n(i))
                } else {
                    (f64::from(n(i)) / f64::from(0xFF_FFFFu32)) as f32
                };
                assert_eq!(got, want.to_bits(), "{float} {:06X}", n(i));
            }
        }
    }

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
        assert_eq!(host_texture_format(F::K8_8_8_8, false, true), H::Rgba8Unorm);
        assert_eq!(host_texture_format(F::K8_8_8_8, true, true), H::Rgba32Float);
        assert_eq!(host_texture_format(F::Dxt1, true, false), H::Rgba16Float);
        assert_eq!(host_texture_format(F::Dxt1, false, true), H::Rgba8Unorm);
        assert_eq!(
            host_texture_format(F::K16Expand, false, true),
            H::Rgba16Float
        );
        assert_eq!(
            host_texture_format(F::K24_8Float, false, true),
            H::Rgba32Float
        );
        assert_eq!(
            host_texture_format(F::K24_8Float, false, false),
            H::Rgba16Float
        );
        assert_eq!(host_texture_format(F::K16_16, false, true), H::Rgba32Float);
        assert_eq!(host_texture_format(F::K16_16, true, true), H::Rgba32Float);
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
        let ps = output_pixel_wgsl("    out.c0 = o_col[0];\n", 0, 0, 31.875, 1.0, 1.0).unwrap();
        assert!(ps.contains("vec3<f32>(31.875)), clamp(o_col[0].w, 0.0, 1.0));"));
        assert!(output_pixel_wgsl("", 0, 0, 1.0, 1.0, 1.0).is_err());
        let ps =
            output_pixel_wgsl("    out.c2 = o_col[2];\n", 2, 0, f32::INFINITY, 1.0, 1.0).unwrap();
        assert!(ps.contains("out.c2 = vec4<f32>(o_col[2].xyz, clamp(o_col[2].w, 0.0, 1.0));"));
        let inf = f32::INFINITY;
        let ps = output_pixel_wgsl("    out.c1 = o_col[1];\n", 1, -2, inf, inf, 1.0).unwrap();
        assert!(ps.contains("out.c1 = vec4<f32>((o_col[1] * 0.25).xyz, (o_col[1] * 0.25).w);"));
        let ps = output_pixel_wgsl("    out.c3 = o_col[3];\n", 3, 1, inf, inf, 0.03125).unwrap();
        assert!(
            ps.contains("out.c3 = vec4<f32>((o_col[3] * 2.0).xyz, (o_col[3] * 2.0).w) * 0.03125;")
        );
    }
}
