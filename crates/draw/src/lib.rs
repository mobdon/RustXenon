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
//! gives the target its source's stamp, so neither is transferred back)
//! and with how many samples per pixel. Two targets at one base are on the
//! same EDRAM tiles when their rows are as wide in samples (pitch times
//! samples across), whatever their sample counts: a 4x MSAA surface at
//! half the pitch of a 2x one, as games clear a tile with a quarter of the
//! pixels. When a draw, clear or resolve uses a target and the newest
//! write to its tiles went to another such target, that one's pixels are
//! first transferred into it, as Xenia's render target cache transfers
//! EDRAM ownership: the EDRAM is one memory, so the newest write wins
//! whatever its format. Between `k_8_8_8_8` colour, kD24S8 and kD24FS8
//! the pixels are drawn as the shared 32-bit EDRAM word (colour `r | g <<
//! 8 | b << 16 | a << 24`; depth `depth24 << 8 | stencil`, converted like
//! the depth clear and the depth resolve) by the fullscreen passes of
//! `transfer.wgsl`, each destination pixel reading the source pixel of its
//! first sample, in the other 40-sample half of its tile between colour and
//! depth (Xenos stores depth tiles with their halves swapped). Depth
//! destinations take `frag_depth`, then, without
//! shader stencil export, the stencil one bit per pass as Xenia does.
//! Other aliasing is not modelled: each key keeps its own pixels. A newer
//! target on the same tiles that cannot be transferred (a colour format
//! other than `k_8_8_8_8`), a newer target at the same base and pitch
//! with rows of another width (64bpp), or a newer target at another base
//! or row width whose tiles overlap
//! (estimated from the rows written), is only reported in
//! [`Report::warnings`]; colour targets of different formats at one base
//! and pitch are not reported.
//!
//! At the end of a replay a target is dropped when a newer one on the same
//! tiles, whose words transfer to it, has written all its rows: used again,
//! it is made anew and takes that newer target's words. Past a bound on
//! their host memory the least recently written targets are dropped too.
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
use std::collections::{HashMap, HashSet};
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
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        self.mem.read_into(addr, out)
    }
}

impl GuestMemory for Tracked<'_> {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        let start = u64::from(addr & PHYSICAL_MASK);
        self.written.push((start, start + bytes.len() as u64));
        self.mem.write_bytes(addr, bytes)
    }
}

impl Tracked<'_> {
    /// Writes bytes the drawer already holds on the GPU (a kept resolve's),
    /// without recording them as a change.
    fn write_back(&mut self, addr: u32, bytes: &[u8]) -> bool {
        self.mem.write_bytes(addr, bytes)
    }
}

/// Guest bytes that GPU resolves wrote, kept in a GPU buffer so that
/// textures fetched from them are decoded on the GPU instead of read back,
/// hashed, decoded and uploaded. The buffer always holds what guest memory
/// holds (or, for the `dirty` range, will hold once written back) over
/// `[lo, hi)`: it starts as a copy of guest memory, resolves write into it,
/// and any other write to the range drops it.
struct Mirror {
    lo: u64,
    hi: u64,
    buf: wgpu::Buffer,
    /// Physical ranges written by resolves and not yet written back to
    /// guest memory, sorted, apart and not touching.
    dirty: Vec<(u64, u64)>,
    /// Hash of each [`KEPT_BLOCK`] bytes of the buffer from `lo`, as it held
    /// them when made from guest memory or last read back, to notice CPU
    /// writes to guest memory since; `None` until a resolve that covers
    /// the block writes it back.
    hashes: Vec<Option<u64>>,
    /// Blocks checked against guest memory (or made from it) this replay.
    checked: Vec<bool>,
    /// Textures decoded from the buffer, by fetch constant.
    views: HashMap<[u32; 6], Kept>,
}

/// A texture decoded from a [`Mirror`].
struct Kept {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    /// Guest bytes it was decoded from, and the bytes of one 32-row block.
    lo: u64,
    hi: u64,
    block: u64,
    /// Rows written by resolves since it was decoded, to decode again.
    stale: Option<(u32, u32)>,
}

/// Kept resolve buffers are written back and dropped past this many bytes.
const MIRROR_LIMIT: u64 = 256 << 20;

/// Bytes of a kept resolve buffer hashed (and read back) as one: a
/// write-back reads only the blocks a resolve wrote.
const KEPT_BLOCK: u64 = 4096;

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
    /// Command buffer submissions made during the replay.
    pub submits: u32,
    /// `submits` by where they come from: resolve copies (GPU copies and
    /// CPU resolve readbacks), EDRAM transfers, kept resolve texture
    /// decodes, and batch flushes and resolve write-backs. Resolve clears
    /// are recorded into the next batch.
    pub submits_by_site: [u32; 4],
    pub texture_bytes: u64,
    /// Guest vertex window bytes read (not those of reused windows).
    pub guest_bytes: u64,
    /// EDRAM transfers and their render passes.
    pub transfers: u32,
    pub transfer_passes: u32,
    /// Bytes read back from the GPU for resolves.
    pub readback_bytes: u64,
    /// Guest texture bytes read and hashed to look up the texture cache.
    pub hashed_bytes: u64,
    /// Texture lookups served from a resolve kept on the GPU (decoded
    /// there, or decoded earlier), each one a guest read, hash, decode and
    /// upload avoided; and the guest texture bytes they did not read.
    pub textures_kept: u32,
    pub kept_bytes: u64,
    /// Bytes hashed to check that guest memory under the kept resolves did
    /// not change between replays (at write-back, and at first use in the
    /// next replay).
    pub kept_hashed_bytes: u64,
    /// Of `textures_uploaded`, those the texture cache held before and
    /// evicted, and the textures it evicted.
    pub textures_reuploaded: u32,
    pub textures_evicted: u32,
    /// `textures_uploaded` and `texture_bytes` by host format: `Rgba8Unorm`,
    /// BC blocks, `Rgba16Float`, `Rgba32Float`, other.
    pub uploads_by_format: [(u32, u64); 5],
    /// Draws whose vertex window lies in one already staged in the same
    /// batch, bound from there instead of read and staged again; and the
    /// guest bytes of their windows.
    pub windows_reused: u32,
    pub reused_bytes: u64,
    /// Draws whose constants block equals the last one staged in the same
    /// batch, bound from there instead of staged again.
    pub consts_reused: u32,
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
        for (a, b) in self.submits_by_site.iter_mut().zip(t.submits_by_site) {
            *a += b;
        }
        self.texture_bytes += t.texture_bytes;
        self.guest_bytes += t.guest_bytes;
        self.transfers += t.transfers;
        self.transfer_passes += t.transfer_passes;
        self.readback_bytes += t.readback_bytes;
        self.hashed_bytes += t.hashed_bytes;
        self.textures_kept += t.textures_kept;
        self.kept_bytes += t.kept_bytes;
        self.kept_hashed_bytes += t.kept_hashed_bytes;
        self.textures_reuploaded += t.textures_reuploaded;
        self.textures_evicted += t.textures_evicted;
        for (a, b) in self.uploads_by_format.iter_mut().zip(t.uploads_by_format) {
            a.0 += b.0;
            a.1 += b.1;
        }
        self.windows_reused += t.windows_reused;
        self.reused_bytes += t.reused_bytes;
        self.consts_reused += t.consts_reused;
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
#[derive(Debug)]
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
    /// Format, depth write, depth compare and stencil.
    depth: Option<(
        wgpu::TextureFormat,
        bool,
        wgpu::CompareFunction,
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
    /// RB_SURFACE_INFO msaa_samples of that write (0 1x, 1 2x, 2 4x).
    msaa: u32,
}

/// Samples per pixel across and down for RB_SURFACE_INFO msaa_samples
/// (2x stacks its samples vertically, 4x is 2 by 2).
fn samples(msaa: u32) -> (u32, u32) {
    match msaa {
        0 => (1, 1),
        1 => (1, 2),
        _ => (2, 2),
    }
}

/// Width of one EDRAM row of `k`'s tiles, in 32-bit samples.
fn row_width(k: RenderTargetKey, msaa: u32) -> u32 {
    let bytes = match k.format {
        TargetFormat::Rgba16Float | TargetFormat::Rgba16Fixed | TargetFormat::Rg32Float => 2,
        _ => 1,
    };
    k.pitch * samples(msaa).0 * bytes
}

/// EDRAM tiles `[start, end)` that `k`, drawn with `msaa`, covers down to
/// row `rows`.
// shortcut: no wrap past tile 2048; wrap if a capture's overlap report
// needs to be exact.
fn edram_range(k: RenderTargetKey, rows: u32, msaa: u32) -> (u32, u32) {
    let len = row_width(k, msaa).div_ceil(80) * (rows * samples(msaa).1).div_ceil(16);
    (k.base_tiles, k.base_tiles + len)
}

/// Whether the EDRAM word of a target of format `a` can be transferred to
/// one of format `b`: between k_8_8_8_8 colour, kD24S8 and kD24FS8.
// shortcut: colour formats other than k_8_8_8_8 (k_32_FLOAT, 2_10_10_10,
// ...) are not transferred, only reported; add their word conversions to
// transfer.wgsl when a capture aliases them.
fn transferable(a: TargetFormat, b: TargetFormat) -> bool {
    let word = |f: TargetFormat| f == TargetFormat::Rgba8 || f.is_depth();
    word(a) && word(b)
}

/// Draw pipelines kept: a long run made about 300, each with its shaders
/// about 1.6 MB of driver memory.
const PIPELINE_CAP: usize = 1024;

/// Key of a shader module: program hash, vertex stage, extra vertex
/// outputs, variant and pixel parameter generation.
type ModuleKey = (u64, bool, u16, Variant, Option<ParamGen>);

/// The vertex and pixel shader modules a pipeline is made from.
fn module_keys(k: &PipelineKey) -> [ModuleKey; 2] {
    [
        (k.vs, true, k.vs_extra, k.vs_variant, None),
        (k.ps, false, 0, k.ps_variant, k.ps_param_gen),
    ]
}

/// `blend` as the pipeline needs it with colour write mask `mask`: a
/// component whose channels are not written is left as replace, and a
/// state that is all replace is no blending.
fn pipeline_blend(
    blend: Option<wgpu::BlendState>,
    mask: wgpu::ColorWrites,
) -> Option<wgpu::BlendState> {
    let mut b = blend?;
    if !mask.contains(wgpu::ColorWrites::ALPHA) {
        b.alpha = wgpu::BlendComponent::REPLACE;
    }
    if !mask.intersects(wgpu::ColorWrites::COLOR) {
        b.color = wgpu::BlendComponent::REPLACE;
    }
    (b != wgpu::BlendState::REPLACE).then_some(b)
}

/// `st` as the pipeline needs it under `cull`: a culled face takes the
/// other's state, operations are keep when nothing is written, the read
/// mask is 0 when nothing compares, and a state that neither tests nor
/// writes is no stencil.
fn pipeline_stencil(
    mut st: wgpu::StencilState,
    cull: Option<wgpu::Face>,
) -> Option<wgpu::StencilState> {
    use wgpu::{CompareFunction as C, StencilOperation as Op};
    match cull {
        Some(wgpu::Face::Back) => st.back = st.front,
        Some(wgpu::Face::Front) => st.front = st.back,
        None => {}
    }
    for f in [&mut st.front, &mut st.back] {
        if st.write_mask == 0 {
            (f.fail_op, f.depth_fail_op, f.pass_op) = (Op::Keep, Op::Keep, Op::Keep);
        }
        if f.compare == C::Always {
            f.fail_op = Op::Keep;
        }
    }
    let always = st.front.compare == C::Always && st.back.compare == C::Always;
    if always {
        st.read_mask = 0;
    }
    let keeps = |f: &wgpu::StencilFaceState| f.depth_fail_op == Op::Keep && f.pass_op == Op::Keep;
    if always && keeps(&st.front) && keeps(&st.back) {
        return None;
    }
    if st.write_mask != 0
        && keeps(&st.front)
        && keeps(&st.back)
        && st.front.fail_op == Op::Keep
        && st.back.fail_op == Op::Keep
    {
        st.write_mask = 0;
    }
    Some(st)
}

/// Bytes of a texture, estimated from its size and format.
fn texture_bytes(t: &wgpu::Texture) -> u64 {
    let f = t.format();
    let texel = f.block_copy_size(None).map_or_else(
        // Depth + stencil formats: 4 or 8 bytes a texel.
        || if f.has_stencil_aspect() { 8 } else { 4 },
        u64::from,
    );
    let e = t.size();
    // Block-compressed formats count bytes per block, not per texel.
    let (bw, bh) = f.block_dimensions();
    texel
        * u64::from(
            e.width.div_ceil(bw)
                * e.height.div_ceil(bh)
                * e.depth_or_array_layers
                * t.sample_count(),
        )
}

/// Bound on the host memory of render targets (EDRAM is 10 MB; host
/// targets are 2048 rows and wider texels): past it the least recently
/// written ones are dropped at the end of a replay.
const TARGET_CAP: u64 = 256 << 20;

/// Replays recorded command lists on a headless wgpu device.
pub struct Drawer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Adapter name, backend and device type (a software one is `Cpu`),
    /// for logs.
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
    modules: HashMap<ModuleKey, wgpu::ShaderModule>,
    /// With the use that last drew with each, for least recently used
    /// eviction past `pipeline_cap`.
    pipelines: HashMap<PipelineKey, (wgpu::RenderPipeline, u64)>,
    pipeline_uses: u64,
    pipeline_cap: usize,
    pipeline_drops: u64,
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
    /// Render targets dropped so far (see [`Drawer::drop_targets`]).
    target_drops: u64,
    target_cap: u64,
    transfer_module: Option<wgpu::ShaderModule>,
    /// Keyed by source and destination host format and pass (stencil bit
    /// 0..8, or 8 for the depth or colour pass).
    transfer_pipelines:
        HashMap<(wgpu::TextureFormat, wgpu::TextureFormat, u32), wgpu::RenderPipeline>,
    /// Destination endian of the last 32bpp resolve to each guest address,
    /// so a swap reads its front buffer in the byte order it was written.
    resolved_endian: HashMap<u32, u32>,
    /// Decoded guest textures keyed by the decode fields of the fetch
    /// constant ([`decode_key`]) and a hash of the guest bytes they read, so
    /// an unchanged texture is not decoded again; with their size and the
    /// replay that last used them, for least recently used eviction.
    texture_cache: HashMap<([u32; 6], u64), (wgpu::TextureView, u64, u64)>,
    texture_cache_bytes: u64,
    /// Hashes of keys evicted from `texture_cache`, to count uploads of a
    /// texture the cache held before (cache too small) apart from new ones.
    texture_evicted: HashSet<u64>,
    /// Replays started, the clock of `texture_cache` uses.
    replays: u64,
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
    /// Vertex windows staged in the batch: guest physical `[lo, hi)` and
    /// their (offset, size) in `arena`. Cleared at each flush, as the next
    /// batch overwrites the arena, and dropped where guest memory changes.
    window_memo: Vec<(u64, u64, (u64, u64))>,
    /// The last `Consts` block staged in the batch and its (offset, size)
    /// in `arena`, bound again by a draw with the same bytes. Cleared with
    /// `window_memo` at each flush.
    staged_consts: Option<(Vec<u8>, (u64, u64))>,
    max_buffer: u64,
    /// 1x1 texture of zeros bound for a slot without a usable texture.
    zero_texture: wgpu::TextureView,
    zero_sampler: wgpu::Sampler,
    /// Whether `Rgba32Float` textures can be sampled with filtering; without
    /// it wide formats are uploaded as `Rgba16Float`.
    float32_filterable: bool,
    /// Compute pipeline of resolve.wgsl, made on the first GPU resolve.
    /// Colour, depth and depth + stencil entry points of resolve.wgsl.
    resolve_pipelines: [Option<wgpu::ComputePipeline>; 3],
    /// kD24FS8 is hosted as `Depth32FloatStencil8` (else `Depth32Float`).
    float_stencil: bool,
    /// `Rg16Snorm` and `Rgba16Snorm` can be render targets (k_16_16 and
    /// k_16_16_16_16 are refused without them).
    snorm16: bool,
    /// The snorm16 targets blend (else their blending is dropped).
    snorm16_blend: bool,
    /// [`formats::half_lut`] tables by source format and exponent bias, for
    /// CPU resolves to 16-bit float destinations.
    half_luts: HashMap<(TargetFormat, i32), Option<Vec<[u16; 2]>>>,
    /// Resolve destinations kept on the GPU, disjoint and not adjacent.
    mirrors: Vec<Mirror>,
    /// Entry points decode_8888 and decode_float of resolve.wgsl.
    decode_pipelines: [Option<wgpu::ComputePipeline>; 2],
    /// Leave resolve results on the GPU at the end of a replay instead of
    /// writing them to guest memory (default false). The drawer still writes
    /// them back before it reads those bytes itself (a texture it cannot
    /// decode on the GPU, vertices, a front buffer, a MEM_WRITE or CPU
    /// resolve over them), but the guest CPU never sees them.
    ///
    /// Unsafe for games whose CPU reads resolved surfaces (a screenshot,
    /// a luminance readback, a resolve to vertex data read by the CPU): they
    /// read stale bytes. A CPU write to such a destination between replays
    /// drops the GPU copy (its pixels are lost).
    pub skip_resolve_writeback: bool,
    /// Bytes of decoded guest textures the texture cache keeps before
    /// evicting the least recently used ones (default 256 MiB).
    pub texture_cache_limit: u64,
    /// Upload unsigned DXT textures as BC blocks for the host to decode
    /// (default: whether the adapter has BC textures). Turning it on
    /// without them makes those uploads fail.
    pub bc_textures: bool,
    /// Diagnostic: report a warning line per distinct texture a draw
    /// samples (see [`texture_log_line`]): for every pixel shader when
    /// `Some(None)`, for the pixel shader of that program hash when
    /// `Some(Some(hash))`. Default from `XENON_TEXTURE_LOG` (`all`, or the
    /// hash in hex); off without it. Draws whose shaders fail translation
    /// (such as a cube or 3D fetch) do not log.
    pub texture_log: Option<Option<u64>>,
    timings: Timings,
    uncaptured: Arc<Mutex<Vec<String>>>,
    max_dim: u32,
    max_storage: u64,
    /// Last, so it drops after the device and every resource made from it.
    instance: wgpu::Instance,
}

/// Draws recorded into one command encoder, in one render pass while the
/// render targets stay the same, and the resolve clears between them, each
/// in a pass of its own. Submitted by [`Drawer::flush`] before anything
/// that reads or writes the targets outside the batch (resolves and EDRAM
/// transfers) and at the end of a replay.
struct Batch {
    /// Dropped (ended) before `enc` is finished.
    pass: Option<(PassKey, wgpu::RenderPass<'static>)>,
    enc: wgpu::CommandEncoder,
}

/// One buffer of a draw's staged data: bytes, or `len` guest bytes at
/// `addr` read straight into the staging area.
#[derive(Clone, Copy)]
enum Part<'a> {
    Bytes(&'a [u8]),
    Guest(u32, usize),
}

impl Part<'_> {
    fn len(&self) -> usize {
        match *self {
            Part::Bytes(b) => b.len(),
            Part::Guest(_, len) => len,
        }
    }
}

/// Where a queue submission comes from: the index of
/// [`Timings::submits_by_site`].
#[derive(Clone, Copy)]
enum Site {
    Resolve,
    Transfer,
    Decode,
    Other,
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
/// The fetch constant fields the decoded texels depend on: type, signs,
/// pitch and tiling (dword 0), format, endian and address (1), size (2),
/// swizzle (3) and dimension (5). Clamp, filters, LOD, exp_adjust, border
/// and mip fields only change sampling, so textures differing in those
/// share one decode.
fn decode_key(c: [u32; 6]) -> [u32; 6] {
    [
        c[0] & !(0x1FF << 10),
        c[1] & !(1 << 11),
        c[2],
        c[3] & (0xFFF << 1),
        0,
        c[5] & (3 << 9),
    ]
}

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

/// The `Consts` uniform of the shader binding contract: float, fetch, bool
/// and loop constants, then the window base and the index offset.
fn consts(r: &regs::RegisterFile, window_base: u32) -> Vec<u8> {
    // Float and fetch constants are adjacent, and so are bool and loop.
    const _: () = assert!(regs::SHADER_CONSTANT_FLOAT + 2048 == regs::SHADER_CONSTANT_FETCH);
    const _: () = assert!(regs::SHADER_CONSTANT_BOOL + 8 == regs::SHADER_CONSTANT_LOOP);
    let tail = [window_base, r.get(regs::VGT_INDX_OFFSET), 0, 0];
    let blocks = [
        r.slice(regs::SHADER_CONSTANT_FLOAT, 2048 + 192),
        r.slice(regs::SHADER_CONSTANT_BOOL, 8 + 32),
        &tail,
    ];
    let mut b = vec![0; 4 * (2048 + 192 + 8 + 32 + 4)];
    let mut at = 0;
    for block in blocks {
        for (o, w) in b[at..][..4 * block.len()].chunks_exact_mut(4).zip(block) {
            o.copy_from_slice(&w.to_le_bytes());
        }
        at += 4 * block.len();
    }
    b
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
    /// Opens a headless device: Vulkan or GL, preferring the discrete GPU
    /// (a desktop's integrated one would otherwise be picked first), first
    /// without and then with `force_fallback_adapter`.
    pub fn new() -> Result<Drawer, Error> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(desc);
        let options = |force_fallback_adapter| wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
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
        let mut required_features = adapter.features()
            & (wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::TEXTURE_COMPRESSION_BC);
        required_features.set(wgpu::Features::DEPTH32FLOAT_STENCIL8, float_stencil);
        required_features.set(snorm16_features, snorm16);
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("xenon-draw"),
            required_limits: limits.clone(),
            required_features,
            // 8-64 MB allocator blocks instead of 128-256 MB: with textures
            // evicted and uploaded all the time, big blocks fragment and stay
            // reserved (a 360 s run held 1,344 MB for 510 MB used).
            memory_hints: wgpu::MemoryHints::MemoryUsage,
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
            instance,
            device,
            queue,
            adapter: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
            timeout: DEFAULT_TIMEOUT,
            regs: RegisterFile::default(),
            vs: None,
            ps: None,
            translated: HashMap::new(),
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            pipeline_uses: 0,
            pipeline_cap: PIPELINE_CAP,
            pipeline_drops: 0,
            texture_layouts: HashMap::new(),
            group0,
            clear_layout,
            clear_module,
            clear_pipelines: HashMap::new(),
            targets: HashMap::new(),
            written: HashMap::new(),
            target_drops: 0,
            target_cap: TARGET_CAP,
            clock: 0,
            transfer_module: None,
            transfer_pipelines: HashMap::new(),
            resolved_endian: HashMap::new(),
            texture_cache: HashMap::new(),
            texture_cache_bytes: 0,
            texture_evicted: HashSet::new(),
            replays: 0,
            texture_memo: HashMap::new(),
            samplers: HashMap::new(),
            batch: None,
            staged: Vec::new(),
            arena: None,
            window_memo: Vec::new(),
            staged_consts: None,
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
            half_luts: HashMap::new(),
            mirrors: Vec::new(),
            decode_pipelines: [None, None],
            skip_resolve_writeback: false,
            texture_cache_limit: TEXTURE_CACHE_LIMIT,
            bc_textures: required_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            texture_log: std::env::var("XENON_TEXTURE_LOG")
                .ok()
                .and_then(|v| match v.trim() {
                    "all" => Some(None),
                    h => {
                        let hex = h.trim_start_matches("0x").trim_start_matches("0X");
                        let r = u64::from_str_radix(hex, 16).ok().map(Some);
                        if r.is_none() {
                            eprintln!("XENON_TEXTURE_LOG={v:?} is not `all` or a hex hash; texture log off");
                        }
                        r
                    }
                }),
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

    /// Uploads those textures as `Rgba16Float` from now on, as on a host
    /// without filterable 32-bit floats, so tests can reach that path on
    /// any adapter. Call it before the first replay: cached textures keep
    /// their format.
    #[doc(hidden)]
    pub fn disable_float32_textures(&mut self) {
        self.float32_filterable = false;
    }

    /// What the drawer holds, for the run log: cached textures, render
    /// targets (bytes estimated from their size and format), the draw upload
    /// arena and its host staging copy, pipeline counts (with the distinct
    /// vertex and pixel shader pairs of the draw pipelines and how many were
    /// dropped), shader module counts and the translated shaders with the
    /// bytes of their WGSL.
    pub fn memory_line(&self) -> String {
        let rt: u64 = self.targets.values().map(texture_bytes).sum();
        let kept: u64 = self.mirrors.iter().map(|m| m.buf.size()).sum::<u64>()
            + self
                .mirrors
                .iter()
                .flat_map(|m| m.views.values())
                .map(|k| texture_bytes(&k.tex))
                .sum::<u64>();
        let pairs: std::collections::HashSet<(u64, u64)> =
            self.pipelines.keys().map(|k| (k.vs, k.ps)).collect();
        let wgsl: Vec<usize> = self
            .translated
            .values()
            .flatten()
            .map(|t| t.wgsl.len())
            .collect();
        format!(
            "draw: textures {} MB in {}, targets {} MB in {}, {} dropped, kept resolves {} MB in {}, arena {} MB, staged {} MB, pipelines {} ({} vs/ps pairs, {} dropped), shaders {}, translated {} ({} KB WGSL)",
            self.texture_cache_bytes >> 20,
            self.texture_cache.len(),
            rt >> 20,
            self.targets.len(),
            self.target_drops,
            kept >> 20,
            self.mirrors.len(),
            self.arena.as_ref().map_or(0, |b| b.size()) >> 20,
            self.staged.capacity() >> 20,
            self.pipelines.len() + self.clear_pipelines.len(),
            pairs.len(),
            self.pipeline_drops,
            self.modules.len(),
            wgsl.len(),
            wgsl.iter().sum::<usize>().div_ceil(1024),
        ) + &self.wgpu_line()
    }

    /// What wgpu itself holds: live objects (a count that keeps rising is a
    /// leak) and the GPU allocator's reserved against allocated bytes (a gap
    /// that keeps widening is memory kept after frees).
    fn wgpu_line(&self) -> String {
        let h = self.device.get_internal_counters().hal;
        let alloc = self.device.generate_allocator_report().map_or_else(
            || "no allocator report".to_string(),
            |r| {
                format!(
                    "GPU memory {} MB used of {} MB reserved in {} blocks",
                    r.total_allocated_bytes >> 20,
                    r.total_reserved_bytes >> 20,
                    r.blocks.len()
                )
            },
        );
        format!(
            "; wgpu live: textures {} MB, buffers {} ({} MB), views {}, bind groups {}, pipelines {}, shaders {}; {alloc}",
            // hal's texture count misses some textures on Vulkan; its bytes do not.
            h.texture_memory.read() >> 20,
            h.buffers.read(),
            h.buffer_memory.read() >> 20,
            h.texture_views.read(),
            h.bind_groups.read(),
            h.render_pipelines.read() + h.compute_pipelines.read(),
            h.shader_modules.read(),
        )
    }

    /// What wgpu itself holds, beside what [`Drawer::memory_line`] counts:
    /// bytes in live allocations (where the backend reports them) and live
    /// textures, texture views, bind groups and buffers. For tests.
    #[doc(hidden)]
    pub fn gpu_live(&self) -> (Option<u64>, [usize; 4]) {
        let bytes = self
            .device
            .generate_allocator_report()
            .map(|r| r.total_allocated_bytes);
        let hub = self.instance.generate_report().map(|r| r.hub);
        let n = hub.map_or([0; 4], |h| {
            [h.textures, h.texture_views, h.bind_groups, h.buffers].map(|r| r.num_kept_from_user)
        });
        (bytes, n)
    }

    /// Sets how many draw pipelines are kept (least recently used ones
    /// past it are dropped, with the shaders only they used). For tests.
    #[doc(hidden)]
    pub fn set_pipeline_cap(&mut self, pipelines: usize) {
        self.pipeline_cap = pipelines;
        self.drop_pipelines();
    }

    /// Drops the least recently used draw pipelines past the cap, and the
    /// shader modules and translations no kept pipeline uses.
    fn drop_pipelines(&mut self) {
        while self.pipelines.len() > self.pipeline_cap {
            let Some(old) = self
                .pipelines
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.pipelines.remove(&old);
            self.pipeline_drops += 1;
            for m in module_keys(&old) {
                if self.pipelines.keys().any(|k| module_keys(k).contains(&m)) {
                    continue;
                }
                self.modules.remove(&m);
                let (hash, vertex, _, _, param_gen) = m;
                if !self
                    .modules
                    .keys()
                    .any(|n| (n.0, n.1, n.4) == (hash, vertex, param_gen))
                {
                    self.translated.remove(&(hash, vertex, param_gen));
                }
            }
        }
    }

    /// Keys of the render targets the drawer holds: every one created so
    /// far, less those dropped at the end of a replay (a newer target on
    /// the same EDRAM tiles holds their words, or targets were over their
    /// memory cap). Reading a dropped one gives [`Error::NoRenderTarget`].
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
        self.replays += 1;
        for m in &mut self.mirrors {
            m.checked.fill(false);
        }
        let mut mem = Tracked {
            mem,
            written: Vec::new(),
        };
        let done = self.replay_commands(commands, &mut mem, &mut report);
        // Submitted on an error too: the batch can hold resolve clears of
        // targets that outlive the failed replay, and the first error wins.
        let flushed = self.flush().map_err(|e| match e {
            Skip::Fatal(e) => e,
            Skip::Unsupported(why) => Error::Gpu(why),
        });
        let done = done.and(flushed);
        self.batch = None;
        self.staged.clear();
        self.window_memo.clear();
        self.staged_consts = None;
        // Resolves done before an error still reach guest memory.
        let kept = self.finish_mirrors(&mut mem);
        done?;
        kept?;
        self.drop_targets(&mut report);
        // Counts made where only the report is mutable (durations there
        // are zero).
        let mut t = std::mem::take(&mut self.timings);
        t += &report.timings;
        report.timings = t;
        Ok(report)
    }

    fn replay_commands(
        &mut self,
        commands: &[Command],
        tracked: &mut Tracked,
        report: &mut Report,
    ) -> Result<(), Error> {
        for c in commands {
            let mem = &mut *tracked;
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
                        Ok(Ok(f)) => report.frames.push(f),
                        Ok(Err(e)) => report.warn(e),
                        Err(e) => return Err(e),
                    }
                    Ok(())
                }
                Command::MemWrite {
                    addr,
                    endian,
                    values,
                } => {
                    let a = u64::from(addr & PHYSICAL_MASK);
                    self.sync(mem, a, a + 4 * values.len() as u64)?;
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
                self.forget_guest(lo, hi);
                // Any write but a resolve's makes a kept copy stale; the
                // writer wrote it back first (see `sync`).
                self.mirrors.retain(|m| {
                    debug_assert!(m.hi <= lo || hi <= m.lo || m.dirty.is_empty());
                    m.hi <= lo || hi <= m.lo
                });
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

    /// Drops the textures looked up this replay from guest bytes in
    /// `[lo, hi)`, which just changed.
    fn forget_guest(&mut self, lo: u64, hi: u64) {
        self.texture_memo
            .retain(|_, &mut (_, a, b)| b <= lo || hi <= a);
        self.window_memo.retain(|&(a, b, _)| b <= lo || hi <= a);
    }

    /// Hash of one [`KEPT_BLOCK`] of a kept resolve.
    fn block_hash(&mut self, bytes: &[u8]) -> u64 {
        self.timings.kept_hashed_bytes += bytes.len() as u64;
        hash_bytes(bytes)
    }

    /// Whether the blocks of kept resolve `i` over `[lo, hi)` still match
    /// guest memory, which the CPU may have written since the last replay.
    /// Only the blocks about to be used are checked, so a replay that uses
    /// one tile of a large buffer does not hash all of it. A mismatch drops
    /// it (in `skip_resolve_writeback` mode with its unwritten pixels, as
    /// the CPU wrote after them).
    fn verify(&mut self, i: usize, mem: &Tracked, lo: u64, hi: u64) -> bool {
        let (m_lo, m_hi) = (self.mirrors[i].lo, self.mirrors[i].hi);
        let k0 = (lo.max(m_lo) - m_lo) / KEPT_BLOCK;
        let k1 = (hi.min(m_hi) - m_lo).div_ceil(KEPT_BLOCK);
        for k in k0 as usize..k1 as usize {
            if self.mirrors[i].checked[k] {
                continue;
            }
            let a = m_lo + k as u64 * KEPT_BLOCK;
            let b = (a + KEPT_BLOCK).min(m_hi);
            let now = mem
                .read_bytes(a as u32, (b - a) as usize)
                .map(|b| self.block_hash(&b));
            if now.is_none() || now != self.mirrors[i].hashes[k] {
                self.mirrors.remove(i);
                return false;
            }
            self.mirrors[i].checked[k] = true;
        }
        true
    }

    /// Writes back to guest memory every kept resolve over `[lo, hi)` that
    /// has unwritten bytes, before the drawer reads or writes that range
    /// itself. One GPU wait for all of them.
    fn sync(&mut self, mem: &mut Tracked, lo: u64, hi: u64) -> Result<(), Error> {
        let mut i = 0;
        let mut copies = Vec::new();
        let mut enc = None;
        while i < self.mirrors.len() {
            let m = &self.mirrors[i];
            if m.hi <= lo || hi <= m.lo || m.dirty.is_empty() {
                i += 1;
                continue;
            }
            // A failed check removed mirror i; the next one moved to i.
            let dirty = m.dirty.clone();
            if !dirty.iter().all(|&(a, b)| self.verify(i, mem, a, b)) {
                continue;
            }
            // Whole blocks come back, not only the dirty ranges: their
            // hashes must be of what the buffer holds, as guest memory may
            // have changed under it since it was made (a CPU write during
            // the replay), and then the next replay's check drops it.
            let m = &self.mirrors[i];
            let len = m.hi - m.lo;
            // Buffer offsets of the dirty blocks, merged.
            let mut spans: Vec<(u64, u64)> = Vec::new();
            for &(a, b) in &m.dirty {
                let s = (a - m.lo) / KEPT_BLOCK * KEPT_BLOCK;
                let e = ((b - m.lo).div_ceil(KEPT_BLOCK) * KEPT_BLOCK).min(len);
                match spans.last_mut() {
                    Some(l) if s <= l.1 => l.1 = l.1.max(e),
                    _ => spans.push((s, e)),
                }
            }
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resolve write-back"),
                size: spans.iter().map(|(s, e)| e - s).sum(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let enc =
                enc.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            let mut at = 0;
            for &(s, e) in &spans {
                enc.copy_buffer_to_buffer(&m.buf, s, &buf, at, e - s);
                at += e - s;
            }
            copies.push((m.lo, m.dirty.clone(), spans, buf));
            i += 1;
        }
        let Some(enc) = enc else {
            return Ok(());
        };
        let mut t = Instant::now();
        self.wait_idle()?;
        self.timings.gpu_wait += lap(&mut t);
        let index = self.queue.submit([enc.finish()]);
        self.submitted(Site::Other);
        for (m_lo, dirty, spans, buf) in copies {
            let data = self.map_read(&buf, index.clone())?;
            self.timings.readback += lap(&mut t);
            self.timings.readback_bytes += data.len() as u64;
            // Index in `data` of each span's first byte.
            let starts: Vec<u64> = spans
                .iter()
                .scan(0, |at, (s, e)| Some(std::mem::replace(at, *at + e - s)))
                .collect();
            let ok = dirty.iter().all(|&(a, b)| {
                let k = spans.partition_point(|&(_, e)| e <= a - m_lo);
                let from = (starts[k] + a - m_lo - spans[k].0) as usize;
                mem.write_back(a as u32, &data[from..from + (b - a) as usize])
            });
            let Some(i) = self.mirrors.iter().position(|m| m.lo == m_lo) else {
                continue;
            };
            self.mirrors[i].dirty.clear();
            if ok {
                for (&(s, e), &at) in spans.iter().zip(&starts) {
                    let span = &data[at as usize..(at + e - s) as usize];
                    for (j, block) in span.chunks(KEPT_BLOCK as usize).enumerate() {
                        let h = self.block_hash(block);
                        self.mirrors[i].hashes[(s / KEPT_BLOCK) as usize + j] = Some(h);
                    }
                }
            } else {
                // Readable when made, so not expected; the pixels are lost.
                self.mirrors.remove(i);
            }
            self.timings.resolve_cpu += lap(&mut t);
        }
        Ok(())
    }

    /// End of a replay: writes back every kept resolve (unless
    /// `skip_resolve_writeback`). A block left unhashed, made for a resolve
    /// that failed before writing to it, fails its next check: hashing
    /// guest memory now could miss a CPU write made after it was filled.
    fn finish_mirrors(&mut self, mem: &mut Tracked) -> Result<(), Error> {
        if !self.skip_resolve_writeback {
            self.sync(mem, 0, u64::MAX)?;
        }
        Ok(())
    }

    /// Index of the kept resolve buffer holding `[lo, hi)`, which a resolve
    /// is about to write, made from guest memory if there is none. Kept buffers it overlaps or touches are
    /// written back and merged into it, so a texture over several resolves
    /// (tiles, one under the other) is in one buffer.
    fn mirror(&mut self, mem: &mut Tracked, lo: u64, hi: u64) -> Result<Option<usize>, Error> {
        if let Some(i) = self.mirrors.iter().position(|m| m.lo <= lo && hi <= m.hi) {
            if self.verify(i, mem, lo, hi) {
                return Ok(Some(i));
            }
        }
        let near = |m: &Mirror| m.lo <= hi && lo <= m.hi;
        let (mut a, mut b) = (lo, hi);
        for m in self.mirrors.iter().filter(|m| near(m)) {
            (a, b) = (a.min(m.lo), b.max(m.hi));
        }
        if b - a > self.max_storage {
            (a, b) = (lo, hi);
        }
        let total: u64 = self.mirrors.iter().map(|m| m.hi - m.lo).sum();
        // shortcut: past the limit every kept resolve is written back and
        // dropped, not the least recently used; track use if a capture
        // keeps more than 256 MB of resolve destinations.
        let all = total + (b - a) > MIRROR_LIMIT;
        let (sa, sb) = if all { (0, u64::MAX) } else { (a, b) };
        // Those touching [sa, sb) too.
        self.sync(mem, sa.saturating_sub(1), sb.saturating_add(1))?;
        self.mirrors.retain(|m| !all && (m.hi < sa || sb < m.lo));
        let Some(bytes) = mem.read_bytes(a as u32, (b - a) as usize) else {
            return Ok(None);
        };
        // With write-back, the blocks the resolve covers are hashed when
        // it writes them back.
        let covers = |k: u64| {
            let s = a + k * KEPT_BLOCK;
            !self.skip_resolve_writeback && lo <= s && (s + KEPT_BLOCK).min(b) <= hi
        };
        let covered: Vec<bool> = (0..(b - a).div_ceil(KEPT_BLOCK)).map(covers).collect();
        let hashes = bytes
            .chunks(KEPT_BLOCK as usize)
            .zip(&covered)
            .map(|(block, &c)| (!c).then(|| self.block_hash(block)))
            .collect();
        let checked = vec![true; covered.len()];
        let buf = self.buffer(
            "kept resolve",
            &bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        self.mirrors.push(Mirror {
            lo: a,
            hi: b,
            buf,
            dirty: Vec::new(),
            hashes,
            checked,
            views: HashMap::new(),
        });
        Ok(Some(self.mirrors.len() - 1))
    }

    /// Counts one queue submission, made by `site`.
    fn submitted(&mut self, site: Site) {
        self.timings.submits += 1;
        self.timings.submits_by_site[site as usize] += 1;
    }

    /// Submits the batched draws, with their staged buffers.
    fn flush(&mut self) -> Result<(), Skip> {
        // The arena upload counts as submit time too.
        let t = Instant::now();
        self.window_memo.clear();
        self.staged_consts = None;
        if let (Some(arena), false) = (&self.arena, self.staged.is_empty()) {
            self.queue.write_buffer(arena, 0, &self.staged);
            self.staged.clear();
        }
        let Some(mut batch) = self.batch.take() else {
            self.timings.submit += t.elapsed();
            return Ok(());
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        drop(batch.pass.take());
        self.queue.submit([batch.enc.finish()]);
        self.submitted(Site::Other);
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
    fn stage(&mut self, mem: &dyn GuestRead, parts: &[Part]) -> Result<Vec<(u64, u64)>, Skip> {
        let size = |p: &Part| (p.len() as u64).max(4).next_multiple_of(4);
        let align = self.arena_align;
        let need: u64 = parts.iter().map(|p| size(p).next_multiple_of(align)).sum();
        let have = self.arena.as_ref().map_or(0, |b| b.size());
        if self.staged.len() as u64 + need > have {
            self.flush()?;
            if need > have {
                if need > self.max_buffer {
                    return unsupported(format!("{need} bytes of draw data"));
                }
                let size = need.max(ARENA_MIN).next_power_of_two().min(self.max_buffer);
                // Flushed, so empty: the batch never stages more than the
                // arena holds, so it never grows past this (a Vec doubling
                // from an odd size would).
                self.staged = Vec::with_capacity(size as usize);
                self.arena = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("draw arena"),
                    size,
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
            match *p {
                Part::Bytes(b) => self.staged.extend_from_slice(b),
                Part::Guest(addr, len) => {
                    self.staged.resize(off as usize + len, 0);
                    if !mem.read_into(addr, &mut self.staged[off as usize..]) {
                        self.staged.truncate(off as usize);
                        let end = u64::from(addr) + len as u64;
                        return unsupported(format!(
                            "vertex data at 0x{addr:08X}..0x{end:08X} not readable"
                        ));
                    }
                }
            }
            self.staged.resize((off + size(p)) as usize, 0);
            out.push((off, size(p)));
            let end = self.staged.len() as u64;
            self.staged.resize(end.next_multiple_of(align) as usize, 0);
        }
        Ok(out)
    }

    /// Whether `parts` can be staged without submitting the batch first.
    fn fits(&self, parts: &[Part]) -> bool {
        let size = |p: &Part| (p.len() as u64).max(4).next_multiple_of(4);
        let need: u64 = parts
            .iter()
            .map(|p| size(p).next_multiple_of(self.arena_align))
            .sum();
        self.staged.len() as u64 + need <= self.arena.as_ref().map_or(0, |b| b.size())
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

    /// The texture of render target `key`, drawn with `msaa` samples, made
    /// current first: when another target on the same EDRAM tiles was
    /// written after it, the newest one's pixels are transferred into it
    /// (the EDRAM holds one set of bytes for both). `rows` is how far down
    /// this use reaches, for the overlap report.
    fn target(
        &mut self,
        key: RenderTargetKey,
        rows: u32,
        msaa: u32,
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
        let range = edram_range(key, rows.max(own.rows), msaa);
        let mut newest: Option<(RenderTargetKey, Written)> = None;
        for (&k, &w) in &self.written {
            if w.stamp <= own.stamp || k == key {
                continue;
            }
            let same_tiles = row_width(k, w.msaa) == row_width(key, msaa);
            if k.base_tiles == key.base_tiles && (k.pitch == key.pitch || same_tiles) {
                // shortcut: colour targets of different formats at one base
                // and pitch keep their own pixels, silently; transfer them
                // too if a capture reinterprets colour in place.
                let with_depth = k.format.is_depth() || key.format.is_depth();
                if k.pitch == key.pitch && !with_depth {
                    continue;
                }
                // A transfer leaves both with one stamp: the one at the
                // same pitch holds the words without resampling.
                if newest.is_none_or(|(n, nw)| {
                    w.stamp > nw.stamp
                        || (w.stamp == nw.stamp && k.pitch == key.pitch && n.pitch != key.pitch)
                }) {
                    newest = Some((k, w));
                }
            } else {
                let other = edram_range(k, w.rows, w.msaa);
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
            let same_tiles = row_width(src, w.msaa) == row_width(key, msaa);
            if transferable(src.format, key.format) && same_tiles {
                // The rows of `key` whose first sample row is in `src`'s.
                let (src_down, dst_down) = (samples(w.msaa).1, samples(msaa).1);
                let height = RENDER_TARGET_HEIGHT.min(self.max_dim);
                let rows = (w.rows * src_down).div_ceil(dst_down).min(height);
                let t = Instant::now();
                self.transfer((src, w.msaa), (key, msaa), &tex, rows)?;
                self.timings.transfer += t.elapsed();
                // Both now hold the same words: the source's stamp, so
                // using the source again does not transfer them back.
                let own = self.written.entry(key).or_default();
                own.stamp = w.stamp;
                own.rows = own.rows.max(rows);
                own.msaa = msaa;
            } else {
                report.warn(format!(
                    "{:?} render target aliases a newer {:?} one at its EDRAM base; not transferred",
                    key.format, src.format
                ));
            }
        }
        Ok(tex)
    }

    /// Drops the host textures of render targets that a newer target on
    /// the same EDRAM tiles, one their words transfer to, has written all
    /// the rows of, with no newer alias whose words do not transfer:
    /// their next use takes that target's words (or newer ones) as it
    /// would without the drop. Then, past the cap, drops the
    /// least recently written targets; one still holding the newest
    /// pixels of its tiles comes back cleared, as EDRAM garbage would, and
    /// is reported. Runs with no batch recorded.
    // shortcut: a dropped target used again at another MSAA mode (no
    // transfer then), or after a newer target that wrote fewer rows,
    // starts cleared where it kept its own stale pixels before; track
    // per-tile ownership if a capture needs those pixels.
    fn drop_targets(&mut self, report: &mut Report) {
        let rows_down = |w: Written| w.rows * samples(w.msaa).1;
        let dead: Vec<RenderTargetKey> = self
            .written
            .iter()
            .filter(|&(&k, &w)| {
                self.written.iter().any(|(&n, &nw)| {
                    nw.stamp > w.stamp
                        && n.base_tiles == k.base_tiles
                        && row_width(n, nw.msaa) == row_width(k, w.msaa)
                        && transferable(n.format, k.format)
                        && rows_down(nw) >= rows_down(w)
                        // `target` would pick a newer alias first and not
                        // transfer from it, so keep k's own words.
                        && !self.written.iter().any(|(&m, &mw)| {
                            mw.stamp > nw.stamp
                                && m.base_tiles == k.base_tiles
                                && !(transferable(m.format, k.format)
                                    && row_width(m, mw.msaa) == row_width(k, w.msaa))
                        })
                })
            })
            .map(|(&k, _)| k)
            .collect();
        for k in dead {
            self.written.remove(&k);
            self.targets.remove(&k);
            self.target_drops += 1;
        }
        let mut total: u64 = self.targets.values().map(texture_bytes).sum();
        if total <= self.target_cap {
            return;
        }
        let mut by_age: Vec<(u64, RenderTargetKey)> = self
            .targets
            .keys()
            .map(|&k| (self.written.get(&k).map_or(0, |w| w.stamp), k))
            .collect();
        // Ties (a transfer shares its stamp) in a fixed order.
        by_age.sort_unstable_by_key(|&(stamp, k)| (stamp, k.base_tiles, k.pitch, k.format as u8));
        for (_, k) in by_age {
            if total <= self.target_cap {
                break;
            }
            total -= self.targets.remove(&k).as_ref().map_or(0, texture_bytes);
            self.target_drops += 1;
            if self.written.remove(&k).is_some() {
                report.warn(
                    "render target memory over its cap: a target still holding its EDRAM tiles' newest pixels was dropped (it comes back cleared)",
                );
            }
        }
    }

    /// Records a write to `key`, drawn with `msaa`, down to row `rows`.
    fn mark_written(&mut self, key: RenderTargetKey, rows: u32, msaa: u32) {
        self.clock += 1;
        let w = self.written.entry(key).or_default();
        w.stamp = self.clock;
        w.rows = w.rows.max(rows);
        w.msaa = msaa;
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
        let from = match src {
            _ if !src.is_depth_stencil_format() => "color",
            _ if src.has_stencil_aspect() => "depth_stencil",
            _ => "depth",
        };
        let to = match (to_depth, bit) {
            (true, 8) => "depth",
            (true, _) => "stencil",
            (false, _) => "color",
        };
        let entry = format!("{from}_to_{to}");
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
                    entry_point: Some(&entry),
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

    /// EDRAM ownership transfer: draws target `src` into the first `rows`
    /// rows of `dst` (same tiles; k_8_8_8_8, kD24S8 or kD24FS8, each with
    /// its RB_SURFACE_INFO msaa_samples) as their shared 32-bit EDRAM
    /// word: colour `r | g << 8 | b << 16 | a << 24`, depth `depth24 << 8
    /// | stencil` (24-bit unorm for kD24S8, 20e4 for kD24FS8, converted as
    /// the depth clear and resolve do). A `dst` pixel takes the `src`
    /// pixel holding its first sample, the tile's halves swapped between
    /// colour and depth.
    fn transfer(
        &mut self,
        (src, src_msaa): (RenderTargetKey, u32),
        (dst, dst_msaa): (RenderTargetKey, u32),
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
        // Without a source stencil the depth pass's stencil 0 is the word's.
        let src_stencil = !sf.is_depth_stencil_format() || sf.has_stencil_aspect();
        let bits = if df.has_stencil_aspect() && src_stencil {
            0..8
        } else {
            0..0
        };
        let mut pipelines = vec![self.transfer_pipeline(sf, df, 8)?];
        for bit in bits {
            pipelines.push(self.transfer_pipeline(sf, df, bit)?);
        }
        // transfer.wgsl reads the samples per pixel from x0, y0 (`dst`) and
        // w, h (`src`), `src`'s pitch from pitch; resolve.wgsl's depth_word
        // reads `src_format`.
        let mut params = [0u32; 12];
        (params[0], params[1]) = samples(dst_msaa);
        (params[2], params[3]) = samples(src_msaa);
        params[4] = src.pitch;
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
        let views = if !sf.is_depth_stencil_format() {
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
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                });
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
        self.submitted(Site::Transfer);
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

    /// The guest memory the vertex fetch constants in `mask` point at, as
    /// one window, made ready to read (kept resolves under it written
    /// back). Returns its physical `[lo, hi)` (none for no window) and base
    /// address; or, when a window staged in this batch holds it, none, that
    /// window's base and its memo entry, with this window's size.
    #[allow(clippy::type_complexity)]
    fn guest_window(
        &mut self,
        mask: u128,
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<
        (
            Option<(u64, u64)>,
            u32,
            Option<((u64, u64, (u64, u64)), u64)>,
        ),
        Skip,
    > {
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
        if lo >= hi {
            return Ok((None, 0, None));
        }
        let mut memo = self.window_memo.iter().rev();
        if let Some(&w) = memo.find(|&&(a, b, _)| a <= lo && hi <= b) {
            return Ok((None, w.0 as u32, Some((w, hi - lo))));
        }
        if hi - lo > self.max_storage {
            return unsupported(format!(
                "vertex data window of {} bytes is larger than the storage buffer limit",
                hi - lo
            ));
        }
        self.sync(mem, lo, hi)?;
        Ok((Some((lo, hi)), lo as u32, None))
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
        consts(&self.regs, window_base)
    }

    /// Texture and sampler for fetch constant slot `n`. A slot without a
    /// texture fetch constant, or whose texture cannot be read or decoded,
    /// gets a 1x1 texture of zeros (as Xenia binds its null image) and a
    /// warning, so the draw still happens.
    fn texture(
        &mut self,
        n: u32,
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<(wgpu::TextureView, wgpu::Sampler), Skip> {
        let fetch = xenon_texture::TextureFetch::decode(self.regs.fetch_constant(n));
        if fetch.fetch_type != 2 {
            report.warn("texture slot without a texture fetch constant drawn with zeros");
            return Ok((self.zero_texture.clone(), self.zero_sampler.clone()));
        }
        let sampler = self.sampler(&fetch, report);
        let constant = self.regs.fetch_constant(n);
        if let Some((view, ..)) = self.texture_memo.get(&constant) {
            return Ok((view.clone(), sampler));
        }
        let addr = u64::from(fetch.base_address & PHYSICAL_MASK);
        if let Ok(len) = xenon_texture::bytes_needed(&fetch) {
            let end = addr + len as u64;
            if let Some(view) = self.kept_texture(&fetch, constant, addr, end, mem)? {
                self.timings.textures_kept += 1;
                self.timings.kept_bytes += len as u64;
                self.texture_memo
                    .insert(constant, (view.clone(), addr, end));
                return Ok((view, sampler));
            }
            self.sync(mem, addr, end)?;
        }
        let view = self
            .guest_texture(n, &fetch, &*mem, report)
            .unwrap_or_else(|why| {
                report.warn(format!("{why}; drawn with zeros"));
                self.zero_texture.clone()
            });
        Ok((view, sampler))
    }

    /// The texture of `fetch` decoded on the GPU from a kept resolve holding
    /// all of its guest bytes `[addr, end)`, as [`Self::guest_texture`]
    /// would decode them; `None` for the guest path. Only tiled 2D
    /// k_8_8_8_8 layouts that host as `Rgba8Unorm`, k_32_FLOAT and
    /// k_16_16_16_16_FLOAT (any endian, swizzle and resolve format, as the
    /// bytes are the guest's) are decoded here.
    fn kept_texture(
        &mut self,
        fetch: &xenon_texture::TextureFetch,
        constant: [u32; 6],
        addr: u64,
        end: u64,
        mem: &Tracked,
    ) -> Result<Option<wgpu::TextureView>, Skip> {
        use xenon_texture::{Dimension, TextureFormat as F};
        let Some(i) = self
            .mirrors
            .iter()
            .position(|m| m.lo <= addr && end <= m.hi)
        else {
            return Ok(None);
        };
        use wgpu::TextureFormat as H;
        let (w, h) = (fetch.width(), fetch.height());
        let pitch = xenon_texture::tile::align_to_tile(fetch.pitch_texels());
        let host = host_texture_format(fetch.format, fetch.sign, self.float32_filterable);
        // (decode_float mode or None for decode_8888, guest bytes and host
        // bytes per texel)
        let (mode, guest_bytes, host_bytes) = match (fetch.format.data_format(), host) {
            (F::K8_8_8_8, H::Rgba8Unorm) => (None, 4, 4),
            (F::K32Float, H::Rgba32Float) => (Some(0), 4, 16),
            (F::K32Float, H::Rgba16Float) => (Some(1), 4, 8),
            (F::K16_16_16_16Float, H::Rgba16Float) => (Some(2), 8, 8),
            _ => return Ok(None),
        };
        if !fetch.tiled
            || fetch.dimension != Dimension::D2OrStacked
            || fetch.stacked
            || !(addr - self.mirrors[i].lo).is_multiple_of(4)
            || pitch < w
            || w > self.max_dim
            || h > self.max_dim
            || !self.verify(i, mem, addr, end)
        {
            return Ok(None);
        }
        // Only the rows resolves wrote since the last decode are decoded
        // again. Nothing in the open batch samples the old texels: the
        // resolve flushed it, and dropped this texture from the memo.
        let (y0, y1) = match self.mirrors[i].views.get(&constant) {
            Some(k) => match k.stale {
                None => return Ok(Some(k.view.clone())),
                Some(rows) => rows,
            },
            None => (0, h),
        };
        let row_words = (w * host_bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) / 4;
        let rows = y1 - y0;
        let params = [
            w,
            rows,
            pitch,
            fetch.endian as u32,
            u32::from(fetch.swizzle),
            ((addr - self.mirrors[i].lo) / 4) as u32,
            row_words,
            y0,
            mode.unwrap_or(0),
            0,
            0,
            0,
        ];
        let uniform = self.buffer(
            "decode params",
            &to_bytes(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        let texels = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decoded texels"),
            size: u64::from(row_words) * 4 * u64::from(rows),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tex = match self.mirrors[i].views.get(&constant) {
            Some(k) => k.tex.clone(),
            None => self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("kept resolve texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: host,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }),
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let device = &self.device;
        let entry = ["decode_8888", "decode_float"][usize::from(mode.is_some())];
        let pipeline =
            self.decode_pipelines[usize::from(mode.is_some())].get_or_insert_with(|| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("resolve"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("resolve.wgsl").into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.mirrors[i].buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: texels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), rows.div_ceil(8), 1);
        }
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &texels,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_words * 4),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: w,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);
        self.submitted(Site::Decode);
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        let view = tex.create_view(&Default::default());
        self.mirrors[i].views.insert(
            constant,
            Kept {
                tex,
                view: view.clone(),
                lo: addr,
                hi: end,
                block: u64::from(pitch) * 32 * guest_bytes,
                stale: None,
            },
        );
        Ok(Some(view))
    }

    /// Host texture of fetch constant slot `n`, from the cache or decoded.
    /// Formats whose values do not fit 8-bit unorm, and fetches with a signed
    /// component, upload as floats; unsigned DXT1..5 as BC blocks where
    /// [`bc_blocks`] takes them. The cache key holds the whole fetch
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
        let constant = self.regs.fetch_constant(n);
        let key = (decode_key(constant), hash_bytes(&bytes));
        let range = (u64::from(addr), u64::from(addr) + bytes.len() as u64);
        if let Some((view, _, used)) = self.texture_cache.get_mut(&key) {
            *used = self.replays;
            let view = view.clone();
            self.texture_memo
                .insert(constant, (view.clone(), range.0, range.1));
            return Ok(view);
        }
        let signed = fetch.sign.contains(&1);
        let mut host = host_texture_format(fetch.format, fetch.sign, self.float32_filterable);
        let bc = if self.bc_textures && !signed {
            bc_blocks(fetch, &bytes).map_err(|e| e.to_string())?
        } else {
            None
        };
        let (width, height, data) = if let Some((format, blocks)) = bc {
            host = format;
            (fetch.width(), fetch.height(), blocks)
        } else if host == wgpu::TextureFormat::Rgba8Unorm {
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
        let bucket = match host {
            wgpu::TextureFormat::Rgba8Unorm => 0,
            _ if host.is_compressed() => 1,
            wgpu::TextureFormat::Rgba16Float => 2,
            wgpu::TextureFormat::Rgba32Float => 3,
            _ => 4,
        };
        let by = &mut report.timings.uploads_by_format[bucket];
        by.0 += 1;
        by.1 += data.len() as u64;
        // Rows of blocks: a BC row holds four rows of texels.
        let rows = height / host.block_dimensions().1;
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
                bytes_per_row: Some(data.len() as u32 / rows),
                rows_per_image: Some(rows),
            },
            size,
        );
        let view = tex.create_view(&Default::default());
        let len = data.len() as u64;
        let key_hash = hash_bytes(&to_bytes(&key.0)) ^ key.1;
        if self.texture_evicted.remove(&key_hash) {
            report.timings.textures_reuploaded += 1;
        }
        let limit = self.texture_cache_limit;
        if self.texture_cache_bytes + len > limit {
            self.evict_textures(limit.saturating_sub(len), report);
        }
        self.texture_cache_bytes += len;
        self.texture_cache
            .insert(key, (view.clone(), len, self.replays));
        self.texture_memo
            .insert(constant, (view.clone(), range.0, range.1));
        Ok(view)
    }

    /// Evicts least recently used textures until the cache holds at most
    /// 3/4 of `room` bytes, so evictions come in batches, not per upload.
    /// Views still bound in the open batch stay alive through their clones.
    fn evict_textures(&mut self, room: u64, report: &mut Report) {
        let goal = room / 4 * 3;
        let mut by_age: Vec<_> = self
            .texture_cache
            .iter()
            .map(|(k, &(_, len, used))| (used, len, *k))
            .collect();
        by_age.sort_unstable_by_key(|e| e.0);
        // shortcut: the evicted-key set only feeds a counter; cleared
        // when large, which undercounts re-uploads after that.
        if self.texture_evicted.len() > 1 << 16 {
            self.texture_evicted.clear();
        }
        for (_, len, k) in by_age {
            if self.texture_cache_bytes <= goal {
                break;
            }
            self.texture_cache.remove(&k);
            self.texture_cache_bytes -= len;
            self.texture_evicted
                .insert(hash_bytes(&to_bytes(&k.0)) ^ k.1);
            report.timings.textures_evicted += 1;
        }
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
    fn draw(&mut self, d: &Draw, mem: &mut Tracked, report: &mut Report) -> Result<bool, Skip> {
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

        let mut idx = {
            if let xenon_gpu::IndexSource::Dma {
                address, format, ..
            } = &d.source
            {
                let a = u64::from(*address & PHYSICAL_MASK);
                let len = u64::from(d.index_count) * format.bytes() as u64;
                self.sync(mem, a, a + len)?;
            }
            indices::guest_indices(&*mem, &d.source, d.index_count).map_err(Skip::Unsupported)?
        };
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
        let msaa = (surface >> 16) & 3;
        let mut color_tex: [Option<wgpu::Texture>; 4] = Default::default();
        for rt in 0..4 {
            color_tex[rt] = color[rt]
                .map(|k| self.target(k, rows, msaa, report))
                .transpose()?;
        }
        let depth_tex = depth_key
            .map(|k| self.target(k, rows, msaa, report))
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

        let key_stencil = stencil
            .as_ref()
            .and_then(|s| pipeline_stencil(s.0.clone(), cull.1));
        // The winding only matters to culling, two-sided stencil and the
        // facing flag of polygon parameter generation.
        let facing = cull.1.is_some()
            || key_stencil.as_ref().is_some_and(|s| s.front != s.back)
            || param_gen.is_some_and(|p| p.primitive == ParamGenPrimitive::Polygon);
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
                    let mask = state::write_mask(color_mask, rt as u32);
                    (k.format.wgpu(), pipeline_blend(blends[rt], mask), mask)
                })
            }),
            depth: depth_format.map(|f| (f, depth.write, depth.compare, key_stencil)),
            cull: (if facing { cull.0 } else { wgpu::FrontFace::Ccw }, cull.1),
            textures: tex_mask,
        };
        self.pipeline_uses += 1;
        let pipeline = match self.pipelines.get_mut(&key) {
            Some((p, used)) => {
                *used = self.pipeline_uses;
                p.clone()
            }
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
                        depth_stencil: k.depth.map(|(format, write, compare, st)| {
                            wgpu::DepthStencilState {
                                format,
                                depth_write_enabled: Some(write),
                                depth_compare: Some(compare),
                                stencil: st.unwrap_or_default(),
                                bias: Default::default(),
                            }
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
                self.pipelines.insert(key, (p.clone(), self.pipeline_uses));
                self.timings.pipelines_created += 1;
                self.drop_pipelines();
                p
            }
        };
        self.timings.shaders += lap(&mut t);

        let (mut read, window_base, staged) = self.guest_window(vf_mask, mem, report)?;
        let consts = self.consts(window_base);
        let epi = to_bytes(&state::epilogue(&self.regs, w, h));
        // Rectangle lists read their guest indices in the vertex shader and
        // draw 6 host vertices per rectangle without an index buffer.
        let idx_bytes = to_bytes(&idx);
        let rect_indices: &[u8] = if rect_list { &idx_bytes } else { &[] };
        let rest = [&epi[..], rect_indices, &idx_bytes].map(Part::Bytes);
        // The window and constants staged earlier in this batch are bound
        // again, unless the batch was submitted since (the arena is then
        // overwritten) or would be to stage the rest.
        let mut kept_window = staged.filter(|(w, _)| self.window_memo.contains(w));
        let mut kept_consts = (self.staged_consts.as_ref())
            .filter(|(b, _)| *b == consts)
            .map(|&(_, s)| s);
        fn parts<'a>(
            w: Option<Part<'a>>,
            c: Option<&'a [u8]>,
            rest: [Part<'a>; 3],
        ) -> Vec<Part<'a>> {
            w.into_iter()
                .chain(c.map(Part::Bytes))
                .chain(rest)
                .collect()
        }
        // The window's size is all `fits` needs of it.
        let window_size = Part::Guest(0, read.map_or(0, |(lo, hi)| (hi - lo) as usize));
        if (kept_window.is_some() || kept_consts.is_some())
            && !self.fits(&parts(
                kept_window.is_none().then_some(window_size),
                kept_consts.is_none().then_some(&consts),
                rest,
            ))
        {
            // Staging all of it submits the batch in `stage`; a kept window
            // is read again below.
            (kept_window, kept_consts) = (None, None);
        }
        if let (Some(((lo, hi, _), _)), None) = (staged, kept_window) {
            self.sync(mem, lo, hi)?;
            read = Some((lo, hi));
        }
        let window = match read {
            Some((lo, hi)) => {
                report.timings.guest_bytes += hi - lo;
                Part::Guest(lo as u32, (hi - lo) as usize)
            }
            None => Part::Bytes(&[]),
        };
        // A full arena submits the batch in `stage`: submit time, not guest data.
        let submit = self.timings.submit;
        let mut fresh = (self.stage(
            &*mem,
            &parts(
                kept_window.is_none().then_some(window),
                kept_consts.is_none().then_some(&consts),
                rest,
            ),
        )?)
        .into_iter();
        let window_slice = match kept_window {
            Some((w, len)) => {
                self.timings.windows_reused += 1;
                self.timings.reused_bytes += len;
                w.2
            }
            None => {
                let s = fresh.next().expect("staged window");
                if let Some((lo, hi)) = read {
                    self.window_memo.push((lo, hi, s));
                }
                s
            }
        };
        let consts_slice = match kept_consts {
            Some(s) => {
                self.timings.consts_reused += 1;
                s
            }
            None => {
                let s = fresh.next().expect("staged constants");
                self.staged_consts = Some((consts, s));
                s
            }
        };
        let slices: Vec<(u64, u64)> = [window_slice, consts_slice]
            .into_iter()
            .chain(fresh)
            .collect();
        let flushed = self.timings.submit - submit;
        self.timings.guest_data += lap(&mut t).saturating_sub(flushed);
        if self
            .texture_log
            .is_some_and(|h| h.is_none_or(|h| h == ps.hash))
        {
            let lod = vs_t.info.texture_explicit_lod | ps_t.info.texture_explicit_lod;
            let bias = vs_t.info.texture_lod_bias | ps_t.info.texture_lod_bias;
            for n in (0..32).filter(|n| tex_mask & (1 << n) != 0) {
                let c = self.regs.fetch_constant(n);
                let (lod, bias) = (lod & (1 << n) != 0, bias & (1 << n) != 0);
                report.warn(texture_log_line(ps.hash, n, c, lod, bias));
            }
        }
        let mut textures = Vec::new();
        for n in (0..32).filter(|n| tex_mask & (1 << n) != 0) {
            textures.push((n, self.texture(n, mem, report)?));
        }
        self.timings.textures += lap(&mut t);
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
            self.mark_written(k, rows, msaa);
        }
        if let Some(k) = depth_key.filter(|_| depth.write || stencil.is_some()) {
            self.mark_written(k, rows, msaa);
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

    /// Clears `rect` of a render target drawn with `msaa` samples. `color`
    /// is RGBA 0..1; for depth targets `depth` and `stencil` are used.
    fn clear(
        &mut self,
        (key, msaa): (RenderTargetKey, u32),
        rect: resolve::Rect,
        color: [f32; 4],
        depth: f32,
        stencil: u32,
        report: &mut Report,
    ) -> Result<(), Skip> {
        let tex = self.target(key, rect.y1, msaa, report)?;
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
        // Recorded into the batch, which the resolve copy before it was
        // submitted ahead of: the clear ends the batch's render pass, so it
        // lands after the draws before it and under those after it, and
        // anything else that reads the target flushes the batch first.
        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            pass: None,
            enc: device.create_command_encoder(&Default::default()),
        });
        batch.pass = None;
        {
            let mut pass = batch.enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        self.mark_written(key, rect.y1, msaa);
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
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<bool, Skip> {
        // The copy and clears see every draw before them.
        self.flush()?;
        let v = u64::from(r.vertex_fetch0.address & PHYSICAL_MASK);
        self.sync(mem, v, v + 24)?;
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
                (key, r.msaa_samples),
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
            self.clear(
                (key, r.msaa_samples),
                rect,
                [0.0; 4],
                depth,
                r.depth_clear & 0xFF,
                report,
            )?;
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
        mem: &mut Tracked,
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
        let tex = self.target(key, rect.y1, r.msaa_samples, report)?;
        let pitch = xenon_texture::tile::align_to_tile(r.dest_pitch);
        let rect = resolve::Rect {
            x1: rect.x1.min(pitch).min(tex.width()),
            ..rect
        };
        if rect.is_empty() {
            return Ok(());
        }
        // shortcut: 16_16_FLOAT copies, and 64bpp copies but
        // 16_16_16_16_FLOAT from 7e3 / 16_16_16_16_FLOAT sources, take the
        // CPU path; add them to resolve.wgsl if they show up in profiles.
        let gpu = if dest_bytes == 4 {
            r.is_depth || r.dest_format != formats::DEST_16_16_FLOAT
        } else {
            r.dest_format == formats::DEST_16_16_16_16_FLOAT
                && matches!(format, TargetFormat::Float7e3 | TargetFormat::Rgba16Float)
        };
        if gpu {
            self.resolve_gpu(r, rect, format, dest_bytes as u32, &tex, mem)?;
            if dest_bytes == 4 {
                self.resolved_endian
                    .insert(r.dest_base & PHYSICAL_MASK, r.dest_endian);
            }
            return Ok(());
        }
        // write_tiled_texels reads and writes the whole tiled extent.
        let base = u64::from(r.dest_base & PHYSICAL_MASK);
        if let Ok(extent) =
            xenon_texture::tile::tiled_extent_2d(rect.x1, rect.y1, pitch, dest_bytes as u32)
        {
            self.sync(mem, base, base + u64::from(extent))?;
        }
        let mut t = Instant::now();
        self.wait_idle()?;
        self.timings.gpu_wait += lap(&mut t);
        let px = self.read_texture(&tex, rect.x0, rect.y0, rect.width(), rect.height())?;
        self.submitted(Site::Resolve);
        self.timings.readback += lap(&mut t);
        self.timings.readback_bytes += px.len() as u64;
        let src_bytes = px.len() / (rect.width() * rect.height()) as usize;
        let mut out = vec![0; px.len() / src_bytes * dest_bytes];
        let half_dest = matches!(
            r.dest_format,
            formats::DEST_16_16_FLOAT | formats::DEST_16_16_16_16_FLOAT
        );
        let lut = if half_dest {
            self.half_luts
                .entry((format, r.dest_exp_bias))
                .or_insert_with(|| formats::half_lut(format, r.dest_exp_bias))
                .as_ref()
        } else {
            None
        };
        for (src, dst) in px
            .chunks_exact(src_bytes)
            .zip(out.chunks_exact_mut(dest_bytes))
        {
            match lut {
                Some(lut) => formats::resolve_half_pixel(lut, src, r.dest_swap, dst),
                None => {
                    let rgba = formats::edram_rgba(format, src);
                    formats::resolve_pixel(rgba, r.dest_exp_bias, r.dest_swap, r.dest_format, dst);
                }
            }
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

    /// 32bpp (or 16_16_16_16_FLOAT, `dest_bytes` 8) colour copy on the
    /// GPU: resolve.wgsl converts, tiles and
    /// swaps each pixel into the destination's guest bytes kept on the GPU
    /// (see [`Mirror`]). Same result as the CPU path in
    /// [`Self::resolve_copy`]. Nothing waits here: the bytes reach guest
    /// memory at the end of the replay, or before the drawer reads them.
    fn resolve_gpu(
        &mut self,
        r: &Resolve,
        rect: resolve::Rect,
        format: TargetFormat,
        dest_bytes: u32,
        tex: &wgpu::Texture,
        mem: &mut Tracked,
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
        if !base.is_multiple_of(4) {
            return unsupported(format!("resolve destination 0x{base:08X} not word aligned"));
        }
        let pitch = xenon_texture::tile::align_to_tile(r.dest_pitch);
        let extent = xenon_texture::tile::tiled_extent_2d(rect.x1, rect.y1, pitch, dest_bytes)
            .map_err(|e| Skip::Unsupported(format!("resolve destination: {e}")))?;
        if u64::from(extent) > self.max_storage {
            return unsupported(format!("resolve destination of {extent} bytes"));
        }
        let (lo, hi) = (u64::from(base), u64::from(base) + u64::from(extent));
        let Some(i) = self.mirror(mem, lo, hi)? else {
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
                formats::DEST_16_16_16_16_FLOAT => 4,
                _ => 2,
            },
            u32::from(r.dest_swap),
            r.dest_endian,
            2f32.powi(r.dest_exp_bias).to_bits(),
            ((lo - self.mirrors[i].lo) / 4) as u32,
            0,
        ];
        let uniform = self.buffer(
            "resolve params",
            &to_bytes(&params),
            wgpu::BufferUsages::UNIFORM,
        );
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
                resource: self.mirrors[i].buf.as_entire_binding(),
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
        self.timings.resolve_cpu += lap(&mut t);
        self.queue.submit([enc.finish()]);
        self.submitted(Site::Resolve);
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        // A software driver finishes the rendering before it here.
        self.timings.gpu_wait += lap(&mut t);
        let m = &mut self.mirrors[i];
        let (mut a, mut b) = (lo, hi);
        m.dirty.retain(|&(x, y)| {
            let apart = y < a || b < x;
            if !apart {
                (a, b) = (a.min(x), b.max(y));
            }
            apart
        });
        m.dirty.push((a, b));
        m.dirty.sort_unstable();
        for k in m.views.values_mut().filter(|k| lo < k.hi && k.lo < hi) {
            // Tiled rows 32n..32n+32 are bytes block*n..block*(n+1).
            let y0 = (lo.saturating_sub(k.lo) / k.block * 32) as u32;
            let y1 = ((hi - k.lo).div_ceil(k.block) * 32).min(u64::from(k.tex.height())) as u32;
            k.stale = Some(k.stale.map_or((y0, y1), |(a, b)| (a.min(y0), b.max(y1))));
        }
        self.forget_guest(lo, hi);
        self.timings.resolve_cpu += lap(&mut t);
        Ok(())
    }

    /// Reads the front buffer: a 32x32-tiled 8_8_8_8 surface, pitch = width
    /// rounded up to 32, R in the low byte. Its byte order is the destination
    /// endian of the last resolve to that address (a game may resolve with
    /// any of them), else k8in32.
    fn front_buffer(
        &mut self,
        addr: u32,
        width: u32,
        height: u32,
        mem: &mut Tracked,
    ) -> Result<Result<Frame, String>, Error> {
        use xenon_texture::tile;
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Ok(Err(format!("front buffer size {width}x{height}")));
        }
        let pitch = tile::align_to_tile(width);
        let extent = match tile::tiled_extent_2d(width, height, pitch, 4) {
            Ok(e) => e,
            Err(e) => return Ok(Err(e.to_string())),
        };
        let addr = addr & PHYSICAL_MASK;
        self.sync(mem, u64::from(addr), u64::from(addr) + u64::from(extent))?;
        Ok(self.detile_front_buffer(addr, width, height, pitch, extent, mem))
    }

    fn detile_front_buffer(
        &self,
        addr: u32,
        width: u32,
        height: u32,
        pitch: u32,
        extent: u32,
        mem: &Tracked,
    ) -> Result<Frame, String> {
        use xenon_texture::tile;
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

/// The base level of an unsigned DXT1, DXT2/3 or DXT4/5 `fetch` as linear
/// BC1, BC2 or BC3 blocks (endian swapped and detiled, as the CPU decoder
/// reads them), which the host decodes to the CPU decoder's texels (up to
/// its rounding of interpolated ones); `None` for the CPU path. Only the
/// identity swizzle passes (a wgpu view cannot swizzle), and only whole
/// blocks: a padded host texture would move normalised coordinates and
/// clamping. DXN stays on the CPU path, which repeats Y into Z and W where
/// BC5 gives 0 and 1.
fn bc_blocks(
    fetch: &xenon_texture::TextureFetch,
    bytes: &[u8],
) -> Result<Option<(wgpu::TextureFormat, Vec<u8>)>, xenon_texture::Error> {
    use xenon_texture::{tile, TextureFormat as F};
    let (format, bpb) = match fetch.format.data_format() {
        F::Dxt1 => (wgpu::TextureFormat::Bc1RgbaUnorm, 8),
        F::Dxt2_3 => (wgpu::TextureFormat::Bc2RgbaUnorm, 16),
        F::Dxt4_5 => (wgpu::TextureFormat::Bc3RgbaUnorm, 16),
        _ => return Ok(None),
    };
    const IDENTITY: u16 = (1 << 3) | (2 << 6) | (3 << 9);
    let (w, h) = (fetch.width(), fetch.height());
    if fetch.swizzle & 0xFFF != IDENTITY || w % 4 != 0 || h % 4 != 0 {
        return Ok(None);
    }
    let (wb, hb) = (w / 4, h / 4);
    let pitch = tile::align_to_tile(fetch.pitch_texels().div_ceil(4));
    let extent = if fetch.tiled {
        tile::tiled_extent_2d(wb, hb, pitch, bpb)? as usize
    } else {
        ((pitch * (hb - 1) + wb) * bpb) as usize
    };
    let Some(raw) = bytes.get(..extent) else {
        return Err(xenon_texture::Error::TooShort {
            needed: extent,
            got: bytes.len(),
        });
    };
    let mut raw = raw.to_vec();
    xenon_texture::endian::swap_in_place(&mut raw, fetch.endian);
    let blocks = if fetch.tiled {
        tile::detile_2d(&raw, wb, hb, pitch, bpb)?
    } else {
        // Index rows rather than chunk by pitch: a single linear row may have
        // a pitch below its width, even 0.
        let (row, pitch) = ((wb * bpb) as usize, (pitch * bpb) as usize);
        (0..hb as usize)
            .flat_map(|y| &raw[y * pitch..][..row])
            .copied()
            .collect()
    };
    Ok(Some((format, blocks)))
}

/// The [`Drawer::texture_log`] line of fetch constant `c` in slot `n` of a
/// draw with pixel shader `ps`: the fields that change sampled values
/// (format, size, swizzle as source per output channel, X Y Z W signs
/// 0 unsigned / 1 signed / 2 biased / 3 gamma, exp_adjust, endian), the mip
/// fields (the drawer uploads the base level only), the fetch constant's
/// LOD bias (1/32 steps), whether a fetch of the slot adds its own non-zero
/// LOD bias (`instr_bias 1`), whether one picks its own LOD (`explicit`)
/// and the raw dwords.
pub fn texture_log_line(
    ps: u64,
    n: u32,
    c: [u32; 6],
    explicit_lod: bool,
    instr_bias: bool,
) -> String {
    let f = xenon_texture::TextureFetch::decode(c);
    let swizzle: String = (0..4)
        .map(|i| b"xyzw0101"[usize::from((f.swizzle >> (3 * i)) & 7)] as char)
        .collect();
    let sign: String = f.sign.iter().map(|s| char::from(b'0' + s)).collect();
    format!(
        "texture log: ps {ps:016x} t{n} type {} {:?} {}x{} at {:08x} swizzle {swizzle} sign {sign} exp_adjust {} endian {:?} num_format {} mips {}-{}{} at {:08x} lod_bias {} instr_bias {} {} lod | {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
        f.fetch_type,
        f.format,
        f.width(),
        f.height(),
        f.base_address,
        f.exp_adjust,
        f.endian,
        f.num_format,
        f.mip_min_level,
        f.mip_max_level,
        if f.packed_mips { " packed" } else { "" },
        f.mip_address,
        f.lod_bias,
        u8::from(instr_bias),
        if explicit_lod { "explicit" } else { "computed" },
        c[0],
        c[1],
        c[2],
        c[3],
        c[4],
        c[5],
    )
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
/// A fetch sign of 1 (signed) on any component moves the 8-bit formats to
/// a float format too: the decoder writes signed values in -1..1, which the
/// host then filters directly. With every component signed that is
/// `Rgba16Float`: n/127 (and the other fields of 11 bits or less) round by
/// at most 2^-12 relative, far below a step. Otherwise it is the wide
/// format, so the texture's unsigned components keep their 8-bit values
/// exactly (a half would not, and the biased and gamma conversions in the
/// shader would carry the rounding to the result).
fn host_texture_format(
    format: xenon_texture::TextureFormat,
    sign: [u8; 4],
    float32: bool,
) -> wgpu::TextureFormat {
    use xenon_texture::TextureFormat as F;
    let signed = sign.contains(&1);
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
        _ if sign == [1; 4] => wgpu::TextureFormat::Rgba16Float,
        _ if signed => wide,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants block holds each register of its four blocks in
    /// order, then the window base and the index offset.
    #[test]
    fn consts_block_is_the_registers_in_order() {
        let mut r = regs::RegisterFile::default();
        for i in 0x3F00..0x4A00 {
            r.set(i, i.wrapping_mul(0x9E37_79B9));
        }
        r.set(regs::VGT_INDX_OFFSET, 77);
        let r = &r;
        let block = |base: u32, n: u32| (0..n).map(move |i| r.get(base + i));
        let mut want: Vec<u32> = block(regs::SHADER_CONSTANT_FLOAT, 2048)
            .chain(block(regs::SHADER_CONSTANT_FETCH, 192))
            .chain(block(regs::SHADER_CONSTANT_BOOL, 8))
            .chain(block(regs::SHADER_CONSTANT_LOOP, 32))
            .collect();
        want.extend([0x1234_5678, 77, 0, 0]);
        assert_eq!(consts(r, 0x1234_5678), to_bytes(&want));
    }

    /// A drawer on the test adapter, held under one lock so the unit tests
    /// never have two devices at once (lavapipe crashes when they do, as
    /// tests/common does for the integration tests), or None without an
    /// adapter (unless XENON_REQUIRE_GPU=1).
    fn gpu() -> Option<TestDrawer> {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
        match Drawer::new() {
            Ok(drawer) => Some(TestDrawer {
                drawer,
                _lock: lock,
            }),
            Err(Error::NoAdapter) if std::env::var("XENON_REQUIRE_GPU").as_deref() != Ok("1") => {
                None
            }
            Err(e) => panic!("{e}"),
        }
    }

    struct TestDrawer {
        drawer: Drawer,
        // Declared after `drawer`, so the device is gone before the next test.
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl std::ops::Deref for TestDrawer {
        type Target = Drawer;
        fn deref(&self) -> &Drawer {
            &self.drawer
        }
    }

    impl std::ops::DerefMut for TestDrawer {
        fn deref_mut(&mut self) -> &mut Drawer {
            &mut self.drawer
        }
    }

    /// transfer.wgsl's `word_depth` makes exactly the depth the depth clear
    /// makes: n / 0xFFFFFF rounded to the nearest f32 for kD24S8 (a host
    /// that keeps Depth24PlusStencil8 as 32-bit float stores it as is;
    /// lavapipe's 24-bit store would hide a mismatch), float20e4 for kD24FS8.
    #[test]
    fn transfer_word_depth_matches_the_clear() {
        let Some(d) = gpu() else { return };
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

    /// Staging that fills the arena in uneven steps keeps its host copy
    /// within the arena's size (doubling from 3 MB would reach 24 MB).
    #[test]
    fn staged_capacity_stays_within_the_arena() {
        let Some(mut d) = gpu() else { return };
        let mem = SliceMemory::default();
        let part = vec![7u8; 3 << 20];
        let part = [Part::Bytes(&part)];
        for _ in 0..12 {
            d.stage(&mem, &part).unwrap();
            let arena = d.arena.as_ref().unwrap().size();
            assert!(
                d.staged.capacity() as u64 <= arena,
                "{}",
                d.staged.capacity()
            );
        }
        let big = vec![1u8; 20 << 20];
        d.stage(&mem, &[Part::Bytes(&big)]).unwrap();
        d.stage(&mem, &part).unwrap();
        assert_eq!(d.arena.as_ref().unwrap().size(), 32 << 20);
        assert!(d.staged.capacity() <= 32 << 20, "{}", d.staged.capacity());
        d.flush().unwrap();
    }

    /// Past the cap the least recently written targets go first, and
    /// dropping one that still held its tiles' newest pixels is reported;
    /// it comes back as a new, cleared target.
    #[test]
    fn target_cap_drops_the_least_recently_written() {
        let Some(mut d) = gpu() else { return };
        let key = |base_tiles| RenderTargetKey {
            base_tiles,
            pitch: 64,
            format: TargetFormat::Rgba8,
        };
        let mut report = Report::default();
        for base in [0, 100, 200] {
            d.target(key(base), 64, 0, &mut report).unwrap();
            d.mark_written(key(base), 64, 0);
        }
        // 64 x 2048 x 4 bytes each.
        d.target_cap = 2 << 20;
        d.drop_targets(&mut report);
        assert!(report.warnings.is_empty(), "{report:?}");
        d.target_cap = 1 << 20;
        d.drop_targets(&mut report);
        let mut left = d.render_targets();
        left.sort_by_key(|k| k.base_tiles);
        assert_eq!(left, [key(100), key(200)]);
        assert!(!d.written.contains_key(&key(0)));
        assert!(d.memory_line().contains("targets 1 MB in 2, 1 dropped,"));
        assert_eq!(report.warnings.len(), 1, "{report:?}");
    }

    #[test]
    fn drop_keeps_a_target_whose_newest_alias_does_not_transfer() {
        let Some(mut d) = gpu() else { return };
        let key = |format| RenderTargetKey {
            base_tiles: 0,
            pitch: 64,
            format,
        };
        let mut report = Report::default();
        // (format, msaa) in write order; the last one transferable but at
        // another row width does not transfer either.
        for (formats, kept) in [
            (
                &[
                    (TargetFormat::D24S8, 0),
                    (TargetFormat::Rgba8, 0),
                    (TargetFormat::Rgba16Float, 0),
                ][..],
                true,
            ),
            (
                &[
                    (TargetFormat::D24S8, 0),
                    (TargetFormat::Rgba8, 0),
                    (TargetFormat::D24FS8, 2),
                ][..],
                true,
            ),
            (
                &[(TargetFormat::D24S8, 0), (TargetFormat::Rgba8, 0)][..],
                false,
            ),
        ] {
            d.targets.clear();
            d.written.clear();
            for &(f, msaa) in formats {
                d.target(key(f), 64, msaa, &mut report).unwrap();
                d.mark_written(key(f), 64, msaa);
            }
            d.drop_targets(&mut report);
            let depth = key(TargetFormat::D24S8);
            assert_eq!(d.render_targets().contains(&depth), kept, "{formats:?}");
            assert_eq!(d.written.contains_key(&depth), kept, "{formats:?}");
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
        let (unsigned, signed, mixed) = ([0; 4], [1; 4], [1, 0, 3, 1]);
        assert_eq!(
            host_texture_format(F::K8_8_8_8, unsigned, true),
            H::Rgba8Unorm
        );
        assert_eq!(
            host_texture_format(F::K8_8_8_8, signed, true),
            H::Rgba16Float
        );
        assert_eq!(
            host_texture_format(F::K8_8_8_8, mixed, true),
            H::Rgba32Float
        );
        assert_eq!(host_texture_format(F::Dxt1, signed, true), H::Rgba16Float);
        assert_eq!(host_texture_format(F::Dxt1, mixed, false), H::Rgba16Float);
        assert_eq!(host_texture_format(F::Dxt1, unsigned, true), H::Rgba8Unorm);
        assert_eq!(
            host_texture_format(F::K16Expand, unsigned, true),
            H::Rgba16Float
        );
        assert_eq!(
            host_texture_format(F::K24_8Float, unsigned, true),
            H::Rgba32Float
        );
        assert_eq!(
            host_texture_format(F::K24_8Float, unsigned, false),
            H::Rgba16Float
        );
        assert_eq!(
            host_texture_format(F::K16_16, unsigned, true),
            H::Rgba32Float
        );
        assert_eq!(host_texture_format(F::K16_16, signed, true), H::Rgba32Float);
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
