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
//! EDRAM holds 2048 rows of 1280 32-bit pixels). Its texture holds
//! [`Drawer::scale`] host pixels per guest pixel on each axis: draws,
//! clears and transfers address host pixels, resolves and readbacks
//! decimate them back to guest pixels ([`formats::decimate`]).
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
//! stencil (the 20e4 float is stored at f32 precision). A pixel shader's
//! oDepth export is the draw's depth (`frag_depth`); a draw with no depth
//! target drops it.
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
use std::collections::hash_map::Entry;
use std::collections::BTreeMap;

/// The drawer's maps are looked up several times per draw with keys it
/// makes itself; Fx hashing costs a fraction of std's SipHash there.
type HashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
type HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use xenon_gpu::regs;
use xenon_gpu::{
    Command, CopyCommand, Draw, FetchConstantType, GuestRead, RegisterFile, Resolve, ShaderStage,
    SliceMemory, PHYSICAL_MASK,
};
use xenon_shader::{
    ParamGen, ParamGenPrimitive, ShaderInfo, Translated, VertexFetchReach, DEPTH_MEMBER,
    DEPTH_STORE,
};
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

    /// The physical `[start, end)` ranges the CPU (or anything else) wrote
    /// since the last call, merged, with the record reset; `None` (the
    /// default) when writes are not recorded and any byte may have changed.
    /// Lets textures looked up in one replay be used in the next without
    /// reading their guest bytes again.
    fn take_written(&mut self) -> Option<Vec<(u64, u64)>> {
        None
    }
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

/// A front buffer copy on its way back: the buffer, its submission, and
/// how to read it (width, height, pitch, endian, swizzle).
struct LateFront {
    /// The frame's RGBA8 rows, decoded on the GPU (byte order, tiling and
    /// fetch constant 0's swizzle, as [`detile_front`] does).
    buf: wgpu::Buffer,
    /// The submitter's batch number of the batch that decodes into `buf`.
    sent: u64,
    size: (u32, u32),
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
    /// Resolve bind groups writing into the buffer, by source texture and
    /// `resolve_pipelines` entry, the parameters in the arena at a dynamic
    /// offset. They hold the buffer, so they go when it does; also dropped
    /// when the arena is replaced, and at the end of a replay for textures
    /// no longer kept.
    groups: HashMap<(wgpu::Texture, usize), wgpu::BindGroup>,
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
    /// `shaders` split by step: shader lookup and state decode, guest
    /// indices, render target lookup, then modules and pipeline lookup.
    /// Sampled: one draw in 32 is timed and counts for 32, so the parts
    /// only approximate `shaders`; sampled draws that made a pipeline,
    /// target, transfer or shader are left out (see `pipeline`).
    pub setup_parts: [Duration; 4],
    /// Guest vertex window, constants and index buffers: reading guest
    /// memory and uploading it, per draw.
    pub guest_data: Duration,
    /// Texture decode (detile and convert) and upload, per draw.
    pub textures: Duration,
    /// Encoding and submitting draws (the GPU runs them asynchronously).
    pub submit: Duration,
    /// Of `submit`: a flush, which writes the staged bytes to the arena,
    /// finishes the encoders and calls `queue.submit`; the rest is
    /// recording draws.
    pub queue: Duration,
    /// Of `queue`: copying the staged draw buffers into upload memory.
    pub upload: Duration,
    /// Of `submit`, the draws' own recording by step: their bind groups
    /// (looked up or made), beginning their render passes, and the `set_*`
    /// and draw calls. Sampled like `setup_parts`.
    pub submit_parts: [Duration; 3],
    /// Commands other than draws, resolves and swaps (register writes,
    /// shader loads, MEM_WRITEs), and the upkeep after every command;
    /// timed by run, between the timed commands, not one by one.
    pub commands: Duration,
    /// The start and end of a replay outside the other phases: memo upkeep
    /// for the guest writes listed, and render target drops.
    pub edges: Duration,
    /// Shader loads replayed.
    pub shader_loads: u32,
    /// Submitting batches on the submit thread, alongside the drawer's
    /// own phases (not part of them).
    pub submit_thread: Duration,
    /// Of the submit thread's work, finishing batches' encoders (not
    /// counted in `submit_thread`).
    pub finish_thread: Duration,
    /// Kept resolve write-backs to guest memory: each waits for the GPU.
    pub writebacks: u32,
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
    /// Of all the phases, the time spent in `create_render_pipeline` and
    /// `create_compute_pipeline` (where a driver compiles to machine
    /// code), including the pipelines' share of `shaders`.
    pub pipeline: Duration,
    /// Counts: pipelines and shader modules created, textures uploaded.
    pub pipelines_created: u32,
    pub textures_uploaded: u32,
    /// Submissions made during the replay. Batch flushes count one each,
    /// though the submit thread may merge queued ones into one
    /// `queue.submit`.
    pub submits: u32,
    /// `submits` by where they come from: resolve copies (GPU copies and
    /// CPU resolve readbacks), EDRAM transfers, kept resolve texture
    /// decodes with no draw to batch them with, and batch flushes and
    /// resolve write-backs. Resolve clears and texture decodes are
    /// recorded into the next batch.
    pub submits_by_site: [u32; 4],
    pub texture_bytes: u64,
    /// Guest vertex window bytes read (not those of reused windows).
    pub guest_bytes: u64,
    /// EDRAM transfers and their render passes.
    pub transfers: u32,
    pub transfer_passes: u32,
    /// Transfer pipelines, bind groups and views made, not kept from an
    /// earlier transfer between the same targets.
    pub transfer_setups: u32,
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
    /// Draws that bound indices staged earlier (by another draw of the same
    /// index buffer, in any replay since the arena last wrapped).
    pub indices_reused: u32,
    /// Draws that bound a texture bind group made earlier in their replay.
    pub texture_groups_reused: u32,
    /// Draws that bound a texture bind group found by their fetch
    /// constants, without looking each texture up.
    pub texture_sets_reused: u32,
    /// Render passes begun for draws: each target change or flush ends one.
    pub render_passes: u32,
    /// GPU time measured by timestamps around the batch's draw passes,
    /// clear passes and resolve copies, with
    /// [`Drawer::enable_gpu_timestamps`] on (zero otherwise); and the
    /// passes measured.
    pub gpu_passes: [Duration; 3],
    pub passes_timed: u32,
}

impl std::ops::AddAssign<&Timings> for Timings {
    /// Sums the phases and counts of several replays.
    fn add_assign(&mut self, t: &Timings) {
        self.shaders += t.shaders;
        for (a, b) in self.setup_parts.iter_mut().zip(t.setup_parts) {
            *a += b;
        }
        self.guest_data += t.guest_data;
        self.textures += t.textures;
        self.submit += t.submit;
        self.gpu_wait += t.gpu_wait;
        self.readback += t.readback;
        self.resolve_cpu += t.resolve_cpu;
        self.clears += t.clears;
        self.swap += t.swap;
        self.transfer += t.transfer;
        self.pipeline += t.pipeline;
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
        self.transfer_setups += t.transfer_setups;
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
        self.indices_reused += t.indices_reused;
        self.texture_groups_reused += t.texture_groups_reused;
        self.texture_sets_reused += t.texture_sets_reused;
        self.queue += t.queue;
        self.upload += t.upload;
        for (a, b) in self.submit_parts.iter_mut().zip(t.submit_parts) {
            *a += b;
        }
        self.commands += t.commands;
        self.edges += t.edges;
        self.shader_loads += t.shader_loads;
        self.submit_thread += t.submit_thread;
        self.finish_thread += t.finish_thread;
        self.writebacks += t.writebacks;
        self.render_passes += t.render_passes;
        for (a, b) in self.gpu_passes.iter_mut().zip(t.gpu_passes) {
            *a += b;
        }
        self.passes_timed += t.passes_timed;
    }
}

/// Copies queued out of kept resolves: the submission, and per kept
/// resolve its start, dirty ranges, buffer spans and readback buffer.
type WriteBack = (
    wgpu::SubmissionIndex,
    Vec<(u64, Vec<(u64, u64)>, Vec<(u64, u64)>, wgpu::Buffer)>,
);

/// Finishes and submits flushed batches on a thread of its own, one at a
/// time in order, so the drawer records the next batch meanwhile: wgpu
/// encodes the recorded passes for the driver at `finish`, and the driver
/// takes them at `submit`, which together cost tens of microseconds a draw
/// on some drivers. Both run on the one thread: with the finish on a
/// thread of its own, a batch ahead, NVIDIA's `vkQueueSubmit` sometimes
/// waited 20-230 ms on a driver lock while the next batch was encoded,
/// with the GPU idle.
struct Submitter {
    tx: Option<std::sync::mpsc::Sender<SubmitJob>>,
    rx: Mutex<std::sync::mpsc::Receiver<SubmitDone>>,
    /// Batches sent and not yet received back, and what came back: time
    /// spent, the first validation error and the last submission.
    state: Mutex<SubmitState>,
    /// The submit thread.
    threads: Vec<std::thread::JoinHandle<()>>,
}

/// Finish time, submit time, the first validation error and the
/// submission of one batch, and its staged bytes' buffer to use again.
type SubmitDone = (
    Duration,
    Duration,
    Option<String>,
    wgpu::SubmissionIndex,
    Option<Vec<u8>>,
);

/// Staged-byte buffers kept to use again.
const SPARE_STAGED: usize = 8;

/// Staged bytes to write to `arena` at an offset before the encoders run,
/// and the encoders.
type SubmitJob = (
    Option<(wgpu::Buffer, u64, Vec<u8>)>,
    Vec<wgpu::CommandEncoder>,
);

#[derive(Default)]
struct SubmitState {
    pending: usize,
    /// Batches sent since the start.
    sent: u64,
    queue: Duration,
    finish: Duration,
    error: Option<String>,
    last: Option<wgpu::SubmissionIndex>,
    /// Buffers of staged bytes the submit thread has written out.
    spare: Vec<Vec<u8>>,
}

impl Submitter {
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Submitter {
        let (tx, jobs) = std::sync::mpsc::channel::<SubmitJob>();
        let (done, rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("xenon-draw submit".into())
            .spawn(move || {
                // Mapped upload chunks, kept and reused: recalled after each
                // submit, so mapped again by the next upload.
                let mut belt = wgpu::util::StagingBelt::new(device.clone(), BELT_CHUNK);
                while let Ok(first) = jobs.recv() {
                    // Batches flushed meanwhile go in the same submit: each
                    // `queue.submit` has a fixed cost, and a run of resolves
                    // flushes many small batches close together.
                    let mut batch = vec![first];
                    batch.extend(jobs.try_iter().take(SUBMIT_MERGE_MAX - 1));
                    // Error scopes are per thread.
                    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                    let mut finished = Vec::with_capacity(batch.len());
                    for (upload, encs) in batch {
                        let t = Instant::now();
                        let bufs: Vec<wgpu::CommandBuffer> =
                            encs.into_iter().map(|e| e.finish()).collect();
                        finished.push((upload, bufs, t.elapsed()));
                    }
                    let finish_error = block_on(scope.pop()).map(|e| e.to_string());
                    let t = Instant::now();
                    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                    // Maps the chunks recalled after earlier submits whose
                    // work the GPU finished.
                    let _ = device.poll(wgpu::PollType::Poll);
                    let mut cmds = Vec::new();
                    let mut dones = Vec::with_capacity(finished.len());
                    for (upload, bufs, finish) in finished {
                        let mut used = None;
                        if let Some((arena, base, bytes)) = upload {
                            let mut enc = device.create_command_encoder(&Default::default());
                            // In chunk-sized pieces: the belt makes a chunk
                            // as large as a write and never frees one, so
                            // batches of growing sizes kept adding larger
                            // chunks.
                            for (i, piece) in bytes.chunks(BELT_CHUNK as usize).enumerate() {
                                let size =
                                    wgpu::BufferSize::new(piece.len() as u64).expect("non-empty");
                                let at = base + i as u64 * BELT_CHUNK;
                                belt.write_buffer(&mut enc, &arena, at, size)
                                    .copy_from_slice(piece);
                            }
                            // Each upload ahead of its own batch, after the
                            // batches before it read the arena.
                            cmds.push(enc.finish());
                            used = Some(bytes);
                        }
                        cmds.extend(bufs);
                        dones.push((finish, used));
                    }
                    belt.finish();
                    let index = queue.submit(cmds);
                    belt.recall();
                    let error = finish_error.or(block_on(scope.pop()).map(|e| e.to_string()));
                    // The submit time goes to the first batch of the merge.
                    let mut busy = t.elapsed();
                    for (i, (finish, used)) in dones.into_iter().enumerate() {
                        let sent = done.send((
                            finish,
                            std::mem::take(&mut busy),
                            if i == 0 { error.clone() } else { None },
                            index.clone(),
                            used,
                        ));
                        if sent.is_err() {
                            return;
                        }
                    }
                }
            })
            .expect("spawn submit thread");
        Submitter {
            tx: Some(tx),
            rx: Mutex::new(rx),
            state: Mutex::new(SubmitState::default()),
            threads: vec![thread],
        }
    }

    fn send(&self, upload: Option<(wgpu::Buffer, u64, Vec<u8>)>, encs: Vec<wgpu::CommandEncoder>) {
        let mut st = self.state.lock().expect("submit state");
        st.pending += 1;
        st.sent += 1;
        drop(st);
        self.tx
            .as_ref()
            .expect("submit thread")
            .send((upload, encs))
            .expect("submit thread alive");
    }

    /// Waits until every batch sent is submitted.
    fn drain(&self) {
        self.drain_to(0);
    }

    /// Waits until at most `n` batches sent are not yet submitted.
    fn drain_to(&self, n: usize) {
        let mut st = self.state.lock().expect("submit state");
        let rx = self.rx.lock().expect("submit results");
        loop {
            // Past `n`, only what is done already: its buffers come back.
            let got = match st.pending > n {
                true => rx.recv().expect("submit thread alive"),
                false => match rx.try_recv() {
                    Ok(got) => got,
                    Err(_) => break,
                },
            };
            let (f, e, error, index, used) = got;
            st.pending -= 1;
            st.finish += f;
            st.queue += e;
            if st.error.is_none() {
                st.error = error;
            }
            st.last = Some(index);
            if let Some(mut used) = used.filter(|_| st.spare.len() < SPARE_STAGED) {
                used.clear();
                st.spare.push(used);
            }
        }
    }

    /// An empty buffer for staged bytes holding at least `capacity`: one
    /// the submit thread is done with (its pages are mapped already), or
    /// a new one.
    fn spare(&self, capacity: usize) -> Vec<u8> {
        let mut st = self.state.lock().expect("submit state");
        match st.spare.iter().position(|v| v.capacity() >= capacity) {
            Some(i) => st.spare.swap_remove(i),
            None => {
                st.spare.pop();
                Vec::with_capacity(capacity)
            }
        }
    }

    /// Waits until batch number `sent` is submitted, and returns that
    /// submission or a later one.
    fn wait_for(&self, sent: u64) -> wgpu::SubmissionIndex {
        let pending = {
            let st = self.state.lock().expect("submit state");
            st.sent.saturating_sub(sent)
        };
        self.drain_to(pending as usize);
        let st = self.state.lock().expect("submit state");
        st.last.clone().expect("a batch was submitted")
    }

    /// The number of batches sent so far.
    fn sent(&self) -> u64 {
        self.state.lock().expect("submit state").sent
    }

    /// The finish and submit time and first error since the last call.
    fn take(&self) -> (Duration, Duration, Option<String>) {
        let mut st = self.state.lock().expect("submit state");
        let finish = std::mem::take(&mut st.finish);
        (finish, std::mem::take(&mut st.queue), st.error.take())
    }
}

impl Drop for Submitter {
    fn drop(&mut self) {
        drop(self.tx.take());
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Flushed batches the submit thread may hold before a flush waits, by
/// default; `XENON_SUBMITS_IN_FLIGHT` sets it.
const SUBMITS_IN_FLIGHT: usize = 4;

/// Most finished batches the submit thread merges into one submit.
const SUBMIT_MERGE_MAX: usize = 16;

/// Batch passes timed per flush; later ones in the same batch go untimed.
const GPU_CLOCK_PASSES: u64 = 1024;

/// Timestamp queries around the batch's passes, read back at each flush.
struct GpuClock {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// Kind of each timed pass in the batch: 0 draw, 1 clear, 2 resolve.
    kinds: Vec<u8>,
    /// Nanoseconds per tick.
    period: f32,
}

impl GpuClock {
    /// The query set and the begin and end indices for the next `kind`
    /// pass, or `None` when the batch has timed as many as fit.
    fn next(clock: &mut Option<GpuClock>, kind: u8) -> Option<(&wgpu::QuerySet, u32)> {
        let c = clock.as_mut()?;
        if c.kinds.len() as u64 >= GPU_CLOCK_PASSES {
            return None;
        }
        c.kinds.push(kind);
        Some((&c.set, 2 * (c.kinds.len() as u32 - 1)))
    }

    fn render(
        clock: &mut Option<GpuClock>,
        kind: u8,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Self::next(clock, kind).map(|(query_set, i)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }

    fn compute(
        clock: &mut Option<GpuClock>,
        kind: u8,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        Self::next(clock, kind).map(|(query_set, i)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(i),
            end_of_pass_write_index: Some(i + 1),
        })
    }
}

impl Report {
    fn unsupported(&mut self, why: String) {
        if !self.unsupported.contains(&why) {
            self.unsupported.push(why);
        }
    }

    /// Allocates only for a warning not reported yet: some repeat per draw.
    fn warn(&mut self, why: impl AsRef<str> + Into<String>) {
        if !self.warnings.iter().any(|w| w == why.as_ref()) {
            self.warnings.push(why.into());
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
    /// Shared, so a draw takes the loaded programs without copying them.
    words: Arc<[u32]>,
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
    /// 1 when the pixel shader's depth export is dropped (no depth target).
    ps_extra: u16,
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

/// New pipelines between writes of the pipeline cache file.
const PIPELINE_SAVE_EVERY: u32 = 16;

/// A pipeline cache loaded from and saved to a file.
struct DiskCache {
    cache: wgpu::PipelineCache,
    path: std::path::PathBuf,
    /// Pipelines created since the file was last written.
    unsaved: u32,
}

/// A draw's texture slot mask and its slots' views and samplers, in slot
/// order.
type TextureGroupKey = (u32, Vec<(wgpu::TextureView, wgpu::Sampler)>);

/// A draw's texture slot mask, and the fetch constants of those slots with
/// each slot's packed fetch filter overrides ([`pack_filters`]).
type TextureSetKey = (u32, Vec<([u32; 6], u32)>);

/// A fetch instruction's filter overrides as one word: 0 for none, else
/// bit 31 with mag, min, mip and aniso in bytes 0 to 3.
fn pack_filters(f: Option<[u8; 4]>) -> u32 {
    f.map_or(0, |f| 1 << 31 | u32::from_le_bytes(f))
}

/// A program's translation, or why it failed.
type Translation = Result<Arc<Translated>, String>;

/// Key of a shader module: program hash, vertex stage, extra vertex
/// outputs (pixel stage: 1 when the depth export is dropped), variant and
/// pixel parameter generation.
type ModuleKey = (u64, bool, u16, Variant, Option<ParamGen>);

/// The vertex and pixel shader modules a pipeline is made from.
fn module_keys(k: &PipelineKey) -> [ModuleKey; 2] {
    [
        (k.vs, true, k.vs_extra, k.vs_variant, None),
        (k.ps, false, k.ps_extra, k.ps_variant, k.ps_param_gen),
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

/// Bounds on the spare render target textures kept for reuse (see
/// [`Spares`]): past either the oldest go.
const SPARE_TARGETS: usize = 32;
const SPARE_TARGET_BYTES: u64 = 128 << 20;

/// Usages of every render target texture.
const TARGET_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::TEXTURE_BINDING);

/// What a spare render target texture can stand in for: format, width,
/// height, samples and usages.
type SpareKey = (wgpu::TextureFormat, u32, u32, u32, wgpu::TextureUsages);

fn spare_key(t: &wgpu::Texture) -> SpareKey {
    (
        t.format(),
        t.width(),
        t.height(),
        t.sample_count(),
        t.usage(),
    )
}

/// Textures of dropped render targets, oldest first, kept for a new target
/// of the same description instead of making one: games drop and remake
/// several targets a frame. Bounded by count and bytes.
struct Spares<K, T> {
    items: std::collections::VecDeque<(K, u64, T)>,
    bytes: u64,
    max_count: usize,
    max_bytes: u64,
}

impl<K: PartialEq, T> Spares<K, T> {
    fn new(max_count: usize, max_bytes: u64) -> Self {
        Self {
            items: Default::default(),
            bytes: 0,
            max_count,
            max_bytes,
        }
    }

    /// Keeps `t` of `bytes` under `key`, freeing the oldest past the caps.
    fn put(&mut self, key: K, bytes: u64, t: T) {
        self.items.push_back((key, bytes, t));
        self.bytes += bytes;
        self.trim(self.max_count, self.max_bytes);
    }

    /// Frees the oldest until at most `max_count` of at most `max_bytes`
    /// are left.
    fn trim(&mut self, max_count: usize, max_bytes: u64) {
        while self.items.len() > max_count || self.bytes > max_bytes {
            let Some((_, b, _)) = self.items.pop_front() else {
                break;
            };
            self.bytes -= b;
        }
    }

    /// The newest spare kept under `key`, taken out.
    fn take(&mut self, key: &K) -> Option<T> {
        let i = self.items.iter().rposition(|(k, _, _)| k == key)?;
        let (_, b, t) = self.items.remove(i)?;
        self.bytes -= b;
        Some(t)
    }
}

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
    /// Shared, so a draw takes a translation without copying its WGSL.
    translated: HashMap<(u64, bool, Option<ParamGen>), Translation>,
    /// Keyed by program hash, stage, extra vertex outputs, variant (the
    /// rectangle list wrapper or the pixel output bias and clamp) and pixel parameter
    /// generation.
    modules: HashMap<ModuleKey, wgpu::ShaderModule>,
    /// With the use that last drew with each, for least recently used
    /// eviction past `pipeline_cap`, and the texture bind group layout of
    /// its key's slots.
    pipelines: HashMap<PipelineKey, (wgpu::RenderPipeline, u64, wgpu::BindGroupLayout)>,
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
    /// Textures of dropped targets for new ones to reuse, and how many
    /// target textures were made and reused so far.
    spare_targets: Spares<SpareKey, wgpu::Texture>,
    targets_made: u64,
    targets_reused: u64,
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
    /// Textures already looked up by fetch constant, with the physical
    /// `[start, end)` guest range they were read from and their
    /// `texture_cache` key (none for kept-resolve decodes). Guest memory
    /// only changes at a resolve or MEM_WRITE during a replay, which drop
    /// the entries whose range they write, so later draws skip reading and
    /// hashing the guest bytes of the others. Cleared at each replay, unless
    /// guest memory lists the pages the CPU wrote between replays
    /// ([`GuestMemory::take_written`]): then only those entries go.
    texture_memo: HashMap<[u32; 6], TextureMemo>,
    /// Guest texture bytes are read into this buffer, kept across lookups:
    /// a fresh one per texture costs zeroing and page faults every frame.
    texture_scratch: Vec<u8>,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    /// Draws recorded but not yet submitted (see [`Batch`]).
    batch: Option<Batch>,
    /// Texture bind groups by texture slot mask and the slots' views and
    /// samplers, with the replay that last bound each. Kept across flushes
    /// and replays (a new texture has a new view, so a stale entry never
    /// matches); at the end of a replay those not bound in it or the one
    /// before go, and all go when textures are evicted and past
    /// [`TEXTURE_GROUP_CAP`], so they keep dropped textures alive no longer
    /// than that. (Cleared at every replay's end, a frame made its
    /// thousands again: a tenth of the draw thread.)
    texture_groups: HashMap<TextureGroupKey, (wgpu::BindGroup, u64)>,
    /// A draw's texture bind group by its slot mask and the fetch constants
    /// of those slots, with the [`Drawer::texture_memo_epoch`] it was found
    /// in and a flag that stays set while the memo keeps its textures: a
    /// later draw with the same constants binds it without looking each
    /// texture up again.
    texture_sets: HashMap<TextureSetKey, (wgpu::BindGroup, u64, Arc<AtomicBool>)>,
    /// The flags of the sets each fetch constant is in: a guest write that
    /// drops the constant's memo entry clears them, so only the sets using
    /// that texture go (dropping them all made a third of draws miss).
    texture_set_flags: HashMap<[u32; 6], Vec<Arc<AtomicBool>>>,
    /// The fetch constants of a draw's texture set, kept so looking a set
    /// up does not allocate.
    texture_set_key: Vec<([u32; 6], u32)>,
    /// Counts the changes that drop all [`Drawer::texture_memo`] entries.
    texture_memo_epoch: u64,
    /// Group 0 bind groups over `arena`, by the sizes of the four slices
    /// they bind (at dynamic offsets). Cleared when the arena is replaced.
    arena_groups: HashMap<[u64; 6], wgpu::BindGroup>,
    /// The clear bind group over `arena`, its value at a dynamic offset.
    /// Dropped when the arena is replaced.
    clear_group: Option<wgpu::BindGroup>,
    /// Views of render target textures for clears, kept as long as the
    /// target.
    target_views: HashMap<wgpu::Texture, wgpu::TextureView>,
    /// Draw buffers (guest window, constants, epilogue, indices) staged
    /// since the last flush, as slices of `arena` from `staged_base`,
    /// written to it at the next flush. The arena is a ring: flushes
    /// append, and staging starts again at 0 only when it is full.
    staged: Vec<u8>,
    staged_base: u64,
    /// This replay began with guest memory listing the pages written since
    /// the last ([`GuestMemory::take_written`]), so the arena and
    /// `window_memo` are kept across replays and dropped only where written.
    watching: bool,
    submitter: Submitter,
    /// Flushed batches the submit thread may hold (`SUBMITS_IN_FLIGHT`).
    submits_in_flight: usize,
    arena: Option<wgpu::Buffer>,
    arena_align: u64,
    /// Lists [`Self::guest_window`] and a draw fill and empty, kept so a
    /// draw does not allocate them again.
    window_scratch: WindowScratch,
    /// Vertex windows staged in the arena since it last wrapped: guest
    /// physical `lo` to `hi` and their (offset, size) in `arena`. Cleared
    /// when the arena wraps or grows and at the end of a replay, and
    /// dropped where guest memory changes (see `forget_guest`). No window
    /// holds another, so `hi` grows with `lo` and the one window that can
    /// hold a range is the last starting at or before it.
    window_memo: BTreeMap<u64, (u64, (u64, u64))>,
    /// Host indices of guest index buffers read this replay or, with
    /// guest writes listed, an earlier one: guest `lo`, `hi` and the
    /// indices. Dropped with the textures (see `forget_guest`).
    index_memo: HashMap<IndexKey, (u64, u64, Arc<HostIndices>)>,
    /// Bytes of the indices `index_memo` holds, counted as they go in.
    index_memo_bytes: usize,
    /// See [`Drawer::transfer_setup`].
    transfer_setups: HashMap<(RenderTargetKey, u32, RenderTargetKey, u32), Arc<TransferSetup>>,
    /// The last `Consts` block staged in the arena, as prefix and suffix,
    /// and its (offset, size), bound again by a draw with the same bytes.
    /// Cleared with `window_memo` when the arena wraps or grows.
    staged_consts: Option<(ConstsBlock, [u32; CONSTS_SUFFIX], (u64, u64))>,
    /// Counts the arena's wraps (from 1): a slice staged in an earlier
    /// epoch is overwritten (see [`HostIndices::staged`]).
    arena_epoch: u64,
    /// Counts the writes to the vertex stage's float constant registers,
    /// and to the pixel stage's.
    float_gens: [u64; 2],
    /// The float constants of each stage staged last, bound again by
    /// draws that read the same ones of unchanged registers.
    staged_floats: [StagedFloats; 2],
    /// Bumped by every register write to the constants a `Consts` block
    /// holds, so a draw rebuilds the block only after one.
    consts_gen: u64,
    /// Draws through pipeline lookup: those at multiples of
    /// [`PART_SAMPLE`] time their sub-phases. A sampled draw that took a
    /// slow path ([`Drawer::slow_work`]) is not counted, so the next draw
    /// takes its slot.
    part_draws: u32,
    /// The last `Consts` block prefix built, by generation: the same
    /// block while its registers hold the same words, so draws compare
    /// blocks by pointer.
    consts_built: Option<(u64, ConstsBlock)>,
    /// Kept-resolve texture decodes recorded since the last flush,
    /// submitted with (before) the batch's draws.
    pre: Option<wgpu::CommandEncoder>,
    /// A GPU resolve is recorded in the batch, not yet submitted: what
    /// reads kept resolve buffers on its own (write-back, decodes, which
    /// run before the batch) flushes first.
    resolve_pending: bool,
    max_buffer: u64,
    /// 1x1 texture of zeros bound for a slot without a usable texture.
    zero_texture: wgpu::TextureView,
    zero_sampler: wgpu::Sampler,
    /// Whether `Rgba32Float` textures can be sampled with filtering; without
    /// it wide formats are uploaded as `Rgba16Float`.
    float32_filterable: bool,
    /// The device has `ADDRESS_MODE_CLAMP_TO_BORDER`; without it border
    /// clamps are drawn as edge clamps.
    clamp_to_border: bool,
    /// Compute pipeline of resolve.wgsl, made on the first GPU resolve.
    /// Colour, depth and depth + stencil entry points of resolve.wgsl.
    resolve_pipelines: [Option<wgpu::ComputePipeline>; 3],
    /// Their bind group and pipeline layouts.
    resolve_layouts: [(wgpu::BindGroupLayout, wgpu::PipelineLayout); 3],
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
    /// `XENON_RESOLVE_WRITEBACK=skip` turns it on at [`Drawer::new`]; the
    /// host turns it on unless that variable is `on` or `late`.
    pub skip_resolve_writeback: bool,
    /// Queue the end-of-replay write-back of kept resolves without waiting
    /// for it, and write it to guest memory when the next replay starts:
    /// the CPU sees resolve output one frame late, in exchange for no GPU
    /// wait at the end of every replay. `XENON_RESOLVE_WRITEBACK=late`
    /// turns it on at [`Drawer::new`].
    pub late_writeback: bool,
    /// Copy each front buffer held by a kept resolve out on the GPU and
    /// hand it over at the next swap, so a swap never waits for the GPU to
    /// finish the frame; the frame image is then one swap late, and the
    /// first swap gives none. `XENON_LATE_FRAME=1` turns it on at
    /// [`Drawer::new`]; the host turns it on unless that variable is `0`.
    pub late_front_buffer: bool,
    /// The front buffer copy queued at the last swap, see
    /// [`Drawer::late_front_buffer`].
    late_front: Option<LateFront>,
    /// Records kept-texture decodes after a resolve in the open batch
    /// instead of flushing it first: about a tenth of the drawer's time in
    /// a frame of thousands of draws. `XENON_DECODE_IN_BATCH=0` turns it
    /// off (it was off for a time: a machine hung twice with it on, and
    /// also hung the same way without it).
    decode_in_batch: bool,
    late: Option<(Vec<Vec<Option<u64>>>, WriteBack)>,
    /// Bytes of decoded guest textures the texture cache keeps before
    /// evicting the least recently used ones (default 256 MiB).
    pub texture_cache_limit: u64,
    /// Decoded guest textures the texture cache keeps however small
    /// (default [`TEXTURE_CACHE_COUNT`]).
    pub texture_cache_count: usize,
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
    /// Diagnostic: anisotropy for linear/linear fetches without any, 1 off
    /// (see [`sampler_diagnostics`]). Default from `XENON_ANISO`.
    pub force_aniso: u16,
    /// Diagnostic: every sampler's mag and min filters linear. Default from
    /// `XENON_FORCE_LINEAR=1`.
    pub force_linear: bool,
    timings: Timings,
    /// GPU timestamps around the batch's passes, when enabled.
    gpu_clock: Option<GpuClock>,
    /// File name for the pipeline cache of this adapter, when the device
    /// has `PIPELINE_CACHE` (Vulkan only).
    cache_key: Option<String>,
    disk_cache: Option<DiskCache>,
    uncaptured: Arc<Mutex<Vec<String>>>,
    /// Set with the first error put in `uncaptured`, so the check after
    /// every command is one load while there is none.
    uncaptured_any: Arc<std::sync::atomic::AtomicBool>,
    max_dim: u32,
    max_storage: u64,
    /// Host pixels per guest pixel on each axis of every render target
    /// (see [`Drawer::set_scale`]).
    scale: u32,
    /// Last, so it drops after the device and every resource made from it.
    instance: wgpu::Instance,
}

impl Drop for Drawer {
    fn drop(&mut self) {
        if let Err(e) = self.save_pipeline_cache() {
            eprintln!("[draw] pipeline cache not saved: {e}");
        }
    }
}

/// Draws recorded into one command encoder, in one render pass while the
/// render targets stay the same, and the resolves, resolve clears and EDRAM
/// transfers between them, each in a pass of its own. Submitted by
/// [`Drawer::flush`] before anything that reads the targets outside the
/// batch (write-backs, readbacks) and at the end of a replay.
struct Batch {
    /// Dropped (ended) before `enc` is finished.
    pass: Option<(PassKey, wgpu::RenderPass<'static>, Bound)>,
    enc: wgpu::CommandEncoder,
}

/// The state a render pass has set, so a draw sets only what changed:
/// wgpu validates and tracks every set call again at submit.
#[derive(Default)]
struct Bound {
    pipeline: Option<wgpu::RenderPipeline>,
    g0: Option<(wgpu::BindGroup, [u32; 6])>,
    g1: Option<wgpu::BindGroup>,
    /// The whole arena, so draws pick their indices by first index.
    index: Option<wgpu::Buffer>,
    viewport: Option<(u32, u32)>,
    scissor: Option<(u32, u32, u32, u32)>,
    stencil: Option<u32>,
    blend: Option<[u32; 4]>,
}

/// What [`Drawer::transfer`] records for one pair of targets.
struct TransferSetup {
    pipelines: Vec<wgpu::RenderPipeline>,
    groups: Vec<wgpu::BindGroup>,
    view: wgpu::TextureView,
    to_depth: bool,
    stencil: bool,
}

/// Where a queue submission comes from: the index of
/// [`Timings::submits_by_site`].
#[derive(Clone, Copy)]
enum Site {
    Resolve,
    // Transfers ride in the batch now; the slot stays so the others keep
    // their indices on the `[draw]` line.
    #[allow(dead_code)]
    Transfer,
    Decode,
    Other,
}

/// Colour and depth targets of a render pass.
type PassKey = ([Option<RenderTargetKey>; 4], Option<RenderTargetKey>);

/// Group 0 bind groups kept for the arena, one per set of slice sizes.
const ARENA_GROUP_CAP: usize = 4096;

/// Texture bind groups kept between evictions and replay ends.
const TEXTURE_GROUP_CAP: usize = 4096;

/// Resolve bind groups one kept resolve buffer keeps between replay ends.
const RESOLVE_GROUP_CAP: usize = 1024;

/// Bytes of each upload staging chunk, and the most one belt write takes.
const BELT_CHUNK: u64 = 4 << 20;

/// Smallest draw upload arena; it grows to fit a larger draw.
const ARENA_MIN: u64 = 16 << 20;

/// Smallest arena while guest memory lists its writes: vertex windows then
/// stay staged across replays until written, so the arena holds several
/// frames of them instead of being refilled every frame.
const ARENA_WATCHED: u64 = 256 << 20;

/// Largest CPU-side staging capacity reserved up front.
const STAGED_RESERVE: u64 = 64 << 20;

/// Time since `t`, restarting `t`.
fn lap(t: &mut Instant) -> Duration {
    let now = Instant::now();
    let d = now - *t;
    *t = now;
    d
}

/// One draw in this many times the sub-phases of `shaders` and `submit`
/// ([`Timings::setup_parts`], [`Timings::submit_parts`]); their sums are
/// scaled by it. Each clock read costs tens of nanoseconds, too many per
/// draw for a split that only shows where a phase goes.
const PART_SAMPLE: u32 = 32;

/// Adds the spans between consecutive `marks` to `parts`, scaled by
/// [`PART_SAMPLE`] as only one draw in that many is timed.
fn add_sampled(parts: &mut [Duration], marks: &[Instant]) {
    for (p, w) in parts.iter_mut().zip(marks.windows(2)) {
        *p += (w[1] - w[0]) * PART_SAMPLE;
    }
}

/// Times a replay's plain commands (all but draws, resolves and swaps,
/// which time their own phases) and the upkeep after every command, with
/// clock reads only at the edges of the timed ones: a run of plain
/// commands costs no read of its own.
struct PlainTimer<F: FnMut() -> Instant> {
    now: F,
    mark: Instant,
}

impl<F: FnMut() -> Instant> PlainTimer<F> {
    fn new(mut now: F) -> Self {
        let mark = now();
        Self { now, mark }
    }

    /// Before a timed command or at the end: charges the run since the
    /// last timed command to `commands`.
    fn charge(&mut self, commands: &mut Duration) {
        let now = (self.now)();
        *commands += now - self.mark;
        self.mark = now;
    }

    /// After a timed command: the next run starts here.
    fn restart(&mut self) {
        self.mark = (self.now)();
    }
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
/// swizzle (3), mip levels (4), and dimension, packed mips and mip address
/// (5), which pick the mip levels decoded. Clamp, filters, LOD bias,
/// exp_adjust and border only change sampling, so textures differing in
/// those share one decode.
fn decode_key(c: [u32; 6]) -> [u32; 6] {
    [
        c[0] & !(0x1FF << 10),
        c[1] & !(1 << 11),
        c[2],
        c[3] & (0xFFF << 1),
        c[4] & (0xFF << 2),
        c[5] & !0x1FF,
    ]
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    // Four independent lanes: one lane's multiply chain held the hash to
    // about 70% of the speed the CPU reads memory at (textures hash every
    // frame).
    let mut chunks = bytes.chunks_exact(32);
    let mut lanes = [bytes.len() as u64, 1, 2, 3];
    for c in &mut chunks {
        for (i, h) in lanes.iter_mut().enumerate() {
            let w = u64::from_le_bytes(c[i * 8..][..8].try_into().unwrap());
            *h = (h.rotate_left(5) ^ w).wrapping_mul(K);
        }
    }
    let mut h = lanes
        .iter()
        .fold(0, |a: u64, &x| (a.rotate_left(5) ^ x).wrapping_mul(K));
    for &b in chunks.remainder() {
        h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }
    h
}

/// The physical byte range `[lo, hi)` vertex fetch constant `k`, at
/// physical byte address `a` with `size_dwords` words, has read by the
/// fetches in `reach` when the vertex shader's starting indices (index plus
/// VGT_INDX_OFFSET, 24 bits) lie in `indices`; `None` when they are not
/// known. Clipped to the constant's buffer, which is all of it for a fetch
/// with an unbounded index or a constant without a reach entry. The flag is
/// set when the draw may read past the buffer: some fetch's words were
/// clipped, or the whole buffer was taken because the reads are not known.
fn fetch_range(
    a: u64,
    size_dwords: u32,
    k: u32,
    reach: &[VertexFetchReach],
    indices: Option<(u32, u32)>,
) -> (u64, u64, bool) {
    let end = a + 4 * u64::from(size_dwords);
    let mut entries = reach.iter().filter(|r| r.fetch_const == k).peekable();
    if entries.peek().is_none() {
        return (a, end, true);
    }
    let (mut lo, mut hi, mut clipped) = (end, a, false);
    for r in entries {
        let (i0, i1) = match (r.stride, indices) {
            (0, _) => (0, 0),
            (_, Some(i)) if r.bounded => i,
            _ => return (a, end, true),
        };
        let stride = i64::from(r.stride);
        let first = a as i64 + 4 * (i64::from(i0) * stride + i64::from(r.first));
        let last = a as i64 + 4 * (i64::from(i1) * stride + i64::from(r.end));
        clipped |= first < a as i64 || last > end as i64;
        lo = lo.min(first.clamp(a as i64, end as i64) as u64);
        hi = hi.max(last.clamp(a as i64, end as i64) as u64);
    }
    (lo, hi.max(lo), clipped)
}

/// The smallest and largest index but the primitive reset (`u32::MAX`).
fn min_max(indices: &[u32]) -> Option<(u32, u32)> {
    let mut used = indices.iter().filter(|&&i| i != u32::MAX);
    let first = *used.next()?;
    Some(used.fold((first, first), |(lo, hi), &i| (lo.min(i), hi.max(i))))
}

/// The lowest and highest vertex index the vertex shader starts with for
/// indices with this [`min_max`] and index offset `offset`, as the shader
/// adds them in 32 bits and keeps 24; `None` when that wraps.
fn offset_range(min_max: Option<(u32, u32)>, offset: u32) -> Option<(u32, u32)> {
    let (lo, hi) = min_max?;
    let (lo, hi) = (lo.wrapping_add(offset), hi.wrapping_add(offset));
    (lo <= hi && lo >> 24 == hi >> 24).then_some((lo & 0xFF_FFFF, hi & 0xFF_FFFF))
}

/// A draw's guest index buffer and how its indices become host ones:
/// address, buffer length, format, endian, count, reset index (when on),
/// expansion and whether the topology is a strip.
type IndexKey = (u32, u32, u8, u8, u32, Option<u32>, u8, bool);

/// A draw's host indices, as staged, with their count and [`min_max`].
struct HostIndices {
    bytes: Vec<u8>,
    len: u32,
    min_max: Option<(u32, u32)>,
    /// Where these bytes were staged last: the [`Drawer::arena_epoch`]
    /// then (0: never) and the arena offset. The slice holds them until
    /// the arena wraps, so draws of the same indices bind it again.
    staged: [std::sync::atomic::AtomicU64; 2],
}

impl HostIndices {
    /// The arena slice holding these bytes in `epoch`, if one does.
    fn staged_in(&self, epoch: u64) -> Option<(u64, u64)> {
        use std::sync::atomic::Ordering::Relaxed;
        (self.staged[0].load(Relaxed) == epoch)
            .then(|| (self.staged[1].load(Relaxed), self.bytes.len() as u64))
    }

    fn set_staged(&self, epoch: u64, offset: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.staged[1].store(offset, Relaxed);
        self.staged[0].store(epoch, Relaxed);
    }
}

/// [`Drawer::index_memo`] entries and index bytes held before it starts
/// over.
const INDEX_MEMO_CAP: usize = 1 << 14;
const INDEX_MEMO_BYTES: usize = 32 << 20;

/// Vertex data ranges closer than this share one window part; farther
/// ranges are staged apart, so the gap between them is not uploaded.
const WINDOW_GAP: u64 = 4096;

/// A [`Drawer::texture_memo`] entry: view, guest range, cache key.
type TextureMemo = (wgpu::TextureView, u64, u64, Option<([u32; 6], u64)>);

/// Entries [`Drawer::texture_memo`] holds before it starts over.
const TEXTURE_MEMO_CAP: usize = 1 << 14;

/// Cached textures are dropped wholesale past this many bytes.
const TEXTURE_CACHE_LIMIT: u64 = 256 << 20;

/// Cached textures past this many are dropped too: small ones each cost a
/// texture and view in the driver however few bytes they hold.
const TEXTURE_CACHE_COUNT: usize = 2048;

fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * words.len());
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// A built `Consts` block prefix ([`consts_prefix`]), shared by the draws
/// that bind it.
type ConstsBlock = Arc<Vec<u8>>;

/// Registers of one stage's 256 float constants (4 each): the vertex
/// stage's, then the pixel stage's.
const FLOAT_STAGE_REGS: u32 = 1024;
const _: () = assert!(xenon_shader::PIXEL_FLOAT_CONST_BASE * 4 == FLOAT_STAGE_REGS);

/// A stage's float constants staged in the arena: the generation of its
/// registers, the mask of the constants staged, and the slice.
type StagedFloats = Option<((u64, [u64; 4]), (u64, u64))>;

/// The float constants of `stage` (0 vertex, 1 pixel) that `mask` names,
/// in ascending order, written to `out` (16 bytes each) as the bytes of a
/// stage's `fc` binding.
fn float_consts(r: &regs::RegisterFile, stage: u32, mask: &[u64; 4], out: &mut [u8]) {
    let all = r.slice(
        regs::SHADER_CONSTANT_FLOAT + FLOAT_STAGE_REGS * stage,
        FLOAT_STAGE_REGS,
    );
    let read = (0..256).filter(|i| mask[i / 64] >> (i % 64) & 1 != 0);
    for (o, i) in out.chunks_exact_mut(16).zip(read) {
        for (o, w) in o.chunks_exact_mut(4).zip(&all[4 * i..4 * i + 4]) {
            o.copy_from_slice(&w.to_le_bytes());
        }
    }
}

/// `words` as little-endian bytes into `out`, which holds them all.
fn words_into(words: &[u32], out: &mut [u8]) {
    for (o, w) in out.chunks_exact_mut(4).zip(words) {
        o.copy_from_slice(&w.to_le_bytes());
    }
}

/// The registers a `Consts` block is built from (see [`consts`]).
const CONSTS_REGS: [std::ops::Range<u32>; 3] = [
    regs::SHADER_CONSTANT_FETCH..regs::SHADER_CONSTANT_FETCH + 192,
    regs::SHADER_CONSTANT_BOOL..regs::SHADER_CONSTANT_BOOL + 8 + 32,
    regs::VGT_INDX_OFFSET..regs::VGT_INDX_OFFSET + 1,
];

/// Words of a `Consts` block's suffix: the window base, the index offset,
/// two of padding and the 96 rebase words.
const CONSTS_SUFFIX: usize = 4 + 96;

/// Bytes of a `Consts` block: its prefix (see [`consts_prefix`]) and suffix.
const CONSTS_BYTES: usize = 4 * (192 + 8 + 32) + 4 * CONSTS_SUFFIX;

/// The `Consts` uniform of the shader binding contract is this prefix, the
/// fetch, bool and loop constants, which only register writes change, then
/// [`consts_suffix`]. (The float constants are bound apart, a span for
/// each stage: see [`FLOAT_STAGE_REGS`].)
fn consts_prefix(r: &regs::RegisterFile) -> Vec<u8> {
    // Bool and loop constants are adjacent.
    const _: () = assert!(regs::SHADER_CONSTANT_BOOL + 8 == regs::SHADER_CONSTANT_LOOP);
    let fetch = r.slice(regs::SHADER_CONSTANT_FETCH, 192);
    let bool_loop = r.slice(regs::SHADER_CONSTANT_BOOL, 8 + 32);
    let mut b = vec![0; 4 * (fetch.len() + bool_loop.len())];
    words_into(fetch, &mut b);
    words_into(bool_loop, &mut b[4 * fetch.len()..]);
    b
}

/// The rest of a `Consts` block, which each draw's window sets: the window
/// base, the index offset and the 96 vertex fetch constants' rebase words.
fn consts_suffix(
    r: &regs::RegisterFile,
    window_base: u32,
    rebase: &[i32; 96],
) -> [u32; CONSTS_SUFFIX] {
    let mut s = [0; CONSTS_SUFFIX];
    s[..2].copy_from_slice(&[window_base, r.get(regs::VGT_INDX_OFFSET)]);
    for (o, &w) in s[4..].iter_mut().zip(rebase) {
        *o = w as u32;
    }
    s
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

/// A translated pixel shader without its oDepth output, for a pass with no
/// depth target (wgpu refuses a `frag_depth` output there).
fn drop_depth_wgsl(ps: &str) -> Result<String, String> {
    if !ps.contains(DEPTH_MEMBER) || !ps.contains(DEPTH_STORE) {
        return Err("pixel shader depth output not recognised".into());
    }
    Ok(ps
        .replacen(DEPTH_MEMBER, "", 1)
        .replacen(DEPTH_STORE, "", 1))
}

/// The address mode of a `ClampMode`, and whether it is exact. `border`:
/// the device has `ADDRESS_MODE_CLAMP_TO_BORDER`.
fn sampler_address(clamp: u8, border: bool) -> (wgpu::AddressMode, bool) {
    use wgpu::AddressMode as A;
    match clamp {
        0 => (A::Repeat, true),
        1 => (A::MirrorRepeat, true),
        2 => (A::ClampToEdge, true),
        6 if border => (A::ClampToBorder, true),
        // Xenia draws the mirror-once modes (to edge, half-way, border) as
        // mirror clamp to edge, which wgpu lacks.
        // shortcut: edge clamp, right for coordinates >= 0 and wrong for
        // negative ones (the guest mirrors them); upgrade when a trace shows
        // negative coordinates on a mirror-once fetch, with abs() on the
        // coordinate from a per-slot bit the translated fetch reads.
        3 | 5 | 7 => (A::ClampToEdge, false),
        // Half-way clamp is edge clamp, as in Xenia; border without the
        // feature too.
        _ => (A::ClampToEdge, false),
    }
}

/// What a guest sampler is made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    address: [wgpu::AddressMode; 3],
    mag: wgpu::FilterMode,
    min: wgpu::FilterMode,
    mip: wgpu::MipmapFilterMode,
    /// 1 is off.
    anisotropy: u16,
    /// Only with a `ClampToBorder` address mode.
    border: Option<wgpu::SamplerBorderColor>,
    /// The first mip level the fetch samples: the lowest LOD.
    min_level: u32,
    /// Base map mip filter: LODs up to `min_level` + 0.25 only.
    base_map: bool,
}

/// The sampler of `fetch` with a fetch instruction's mag, min, mip and
/// aniso filter `overrides` (3, 3, 3 and 7 keep the constant's), as Xenia's
/// GetSamplerParameters and UseSampler. `border`: the device has
/// `ADDRESS_MODE_CLAMP_TO_BORDER`.
fn sampler_key(
    fetch: &xenon_texture::TextureFetch,
    overrides: Option<[u8; 4]>,
    border: bool,
) -> SamplerKey {
    let [mag, min, mip, aniso] = overrides.unwrap_or([3, 3, 3, 7]);
    let pick = |o: u8, keep: u8, f: u8| if o == keep { f } else { o };
    let (mag, min) = (
        pick(mag, 3, fetch.mag_filter),
        pick(min, 3, fetch.min_filter),
    );
    let (mip, aniso) = (
        pick(mip, 3, fetch.mip_filter),
        pick(aniso, 7, fetch.aniso_filter),
    );
    let address = fetch.clamp.map(|c| sampler_address(c, border).0);
    // As Xenia, any anisotropy (1:1 too) makes every filter linear, which
    // wgpu also needs for anisotropy. Xenia: border colours other than
    // opaque white are transparent black (CrYCb ones are not done).
    let (mag_f, min_f, mip_linear) = if aniso != 0 {
        (1, 1, true)
    } else {
        (mag, min, mip == 1)
    };
    SamplerKey {
        address,
        mag: filter(mag_f),
        min: filter(min_f),
        mip: if mip_linear {
            wgpu::MipmapFilterMode::Linear
        } else {
            wgpu::MipmapFilterMode::Nearest
        },
        anisotropy: match aniso {
            0 => 1,
            a => 1 << (a.min(5) - 1),
        },
        border: address
            .contains(&wgpu::AddressMode::ClampToBorder)
            .then_some(if fetch.border_color == 1 {
                wgpu::SamplerBorderColor::OpaqueWhite
            } else {
                wgpu::SamplerBorderColor::TransparentBlack
            }),
        min_level: fetch.mip_range().0,
        base_map: mip == 2,
    }
}

/// The diagnostic overrides on a guest sampler. `force_linear` makes mag
/// and min linear even for point fetches (diagnosis only: it breaks lookup
/// and data textures). `force_aniso` (1 is off) gives that anisotropy, with
/// linear mip filtering, to linear/linear fetches without any; it never
/// touches point filters, base maps or the guest's own anisotropy.
fn sampler_diagnostics(mut k: SamplerKey, force_aniso: u16, force_linear: bool) -> SamplerKey {
    if force_linear {
        (k.mag, k.min) = (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear);
    }
    if force_aniso > 1
        && k.anisotropy == 1
        && k.mag == wgpu::FilterMode::Linear
        && k.min == wgpu::FilterMode::Linear
        && !k.base_map
    {
        k.anisotropy = force_aniso;
        k.mip = wgpu::MipmapFilterMode::Linear;
    }
    k
}

/// `XENON_ANISO`: 2, 4, 8 or 16; anything else warns and is off (1).
fn force_aniso_from(v: Option<&str>) -> u16 {
    match v.map(str::trim) {
        None => 1,
        Some(s @ ("2" | "4" | "8" | "16")) => s.parse().unwrap_or(1),
        Some(s) => {
            eprintln!("XENON_ANISO={s:?} is not 2, 4, 8 or 16; forced anisotropy off");
            1
        }
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
        // wgpu's release default, whatever the build: it turns Vulkan
        // validation on in builds with debug assertions, which costs
        // microseconds a draw where the layer is installed.
        // WGPU_VALIDATION=1 and WGPU_DEBUG=1 still do.
        desc.flags = wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL.with_env();
        // Requested only when used, so the device is made as it was before
        // timestamps: runs with them (and the flags above) unused on it
        // held about 0.6 GB more host memory outside the Rust heap.
        let timestamps = std::env::var_os("XENON_GPU_TIMESTAMPS").is_some_and(|v| v != "0");
        let flags = desc.flags;
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
        required_features |= adapter.features()
            & (wgpu::Features::PIPELINE_CACHE | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER);
        if timestamps {
            required_features |= adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        }
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
        let uncaptured_any = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (sink, any) = (uncaptured.clone(), uncaptured_any.clone());
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock() {
                v.push(e.to_string());
                any.store(true, std::sync::atomic::Ordering::Release);
            }
        }));
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        // Arena slices are bound at dynamic offsets, so draws whose slices
        // have the same sizes share one bind group.
        let dynamic = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: true,
            min_binding_size: None,
        };
        let (dyn_storage, dyn_uniform) = (
            dynamic(wgpu::BufferBindingType::Storage { read_only: true }),
            dynamic(wgpu::BufferBindingType::Uniform),
        );
        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("guest window, constants, epilogue, rectangle indices, float constants"),
            entries: &[
                entry(0, vf, dyn_storage),
                entry(1, vf, dyn_uniform),
                entry(2, wgpu::ShaderStages::VERTEX, dyn_uniform),
                entry(3, wgpu::ShaderStages::VERTEX, dyn_storage),
                entry(
                    xenon_shader::VERTEX_FLOAT_CONSTS_BINDING,
                    wgpu::ShaderStages::VERTEX,
                    dyn_uniform,
                ),
                entry(
                    xenon_shader::PIXEL_FLOAT_CONSTS_BINDING,
                    wgpu::ShaderStages::FRAGMENT,
                    dyn_uniform,
                ),
            ],
        });
        // Clear values and resolve parameters are staged in the arena too.
        let clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear"),
            entries: &[entry(0, vf, dyn_uniform)],
        });
        // resolve.wgsl's main, depth_main and depth_stencil_main: the
        // source (colour, depth, depth and stencil), the kept resolve
        // buffer and the parameters.
        let resolve_layouts = {
            let cs = wgpu::ShaderStages::COMPUTE;
            let texture = |sample_type| wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            };
            let dest = entry(
                1,
                cs,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            );
            let color = entry(
                0,
                cs,
                texture(wgpu::TextureSampleType::Float { filterable: false }),
            );
            let depth = entry(3, cs, texture(wgpu::TextureSampleType::Depth));
            let stencil = entry(4, cs, texture(wgpu::TextureSampleType::Uint));
            let params = entry(2, cs, dyn_uniform);
            [
                vec![color, dest, params],
                vec![dest, params, depth],
                vec![dest, params, depth, stencil],
            ]
            .map(|entries| {
                let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("resolve"),
                    entries: &entries,
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("resolve"),
                    bind_group_layouts: &[Some(&group)],
                    immediate_size: 0,
                });
                (group, layout)
            })
        };
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
        let submitter = Submitter::new(device.clone(), queue.clone());
        let mut d = Drawer {
            instance,
            device,
            queue,
            adapter: format!(
                "{} ({:?}, {:?}){}{}",
                info.name,
                info.backend,
                info.device_type,
                if flags.contains(wgpu::InstanceFlags::VALIDATION) {
                    ", validation on"
                } else {
                    ""
                },
                if cfg!(debug_assertions) {
                    ", built with debug assertions"
                } else {
                    ""
                }
            ),
            timeout: DEFAULT_TIMEOUT,
            regs: RegisterFile::default(),
            vs: None,
            ps: None,
            translated: HashMap::default(),
            modules: HashMap::default(),
            pipelines: HashMap::default(),
            pipeline_uses: 0,
            pipeline_cap: PIPELINE_CAP,
            pipeline_drops: 0,
            texture_layouts: HashMap::default(),
            group0,
            clear_layout,
            clear_module,
            clear_pipelines: HashMap::default(),
            targets: HashMap::default(),
            written: HashMap::default(),
            target_drops: 0,
            target_cap: TARGET_CAP,
            spare_targets: Spares::new(SPARE_TARGETS, SPARE_TARGET_BYTES),
            targets_made: 0,
            targets_reused: 0,
            clock: 0,
            transfer_module: None,
            transfer_pipelines: HashMap::default(),
            resolved_endian: HashMap::default(),
            texture_cache: HashMap::default(),
            texture_cache_bytes: 0,
            texture_evicted: HashSet::default(),
            replays: 0,
            texture_memo: HashMap::default(),
            texture_scratch: Vec::new(),
            samplers: HashMap::default(),
            batch: None,
            texture_groups: HashMap::default(),
            texture_sets: HashMap::default(),
            texture_set_flags: HashMap::default(),
            texture_set_key: Vec::new(),
            texture_memo_epoch: 0,
            arena_groups: HashMap::default(),
            clear_group: None,
            target_views: HashMap::default(),
            staged: Vec::new(),
            arena: None,
            window_memo: BTreeMap::new(),
            index_memo: HashMap::default(),
            index_memo_bytes: 0,
            transfer_setups: HashMap::default(),
            staged_consts: None,
            arena_epoch: 1,
            float_gens: [0; 2],
            staged_floats: [None; 2],
            consts_gen: 0,
            part_draws: 0,
            consts_built: None,
            staged_base: 0,
            watching: false,
            submitter,
            submits_in_flight: std::env::var("XENON_SUBMITS_IN_FLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SUBMITS_IN_FLIGHT),
            pre: None,
            resolve_pending: false,
            window_scratch: WindowScratch::default(),
            arena_align: u64::from(
                limits
                    .min_uniform_buffer_offset_alignment
                    .max(limits.min_storage_buffer_offset_alignment),
            ),
            max_buffer: limits.max_buffer_size,
            zero_texture,
            zero_sampler,
            float32_filterable: required_features.contains(wgpu::Features::FLOAT32_FILTERABLE),
            clamp_to_border: required_features
                .contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER),
            resolve_pipelines: [None, None, None],
            resolve_layouts,
            float_stencil,
            snorm16,
            snorm16_blend,
            half_luts: HashMap::default(),
            mirrors: Vec::new(),
            decode_pipelines: [None, None],
            skip_resolve_writeback: std::env::var("XENON_RESOLVE_WRITEBACK").as_deref() == Ok("skip"),
            late_writeback: std::env::var("XENON_RESOLVE_WRITEBACK").as_deref() == Ok("late"),
            late_front_buffer: std::env::var("XENON_LATE_FRAME").as_deref() == Ok("1"),
            late_front: None,
            decode_in_batch: std::env::var("XENON_DECODE_IN_BATCH").as_deref() != Ok("0"),
            late: None,
            texture_cache_limit: TEXTURE_CACHE_LIMIT,
            texture_cache_count: TEXTURE_CACHE_COUNT,
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
            force_aniso: force_aniso_from(std::env::var("XENON_ANISO").ok().as_deref()),
            force_linear: std::env::var("XENON_FORCE_LINEAR").as_deref() == Ok("1") && {
                eprintln!("XENON_FORCE_LINEAR=1: every sampler linear (diagnostic only; breaks lookup textures)");
                true
            },
            timings: Timings::default(),
            gpu_clock: None,
            cache_key: required_features
                .contains(wgpu::Features::PIPELINE_CACHE)
                .then(|| wgpu::util::pipeline_cache_key(&info))
                .flatten(),
            disk_cache: None,
            uncaptured,
            uncaptured_any,
            max_dim: limits.max_texture_dimension_2d,
            max_storage: limits.max_storage_buffer_binding_size,
            scale: 1,
        };
        if timestamps {
            d.enable_gpu_timestamps();
        }
        if let Ok(v) = std::env::var("XENON_SCALE") {
            let s = v.trim().parse().unwrap_or(0);
            if d.set_scale(s) != s {
                eprintln!(
                    "XENON_SCALE={v:?} is not 1 to 4 or too large for this GPU (2048 x scale rows past its {} texels); drawing at 1",
                    d.max_dim
                );
            }
        }
        Ok(d)
    }

    /// Draws render targets at `s` host pixels per guest pixel on each
    /// axis (1 to 4; default from `XENON_SCALE`, else 1). Everything the
    /// guest or a caller can see stays at guest size: viewports, scissors
    /// and clears are scaled up, and every read of a target into guest
    /// memory or out of [`Drawer::read_render_target_raw`] takes each guest
    /// pixel from its host pixels ([`formats::decimate`]). A scale outside
    /// 1..=4, or one whose 2048 x `s` rows the device cannot hold, is
    /// refused and the scale stays as it was. Drops every render target,
    /// so set it before replaying. Returns the scale now in use.
    pub fn set_scale(&mut self, s: u32) -> u32 {
        if !(1..=4).contains(&s) || RENDER_TARGET_HEIGHT * s > self.max_dim || s == self.scale {
            return self.scale;
        }
        self.scale = s;
        let s2 = u64::from(s * s);
        self.target_cap = TARGET_CAP * s2;
        self.spare_targets = Spares::new(SPARE_TARGETS, SPARE_TARGET_BYTES * s2);
        self.targets.clear();
        self.written.clear();
        self.transfer_setups.clear();
        self.target_views.clear();
        for m in &mut self.mirrors {
            m.groups.clear();
        }
        // Pixel shaders that read the pixel position take the scale as a
        // pipeline constant.
        self.pipelines.retain(|k, _| k.ps_param_gen.is_none());
        self.scale
    }

    /// Host pixels per guest pixel of every render target (see
    /// [`Drawer::set_scale`]).
    pub fn scale(&self) -> u32 {
        self.scale
    }

    /// Guest rows of every render target: [`RENDER_TARGET_HEIGHT`], fewer
    /// where the device cannot hold them at the scale.
    fn target_rows(&self) -> u32 {
        RENDER_TARGET_HEIGHT.min(self.max_dim / self.scale)
    }

    /// Times the batch's draw, clear and resolve passes on the GPU with
    /// timestamp queries, into [`Timings::gpu_passes`]. Each flush then
    /// waits for its batch, so frames run slower and `submit` no longer
    /// includes time queued behind the GPU. Returns false where the device
    /// has no timestamp queries, which [`Drawer::new`] requests only with
    /// `XENON_GPU_TIMESTAMPS=1` (and then turns this on).
    pub fn enable_gpu_timestamps(&mut self) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        let size = 16 * GPU_CLOCK_PASSES;
        let buffer = |usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu clock"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        self.gpu_clock = Some(GpuClock {
            set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu clock"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * GPU_CLOCK_PASSES as u32,
            }),
            resolve: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            kinds: Vec::new(),
            period: self.queue.get_timestamp_period(),
        });
        true
    }

    /// Keeps compiled pipelines in a file under `dir` (made if missing),
    /// loaded now and written every few new pipelines, when the drawer
    /// drops and on [`Drawer::save_pipeline_cache`] (an owner that never
    /// drops the drawer calls it now and then), so pipelines compiled in
    /// earlier runs are not compiled again. Returns the file, or `None`
    /// where the device has no pipeline cache (GL). A file from another
    /// driver version, or unreadable, is replaced once a pipeline is made. Pipelines made before this call are not in the cache.
    pub fn set_pipeline_cache_dir(
        &mut self,
        dir: &std::path::Path,
    ) -> std::io::Result<Option<std::path::PathBuf>> {
        let Some(key) = &self.cache_key else {
            return Ok(None);
        };
        std::fs::create_dir_all(dir)?;
        let path = dir.join(key);
        let data = match std::fs::read(&path) {
            Ok(d) => Some(d),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        let create = |data: Option<&[u8]>| {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            // SAFETY: the data was written by `get_data` (from the file named
            // by this adapter's cache key); wgpu checks its header, version
            // and adapter and falls back to an empty cache for a stale one.
            let cache = unsafe {
                self.device
                    .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("pipelines"),
                        data,
                        fallback: true,
                    })
            };
            block_on(scope.pop()).is_none().then_some(cache)
        };
        // wgpu refuses data for another adapter even with `fallback` (a
        // copied file): start empty then.
        let cache = create(data.as_deref())
            .or_else(|| create(None))
            .ok_or_else(|| std::io::Error::other("could not create a pipeline cache"))?;
        self.disk_cache = Some(DiskCache {
            cache,
            path: path.clone(),
            unsaved: 0,
        });
        Ok(Some(path))
    }

    /// Writes the pipeline cache file, if there is one (see
    /// [`Drawer::set_pipeline_cache_dir`]) and pipelines were created since
    /// it was last written, so it is cheap to call often; through a
    /// temporary file of its
    /// own so a crash, or another process saving to the same directory,
    /// leaves a whole file (the last one renamed wins).
    pub fn save_pipeline_cache(&mut self) -> std::io::Result<()> {
        static SAVES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let Some(c) = &mut self.disk_cache else {
            return Ok(());
        };
        if c.unsaved == 0 {
            return Ok(());
        }
        c.unsaved = 0;
        let Some(data) = c.cache.get_data() else {
            return Ok(());
        };
        let n = SAVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = c
            .path
            .with_extension(format!("{}-{n}.tmp", std::process::id()));
        let r = std::fs::write(&tmp, data).and_then(|()| std::fs::rename(&tmp, &c.path));
        if r.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        r
    }

    /// The pipeline cache, for pipeline descriptors.
    fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.disk_cache.as_ref().map(|c| &c.cache)
    }

    /// Marks that move when a draw takes a slow path: pipelines (any kind),
    /// EDRAM transfers, targets, shader translations and modules made. A
    /// sampled draw that moves them is left out of the sampled splits, as
    /// scaling a one-off compile by [`PART_SAMPLE`] would swamp them; that
    /// time is exact in `pipeline` and the phase totals.
    fn slow_work(&self) -> (Duration, u32, u64, usize, usize) {
        (
            self.timings.pipeline,
            self.timings.transfers,
            self.targets_made,
            self.translated.len(),
            self.modules.len(),
        )
    }

    /// Counts a pipeline created since `t0` and writes the cache file every
    /// [`PIPELINE_SAVE_EVERY`] new ones.
    fn pipeline_made(&mut self, t0: Instant) {
        self.timings.pipeline += t0.elapsed();
        let Some(c) = &mut self.disk_cache else {
            return;
        };
        c.unsaved += 1;
        if c.unsaved >= PIPELINE_SAVE_EVERY {
            if let Err(e) = self.save_pipeline_cache() {
                eprintln!("[draw] pipeline cache not saved: {e}");
            }
        }
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
        let pairs: HashSet<(u64, u64)> = self.pipelines.keys().map(|k| (k.vs, k.ps)).collect();
        let wgsl: Vec<usize> = self
            .translated
            .values()
            .flatten()
            .map(|t| t.wgsl.len())
            .collect();
        format!(
            "draw: textures {} MB in {}, targets {} MB in {}, {} dropped, {} made, {} reused, {} spare ({} MB), kept resolves {} MB in {}, arena {} MB, staged {} MB, pipelines {} ({} vs/ps pairs, {} dropped), shaders {}, translated {} ({} KB WGSL)",
            self.texture_cache_bytes >> 20,
            self.texture_cache.len(),
            rt >> 20,
            self.targets.len(),
            self.target_drops,
            self.targets_made,
            self.targets_reused,
            self.spare_targets.items.len(),
            self.spare_targets.bytes >> 20,
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
                .min_by_key(|(_, (_, used, _))| *used)
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

    /// Width and height of render target `key` in guest pixels (its host
    /// texture is [`Drawer::scale`] times that).
    pub fn render_target_size(&self, key: RenderTargetKey) -> Option<(u32, u32)> {
        let s = self.scale;
        self.targets
            .get(&key)
            .map(|t| (t.width() / s, t.height() / s))
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
        let start = Instant::now();
        let was_watching = self.watching;
        match mem.take_written() {
            Some(written) if self.texture_memo.len() < TEXTURE_MEMO_CAP => {
                self.watching = true;
                if was_watching {
                    // Kept resolve blocks hashed against guest memory stay
                    // checked while nothing writes them: no hash again.
                    for m in &mut self.mirrors {
                        for (k, c) in m.checked.iter_mut().enumerate() {
                            *c &= m.hashes[k].is_some();
                        }
                    }
                }
                // One pass over the memos for all of them: a frame lists
                // thousands of written pages.
                let written = merge_ranges(written);
                self.forget_guest_in(&written);
                for &(lo, hi) in &written {
                    self.uncheck(lo, hi);
                }
            }
            _ => {
                self.watching = false;
                self.texture_memo.clear();
                self.texture_memo_epoch += 1;
                self.index_memo.clear();
                self.index_memo_bytes = 0;
                // Windows staged by a watched replay: anything may have
                // changed since.
                self.wrap();
            }
        }
        self.replays += 1;
        if !(was_watching && self.watching) {
            for m in &mut self.mirrors {
                m.checked.fill(false);
            }
        }
        let mut mem = Tracked {
            mem,
            written: Vec::new(),
        };
        self.timings.edges += start.elapsed();
        if let Some((before, w)) = self.late.take() {
            self.end_writeback(&mut mem, w, Some(before))?;
        }
        let done = self.replay_commands(commands, &mut mem, &mut report);
        // Submitted on an error too: the batch can hold resolve clears of
        // targets that outlive the failed replay, and the first error wins.
        let flushed = self.flush().map_err(|e| match e {
            Skip::Fatal(e) => e,
            Skip::Unsupported(why) => Error::Gpu(why),
        });
        let done = done.and(flushed);
        self.batch = None;
        self.pre = None;
        let t = Instant::now();
        let replay = self.replays;
        (self.texture_groups).retain(|_, &mut (_, used)| used + 1 >= replay);
        self.timings.edges += t.elapsed();
        if !self.watching || done.is_err() {
            self.wrap();
        }
        // Resolves done before an error still reach guest memory.
        let kept = self.finish_mirrors(&mut mem);
        done?;
        kept?;
        let t = Instant::now();
        self.drop_targets(&mut report);
        // Views and resolve groups hold their textures and buffers: only
        // those of kept targets and kept resolves stay.
        let targets = &self.targets;
        let live = |tex: &wgpu::Texture| targets.values().any(|t| t == tex);
        self.target_views.retain(|t, _| live(t));
        for m in &mut self.mirrors {
            m.groups.retain(|(tex, _), _| live(tex));
        }
        self.timings.edges += t.elapsed();
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
        // Two clock reads per draw, resolve and swap, none for the commands
        // between them: a run of those is charged once, before the next
        // timed command (and at the end).
        let mut clock = PlainTimer::new(Instant::now);
        for c in commands {
            let mem = &mut *tracked;
            // Draws, resolves and swaps time their own phases; the rest,
            // and the upkeep below, count as `commands`.
            let own = matches!(
                c,
                Command::Draw(_) | Command::Resolve(_) | Command::Swap { .. }
            );
            if own {
                clock.charge(&mut self.timings.commands);
            }
            let done = match c {
                Command::WriteRegisters { base, values } => {
                    self.regs.set_many(*base, values);
                    let end = base.saturating_add(values.len() as u32);
                    if CONSTS_REGS.iter().any(|r| *base < r.end && r.start < end) {
                        self.consts_gen += 1;
                    }
                    for (stage, written) in self.float_gens.iter_mut().enumerate() {
                        let lo = regs::SHADER_CONSTANT_FLOAT + FLOAT_STAGE_REGS * stage as u32;
                        if *base < lo + FLOAT_STAGE_REGS && lo < end {
                            *written += 1;
                        }
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
                    self.timings.shader_loads += 1;
                    let p = Some(Program {
                        hash: hash_words(words),
                        words: words.as_slice().into(),
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
            if own {
                clock.restart();
            }
            // Per command, not per run: a MEM_WRITE's `sync` must not find a
            // kept copy an earlier write in the run made stale, and errors
            // stop the replay at the command they came after.
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
        clock.charge(&mut self.timings.commands);
        Ok(())
    }

    /// Drops the textures looked up this replay from guest bytes in
    /// `[lo, hi)`, which just changed.
    fn forget_guest(&mut self, lo: u64, hi: u64) {
        self.forget_guest_in(&[(lo, hi)]);
    }

    /// [`Self::forget_guest`] for every range of `ranges`, which
    /// [`merge_ranges`] made.
    fn forget_guest_in(&mut self, ranges: &[(u64, u64)]) {
        if ranges.is_empty() {
            return;
        }
        let mut gone = Vec::new();
        self.texture_memo.retain(|c, &mut (_, a, b, _)| {
            let keep = !meets(ranges, a, b);
            if !keep {
                gone.push(*c);
            }
            keep
        });
        // The sets that used those textures; the others stay bound.
        for c in gone {
            for flag in self.texture_set_flags.remove(&c).into_iter().flatten() {
                flag.store(false, Ordering::Relaxed);
            }
        }
        self.window_memo
            .retain(|&a, &mut (b, _)| !meets(ranges, a, b));
        self.index_memo
            .retain(|_, &mut (a, b, _)| !meets(ranges, a, b));
    }

    /// Marks the kept resolve blocks over `[lo, hi)` unchecked: guest
    /// memory there changed, so the next use hashes it again.
    fn uncheck(&mut self, lo: u64, hi: u64) {
        for m in self.mirrors.iter_mut().filter(|m| lo < m.hi && m.lo < hi) {
            let k0 = (lo.max(m.lo) - m.lo) / KEPT_BLOCK;
            let k1 = (hi.min(m.hi) - m.lo).div_ceil(KEPT_BLOCK);
            m.checked[k0 as usize..k1 as usize].fill(false);
        }
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
        match self.start_writeback(mem, lo, hi)? {
            Some(w) => self.end_writeback(mem, w, None),
            None => Ok(()),
        }
    }

    /// Queues copies out of every kept resolve over `[lo, hi)` with
    /// unwritten bytes, for [`Drawer::end_writeback`].
    fn start_writeback(
        &mut self,
        mem: &mut Tracked,
        lo: u64,
        hi: u64,
    ) -> Result<Option<WriteBack>, Error> {
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
            return Ok(None);
        };
        if self.resolve_pending {
            self.flush().map_err(|e| match e {
                Skip::Fatal(e) => e,
                Skip::Unsupported(why) => Error::Gpu(why),
            })?;
        }
        // No wait for idle first: the copies are queued after the work
        // that wrote the buffers, and the first map waits for them.
        self.timings.writebacks += 1;
        let index = self.submit_now([enc.finish()]);
        self.submitted(Site::Other);
        Ok(Some((index, copies)))
    }

    /// Hashes of guest memory under each copy's dirty ranges, to tell at
    /// a late write-back whether the CPU wrote there meanwhile.
    fn guest_hashes(&mut self, mem: &Tracked, w: &WriteBack) -> Vec<Vec<Option<u64>>> {
        w.1.iter()
            .map(|(_, dirty, _, _)| {
                dirty
                    .iter()
                    .map(|&(a, b)| {
                        mem.read_bytes(a as u32, (b - a) as usize)
                            .map(|b| self.block_hash(&b))
                    })
                    .collect()
            })
            .collect()
    }

    /// Waits for the copies of `w` and writes them to guest memory. With
    /// `before` (a late write-back: the guest hashes when it was queued),
    /// a copy whose ranges the CPU has written since is dropped with its
    /// kept resolve.
    fn end_writeback(
        &mut self,
        mem: &mut Tracked,
        w: WriteBack,
        before: Option<Vec<Vec<Option<u64>>>>,
    ) -> Result<(), Error> {
        let now = before.as_ref().map(|_| self.guest_hashes(mem, &w));
        let (index, copies) = w;
        let mut t = Instant::now();
        let mut first = true;
        for (k, (m_lo, dirty, spans, buf)) in copies.into_iter().enumerate() {
            if let (Some(before), Some(now)) = (&before, &now) {
                if before[k].iter().any(Option::is_none) || before[k] != now[k] {
                    self.mirrors.retain(|m| m.lo != m_lo);
                    continue;
                }
            }
            let data = self.map_read(&buf, index.clone())?;
            if std::mem::take(&mut first) {
                self.timings.gpu_wait += lap(&mut t);
            } else {
                self.timings.readback += lap(&mut t);
            }
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
    /// With `late_writeback`, the copies are only queued, and the next
    /// replay writes them to guest memory when it starts.
    fn finish_mirrors(&mut self, mem: &mut Tracked) -> Result<(), Error> {
        if self.skip_resolve_writeback {
            return Ok(());
        }
        if self.late_writeback {
            self.late = self
                .start_writeback(mem, 0, u64::MAX)?
                .map(|w| (self.guest_hashes(mem, &w), w));
            return Ok(());
        }
        self.sync(mem, 0, u64::MAX)
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
            views: HashMap::default(),
            groups: HashMap::default(),
        });
        Ok(Some(self.mirrors.len() - 1))
    }

    /// Counts one queue submission, made by `site`.
    fn submitted(&mut self, site: Site) {
        self.timings.submits += 1;
        self.timings.submits_by_site[site as usize] += 1;
    }

    /// Submits the recorded texture decodes and the batched draws, after
    /// writing the newly staged bytes to the arena. The windows and
    /// constants staged so far stay in the arena for later batches.
    fn flush(&mut self) -> Result<(), Skip> {
        // The arena upload counts as submit time too.
        let t = Instant::now();
        // A few batches in flight, so their staged copies stay few; the
        // drawer records on while the submit thread catches up. With one,
        // flushes close together waited on each other's submit.
        self.submitter.drain_to(self.submits_in_flight);
        let mut upload = None;
        if let (Some(arena), false) = (&self.arena, self.staged.is_empty()) {
            // The bytes go to the submit thread, which writes them to the
            // arena ahead of this batch and hands the buffer back: no copy
            // here, and `staged` keeps a reserve of the same size.
            let next = self.submitter.spare(self.staged.capacity());
            let bytes = std::mem::replace(&mut self.staged, next);
            self.staged_base += bytes.len() as u64;
            upload = Some((arena.clone(), self.staged_base - bytes.len() as u64, bytes));
        }
        self.timings.upload += t.elapsed();
        self.resolve_pending = false;
        let (pre, batch) = (self.pre.take(), self.batch.take());
        if pre.is_none() && batch.is_none() && upload.is_none() {
            let e = t.elapsed();
            self.timings.submit += e;
            self.timings.queue += e;
            return self.take_submitted();
        }
        let site = if pre.is_some() && batch.is_none() {
            Site::Decode
        } else {
            Site::Other
        };
        // Decodes first: the draws sample what they write, and they read
        // kept resolves written by work submitted before.
        let clock = &self.gpu_clock;
        let encs = pre
            .into_iter()
            .chain(batch.map(|mut b| {
                drop(b.pass.take());
                if let Some(c) = clock.as_ref().filter(|c| !c.kinds.is_empty()) {
                    let n = c.kinds.len() as u64;
                    b.enc
                        .resolve_query_set(&c.set, 0..2 * n as u32, &c.resolve, 0);
                    b.enc
                        .copy_buffer_to_buffer(&c.resolve, 0, &c.read, 0, 16 * n);
                }
                b.enc
            }))
            .collect();
        self.submitter.send(upload, encs);
        self.submitted(site);
        // One reading for both, so `queue` never exceeds `submit`.
        let e = t.elapsed();
        self.timings.submit += e;
        self.timings.queue += e;
        if self.gpu_clock.as_ref().is_some_and(|c| !c.kinds.is_empty()) {
            self.submitter.drain();
            let index = self
                .submitter
                .state
                .lock()
                .expect("submit state")
                .last
                .clone();
            self.read_gpu_clock(index.expect("submitted"))?;
        }
        self.take_submitted()
    }

    /// Submits now, after the batches the submit thread holds.
    fn submit_now(
        &self,
        bufs: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.submitter.drain();
        self.queue.submit(bufs)
    }

    /// Adds the submit thread's time so far to the timings, and fails on
    /// the first validation error it met.
    fn take_submitted(&mut self) -> Result<(), Skip> {
        let (finish, busy, error) = self.submitter.take();
        self.timings.finish_thread += finish;
        self.timings.submit_thread += busy;
        match error {
            Some(e) => Err(Skip::Fatal(Error::Gpu(e))),
            None => Ok(()),
        }
    }

    /// Waits for the timestamps of the batch just submitted (`index`) and
    /// adds its passes' GPU time to the timings. The wait counts as GPU
    /// wait; it also stops submits from queueing up behind the GPU.
    fn read_gpu_clock(&mut self, index: wgpu::SubmissionIndex) -> Result<(), Skip> {
        let Some(kinds) = self
            .gpu_clock
            .as_mut()
            .filter(|c| !c.kinds.is_empty())
            .map(|c| std::mem::take(&mut c.kinds))
        else {
            return Ok(());
        };
        let t = Instant::now();
        let c = self.gpu_clock.as_ref().expect("clock");
        let data = self.map_read(&c.read, index).map_err(Skip::Fatal)?;
        let period = f64::from(c.period);
        for (i, &kind) in kinds.iter().enumerate() {
            let tick = |j: usize| u64::from_le_bytes(data[8 * j..][..8].try_into().unwrap());
            let ns = tick(2 * i + 1).saturating_sub(tick(2 * i)) as f64 * period;
            self.timings.gpu_passes[kind as usize] += Duration::from_nanos(ns as u64);
        }
        self.timings.passes_timed += kinds.len() as u32;
        self.timings.gpu_wait += t.elapsed();
        Ok(())
    }

    /// Forgets what the arena holds, to stage from its start again: the
    /// staged windows and constants are no longer bound by later draws.
    /// Only after a flush, or once nothing will be submitted.
    fn wrap(&mut self) {
        // Index bytes staged before are overwritten from here.
        self.arena_epoch += 1;
        self.staged.clear();
        self.staged_base = 0;
        self.window_memo.clear();
        self.staged_consts = None;
        self.staged_floats = [None; 2];
    }

    /// Arena bytes a part of `len` bytes takes when [`Self::push`]ed: its
    /// size, then padding to the next part's alignment.
    fn staged_size(&self, len: u64) -> u64 {
        len.max(4)
            .next_multiple_of(4)
            .next_multiple_of(self.arena_align)
    }

    /// Makes room for `need` bytes of parts (see [`Self::staged_size`]) in
    /// the batch's staging area: submits the batch and wraps the arena
    /// first when it is full, and grows the arena when one draw does not
    /// fit. A grown arena is a new buffer: submitted draws keep the old one.
    fn reserve(&mut self, need: u64) -> Result<(), Skip> {
        let have = self.arena.as_ref().map_or(0, |b| b.size());
        if self.staged_base + self.staged.len() as u64 + need > have {
            self.flush()?;
            self.wrap();
            // A watched replay grows an arena made before it to keep windows.
            if need > have || (self.watching && have < ARENA_WATCHED.min(self.max_buffer)) {
                if need > self.max_buffer {
                    return unsupported(format!("{need} bytes of draw data"));
                }
                let min = if self.watching {
                    ARENA_WATCHED
                } else {
                    ARENA_MIN
                };
                let size = need.max(min).next_power_of_two().min(self.max_buffer);
                // Flushed, so empty: the batch never stages more than the
                // arena holds, so it never grows past this (a Vec doubling
                // from an odd size would).
                self.staged = Vec::with_capacity(size.min(STAGED_RESERVE) as usize);
                self.arena_groups.clear();
                self.clear_group = None;
                self.mirrors.iter_mut().for_each(|m| m.groups.clear());
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
        Ok(())
    }

    /// Appends a part of `len` bytes, which `fill` writes over zeros, to
    /// the staging area at an offset aligned for uniform and storage
    /// bindings, and returns its (offset, size) in the arena; `None`, with
    /// nothing appended, when `fill` fails. Only after [`Self::reserve`]
    /// made room for it.
    fn push(&mut self, len: usize, fill: impl FnOnce(&mut [u8]) -> bool) -> Option<(u64, u64)> {
        let off = self.staged.len();
        let size = (len as u64).max(4).next_multiple_of(4);
        self.staged.resize(off + size as usize, 0);
        if !fill(&mut self.staged[off..off + len]) {
            self.staged.truncate(off);
            return None;
        }
        let end = (off as u64 + size).next_multiple_of(self.arena_align);
        self.staged.resize(end as usize, 0);
        Some((self.staged_base + off as u64, size))
    }

    /// [`Self::push`] of `bytes`.
    fn push_bytes(&mut self, bytes: &[u8]) -> (u64, u64) {
        self.push(bytes.len(), |b| {
            b.copy_from_slice(bytes);
            true
        })
        .expect("bytes staged")
    }

    /// Stages `words` (a clear value or resolve parameters) for a uniform
    /// binding of the arena at a dynamic offset, and returns the offset.
    /// Submits the batch first when the arena is full, like a draw.
    fn stage_uniform(&mut self, words: &[u32]) -> Result<u32, Skip> {
        self.reserve(self.staged_size(4 * words.len() as u64))?;
        let (offset, _) = (self.push(4 * words.len(), |b| {
            words_into(words, b);
            true
        }))
        .expect("uniform staged");
        u32::try_from(offset)
            .map_err(|_| Skip::Unsupported("draw data past 4 GiB of the arena".into()))
    }

    /// Whether `need` bytes of parts can be staged without submitting the
    /// batch first.
    fn fits(&self, need: u64) -> bool {
        self.staged_base + self.staged.len() as u64 + need
            <= self.arena.as_ref().map_or(0, |b| b.size())
    }

    /// The batch's render pass, begun on `key`'s targets unless the current
    /// pass already draws to them.
    /// Whether the batch's render pass is open on `key`'s targets.
    fn pass_open(&self, key: &PassKey) -> bool {
        (self.batch.as_ref())
            .and_then(|b| b.pass.as_ref())
            .is_some_and(|p| p.0 == *key)
    }

    /// The textures of a draw's colour and depth targets, made current
    /// (see [`Self::target`]).
    #[allow(clippy::type_complexity)]
    fn targets_of(
        &mut self,
        (color, depth): PassKey,
        rows: u32,
        msaa: u32,
        report: &mut Report,
    ) -> Result<([Option<wgpu::Texture>; 4], Option<wgpu::Texture>), Skip> {
        let mut color_tex: [Option<wgpu::Texture>; 4] = Default::default();
        for rt in 0..4 {
            color_tex[rt] = color[rt]
                .map(|k| self.target(k, rows, msaa, report))
                .transpose()?;
        }
        let depth_tex = depth
            .map(|k| self.target(k, rows, msaa, report))
            .transpose()?;
        Ok((color_tex, depth_tex))
    }

    fn pass(
        &mut self,
        key: PassKey,
        color: &[Option<wgpu::Texture>; 4],
        depth: Option<&wgpu::Texture>,
    ) -> (&mut wgpu::RenderPass<'static>, &mut Bound) {
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
            self.timings.render_passes += 1;
            let timestamp_writes = GpuClock::render(&mut self.gpu_clock, 0);
            let pass = batch
                .enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("draw"),
                    timestamp_writes,
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
            batch.pass = Some((key, pass, Bound::default()));
        }
        let (_, pass, bound) = batch.pass.as_mut().expect("pass begun");
        (pass, bound)
    }

    fn check_uncaptured(&self) -> Result<(), Error> {
        use std::sync::atomic::Ordering;
        if !self.uncaptured_any.swap(false, Ordering::Acquire) {
            return Ok(());
        }
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
    /// Coordinates are guest pixels; at a [`Drawer::scale`] above 1 each
    /// is decimated from its host pixels ([`formats::decimate`]).
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
        self.read_target(key.format, tex, x, y, w, h)
    }

    /// [`Self::read_texture`] of a render target texture of `format` in
    /// guest pixels, decimated from the host pixels at a scale above 1.
    fn read_target(
        &self,
        format: TargetFormat,
        tex: &wgpu::Texture,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, Error> {
        let s = self.scale;
        if s == 1 {
            return self.read_texture(tex, x, y, w, h);
        }
        let (gw, gh) = (tex.width() / s, tex.height() / s);
        if x.saturating_add(w) > gw || y.saturating_add(h) > gh {
            return Err(Error::Unsupported(format!(
                "readback {w}x{h} at ({x}, {y}) outside the {gw}x{gh} target"
            )));
        }
        let host = self.read_texture(tex, x * s, y * s, w * s, h * s)?;
        Ok(formats::decimate(format, &host, w, h, s))
    }

    /// Waits for all submitted GPU work, bounded by the drawer's timeout.
    fn wait_idle(&self) -> Result<(), Error> {
        self.submitter.drain();
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
        let index = self.submit_now([enc.finish()]);
        let data = self.map_read(&buf, index)?;
        let mut out = Vec::with_capacity((row * h) as usize);
        for r in 0..h as usize {
            out.extend_from_slice(&data[r * padded as usize..][..row as usize]);
        }
        Ok(out)
    }

    /// Waits for submission `index` and copies out the mapped `buf`.
    fn map_read(&self, buf: &wgpu::Buffer, index: wgpu::SubmissionIndex) -> Result<Vec<u8>, Error> {
        self.submitter.drain();
        self.map_submitted(buf, index)
    }

    /// [`Drawer::map_read`] of a buffer that no batch still queued on the
    /// submit thread uses, so without waiting for those.
    fn map_submitted(
        &self,
        buf: &wgpu::Buffer,
        index: wgpu::SubmissionIndex,
    ) -> Result<Vec<u8>, Error> {
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
        let s = self.scale;
        if key.pitch == 0 || key.pitch > self.max_dim / s {
            return unsupported(format!("surface pitch {}", key.pitch));
        }
        if key.format.is_snorm16() && !self.snorm16 {
            return unsupported(format!(
                "{:?} render target: the host cannot render into 16-bit snorm",
                key.format
            ));
        }
        let tex = match self.targets.get(&key) {
            Some(t) => t.clone(),
            None => {
                let format = self.host_format(key.format);
                let (width, height) = (key.pitch * s, self.target_rows() * s);
                let spare = (format, width, height, 1, TARGET_USAGE);
                let tex = if let Some(t) = self.spare_targets.take(&spare) {
                    // A new target starts zeroed, as wgpu makes it.
                    self.clear_target(&t);
                    self.targets_reused += 1;
                    t
                } else {
                    self.targets_made += 1;
                    self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("render target"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: TARGET_USAGE,
                        view_formats: &[],
                    })
                };
                self.targets.insert(key, tex.clone());
                tex
            }
        };
        let own = self.written.get(&key).copied().unwrap_or_default();
        let range = edram_range(key, rows.max(own.rows), msaa);
        let mut newest: Option<(RenderTargetKey, Written)> = None;
        // Stamps come from `clock`, so a target stamped at the clock has no
        // newer one to find: the scan below would skip every entry. Most
        // draws go to the target drawn to last (4 ms a frame of scans).
        let others = if own.stamp < self.clock {
            self.written.len()
        } else {
            0
        };
        for (&k, &w) in self.written.iter().take(others) {
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
                let rows = (w.rows * src_down)
                    .div_ceil(dst_down)
                    .min(self.target_rows());
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

    /// Records a clear of every texel of `tex` to zero in the batch, as a
    /// new texture holds.
    fn clear_target(&mut self, tex: &wgpu::Texture) {
        let view = tex.create_view(&Default::default());
        let f = tex.format();
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })];
        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            pass: None,
            enc: device.create_command_encoder(&Default::default()),
        });
        batch.pass = None;
        let depth = f.is_depth_stencil_format();
        batch.enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear reused target"),
            color_attachments: if depth { &[] } else { &color_attachment },
            depth_stencil_attachment: depth.then(|| wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: f.has_depth_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: f.has_stencil_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            ..Default::default()
        });
    }

    /// Drops the host textures of render targets that a newer target on
    /// the same EDRAM tiles, one their words transfer to, has written all
    /// the rows of, with no newer alias whose words do not transfer:
    /// their next use takes that target's words (or newer ones) as it
    /// would without the drop; their textures are kept as spares for new
    /// targets. Then frees the oldest spares past the cap on targets and
    /// spares together, and past it on targets alone drops the
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
            if let Some(t) = self.targets.remove(&k) {
                self.spare_targets.put(spare_key(&t), texture_bytes(&t), t);
            }
            self.transfer_setups.retain(|t, _| t.0 != k && t.2 != k);
            self.target_drops += 1;
        }
        let mut total: u64 = self.targets.values().map(texture_bytes).sum();
        // The spares count against the cap and go first; targets dropped
        // below are freed, not kept.
        let room = self.target_cap.saturating_sub(total);
        self.spare_targets.trim(usize::MAX, room);
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
            self.transfer_setups.retain(|t, _| t.0 != k && t.2 != k);
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
        self.stamp_written(key, rows, msaa);
    }

    /// [`Self::mark_written`] at the current `clock`, for the targets of one
    /// draw.
    fn stamp_written(&mut self, key: RenderTargetKey, rows: u32, msaa: u32) {
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
        let cache = self.pipeline_cache();
        let t0 = Instant::now();
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
                cache,
            })
        })?;
        self.pipeline_made(t0);
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
        // Host rows: the pass works on host pixels.
        let rows = (rows * self.scale).min(dst_tex.height());
        if rows == 0 {
            return Ok(());
        }
        // Recorded in the batch after its draws, like a clear: its render
        // pass ends theirs, so the source's draws (and reads of `dst`) land
        // before it and later draws after it, with no submit of its own.
        let setup = self.transfer_setup((src, src_msaa), (dst, dst_msaa), dst_tex)?;
        let (view, to_depth) = (&setup.view, setup.to_depth);
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view,
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
        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            pass: None,
            enc: device.create_command_encoder(&Default::default()),
        });
        batch.pass = None;
        {
            let mut pass = batch.enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transfer"),
                color_attachments: if to_depth { &[] } else { &color_attachment },
                depth_stencil_attachment: to_depth.then(|| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(keep),
                        stencil_ops: setup.stencil.then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                ..Default::default()
            });
            pass.set_scissor_rect(0, 0, dst_tex.width(), rows);
            for (i, (p, g)) in setup.pipelines.iter().zip(&setup.groups).enumerate() {
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                pass.set_stencil_reference(if i == 0 { 0 } else { 0xFF });
                pass.draw(0..3, 0..1);
            }
        }
        self.timings.transfers += 1;
        self.timings.transfer_passes += setup.pipelines.len() as u32;
        Ok(())
    }

    /// The pipelines, bind groups and `dst` view of a transfer from `src`
    /// to `dst`, made once per pair of targets and kept until a target is
    /// dropped (their textures go with it).
    fn transfer_setup(
        &mut self,
        (src, src_msaa): (RenderTargetKey, u32),
        (dst, dst_msaa): (RenderTargetKey, u32),
        dst_tex: &wgpu::Texture,
    ) -> Result<Arc<TransferSetup>, Skip> {
        let key = (src, src_msaa, dst, dst_msaa);
        if let Some(setup) = self.transfer_setups.get(&key) {
            return Ok(setup.clone());
        }
        let src_tex = self.targets[&src].clone();
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
        // w, h (`src`), `src`'s pitch (guest pixels) from pitch and the host
        // pixels per guest pixel from px_scale; resolve.wgsl's depth_word
        // reads `src_format`. set_scale drops these setups.
        let mut params = [0u32; 12];
        (params[0], params[1]) = samples(dst_msaa);
        (params[2], params[3]) = samples(src_msaa);
        params[4] = src.pitch;
        params[5] = if src.format == TargetFormat::D24FS8 {
            6
        } else {
            5
        };
        params[11] = self.scale;
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
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        let setup = Arc::new(TransferSetup {
            pipelines,
            groups,
            view,
            to_depth,
            stencil: df.has_stencil_aspect(),
        });
        self.transfer_setups.insert(key, setup.clone());
        self.timings.transfer_setups += 1;
        Ok(setup)
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
    ) -> Result<Arc<Translated>, Skip> {
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
                xenon_shader::translate_with(&p.words, s, param_gen)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
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
            (false, _) if extra != 0 => drop_depth_wgsl(&t.wgsl),
            (false, _) => Ok(t.wgsl.clone()),
        }
        .and_then(|src| match variant {
            Variant::Output(c) if !vertex => {
                c.iter().enumerate().try_fold(src, |ps, (rt, c)| match c {
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
            _ => Ok(src),
        })
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
    /// far as `reach` reads with vertex indices in `indices` (see
    /// [`fetch_range`]), made ready to read (kept resolves under it written
    /// back). Ranges less than [`WINDOW_GAP`] apart share one part (all of
    /// them when some fetch reads past its buffer); the parts are staged
    /// one after the other as one window. Returns the
    /// parts' physical `[lo, hi)` (none for no window), the window base and
    /// each constant's rebase words (see the shader binding contract); or,
    /// when the window is one part and a window staged in the arena holds
    /// it, no parts, that window's base and its memo entry, with this
    /// window's size.
    #[allow(clippy::type_complexity)]
    fn guest_window(
        &mut self,
        mask: u128,
        indices: Option<(u32, u32)>,
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<
        (
            Vec<(u64, u64)>,
            u32,
            [i32; 96],
            Option<((u64, u64, (u64, u64)), u64)>,
        ),
        Skip,
    > {
        let mut scratch = std::mem::take(&mut self.window_scratch);
        let window = self.guest_window_in(&mut scratch, mask, indices, mem, report);
        self.window_scratch = scratch;
        window
    }

    /// [`Self::guest_window`] with `scratch.reach` as the reach.
    #[allow(clippy::type_complexity)]
    fn guest_window_in(
        &mut self,
        scratch: &mut WindowScratch,
        mask: u128,
        indices: Option<(u32, u32)>,
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<
        (
            Vec<(u64, u64)>,
            u32,
            [i32; 96],
            Option<((u64, u64, (u64, u64)), u64)>,
        ),
        Skip,
    > {
        let WindowScratch {
            reach,
            ranges,
            part_of,
            at,
        } = scratch;
        ranges.clear();
        part_of.clear();
        at.clear();
        let mut clipped = false;
        for k in (0..96).filter(|k| mask & (1u128 << k) != 0) {
            let vf = self.regs.vertex_fetch(k);
            if vf.fetch_type != FetchConstantType::Vertex {
                report.warn(format!("vertex fetch constant {k} is not a vertex fetch"));
            }
            // Physical, as the translated shader masks fetch addresses.
            let a = u64::from(vf.address & PHYSICAL_MASK);
            let (lo, hi, c) = fetch_range(a, vf.size_dwords, k, reach, indices);
            clipped |= c;
            if lo < hi {
                ranges.push((lo, hi, k));
            }
        }
        ranges.sort_unstable();
        let mut parts: Vec<(u64, u64)> = Vec::new();
        // A read past its constant's buffer must find the guest's word or 0,
        // not another part's, so such a draw stages one span with no rebase.
        let gap = if clipped { u64::MAX } else { WINDOW_GAP };
        for &(lo, hi, k) in ranges.iter() {
            match parts.last_mut() {
                Some(p) if lo <= p.1.saturating_add(gap) => p.1 = p.1.max(hi),
                _ => parts.push((lo, hi)),
            }
            part_of.push((k, parts.len() - 1));
        }
        let mut rebase = [0; 96];
        let Some(&(lo, hi)) = parts.first() else {
            return Ok((parts, 0, rebase, None));
        };
        if parts.len() == 1 {
            let memo = self.window_memo.range(..=lo).next_back();
            if let Some((&a, &(b, s))) = memo.filter(|(_, (b, _))| hi <= *b) {
                return Ok((Vec::new(), a as u32, rebase, Some(((a, b, s), hi - lo))));
            }
        }
        // Where each part lands in the window, as `stage` lays them out.
        let mut next = 0;
        for &(lo, hi) in &parts {
            at.push(next);
            next += (hi - lo)
                .max(4)
                .next_multiple_of(4)
                .next_multiple_of(self.arena_align);
        }
        let (last_lo, last_hi) = parts[parts.len() - 1];
        let size = at[at.len() - 1] + last_hi - last_lo;
        if size > self.max_storage {
            return unsupported(format!(
                "vertex data window of {size} bytes is larger than the storage buffer limit"
            ));
        }
        for &(k, p) in part_of.iter() {
            rebase[k as usize] = ((lo + at[p]) as i64 - parts[p].0 as i64) as i32 / 4;
        }
        for &(lo, hi) in &parts {
            self.sync(mem, lo, hi)?;
        }
        Ok((parts, lo as u32, rebase, None))
    }

    /// The texture bind group of `textures` at the slots of `mask`, from
    /// [`Drawer::texture_groups`] or made and kept there.
    fn texture_group(
        &mut self,
        tex_mask: u32,
        textures: Vec<(wgpu::TextureView, wgpu::Sampler)>,
        g1_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        match self.texture_groups.entry((tex_mask, textures)) {
            Entry::Occupied(mut g) => {
                self.timings.texture_groups_reused += 1;
                g.get_mut().1 = self.replays;
                g.get().0.clone()
            }
            Entry::Vacant(v) => {
                let slots = (0..32).filter(|n| tex_mask & (1 << n) != 0);
                let mut entries = Vec::new();
                for (n, (view, sampler)) in slots.zip(&v.key().1) {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 * n,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 * n + 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
                let g = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("draw group 1"),
                    layout: g1_layout,
                    entries: &entries,
                });
                v.insert((g, self.replays)).0.clone()
            }
        }
    }

    /// The compute pipeline of `resolve.wgsl`'s `decode_8888`, or with
    /// `float` its `decode_float`, made at first use.
    fn decode_pipeline(&mut self, float: bool) -> wgpu::ComputePipeline {
        let device = &self.device;
        let cache = self.disk_cache.as_ref().map(|c| &c.cache);
        let entry = ["decode_8888", "decode_float"][usize::from(float)];
        // Set once the shader module is made, so it times the pipeline alone.
        let mut t0 = None;
        let pipeline = self.decode_pipelines[usize::from(float)]
            .get_or_insert_with(|| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("resolve"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("resolve.wgsl").into()),
                });
                t0 = Some(Instant::now());
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache,
                })
            })
            .clone();
        if let Some(t0) = t0 {
            self.pipeline_made(t0);
        }
        pipeline
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

    /// Texture and sampler for fetch constant slot `n`. A slot without a
    /// texture fetch constant, or whose texture cannot be read or decoded,
    /// gets a 1x1 texture of zeros (as Xenia binds its null image) and a
    /// warning, so the draw still happens. `filters`: the fetch
    /// instruction's filter overrides of the slot.
    fn texture(
        &mut self,
        n: u32,
        filters: Option<[u8; 4]>,
        mem: &mut Tracked,
        report: &mut Report,
    ) -> Result<(wgpu::TextureView, wgpu::Sampler), Skip> {
        let fetch = xenon_texture::TextureFetch::decode(self.regs.fetch_constant(n));
        if fetch.fetch_type != 2 {
            report.warn("texture slot without a texture fetch constant drawn with zeros");
            return Ok((self.zero_texture.clone(), self.zero_sampler.clone()));
        }
        let sampler = self.sampler(&fetch, filters, report);
        let constant = self.regs.fetch_constant(n);
        if let Some((view, _, _, key)) = self.texture_memo.get(&constant) {
            let view = view.clone();
            // Kept in use, so eviction does not take it for an old one.
            if let Some((_, _, used)) = key.and_then(|k| self.texture_cache.get_mut(&k)) {
                *used = self.replays;
            }
            return Ok((view, sampler));
        }
        let addr = u64::from(fetch.base_address & PHYSICAL_MASK);
        if let Ok(len) = xenon_texture::bytes_needed(&fetch) {
            let end = addr + len as u64;
            if let Some((view, lo, hi)) = self.kept_texture(&fetch, constant, addr, end, mem)? {
                self.timings.textures_kept += 1;
                self.timings.kept_bytes += hi - lo;
                self.texture_memo
                    .insert(constant, (view.clone(), lo, hi, None));
                return Ok((view, sampler));
            }
            // One sync over the whole chain: at most one writeback wait.
            if let Some((lo, hi)) = chain_range(&fetch) {
                self.sync(mem, lo, hi)?;
            }
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
    /// k_8_8_8_8 layouts that host as `Rgba8Unorm`, k_32_FLOAT,
    /// k_16_16_16_16_FLOAT and the depth formats k_24_8 and k_24_8_FLOAT
    /// (any endian, swizzle and resolve format, as the bytes are the
    /// guest's) are decoded here. Mip levels before the packed tail, up to
    /// the fetch's last one, that the same kept resolve holds are decoded
    /// too, so explicit and computed LODs reach them. Returns the view and
    /// the guest range it was decoded from.
    fn kept_texture(
        &mut self,
        fetch: &xenon_texture::TextureFetch,
        constant: [u32; 6],
        addr: u64,
        end: u64,
        mem: &Tracked,
    ) -> Result<Option<(wgpu::TextureView, u64, u64)>, Skip> {
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
            (F::K24_8, H::Rgba32Float) => (Some(3), 4, 16),
            (F::K24_8, H::Rgba16Float) => (Some(4), 4, 8),
            (F::K24_8Float, H::Rgba32Float) => (Some(5), 4, 16),
            (F::K24_8Float, H::Rgba16Float) => (Some(6), 4, 8),
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
        // (width, height, pitch, guest address) of each level decoded.
        let mut levels = vec![(w, h, pitch, addr)];
        let (mut lo, mut hi) = (addr, end);
        // shortcut: the levels stop at the first one the kept resolve does
        // not hold (sampled LODs past it clamp to the last decoded level);
        // read those from guest memory if a capture keeps only some mips.
        for level in 1..=fetch.mip_range().1 {
            let Ok(m) = xenon_texture::mip_level(fetch, level) else {
                break;
            };
            let a = u64::from(fetch.mip_address & PHYSICAL_MASK) + m.offset as u64;
            let b = a + m.extent as u64;
            let mirror = &self.mirrors[i];
            if a < mirror.lo || mirror.hi < b || !(a - mirror.lo).is_multiple_of(4) {
                break;
            }
            if !self.verify(i, mem, a, b) {
                return Ok(None);
            }
            levels.push((m.width, m.height, m.pitch_blocks, a));
            // shortcut: one range spans the base, the mips and any gap between
            // them, so a resolve into the gap re-decodes the chain; keep a
            // range per level if mips far from their base show up.
            (lo, hi) = (lo.min(a), hi.max(b));
        }
        // Only the rows resolves wrote since the last decode are decoded
        // again (all levels with mips). With a resolve in the open batch the
        // decode is recorded after it, so the batch's earlier draws sample
        // the old texels and later ones the new; otherwise it runs before
        // the batch (`pre`). A chain with a new mip count gets a new
        // texture, decoded whole, which no recorded draw can sample yet.
        let old = self.mirrors[i]
            .views
            .get(&constant)
            .filter(|k| k.tex.mip_level_count() as usize == levels.len());
        let (y0, y1) = match old {
            Some(k) => match k.stale {
                None => return Ok(Some((k.view.clone(), k.lo, k.hi))),
                Some(rows) => rows,
            },
            None => (0, h),
        };
        let tex = match old {
            Some(k) => k.tex.clone(),
            None => self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("kept resolve texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: host,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }),
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let entry = ["decode_8888", "decode_float"][usize::from(mode.is_some())];
        let pipeline = self.decode_pipeline(mode.is_some());
        // After a resolve recorded in the batch, the decode runs after it:
        // with `decode_in_batch` (the default) it is recorded in the batch,
        // after the resolve, so the batch's earlier draws still sample the
        // old texels first; without, the batch is flushed and the decode
        // runs before the next one (`pre`).
        if self.resolve_pending && !self.decode_in_batch {
            self.flush()?;
        }
        let mut batch = match self.resolve_pending {
            true => self.batch.take(),
            false => None,
        };
        let mut pre = match batch {
            Some(ref mut b) => {
                b.pass = None;
                None
            }
            None => Some(
                self.pre
                    .take()
                    .unwrap_or_else(|| self.device.create_command_encoder(&Default::default())),
            ),
        };
        let enc = match (&mut batch, &mut pre) {
            (Some(b), _) => &mut b.enc,
            (None, Some(e)) => e,
            (None, None) => unreachable!("one encoder"),
        };
        let pipeline = &pipeline;
        for (level, &(lw, lh, lpitch, la)) in levels.iter().enumerate() {
            let (ly0, rows) = if level == 0 { (y0, y1 - y0) } else { (0, lh) };
            let row_words =
                (lw * host_bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) / 4;
            let params = [
                lw,
                rows,
                lpitch,
                fetch.endian as u32,
                u32::from(fetch.swizzle),
                ((la - self.mirrors[i].lo) / 4) as u32,
                row_words,
                ly0,
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
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(lw.div_ceil(8), rows.div_ceil(8), 1);
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
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: ly0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: lw,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
        }
        match batch {
            Some(b) => self.batch = Some(b),
            None => self.pre = pre,
        }
        if let Some(e) = block_on(scope.pop()) {
            return Err(Skip::Fatal(Error::Gpu(e.to_string())));
        }
        let view = tex.create_view(&Default::default());
        self.mirrors[i].views.insert(
            constant,
            Kept {
                tex,
                view: view.clone(),
                lo,
                hi,
                block: u64::from(pitch) * 32 * guest_bytes,
                stale: None,
            },
        );
        Ok(Some((view, lo, hi)))
    }

    /// Host texture of fetch constant slot `n`, from the cache or decoded.
    /// Formats whose values do not fit 8-bit unorm, and fetches with a signed
    /// component, upload as floats; unsigned DXT1..5 as BC blocks where
    /// [`bc_blocks`] takes them. The cache key holds the whole fetch
    /// constant (sign bits included), so signed and unsigned views of the
    /// same bytes are separate textures. The mip levels of
    /// [`xenon_texture::mip_fetches`] are read after the base (left out
    /// when [`reads_base`] is false), up to the first that cannot be read.
    fn guest_texture(
        &mut self,
        n: u32,
        fetch: &xenon_texture::TextureFetch,
        mem: &dyn GuestMemory,
        report: &mut Report,
    ) -> Result<wgpu::TextureView, String> {
        let addr = fetch.base_address & PHYSICAL_MASK;
        let base = reads_base(fetch);
        let len = match base {
            true => xenon_texture::bytes_needed(fetch).map_err(|e| e.to_string())?,
            false => 0,
        };
        let mut bytes = std::mem::take(&mut self.texture_scratch);
        bytes.resize(len, 0);
        let read = !base || mem.read_into(addr, &mut bytes);
        if read {
            for (a, b) in xenon_texture::mip_fetches(fetch).map_while(|m| level_range(&m)) {
                let at = bytes.len();
                bytes.resize(at + (b - a) as usize, 0);
                if !mem.read_into(a as u32, &mut bytes[at..]) {
                    bytes.truncate(at);
                    break;
                }
            }
        }
        let view = if read {
            self.texture_from(n, fetch, &bytes, report)
        } else {
            Err(xenon_texture::Error::TooShort {
                needed: len,
                got: 0,
            }
            .to_string())
        };
        self.texture_scratch = bytes;
        view
    }

    /// [`Self::guest_texture`] from the guest bytes `bytes` of `fetch`: the
    /// base level's when [`reads_base`], then each mip level's in turn.
    fn texture_from(
        &mut self,
        n: u32,
        fetch: &xenon_texture::TextureFetch,
        bytes: &[u8],
        report: &mut Report,
    ) -> Result<wgpu::TextureView, String> {
        let base = reads_base(fetch);
        let base_len = match base {
            true => xenon_texture::bytes_needed(fetch).map_err(|e| e.to_string())?,
            false => 0,
        };
        // shortcut: one range spans the base, the mips and any gap between
        // them, so a write into the gap decodes the chain again; keep a
        // range per level if mips far from their base show up.
        let range = chain_range(fetch).ok_or("texture without levels to read")?;
        // (fetch, offset in `bytes`, length) of each mip level read.
        let mut mips = Vec::new();
        let mut at = base_len;
        for m in xenon_texture::mip_fetches(fetch) {
            let Some((a, b)) = level_range(&m) else { break };
            let len = (b - a) as usize;
            if bytes.len() < at + len {
                break;
            }
            mips.push((m, at, len));
            at += len;
        }
        if !base && mips.is_empty() {
            return Err("texture without its base or a mip level read".into());
        }
        report.timings.hashed_bytes += bytes.len() as u64;
        let constant = self.regs.fetch_constant(n);
        let key = (decode_key(constant), hash_bytes(bytes));
        if let Some((view, _, used)) = self.texture_cache.get_mut(&key) {
            *used = self.replays;
            let view = view.clone();
            self.texture_memo
                .insert(constant, (view.clone(), range.0, range.1, Some(key)));
            return Ok(view);
        }
        let mut bc = self.bc_textures && !fetch.sign.contains(&1);
        let float32 = self.float32_filterable;
        // Width, height and texels of each level; no texels for levels no
        // LOD samples (the base not read, levels below the first sampled),
        // which stay wgpu's zeros.
        let mut levels = Vec::new();
        let mut host = None;
        if base {
            let (h, width, height, data) =
                decode_level(fetch, &bytes[..base_len], bc, float32).map_err(|e| e.to_string())?;
            host = Some(h);
            levels.push((width, height, Some(data)));
        } else {
            // A BC texture's base is whole blocks, written or not.
            bc &= fetch.width().is_multiple_of(4) && fetch.height().is_multiple_of(4);
            levels.push((fetch.width(), fetch.height(), None));
        }
        let (width, height) = (levels[0].0, levels[0].1);
        if width > self.max_dim || height > self.max_dim {
            return Err(format!("texture {width}x{height}"));
        }
        let first = fetch.mip_range().0 as usize;
        for (level, &(m, at, len)) in (1..).zip(&mips) {
            if level < first {
                levels.push((m.width(), m.height(), None));
                continue;
            }
            // shortcut: the chain stops at a level that does not decode to
            // the first level's host format (a DXT level not whole blocks)
            // or at all; sampled LODs past it clamp to the last level kept.
            let bc = bc && host.is_none_or(|h| h.is_compressed());
            match decode_level(&m, &bytes[at..at + len], bc, float32) {
                Ok((h, w, hh, d)) if host.is_none_or(|host| host == h) => {
                    host = Some(h);
                    levels.push((w, hh, Some(d)));
                }
                _ => break,
            }
        }
        // A cut chain ends at a level with texels, which LODs past it reach.
        while levels.last().is_some_and(|l| l.2.is_none()) {
            levels.pop();
        }
        let Some(host) = host else {
            return Err("texture whose first sampled mip level does not decode".into());
        };
        let uploaded: u64 = levels
            .iter()
            .flat_map(|l| &l.2)
            .map(|d| d.len() as u64)
            .sum();
        // GPU bytes, the levels left as zeros included.
        let (bw, bh) = host.block_dimensions();
        let block = u64::from(host.block_copy_size(None).unwrap_or(4));
        let len: u64 = levels
            .iter()
            .map(|&(w, h, _)| u64::from(w.div_ceil(bw)) * u64::from(h.div_ceil(bh)) * block)
            .sum();
        report.timings.textures_uploaded += 1;
        report.timings.texture_bytes += uploaded;
        let bucket = match host {
            wgpu::TextureFormat::Rgba8Unorm => 0,
            _ if host.is_compressed() => 1,
            wgpu::TextureFormat::Rgba16Float => 2,
            wgpu::TextureFormat::Rgba32Float => 3,
            _ => 4,
        };
        let by = &mut report.timings.uploads_by_format[bucket];
        by.0 += 1;
        by.1 += uploaded;
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("guest texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: host,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, (w, h, data)) in levels.iter().enumerate() {
            let Some(data) = data else { continue };
            // Rows of blocks: a BC row holds four rows of texels.
            let rows = h / host.block_dimensions().1;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(data.len() as u32 / rows),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: *w,
                    height: *h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = tex.create_view(&Default::default());
        let key_hash = hash_bytes(&to_bytes(&key.0)) ^ key.1;
        if self.texture_evicted.remove(&key_hash) {
            report.timings.textures_reuploaded += 1;
        }
        let limit = self.texture_cache_limit;
        if self.texture_cache_bytes + len > limit
            || self.texture_cache.len() >= self.texture_cache_count
        {
            self.evict_textures(limit.saturating_sub(len), report);
        }
        self.texture_cache_bytes += len;
        self.texture_cache
            .insert(key, (view.clone(), len, self.replays));
        self.texture_memo
            .insert(constant, (view.clone(), range.0, range.1, Some(key)));
        Ok(view)
    }

    /// Evicts least recently used textures until the cache holds at most
    /// 3/4 of `room` bytes and of `texture_cache_count` textures, so
    /// evictions come in batches, not per upload.
    /// Views still bound in the open batch stay alive through their clones.
    fn evict_textures(&mut self, room: u64, report: &mut Report) {
        let goal = room / 4 * 3;
        let mut by_age: Vec<_> = self
            .texture_cache
            .iter()
            .map(|(k, &(_, len, used))| (used, len, *k))
            .collect();
        by_age.sort_unstable_by_key(|e| e.0);
        // Groups and the memo hold views; drop them so evicted textures go too.
        self.texture_groups.clear();
        self.texture_sets.clear();
        self.texture_set_flags.clear();
        self.texture_memo.clear();
        self.texture_memo_epoch += 1;
        // shortcut: the evicted-key set only feeds a counter; cleared
        // when large, which undercounts re-uploads after that.
        if self.texture_evicted.len() > 1 << 16 {
            self.texture_evicted.clear();
        }
        for (_, len, k) in by_age {
            if self.texture_cache_bytes <= goal
                && self.texture_cache.len() <= self.texture_cache_count * 3 / 4
            {
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
        overrides: Option<[u8; 4]>,
        report: &mut Report,
    ) -> wgpu::Sampler {
        for c in fetch.clamp {
            let (m, exact) = sampler_address(c, self.clamp_to_border);
            if !exact {
                report.warn(format!("texture clamp mode {c} drawn as {m:?}"));
            }
        }
        let k = sampler_diagnostics(
            sampler_key(fetch, overrides, self.clamp_to_border),
            self.force_aniso,
            self.force_linear,
        );
        let device = &self.device;
        // As Xenia's sampler: LODs from the first mip level the fetch
        // samples; base map mip filtering is point at that level (+0.25
        // keeps magnification apart from minification).
        self.samplers
            .entry(k)
            .or_insert_with(|| {
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("guest sampler"),
                    address_mode_u: k.address[0],
                    address_mode_v: k.address[1],
                    address_mode_w: k.address[2],
                    mag_filter: k.mag,
                    min_filter: k.min,
                    mipmap_filter: k.mip,
                    lod_min_clamp: k.min_level as f32,
                    lod_max_clamp: if k.base_map {
                        k.min_level as f32 + 0.25
                    } else {
                        32.0
                    },
                    anisotropy_clamp: k.anisotropy,
                    border_color: k.border,
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
        let mut t = Instant::now();
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

        let sample = self.part_draws.is_multiple_of(PART_SAMPLE);
        let stamp = || sample.then(Instant::now);
        let slow0 = sample.then(|| self.slow_work());
        let t_state = stamp();
        let strip = matches!(
            topology,
            wgpu::PrimitiveTopology::TriangleStrip | wgpu::PrimitiveTopology::LineStrip
        );
        let reset_on = self.regs.get(regs::PA_SU_SC_MODE_CNTL) & (1 << 21) != 0;
        let key = match d.source {
            xenon_gpu::IndexSource::Dma {
                address,
                buffer_indices,
                format,
                endian,
            } => Some((
                address,
                buffer_indices,
                format as u8,
                endian as u8,
                d.index_count,
                reset_on.then(|| self.regs.get(regs::VGT_MULTI_PRIM_IB_RESET_INDX)),
                expand as u8,
                strip,
            )),
            _ => None,
        };
        let memo = key
            .and_then(|k| self.index_memo.get(&k))
            .map(|e| e.2.clone());
        let idx = match memo {
            Some(idx) => idx,
            None => {
                let mut guest = (0, 0);
                if let xenon_gpu::IndexSource::Dma {
                    address, format, ..
                } = &d.source
                {
                    let a = u64::from(*address & PHYSICAL_MASK);
                    let len = u64::from(d.index_count) * format.bytes() as u64;
                    self.sync(mem, a, a + len)?;
                    guest = (a, a + len);
                }
                let mut idx = indices::guest_indices(&*mem, &d.source, d.index_count)
                    .map_err(Skip::Unsupported)?;
                if reset_on
                    && matches!(d.source, xenon_gpu::IndexSource::Dma { .. })
                    && indices::mark_resets(
                        &mut idx,
                        self.regs.get(regs::VGT_MULTI_PRIM_IB_RESET_INDX),
                    )
                    && !strip
                {
                    return unsupported("primitive reset outside strips");
                }
                let idx = indices::expand(idx, expand);
                let host = Arc::new(HostIndices {
                    bytes: to_bytes(&idx),
                    len: idx.len() as u32,
                    min_max: min_max(&idx),
                    staged: Default::default(),
                });
                if let Some(k) = key {
                    // Entries dropped by forget_guest still count until
                    // this clears them all.
                    self.index_memo_bytes += host.bytes.len();
                    if self.index_memo.len() >= INDEX_MEMO_CAP
                        || self.index_memo_bytes > INDEX_MEMO_BYTES
                    {
                        self.index_memo.clear();
                        self.index_memo_bytes = host.bytes.len();
                    }
                    self.index_memo.insert(k, (guest.0, guest.1, host.clone()));
                }
                host
            }
        };
        if idx.len == 0 {
            return Ok(false);
        }

        // Every target of a draw has the surface pitch and the same height.
        let t_indices = stamp();
        let (w, h) = (surface & 0x3FFF, self.target_rows());
        let (sx, sy, sw, sh) = state::scissor(&self.regs, w, h);
        let rows = sy + sh;
        let msaa = (surface >> 16) & 3;
        // The open pass draws to these targets and they were written last:
        // `target` would transfer and make nothing, and the pass needs no
        // textures, so the lookups and handle copies are skipped (most draws
        // follow one to the same targets).
        let pass_key = (color, depth_key);
        let current = self.pass_open(&pass_key)
            && (color.iter().flatten().chain(&depth_key))
                .all(|k| self.written.get(k).is_some_and(|w| w.stamp == self.clock));
        let mut color_tex: [Option<wgpu::Texture>; 4] = Default::default();
        let mut depth_tex = None;
        if !current {
            (color_tex, depth_tex) = self.targets_of(pass_key, rows, msaa, report)?;
        }
        if sw == 0 || sh == 0 {
            return Ok(false);
        }

        let t_targets = stamp();
        let info = |a: &ShaderInfo, b: &ShaderInfo| {
            (
                a.vertex_fetch_consts | b.vertex_fetch_consts,
                a.textures_used | b.textures_used,
            )
        };
        let (vf_mask, tex_mask) = info(&vs_t.info, &ps_t.info);
        // Each slot has one sampler: the vertex stage's filters, else the
        // pixel stage's.
        let (vs_i, ps_i) = (&vs_t.info, &ps_t.info);
        let filters =
            |n: u32| vs_i.texture_filters[n as usize].or(ps_i.texture_filters[n as usize]);
        let both = vs_i.textures_used & ps_i.textures_used;
        let differ = (0..32)
            .any(|n| both & (1 << n) != 0 && vs_i.texture_filters[n] != ps_i.texture_filters[n]);
        if differ || (vs_i.texture_filter_conflicts | ps_i.texture_filter_conflicts) != 0 {
            report.warn("texture fetches of one slot with different filters drawn with one");
        }
        let extra = ps_t.info.interpolators_read & !vs_t.info.interpolators_written;
        let ps_extra = u16::from(ps_t.info.writes_depth && depth_format.is_none());

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
            ps_extra,
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
        // The key names the modules and the texture layout: a cached
        // pipeline needs neither looked up.
        let (pipeline, g1_layout) = match self.pipelines.get_mut(&key) {
            Some((p, used, g1)) => {
                *used = self.pipeline_uses;
                (p.clone(), g1.clone())
            }
            None => {
                let vs_mod = self.module(vs.hash, true, extra, vs_variant, None, &vs_t)?;
                let ps_mod = self.module(ps.hash, false, ps_extra, ps_variant, param_gen, &ps_t)?;
                let (g1_layout, layout) = self.texture_layout(tex_mask);
                let k = key.clone();
                // Only a pixel shader with parameter generation declares
                // xe_scale; at scale 1 its default holds.
                let scale = [("xe_scale", f64::from(self.scale))];
                let ps_constants: &[(&str, f64)] = if param_gen.is_some() && self.scale != 1 {
                    &scale
                } else {
                    &[]
                };
                let cache = self.pipeline_cache();
                let t0 = Instant::now();
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
                            compilation_options: wgpu::PipelineCompilationOptions {
                                constants: ps_constants,
                                ..Default::default()
                            },
                            targets: &k.color.map(|c| {
                                c.map(|(format, blend, write_mask)| wgpu::ColorTargetState {
                                    format,
                                    blend,
                                    write_mask,
                                })
                            }),
                        }),
                        multiview_mask: None,
                        cache,
                    })
                })?;
                self.pipeline_made(t0);
                let entry = (p.clone(), self.pipeline_uses, g1_layout.clone());
                self.pipelines.insert(key, entry);
                self.timings.pipelines_created += 1;
                self.drop_pipelines();
                (p, g1_layout)
            }
        };
        let t0 = t;
        self.timings.shaders += lap(&mut t);
        let slow = slow0.is_some_and(|s| s != self.slow_work());
        if !slow {
            self.part_draws = self.part_draws.wrapping_add(1);
        }
        if let (false, Some(a), Some(b), Some(c)) = (slow, t_state, t_indices, t_targets) {
            add_sampled(&mut self.timings.setup_parts, &[t0, a, b, c, t]);
        }

        let reach = &mut self.window_scratch.reach;
        reach.clear();
        reach.extend(&vs_t.info.vertex_fetch_reach);
        reach.extend(&ps_t.info.vertex_fetch_reach);
        let indices = offset_range(idx.min_max, self.regs.get(regs::VGT_INDX_OFFSET));
        let (mut read, window_base, rebase, staged) =
            self.guest_window(vf_mask, indices, mem, report)?;
        let gen = self.consts_gen;
        let consts = match &self.consts_built {
            Some((g, c)) if *g == gen => c.clone(),
            built => {
                // Registers written with the words they held keep the block.
                let c = consts_prefix(&self.regs);
                let c = match built {
                    Some((_, old)) if **old == c => old.clone(),
                    _ => Arc::new(c),
                };
                self.consts_built = Some((gen, c.clone()));
                c
            }
        };
        let suffix = consts_suffix(&self.regs, window_base, &rebase);
        let epi = state::epilogue(&self.regs, w, h, self.scale);
        // The indices are staged once and bound again by later draws of
        // them until the arena wraps: a frame draws the same meshes many
        // times, kilobytes of indices each. (Rectangle lists read them in
        // the vertex shader, binding 3, and draw without an index buffer.)
        let mut kept_index = idx.staged_in(self.arena_epoch);
        // The window and constants staged earlier (in this batch or one
        // submitted since) are bound again, unless the arena wrapped since
        // (it is then overwritten) or would wrap to stage the rest.
        let mut kept_window =
            staged.filter(|((a, b, s), _)| self.window_memo.get(a) == Some(&(*b, *s)));
        let mut kept_consts = (self.staged_consts.as_ref())
            .filter(|(b, s, _)| *s == suffix && (Arc::ptr_eq(b, &consts) || **b == *consts))
            .map(|&(_, _, s)| s);
        // Each stage's float constants: those it reads, staged again only
        // when one of the stage's registers was written since.
        let float_masks = [&vs_t.info, &ps_t.info].map(|info| info.float_consts_bound());
        let float_keys = [0, 1].map(|i| float_masks[i].map(|m| (self.float_gens[i], m)));
        let mut kept_floats = [0, 1].map(|i| {
            (self.staged_floats[i].zip(float_keys[i]))
                .filter(|((k, _), key)| k == key)
                .map(|((_, s), _)| s)
        });
        // Whether all of it fits is decided on the sizes alone: a stage's
        // float constants are written only once it is known to stage them.
        let float_len = [0, 1].map(|i| {
            float_masks[i].map(|m| 16 * m.iter().map(|w| w.count_ones() as usize).sum::<usize>())
        });
        // The arena bytes of each part, staged in this order: the window
        // (its parts, the kept window read again, or one empty part for no
        // window), constants, float constants, epilogue and indices.
        let window_need = match staged {
            Some(((lo, hi, _), _)) => self.staged_size(hi - lo),
            None if read.is_empty() => self.staged_size(0),
            None => (read.iter())
                .map(|&(lo, hi)| self.staged_size(hi - lo))
                .sum(),
        };
        let consts_need = self.staged_size(CONSTS_BYTES as u64);
        let float_need = float_len.map(|n| n.map_or(0, |n| self.staged_size(n as u64)));
        let epi_need = self.staged_size(4 * epi.len() as u64);
        let index_need = self.staged_size(idx.bytes.len() as u64);
        // Of the parts, those staged (not kept).
        let need = |window: bool, consts: bool, floats: [bool; 2], index: bool| {
            let part = |staged: bool, n: u64| if staged { n } else { 0 };
            part(window, window_need)
                + part(consts, consts_need)
                + part(floats[0], float_need[0])
                + part(floats[1], float_need[1])
                + epi_need
                + part(index, index_need)
        };
        let any_kept = kept_window.is_some()
            || kept_consts.is_some()
            || kept_floats != [None; 2]
            || kept_index.is_some();
        if any_kept
            && !self.fits(need(
                kept_window.is_none(),
                kept_consts.is_none(),
                kept_floats.map(|k| k.is_none()),
                kept_index.is_none(),
            ))
        {
            // Staging all of it submits the batch in `reserve`; a kept
            // window is read again below.
            (kept_window, kept_consts, kept_floats, kept_index) = (None, None, [None; 2], None);
        }
        if let (Some(((lo, hi, _), _)), None) = (staged, kept_window) {
            self.sync(mem, lo, hi)?;
            read = vec![(lo, hi)];
        }
        if kept_window.is_none() {
            report.timings.guest_bytes += read.iter().map(|(lo, hi)| hi - lo).sum::<u64>();
        }
        // A full arena submits the batch in `reserve`: submit time, not
        // guest data.
        let submit = self.timings.submit;
        self.reserve(need(
            kept_window.is_none(),
            kept_consts.is_none(),
            kept_floats.map(|k| k.is_none()),
            kept_index.is_none(),
        ))?;
        let window_slice = match kept_window {
            Some((w, len)) => {
                self.timings.windows_reused += 1;
                self.timings.reused_bytes += len;
                w.2
            }
            None => {
                // The window's parts are read straight into the staging
                // area; the parts after the first end the window's binding.
                let mut first_last = None;
                for &(lo, hi) in &read {
                    let Some(s) = self.push((hi - lo) as usize, |b| mem.read_into(lo as u32, b))
                    else {
                        return unsupported(format!(
                            "vertex data at 0x{lo:08X}..0x{hi:08X} not readable"
                        ));
                    };
                    first_last = Some((first_last.map_or(s, |(f, _)| f), s));
                }
                let (first, last) = first_last.unwrap_or_else(|| {
                    let s = self.push_bytes(&[]);
                    (s, s)
                });
                if let [(lo, hi)] = read[..] {
                    // None holds this window (it was read), so those it
                    // holds are the ones from `lo` that end by `hi`.
                    let inside: Vec<u64> = (self.window_memo.range(lo..))
                        .take_while(|(_, (b, _))| *b <= hi)
                        .map(|(&a, _)| a)
                        .collect();
                    for a in inside {
                        self.window_memo.remove(&a);
                    }
                    self.window_memo.insert(lo, (hi, first));
                }
                (first.0, last.0 + last.1 - first.0)
            }
        };
        let consts_slice = match kept_consts {
            Some(s) => {
                self.timings.consts_reused += 1;
                s
            }
            None => {
                let s = (self.push(CONSTS_BYTES, |b| {
                    let (prefix, rest) = b.split_at_mut(consts.len());
                    prefix.copy_from_slice(&consts);
                    words_into(&suffix, rest);
                    true
                }))
                .expect("constants staged");
                self.staged_consts = Some((consts, suffix, s));
                s
            }
        };
        // A stage that reads no float constant declares no binding for
        // them: the group still needs one, so the constants' first words.
        let mut float_slices = [(consts_slice.0, 16); 2];
        for i in 0..2 {
            let Some(key) = float_keys[i] else { continue };
            float_slices[i] = match kept_floats[i] {
                Some(s) => s,
                None => {
                    let len = float_len[i].expect("float constants read");
                    let s = self.push(len, |_| true).expect("float constants staged");
                    let at = (s.0 - self.staged_base) as usize;
                    float_consts(&self.regs, i as u32, &key.1, &mut self.staged[at..at + len]);
                    self.staged_floats[i] = Some((key, s));
                    s
                }
            };
        }
        let epilogue_slice = (self.push(4 * epi.len(), |b| {
            words_into(&epi, b);
            true
        }))
        .expect("epilogue staged");
        let index_slice = match kept_index {
            Some(s) => {
                self.timings.indices_reused += 1;
                s
            }
            None => {
                let s = self.push_bytes(&idx.bytes);
                idx.set_staged(self.arena_epoch, s.0);
                s
            }
        };
        // Binding 3 is unused but for rectangle lists: then the indices.
        let rect_slice = if rect_list {
            index_slice
        } else {
            epilogue_slice
        };
        // In binding order.
        let slices = [
            window_slice,
            consts_slice,
            epilogue_slice,
            rect_slice,
            float_slices[0],
            float_slices[1],
        ];
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
        // shortcut: a set found here neither reports its textures'
        // warnings again nor marks them used for eviction; the first draw
        // of the set in each memo epoch does both.
        // The key is built in a kept list: a new one only when a set is
        // recorded below.
        let mut key = (tex_mask, std::mem::take(&mut self.texture_set_key));
        key.1.clear();
        let regs = &self.regs;
        (key.1).extend(
            (0..32)
                .filter(|n| tex_mask & (1 << n) != 0)
                .map(|n| (regs.fetch_constant(n), pack_filters(filters(n)))),
        );
        let set = (self.texture_sets.get(&key))
            .filter(|(_, epoch, kept)| {
                *epoch == self.texture_memo_epoch && kept.load(Ordering::Relaxed)
            })
            .map(|(g, _, _)| g.clone());
        self.texture_set_key = key.1;
        let mut textures = Vec::new();
        if set.is_none() {
            for n in (0..32).filter(|n| tex_mask & (1 << n) != 0) {
                textures.push(self.texture(n, filters(n), mem, report)?);
            }
        }
        self.timings.textures += lap(&mut t);
        let arena = self.arena.clone().expect("staged");
        let mut offsets = [0u32; 6];
        for (o, &(offset, _)) in offsets.iter_mut().zip(&slices) {
            *o = u32::try_from(offset)
                .map_err(|_| Skip::Unsupported("draw data past 4 GiB of the arena".into()))?;
        }
        let mut sizes = slices.map(|s| s.1);
        // The float constant bindings at a power of two of at least their
        // size (the shader declares its count; the words past it go
        // unread), so draws of different shaders share a bind group: exact
        // sizes made thousands, past the cap that drops them all.
        for i in [4, 5] {
            let up = sizes[i].max(16).next_power_of_two();
            if slices[i].0 + up <= arena.size() {
                sizes[i] = up;
            }
        }
        // Window sizes vary without end over a long run; start over past a
        // cap rather than keep every one.
        if self.arena_groups.len() >= ARENA_GROUP_CAP {
            self.arena_groups.clear();
        }
        let (device, group0) = (&self.device, &self.group0);
        let g0 = self
            .arena_groups
            .entry(sizes)
            .or_insert_with(|| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("draw group 0"),
                    layout: group0,
                    entries: &[0, 1, 2, 3, 4, 5].map(|i| wgpu::BindGroupEntry {
                        binding: i,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena,
                            offset: 0,
                            size: wgpu::BufferSize::new(sizes[i as usize]),
                        }),
                    }),
                })
            })
            .clone();
        if self.texture_groups.len() >= TEXTURE_GROUP_CAP {
            self.texture_groups.clear();
        }
        let g1 = if let Some(g) = set {
            self.timings.texture_sets_reused += 1;
            g
        } else {
            let g = self.texture_group(tex_mask, textures, &g1_layout);
            if self.texture_sets.len() >= TEXTURE_GROUP_CAP {
                self.texture_sets.clear();
                self.texture_set_flags.clear();
            }
            let constants: Vec<([u32; 6], u32)> = (0..32)
                .filter(|n| tex_mask & (1 << n) != 0)
                .map(|n| (self.regs.fetch_constant(n), pack_filters(filters(n))))
                .collect();
            // Only a set whose textures all came from the texture memo: a
            // texture that could not be read or decoded is drawn as zeros
            // with no memo entry, so no guest write would move the epoch
            // and the set would keep its zeros after the data is fixed.
            // (Slots without a texture fetch constant read no memory.)
            let found = constants.iter().all(|(c, _)| {
                xenon_texture::TextureFetch::decode(*c).fetch_type != 2
                    || self.texture_memo.contains_key(c)
            });
            if found {
                let kept = Arc::new(AtomicBool::new(true));
                for (c, _) in constants
                    .iter()
                    .filter(|(c, _)| self.texture_memo.contains_key(c))
                {
                    let flags = self.texture_set_flags.entry(*c).or_default();
                    // A set made again after its flag was cleared leaves the
                    // old one here: drop those as the list grows.
                    if flags.len() >= 64 {
                        flags.retain(|f| f.load(Ordering::Relaxed));
                    }
                    flags.push(kept.clone());
                }
                let epoch = self.texture_memo_epoch;
                let key = (tex_mask, constants);
                (self.texture_sets).insert(key, (g.clone(), epoch, kept));
            }
            g
        };
        let t_groups = stamp();
        // Staging may have flushed the batch, ending the pass: then it
        // begins again and needs the textures.
        if current && !self.pass_open(&pass_key) {
            (color_tex, depth_tex) = self.targets_of(pass_key, rows, msaa, report)?;
        }
        let s = self.scale;
        let (pass, bound) = self.pass(pass_key, &color_tex, depth_tex.as_ref());
        let t_pass = stamp();
        if bound.pipeline.as_ref() != Some(&pipeline) {
            pass.set_pipeline(&pipeline);
            bound.pipeline = Some(pipeline);
        }
        if bound
            .g0
            .as_ref()
            .is_none_or(|(g, o)| *g != g0 || *o != offsets)
        {
            pass.set_bind_group(0, &g0, &offsets);
            bound.g0 = Some((g0, offsets));
        }
        if bound.g1.as_ref() != Some(&g1) {
            pass.set_bind_group(1, &g1, &[]);
            bound.g1 = Some(g1);
        }
        if bound.index.as_ref() != Some(&arena) {
            pass.set_index_buffer(arena.slice(..), wgpu::IndexFormat::Uint32);
            bound.index = Some(arena.clone());
        }
        // Bound in guest pixels, set in host ones.
        if bound.viewport != Some((w, h)) {
            pass.set_viewport(0.0, 0.0, (w * s) as f32, (h * s) as f32, 0.0, 1.0);
            bound.viewport = Some((w, h));
        }
        if bound.scissor != Some((sx, sy, sw, sh)) {
            pass.set_scissor_rect(sx * s, sy * s, sw * s, sh * s);
            bound.scissor = Some((sx, sy, sw, sh));
        }
        if let Some(&(_, reference)) = stencil.as_ref().filter(|s| bound.stencil != Some(s.1)) {
            pass.set_stencil_reference(reference);
            bound.stencil = Some(reference);
        }
        let blend = blend_rgba.map(f32::to_bits);
        if bound.blend != Some(blend) {
            pass.set_blend_constant(wgpu::Color {
                r: f64::from(blend_rgba[0]),
                g: f64::from(blend_rgba[1]),
                b: f64::from(blend_rgba[2]),
                a: f64::from(blend_rgba[3]),
            });
            bound.blend = Some(blend);
        }
        if rect_list {
            pass.draw(0..idx.len * 2, 0..1);
        } else {
            // Staged at the arena's alignment, so a whole u32 index.
            let first = (index_slice.0 / 4) as u32;
            pass.draw_indexed(first..first + idx.len, 0, 0..1);
        }
        // Recorded, not yet submitted: everything else that touches the
        // targets flushes the batch first, so stamps follow submit order.
        // The targets of one draw share its stamp: none is newer than
        // another, and the next draw to any of them finds no newer target.
        self.clock += 1;
        for k in color.into_iter().flatten() {
            self.stamp_written(k, rows, msaa);
        }
        if let Some(k) = depth_key.filter(|_| depth.write || stencil.is_some()) {
            self.stamp_written(k, rows, msaa);
        }
        let t0 = t;
        self.timings.submit += lap(&mut t);
        let slow = slow0.is_some_and(|s| s != self.slow_work());
        if let (false, Some(a), Some(b)) = (slow, t_groups, t_pass) {
            add_sampled(&mut self.timings.submit_parts, &[t0, a, b, t]);
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
        let cache = self.pipeline_cache();
        let t0 = Instant::now();
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
                cache,
            })
        })?;
        self.pipeline_made(t0);
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
        // Clamped in guest pixels, drawn in host ones.
        let s = self.scale;
        let (gw, gh) = (tex.width() / s, tex.height() / s);
        let rect = resolve::Rect {
            x0: rect.x0.min(gw),
            y0: rect.y0.min(gh),
            x1: rect.x1.min(gw),
            y1: rect.y1.min(gh),
        };
        if rect.is_empty() {
            return Ok(());
        }
        let format = self.host_format(key.format);
        let pipeline = self.clear_pipeline(format)?;
        let c = color.map(f32::to_bits);
        let value = [c[0], c[1], c[2], c[3], depth.to_bits(), 0, 0, 0];
        let offset = self.stage_uniform(&value)?;
        let (device, layout) = (&self.device, &self.clear_layout);
        let arena = self.arena.as_ref().expect("staged");
        let bg = (self.clear_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clear"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: arena,
                        offset: 0,
                        size: wgpu::BufferSize::new(4 * value.len() as u64),
                    }),
                }],
            })
        }))
        .clone();
        let view = (self.target_views.entry(tex.clone()))
            .or_insert_with(|| tex.create_view(&Default::default()))
            .clone();
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
                timestamp_writes: GpuClock::render(&mut self.gpu_clock, 1),
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
            pass.set_bind_group(0, &bg, &[offset]);
            pass.set_scissor_rect(
                rect.x0 * s,
                rect.y0 * s,
                rect.width() * s,
                rect.height() * s,
            );
            pass.set_stencil_reference(stencil);
            pass.draw(0..3, 0..1);
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
        // The copy and clears are recorded after every draw before them (a
        // CPU copy flushes first).
        let v = u64::from(r.vertex_fetch0.address & PHYSICAL_MASK);
        self.sync(mem, v, v + 24)?;
        if r.msaa_samples != 0 {
            report.warn("MSAA surfaces drawn and resolved single-sampled");
        }
        let height = self.target_rows();
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
            x1: rect.x1.min(pitch).min(tex.width() / self.scale),
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
        self.flush()?;
        let mut t = Instant::now();
        self.wait_idle()?;
        self.timings.gpu_wait += lap(&mut t);
        let px = self.read_target(format, &tex, rect.x0, rect.y0, rect.width(), rect.height())?;
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
        // `rect` is in guest pixels; resolve.wgsl reads each one's host
        // pixels at the scale in the last parameter.
        let s = self.scale;
        if rect.y1 > tex.height() / s {
            return Err(Error::Unsupported(format!(
                "readback {}x{} at ({}, {}) outside the {}x{} target",
                rect.width(),
                rect.height(),
                rect.x0,
                rect.y0,
                tex.width() / s,
                tex.height() / s
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
            s,
        ];
        let offset = self.stage_uniform(&params)?;
        let device = &self.device;
        let cache = self.disk_cache.as_ref().map(|c| &c.cache);
        let layouts = &self.resolve_layouts[entry];
        // Set once the shader module is made, so it times the pipeline alone.
        let mut t0 = None;
        let pipeline = self.resolve_pipelines[entry].get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("resolve"),
                source: wgpu::ShaderSource::Wgsl(include_str!("resolve.wgsl").into()),
            });
            t0 = Some(Instant::now());
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resolve"),
                layout: Some(&layouts.1),
                module: &module,
                entry_point: Some(["main", "depth_main", "depth_stencil_main"][entry]),
                compilation_options: Default::default(),
                cache,
            })
        });
        let pipeline = &pipeline.clone();
        if let Some(t0) = t0 {
            self.pipeline_made(t0);
        }
        // The source's aspect views and the buffer, made once for each
        // pair: a frame resolves the same targets into the same buffers.
        let (device, layouts) = (&self.device, &self.resolve_layouts[entry]);
        let arena = self.arena.as_ref().expect("staged");
        let Mirror { buf, groups, .. } = &mut self.mirrors[i];
        if groups.len() >= RESOLVE_GROUP_CAP {
            groups.clear();
        }
        let bg = (groups.entry((tex.clone(), entry)))
            .or_insert_with(|| {
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
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: arena,
                            offset: 0,
                            size: wgpu::BufferSize::new(4 * params.len() as u64),
                        }),
                    },
                ]);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("resolve"),
                    layout: &layouts.0,
                    entries: &entries,
                })
            })
            .clone();
        // Recorded in the batch after its draws, which end their render
        // pass here; submitted with them.
        let device = &self.device;
        let batch = self.batch.get_or_insert_with(|| Batch {
            pass: None,
            enc: device.create_command_encoder(&Default::default()),
        });
        batch.pass = None;
        {
            let mut pass = batch.enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resolve"),
                timestamp_writes: GpuClock::compute(&mut self.gpu_clock, 2),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[offset]);
            pass.dispatch_workgroups(rect.width().div_ceil(8), rect.height().div_ceil(8), 1);
        }
        self.resolve_pending = true;
        self.timings.resolve_cpu += lap(&mut t);
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
            if k.tex.mip_level_count() > 1 {
                k.stale = Some((0, k.tex.height()));
                continue;
            }
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
        let (lo, hi) = (u64::from(addr), u64::from(addr) + u64::from(extent));
        let mut kept = self.mirrors.iter().position(|m| m.lo <= lo && hi <= m.hi);
        // A CPU write to the front buffer since the resolve drops the kept
        // copy, as the synchronous path's check does.
        if let Some(i) = kept.filter(|_| self.late_front_buffer) {
            if !self.verify(i, mem, lo, hi) {
                kept = None;
            }
        }
        if let (true, Some(i)) = (self.late_front_buffer, kept) {
            // The GPU decodes the frame (byte order, tiling, swizzle) into
            // RGBA8 rows, so the CPU only reads them back: untiling a 720p
            // frame on the draw thread was a tenth of its time.
            let size = u64::from(width) * u64::from(height) * 4;
            let usage = |u| wgpu::BufferDescriptor {
                label: Some("front buffer"),
                size,
                usage: u,
                mapped_at_creation: false,
            };
            let buf = (self.device).create_buffer(&usage(
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ));
            let rows = (self.device).create_buffer(&usage(
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ));
            let (endian, swizzle) = self.front_buffer_read(addr);
            let pipeline = self.decode_pipeline(false);
            let m = &self.mirrors[i];
            let params = [
                width,
                height,
                pitch,
                endian as u32,
                // x, y, z, w from themselves: none.
                u32::from(swizzle.unwrap_or((1 << 3) | (2 << 6) | (3 << 9))),
                ((lo - m.lo) / 4) as u32,
                width,
                0,
                0,
                0,
                0,
                0,
            ];
            let uniform = self.buffer(
                "front params",
                &to_bytes(&params),
                wgpu::BufferUsages::UNIFORM,
            );
            let m = &self.mirrors[i];
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("front buffer decode"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: m.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: rows.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            // At the end of the batch, after this frame's resolves, through
            // the submit thread: the swap does not wait for the queued
            // batches, only for last frame's decode below.
            let device = &self.device;
            let b = self.batch.get_or_insert_with(|| Batch {
                pass: None,
                enc: device.create_command_encoder(&Default::default()),
            });
            b.pass = None;
            {
                let mut pass = b.enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
            b.enc.copy_buffer_to_buffer(&rows, 0, &buf, 0, size);
            self.flush().map_err(|e| match e {
                Skip::Fatal(e) => e,
                Skip::Unsupported(why) => Error::Gpu(why),
            })?;
            let sent = self.submitter.sent();
            let new = LateFront {
                buf,
                sent,
                size: (width, height),
            };
            let Some(old) = self.late_front.replace(new) else {
                return Ok(Err("first late front buffer: no frame yet".into()));
            };
            // A frame later, its batch is most likely submitted already.
            let index = self.submitter.wait_for(old.sent);
            let rgba8 = self.map_submitted(&old.buf, index)?;
            let (width, height) = old.size;
            return Ok(Ok(Frame {
                width,
                height,
                rgba8,
            }));
        }
        // An older late copy would come out of order after this one.
        self.late_front = None;
        self.sync(mem, lo, hi)?;
        let raw = mem
            .read_bytes(addr, extent as usize)
            .ok_or_else(|| format!("front buffer at 0x{addr:08X} not readable"));
        let (endian, swizzle) = self.front_buffer_read(addr);
        Ok(raw.and_then(|raw| detile_front(raw, width, height, pitch, endian, swizzle)))
    }

    /// The endian the front buffer at `addr` was resolved with, and the
    /// swizzle scan-out reads it through: fetch constant 0's, as Xenia's
    /// swap texture, when it is a texture fetch of this address (it undoes
    /// a red/blue swap the final resolve put in with copy_dest_swap).
    fn front_buffer_read(&self, addr: u32) -> (xenon_texture::Endian, Option<u16>) {
        let endian = match self.resolved_endian.get(&addr) {
            Some(&e) if e <= 3 => xenon_texture::Endian::from_bits(e),
            _ => xenon_texture::Endian::K8in32,
        };
        let f = xenon_texture::TextureFetch::decode(self.regs.fetch_constant(0));
        let swizzle =
            (f.fetch_type == 2 && f.base_address & PHYSICAL_MASK == addr).then_some(f.swizzle);
        (endian, swizzle)
    }
}

/// See [`Drawer::window_scratch`].
#[derive(Default)]
struct WindowScratch {
    /// The vertex fetches the draw's shaders reach, vertex shader first.
    reach: Vec<VertexFetchReach>,
    /// Each fetch constant's `[lo, hi)` and number, sorted.
    ranges: Vec<(u64, u64, u32)>,
    /// Each fetch constant's part.
    part_of: Vec<(u32, usize)>,
    /// Where each part lands in the window.
    at: Vec<u64>,
}

/// `ranges` sorted by start, with those that touch or overlap made one.
fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (lo, hi) in ranges {
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// Whether `[a, b)` shares a byte with one of `ranges`, which
/// [`merge_ranges`] made (or is a single range).
fn meets(ranges: &[(u64, u64)], a: u64, b: u64) -> bool {
    // Ends are sorted like starts: the first range ending past `a`.
    let i = ranges.partition_point(|&(_, hi)| hi <= a);
    ranges.get(i).is_some_and(|&(lo, _)| lo < b)
}

/// A tiled front buffer's bytes as a frame.
fn detile_front(
    mut raw: Vec<u8>,
    width: u32,
    height: u32,
    pitch: u32,
    endian: xenon_texture::Endian,
    swizzle: Option<u16>,
) -> Result<Frame, String> {
    xenon_texture::endian::swap_in_place(&mut raw, endian);
    let mut rgba8 =
        xenon_texture::tile::detile_2d(&raw, width, height, pitch, 4).map_err(|e| e.to_string())?;
    if let Some(s) = swizzle {
        xenon_texture::apply_swizzle(&mut rgba8, s);
    }
    Ok(Frame {
        width,
        height,
        rgba8,
    })
}

impl Drawer {}

/// The base level of `fetch` from its guest bytes as host texels: host
/// format, width, height and texel bytes. BC blocks when `bc` and
/// [`bc_blocks`] takes them; Rgba8Unorm, or floats where
/// [`host_texture_format`] picks them (32-bit ones with `float32`).
fn decode_level(
    fetch: &xenon_texture::TextureFetch,
    bytes: &[u8],
    bc: bool,
    float32: bool,
) -> Result<(wgpu::TextureFormat, u32, u32, Vec<u8>), xenon_texture::Error> {
    if bc {
        if let Some((format, blocks)) = bc_blocks(fetch, bytes)? {
            return Ok((format, fetch.width(), fetch.height(), blocks));
        }
    }
    let host = host_texture_format(fetch.format, fetch.sign, float32);
    if host == wgpu::TextureFormat::Rgba8Unorm {
        let img = xenon_texture::decode_base_level(fetch, bytes)?;
        return Ok((host, img.width, img.height, img.rgba8));
    }
    let img = xenon_texture::decode_base_level_f32(fetch, bytes)?;
    let data = if host == wgpu::TextureFormat::Rgba32Float {
        img.rgba.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        img.rgba
            .iter()
            .flat_map(|&v| formats::f32_to_half(v).to_le_bytes())
            .collect()
    };
    Ok((host, img.width, img.height, data))
}

/// Whether the guest path reads the base level of `fetch`: not when its
/// base page is 0 and it lists a mip level, as Xenia skips it there (its
/// [`xenon_texture::TextureFetch::mip_range`] then starts at 1, so no LOD
/// samples the base).
fn reads_base(fetch: &xenon_texture::TextureFetch) -> bool {
    fetch.base_address & 0x1FFF_F000 != 0 || xenon_texture::mip_fetches(fetch).next().is_none()
}

/// The physical guest range `[lo, hi)` spanning the levels of `fetch` the
/// guest path reads: the base when [`reads_base`], then each of
/// [`xenon_texture::mip_fetches`] that [`level_range`] places; `None` when
/// there are none.
fn chain_range(fetch: &xenon_texture::TextureFetch) -> Option<(u64, u64)> {
    let base = level_range(fetch).filter(|_| reads_base(fetch));
    let mips = xenon_texture::mip_fetches(fetch).map_while(|m| level_range(&m));
    base.into_iter()
        .chain(mips)
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

/// The physical guest range `[lo, hi)` the base level of `fetch` reads.
fn level_range(fetch: &xenon_texture::TextureFetch) -> Option<(u64, u64)> {
    let a = u64::from(fetch.base_address & PHYSICAL_MASK);
    let len = xenon_texture::bytes_needed(fetch).ok()?;
    Some((a, a + len as u64))
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

    /// Plain commands are charged by run, timed ones not at all, with two
    /// clock reads per timed command and none per plain one.
    #[test]
    fn plain_timer_charges_runs_between_timed_commands() {
        use std::cell::Cell;
        let base = Instant::now();
        let (now, reads) = (Cell::new(0u64), Cell::new(0u32));
        let clock = || {
            reads.set(reads.get() + 1);
            base + Duration::from_micros(now.get())
        };
        let mut timer = PlainTimer::new(clock);
        let mut commands = Duration::ZERO;
        // (timed, microseconds of work): runs of plain commands between
        // draws, as the replay loop sees them.
        let list = [(false, 3), (false, 4), (true, 100), (false, 5), (true, 200)];
        for (timed, work) in list {
            if timed {
                timer.charge(&mut commands);
            }
            now.set(now.get() + work);
            if timed {
                timer.restart();
            }
            // Upkeep after the command, charged as plain.
            now.set(now.get() + 1);
        }
        timer.charge(&mut commands);
        assert_eq!(commands, Duration::from_micros(3 + 4 + 5 + 5));
        assert_eq!(reads.get(), 1 + 2 * 2 + 1);
    }

    #[test]
    fn sampled_parts_scale_by_the_sample_rate() {
        let t = Instant::now();
        let us = |n| t + Duration::from_micros(n);
        let mut parts = [Duration::ZERO; 3];
        add_sampled(&mut parts, &[t, us(1), us(3), us(6)]);
        let n = PART_SAMPLE;
        assert_eq!(
            parts.map(|p| p.as_micros()),
            [1, 2, 3].map(|p| p * u128::from(n))
        );
    }

    /// A fetch constant with clamp modes, filters (mag, min, mip, aniso)
    /// and border colour.
    fn sampler_fetch(clamp: [u32; 3], f: [u32; 4], border: u32) -> xenon_texture::TextureFetch {
        let w0 = 2 | clamp[0] << 10 | clamp[1] << 13 | clamp[2] << 16;
        let w3 = f[0] << 19 | f[1] << 21 | f[2] << 23 | f[3] << 25;
        xenon_texture::TextureFetch::decode([w0, 0, 0, w3, 0, border])
    }

    /// Every clamp mode, with and without the border feature.
    #[test]
    fn sampler_key_address_modes() {
        use wgpu::AddressMode as A;
        let want = [
            (A::Repeat, A::Repeat),
            (A::MirrorRepeat, A::MirrorRepeat),
            (A::ClampToEdge, A::ClampToEdge),
            (A::ClampToEdge, A::ClampToEdge),
            (A::ClampToEdge, A::ClampToEdge),
            (A::ClampToEdge, A::ClampToEdge),
            (A::ClampToEdge, A::ClampToBorder),
            (A::ClampToEdge, A::ClampToEdge),
        ];
        for (c, &(plain, border)) in want.iter().enumerate() {
            let f = sampler_fetch([c as u32, 0, 2], [1, 1, 0, 0], 1);
            let k = sampler_key(&f, None, false);
            assert_eq!(k.address, [plain, A::Repeat, A::ClampToEdge], "mode {c}");
            assert_eq!(k.border, None, "mode {c}");
            let k = sampler_key(&f, None, true);
            assert_eq!(k.address[0], border, "mode {c}");
            let colour = (c == 6).then_some(wgpu::SamplerBorderColor::OpaqueWhite);
            assert_eq!(k.border, colour, "mode {c}");
            let exact = sampler_address(c as u8, true).1;
            assert_eq!(exact, matches!(c, 0 | 1 | 2 | 6), "mode {c}");
        }
        // Other border colours are transparent black, as in Xenia.
        let f = sampler_fetch([6, 6, 6], [0; 4], 0);
        let k = sampler_key(&f, None, true);
        assert_eq!(k.border, Some(wgpu::SamplerBorderColor::TransparentBlack));
    }

    /// Instruction filter fields override the fetch constant's unless 3
    /// (aniso 7); any anisotropy makes every filter linear, as in Xenia.
    #[test]
    fn sampler_key_filters_and_anisotropy() {
        use wgpu::FilterMode::{Linear, Nearest};
        use wgpu::MipmapFilterMode as M;
        let point = sampler_fetch([0; 3], [0, 0, 0, 0], 0);
        let k = sampler_key(&point, None, false);
        assert_eq!(
            (k.mag, k.min, k.mip, k.anisotropy),
            (Nearest, Nearest, M::Nearest, 1)
        );
        assert_eq!(sampler_key(&point, Some([3, 3, 3, 7]), false), k);
        let k = sampler_key(&point, Some([1, 3, 1, 7]), false);
        assert_eq!((k.mag, k.min, k.mip), (Linear, Nearest, M::Linear));
        let k = sampler_key(&point, Some([1, 1, 2, 7]), false);
        assert!(k.base_map);
        for (aniso, want) in [(0, 1), (1, 1), (2, 2), (3, 4), (4, 8), (5, 16), (6, 16)] {
            let f = sampler_fetch([0; 3], [1, 1, 1, aniso], 0);
            assert_eq!(
                sampler_key(&f, None, false).anisotropy,
                want,
                "aniso {aniso}"
            );
            // The instruction's aniso over point filters: linear filters
            // with it, point ones without.
            let k = sampler_key(&point, Some([3, 3, 3, aniso as u8]), false);
            let (f_want, m_want) = if aniso != 0 {
                (Linear, M::Linear)
            } else {
                (Nearest, M::Nearest)
            };
            assert_eq!(
                (k.mag, k.min, k.mip, k.anisotropy),
                (f_want, f_want, m_want, want),
                "aniso {aniso} over point"
            );
            // A base map mip filter stays one under anisotropy.
            let k = sampler_key(&point, Some([0, 0, 2, aniso as u8]), false);
            assert!(k.base_map, "aniso {aniso}");
        }
        // The instruction turns the constant's anisotropy off.
        let f = sampler_fetch([0; 3], [1, 1, 1, 5], 0);
        assert_eq!(sampler_key(&f, Some([3, 3, 3, 0]), false).anisotropy, 1);

        // Diagnostic overrides: forced anisotropy reaches linear/linear
        // fetches only; off keeps every key.
        let linear = sampler_key(&sampler_fetch([0; 3], [1, 1, 0, 0], 0), None, false);
        let k = sampler_diagnostics(linear, 8, false);
        assert_eq!((k.anisotropy, k.mip), (8, M::Linear));
        let base = sampler_key(&point, Some([1, 1, 2, 0]), false);
        let guest = sampler_key(&sampler_fetch([0; 3], [1, 1, 1, 3], 0), None, false);
        let mixed = sampler_key(&point, Some([1, 3, 3, 7]), false);
        for k in [
            k,
            linear,
            base,
            guest,
            mixed,
            sampler_key(&point, None, false),
        ] {
            assert_eq!(sampler_diagnostics(k, 1, false), k);
        }
        for k in [base, guest, mixed, sampler_key(&point, None, false)] {
            assert_eq!(sampler_diagnostics(k, 16, false), k);
        }
        // Forced linear turns point filters linear, and then forced
        // anisotropy applies.
        let k = sampler_diagnostics(sampler_key(&point, None, false), 4, true);
        assert_eq!((k.mag, k.min, k.anisotropy), (Linear, Linear, 4));
        assert_eq!(force_aniso_from(None), 1);
        assert_eq!(force_aniso_from(Some("16")), 16);
        assert_eq!(force_aniso_from(Some("3")), 1);
    }

    /// Packed filter overrides tell no override from all-zero ones and
    /// keep every field.
    #[test]
    fn packed_filters_differ() {
        assert_eq!(pack_filters(None), 0);
        assert_ne!(pack_filters(Some([0; 4])), 0);
        let base = pack_filters(Some([1, 2, 3, 4]));
        for i in 0..4 {
            let mut f = [1, 2, 3, 4];
            f[i] = 0;
            assert_ne!(pack_filters(Some(f)), base, "field {i}");
        }
    }

    /// `meets` over merged ranges agrees with checking every range.
    #[test]
    fn merged_ranges_meet_what_any_range_overlaps() {
        let raw = vec![(40, 50), (0, 4), (4, 8), (20, 30), (25, 35), (90, 91)];
        let merged = merge_ranges(raw.clone());
        assert_eq!(merged, [(0, 8), (20, 35), (40, 50), (90, 91)]);
        for a in 0..100u64 {
            for b in a + 1..=100 {
                let any = raw.iter().any(|&(lo, hi)| lo < b && a < hi);
                assert_eq!(meets(&merged, a, b), any, "[{a}, {b})");
            }
        }
        assert!(!meets(&[], 0, 10));
        assert!(merge_ranges(Vec::new()).is_empty());
    }

    /// Every byte, in every lane and in the tail, and the length change
    /// the hash.
    #[test]
    fn hash_sees_every_byte() {
        let base: Vec<u8> = (0..77u8).collect();
        let h = hash_bytes(&base);
        for i in 0..base.len() {
            let mut b = base.clone();
            b[i] ^= 1;
            assert_ne!(hash_bytes(&b), h, "byte {i}");
        }
        assert_ne!(hash_bytes(&base[..76]), h);
        assert_ne!(hash_bytes(&[0; 32]), hash_bytes(&[0; 64]));
    }

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
        let mut want: Vec<u32> = block(regs::SHADER_CONSTANT_FETCH, 192)
            .chain(block(regs::SHADER_CONSTANT_BOOL, 8))
            .chain(block(regs::SHADER_CONSTANT_LOOP, 32))
            .collect();
        want.extend([0x1234_5678, 77, 0, 0]);
        let rebase = std::array::from_fn(|k| k as i32 - 3);
        want.extend(rebase.map(|w: i32| w as u32));
        let mut got = consts_prefix(r);
        got.extend(to_bytes(&consts_suffix(r, 0x1234_5678, &rebase)));
        assert_eq!(got, to_bytes(&want));
        assert_eq!(got.len(), CONSTS_BYTES);
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
        let stage = |d: &mut Drawer, part: &[u8]| {
            d.reserve(d.staged_size(part.len() as u64)).unwrap();
            d.push_bytes(part)
        };
        let part = vec![7u8; 3 << 20];
        for _ in 0..12 {
            stage(&mut d, &part);
            let arena = d.arena.as_ref().unwrap().size();
            assert!(
                d.staged.capacity() as u64 <= arena,
                "{}",
                d.staged.capacity()
            );
        }
        let big = vec![1u8; 20 << 20];
        stage(&mut d, &big);
        stage(&mut d, &part);
        assert_eq!(d.arena.as_ref().unwrap().size(), 32 << 20);
        assert!(d.staged.capacity() <= 32 << 20, "{}", d.staged.capacity());
        d.flush().unwrap();
    }

    /// Each part is padded with zeros to a multiple of 4 (at least 4) and
    /// starts at the arena's alignment; a part whose fill fails leaves
    /// nothing behind.
    #[test]
    fn pushed_parts_are_padded_and_aligned() {
        let Some(mut d) = gpu() else { return };
        d.wrap();
        let align = d.arena_align as usize;
        assert_eq!(d.push_bytes(&[1, 2, 3, 4, 5]), (0, 8));
        assert_eq!(d.staged.len(), align);
        assert_eq!(d.staged[..8], [1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(d.push(12, |_| false), None);
        assert_eq!(d.staged.len(), align);
        assert_eq!(d.push_bytes(&[]), (align as u64, 4));
        let s = d.push(3, |b| {
            b.copy_from_slice(&[9; 3]);
            true
        });
        assert_eq!(s, Some((2 * align as u64, 4)));
        assert_eq!(d.staged[2 * align..2 * align + 4], [9, 9, 9, 0]);
        assert_eq!(d.staged_size(5), align as u64);
        d.wrap();
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
    fn spares_take_the_newest_match_and_free_the_oldest_past_the_caps() {
        let mut s = Spares::new(3, 10);
        s.put('a', 2, 1);
        s.put('b', 2, 2);
        s.put('a', 2, 3);
        assert_eq!(s.take(&'a'), Some(3));
        assert_eq!(s.take(&'c'), None);
        assert_eq!(s.bytes, 4);
        // Past the count the oldest goes (1), then past the bytes (2).
        s.put('c', 2, 4);
        s.put('c', 2, 5);
        assert_eq!(s.take(&'a'), None);
        s.put('d', 6, 6);
        assert_eq!(s.take(&'b'), None);
        assert_eq!((s.items.len(), s.bytes), (3, 10));
        s.put('d', 1, 7);
        assert_eq!((s.take(&'c'), s.take(&'c')), (Some(5), None));
        // One bigger than the byte cap is not kept.
        s.put('e', 11, 7);
        assert_eq!((s.items.len(), s.bytes), (0, 0));
        s.put('f', 1, 8);
        s.put('g', 2, 9);
        s.trim(usize::MAX, 2);
        assert_eq!((s.take(&'f'), s.bytes), (None, 2));
        s.trim(usize::MAX, 0);
        assert_eq!((s.take(&'g'), s.bytes), (None, 0));
    }

    /// The spares count against the target cap: past it with the live
    /// targets under it, the oldest spares go and the targets stay.
    #[test]
    fn target_cap_counts_the_spares() {
        let Some(mut d) = gpu() else { return };
        let key = |base_tiles| RenderTargetKey {
            base_tiles,
            pitch: 64,
            format: TargetFormat::Rgba8,
        };
        let mut report = Report::default();
        for base in [0, 100, 200, 300, 400] {
            d.target(key(base), 64, 0, &mut report).unwrap();
        }
        for base in [0, 100, 200] {
            d.mark_written(key(base), 64, 0);
        }
        for base in [300, 400] {
            let tex = d.targets.remove(&key(base)).unwrap();
            d.spare_targets
                .put(spare_key(&tex), texture_bytes(&tex), tex);
        }
        // 64 x 2048 x 4 bytes each: three live and two spare are 2.5 MB.
        d.target_cap = 2 << 20;
        d.drop_targets(&mut report);
        assert_eq!(d.render_targets().len(), 3);
        assert_eq!(d.spare_targets.items.len(), 1);
        let live: u64 = d.targets.values().map(texture_bytes).sum();
        assert!(live + d.spare_targets.bytes <= d.target_cap);
        assert!(report.warnings.is_empty(), "{report:?}");
        d.flush().unwrap();
    }

    /// A new target that takes a dropped one's texture starts zeroed, as
    /// a made one does, not with the old pixels.
    #[test]
    fn reused_target_texture_starts_cleared() {
        let Some(mut d) = gpu() else { return };
        let key = |base_tiles| RenderTargetKey {
            base_tiles,
            pitch: 64,
            format: TargetFormat::Rgba8,
        };
        let mut report = Report::default();
        let tex = d.target(key(0), 64, 0, &mut report).unwrap();
        d.flush().unwrap();
        d.queue.write_texture(
            tex.as_image_copy(),
            &vec![0xAB; 64 * 4 * tex.height() as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: None,
            },
            tex.size(),
        );
        assert_eq!(d.read_render_target(key(0), 0, 0, 1, 1).unwrap(), [0xAB; 4]);
        let tex = d.targets.remove(&key(0)).unwrap();
        d.spare_targets
            .put(spare_key(&tex), texture_bytes(&tex), tex);
        d.target(key(100), 64, 0, &mut report).unwrap();
        d.flush().unwrap();
        assert!(d.spare_targets.items.is_empty());
        assert_eq!(d.read_render_target(key(100), 0, 0, 2, 2).unwrap(), [0; 16]);
        let m = d.memory_line();
        assert!(m.contains(", 1 made, 1 reused, 0 spare (0 MB),"), "{m}");
    }

    /// Texel (0, 0) of a depth target: depth (loaded in a compute pass, as
    /// Depth24PlusStencil8 depth cannot be copied out) and stencil (0
    /// without a stencil aspect).
    fn read_depth_stencil(d: &Drawer, tex: &wgpu::Texture) -> (f32, u8) {
        let module = d.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var depth: texture_depth_2d;\n\
                 @group(0) @binding(1) var<storage, read_write> out: array<f32>;\n\
                 @compute @workgroup_size(1)\n\
                 fn main() { out[0] = textureLoad(depth, vec2<i32>(0, 0), 0); }\n"
                    .into(),
            ),
        });
        let pipeline = d
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let buffer = |usage| {
            d.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 + 256 * u64::from(tex.height()),
                usage,
                mapped_at_creation: false,
            })
        };
        let out = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let read = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let group = d.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut enc = d.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, Some(4));
        if tex.format().has_stencil_aspect() {
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::StencilOnly,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 256,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                // Depth and stencil copies take the whole texture.
                tex.size(),
            );
        }
        let scope = d.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let cmd = enc.finish();
        if let Some(e) = block_on(scope.pop()) {
            panic!("{e}");
        }
        let data = d.map_read(&read, d.submit_now([cmd])).unwrap();
        (f32::from_le_bytes(data[..4].try_into().unwrap()), data[256])
    }

    /// A depth target that takes a dropped one's texture starts with depth
    /// and stencil 0, and its clear passes validation.
    #[test]
    fn reused_depth_target_texture_starts_cleared() {
        let Some(mut d) = gpu() else { return };
        for format in [TargetFormat::D24S8, TargetFormat::D24FS8] {
            let key = |base_tiles| RenderTargetKey {
                base_tiles,
                pitch: 64,
                format,
            };
            let mut report = Report::default();
            let tex = d.target(key(0), 64, 0, &mut report).unwrap();
            d.flush().unwrap();
            let stencil = tex.format().has_stencil_aspect();
            let view = tex.create_view(&Default::default());
            let mut enc = d.device.create_command_encoder(&Default::default());
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.75),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: stencil.then_some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0x5A),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                ..Default::default()
            });
            d.submit_now([enc.finish()]);
            let want = if stencil { 0x5A } else { 0 };
            assert_eq!(read_depth_stencil(&d, &tex), (0.75, want), "{format:?}");
            let tex = d.targets.remove(&key(0)).unwrap();
            d.spare_targets
                .put(spare_key(&tex), texture_bytes(&tex), tex);
            let tex = d.target(key(100), 64, 0, &mut report).unwrap();
            d.flush().unwrap();
            assert!(d.spare_targets.items.is_empty(), "{format:?}");
            assert_eq!(read_depth_stencil(&d, &tex), (0.0, 0), "{format:?}");
            // A validation error on the submit thread comes out here.
            d.flush().unwrap();
            assert!(report.warnings.is_empty(), "{report:?}");
            d.targets.clear();
        }
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
        let ps = format!("struct PsOut {{\n{DEPTH_MEMBER}}}\n{DEPTH_STORE}    return out;\n");
        assert_eq!(
            drop_depth_wgsl(&ps).unwrap(),
            "struct PsOut {\n}\n    return out;\n"
        );
        assert!(drop_depth_wgsl(DEPTH_MEMBER).is_err());
    }

    #[test]
    fn index_range_skips_resets_and_applies_the_offset() {
        let range = |i: &[u32], o| offset_range(min_max(i), o);
        assert_eq!(range(&[7, u32::MAX, 3, 9], 0), Some((3, 9)));
        assert_eq!(range(&[u32::MAX], 0), None);
        assert_eq!(range(&[1, 2], 10), Some((11, 12)));
        // The shader keeps 24 bits of the 32-bit sum.
        assert_eq!(range(&[1, 2], 0xFF00_0000), Some((1, 2)));
        assert_eq!(range(&[0, 5], 0xFF_FFFE), None);
        assert_eq!(range(&[0, 5], u32::MAX - 1), None);
        assert_eq!(range(&[3, 5], u32::MAX - 1), Some((1, 3)));
    }

    #[test]
    fn fetch_range_bounds_clips_and_falls_back() {
        let r = |stride, bounded, first, end| VertexFetchReach {
            fetch_const: 2,
            stride,
            bounded,
            first,
            end,
        };
        let a = 0x1000;
        // Vertices 4..=6 of stride 4, words 0..3 and 1..4.
        let reach = [r(4, true, 0, 3), r(4, true, 1, 4)];
        assert_eq!(
            fetch_range(a, 256, 2, &reach, Some((4, 6))),
            (a + 64, a + 112, false)
        );
        // Another constant's reach, or unknown indices: the whole buffer,
        // which such a fetch may read past.
        let whole = (a, a + 1024, true);
        assert_eq!(fetch_range(a, 256, 3, &reach, Some((4, 6))), whole);
        assert_eq!(fetch_range(a, 256, 2, &reach, None), whole);
        assert_eq!(
            fetch_range(a, 256, 2, &[r(4, false, 0, 1)], Some((0, 0))),
            whole
        );
        // Stride 0 needs no indices; a negative offset and an end past the
        // buffer are clipped.
        assert_eq!(
            fetch_range(a, 256, 2, &[r(0, true, 5, 7)], None),
            (a + 20, a + 28, false)
        );
        assert_eq!(
            fetch_range(a, 16, 2, &[r(4, true, -2, 1)], Some((0, 9))),
            (a, a + 64, true)
        );
        // Entirely past the buffer: nothing, and clipped.
        let (lo, hi, clipped) = fetch_range(a, 16, 2, &[r(4, true, 0, 1)], Some((20, 30)));
        assert_eq!((lo == hi, clipped), (true, true));
    }
}
