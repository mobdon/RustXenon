//! Synthetic command streams, shaders and memory for the drawer tests and
//! the triangle example. Nothing here comes from a game.

#![allow(dead_code)]

use xenon_draw::{Drawer, Error, GuestMemory, RenderTargetKey, Report, TargetFormat};
use xenon_gpu::pm4::{build, PM4_DRAW_INDX, PM4_DRAW_INDX_2, PM4_IM_LOAD_IMMEDIATE, PM4_XE_SWAP};
use xenon_gpu::regs::*;
use xenon_gpu::{Command, Recorder, SliceMemory};
use xenon_shader::asm::*;
use xenon_shader::ucode::*;

/// Guest memory layout of the tests.
pub const MEM_SIZE: usize = 0x10_0000;
pub const CMD: u32 = 0x1000;
pub const VERTS: u32 = 0x2_0000;
pub const INDICES: u32 = 0x3_0000;
pub const RECT: u32 = 0x3_8000;
pub const TEXTURE: u32 = 0x4_0000;
pub const DEST: u32 = 0x8_0000;

/// Render target size the tests draw in.
pub const W: u32 = 64;
pub const H: u32 = 64;
pub const COLOR_BASE: u32 = 0;
pub const DEPTH_BASE: u32 = 512;

pub const COLOR_KEY: RenderTargetKey = RenderTargetKey {
    base_tiles: COLOR_BASE,
    pitch: W,
    format: TargetFormat::Rgba8,
};

/// A drawer that holds the test GPU lock while it lives.
///
/// One drawer at a time: on GitHub's runner the software Vulkan driver
/// crashed (SIGSEGV) once with many devices alive in parallel test threads,
/// and the host only ever uses one device.
pub struct TestDrawer {
    drawer: Drawer,
    // Dropped after `drawer`, so the device is gone before the next test.
    _gpu: std::sync::MutexGuard<'static, ()>,
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

/// A drawer, or None (after a skip line) when there is no adapter and
/// XENON_REQUIRE_GPU is not 1.
pub fn drawer() -> Option<TestDrawer> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    match Drawer::new() {
        Ok(drawer) => Some(TestDrawer { drawer, _gpu: gpu }),
        Err(Error::NoAdapter) if std::env::var("XENON_REQUIRE_GPU").as_deref() != Ok("1") => {
            eprintln!("SKIP: no wgpu adapter (set XENON_REQUIRE_GPU=1 to fail instead)");
            None
        }
        Err(e) => panic!("no drawer: {e}"),
    }
}

pub fn f(v: f32) -> u32 {
    v.to_bits()
}

/// RGBA bytes -> the 8_8_8_8 value with R in the low byte.
pub fn rgba(c: [u8; 4]) -> u32 {
    u32::from_le_bytes(c)
}

/// A command stream under construction.
#[derive(Default)]
pub struct Stream(pub Vec<u32>);

impl Stream {
    pub fn regs(&mut self, base: u32, values: &[u32]) -> &mut Self {
        self.0.extend(build::type0(base, values));
        self
    }

    pub fn shader(&mut self, pixel: bool, words: &[u32]) -> &mut Self {
        let mut p = vec![u32::from(pixel), words.len() as u32];
        p.extend_from_slice(words);
        self.0.extend(build::type3(PM4_IM_LOAD_IMMEDIATE, &p));
        self
    }

    /// Vertex fetch constant `k` (two dwords) for `dwords` at `addr`, k8in32.
    pub fn vertex_fetch(&mut self, k: u32, addr: u32, dwords: u32) -> &mut Self {
        self.regs(
            SHADER_CONSTANT_FETCH + 2 * k,
            &[addr | 3, (dwords << 2) | 2],
        )
    }

    /// Render state shared by the tests: 64x64 colour target at tile 0,
    /// D24S8 depth at tile 512, no blending, no depth, no culling, full
    /// scissor, and positions in window pixels (viewport transform off,
    /// OpenGL pixel centres).
    pub fn setup(&mut self) -> &mut Self {
        self.regs(RB_MODECONTROL, &[4])
            .regs(RB_SURFACE_INFO, &[W, COLOR_BASE, DEPTH_BASE])
            .regs(RB_COLOR_MASK, &[0xF])
            .regs(RB_BLENDCONTROL0, &[0x0001_0001])
            .regs(RB_DEPTHCONTROL, &[0])
            .regs(PA_SU_SC_MODE_CNTL, &[0])
            .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, W | (H << 16)])
            .regs(PA_SC_SCREEN_SCISSOR_TL, &[0, 8192 | (8192 << 16)])
            .regs(PA_CL_VTE_CNTL, &[1 << 10])
            .regs(PA_SU_VTX_CNTL, &[1])
    }

    /// DRAW_INDX_2 with auto-generated indices.
    pub fn draw_auto(&mut self, prim: u32, count: u32) -> &mut Self {
        self.0.extend(build::type3(
            PM4_DRAW_INDX_2,
            &[prim | (2 << 6) | (count << 16)],
        ));
        self
    }

    /// DRAW_INDX with a 16-bit DMA index buffer at INDICES, k8in16.
    pub fn draw_dma16(&mut self, prim: u32, count: u32) -> &mut Self {
        self.0.extend(build::type3(
            PM4_DRAW_INDX,
            &[0, prim | (count << 16), INDICES, count | (1 << 30)],
        ));
        self
    }

    /// A Direct3D-style resolve of the (0,0)-(w,h) rectangle: the corners go
    /// to RECT and vertex fetch 0, RB_MODECONTROL switches to copy mode for
    /// one rectangle-list draw, then back. `copy` false clears only.
    pub fn resolve(
        &mut self,
        mem: &mut SliceMemory,
        w: u32,
        h: u32,
        copy: bool,
        clear: Option<(u32, u32)>,
    ) -> &mut Self {
        let mut control = if copy { 0 } else { COPY_NULL };
        let (color_clear, depth_clear) = clear.unwrap_or((0, 0));
        if clear.is_some() {
            control |= CLEAR_COLOR | CLEAR_DEPTH;
        }
        // Destination info: k8in32, k_8_8_8_8.
        self.resolve_with(
            mem,
            (w, h),
            control,
            (DEST, W | (H << 16), dest_info(6, 0, false)),
            [depth_clear, color_clear, 0],
        )
    }

    /// A resolve of the (0,0)-(w,h) rectangle with explicit registers:
    /// RB_COPY_CONTROL, (RB_COPY_DEST_BASE, RB_COPY_DEST_PITCH,
    /// RB_COPY_DEST_INFO) and (RB_DEPTH_CLEAR, RB_COLOR_CLEAR,
    /// RB_COLOR_CLEAR_LO).
    pub fn resolve_with(
        &mut self,
        mem: &mut SliceMemory,
        (w, h): (u32, u32),
        control: u32,
        (dest, pitch_height, info): (u32, u32, u32),
        clear: [u32; 3],
    ) -> &mut Self {
        let (wf, hf) = (w as f32, h as f32);
        mem.write_words(RECT, &[f(0.0), f(0.0), f(wf), f(0.0), f(wf), f(hf)]);
        self.vertex_fetch(0, RECT, 6)
            .regs(RB_MODECONTROL, &[6])
            .regs(RB_COPY_CONTROL, &[control, dest, pitch_height, info])
            .regs(RB_DEPTH_CLEAR, &clear)
            .draw_auto(8, 3)
            .regs(RB_MODECONTROL, &[4])
    }

    pub fn swap(&mut self, w: u32, h: u32) -> &mut Self {
        self.swap_at(DEST, w, h)
    }

    pub fn swap_at(&mut self, addr: u32, w: u32, h: u32) -> &mut Self {
        self.0.extend(build::type3(
            PM4_XE_SWAP,
            &[xenon_gpu::pm4::SWAP_SIGNATURE, addr, w, h],
        ));
        self
    }
}

/// RB_COPY_CONTROL copy commands and clear enables.
pub const COPY_RAW: u32 = 0;
pub const COPY_CONVERT: u32 = 1 << 20;
pub const COPY_NULL: u32 = 3 << 20;
pub const CLEAR_COLOR: u32 = 1 << 8;
pub const CLEAR_DEPTH: u32 = 1 << 9;

/// RB_COPY_DEST_INFO: k8in32, `format` (`xenos::ColorFormat`), exponent
/// bias (6-bit signed) and red/blue swap.
pub fn dest_info(format: u32, exp_bias: i32, swap: bool) -> u32 {
    2 | (format << 7) | (((exp_bias as u32) & 0x3F) << 16) | (u32::from(swap) << 24)
}

/// `w` x `h` texels of `bytes` each at `addr`, tiled with pitch `pitch`
/// and k8in32, back to linear rows.
pub fn detiled(mem: &SliceMemory, addr: u32, w: u32, h: u32, pitch: u32, bytes: u32) -> Vec<u8> {
    use xenon_texture::{endian, tile};
    let extent = tile::tiled_extent_2d(w, h, pitch, bytes).unwrap() as usize;
    let mut raw = mem.data[(addr - mem.base) as usize..][..extent].to_vec();
    endian::swap_in_place(&mut raw, xenon_texture::Endian::K8in32);
    tile::detile_2d(&raw, w, h, pitch, bytes).unwrap()
}

pub fn memory() -> SliceMemory {
    SliceMemory {
        base: 0,
        data: vec![0; MEM_SIZE],
    }
}

/// Records `s` from guest memory and replays it.
pub fn run(d: &mut Drawer, s: &Stream, mem: &mut SliceMemory) -> Report {
    mem.write_words(CMD, &s.0);
    let mut rec = Recorder::new();
    rec.record_buffer(mem, CMD, s.0.len() as u32).unwrap();
    assert!(rec.warnings.is_empty(), "{:?}", rec.warnings);
    let cmds: Vec<Command> = rec.commands;
    d.replay(&cmds, mem as &mut dyn GuestMemory).unwrap()
}

pub fn exec(addr: u32, count: u32, end: bool) -> CfInstr {
    CfInstr::Exec(Exec::new(addr, count, end))
}

fn r(n: u32) -> AluSrc {
    AluSrc::temp(n, SWZ_XYZW)
}

pub fn export(reg: u32, src: AluSrc) -> Instr {
    Instr::Alu(AluInstr::vector(VectorOp::Max, reg, mask("xyzw"), &[src, src]).exported(reg))
}

/// Vertex shader: vertex = 4 dwords at vf0 (x, y, z float, 8_8_8_8
/// colour); oPos = (x, y, z, 1), o0 = colour.
pub fn vs_color() -> Vec<u32> {
    vs_color_from(0)
}

/// `vs_color` reading its vertices through vertex fetch constant `k`.
pub fn vs_color_from(k: u32) -> Vec<u32> {
    let mut pos = VertexFetch::full(1, 0, k, VertexFormat::F32_32_32_FLOAT, 4, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let mut col = VertexFetch::full(2, 0, 0, VertexFormat::F8_8_8_8, 4, 3);
    col.mini = true;
    assemble(
        &[exec(0, 4, true)],
        &[
            Instr::Fetch(FetchInstr::Vertex(pos)),
            Instr::Fetch(FetchInstr::Vertex(col)),
            export(62, r(1)),
            export(0, r(2)),
        ],
    )
}

/// Vertex shader: vertex = 5 dwords at vf0 (x, y, z, u, v floats);
/// oPos = (x, y, z, 1), o0 = (u, v, 0, 1).
pub fn vs_uv() -> Vec<u32> {
    let mut pos = VertexFetch::full(1, 0, 0, VertexFormat::F32_32_32_FLOAT, 5, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let mut uv = VertexFetch::full(2, 0, 0, VertexFormat::F32_32_FLOAT, 5, 3);
    uv.mini = true;
    uv.dest_swizzle = fetch_dest_swz("xy01");
    assemble(
        &[exec(0, 4, true)],
        &[
            Instr::Fetch(FetchInstr::Vertex(pos)),
            Instr::Fetch(FetchInstr::Vertex(uv)),
            export(62, r(1)),
            export(0, r(2)),
        ],
    )
}

/// Pixel shader: oC0 = interpolator 0.
pub fn ps_interp() -> Vec<u32> {
    assemble(&[exec(0, 1, true)], &[export(0, r(0))])
}

/// Pixel shader: oC0 = pixel constant c0.
pub fn ps_const() -> Vec<u32> {
    assemble(
        &[exec(0, 1, true)],
        &[export(0, AluSrc::constant(0, SWZ_XYZW))],
    )
}

/// Pixel shader: oC0 = texture slot `slot` at interpolator 0's xy.
pub fn ps_texture(slot: u32) -> Vec<u32> {
    assemble(
        &[exec(0, 2, true)],
        &[
            Instr::Fetch(FetchInstr::Texture(TextureFetch::fetch_2d(1, 0, slot))),
            export(0, r(1)),
        ],
    )
}

/// Vertex words for `vs_color`: (x, y, z, colour) per vertex.
pub fn color_verts(v: &[(f32, f32, f32, [u8; 4])]) -> Vec<u32> {
    v.iter()
        .flat_map(|&(x, y, z, c)| [f(x), f(y), f(z), rgba(c)])
        .collect()
}

/// Pixel (x, y) of a `w`-wide RGBA8 image.
pub fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * w + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2], img[i + 3]]
}

pub fn near(a: [u8; 4], b: [u8; 4], tol: u8) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= tol)
}
