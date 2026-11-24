// Register addresses taken from Xenia src/xenia/gpu/register_table.inc and the
// register file size from src/xenia/gpu/register_file.h (BSD 3-Clause, see
// CREDITS.md).

//! Xenos register addresses (in dwords, as PM4 packets address them) and the
//! register file the recorder keeps.

/// Number of dword registers, as Xenia's `RegisterFile::kRegisterCount`.
pub const REGISTER_COUNT: usize = 0x5003;

// Command processor ring buffer. Only CP_RB_WPTR (0x01C5) is named in Xenia
// (graphics_system.cc); CP_RB_CNTL and CP_RB_RPTR_ADDR come from the byte
// offsets 0x704 and 0x70C in command_processor.cc comments. CP_RB_BASE and
// CP_RB_RPTR follow the same Adreno (yamato) layout and are not in Xenia's table.
pub const CP_RB_BASE: u32 = 0x01C0;
pub const CP_RB_CNTL: u32 = 0x01C1;
pub const CP_RB_RPTR_ADDR: u32 = 0x01C3;
pub const CP_RB_RPTR: u32 = 0x01C4;
pub const CP_RB_WPTR: u32 = 0x01C5;

pub const SCRATCH_UMSK: u32 = 0x01DC;
pub const SCRATCH_ADDR: u32 = 0x01DD;
pub const SCRATCH_REG0: u32 = 0x0578;
pub const SCRATCH_REG7: u32 = 0x057F;
pub const COHER_SIZE_HOST: u32 = 0x0A2F;
pub const COHER_BASE_HOST: u32 = 0x0A30;
pub const COHER_STATUS_HOST: u32 = 0x0A31;
pub const PA_SC_VIZ_QUERY_STATUS_0: u32 = 0x0C44;
pub const PA_SC_VIZ_QUERY_STATUS_1: u32 = 0x0C45;
pub const RB_EDRAM_INFO: u32 = 0x0F02;

pub const RB_SURFACE_INFO: u32 = 0x2000;
pub const RB_COLOR_INFO: u32 = 0x2001;
pub const RB_DEPTH_INFO: u32 = 0x2002;
pub const RB_COLOR1_INFO: u32 = 0x2003;
pub const RB_COLOR2_INFO: u32 = 0x2004;
pub const RB_COLOR3_INFO: u32 = 0x2005;
/// The four color render target info registers, by render target index.
pub const RB_COLOR_INFO_RT: [u32; 4] = [
    RB_COLOR_INFO,
    RB_COLOR1_INFO,
    RB_COLOR2_INFO,
    RB_COLOR3_INFO,
];
pub const PA_SC_SCREEN_SCISSOR_TL: u32 = 0x200E;
pub const PA_SC_SCREEN_SCISSOR_BR: u32 = 0x200F;
pub const PA_SC_WINDOW_OFFSET: u32 = 0x2080;
pub const PA_SC_WINDOW_SCISSOR_TL: u32 = 0x2081;
pub const PA_SC_WINDOW_SCISSOR_BR: u32 = 0x2082;
pub const VGT_MAX_VTX_INDX: u32 = 0x2100;
pub const VGT_MIN_VTX_INDX: u32 = 0x2101;
pub const VGT_INDX_OFFSET: u32 = 0x2102;
pub const VGT_MULTI_PRIM_IB_RESET_INDX: u32 = 0x2103;
pub const RB_COLOR_MASK: u32 = 0x2104;
pub const RB_BLEND_RED: u32 = 0x2105;
pub const RB_BLEND_GREEN: u32 = 0x2106;
pub const RB_BLEND_BLUE: u32 = 0x2107;
pub const RB_BLEND_ALPHA: u32 = 0x2108;
pub const RB_STENCILREFMASK_BF: u32 = 0x210C;
pub const RB_STENCILREFMASK: u32 = 0x210D;
pub const RB_ALPHA_REF: u32 = 0x210E;
pub const PA_CL_VPORT_XSCALE: u32 = 0x210F;
pub const PA_CL_VPORT_XOFFSET: u32 = 0x2110;
pub const PA_CL_VPORT_YSCALE: u32 = 0x2111;
pub const PA_CL_VPORT_YOFFSET: u32 = 0x2112;
pub const PA_CL_VPORT_ZSCALE: u32 = 0x2113;
pub const PA_CL_VPORT_ZOFFSET: u32 = 0x2114;
pub const SQ_PROGRAM_CNTL: u32 = 0x2180;
pub const SQ_CONTEXT_MISC: u32 = 0x2181;
pub const VGT_EVENT_INITIATOR: u32 = 0x21F9;
pub const VGT_DMA_BASE: u32 = 0x21FA;
pub const VGT_DMA_SIZE: u32 = 0x21FB;
pub const VGT_DRAW_INITIATOR: u32 = 0x21FC;
pub const RB_DEPTHCONTROL: u32 = 0x2200;
pub const RB_BLENDCONTROL0: u32 = 0x2201;
pub const RB_COLORCONTROL: u32 = 0x2202;
pub const PA_CL_CLIP_CNTL: u32 = 0x2204;
pub const PA_SU_SC_MODE_CNTL: u32 = 0x2205;
pub const PA_CL_VTE_CNTL: u32 = 0x2206;
pub const RB_MODECONTROL: u32 = 0x2208;
pub const RB_BLENDCONTROL1: u32 = 0x2209;
pub const RB_BLENDCONTROL2: u32 = 0x220A;
pub const RB_BLENDCONTROL3: u32 = 0x220B;
/// The four blend control registers, by render target index.
pub const RB_BLENDCONTROL_RT: [u32; 4] = [
    RB_BLENDCONTROL0,
    RB_BLENDCONTROL1,
    RB_BLENDCONTROL2,
    RB_BLENDCONTROL3,
];
pub const PA_SU_POINT_SIZE: u32 = 0x2280;
pub const PA_SU_POINT_MINMAX: u32 = 0x2281;
pub const PA_SU_LINE_CNTL: u32 = 0x2282;
pub const VGT_OUTPUT_PATH_CNTL: u32 = 0x2284;
pub const PA_SC_VIZ_QUERY: u32 = 0x2293;
pub const PA_SC_AA_CONFIG: u32 = 0x2301;
pub const PA_SU_VTX_CNTL: u32 = 0x2302;
pub const PA_CL_GB_VERT_CLIP_ADJ: u32 = 0x2303;
pub const PA_CL_GB_VERT_DISC_ADJ: u32 = 0x2304;
pub const PA_CL_GB_HORZ_CLIP_ADJ: u32 = 0x2305;
pub const PA_CL_GB_HORZ_DISC_ADJ: u32 = 0x2306;
pub const PA_SC_AA_MASK: u32 = 0x2312;
pub const RB_COPY_CONTROL: u32 = 0x2318;
pub const RB_COPY_DEST_BASE: u32 = 0x2319;
pub const RB_COPY_DEST_PITCH: u32 = 0x231A;
pub const RB_COPY_DEST_INFO: u32 = 0x231B;
pub const RB_DEPTH_CLEAR: u32 = 0x231D;
pub const RB_COLOR_CLEAR: u32 = 0x231E;
pub const RB_COLOR_CLEAR_LO: u32 = 0x231F;
pub const RB_COPY_FUNC: u32 = 0x2320;
pub const RB_COPY_REF: u32 = 0x2321;
pub const RB_COPY_MASK: u32 = 0x2322;
pub const RB_COPY_SURFACE_SLICE: u32 = 0x2323;
pub const RB_SAMPLE_COUNT_CTL: u32 = 0x2324;
pub const RB_SAMPLE_COUNT_ADDR: u32 = 0x2325;
pub const PA_SU_POLY_OFFSET_FRONT_SCALE: u32 = 0x2380;
pub const PA_SU_POLY_OFFSET_FRONT_OFFSET: u32 = 0x2381;
pub const PA_SU_POLY_OFFSET_BACK_SCALE: u32 = 0x2382;
pub const PA_SU_POLY_OFFSET_BACK_OFFSET: u32 = 0x2383;

/// 512 float4 ALU constants (`SHADER_CONSTANT_000_X` .. `SHADER_CONSTANT_511_W`).
pub const SHADER_CONSTANT_FLOAT: u32 = 0x4000;
/// 32 six-dword fetch constants (`SHADER_CONSTANT_FETCH_00_0` .. `_31_5`).
pub const SHADER_CONSTANT_FETCH: u32 = 0x4800;
/// 8 dwords of 256 boolean constants (`SHADER_CONSTANT_BOOL_000_031` ..).
pub const SHADER_CONSTANT_BOOL: u32 = 0x4900;
/// 32 loop constants (`SHADER_CONSTANT_LOOP_00` .. `_31`).
pub const SHADER_CONSTANT_LOOP: u32 = 0x4908;

/// RB_MODECONTROL edram_mode value (bits 0:2) that turns a draw into a resolve.
pub const EDRAM_MODE_COPY: u32 = 6;

/// Every GPU register value, indexed by dword register address.
#[derive(Clone)]
pub struct RegisterFile {
    values: Box<[u32]>,
}

impl Default for RegisterFile {
    fn default() -> Self {
        Self {
            values: vec![0; REGISTER_COUNT].into_boxed_slice(),
        }
    }
}

impl std::fmt::Debug for RegisterFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nonzero = self.values.iter().filter(|v| **v != 0).count();
        write!(f, "RegisterFile {{ {nonzero} non-zero }}")
    }
}

impl RegisterFile {
    /// Value of register `index`; registers past the end read as 0.
    pub fn get(&self, index: u32) -> u32 {
        self.values.get(index as usize).copied().unwrap_or(0)
    }

    /// Sets register `index`. Returns false (and writes nothing) when out of
    /// range, as Xenia's `WriteRegister` drops such writes.
    pub fn set(&mut self, index: u32, value: u32) -> bool {
        match self.values.get_mut(index as usize) {
            Some(slot) => {
                *slot = value;
                true
            }
            None => false,
        }
    }

    /// The six dwords of fetch constant `n` (0..32): a texture fetch
    /// constant, or three two-dword vertex fetch constants.
    pub fn fetch_constant(&self, n: u32) -> [u32; 6] {
        // Saturating: an out-of-range `n` reads zeros instead of panicking.
        let base = SHADER_CONSTANT_FETCH.saturating_add(n.saturating_mul(6));
        std::array::from_fn(|i| self.get(base.saturating_add(i as u32)))
    }

    /// The two dwords of vertex fetch constant `n` (0..96).
    pub fn vertex_fetch(&self, n: u32) -> crate::VertexFetch {
        let base = SHADER_CONSTANT_FETCH.saturating_add(n.saturating_mul(2));
        crate::VertexFetch::decode([self.get(base), self.get(base.saturating_add(1))])
    }
}
