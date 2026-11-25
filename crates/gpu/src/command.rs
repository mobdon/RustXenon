// Command fields follow the packet handlers in Xenia
// src/xenia/gpu/command_processor.cc; the resolve register bitfields follow
// src/xenia/gpu/registers.h (RB_COPY_*, RB_SURFACE_INFO, RB_COLOR_INFO,
// RB_DEPTH_INFO, PA_SC_WINDOW_*) as used by GetResolveInfo in draw_util.cc
// (BSD 3-Clause, see CREDITS.md).

//! The typed commands the recorder produces.

use crate::regs::{self, RegisterFile};
use crate::xenos::{sign_extend, CopyCommand, Endian, IndexFormat, PrimitiveType, VertexFetch};

/// One recorded GPU command, in stream order.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub enum Command {
    /// Consecutive register writes from `base`. Covers type-0/1 packets,
    /// SET_CONSTANT(2), SET_SHADER_CONSTANTS, LOAD_ALU_CONSTANT, REG_RMW and
    /// the registers other packets write as a side effect (draw initiator,
    /// event initiator, viz query status). Replaying these in order rebuilds
    /// the register file.
    WriteRegisters {
        base: u32,
        values: Vec<u32>,
    },
    Draw(Draw),
    /// A draw issued while RB_MODECONTROL is in copy mode: an EDRAM resolve.
    Resolve(Box<Resolve>),
    LoadShader {
        stage: ShaderStage,
        /// Physical address for IM_LOAD; None for IM_LOAD_IMMEDIATE.
        source_addr: Option<u32>,
        /// Start offset in instruction memory (Xenia expects 0).
        start: u32,
        words: Vec<u32>,
    },
    /// PM4_XE_SWAP: present the front buffer.
    Swap {
        signature_ok: bool,
        frontbuffer: u32,
        width: u32,
        height: u32,
    },
    EventWrite(EventWrite),
    /// Memory writes the host must apply itself; the recorder does not
    /// write guest memory. Value `i` goes to `addr + 4 * i` as
    /// `endian.to_guest_bytes(value)` (little-endian after the swap, as Xenia).
    MemWrite {
        addr: u32,
        endian: Endian,
        values: Vec<u32>,
    },
    /// REG_TO_MEM: store register `register` at `addr` as
    /// `endian.to_guest_bytes(value)`.
    RegToMem {
        register: u32,
        addr: u32,
        endian: Endian,
    },
    /// WAIT_REG_MEM. The recorder cannot wait; the host decides.
    WaitRegMem(WaitRegMem),
    /// COND_WRITE: if the condition holds at execution time, write
    /// `write_data` to register `write_addr` or, with `write_memory`, to
    /// guest memory as `write_endian.to_guest_bytes(write_data)`. Not applied
    /// by the recorder.
    CondWrite {
        wait: WaitRegMem,
        write_memory: bool,
        write_addr: u32,
        write_endian: Endian,
        write_data: u32,
    },
    /// Raise the guest interrupt callback for each CPU bit in `cpu_mask`.
    Interrupt {
        cpu_mask: u32,
    },
    WaitForIdle {
        value: u32,
    },
    VizQuery {
        id: u32,
        end: bool,
    },
    Nop,
    ContextUpdate {
        value: u32,
    },
    InvalidateState {
        mask: u32,
    },
    MeInit {
        words: Vec<u32>,
    },
    /// An opcode the recorder does not interpret, with its payload.
    Unknown {
        opcode: u32,
        words: Vec<u32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub enum ShaderStage {
    Vertex,
    Pixel,
}

/// A DRAW_INDX or DRAW_INDX_2 packet.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub struct Draw {
    pub primitive: PrimitiveType,
    pub index_count: u32,
    pub source: IndexSource,
    /// Raw VGT_DRAW_INITIATOR.
    pub initiator: u32,
    /// DRAW_INDX's leading viz query token; None for DRAW_INDX_2.
    pub viz_query: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexSource {
    /// Indices 0..index_count are generated.
    Auto,
    /// Indices read from guest memory (VGT_DMA_BASE / VGT_DMA_SIZE).
    Dma {
        /// Physical address, aligned down to the index size as in Xenia.
        address: u32,
        /// VGT_DMA_SIZE::num_words: buffer length in indices.
        buffer_indices: u32,
        format: IndexFormat,
        endian: Endian,
    },
    /// Indices carried in the packet after the initiator. Xenia does not
    /// support this source; the words are kept raw because their packing is
    /// not verified.
    Immediate {
        format: IndexFormat,
        words: Vec<u32>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub enum EventKind {
    /// EVENT_WRITE.
    Plain,
    /// EVENT_WRITE_SHD: write `value` (or the swap counter) to `address` as
    /// `endian.to_guest_bytes(..)`.
    Shader,
    /// EVENT_WRITE_EXT: write screen extents to `address`.
    ScreenExtent,
    /// EVENT_WRITE_ZPD: occlusion query, samples at RB_SAMPLE_COUNT_ADDR.
    ZPassDone,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub struct EventWrite {
    pub kind: EventKind,
    pub initiator: u32,
    /// Physical address (low two bits removed) for Shader and ScreenExtent.
    pub address: Option<u32>,
    pub endian: Endian,
    /// Value for Shader events.
    pub value: Option<u32>,
    /// Shader events with initiator bit 31 write the frame counter instead.
    pub use_counter: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub struct WaitRegMem {
    /// Compare function (wait_info bits 0:2): 0 never, 1 <, 2 <=, 3 ==,
    /// 4 !=, 5 >=, 6 >, 7 always; applied to `(value & mask)` vs `reference`.
    pub function: u32,
    /// Poll memory (wait_info bit 4) instead of a register.
    pub memory: bool,
    /// Register index, or physical address with the low two bits removed.
    pub poll_addr: u32,
    pub endian: Endian,
    pub reference: u32,
    pub mask: u32,
    /// Wait interval hint (WAIT_REG_MEM only; 0 for COND_WRITE).
    pub wait: u32,
}

impl WaitRegMem {
    /// Evaluates the compare function against `value`: the register value,
    /// or for memory polls `endian.from_guest_bytes(bytes at poll_addr)`.
    pub fn matches(&self, value: u32) -> bool {
        let v = value & self.mask;
        match self.function {
            1 => v < self.reference,
            2 => v <= self.reference,
            3 => v == self.reference,
            4 => v != self.reference,
            5 => v >= self.reference,
            6 => v > self.reference,
            7 => true,
            _ => false,
        }
    }
}

/// Pixel rectangle, right/bottom exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// An EDRAM resolve: the copy registers at the time of the copy draw, raw and
/// decoded. The source rectangle itself comes from three vertices in
/// `vertex_fetch0` (Direct3D 9 convention, see Xenia GetResolveInfo), which
/// the drawer reads from guest memory; it is not decoded here.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bundle", derive(serde::Serialize, serde::Deserialize))]
pub struct Resolve {
    // Raw registers.
    pub copy_control: u32,
    pub dest_base: u32,
    pub dest_pitch_reg: u32,
    pub dest_info: u32,
    pub surface_slice: u32,
    pub surface_info: u32,
    pub color_info: [u32; 4],
    pub depth_info: u32,
    pub depth_clear: u32,
    pub color_clear: u32,
    pub color_clear_lo: u32,
    pub window_offset_reg: u32,
    pub window_scissor_tl: u32,
    pub window_scissor_br: u32,
    pub sc_mode_cntl: u32,
    pub vertex_fetch0: VertexFetch,

    // RB_COPY_CONTROL.
    /// 0..=3 color render target, 4 depth.
    pub src_select: u32,
    pub sample_select: u32,
    pub color_clear_enable: bool,
    pub depth_clear_enable: bool,
    pub copy_command: CopyCommand,
    // RB_COPY_DEST_PITCH.
    pub dest_pitch: u32,
    pub dest_height: u32,
    // RB_COPY_DEST_INFO.
    /// `xenos::Endian128` value (0..=5).
    pub dest_endian: u32,
    pub dest_array: bool,
    pub dest_slice: u32,
    /// `xenos::ColorFormat` value (6 bits).
    pub dest_format: u32,
    pub dest_number: u32,
    pub dest_exp_bias: i32,
    pub dest_swap: bool,
    // RB_SURFACE_INFO.
    pub surface_pitch: u32,
    /// 0 = 1x, 1 = 2x, 2 = 4x.
    pub msaa_samples: u32,
    // Source surface: RB_COLOR_INFO[src_select] or RB_DEPTH_INFO.
    pub is_depth: bool,
    /// EDRAM base in tiles (12 bits, including Xenia's separate bit 11).
    pub src_edram_base: u32,
    /// `ColorRenderTargetFormat` (4 bits) or `DepthRenderTargetFormat` (1 bit).
    pub src_format: u32,
    // PA_SC_WINDOW_*.
    pub window_offset_x: i32,
    pub window_offset_y: i32,
    /// PA_SU_SC_MODE_CNTL::vtx_window_offset_enable (bit 16).
    pub vtx_window_offset_enable: bool,
    pub window_scissor: Rect,
    pub window_offset_disable: bool,
}

impl Resolve {
    /// Captures and decodes the resolve state from `r`.
    pub fn from_registers(r: &RegisterFile) -> Self {
        let copy_control = r.get(regs::RB_COPY_CONTROL);
        let dest_pitch_reg = r.get(regs::RB_COPY_DEST_PITCH);
        let dest_info = r.get(regs::RB_COPY_DEST_INFO);
        let surface_info = r.get(regs::RB_SURFACE_INFO);
        let color_info = regs::RB_COLOR_INFO_RT.map(|i| r.get(i));
        let depth_info = r.get(regs::RB_DEPTH_INFO);
        let window_offset_reg = r.get(regs::PA_SC_WINDOW_OFFSET);
        let tl = r.get(regs::PA_SC_WINDOW_SCISSOR_TL);
        let br = r.get(regs::PA_SC_WINDOW_SCISSOR_BR);
        let sc_mode_cntl = r.get(regs::PA_SU_SC_MODE_CNTL);
        let src_select = copy_control & 7;
        let is_depth = src_select >= 4;
        let src_info = if is_depth {
            depth_info
        } else {
            color_info[src_select as usize]
        };
        Self {
            copy_control,
            dest_base: r.get(regs::RB_COPY_DEST_BASE),
            dest_pitch_reg,
            dest_info,
            surface_slice: r.get(regs::RB_COPY_SURFACE_SLICE),
            surface_info,
            color_info,
            depth_info,
            depth_clear: r.get(regs::RB_DEPTH_CLEAR),
            color_clear: r.get(regs::RB_COLOR_CLEAR),
            color_clear_lo: r.get(regs::RB_COLOR_CLEAR_LO),
            window_offset_reg,
            window_scissor_tl: tl,
            window_scissor_br: br,
            sc_mode_cntl,
            vertex_fetch0: r.vertex_fetch(0),
            src_select,
            sample_select: (copy_control >> 4) & 7,
            color_clear_enable: copy_control & (1 << 8) != 0,
            depth_clear_enable: copy_control & (1 << 9) != 0,
            copy_command: CopyCommand::from_bits(copy_control >> 20),
            dest_pitch: dest_pitch_reg & 0x3FFF,
            dest_height: (dest_pitch_reg >> 16) & 0x3FFF,
            dest_endian: dest_info & 7,
            dest_array: dest_info & (1 << 3) != 0,
            dest_slice: (dest_info >> 4) & 7,
            dest_format: (dest_info >> 7) & 0x3F,
            dest_number: (dest_info >> 13) & 7,
            dest_exp_bias: sign_extend(dest_info >> 16, 6),
            dest_swap: dest_info & (1 << 24) != 0,
            surface_pitch: surface_info & 0x3FFF,
            msaa_samples: (surface_info >> 16) & 3,
            is_depth,
            src_edram_base: src_info & 0xFFF,
            src_format: if is_depth {
                (src_info >> 16) & 1
            } else {
                (src_info >> 16) & 0xF
            },
            window_offset_x: sign_extend(window_offset_reg, 15),
            window_offset_y: sign_extend(window_offset_reg >> 16, 15),
            vtx_window_offset_enable: sc_mode_cntl & (1 << 16) != 0,
            window_scissor: Rect {
                left: (tl & 0x3FFF) as i32,
                top: ((tl >> 16) & 0x3FFF) as i32,
                right: (br & 0x3FFF) as i32,
                bottom: ((br >> 16) & 0x3FFF) as i32,
            },
            window_offset_disable: tl & (1 << 31) != 0,
        }
    }
}
