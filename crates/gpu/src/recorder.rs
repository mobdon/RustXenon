// Packet handling ported from Xenia src/xenia/gpu/command_processor.cc
// (ExecutePrimaryBuffer, ExecutePacket*, ExecutePacketType3_*) (BSD 3-Clause,
// see CREDITS.md).

//! Walks PM4 packet streams in guest memory and records typed commands.

use std::fmt;

use crate::command::{
    Command, Draw, EventKind, EventWrite, IndexSource, Resolve, ShaderStage, WaitRegMem,
};
use crate::memory::{read_words, GuestRead, PHYSICAL_MASK};
use crate::pm4::{self, Header};
use crate::regs::{self, RegisterFile, REGISTER_COUNT};
use crate::xenos::{Endian, IndexFormat, PrimitiveType};

/// Default nesting limit for indirect buffers. Direct3D 9 on the console
/// uses one level below the ring; this leaves room without letting a
/// self-referencing buffer recurse forever.
pub const DEFAULT_MAX_INDIRECT_DEPTH: u32 = 4;
/// Default total of indirect-buffer dwords followed per record call (64 MiB).
pub const DEFAULT_MAX_INDIRECT_DWORDS: u64 = 16 << 20;

/// Fatal problems with the buffer the caller handed in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A ring pointer is not below the ring size, or the size is 0.
    RingPointerOutOfRange {
        read_ptr: u32,
        write_ptr: u32,
        size_dwords: u32,
    },
    /// The top-level buffer is not in guest memory.
    Unreadable { addr: u32, dwords: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RingPointerOutOfRange { read_ptr, write_ptr, size_dwords } => write!(
                f,
                "ring pointers read={read_ptr} write={write_ptr} out of range for {size_dwords} dwords"
            ),
            Error::Unreadable { addr, dwords } => write!(f, "cannot read {dwords} dwords at 0x{addr:08X}"),
        }
    }
}

impl std::error::Error for Error {}

/// Problems inside the stream. The recorder notes them and goes on, as
/// Xenia logs and continues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Warning {
    /// A packet claims more dwords than the buffer has left; the rest of
    /// that buffer is dropped (Xenia stops the buffer here too).
    Truncated { header: u32, available: u32 },
    /// A type-3 packet is too short for its opcode; the packet is skipped.
    /// Xenia instead reads past the packet (most opcodes) or stops the
    /// buffer (DRAW_INDX, a DMA draw without base and size).
    PacketTooShort { opcode: u32, count: u32 },
    /// An indirect buffer nested deeper than the limit; not followed.
    IndirectDepthExceeded { addr: u32, dwords: u32 },
    /// Indirect buffers in one record call added up to more than
    /// `max_indirect_dwords`; this one is not followed. Bounds the work a
    /// buffer that calls itself many times can cause.
    IndirectBudgetExceeded { addr: u32, dwords: u32 },
    /// Guest memory a packet points at is not readable; packet skipped.
    Unreadable { addr: u32, dwords: u32 },
    /// Register writes from this index on fell past the register file.
    RegisterOutOfRange { index: u32 },
    /// SET_CONSTANT / LOAD_ALU_CONSTANT with a constant type above 4.
    BadConstantType { offset_type: u32 },
    /// IM_LOAD(_IMMEDIATE) with a shader type other than vertex or pixel.
    BadShaderType { value: u32 },
    /// A draw with source select 3 (invalid); no draw recorded.
    BadIndexSource { initiator: u32 },
}

/// Parses PM4 streams into [`Command`]s and keeps the register state they
/// produce.
#[derive(Debug)]
pub struct Recorder {
    /// Register state after the commands recorded so far.
    pub regs: RegisterFile,
    /// Commands in stream order; the caller drains it.
    pub commands: Vec<Command>,
    /// Non-fatal problems in stream order.
    pub warnings: Vec<Warning>,
    pub max_indirect_depth: u32,
    /// Indirect-buffer dwords one record call may follow in total.
    pub max_indirect_dwords: u64,
    indirect_left: u64,
    // Predication state for type-3 packets with the predicate bit, as Xenia.
    bin_mask: u64,
    bin_select: u64,
}

impl Default for Recorder {
    fn default() -> Self {
        Self {
            regs: RegisterFile::default(),
            commands: Vec::new(),
            warnings: Vec::new(),
            max_indirect_depth: DEFAULT_MAX_INDIRECT_DEPTH,
            max_indirect_dwords: DEFAULT_MAX_INDIRECT_DWORDS,
            indirect_left: 0,
            bin_mask: 0xFFFF_FFFF,
            bin_select: 0xFFFF_FFFF,
        }
    }
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the packets between `read_ptr` and `write_ptr` (dword indices)
    /// of the ring at `ring_base`, wrapping at `ring_size_dwords`. Returns the
    /// new read pointer, which is `write_ptr`: like Xenia, the whole span is
    /// consumed even when the last packet is cut short (that gives a warning).
    pub fn record_ring(
        &mut self,
        mem: &dyn GuestRead,
        ring_base: u32,
        ring_size_dwords: u32,
        read_ptr: u32,
        write_ptr: u32,
    ) -> Result<u32, Error> {
        if ring_size_dwords == 0 || read_ptr >= ring_size_dwords || write_ptr >= ring_size_dwords {
            return Err(Error::RingPointerOutOfRange {
                read_ptr,
                write_ptr,
                size_dwords: ring_size_dwords,
            });
        }
        let base = ring_base & PHYSICAL_MASK;
        let read_span = |from: u32, to: u32| {
            // wrapping_mul: a hostile ring size above 2^30 dwords must not panic.
            let addr = base.wrapping_add(from.wrapping_mul(4)) & PHYSICAL_MASK;
            read_words(mem, addr, to - from).ok_or(Error::Unreadable {
                addr,
                dwords: to - from,
            })
        };
        let words = if read_ptr <= write_ptr {
            read_span(read_ptr, write_ptr)?
        } else {
            let mut words = read_span(read_ptr, ring_size_dwords)?;
            words.extend(read_span(0, write_ptr)?);
            words
        };
        self.indirect_left = self.max_indirect_dwords;
        self.run(mem, &words, 0);
        Ok(write_ptr)
    }

    /// Records a plain buffer of `dword_count` dwords at `addr`.
    pub fn record_buffer(
        &mut self,
        mem: &dyn GuestRead,
        addr: u32,
        dword_count: u32,
    ) -> Result<(), Error> {
        let addr = addr & PHYSICAL_MASK;
        let words = read_words(mem, addr, dword_count).ok_or(Error::Unreadable {
            addr,
            dwords: dword_count,
        })?;
        self.indirect_left = self.max_indirect_dwords;
        self.run(mem, &words, 0);
        Ok(())
    }

    /// Records packets already copied out of guest memory. `mem` is still
    /// needed for packets that point into memory.
    pub fn record_words(&mut self, mem: &dyn GuestRead, words: &[u32]) {
        self.indirect_left = self.max_indirect_dwords;
        self.run(mem, words, 0);
    }

    fn run(&mut self, mem: &dyn GuestRead, words: &[u32], depth: u32) {
        let mut pos = 0;
        while pos < words.len() {
            let header = words[pos];
            pos += 1;
            // Xenia skips a zero dword as a one-dword no-op.
            if header == 0 {
                continue;
            }
            let h = Header::decode(header);
            let len = h.payload_len() as usize;
            if words.len() - pos < len {
                self.warnings.push(Warning::Truncated {
                    header,
                    available: (words.len() - pos) as u32,
                });
                return;
            }
            let p = &words[pos..pos + len];
            pos += len;
            match h {
                Header::Type0 {
                    base,
                    one_reg: false,
                    ..
                } => self.write_regs(base, p.to_vec()),
                Header::Type0 {
                    base,
                    one_reg: true,
                    ..
                } => {
                    for &v in p {
                        self.write_regs(base, vec![v]);
                    }
                }
                Header::Type1 { reg1, reg2 } => {
                    self.write_regs(reg1, vec![p[0]]);
                    self.write_regs(reg2, vec![p[1]]);
                }
                Header::Type2 => self.commands.push(Command::Nop),
                Header::Type3 {
                    opcode, predicate, ..
                } => {
                    // Predicated packets run only if a selected bin is enabled;
                    // predicated swaps never run.
                    if predicate
                        && ((self.bin_select & self.bin_mask) == 0 || opcode == pm4::PM4_XE_SWAP)
                    {
                        continue;
                    }
                    self.type3(mem, opcode, p, depth);
                }
            }
        }
    }

    /// Applies consecutive register writes and records them.
    fn write_regs(&mut self, base: u32, mut values: Vec<u32>) {
        let room = (REGISTER_COUNT as u32).saturating_sub(base) as usize;
        if values.len() > room {
            self.warnings.push(Warning::RegisterOutOfRange {
                index: base.saturating_add(room as u32),
            });
            values.truncate(room);
        }
        if values.is_empty() {
            return;
        }
        for (i, &v) in values.iter().enumerate() {
            self.regs.set(base + i as u32, v);
        }
        self.commands.push(Command::WriteRegisters { base, values });
    }

    fn type3(&mut self, mem: &dyn GuestRead, opcode: u32, p: &[u32], depth: u32) {
        let need = |n: usize, warnings: &mut Vec<Warning>| {
            let ok = p.len() >= n;
            if !ok {
                warnings.push(Warning::PacketTooShort {
                    opcode,
                    count: p.len() as u32,
                });
            }
            ok
        };
        use pm4::*;
        match opcode {
            PM4_ME_INIT => self.commands.push(Command::MeInit { words: p.to_vec() }),
            PM4_NOP => self.commands.push(Command::Nop),
            PM4_INTERRUPT => self.commands.push(Command::Interrupt { cpu_mask: p[0] }),
            PM4_XE_SWAP => {
                if need(4, &mut self.warnings) {
                    self.commands.push(Command::Swap {
                        signature_ok: p[0] == SWAP_SIGNATURE,
                        frontbuffer: p[1],
                        width: p[2],
                        height: p[3],
                    });
                }
            }
            PM4_INDIRECT_BUFFER | PM4_INDIRECT_BUFFER_PFD => {
                if need(2, &mut self.warnings) {
                    let addr = p[0] & PHYSICAL_MASK;
                    let dwords = p[1] & 0xF_FFFF;
                    if depth >= self.max_indirect_depth {
                        self.warnings
                            .push(Warning::IndirectDepthExceeded { addr, dwords });
                    } else if u64::from(dwords) > self.indirect_left {
                        self.warnings
                            .push(Warning::IndirectBudgetExceeded { addr, dwords });
                    } else if let Some(words) = read_words(mem, addr, dwords) {
                        self.indirect_left -= u64::from(dwords);
                        self.run(mem, &words, depth + 1);
                    } else {
                        self.warnings.push(Warning::Unreadable { addr, dwords });
                    }
                }
            }
            PM4_WAIT_REG_MEM => {
                if need(5, &mut self.warnings) {
                    self.commands
                        .push(Command::WaitRegMem(wait_info(p[0], p[1], p[2], p[3], p[4])));
                }
            }
            PM4_COND_WRITE => {
                if need(6, &mut self.warnings) {
                    let write_memory = p[0] & 0x100 != 0;
                    let (write_addr, write_endian) = if write_memory {
                        ((p[4] & !3) & PHYSICAL_MASK, Endian::from_bits(p[4]))
                    } else {
                        (p[4], Endian::None)
                    };
                    self.commands.push(Command::CondWrite {
                        wait: wait_info(p[0], p[1], p[2], p[3], 0),
                        write_memory,
                        write_addr,
                        write_endian,
                        write_data: p[5],
                    });
                }
            }
            PM4_REG_RMW => {
                if need(3, &mut self.warnings) {
                    let (info, and_mask, or_mask) = (p[0], p[1], p[2]);
                    // Indices are 13 bits as in Xenia: only registers below 0x2000.
                    let reg = info & 0x1FFF;
                    let mut value = self.regs.get(reg);
                    value &= if info >> 31 != 0 {
                        self.regs.get(and_mask & 0x1FFF)
                    } else {
                        and_mask
                    };
                    value |= if (info >> 30) & 1 != 0 {
                        self.regs.get(or_mask & 0x1FFF)
                    } else {
                        or_mask
                    };
                    self.write_regs(reg, vec![value]);
                }
            }
            PM4_REG_TO_MEM => {
                if need(2, &mut self.warnings) {
                    self.commands.push(Command::RegToMem {
                        register: p[0],
                        addr: (p[1] & !3) & PHYSICAL_MASK,
                        endian: Endian::from_bits(p[1]),
                    });
                }
            }
            PM4_MEM_WRITE => self.commands.push(Command::MemWrite {
                addr: (p[0] & !3) & PHYSICAL_MASK,
                endian: Endian::from_bits(p[0]),
                values: p[1..].to_vec(),
            }),
            PM4_EVENT_WRITE | PM4_EVENT_WRITE_SHD | PM4_EVENT_WRITE_EXT | PM4_EVENT_WRITE_ZPD => {
                let (kind, n) = match opcode {
                    PM4_EVENT_WRITE => (EventKind::Plain, 1),
                    PM4_EVENT_WRITE_SHD => (EventKind::Shader, 3),
                    PM4_EVENT_WRITE_EXT => (EventKind::ScreenExtent, 2),
                    _ => (EventKind::ZPassDone, 1),
                };
                if need(n, &mut self.warnings) {
                    let initiator = p[0];
                    self.write_regs(regs::VGT_EVENT_INITIATOR, vec![initiator & 0x3F]);
                    let address = (n >= 2).then(|| (p[1] & !3) & PHYSICAL_MASK);
                    let endian = if n >= 2 {
                        Endian::from_bits(p[1])
                    } else {
                        Endian::None
                    };
                    self.commands.push(Command::EventWrite(EventWrite {
                        kind,
                        initiator,
                        address,
                        endian,
                        value: (n >= 3).then(|| p[2]),
                        use_counter: kind == EventKind::Shader && initiator >> 31 != 0,
                    }));
                }
            }
            PM4_DRAW_INDX => {
                if need(2, &mut self.warnings) {
                    self.draw(Some(p[0]), &p[1..]);
                }
            }
            PM4_DRAW_INDX_2 => self.draw(None, p),
            PM4_SET_CONSTANT => {
                let offset_type = p[0];
                match constant_base(offset_type) {
                    Some(base) => {
                        if p.len() > 1 {
                            self.write_regs(base + (offset_type & 0x7FF), p[1..].to_vec());
                        }
                    }
                    None => self.warnings.push(Warning::BadConstantType { offset_type }),
                }
            }
            PM4_SET_CONSTANT2 | PM4_SET_SHADER_CONSTANTS => {
                if p.len() > 1 {
                    self.write_regs(p[0] & 0xFFFF, p[1..].to_vec());
                }
            }
            PM4_LOAD_ALU_CONSTANT => {
                if need(3, &mut self.warnings) {
                    let addr = p[0] & 0x3FFF_FFFF & PHYSICAL_MASK;
                    let offset_type = p[1];
                    let dwords = p[2] & 0xFFF;
                    match constant_base(offset_type) {
                        None => self.warnings.push(Warning::BadConstantType { offset_type }),
                        Some(_) if dwords == 0 => {}
                        Some(base) => match read_words(mem, addr, dwords) {
                            Some(values) => self.write_regs(base + (offset_type & 0x7FF), values),
                            None => self.warnings.push(Warning::Unreadable { addr, dwords }),
                        },
                    }
                }
            }
            PM4_IM_LOAD => {
                if need(2, &mut self.warnings) {
                    let addr = (p[0] & !3) & PHYSICAL_MASK;
                    let (start, dwords) = (p[1] >> 16, p[1] & 0xFFFF);
                    match (shader_stage(p[0] & 3), read_words(mem, addr, dwords)) {
                        (None, _) => self
                            .warnings
                            .push(Warning::BadShaderType { value: p[0] & 3 }),
                        (_, None) => self.warnings.push(Warning::Unreadable { addr, dwords }),
                        (Some(stage), Some(words)) => self.commands.push(Command::LoadShader {
                            stage,
                            source_addr: Some(addr),
                            start,
                            words,
                        }),
                    }
                }
            }
            PM4_IM_LOAD_IMMEDIATE => {
                if need(2, &mut self.warnings) {
                    let (start, dwords) = (p[1] >> 16, (p[1] & 0xFFFF) as usize);
                    match shader_stage(p[0]) {
                        None => self.warnings.push(Warning::BadShaderType { value: p[0] }),
                        Some(_) if p.len() - 2 < dwords => {
                            self.warnings.push(Warning::PacketTooShort {
                                opcode,
                                count: p.len() as u32,
                            })
                        }
                        Some(stage) => self.commands.push(Command::LoadShader {
                            stage,
                            source_addr: None,
                            start,
                            words: p[2..2 + dwords].to_vec(),
                        }),
                    }
                }
            }
            PM4_INVALIDATE_STATE => self.commands.push(Command::InvalidateState { mask: p[0] }),
            PM4_VIZ_QUERY => {
                let id = p[0] & 0x3F;
                let end = p[0] & 0x100 != 0;
                // VIZQUERY_START = 7, VIZQUERY_END = 8 (xenos.h Event).
                self.write_regs(regs::VGT_EVENT_INITIATOR, vec![if end { 8 } else { 7 }]);
                if end {
                    // Xenia reports every query as visible.
                    let (reg, bit) = if id < 32 {
                        (regs::PA_SC_VIZ_QUERY_STATUS_0, id)
                    } else {
                        (regs::PA_SC_VIZ_QUERY_STATUS_1, id - 32)
                    };
                    let value = self.regs.get(reg) | (1 << bit);
                    self.write_regs(reg, vec![value]);
                }
                self.commands.push(Command::VizQuery { id, end });
            }
            PM4_SET_BIN_MASK_LO => self.bin_mask = (self.bin_mask & !0xFFFF_FFFF) | u64::from(p[0]),
            PM4_SET_BIN_MASK_HI => {
                self.bin_mask = (self.bin_mask & 0xFFFF_FFFF) | (u64::from(p[0]) << 32)
            }
            PM4_SET_BIN_SELECT_LO => {
                self.bin_select = (self.bin_select & !0xFFFF_FFFF) | u64::from(p[0])
            }
            PM4_SET_BIN_SELECT_HI => {
                self.bin_select = (self.bin_select & 0xFFFF_FFFF) | (u64::from(p[0]) << 32)
            }
            PM4_SET_BIN_MASK | PM4_SET_BIN_SELECT => {
                if need(2, &mut self.warnings) {
                    let v = (u64::from(p[0]) << 32) | u64::from(p[1]);
                    if opcode == PM4_SET_BIN_MASK {
                        self.bin_mask = v;
                    } else {
                        self.bin_select = v;
                    }
                }
            }
            PM4_CONTEXT_UPDATE => self.commands.push(Command::ContextUpdate { value: p[0] }),
            PM4_WAIT_FOR_IDLE => self.commands.push(Command::WaitForIdle { value: p[0] }),
            _ => self.commands.push(Command::Unknown {
                opcode,
                words: p.to_vec(),
            }),
        }
    }

    /// DRAW_INDX (with `viz_query`) and DRAW_INDX_2; `p` starts at
    /// VGT_DRAW_INITIATOR.
    fn draw(&mut self, viz_query: Option<u32>, p: &[u32]) {
        let opcode = if viz_query.is_some() {
            pm4::PM4_DRAW_INDX
        } else {
            pm4::PM4_DRAW_INDX_2
        };
        let initiator = p[0];
        self.write_regs(regs::VGT_DRAW_INITIATOR, vec![initiator]);
        let format = if initiator & (1 << 11) != 0 {
            IndexFormat::Int32
        } else {
            IndexFormat::Int16
        };
        let source = match (initiator >> 6) & 3 {
            0 => {
                if p.len() < 3 {
                    self.warnings.push(Warning::PacketTooShort {
                        opcode,
                        count: p.len() as u32,
                    });
                    return;
                }
                self.write_regs(regs::VGT_DMA_BASE, vec![p[1]]);
                self.write_regs(regs::VGT_DMA_SIZE, vec![p[2]]);
                IndexSource::Dma {
                    address: (p[1] & PHYSICAL_MASK) & !(format.bytes() - 1),
                    buffer_indices: p[2] & 0x00FF_FFFF,
                    format,
                    endian: Endian::from_bits(p[2] >> 30),
                }
            }
            1 => IndexSource::Immediate {
                format,
                words: p[1..].to_vec(),
            },
            2 => IndexSource::Auto,
            _ => {
                self.warnings.push(Warning::BadIndexSource { initiator });
                return;
            }
        };
        // Direct3D 9 resolves by drawing a rectangle with RB_MODECONTROL in
        // copy mode; Xenia's IssueDraw turns that into IssueCopy.
        if self.regs.get(regs::RB_MODECONTROL) & 7 == regs::EDRAM_MODE_COPY {
            self.commands
                .push(Command::Resolve(Box::new(Resolve::from_registers(
                    &self.regs,
                ))));
            return;
        }
        self.commands.push(Command::Draw(Draw {
            primitive: PrimitiveType::from_raw(initiator & 0x3F),
            index_count: initiator >> 16,
            source,
            initiator,
            viz_query,
        }));
    }
}

/// Register base for a SET_CONSTANT / LOAD_ALU_CONSTANT type field.
fn constant_base(offset_type: u32) -> Option<u32> {
    match (offset_type >> 16) & 0xFF {
        0 => Some(regs::SHADER_CONSTANT_FLOAT),
        1 => Some(regs::SHADER_CONSTANT_FETCH),
        2 => Some(regs::SHADER_CONSTANT_BOOL),
        3 => Some(regs::SHADER_CONSTANT_LOOP),
        4 => Some(0x2000),
        _ => None,
    }
}

fn shader_stage(value: u32) -> Option<ShaderStage> {
    match value {
        0 => Some(ShaderStage::Vertex),
        1 => Some(ShaderStage::Pixel),
        _ => None,
    }
}

fn wait_info(info: u32, poll: u32, reference: u32, mask: u32, wait: u32) -> WaitRegMem {
    let memory = info & 0x10 != 0;
    WaitRegMem {
        function: info & 7,
        memory,
        poll_addr: if memory {
            (poll & !3) & PHYSICAL_MASK
        } else {
            poll
        },
        endian: if memory {
            Endian::from_bits(poll)
        } else {
            Endian::None
        },
        reference,
        mask,
        wait,
    }
}
