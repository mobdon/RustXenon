// Packet header layout ported from Xenia src/xenia/gpu/command_processor.cc
// and src/xenia/gpu/packet_disassembler.cc, opcode values from
// src/xenia/gpu/xenos.h (BSD 3-Clause, see CREDITS.md).

//! PM4 packet headers: decode, opcode names, and a builder for hand-made
//! command streams.

use std::fmt;

pub const PM4_ME_INIT: u32 = 0x48;
pub const PM4_NOP: u32 = 0x10;
pub const PM4_INDIRECT_BUFFER: u32 = 0x3F;
pub const PM4_INDIRECT_BUFFER_PFD: u32 = 0x37;
pub const PM4_WAIT_FOR_IDLE: u32 = 0x26;
pub const PM4_WAIT_REG_MEM: u32 = 0x3C;
pub const PM4_WAIT_REG_EQ: u32 = 0x52;
pub const PM4_WAIT_REG_GTE: u32 = 0x53;
pub const PM4_WAIT_UNTIL_READ: u32 = 0x5C;
pub const PM4_WAIT_IB_PFD_COMPLETE: u32 = 0x5D;
pub const PM4_REG_RMW: u32 = 0x21;
pub const PM4_REG_TO_MEM: u32 = 0x3E;
pub const PM4_MEM_WRITE: u32 = 0x3D;
pub const PM4_MEM_WRITE_CNTR: u32 = 0x4F;
pub const PM4_COND_EXEC: u32 = 0x44;
pub const PM4_COND_WRITE: u32 = 0x45;
pub const PM4_EVENT_WRITE: u32 = 0x46;
pub const PM4_EVENT_WRITE_SHD: u32 = 0x58;
pub const PM4_EVENT_WRITE_CFL: u32 = 0x59;
pub const PM4_EVENT_WRITE_EXT: u32 = 0x5A;
pub const PM4_EVENT_WRITE_ZPD: u32 = 0x5B;
pub const PM4_DRAW_INDX: u32 = 0x22;
pub const PM4_DRAW_INDX_2: u32 = 0x36;
pub const PM4_DRAW_INDX_BIN: u32 = 0x34;
pub const PM4_DRAW_INDX_2_BIN: u32 = 0x35;
pub const PM4_VIZ_QUERY: u32 = 0x23;
pub const PM4_SET_STATE: u32 = 0x25;
pub const PM4_SET_CONSTANT: u32 = 0x2D;
pub const PM4_SET_CONSTANT2: u32 = 0x55;
pub const PM4_SET_SHADER_CONSTANTS: u32 = 0x56;
pub const PM4_LOAD_ALU_CONSTANT: u32 = 0x2F;
pub const PM4_IM_LOAD: u32 = 0x27;
pub const PM4_IM_LOAD_IMMEDIATE: u32 = 0x2B;
pub const PM4_LOAD_CONSTANT_CONTEXT: u32 = 0x2E;
pub const PM4_INVALIDATE_STATE: u32 = 0x3B;
pub const PM4_SET_SHADER_BASES: u32 = 0x4A;
pub const PM4_SET_BIN_BASE_OFFSET: u32 = 0x4B;
pub const PM4_SET_BIN_MASK: u32 = 0x50;
pub const PM4_SET_BIN_SELECT: u32 = 0x51;
pub const PM4_CONTEXT_UPDATE: u32 = 0x5E;
pub const PM4_INTERRUPT: u32 = 0x54;
/// Not a hardware opcode: Xenia's convention for the swap the kernel's
/// VdSwap posts. The host here uses the same packet.
pub const PM4_XE_SWAP: u32 = 0x64;
pub const PM4_IM_STORE: u32 = 0x2C;
pub const PM4_SET_BIN_MASK_LO: u32 = 0x60;
pub const PM4_SET_BIN_MASK_HI: u32 = 0x61;
pub const PM4_SET_BIN_SELECT_LO: u32 = 0x62;
pub const PM4_SET_BIN_SELECT_HI: u32 = 0x63;

/// First payload dword of a PM4_XE_SWAP packet: the fourcc "SWAP".
pub const SWAP_SIGNATURE: u32 = u32::from_be_bytes(*b"SWAP");

/// Name of a type-3 opcode, without the `PM4_` prefix.
pub fn opcode_name(opcode: u32) -> Option<&'static str> {
    Some(match opcode {
        PM4_ME_INIT => "ME_INIT",
        PM4_NOP => "NOP",
        PM4_INDIRECT_BUFFER => "INDIRECT_BUFFER",
        PM4_INDIRECT_BUFFER_PFD => "INDIRECT_BUFFER_PFD",
        PM4_WAIT_FOR_IDLE => "WAIT_FOR_IDLE",
        PM4_WAIT_REG_MEM => "WAIT_REG_MEM",
        PM4_WAIT_REG_EQ => "WAIT_REG_EQ",
        PM4_WAIT_REG_GTE => "WAIT_REG_GTE",
        PM4_WAIT_UNTIL_READ => "WAIT_UNTIL_READ",
        PM4_WAIT_IB_PFD_COMPLETE => "WAIT_IB_PFD_COMPLETE",
        PM4_REG_RMW => "REG_RMW",
        PM4_REG_TO_MEM => "REG_TO_MEM",
        PM4_MEM_WRITE => "MEM_WRITE",
        PM4_MEM_WRITE_CNTR => "MEM_WRITE_CNTR",
        PM4_COND_EXEC => "COND_EXEC",
        PM4_COND_WRITE => "COND_WRITE",
        PM4_EVENT_WRITE => "EVENT_WRITE",
        PM4_EVENT_WRITE_SHD => "EVENT_WRITE_SHD",
        PM4_EVENT_WRITE_CFL => "EVENT_WRITE_CFL",
        PM4_EVENT_WRITE_EXT => "EVENT_WRITE_EXT",
        PM4_EVENT_WRITE_ZPD => "EVENT_WRITE_ZPD",
        PM4_DRAW_INDX => "DRAW_INDX",
        PM4_DRAW_INDX_2 => "DRAW_INDX_2",
        PM4_DRAW_INDX_BIN => "DRAW_INDX_BIN",
        PM4_DRAW_INDX_2_BIN => "DRAW_INDX_2_BIN",
        PM4_VIZ_QUERY => "VIZ_QUERY",
        PM4_SET_STATE => "SET_STATE",
        PM4_SET_CONSTANT => "SET_CONSTANT",
        PM4_SET_CONSTANT2 => "SET_CONSTANT2",
        PM4_SET_SHADER_CONSTANTS => "SET_SHADER_CONSTANTS",
        PM4_LOAD_ALU_CONSTANT => "LOAD_ALU_CONSTANT",
        PM4_IM_LOAD => "IM_LOAD",
        PM4_IM_LOAD_IMMEDIATE => "IM_LOAD_IMMEDIATE",
        PM4_LOAD_CONSTANT_CONTEXT => "LOAD_CONSTANT_CONTEXT",
        PM4_INVALIDATE_STATE => "INVALIDATE_STATE",
        PM4_SET_SHADER_BASES => "SET_SHADER_BASES",
        PM4_SET_BIN_BASE_OFFSET => "SET_BIN_BASE_OFFSET",
        PM4_SET_BIN_MASK => "SET_BIN_MASK",
        PM4_SET_BIN_SELECT => "SET_BIN_SELECT",
        PM4_CONTEXT_UPDATE => "CONTEXT_UPDATE",
        PM4_INTERRUPT => "INTERRUPT",
        PM4_XE_SWAP => "XE_SWAP",
        PM4_IM_STORE => "IM_STORE",
        PM4_SET_BIN_MASK_LO => "SET_BIN_MASK_LO",
        PM4_SET_BIN_MASK_HI => "SET_BIN_MASK_HI",
        PM4_SET_BIN_SELECT_LO => "SET_BIN_SELECT_LO",
        PM4_SET_BIN_SELECT_HI => "SET_BIN_SELECT_HI",
        _ => return None,
    })
}

/// A decoded packet header. `payload_len` is the number of dwords that
/// follow the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// Write `count` registers from `base`; with `one_reg` every value goes
    /// to `base`.
    Type0 {
        base: u32,
        count: u32,
        one_reg: bool,
    },
    /// Write two registers, each from an 11-bit index.
    Type1 { reg1: u32, reg2: u32 },
    /// No-op, header only.
    Type2,
    /// Opcode packet; `predicate` asks for the bin check before executing.
    Type3 {
        opcode: u32,
        count: u32,
        predicate: bool,
    },
}

impl Header {
    pub fn decode(word: u32) -> Self {
        match word >> 30 {
            0 => Header::Type0 {
                base: word & 0x7FFF,
                count: ((word >> 16) & 0x3FFF) + 1,
                one_reg: (word >> 15) & 1 != 0,
            },
            1 => Header::Type1 {
                reg1: word & 0x7FF,
                reg2: (word >> 11) & 0x7FF,
            },
            2 => Header::Type2,
            _ => Header::Type3 {
                opcode: (word >> 8) & 0x7F,
                count: ((word >> 16) & 0x3FFF) + 1,
                predicate: word & 1 != 0,
            },
        }
    }

    /// Dwords that follow the header.
    pub fn payload_len(self) -> u32 {
        match self {
            Header::Type0 { count, .. } | Header::Type3 { count, .. } => count,
            Header::Type1 { .. } => 2,
            Header::Type2 => 0,
        }
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Header::Type0 {
                base,
                count,
                one_reg,
            } => {
                write!(f, "TYPE0 base=0x{base:04X} count={count}")?;
                if one_reg {
                    write!(f, " one_reg")?;
                }
                Ok(())
            }
            Header::Type1 { reg1, reg2 } => write!(f, "TYPE1 reg1=0x{reg1:04X} reg2=0x{reg2:04X}"),
            Header::Type2 => write!(f, "TYPE2"),
            Header::Type3 {
                opcode,
                count,
                predicate,
            } => {
                match opcode_name(opcode) {
                    Some(name) => write!(f, "TYPE3 {name} count={count}")?,
                    None => write!(f, "TYPE3 0x{opcode:02X} count={count}")?,
                }
                if predicate {
                    write!(f, " predicated")?;
                }
                Ok(())
            }
        }
    }
}

/// Builders for hand-made packet streams (tests, tools, drawer tests).
/// Every function returns host-order dwords; `to_be_bytes` turns a stream
/// into guest memory bytes.
pub mod build {
    /// Type-0 packet writing `values` to consecutive registers from `base`.
    pub fn type0(base: u32, values: &[u32]) -> Vec<u32> {
        header0(base, values, false)
    }

    /// Type-0 packet writing every value to the single register `reg`.
    pub fn type0_one_reg(reg: u32, values: &[u32]) -> Vec<u32> {
        header0(reg, values, true)
    }

    fn header0(base: u32, values: &[u32], one_reg: bool) -> Vec<u32> {
        assert!(!values.is_empty() && values.len() <= 0x4000 && base <= 0x7FFF);
        let mut out = vec![((values.len() as u32 - 1) << 16) | (u32::from(one_reg) << 15) | base];
        out.extend_from_slice(values);
        out
    }

    /// Type-1 packet writing two registers (indices below 0x800).
    pub fn type1(reg1: u32, value1: u32, reg2: u32, value2: u32) -> Vec<u32> {
        assert!(reg1 < 0x800 && reg2 < 0x800);
        vec![(1 << 30) | (reg2 << 11) | reg1, value1, value2]
    }

    /// Type-2 no-op packet.
    pub fn type2() -> Vec<u32> {
        vec![2 << 30]
    }

    /// Type-3 packet header for `opcode` with `count` payload dwords; lets
    /// tests write a header whose count disagrees with the data.
    pub fn type3_header(opcode: u32, count: u32, predicate: bool) -> u32 {
        assert!((1..=0x4000).contains(&count) && opcode < 0x80);
        (3 << 30) | ((count - 1) << 16) | (opcode << 8) | u32::from(predicate)
    }

    /// Type-3 packet with `payload` (at least one dword, as the count field
    /// stores count minus one).
    pub fn type3(opcode: u32, payload: &[u32]) -> Vec<u32> {
        let mut out = vec![type3_header(opcode, payload.len() as u32, false)];
        out.extend_from_slice(payload);
        out
    }

    /// Type-3 packet with the predicate bit set.
    pub fn type3_predicated(opcode: u32, payload: &[u32]) -> Vec<u32> {
        let mut out = vec![type3_header(opcode, payload.len() as u32, true)];
        out.extend_from_slice(payload);
        out
    }

    /// Dwords as big-endian guest memory bytes.
    pub fn to_be_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_be_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_each_type() {
        assert_eq!(
            Header::decode(0x0002_8123),
            Header::Type0 {
                base: 0x0123,
                count: 3,
                one_reg: true
            }
        );
        assert_eq!(
            Header::decode((1 << 30) | (0x22 << 11) | 0x11),
            Header::Type1 {
                reg1: 0x11,
                reg2: 0x22
            }
        );
        assert_eq!(Header::decode(0x8000_0000), Header::Type2);
        assert_eq!(
            Header::decode(0xC003_2201),
            Header::Type3 {
                opcode: PM4_DRAW_INDX,
                count: 4,
                predicate: true
            }
        );
    }

    #[test]
    fn display_is_readable() {
        assert_eq!(
            Header::decode(0xC003_2201).to_string(),
            "TYPE3 DRAW_INDX count=4 predicated"
        );
        assert_eq!(
            Header::decode(0xC000_7F00).to_string(),
            "TYPE3 0x7F count=1"
        );
        assert_eq!(
            Header::decode(0x0001_2000).to_string(),
            "TYPE0 base=0x2000 count=2"
        );
    }

    #[test]
    fn builder_round_trips() {
        let p = build::type3(PM4_NOP, &[1, 2]);
        assert_eq!(
            Header::decode(p[0]),
            Header::Type3 {
                opcode: PM4_NOP,
                count: 2,
                predicate: false
            }
        );
        let p = build::type0_one_reg(0x2000, &[5, 6]);
        assert_eq!(
            Header::decode(p[0]),
            Header::Type0 {
                base: 0x2000,
                count: 2,
                one_reg: true
            }
        );
        assert_eq!(SWAP_SIGNATURE, 0x5357_4150);
    }
}
