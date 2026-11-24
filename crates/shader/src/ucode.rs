// Instruction formats ported from Xenia src/xenia/gpu/ucode.h and ucode.cc
// (BSD 3-Clause, see CREDITS.md); XenosRecomp shader_code.h (MIT) was used as
// a cross-check of the same bit layouts.

//! Decoder for the console GPU's shader microcode.
//!
//! A program is a run of 32-bit words. It starts with control-flow
//! instructions, packed two per three words. ALU and fetch instructions
//! follow, three words each, and are addressed in units of three words from
//! the start of the program, so the control-flow area also counts as slots.

use std::fmt;

/// A word sequence could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DecodeError {}

fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1u32 << n) - 1)
}

fn sbits(v: u32, lo: u32, n: u32) -> i32 {
    ((v << (32 - lo - n)) as i32) >> (32 - n)
}

fn bit(v: u32, lo: u32) -> bool {
    (v >> lo) & 1 != 0
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

/// What gates an exec block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecCond {
    /// `exec` / `exece`.
    Always { predicate_clean: bool },
    /// `cexec b<index>`; equal to `value` runs the block. `predicate_clean`
    /// selects the CondExecPredClean opcodes, which also test a bool constant.
    Bool {
        index: u32,
        value: bool,
        predicate_clean: bool,
    },
    /// `exec` on the predicate register (CondExecPred opcodes).
    Predicate { value: bool, predicate_clean: bool },
}

/// One exec control-flow instruction: runs `count` ALU/fetch instructions
/// starting at slot `address`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exec {
    pub cond: ExecCond,
    /// Ends the program after running (only when the condition passed).
    pub end: bool,
    /// Slot of the first instruction, in units of three words.
    pub address: u32,
    /// Number of instructions, 0..=6.
    pub count: u32,
    /// Two bits per instruction: bit 2i set means instruction i is a fetch,
    /// bit 2i+1 is the serialize hint.
    pub sequence: u32,
    pub is_yield: bool,
    /// Vertex cache hint bits, kept only for round trips.
    pub vc: u32,
}

/// Condition of a jump or call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchCond {
    Always,
    Bool { index: u32, value: bool },
    Predicate(bool),
}

/// A decoded control-flow instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CfInstr {
    Nop,
    Exec(Exec),
    /// Pushes a loop. `address` is where to go when the loop count is zero
    /// (normally right after the matching `LoopEnd`).
    LoopStart {
        address: u32,
        repeat: bool,
        loop_id: u32,
    },
    /// Counts down and jumps back to `address` (the loop body).
    /// `predicated_break`: break out when the predicate equals the value.
    LoopEnd {
        address: u32,
        loop_id: u32,
        predicated_break: Option<bool>,
    },
    CondCall {
        address: u32,
        cond: BranchCond,
    },
    Return,
    CondJmp {
        address: u32,
        cond: BranchCond,
        direction: bool,
    },
    Alloc {
        size: u32,
        alloc_type: u32,
        unserialized: bool,
    },
    MarkVsFetchDone,
}

/// Splits three words into the two 48-bit control-flow halves, as
/// `(low 32 bits, high 16 bits)` each.
pub fn split_cf_pair(w: [u32; 3]) -> [(u32, u32); 2] {
    [
        (w[0], w[1] & 0xFFFF),
        ((w[1] >> 16) | (w[2] << 16), w[2] >> 16),
    ]
}

/// Decodes one control-flow instruction from its two halves.
pub fn decode_cf(lo: u32, hi: u32) -> CfInstr {
    let opcode = bits(hi, 12, 4);
    let exec = |cond| {
        CfInstr::Exec(Exec {
            cond,
            end: matches!(opcode, 2 | 4 | 6 | 14),
            address: bits(lo, 0, 12),
            count: bits(lo, 12, 3),
            is_yield: bit(lo, 15),
            sequence: bits(lo, 16, 12),
            vc: bits(lo, 28, 4) | (bits(hi, 0, 2) << 4),
        })
    };
    let branch = || {
        if bit(lo, 13) {
            BranchCond::Always
        } else if bit(lo, 14) {
            BranchCond::Predicate(bit(hi, 10))
        } else {
            BranchCond::Bool {
                index: bits(hi, 2, 8),
                value: bit(hi, 10),
            }
        }
    };
    match opcode {
        0 => CfInstr::Nop,
        1 | 2 => exec(ExecCond::Always {
            predicate_clean: bit(hi, 9),
        }),
        3 | 4 | 13 | 14 => exec(ExecCond::Bool {
            index: bits(hi, 2, 8),
            value: bit(hi, 10),
            predicate_clean: opcode >= 13,
        }),
        5 | 6 => exec(ExecCond::Predicate {
            value: bit(hi, 10),
            predicate_clean: bit(hi, 9),
        }),
        7 => CfInstr::LoopStart {
            address: bits(lo, 0, 13),
            repeat: bit(lo, 13),
            loop_id: bits(lo, 16, 5),
        },
        8 => CfInstr::LoopEnd {
            address: bits(lo, 0, 13),
            loop_id: bits(lo, 16, 5),
            predicated_break: bit(lo, 21).then(|| bit(hi, 10)),
        },
        9 => CfInstr::CondCall {
            address: bits(lo, 0, 13),
            cond: branch(),
        },
        10 => CfInstr::Return,
        11 => CfInstr::CondJmp {
            address: bits(lo, 0, 13),
            cond: branch(),
            direction: bit(hi, 1),
        },
        12 => CfInstr::Alloc {
            size: bits(lo, 0, 3),
            alloc_type: bits(hi, 9, 2),
            unserialized: bit(hi, 8),
        },
        _ => CfInstr::MarkVsFetchDone,
    }
}

/// Decodes a pair of control-flow instructions from three words.
pub fn decode_cf_pair(w: [u32; 3]) -> [CfInstr; 2] {
    let [a, b] = split_cf_pair(w);
    [decode_cf(a.0, a.1), decode_cf(b.0, b.1)]
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------

/// Vertex fetch data format (6-bit field). Constants hold the known values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFormat(pub u32);

impl VertexFormat {
    pub const F8_8_8_8: Self = Self(6);
    pub const F2_10_10_10: Self = Self(7);
    pub const F10_11_11: Self = Self(16);
    pub const F11_11_10: Self = Self(17);
    pub const F16_16: Self = Self(25);
    pub const F16_16_16_16: Self = Self(26);
    pub const F16_16_FLOAT: Self = Self(31);
    pub const F16_16_16_16_FLOAT: Self = Self(32);
    pub const F32: Self = Self(33);
    pub const F32_32: Self = Self(34);
    pub const F32_32_32_32: Self = Self(35);
    pub const F32_FLOAT: Self = Self(36);
    pub const F32_32_FLOAT: Self = Self(37);
    pub const F32_32_32_32_FLOAT: Self = Self(38);
    pub const F32_32_32_FLOAT: Self = Self(57);

    /// Number of components the format holds, or `None` for unknown formats.
    pub fn component_count(self) -> Option<u32> {
        Some(match self {
            Self::F32 | Self::F32_FLOAT => 1,
            Self::F16_16 | Self::F16_16_FLOAT | Self::F32_32 | Self::F32_32_FLOAT => 2,
            Self::F10_11_11 | Self::F11_11_10 | Self::F32_32_32_FLOAT => 3,
            Self::F8_8_8_8
            | Self::F2_10_10_10
            | Self::F16_16_16_16
            | Self::F16_16_16_16_FLOAT
            | Self::F32_32_32_32
            | Self::F32_32_32_32_FLOAT => 4,
            _ => return None,
        })
    }
}

/// `vfetch_full` / `vfetch_mini`. Stride and offset count 32-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFetch {
    pub dest: u32,
    pub dest_relative: bool,
    /// Three bits per component: 0..=3 select x..w, 4 is 0.0, 5 is 1.0, 7
    /// keeps the register's component.
    pub dest_swizzle: u32,
    pub src: u32,
    pub src_relative: bool,
    /// Component of `src` holding the vertex index (0..=3).
    pub src_swizzle: u32,
    /// Vertex fetch constant, 0..=95 (three per 6-dword fetch slot).
    pub fetch_const: u32,
    pub prefetch_count: u32,
    pub mini: bool,
    pub format: VertexFormat,
    pub signed: bool,
    /// Normalize to -1..1 / 0..1 instead of converting the integer as is.
    pub normalized: bool,
    /// Signed normalization without a zero code (OpenGL style).
    pub signed_rf_no_zero: bool,
    pub index_rounded: bool,
    /// Power-of-two exponent bias applied to the result (-32..=31).
    pub exp_adjust: i32,
    pub stride: u32,
    pub offset: i32,
    /// `Some(v)`: runs only when the predicate equals `v`.
    pub predicate: Option<bool>,
}

/// Texture fetch family opcodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureOp {
    Fetch,
    GetBorderColorFrac,
    GetComputedLod,
    GetGradients,
    GetWeights,
    SetLod,
    SetGradientsHorz,
    SetGradientsVert,
}

impl TextureOp {
    fn from_opcode(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Fetch,
            16 => Self::GetBorderColorFrac,
            17 => Self::GetComputedLod,
            18 => Self::GetGradients,
            19 => Self::GetWeights,
            24 => Self::SetLod,
            25 => Self::SetGradientsHorz,
            26 => Self::SetGradientsVert,
            _ => return None,
        })
    }

    pub fn opcode(self) -> u32 {
        match self {
            Self::Fetch => 1,
            Self::GetBorderColorFrac => 16,
            Self::GetComputedLod => 17,
            Self::GetGradients => 18,
            Self::GetWeights => 19,
            Self::SetLod => 24,
            Self::SetGradientsHorz => 25,
            Self::SetGradientsVert => 26,
        }
    }
}

/// Texture dimension of a texture fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureDim {
    D1,
    D2,
    D3OrStacked,
    Cube,
}

/// `tfetch*` and the related texture instructions. Filter fields use 3 for
/// "take it from the fetch constant" (7 for the 3-bit ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureFetch {
    pub op: TextureOp,
    pub dest: u32,
    pub dest_relative: bool,
    pub dest_swizzle: u32,
    pub src: u32,
    pub src_relative: bool,
    /// Two bits per coordinate component, each a direct component index.
    pub src_swizzle: u32,
    /// Texture fetch constant slot, 0..=31.
    pub slot: u32,
    pub fetch_valid_only: bool,
    pub unnormalized: bool,
    pub mag_filter: u32,
    pub min_filter: u32,
    pub mip_filter: u32,
    pub aniso_filter: u32,
    pub arbitrary_filter: u32,
    pub vol_mag_filter: u32,
    pub vol_min_filter: u32,
    pub use_computed_lod: bool,
    pub use_register_lod: bool,
    pub use_register_gradients: bool,
    /// Sample at the pixel centre rather than the centroid.
    pub sample_at_center: bool,
    /// LOD bias in 1/16 steps (-64..=63).
    pub lod_bias: i32,
    pub dim: TextureDim,
    /// Texel offsets in half-texel steps (-16..=15).
    pub offset: [i32; 3],
    pub predicate: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchInstr {
    Vertex(VertexFetch),
    Texture(TextureFetch),
}

pub fn decode_fetch(w: [u32; 3]) -> Result<FetchInstr, DecodeError> {
    let opcode = bits(w[0], 0, 5);
    let predicate = bit(w[1], 31).then(|| bit(w[2], 31));
    if opcode == 0 {
        let sel = bits(w[0], 25, 2);
        if sel == 3 {
            return Err(DecodeError(
                "vertex fetch constant selector 3 is invalid".into(),
            ));
        }
        return Ok(FetchInstr::Vertex(VertexFetch {
            src: bits(w[0], 5, 6),
            src_relative: bit(w[0], 11),
            dest: bits(w[0], 12, 6),
            dest_relative: bit(w[0], 18),
            fetch_const: bits(w[0], 20, 5) * 3 + sel,
            prefetch_count: bits(w[0], 27, 3),
            src_swizzle: bits(w[0], 30, 2),
            dest_swizzle: bits(w[1], 0, 12),
            signed: bit(w[1], 12),
            normalized: !bit(w[1], 13),
            signed_rf_no_zero: bit(w[1], 14),
            index_rounded: bit(w[1], 15),
            format: VertexFormat(bits(w[1], 16, 6)),
            exp_adjust: sbits(w[1], 24, 6),
            mini: bit(w[1], 30),
            stride: bits(w[2], 0, 8),
            offset: sbits(w[2], 8, 23),
            predicate,
        }));
    }
    let op = TextureOp::from_opcode(opcode)
        .ok_or_else(|| DecodeError(format!("unknown fetch opcode {opcode}")))?;
    Ok(FetchInstr::Texture(TextureFetch {
        op,
        src: bits(w[0], 5, 6),
        src_relative: bit(w[0], 11),
        dest: bits(w[0], 12, 6),
        dest_relative: bit(w[0], 18),
        fetch_valid_only: bit(w[0], 19),
        slot: bits(w[0], 20, 5),
        unnormalized: bit(w[0], 25),
        src_swizzle: bits(w[0], 26, 6),
        dest_swizzle: bits(w[1], 0, 12),
        mag_filter: bits(w[1], 12, 2),
        min_filter: bits(w[1], 14, 2),
        mip_filter: bits(w[1], 16, 2),
        aniso_filter: bits(w[1], 18, 3),
        arbitrary_filter: bits(w[1], 21, 3),
        vol_mag_filter: bits(w[1], 24, 2),
        vol_min_filter: bits(w[1], 26, 2),
        use_computed_lod: bit(w[1], 28),
        use_register_lod: bit(w[1], 29),
        use_register_gradients: bit(w[2], 0),
        sample_at_center: bit(w[2], 1),
        lod_bias: sbits(w[2], 2, 7),
        dim: match bits(w[2], 14, 2) {
            0 => TextureDim::D1,
            1 => TextureDim::D2,
            2 => TextureDim::D3OrStacked,
            _ => TextureDim::Cube,
        },
        offset: [sbits(w[2], 16, 5), sbits(w[2], 21, 5), sbits(w[2], 26, 5)],
        predicate,
    }))
}

// ---------------------------------------------------------------------------
// ALU
// ---------------------------------------------------------------------------

macro_rules! opcode_enum {
    ($(#[$m:meta])* $name:ident { $($v:ident = $n:expr),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name { $($v = $n),* }
        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$v),*];
            pub fn from_u32(v: u32) -> Option<Self> {
                match v { $($n => Some(Self::$v),)* _ => None }
            }
        }
    };
}

opcode_enum! {
    /// Vector ALU opcodes (5-bit field).
    VectorOp {
        Add = 0, Mul = 1, Max = 2, Min = 3, Seq = 4, Sgt = 5, Sge = 6, Sne = 7,
        Frc = 8, Trunc = 9, Floor = 10, Mad = 11, CndEq = 12, CndGe = 13,
        CndGt = 14, Dp4 = 15, Dp3 = 16, Dp2Add = 17, Cube = 18, Max4 = 19,
        SetpEqPush = 20, SetpNePush = 21, SetpGtPush = 22, SetpGePush = 23,
        KillEq = 24, KillGt = 25, KillGe = 26, KillNe = 27, Dst = 28, MaxA = 29,
    }
}

opcode_enum! {
    /// Scalar ALU opcodes (6-bit field; 41 and 51.. are invalid).
    ScalarOp {
        Adds = 0, AddsPrev = 1, Muls = 2, MulsPrev = 3, MulsPrev2 = 4, Maxs = 5,
        Mins = 6, Seqs = 7, Sgts = 8, Sges = 9, Snes = 10, Frcs = 11,
        Truncs = 12, Floors = 13, Exp = 14, Logc = 15, Log = 16, Rcpc = 17,
        Rcpf = 18, Rcp = 19, Rsqc = 20, Rsqf = 21, Rsq = 22, MaxAs = 23,
        MaxAsf = 24, Subs = 25, SubsPrev = 26, SetpEq = 27, SetpNe = 28,
        SetpGt = 29, SetpGe = 30, SetpInv = 31, SetpPop = 32, SetpClr = 33,
        SetpRstr = 34, KillsEq = 35, KillsGt = 36, KillsGe = 37, KillsNe = 38,
        KillsOne = 39, Sqrt = 40, Mulsc0 = 42, Mulsc1 = 43, Addsc0 = 44,
        Addsc1 = 45, Subsc0 = 46, Subsc1 = 47, Sin = 48, Cos = 49,
        RetainPrev = 50,
    }
}

impl VectorOp {
    /// How many of src1..src3 the operation reads.
    pub fn operand_count(self) -> usize {
        use VectorOp::*;
        match self {
            Frc | Trunc | Floor | Max4 => 1,
            Mad | CndEq | CndGe | CndGt | Dp2Add => 3,
            _ => 2,
        }
    }
}

impl ScalarOp {
    /// Operands: 0 none, 1 the src3 register (`a` = its W swizzle, `b` = its
    /// X swizzle), 2 the constant/temporary pair of the `*sc` ops.
    pub fn operand_count(self) -> usize {
        use ScalarOp::*;
        match self {
            SetpClr | RetainPrev => 0,
            Mulsc0 | Mulsc1 | Addsc0 | Addsc1 | Subsc0 | Subsc1 => 2,
            _ => 1,
        }
    }
}

/// One ALU source slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AluSrc {
    /// For a temporary: bits 0..5 register, bit 6 loop-relative, bit 7
    /// absolute value. For a constant: the float constant index.
    pub reg: u32,
    /// Temporary register (true) or float constant (false).
    pub is_temp: bool,
    /// Per component i, `(swizzle >> 2i) + i` mod 4 is the source component.
    pub swizzle: u32,
    pub negate: bool,
}

/// One ALU instruction: a vector and a scalar operation issued together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AluInstr {
    pub vector_op: VectorOp,
    pub scalar_op: ScalarOp,
    pub vector_dest: u32,
    pub vector_dest_relative: bool,
    pub scalar_dest: u32,
    /// For exports this instead requests zeros in unwritten components.
    pub scalar_dest_relative: bool,
    /// Destinations are export registers instead of temporaries.
    pub export: bool,
    pub vector_write_mask: u32,
    pub scalar_write_mask: u32,
    pub vector_saturate: bool,
    pub scalar_saturate: bool,
    /// Take the absolute value of constant operands.
    pub abs_constants: bool,
    pub src: [AluSrc; 3],
    pub predicate: Option<bool>,
    /// Relative constants add a0 (true) or aL (false).
    pub const_relative_a0: bool,
    pub const0_relative: bool,
    pub const1_relative: bool,
}

impl AluInstr {
    /// Whether constant source `i` (1-based) uses relative addressing.
    pub fn src_const_relative(&self, i: usize) -> bool {
        let first = match i {
            1 => true,
            2 => self.src[0].is_temp,
            _ => self.src[0].is_temp && self.src[1].is_temp,
        };
        if first {
            self.const0_relative
        } else {
            self.const1_relative
        }
    }

    /// Temporary register of the `*sc` scalar ops' second operand.
    pub fn scalar_const_op_temp_reg(&self) -> u32 {
        (self.scalar_op as u32 & 1)
            | ((self.src[2].is_temp as u32) << 1)
            | (self.src[2].swizzle & 0x3C)
    }
}

/// Source component selected by an ALU swizzle for result component `i`.
pub fn alu_swizzle_component(swizzle: u32, i: u32) -> u32 {
    ((swizzle >> (2 * i)) + i) & 3
}

pub fn decode_alu(w: [u32; 3]) -> Result<AluInstr, DecodeError> {
    let vop = bits(w[2], 24, 5);
    let sop = bits(w[0], 26, 6);
    let vector_op = VectorOp::from_u32(vop)
        .ok_or_else(|| DecodeError(format!("invalid ALU vector opcode {vop}")))?;
    let scalar_op = ScalarOp::from_u32(sop)
        .ok_or_else(|| DecodeError(format!("invalid ALU scalar opcode {sop}")))?;
    let src = |i: u32| AluSrc {
        reg: bits(w[2], 16 - 8 * i, 8),
        is_temp: bit(w[2], 31 - i),
        swizzle: bits(w[1], 16 - 8 * i, 8),
        negate: bit(w[1], 26 - i),
    };
    Ok(AluInstr {
        vector_op,
        scalar_op,
        vector_dest: bits(w[0], 0, 6),
        vector_dest_relative: bit(w[0], 6),
        abs_constants: bit(w[0], 7),
        scalar_dest: bits(w[0], 8, 6),
        scalar_dest_relative: bit(w[0], 14),
        export: bit(w[0], 15),
        vector_write_mask: bits(w[0], 16, 4),
        scalar_write_mask: bits(w[0], 20, 4),
        vector_saturate: bit(w[0], 24),
        scalar_saturate: bit(w[0], 25),
        src: [src(0), src(1), src(2)],
        predicate: bit(w[1], 28).then(|| bit(w[1], 27)),
        const_relative_a0: bit(w[1], 29),
        const1_relative: bit(w[1], 30),
        const0_relative: bit(w[1], 31),
    })
}

/// An instruction in the ALU/fetch area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Instr {
    Alu(AluInstr),
    Fetch(FetchInstr),
}

/// Decodes the instruction at `w`; `is_fetch` comes from the exec sequence.
pub fn decode_instr(w: [u32; 3], is_fetch: bool) -> Result<Instr, DecodeError> {
    if is_fetch {
        decode_fetch(w).map(Instr::Fetch)
    } else {
        decode_alu(w).map(Instr::Alu)
    }
}
