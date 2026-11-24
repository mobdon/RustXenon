// Bit layouts ported from Xenia src/xenia/gpu/ucode.h (BSD 3-Clause, see
// CREDITS.md).

//! Encoder for shader microcode, the inverse of [`crate::ucode`].
//!
//! Tests here and in later crates build programs by hand with it, so no
//! captured shader bytes are needed.

use crate::ucode::*;

fn put(v: u32, lo: u32, n: u32) -> u32 {
    (v & ((1u32 << n) - 1)) << lo
}

fn flag(b: bool, lo: u32) -> u32 {
    (b as u32) << lo
}

/// Encodes one control-flow instruction as `(low 32 bits, high 16 bits)`.
pub fn encode_cf(cf: &CfInstr) -> (u32, u32) {
    let op = |o: u32| o << 12;
    let branch = |cond: &BranchCond| -> (u32, u32) {
        match *cond {
            BranchCond::Always => (1 << 13, 0),
            BranchCond::Predicate(v) => (1 << 14, flag(v, 10)),
            BranchCond::Bool { index, value } => (0, put(index, 2, 8) | flag(value, 10)),
        }
    };
    match cf {
        CfInstr::Nop => (0, 0),
        CfInstr::Exec(e) => {
            let lo = put(e.address, 0, 12)
                | put(e.count, 12, 3)
                | flag(e.is_yield, 15)
                | put(e.sequence, 16, 12)
                | put(e.vc, 28, 4);
            let mut hi = put(e.vc >> 4, 0, 2);
            let end = e.end as u32;
            hi |= match e.cond {
                ExecCond::Always { predicate_clean } => op(1 + end) | flag(predicate_clean, 9),
                ExecCond::Bool {
                    index,
                    value,
                    predicate_clean,
                } => {
                    let base = if predicate_clean { 13 } else { 3 };
                    op(base + end) | put(index, 2, 8) | flag(value, 10)
                }
                ExecCond::Predicate {
                    value,
                    predicate_clean,
                } => op(5 + end) | flag(predicate_clean, 9) | flag(value, 10),
            };
            (lo, hi)
        }
        CfInstr::LoopStart {
            address,
            repeat,
            loop_id,
        } => (
            put(*address, 0, 13) | flag(*repeat, 13) | put(*loop_id, 16, 5),
            op(7),
        ),
        CfInstr::LoopEnd {
            address,
            loop_id,
            predicated_break,
        } => (
            put(*address, 0, 13) | put(*loop_id, 16, 5) | flag(predicated_break.is_some(), 21),
            op(8) | flag(predicated_break.unwrap_or(false), 10),
        ),
        CfInstr::CondCall { address, cond } => {
            let (lo, hi) = branch(cond);
            (put(*address, 0, 13) | lo, op(9) | hi)
        }
        CfInstr::Return => (0, op(10)),
        CfInstr::CondJmp {
            address,
            cond,
            direction,
        } => {
            let (lo, hi) = branch(cond);
            (put(*address, 0, 13) | lo, op(11) | hi | flag(*direction, 1))
        }
        CfInstr::Alloc {
            size,
            alloc_type,
            unserialized,
        } => (
            put(*size, 0, 3),
            op(12) | flag(*unserialized, 8) | put(*alloc_type, 9, 2),
        ),
        CfInstr::MarkVsFetchDone => (0, op(15)),
    }
}

/// Packs two control-flow instructions into three words.
pub fn encode_cf_pair(a: &CfInstr, b: &CfInstr) -> [u32; 3] {
    let (a0, a1) = encode_cf(a);
    let (b0, b1) = encode_cf(b);
    [a0, (a1 & 0xFFFF) | (b0 << 16), (b0 >> 16) | (b1 << 16)]
}

pub fn encode_fetch(f: &FetchInstr) -> [u32; 3] {
    match f {
        FetchInstr::Vertex(v) => {
            let (pred, cond) = (v.predicate.is_some(), v.predicate.unwrap_or(false));
            [
                put(v.src, 5, 6)
                    | flag(v.src_relative, 11)
                    | put(v.dest, 12, 6)
                    | flag(v.dest_relative, 18)
                    | (1 << 19)
                    | put(v.fetch_const / 3, 20, 5)
                    | put(v.fetch_const % 3, 25, 2)
                    | put(v.prefetch_count, 27, 3)
                    | put(v.src_swizzle, 30, 2),
                put(v.dest_swizzle, 0, 12)
                    | flag(v.signed, 12)
                    | flag(!v.normalized, 13)
                    | flag(v.signed_rf_no_zero, 14)
                    | flag(v.index_rounded, 15)
                    | put(v.format.0, 16, 6)
                    | put(v.exp_adjust as u32, 24, 6)
                    | flag(v.mini, 30)
                    | flag(pred, 31),
                put(v.stride, 0, 8) | put(v.offset as u32, 8, 23) | flag(cond, 31),
            ]
        }
        FetchInstr::Texture(t) => {
            let (pred, cond) = (t.predicate.is_some(), t.predicate.unwrap_or(false));
            let dim = match t.dim {
                TextureDim::D1 => 0,
                TextureDim::D2 => 1,
                TextureDim::D3OrStacked => 2,
                TextureDim::Cube => 3,
            };
            [
                t.op.opcode()
                    | put(t.src, 5, 6)
                    | flag(t.src_relative, 11)
                    | put(t.dest, 12, 6)
                    | flag(t.dest_relative, 18)
                    | flag(t.fetch_valid_only, 19)
                    | put(t.slot, 20, 5)
                    | flag(t.unnormalized, 25)
                    | put(t.src_swizzle, 26, 6),
                put(t.dest_swizzle, 0, 12)
                    | put(t.mag_filter, 12, 2)
                    | put(t.min_filter, 14, 2)
                    | put(t.mip_filter, 16, 2)
                    | put(t.aniso_filter, 18, 3)
                    | put(t.arbitrary_filter, 21, 3)
                    | put(t.vol_mag_filter, 24, 2)
                    | put(t.vol_min_filter, 26, 2)
                    | flag(t.use_computed_lod, 28)
                    | flag(t.use_register_lod, 29)
                    | flag(pred, 31),
                flag(t.use_register_gradients, 0)
                    | flag(t.sample_at_center, 1)
                    | put(t.lod_bias as u32, 2, 7)
                    | put(dim, 14, 2)
                    | put(t.offset[0] as u32, 16, 5)
                    | put(t.offset[1] as u32, 21, 5)
                    | put(t.offset[2] as u32, 26, 5)
                    | flag(cond, 31),
            ]
        }
    }
}

pub fn encode_alu(a: &AluInstr) -> [u32; 3] {
    let (pred, cond) = (a.predicate.is_some(), a.predicate.unwrap_or(false));
    let mut w1 = flag(pred, 28)
        | flag(cond, 27)
        | flag(a.const_relative_a0, 29)
        | flag(a.const1_relative, 30)
        | flag(a.const0_relative, 31);
    let mut w2 = put(a.vector_op as u32, 24, 5);
    for (i, s) in a.src.iter().enumerate() {
        let i = i as u32;
        w1 |= put(s.swizzle, 16 - 8 * i, 8) | flag(s.negate, 26 - i);
        w2 |= put(s.reg, 16 - 8 * i, 8) | flag(s.is_temp, 31 - i);
    }
    [
        put(a.vector_dest, 0, 6)
            | flag(a.vector_dest_relative, 6)
            | flag(a.abs_constants, 7)
            | put(a.scalar_dest, 8, 6)
            | flag(a.scalar_dest_relative, 14)
            | flag(a.export, 15)
            | put(a.vector_write_mask, 16, 4)
            | put(a.scalar_write_mask, 20, 4)
            | flag(a.vector_saturate, 24)
            | flag(a.scalar_saturate, 25)
            | put(a.scalar_op as u32, 26, 6),
        w1,
        w2,
    ]
}

pub fn encode_instr(i: &Instr) -> [u32; 3] {
    match i {
        Instr::Alu(a) => encode_alu(a),
        Instr::Fetch(f) => encode_fetch(f),
    }
}

/// Builds a whole program. Exec addresses in `cf` are indices into `instrs`;
/// the control-flow area size is added here. The fetch bits of every exec
/// sequence are filled in from the instruction kinds (the serialize bits are
/// kept as given). Jump, call and loop addresses are control-flow indices
/// and stay as given.
pub fn assemble(cf: &[CfInstr], instrs: &[Instr]) -> Vec<u32> {
    let pairs = cf.len().div_ceil(2).max(1) as u32;
    let mut out = Vec::new();
    let fixed: Vec<CfInstr> = cf
        .iter()
        .map(|c| match *c {
            CfInstr::Exec(mut e) => {
                let mut seq = e.sequence & 0xAAA;
                for i in 0..e.count {
                    if let Some(Instr::Fetch(_)) = instrs.get((e.address + i) as usize) {
                        seq |= 1 << (2 * i);
                    }
                }
                e.sequence = seq;
                e.address += pairs;
                CfInstr::Exec(e)
            }
            other => other,
        })
        .collect();
    for i in 0..pairs as usize {
        let a = fixed.get(2 * i).copied().unwrap_or(CfInstr::Nop);
        let b = fixed.get(2 * i + 1).copied().unwrap_or(CfInstr::Nop);
        out.extend_from_slice(&encode_cf_pair(&a, &b));
    }
    for i in instrs {
        out.extend_from_slice(&encode_instr(i));
    }
    out
}

// ---------------------------------------------------------------------------
// Conveniences for hand-written programs
// ---------------------------------------------------------------------------

/// Encodes an ALU source swizzle written as up to four of `xyzw`
/// (a shorter string repeats its last letter, like the assembler).
pub fn swz(s: &str) -> u32 {
    let comps: Vec<u32> = s
        .chars()
        .map(|c| match c {
            'x' | 'r' => 0,
            'y' | 'g' => 1,
            'z' | 'b' => 2,
            'w' | 'a' => 3,
            _ => panic!("bad swizzle letter {c}"),
        })
        .collect();
    assert!(!comps.is_empty() && comps.len() <= 4, "bad swizzle {s}");
    (0..4u32)
        .map(|i| {
            let c = comps[(i as usize).min(comps.len() - 1)];
            ((c + 4 - i) & 3) << (2 * i)
        })
        .sum()
}

/// Identity ALU swizzle (`xyzw`), which encodes as 0.
pub const SWZ_XYZW: u32 = 0;

/// Encodes a fetch destination swizzle from four of `xyzw01_` (`_` keeps).
pub fn fetch_dest_swz(s: &str) -> u32 {
    assert_eq!(s.len(), 4, "fetch destination swizzle needs 4 letters");
    s.chars()
        .enumerate()
        .map(|(i, c)| {
            let v = match c {
                'x' => 0,
                'y' => 1,
                'z' => 2,
                'w' => 3,
                '0' => 4,
                '1' => 5,
                '_' => 7,
                _ => panic!("bad fetch swizzle letter {c}"),
            };
            v << (3 * i)
        })
        .sum()
}

/// Write mask from a subset of `xyzw`.
pub fn mask(s: &str) -> u32 {
    s.chars()
        .map(|c| match c {
            'x' => 1,
            'y' => 2,
            'z' => 4,
            'w' => 8,
            _ => panic!("bad mask letter {c}"),
        })
        .fold(0, |a, b| a | b)
}

impl AluSrc {
    /// Temporary register `r<reg>` with a swizzle from [`swz`].
    pub fn temp(reg: u32, swizzle: u32) -> Self {
        AluSrc {
            reg,
            is_temp: true,
            swizzle,
            negate: false,
        }
    }

    /// Float constant `c<index>` with a swizzle from [`swz`].
    pub fn constant(index: u32, swizzle: u32) -> Self {
        AluSrc {
            reg: index,
            is_temp: false,
            swizzle,
            negate: false,
        }
    }

    pub fn negated(mut self) -> Self {
        self.negate = true;
        self
    }

    /// Absolute value of a temporary (bit 7 of the register field).
    pub fn abs(mut self) -> Self {
        self.reg |= 0x80;
        self
    }

    /// Loop-relative temporary (bit 6 of the register field).
    pub fn rel(mut self) -> Self {
        self.reg |= 0x40;
        self
    }
}

impl AluInstr {
    /// An instruction that does nothing: `max r0._, r0, r0` with
    /// `retain_prev`. Fill in fields from here.
    pub fn nop() -> Self {
        AluInstr {
            vector_op: VectorOp::Max,
            scalar_op: ScalarOp::RetainPrev,
            vector_dest: 0,
            vector_dest_relative: false,
            scalar_dest: 0,
            scalar_dest_relative: false,
            export: false,
            vector_write_mask: 0,
            scalar_write_mask: 0,
            vector_saturate: false,
            scalar_saturate: false,
            abs_constants: false,
            src: [AluSrc::temp(0, 0); 3],
            predicate: None,
            const_relative_a0: false,
            const0_relative: false,
            const1_relative: false,
        }
    }

    /// Vector operation into temporary `dest` with `write_mask`.
    pub fn vector(op: VectorOp, dest: u32, write_mask: u32, src: &[AluSrc]) -> Self {
        let mut a = Self::nop();
        a.vector_op = op;
        a.vector_dest = dest;
        a.vector_write_mask = write_mask;
        for (slot, s) in a.src.iter_mut().zip(src) {
            *slot = *s;
        }
        a
    }

    /// Scalar operation reading src3 into temporary `dest`.
    pub fn scalar(op: ScalarOp, dest: u32, write_mask: u32, src3: AluSrc) -> Self {
        let mut a = Self::nop();
        a.scalar_op = op;
        a.scalar_dest = dest;
        a.scalar_write_mask = write_mask;
        a.src[2] = src3;
        a
    }

    /// Turns the destinations into export register `reg`.
    pub fn exported(mut self, reg: u32) -> Self {
        self.export = true;
        self.vector_dest = reg;
        self.scalar_dest = reg;
        self
    }

    pub fn predicated(mut self, value: bool) -> Self {
        self.predicate = Some(value);
        self
    }
}

impl VertexFetch {
    /// `vfetch_full r<dest>, r<src>.x, vf<fetch_const>` with 32-bit float
    /// data, stride and offset in words.
    pub fn full(
        dest: u32,
        src: u32,
        fetch_const: u32,
        format: VertexFormat,
        stride: u32,
        offset: i32,
    ) -> Self {
        VertexFetch {
            dest,
            dest_relative: false,
            dest_swizzle: fetch_dest_swz("xyzw"),
            src,
            src_relative: false,
            src_swizzle: 0,
            fetch_const,
            prefetch_count: 0,
            mini: false,
            format,
            signed: false,
            normalized: true,
            signed_rf_no_zero: false,
            index_rounded: false,
            exp_adjust: 0,
            stride,
            offset,
            predicate: None,
        }
    }
}

impl TextureFetch {
    /// `tfetch2D r<dest>, r<src>.xy, tf<slot>` with every filter taken
    /// from the fetch constant.
    pub fn fetch_2d(dest: u32, src: u32, slot: u32) -> Self {
        TextureFetch {
            op: TextureOp::Fetch,
            dest,
            dest_relative: false,
            dest_swizzle: fetch_dest_swz("xyzw"),
            src,
            src_relative: false,
            src_swizzle: 0b01_00, // x, y
            slot,
            fetch_valid_only: true,
            unnormalized: false,
            mag_filter: 3,
            min_filter: 3,
            mip_filter: 3,
            aniso_filter: 7,
            arbitrary_filter: 7,
            vol_mag_filter: 3,
            vol_min_filter: 3,
            use_computed_lod: true,
            use_register_lod: false,
            use_register_gradients: false,
            sample_at_center: false,
            lod_bias: 0,
            dim: TextureDim::D2,
            offset: [0; 3],
            predicate: None,
        }
    }
}

impl Exec {
    /// Unconditional exec of `count` instructions starting at `address`.
    pub fn new(address: u32, count: u32, end: bool) -> Self {
        Exec {
            cond: ExecCond::Always {
                predicate_clean: true,
            },
            end,
            address,
            count,
            sequence: 0,
            is_yield: false,
            vc: 0,
        }
    }

    pub fn with_cond(mut self, cond: ExecCond) -> Self {
        self.cond = cond;
        self
    }
}
