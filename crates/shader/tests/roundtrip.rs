//! decode(encode(x)) == x for every instruction kind, plus fuzzing of the
//! decoders with random words.

mod common;

use common::Lcg;
use xenon_shader::asm::*;
use xenon_shader::ucode::*;

fn cf_roundtrip(a: CfInstr, b: CfInstr) {
    let w = encode_cf_pair(&a, &b);
    assert_eq!(decode_cf_pair(w), [a, b], "words {w:08x?}");
}

#[test]
fn control_flow_roundtrip() {
    let exec = Exec {
        cond: ExecCond::Always {
            predicate_clean: true,
        },
        end: false,
        address: 0xABC,
        count: 6,
        sequence: 0b1001_0110_0101,
        is_yield: true,
        vc: 0b10_1101,
    };
    let all = [
        CfInstr::Nop,
        CfInstr::Exec(exec),
        CfInstr::Exec(Exec { end: true, ..exec }),
        CfInstr::Exec(Exec {
            cond: ExecCond::Bool {
                index: 201,
                value: true,
                predicate_clean: false,
            },
            ..exec
        }),
        CfInstr::Exec(Exec {
            cond: ExecCond::Bool {
                index: 3,
                value: false,
                predicate_clean: true,
            },
            end: true,
            ..exec
        }),
        CfInstr::Exec(Exec {
            cond: ExecCond::Predicate {
                value: false,
                predicate_clean: false,
            },
            ..exec
        }),
        CfInstr::Exec(Exec {
            cond: ExecCond::Predicate {
                value: true,
                predicate_clean: true,
            },
            end: true,
            ..exec
        }),
        CfInstr::LoopStart {
            address: 0x1FFF,
            repeat: true,
            loop_id: 31,
        },
        CfInstr::LoopEnd {
            address: 5,
            loop_id: 7,
            predicated_break: None,
        },
        CfInstr::LoopEnd {
            address: 5,
            loop_id: 7,
            predicated_break: Some(true),
        },
        CfInstr::LoopEnd {
            address: 5,
            loop_id: 7,
            predicated_break: Some(false),
        },
        CfInstr::CondCall {
            address: 77,
            cond: BranchCond::Always,
        },
        CfInstr::CondCall {
            address: 77,
            cond: BranchCond::Predicate(true),
        },
        CfInstr::CondCall {
            address: 77,
            cond: BranchCond::Bool {
                index: 255,
                value: true,
            },
        },
        CfInstr::Return,
        CfInstr::CondJmp {
            address: 9,
            cond: BranchCond::Bool {
                index: 17,
                value: false,
            },
            direction: true,
        },
        CfInstr::CondJmp {
            address: 9,
            cond: BranchCond::Predicate(false),
            direction: false,
        },
        CfInstr::Alloc {
            size: 5,
            alloc_type: 2,
            unserialized: true,
        },
        CfInstr::MarkVsFetchDone,
    ];
    for a in all {
        for b in all {
            cf_roundtrip(a, b);
        }
    }
}

fn random_alu(g: &mut Lcg) -> AluInstr {
    let vops = VectorOp::ALL;
    let sops = ScalarOp::ALL;
    let mut src = [AluSrc::temp(0, 0); 3];
    for s in &mut src {
        *s = AluSrc {
            reg: g.next() & 0xFF,
            is_temp: g.next() & 1 != 0,
            swizzle: g.next() & 0xFF,
            negate: g.next() & 1 != 0,
        };
    }
    let b = |g: &mut Lcg| g.next() & 1 != 0;
    AluInstr {
        vector_op: vops[g.next() as usize % vops.len()],
        scalar_op: sops[g.next() as usize % sops.len()],
        vector_dest: g.next() & 63,
        vector_dest_relative: b(g),
        scalar_dest: g.next() & 63,
        scalar_dest_relative: b(g),
        export: b(g),
        vector_write_mask: g.next() & 15,
        scalar_write_mask: g.next() & 15,
        vector_saturate: b(g),
        scalar_saturate: b(g),
        abs_constants: b(g),
        src,
        predicate: if b(g) { Some(b(g)) } else { None },
        const_relative_a0: b(g),
        const0_relative: b(g),
        const1_relative: b(g),
    }
}

#[test]
fn alu_roundtrip() {
    let mut g = Lcg(1);
    for _ in 0..5000 {
        let a = random_alu(&mut g);
        assert_eq!(decode_alu(encode_alu(&a)), Ok(a));
    }
}

#[test]
fn fetch_roundtrip() {
    let mut g = Lcg(2);
    for _ in 0..5000 {
        let b = |g: &mut Lcg| g.next() & 1 != 0;
        let pred = if b(&mut g) { Some(b(&mut g)) } else { None };
        let v = VertexFetch {
            dest: g.next() & 63,
            dest_relative: b(&mut g),
            dest_swizzle: g.next() & 0xFFF,
            src: g.next() & 63,
            src_relative: b(&mut g),
            src_swizzle: g.next() & 3,
            fetch_const: g.next() % 96,
            prefetch_count: g.next() & 7,
            mini: b(&mut g),
            format: VertexFormat(g.next() & 63),
            signed: b(&mut g),
            normalized: b(&mut g),
            signed_rf_no_zero: b(&mut g),
            index_rounded: b(&mut g),
            exp_adjust: (g.next() & 63) as i32 - 32,
            stride: g.next() & 0xFF,
            offset: (g.next() & 0x7F_FFFF) as i32 - 0x40_0000,
            predicate: pred,
        };
        let f = FetchInstr::Vertex(v);
        assert_eq!(decode_fetch(encode_fetch(&f)), Ok(f));

        let ops = [
            TextureOp::Fetch,
            TextureOp::GetBorderColorFrac,
            TextureOp::GetComputedLod,
            TextureOp::GetGradients,
            TextureOp::GetWeights,
            TextureOp::SetLod,
            TextureOp::SetGradientsHorz,
            TextureOp::SetGradientsVert,
        ];
        let dims = [
            TextureDim::D1,
            TextureDim::D2,
            TextureDim::D3OrStacked,
            TextureDim::Cube,
        ];
        let t = TextureFetch {
            op: ops[g.next() as usize % ops.len()],
            dest: g.next() & 63,
            dest_relative: b(&mut g),
            dest_swizzle: g.next() & 0xFFF,
            src: g.next() & 63,
            src_relative: b(&mut g),
            src_swizzle: g.next() & 63,
            slot: g.next() & 31,
            fetch_valid_only: b(&mut g),
            unnormalized: b(&mut g),
            mag_filter: g.next() & 3,
            min_filter: g.next() & 3,
            mip_filter: g.next() & 3,
            aniso_filter: g.next() & 7,
            arbitrary_filter: g.next() & 7,
            vol_mag_filter: g.next() & 3,
            vol_min_filter: g.next() & 3,
            use_computed_lod: b(&mut g),
            use_register_lod: b(&mut g),
            use_register_gradients: b(&mut g),
            sample_at_center: b(&mut g),
            lod_bias: (g.next() & 127) as i32 - 64,
            dim: dims[g.next() as usize % 4],
            offset: [
                (g.next() & 31) as i32 - 16,
                (g.next() & 31) as i32 - 16,
                (g.next() & 31) as i32 - 16,
            ],
            predicate: pred,
        };
        let f = FetchInstr::Texture(t);
        assert_eq!(decode_fetch(encode_fetch(&f)), Ok(f));
    }
}

/// Random words: decoding never panics, and whatever decodes re-encodes to
/// something that decodes the same.
#[test]
fn decoders_are_stable_on_random_words() {
    let mut g = Lcg(3);
    for _ in 0..20000 {
        let w = [g.next(), g.next(), g.next()];
        let [a, b] = decode_cf_pair(w);
        assert_eq!(decode_cf_pair(encode_cf_pair(&a, &b)), [a, b]);
        if let Ok(a) = decode_alu(w) {
            assert_eq!(decode_alu(encode_alu(&a)), Ok(a));
        }
        if let Ok(f) = decode_fetch(w) {
            assert_eq!(decode_fetch(encode_fetch(&f)), Ok(f));
        }
    }
}

#[test]
fn invalid_opcodes_are_errors() {
    let mut a = encode_alu(&AluInstr::nop());
    a[2] |= 30 << 24; // vector opcode 30
    assert!(decode_alu(a).is_err());
    let mut a = encode_alu(&AluInstr::nop());
    a[0] = (a[0] & !(63 << 26)) | (41 << 26); // scalar opcode 41
    assert!(decode_alu(a).is_err());
    assert!(decode_fetch([2, 0, 0]).is_err()); // fetch opcode 2
    assert!(decode_fetch([3 << 25, 0, 0]).is_err()); // vertex const selector 3
}

#[test]
fn swizzle_helpers() {
    assert_eq!(swz("xyzw"), 0);
    // Component i stores (source - i) mod 4.
    assert_eq!(alu_swizzle_component(swz("wzyx"), 0), 3);
    assert_eq!(alu_swizzle_component(swz("wzyx"), 3), 0);
    assert_eq!(alu_swizzle_component(swz("x"), 2), 0);
    assert_eq!(mask("xw"), 0b1001);
    assert_eq!(fetch_dest_swz("xyz1"), (1 << 3) | (2 << 6) | (5 << 9));
}

// The tests above only show that the encoder and decoder agree with each
// other. These build words bit by bit from the field positions in Xenia's
// src/xenia/gpu/ucode.h, so a layout mistake shared by both sides shows up.

#[test]
fn alu_words_match_xenia_layout() {
    // ucode.h AluInstruction::Data.
    let w0 = 5 // vector_dest
        | 1 << 6 // vector_dest_rel
        | 1 << 7 // abs_constants
        | 9 << 8 // scalar_dest
        | 1 << 15 // export_data
        | 0b0011 << 16 // vector_write_mask
        | 0b0100 << 20 // scalar_write_mask
        | 1 << 24 // vector_clamp
        | 27 << 26; // scalar_opc = kSetpEq
    let w1 = 0x1B // src3_swiz
        | 0x24 << 8 // src2_swiz
        | 0xE4 << 16 // src1_swiz
        | 1 << 24 // src3_reg_negate
        | 1 << 26 // src1_reg_negate
        | 1 << 27 // pred_condition
        | 1 << 28 // is_predicated
        | 1 << 29 // const_address_register_relative
        | 1 << 31; // const_0_rel_abs
    let w2 = 0x12 // src3_reg
        | 0x34 << 8 // src2_reg
        | 0x56 << 16 // src1_reg
        | 11 << 24 // vector_opc = kMad
        | 1 << 29 // src3_sel
        | 1 << 31; // src1_sel
    let a = decode_alu([w0, w1, w2]).unwrap();
    assert_eq!(a.vector_op, VectorOp::Mad);
    assert_eq!(a.scalar_op, ScalarOp::SetpEq);
    assert_eq!((a.vector_dest, a.vector_dest_relative), (5, true));
    assert_eq!((a.scalar_dest, a.scalar_dest_relative), (9, false));
    assert!(a.abs_constants && a.export && a.vector_saturate && !a.scalar_saturate);
    assert_eq!((a.vector_write_mask, a.scalar_write_mask), (0b0011, 0b0100));
    let regs = a.src.map(|s| (s.reg, s.is_temp, s.swizzle, s.negate));
    assert_eq!(
        regs,
        [
            (0x56, true, 0xE4, true),
            (0x34, false, 0x24, false),
            (0x12, true, 0x1B, true),
        ]
    );
    assert_eq!(a.predicate, Some(true));
    assert!(a.const_relative_a0 && a.const0_relative && !a.const1_relative);
    // ucode.h src_const_is_addressed: src2 is the first constant, so it uses
    // const_0_rel_abs.
    assert!(a.src_const_relative(2));
    // scalar_const_reg_op_src_temp_reg: opcode bit 0 (1 for kSetpEq = 27),
    // src3_sel << 1, src3_swiz & 0x3C.
    assert_eq!(a.scalar_const_op_temp_reg(), 1 | 2 | (0x1B & 0x3C));
    assert_eq!(encode_alu(&a), [w0, w1, w2]);
}

#[test]
fn fetch_words_match_xenia_layout() {
    // ucode.h VertexFetchInstruction::Data.
    let w0 = 3 << 5 // src_reg
        | 1 << 11 // src_reg_am
        | 7 << 12 // dst_reg
        | 1 << 19 // must_be_one
        | 10 << 20 // const_index
        | 2 << 25 // const_index_sel
        | 5 << 27 // prefetch_count
        | 2 << 30; // src_swiz
    let w1 = 0x688 // dst_swiz
        | 1 << 12 // fomat_comp_all (signed)
        | 1 << 13 // num_format_all (integer, not normalized)
        | 1 << 15 // is_index_rounded
        | 26 << 16 // format = k_16_16_16_16
        | 0b111101 << 24 // exp_adjust = -3
        | 1 << 31; // is_predicated
    let w2 = 12 // stride
        | ((-5i32 as u32) & 0x7F_FFFF) << 8; // offset, pred_condition 0
    let FetchInstr::Vertex(v) = decode_fetch([w0, w1, w2]).unwrap() else {
        panic!("not a vertex fetch");
    };
    assert_eq!(
        (v.src, v.src_relative, v.dest, v.dest_relative),
        (3, true, 7, false)
    );
    assert_eq!(v.fetch_const, 10 * 3 + 2);
    assert_eq!(
        (v.prefetch_count, v.src_swizzle, v.dest_swizzle),
        (5, 2, 0x688)
    );
    assert!(v.signed && !v.normalized && !v.signed_rf_no_zero && v.index_rounded);
    assert_eq!(v.format, VertexFormat::F16_16_16_16);
    assert_eq!((v.exp_adjust, v.stride, v.offset), (-3, 12, -5));
    assert!(!v.mini);
    assert_eq!(v.predicate, Some(false));
    assert_eq!(encode_fetch(&FetchInstr::Vertex(v)), [w0, w1, w2]);

    // ucode.h TextureFetchInstruction::Data.
    let w0 = 1 // opcode = kTextureFetch
        | 4 << 5 // src_reg
        | 8 << 12 // dst_reg
        | 1 << 18 // dst_reg_am
        | 1 << 19 // fetch_valid_only
        | 17 << 20 // const_index
        | 1 << 25 // tx_coord_denorm
        | 0b10_01_00 << 26; // src_swiz
    let w1 = 0xFAC // dst_swiz
        | 1 << 12 // mag_filter
        | 2 << 14 // min_filter
        | 3 << 16 // mip_filter
        | 5 << 18 // aniso_filter
        | 6 << 21 // arbitrary_filter
        | 1 << 24 // vol_mag_filter
        | 1 << 28; // use_comp_lod
    let w2 = 1 << 1 // sample_location = kCenter
        | ((-7i32 as u32) & 0x7F) << 2 // lod_bias
        | 2 << 14 // dimension = k3DOrStacked
        | 3 << 16 // offset_x
        | ((-4i32 as u32) & 0x1F) << 21 // offset_y
        | ((-16i32 as u32) & 0x1F) << 26; // offset_z
    let FetchInstr::Texture(t) = decode_fetch([w0, w1, w2]).unwrap() else {
        panic!("not a texture fetch");
    };
    assert_eq!(t.op, TextureOp::Fetch);
    assert_eq!(
        (t.src, t.src_relative, t.dest, t.dest_relative),
        (4, false, 8, true)
    );
    assert!(t.fetch_valid_only && t.unnormalized);
    assert_eq!(
        (t.slot, t.src_swizzle, t.dest_swizzle),
        (17, 0b10_01_00, 0xFAC)
    );
    assert_eq!(
        [t.mag_filter, t.min_filter, t.mip_filter, t.aniso_filter],
        [1, 2, 3, 5]
    );
    assert_eq!(
        [t.arbitrary_filter, t.vol_mag_filter, t.vol_min_filter],
        [6, 1, 0]
    );
    assert!(t.use_computed_lod && !t.use_register_lod && !t.use_register_gradients);
    assert!(t.sample_at_center);
    assert_eq!(t.lod_bias, -7);
    assert_eq!(t.dim, TextureDim::D3OrStacked);
    assert_eq!(t.offset, [3, -4, -16]);
    assert_eq!(t.predicate, None);
    assert_eq!(encode_fetch(&FetchInstr::Texture(t)), [w0, w1, w2]);
}

#[test]
fn control_flow_words_match_xenia_layout() {
    // ucode.h ControlFlowExecInstruction (kExecEnd) and
    // ControlFlowCondJmpInstruction, packed as UnpackControlFlowInstructions
    // unpacks them: A is dword 0 + low half of dword 1, B the rest.
    let a_lo = 0x123 // address
        | 5 << 12 // count
        | 1 << 15 // is_yield
        | 0x9A5 << 16 // sequence
        | 0xA << 28; // vc_hi
    let a_hi = 2 // vc_lo
        | 1 << 9 // is_predicate_clean
        | 2 << 12; // opcode = kExecEnd
    let b_lo = 0x42 // address
        | 1 << 14; // is_predicated
    let b_hi = 1 << 1 // direction
        | 1 << 10 // condition
        | 11 << 12; // opcode = kCondJmp
    let w = [a_lo, a_hi | b_lo << 16, b_lo >> 16 | b_hi << 16];
    let [a, b] = decode_cf_pair(w);
    assert_eq!(
        a,
        CfInstr::Exec(Exec {
            cond: ExecCond::Always {
                predicate_clean: true
            },
            end: true,
            address: 0x123,
            count: 5,
            sequence: 0x9A5,
            is_yield: true,
            vc: 0xA | 2 << 4,
        })
    );
    assert_eq!(
        b,
        CfInstr::CondJmp {
            address: 0x42,
            cond: BranchCond::Predicate(true),
            direction: true,
        }
    );
    assert_eq!(encode_cf_pair(&a, &b), w);

    // ControlFlowCondExecInstruction with kCondExecPredCleanEnd (a bool
    // constant test) and ControlFlowLoopEndInstruction.
    let a_lo = 0x040 | 2 << 12;
    let a_hi = 0x5C << 2 // bool_address
        | 1 << 10 // condition
        | 14 << 12; // opcode
    let b_lo = 0xABC // address
        | 0x13 << 16 // loop_id
        | 1 << 21; // is_predicated_break
    let b_hi = 8 << 12; // opcode = kLoopEnd, condition 0
    let w = [a_lo, a_hi | b_lo << 16, b_lo >> 16 | b_hi << 16];
    let [a, b] = decode_cf_pair(w);
    let CfInstr::Exec(e) = a else {
        panic!("not an exec");
    };
    assert_eq!(
        e.cond,
        ExecCond::Bool {
            index: 0x5C,
            value: true,
            predicate_clean: true
        }
    );
    assert!(e.end);
    assert_eq!((e.address, e.count), (0x40, 2));
    assert_eq!(
        b,
        CfInstr::LoopEnd {
            address: 0xABC,
            loop_id: 0x13,
            predicated_break: Some(false),
        }
    );
}
