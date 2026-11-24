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
