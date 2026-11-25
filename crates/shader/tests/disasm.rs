//! The disassembler lists every instruction with its fields and collects
//! the constants and fetches a program uses.

use xenon_shader::asm::*;
use xenon_shader::disasm::disassemble;
use xenon_shader::ucode::*;
use xenon_shader::ShaderStage;

#[test]
fn lists_fetches_alu_and_constants() {
    let vf = VertexFetch {
        signed: true,
        exp_adjust: -2,
        dest_swizzle: fetch_dest_swz("xyz1"),
        ..VertexFetch::full(1, 0, 4, VertexFormat::F2_10_10_10, 3, 2)
    };
    let mad = AluInstr::vector(
        VectorOp::Mad,
        2,
        mask("xyz"),
        &[
            AluSrc::temp(1, swz("xyzw")).abs(),
            AluSrc::constant(7, swz("wzyx")).negated(),
            AluSrc::constant(9, swz("x")),
        ],
    );
    let rsq = AluInstr::scalar(ScalarOp::Rsq, 3, mask("w"), AluSrc::temp(2, swz("x")));
    let mut rel = AluInstr::vector(
        VectorOp::Mul,
        0,
        mask("xyzw"),
        &[
            AluSrc::temp(2, swz("xyzw")),
            AluSrc::constant(20, swz("xyzw")),
        ],
    )
    .exported(62);
    rel.const0_relative = true;
    rel.const_relative_a0 = true;
    let words = assemble(
        &[CfInstr::Exec(Exec::new(0, 4, true))],
        &[
            Instr::Fetch(FetchInstr::Vertex(vf)),
            Instr::Alu(mad),
            Instr::Alu(rsq),
            Instr::Alu(rel),
        ],
    );
    let d = disassemble(&words, ShaderStage::Vertex);
    let t = &d.text;
    for want in [
        "exece addr 1 count 4",
        "vfetch_full r1.xyz1, r0.x, vf4 format 2_10_10_10(7) signed normalized exp_adjust -2 stride 3 offset 2",
        "mad r2.xyz_, |r1|, -c7.wzyx, c9.xxxx",
        "rsq r3.___w, r2.xxxx",
        "mul oPos.xyzw, r2, c[20+a0]",
    ] {
        assert!(t.contains(want), "missing {want:?} in\n{t}");
    }
    assert_eq!(d.float_consts, [7, 9]);
    assert!(d.float_consts_relative);
    assert_eq!(d.fetches, [FetchInstr::Vertex(vf)]);
}

#[test]
fn lists_texture_fetch_fields_and_pixel_exports() {
    let tf = TextureFetch {
        lod_bias: -8,
        mag_filter: 1,
        ..TextureFetch::fetch_2d(0, 0, 15)
    };
    let out = AluInstr::vector(
        VectorOp::Max,
        0,
        mask("xyzw"),
        &[AluSrc::temp(0, swz("xyzw")); 2],
    )
    .exported(0);
    let words = assemble(
        &[CfInstr::Exec(Exec::new(0, 2, true))],
        &[Instr::Fetch(FetchInstr::Texture(tf)), Instr::Alu(out)],
    );
    let d = disassemble(&words, ShaderStage::Pixel);
    assert!(
        d.text
            .contains("tfetch2d r0.xyzw, r0.xy, tf15 filter mag 1"),
        "{}",
        d.text
    );
    assert!(d.text.contains("lod_bias -0.5"), "{}", d.text);
    assert!(d.text.contains("max oC0.xyzw, r0, r0"), "{}", d.text);
    assert!(d.text.contains("float constants read: none"), "{}", d.text);
    assert_eq!(d.fetches, [FetchInstr::Texture(tf)]);
}

#[test]
fn never_panics_on_garbage() {
    let mut x = 0x1234_5678u32;
    for len in 0..60 {
        let words: Vec<u32> = (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            })
            .collect();
        disassemble(&words, ShaderStage::Pixel);
    }
}

#[test]
fn export_components_in_both_masks_list_as_one() {
    let mut a = AluInstr::vector(
        VectorOp::Mul,
        0,
        mask("xyz"),
        &[AluSrc::temp(1, swz("xyzw")), AluSrc::temp(2, swz("xyzw"))],
    )
    .exported(62);
    a.scalar_op = ScalarOp::Rcp;
    a.scalar_write_mask = mask("zw");
    a.src[2] = AluSrc::temp(3, swz("x"));
    let words = assemble(&[CfInstr::Exec(Exec::new(0, 1, true))], &[Instr::Alu(a)]);
    let d = disassemble(&words, ShaderStage::Vertex);
    assert!(
        d.text
            .contains("mul oPos.xy__, r1, r2 + rcp oPos.___w, r3.xxxx one __z_"),
        "{}",
        d.text
    );
}

#[test]
fn abs_constants_applies_to_both_scalar_constant_op_operands() {
    let mut a = AluInstr::scalar(
        ScalarOp::Mulsc0,
        1,
        mask("w"),
        AluSrc::constant(5, swz("xyzw")),
    );
    a.abs_constants = true;
    let temp = a.scalar_const_op_temp_reg();
    let words = assemble(&[CfInstr::Exec(Exec::new(0, 1, true))], &[Instr::Alu(a)]);
    let d = disassemble(&words, ShaderStage::Vertex);
    let want = format!("mulsc0 r1.___w, |c5|.w, |r{temp}|.x");
    assert!(d.text.contains(&want), "missing {want:?} in\n{}", d.text);
}

#[test]
fn mini_fetch_omits_fields_it_takes_from_the_full_fetch() {
    let full = VertexFetch::full(1, 0, 4, VertexFormat::F32_32_FLOAT, 3, 0);
    // Encoded index register, constant, stride and rounding that differ
    // from the full fetch's: hardware ignores them.
    let mini = VertexFetch {
        mini: true,
        index_rounded: true,
        ..VertexFetch::full(2, 5, 7, VertexFormat::F32_FLOAT, 9, 2)
    };
    let words = assemble(
        &[CfInstr::Exec(Exec::new(0, 2, true))],
        &[
            Instr::Fetch(FetchInstr::Vertex(full)),
            Instr::Fetch(FetchInstr::Vertex(mini)),
        ],
    );
    let t = disassemble(&words, ShaderStage::Vertex).text;
    assert!(
        t.contains(
            "vfetch_mini r2.xyzw format 32_FLOAT(36) unsigned normalized exp_adjust 0 offset 2"
        ),
        "{t}"
    );
    for absent in ["r5", "vf7", "stride 9", "index_rounded"] {
        assert!(!t.contains(absent), "{absent:?} in\n{t}");
    }
}
