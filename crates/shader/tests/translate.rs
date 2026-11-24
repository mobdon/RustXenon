//! Translation tests on hand-assembled programs. Every successful output is
//! parsed and validated with naga.

mod common;

use common::{has, ok, Lcg};
use xenon_shader::asm::*;
use xenon_shader::ucode::*;
use xenon_shader::{
    translate, translate_with, Error, ParamGen, ParamGenPrimitive, ShaderStage,
    PIXEL_FLOAT_CONST_BASE,
};

use ShaderStage::{Pixel, Vertex};

fn exec(addr: u32, count: u32, end: bool) -> CfInstr {
    CfInstr::Exec(Exec::new(addr, count, end))
}

fn alu(a: AluInstr) -> Instr {
    Instr::Alu(a)
}

/// `mov` as the assembler writes it: `max dest, src, src`.
fn mov(dest: u32, m: u32, src: AluSrc) -> AluInstr {
    AluInstr::vector(VectorOp::Max, dest, m, &[src, src])
}

fn export(reg: u32, src: AluSrc) -> AluInstr {
    mov(reg, mask("xyzw"), src).exported(reg)
}

fn vfetch(v: VertexFetch) -> Instr {
    Instr::Fetch(FetchInstr::Vertex(v))
}

fn tfetch(t: TextureFetch) -> Instr {
    Instr::Fetch(FetchInstr::Texture(t))
}

fn r(n: u32) -> AluSrc {
    AluSrc::temp(n, SWZ_XYZW)
}

#[test]
fn passthrough_vertex_shader() {
    // r1 = position (3 floats), r2 = colour (8_8_8_8 unorm) from vf0 with a
    // 4-word stride; oPos = r1, o0 = r2.
    let mut pos = VertexFetch::full(1, 0, 0, VertexFormat::F32_32_32_FLOAT, 4, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let mut col = VertexFetch::full(2, 0, 0, VertexFormat::F8_8_8_8, 4, 3);
    col.mini = true;
    let words = assemble(
        &[exec(0, 4, true)],
        &[
            vfetch(pos),
            vfetch(col),
            alu(export(62, r(1))),
            alu(export(0, r(2))),
        ],
    );
    let t = ok(&words, Vertex);
    assert!(t.info.uses_vertex_fetch);
    assert!(t.info.writes_position);
    assert_eq!(t.info.interpolators_written, 0b1);
    assert_eq!(t.info.vertex_fetch_consts, 1);
    assert_eq!(t.info.textures_used, 0);
    has(
        &t,
        "fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut",
    );
    has(
        &t,
        "r[0].x = f32((vertex_index + consts.vertex_base) & 0xFFFFFFu);",
    );
    has(
        &t,
        "@group(0) @binding(0) var<storage, read> guest: array<u32>;",
    );
    has(&t, "let idx = i32(floor(r[0].x));");
    has(
        &t,
        "vf_addr = i32((consts.fetch[0u][0u] >> 2u) & 0x07FFFFFFu) + idx * 4;",
    );
    has(&t, "let e = consts.fetch[0u][1u] & 3u;");
    has(&t, "r[1].w = 1.0;");
    has(&t, "let w0 = vword(vf_addr + 3, e);");
    has(&t, "f32(extractBits(w0, 24u, 8u)) * 0.003921569f");
    has(&t, "@location(0) o0: vec4<f32>,");
    has(&t, "out.pos = o_pos;");
}

#[test]
fn vertex_fetch_constant_layout() {
    // Fetch constant 7 is dwords 14 and 15: fetch[3][2] and fetch[3][3].
    let words = assemble(
        &[exec(0, 2, true)],
        &[
            vfetch(VertexFetch::full(1, 0, 7, VertexFormat::F32_FLOAT, 1, 0)),
            alu(export(62, r(1))),
        ],
    );
    let t = ok(&words, Vertex);
    has(&t, "consts.fetch[3u][2u] >> 2u");
    has(&t, "consts.fetch[3u][3u] & 3u");
    assert_eq!(t.info.vertex_fetch_consts, 1 << 7);
}

#[test]
fn passthrough_pixel_shader() {
    let words = assemble(&[exec(0, 1, true)], &[alu(export(0, r(0)))]);
    let t = ok(&words, Pixel);
    assert_eq!(t.info.interpolators_read, 0b1);
    assert_eq!(t.info.colour_targets_written, 0b1);
    assert!(!t.info.uses_vertex_fetch);
    has(&t, "diagnostic(off, derivative_uniformity);");
    has(&t, "fn ps_main(input: PsIn) -> PsOut");
    has(&t, "r[0] = input.i0;");
    has(&t, "o_col[0].x = vr.x;");
    has(&t, "out.c0 = o_col[0];");
    assert!(!t.wgsl.contains("var<storage"));
}

/// SQ_PROGRAM_CNTL.param_gen: r1 holds the pixel parameters instead of an
/// interpolator; the sign bits carry facing, point and line.
#[test]
fn pixel_param_gen() {
    let words = assemble(
        &[exec(0, 2, true)],
        &[alu(export(0, r(0))), alu(export(1, r(1)))],
    );
    let plain = ok(&words, Pixel);
    assert_eq!(plain.info.interpolators_read, 0b11);
    assert!(!plain.wgsl.contains("xe_frag_coord"));
    let gen = |primitive| {
        let t = translate_with(
            &words,
            Pixel,
            Some(ParamGen {
                register: 1,
                primitive,
            }),
        )
        .unwrap();
        common::validate(&t.wgsl);
        assert_eq!(t.info.interpolators_read, 0b01);
        has(&t, "@builtin(position) xe_frag_coord: vec4<f32>,");
        has(&t, "let xy = floor(input.xe_frag_coord.xy);");
        assert!(!t.wgsl.contains("r[1] = input.i1;"));
        t
    };
    let t = gen(ParamGenPrimitive::Polygon);
    has(&t, "@builtin(front_facing) xe_front_facing: bool,");
    has(
        &t,
        "r[1] = vec4<f32>(bitcast<f32>(bitcast<u32>(xy.x) | select(0x80000000u, 0u, input.xe_front_facing)), xy.y, 0.0, 0.0);",
    );
    let t = gen(ParamGenPrimitive::Line);
    assert!(!t.wgsl.contains("front_facing"));
    has(
        &t,
        "r[1] = vec4<f32>(xy.x, xy.y, bitcast<f32>(0x80000000u), 0.0);",
    );
    let t = gen(ParamGenPrimitive::Point);
    has(
        &t,
        "r[1] = vec4<f32>(xy.x, bitcast<f32>(bitcast<u32>(xy.y) | 0x80000000u), 0.0, 0.0);",
    );
    let t = gen(ParamGenPrimitive::Other);
    has(&t, "r[1] = vec4<f32>(xy.x, xy.y, 0.0, 0.0);");
    // Registers past the 16 interpolators and the vertex stage get nothing.
    let pg = |register| {
        Some(ParamGen {
            register,
            primitive: ParamGenPrimitive::Polygon,
        })
    };
    let t = translate_with(&words, Pixel, pg(16)).unwrap();
    assert_eq!(t.wgsl, plain.wgsl);
    let v = assemble(&[exec(0, 1, true)], &[alu(export(62, r(0)))]);
    assert_eq!(
        translate_with(&v, Vertex, pg(0)).unwrap().wgsl,
        translate(&v, Vertex).unwrap().wgsl
    );
}

#[test]
fn pixel_shader_texture_fetch() {
    let mut tf = TextureFetch::fetch_2d(1, 0, 5);
    tf.src_swizzle = 0b00_01; // r0.yx
    tf.offset = [1, -2, 0];
    tf.lod_bias = 8;
    tf.mag_filter = 1;
    let words = assemble(&[exec(0, 2, true)], &[tfetch(tf), alu(export(1, r(1)))]);
    let t = ok(&words, Pixel);
    assert_eq!(t.info.textures_used, 1 << 5);
    assert_eq!(t.info.texture_filter_overrides, 1 << 5);
    assert_eq!(t.info.colour_targets_written, 0b10);
    has(&t, "@group(1) @binding(10) var tex5: texture_2d<f32>;");
    has(&t, "@group(1) @binding(11) var samp5: sampler;");
    has(&t, "let uv = r[0].yx + vec2<f32>(0.5f, -1.0f) / size;");
    // Slot 5's dword 4 is fetch dword 34: fetch[8][2].
    has(
        &t,
        "let lod = f32(extractBits(bitcast<i32>(consts.fetch[8u][2u]), 12u, 10u)) / 32.0 + 0.5f;",
    );
    has(&t, "textureSampleBias(tex5, samp5, uv, lod)");
    has(&t, "@location(1) c1: vec4<f32>,");
}

#[test]
fn texture_fetch_variants() {
    // Unnormalized coordinates, vertex stage (explicit LOD), predicated,
    // partial destination swizzle.
    let mut tf = TextureFetch::fetch_2d(2, 3, 0);
    tf.unnormalized = true;
    tf.dest_swizzle = fetch_dest_swz("x_01");
    tf.predicate = Some(true);
    let words = assemble(&[exec(0, 2, true)], &[tfetch(tf), alu(export(62, r(2)))]);
    let t = ok(&words, Vertex);
    has(&t, "let uv = (r[3].xy) / size;");
    has(&t, "textureSampleLevel(tex0, samp0, uv, lod)");
    has(&t, "r[2].z = 0.0;");
    has(&t, "r[2].w = 1.0;");
    assert!(!t.wgsl.contains("r[2].y ="));
    // Pixel stage without computed LOD also uses an explicit level.
    let mut tf = TextureFetch::fetch_2d(0, 0, 1);
    tf.use_computed_lod = false;
    let words = assemble(&[exec(0, 2, true)], &[tfetch(tf), alu(export(0, r(0)))]);
    let t = ok(&words, Pixel);
    has(&t, "textureSampleLevel(tex1, samp1, uv, lod)");
    // The implicit-LOD case still applies the fetch constant's bias.
    let words = assemble(
        &[exec(0, 2, true)],
        &[
            tfetch(TextureFetch::fetch_2d(0, 0, 1)),
            alu(export(0, r(0))),
        ],
    );
    has(
        &ok(&words, Pixel),
        "textureSampleBias(tex1, samp1, uv, lod)",
    );
}

#[test]
fn float_constants() {
    // VS: r1 = c3 + c[5 + a0]; oPos = r1 * c[2 + aL].
    let mut add = AluInstr::vector(
        VectorOp::Add,
        1,
        mask("xyzw"),
        &[AluSrc::constant(3, SWZ_XYZW), AluSrc::constant(5, SWZ_XYZW)],
    );
    // src1 is a constant, so src2 uses const1_relative.
    add.const1_relative = true;
    add.const_relative_a0 = true;
    let mut mul = AluInstr::vector(
        VectorOp::Mul,
        62,
        mask("xyzw"),
        &[r(1), AluSrc::constant(2, swz("wzyx"))],
    )
    .exported(62);
    // src1 is a temporary, so src2 is the first constant: const0_relative.
    mul.const0_relative = true;
    let words = assemble(&[exec(0, 2, true)], &[alu(add), alu(mul)]);
    let t = ok(&words, Vertex);
    has(&t, "let v0 = consts.float_consts[3];");
    has(&t, "let v1 = cnst(5 + a0);");
    has(&t, "let v1 = cnst(2 + al()).wzyx;");
    has(&t, "return consts.float_consts[u32(i) + 0u];");
    assert_eq!(t.info.float_consts_used, Some((3, 3)));
    assert!(t.info.float_consts_dynamic);

    // PS constants live at 256 and up.
    let mut a = mov(0, mask("xyzw"), AluSrc::constant(7, SWZ_XYZW)).exported(0);
    a.abs_constants = true;
    let words = assemble(&[exec(0, 1, true)], &[alu(a)]);
    let t = ok(&words, Pixel);
    has(
        &t,
        &format!("abs(consts.float_consts[{}])", 7 + PIXEL_FLOAT_CONST_BASE),
    );
    assert_eq!(t.info.interpolators_read, 0);
    has(&t, "fn ps_main() -> PsOut");
}

#[test]
fn source_modifiers_and_saturate() {
    let mut a = AluInstr::vector(
        VectorOp::Add,
        3,
        mask("xz"),
        &[r(1).abs().negated(), AluSrc::temp(2, swz("yyyy")).rel()],
    );
    a.vector_saturate = true;
    a.scalar_op = ScalarOp::Adds;
    a.scalar_dest = 4;
    a.scalar_write_mask = mask("w");
    a.scalar_saturate = true;
    a.src[2] = AluSrc::temp(5, swz("yzwx")).negated();
    let words = assemble(&[exec(0, 2, true)], &[alu(a), alu(export(62, r(3)))]);
    let t = ok(&words, Vertex);
    has(&t, "let v0 = (-abs(r[1]));");
    has(&t, "let v1 = r[u32(2 + al()) & 63u].yyyy;");
    has(&t, "let vr = saturate(v0 + v1);");
    has(&t, "r[3].x = vr.x;");
    has(&t, "r[3].z = vr.z;");
    assert!(!t.wgsl.contains("r[3].y = vr.y;"));
    // Scalar `a` is the W swizzle slot (x here), `b` the X slot (y).
    has(&t, "let sa = (-r[5].x);");
    has(&t, "let sb = (-r[5].y);");
    has(&t, "r[4].w = saturate(ps);");
}

#[test]
fn export_constant_writes() {
    // Vector writes xy, scalar writes yz: y becomes 1.0 (both), w becomes
    // 0.0 when scalar_dest_relative is set on an export.
    let mut a = AluInstr::vector(VectorOp::Max, 0, mask("xy"), &[r(1), r(1)]);
    a.scalar_op = ScalarOp::Maxs;
    a.scalar_write_mask = mask("yz");
    a.src[2] = r(2);
    a = a.exported(62);
    a.scalar_dest_relative = true;
    let words = assemble(&[exec(0, 1, true)], &[alu(a)]);
    let t = ok(&words, Vertex);
    has(&t, "o_pos.x = vr.x;");
    has(&t, "o_pos.z = ps;");
    has(&t, "o_pos.y = 1.0;");
    has(&t, "o_pos.w = 0.0;");
}

#[test]
fn predicated_exec_and_instructions() {
    // setp_gt sets p0; a predicated exec runs; a predicated ALU inside.
    let setp = AluInstr::scalar(ScalarOp::SetpGt, 0, 0, r(0));
    let cf = [
        exec(0, 1, false),
        CfInstr::Exec(Exec::new(1, 2, false).with_cond(ExecCond::Predicate {
            value: true,
            predicate_clean: false,
        })),
        exec(3, 1, true),
    ];
    let words = assemble(
        &cf,
        &[
            alu(setp),
            alu(mov(1, mask("xyzw"), r(0)).predicated(true)),
            alu(mov(1, mask("x"), r(2)).predicated(false)),
            alu(export(0, r(1))),
        ],
    );
    let t = ok(&words, Pixel);
    has(&t, "p0 = sa > 0.0;");
    has(&t, "if (p0 == true) {");
    has(&t, "if (p0 == false) {");
    assert_eq!(t.info.interpolators_read, 0b111);
}

#[test]
fn loop_with_loop_constant_and_relative_registers() {
    // loop i3 { r[aL+4] = r[aL+4] + c0 } then export r4.
    let mut body = AluInstr::vector(
        VectorOp::Add,
        4,
        mask("xyzw"),
        &[
            AluSrc::temp(4, SWZ_XYZW).rel(),
            AluSrc::constant(0, SWZ_XYZW),
        ],
    );
    body.vector_dest_relative = true;
    let cf = [
        CfInstr::LoopStart {
            address: 3,
            repeat: false,
            loop_id: 3,
        },
        exec(0, 1, false),
        CfInstr::LoopEnd {
            address: 1,
            loop_id: 3,
            predicated_break: Some(true),
        },
        exec(1, 1, true),
    ];
    let words = assemble(&cf, &[alu(body), alu(export(62, r(4)))]);
    let t = ok(&words, Vertex);
    has(&t, "let lc = consts.loop_consts[0u][3u];");
    has(&t, "let start = i32((lc >> 8u) & 0xFFu);");
    has(&t, "if (loop_count[d] != 0u && p0 != true) {");
    has(
        &t,
        "loop_al[d] += extractBits(bitcast<i32>(consts.loop_consts[0u][3u]), 16u, 8u);",
    );
    has(&t, "r[u32(4 + al()) & 63u].x = vr.x;");
    has(&t, "steps >= 65536u");
    // A repeat loop keeps the enclosing aL.
    let cf = [
        CfInstr::LoopStart {
            address: 3,
            repeat: true,
            loop_id: 9,
        },
        exec(0, 1, false),
        CfInstr::LoopEnd {
            address: 1,
            loop_id: 9,
            predicated_break: None,
        },
        exec(1, 1, true),
    ];
    let words = assemble(&cf, &[alu(body), alu(export(62, r(4)))]);
    let t = ok(&words, Vertex);
    has(&t, "let start = al();");
    has(&t, "consts.loop_consts[2u][1u]");
}

#[test]
fn conditional_jump_call_and_return() {
    let cf = [
        // 0: if b37 is false skip to 3.
        CfInstr::CondJmp {
            address: 3,
            cond: BranchCond::Bool {
                index: 37,
                value: false,
            },
            direction: false,
        },
        // 1: call the function at 5.
        CfInstr::CondCall {
            address: 5,
            cond: BranchCond::Always,
        },
        // 2: predicated call.
        CfInstr::CondCall {
            address: 5,
            cond: BranchCond::Predicate(false),
        },
        // 3: export and end.
        exec(1, 1, true),
        CfInstr::Nop,
        // 5: function body, then return.
        exec(0, 1, false),
        CfInstr::Return,
    ];
    let words = assemble(
        &cf,
        &[
            alu(AluInstr::vector(
                VectorOp::Add,
                1,
                mask("xyzw"),
                &[r(1), r(0)],
            )),
            alu(export(0, r(1))),
        ],
    );
    let t = ok(&words, Pixel);
    // b37 is bit 5 of word 1: bool_consts[0][1].
    has(
        &t,
        "if ((((consts.bool_consts[0u][1u] >> 5u) & 1u) != 0u) == false) {",
    );
    has(&t, "if (true && call_depth < 4u) {");
    has(&t, "if ((p0 == false) && call_depth < 4u) {");
    has(&t, "call_ret[call_depth] = 2u;");
    has(&t, "pc = call_ret[call_depth];");
}

#[test]
fn conditional_execs_on_bool_constants() {
    let cf = [
        CfInstr::Exec(Exec::new(0, 1, false).with_cond(ExecCond::Bool {
            index: 200,
            value: true,
            predicate_clean: false,
        })),
        CfInstr::Exec(Exec::new(1, 1, true).with_cond(ExecCond::Bool {
            index: 200,
            value: false,
            predicate_clean: true,
        })),
        exec(0, 1, true),
    ];
    let words = assemble(&cf, &[alu(export(0, r(0))), alu(export(0, r(1)))]);
    let t = ok(&words, Pixel);
    // b200: word 6 -> bool_consts[1][2], bit 8.
    has(
        &t,
        "if ((((consts.bool_consts[1u][2u] >> 8u) & 1u) != 0u) == true) {",
    );
    has(
        &t,
        "if ((((consts.bool_consts[1u][2u] >> 8u) & 1u) != 0u) == false) {",
    );
}

fn stage_for_vector(op: VectorOp) -> ShaderStage {
    match op {
        VectorOp::KillEq | VectorOp::KillGt | VectorOp::KillGe | VectorOp::KillNe => Pixel,
        _ => Vertex,
    }
}

#[test]
fn every_vector_opcode() {
    for &op in VectorOp::ALL {
        let a = AluInstr::vector(
            op,
            5,
            mask("xyzw"),
            &[
                AluSrc::temp(1, swz("zxyw")),
                r(2),
                AluSrc::constant(4, swz("x")),
            ],
        );
        let stage = stage_for_vector(op);
        let out = if stage == Pixel { 0 } else { 62 };
        let words = assemble(&[exec(0, 2, true)], &[alu(a), alu(export(out, r(5)))]);
        let t = ok(&words, stage);
        has(&t, "r[5].w = vr.w;");
        let expect = match op {
            VectorOp::Add => "let vr = v0 + v1;",
            VectorOp::Mul => "let vr = mulz(v0, v1);",
            VectorOp::Max => "let vr = vmax(v0, v1);",
            VectorOp::Min => "let vr = vmin(v0, v1);",
            VectorOp::Seq => "v0 == v1",
            VectorOp::Sgt => "v0 > v1",
            VectorOp::Sge => "v0 >= v1",
            VectorOp::Sne => "v0 != v1",
            VectorOp::Frc => "fract(v0)",
            VectorOp::Trunc => "trunc(v0)",
            VectorOp::Floor => "floor(v0)",
            VectorOp::Mad => "mulz(v0, v1) + v2",
            VectorOp::CndEq => "select(v2, v1, v0 == vec4<f32>(0.0))",
            VectorOp::CndGe => "select(v2, v1, v0 >= vec4<f32>(0.0))",
            VectorOp::CndGt => "select(v2, v1, v0 > vec4<f32>(0.0))",
            VectorOp::Dp4 => "dot_p.x + dot_p.y + dot_p.z + dot_p.w",
            VectorOp::Dp3 => "vec4<f32>(dot_p.x + dot_p.y + dot_p.z)",
            VectorOp::Dp2Add => "dot_p.x + dot_p.y + v2.x",
            VectorOp::Cube => "cube(v0)",
            VectorOp::Max4 => "max(max(max(v0.x, v0.y), v0.z), v0.w)",
            VectorOp::SetpEqPush => "p0 = v0.w == 0.0 && v1.w == 0.0;",
            VectorOp::SetpNePush => "p0 = v0.w == 0.0 && v1.w != 0.0;",
            VectorOp::SetpGtPush => "p0 = v0.w == 0.0 && v1.w > 0.0;",
            VectorOp::SetpGePush => "p0 = v0.w == 0.0 && v1.w >= 0.0;",
            VectorOp::KillEq => "if (any(v0 == v1)) { discard; }",
            VectorOp::KillGt => "if (any(v0 > v1)) { discard; }",
            VectorOp::KillGe => "if (any(v0 >= v1)) { discard; }",
            VectorOp::KillNe => "if (any(v0 != v1)) { discard; }",
            VectorOp::Dst => "vec4<f32>(1.0, mulzs(v0.y, v1.y), v0.z, v1.w)",
            VectorOp::MaxA => "a0 = i32(clamp(floor(v0.w + 0.5), -256.0, 255.0));",
        };
        has(&t, expect);
        has(&t, "let v0 = r[1].zxyw;");
        if stage == Pixel {
            assert!(t.info.kills);
        }
    }
}

#[test]
fn every_scalar_opcode() {
    for &op in ScalarOp::ALL {
        let stage = match op {
            ScalarOp::KillsEq
            | ScalarOp::KillsGt
            | ScalarOp::KillsGe
            | ScalarOp::KillsNe
            | ScalarOp::KillsOne => Pixel,
            _ => Vertex,
        };
        let src = if op.operand_count() == 2 {
            AluSrc::constant(9, swz("xyzw"))
        } else {
            AluSrc::temp(3, swz("wzyx"))
        };
        let a = AluInstr::scalar(op, 6, mask("xyzw"), src);
        let out = if stage == Pixel { 0 } else { 62 };
        let words = assemble(&[exec(0, 2, true)], &[alu(a), alu(export(out, r(6)))]);
        let t = ok(&words, stage);
        has(&t, "r[6].y = ps;");
        let expect = match op {
            ScalarOp::Adds => "sa + sb",
            ScalarOp::AddsPrev => "sa + ps",
            ScalarOp::Muls => "mulzs(sa, sb)",
            ScalarOp::MulsPrev => "mulzs(sa, ps)",
            ScalarOp::MulsPrev2 => "mulzs(sa, ps), ps != -",
            ScalarOp::Maxs => "smax(sa, sb)",
            ScalarOp::Mins => "smin(sa, sb)",
            ScalarOp::Seqs => "select(0.0, 1.0, sa == 0.0)",
            ScalarOp::Sgts => "select(0.0, 1.0, sa > 0.0)",
            ScalarOp::Sges => "select(0.0, 1.0, sa >= 0.0)",
            ScalarOp::Snes => "select(0.0, 1.0, sa != 0.0)",
            ScalarOp::Frcs => "fract(sa)",
            ScalarOp::Truncs => "trunc(sa)",
            ScalarOp::Floors => "floor(sa)",
            ScalarOp::Exp => "exp2(sa)",
            ScalarOp::Logc => "clamp_neg_inf(log2(sa))",
            ScalarOp::Log => "= log2(sa)",
            ScalarOp::Rcpc => "clamp_inf(1.0 / sa)",
            ScalarOp::Rcpf => "zero_inf(1.0 / sa)",
            ScalarOp::Rcp => "= 1.0 / sa",
            ScalarOp::Rsqc => "clamp_inf(inverseSqrt(sa))",
            ScalarOp::Rsqf => "zero_inf(inverseSqrt(sa))",
            ScalarOp::Rsq => "= inverseSqrt(sa)",
            ScalarOp::MaxAs => "a0 = i32(clamp(floor(sa + 0.5), -256.0, 255.0));",
            ScalarOp::MaxAsf => "a0 = i32(clamp(floor(sa), -256.0, 255.0));",
            ScalarOp::Subs => "sa - sb",
            ScalarOp::SubsPrev => "sa - ps",
            ScalarOp::SetpEq => "p0 = sa == 0.0;",
            ScalarOp::SetpNe => "p0 = sa != 0.0;",
            ScalarOp::SetpGt => "p0 = sa > 0.0;",
            ScalarOp::SetpGe => "p0 = sa >= 0.0;",
            ScalarOp::SetpInv => "p0 = sa == 1.0;",
            ScalarOp::SetpPop => "p0 = pop <= 0.0;",
            ScalarOp::SetpClr => "p0 = false;",
            ScalarOp::SetpRstr => "select(sa, 0.0, p0)",
            ScalarOp::KillsEq => "if (sa == 0.0) { discard; }",
            ScalarOp::KillsGt => "if (sa > 0.0) { discard; }",
            ScalarOp::KillsGe => "if (sa >= 0.0) { discard; }",
            ScalarOp::KillsNe => "if (sa != 0.0) { discard; }",
            ScalarOp::KillsOne => "if (sa == 1.0) { discard; }",
            ScalarOp::Sqrt => "sqrt(sa)",
            ScalarOp::Mulsc0 | ScalarOp::Mulsc1 => "mulzs(sa, sb)",
            ScalarOp::Addsc0 | ScalarOp::Addsc1 => "sa + sb",
            ScalarOp::Subsc0 | ScalarOp::Subsc1 => "sa - sb",
            ScalarOp::Sin => "sin(sa)",
            ScalarOp::Cos => "cos(sa)",
            ScalarOp::RetainPrev => "r[6].x = ps;",
        };
        has(&t, expect);
        match op.operand_count() {
            1 => {
                has(&t, "let sa = r[3].x;");
                has(&t, "let sb = r[3].w;");
            }
            2 => {
                // Constant `a` from c9.w, temporary `b` from the register
                // number packed into the opcode and swizzle bits.
                let reg = (op as u32 & 1) | (swz("xyzw") & 0x3C);
                has(&t, "let sa = consts.float_consts[9].w;");
                has(&t, &format!("let sb = r[{reg}].x;"));
            }
            _ => {}
        }
    }
}

#[test]
fn scalar_constant_ops_pick_the_temp_register() {
    // muls_c1 with src3 select = temp: register = 1 | 2 | (swizzle & 0x3C).
    let mut a = AluInstr::scalar(
        ScalarOp::Mulsc1,
        1,
        mask("x"),
        AluSrc::constant(4, 0b0011_0100),
    );
    a.src[2].is_temp = true;
    assert_eq!(a.scalar_const_op_temp_reg(), 1 | 2 | 0x34);
    let words = assemble(&[exec(0, 2, true)], &[alu(a), alu(export(62, r(1)))]);
    let t = ok(&words, Vertex);
    has(&t, "let sb = r[55]");
}

fn fetch_one(f: VertexFetch) -> Result<xenon_shader::Translated, Error> {
    let words = assemble(&[exec(0, 2, true)], &[vfetch(f), alu(export(62, r(1)))]);
    translate(&words, Vertex)
}

#[test]
fn every_vertex_format() {
    let formats = [
        (VertexFormat::F8_8_8_8, 1),
        (VertexFormat::F2_10_10_10, 1),
        (VertexFormat::F10_11_11, 1),
        (VertexFormat::F11_11_10, 1),
        (VertexFormat::F16_16, 1),
        (VertexFormat::F16_16_16_16, 2),
        (VertexFormat::F16_16_FLOAT, 1),
        (VertexFormat::F16_16_16_16_FLOAT, 2),
        (VertexFormat::F32, 1),
        (VertexFormat::F32_32, 2),
        (VertexFormat::F32_32_32_32, 4),
        (VertexFormat::F32_FLOAT, 1),
        (VertexFormat::F32_32_FLOAT, 2),
        (VertexFormat::F32_32_32_32_FLOAT, 4),
        (VertexFormat::F32_32_32_FLOAT, 3),
    ];
    for (format, words) in formats {
        for flags in 0..8u32 {
            let mut f = VertexFetch::full(1, 2, 4, format, 7, -2);
            f.signed = flags & 1 != 0;
            f.normalized = flags & 2 != 0;
            f.signed_rf_no_zero = flags & 4 != 0;
            f.exp_adjust = if flags == 3 { -4 } else { 0 };
            f.index_rounded = flags == 5;
            let t = fetch_one(f).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            common::validate(&t.wgsl);
            has(
                &t,
                &format!("let w{} = vword(vf_addr + {}, e);", words - 1, words - 3),
            );
            assert!(!t.wgsl.contains(&format!("let w{words} ")));
            if flags == 3 {
                has(&t, "* 0.0625f");
            }
            if flags == 5 {
                has(&t, "floor(r[2].x + 0.5)");
            }
        }
    }
    // Spot checks of the conversions.
    let mut f = VertexFetch::full(1, 0, 0, VertexFormat::F2_10_10_10, 1, 0);
    f.signed = true;
    has(
        &fetch_one(f).unwrap(),
        "max(f32(extractBits(bitcast<i32>(w0), 30u, 2u)) * 1.0f, -1.0)",
    );
    f.signed_rf_no_zero = true;
    has(
        &fetch_one(f).unwrap(),
        "f32(extractBits(bitcast<i32>(w0), 10u, 10u)) * 0.0019550342f + 0.0009775171f",
    );
    let f = VertexFetch::full(1, 0, 0, VertexFormat::F16_16_16_16_FLOAT, 1, 0);
    has(&fetch_one(f).unwrap(), "half(extractBits(w1, 16u, 16u))");
    let mut f = VertexFetch::full(1, 0, 0, VertexFormat::F32_32, 1, 0);
    f.normalized = false;
    f.signed = true;
    has(
        &fetch_one(f).unwrap(),
        "vec4<f32>(f32(bitcast<i32>(w0)), f32(bitcast<i32>(w1)), 0.0, 0.0)",
    );
    // Stride 0 does not read the index register.
    let f = VertexFetch::full(1, 0, 0, VertexFormat::F32_FLOAT, 0, 0);
    let t = fetch_one(f).unwrap();
    assert!(!t.wgsl.contains("let idx"));
    has(
        &t,
        "vf_addr = i32((consts.fetch[0u][0u] >> 2u) & 0x07FFFFFFu);",
    );
}

#[test]
fn unsupported_vertex_format_is_an_error() {
    let f = VertexFetch::full(1, 0, 0, VertexFormat(12), 1, 0);
    assert!(matches!(fetch_one(f), Err(Error::Unsupported(_))));
}

#[test]
fn vfetch_mini_without_full_is_an_error() {
    let mut f = VertexFetch::full(1, 0, 0, VertexFormat::F32_FLOAT, 1, 0);
    f.mini = true;
    assert!(matches!(fetch_one(f), Err(Error::Malformed(_))));
}

#[test]
fn unsupported_features_are_errors() {
    let unsupported = |words: Vec<u32>, stage| match translate(&words, stage) {
        Err(Error::Unsupported(m)) => m,
        other => panic!("expected Unsupported, got {other:?}"),
    };
    // Depth export.
    let m = unsupported(
        assemble(&[exec(0, 1, true)], &[alu(export(61, r(0)))]),
        Pixel,
    );
    assert!(m.contains("depth"), "{m}");
    // Memory export.
    let m = unsupported(
        assemble(&[exec(0, 1, true)], &[alu(export(32, r(0)))]),
        Vertex,
    );
    assert!(m.contains("memory export"), "{m}");
    // Cube and 3D fetches, other texture ops, register LOD.
    for dim in [TextureDim::D1, TextureDim::D3OrStacked, TextureDim::Cube] {
        let mut tf = TextureFetch::fetch_2d(0, 0, 0);
        tf.dim = dim;
        unsupported(assemble(&[exec(0, 1, true)], &[tfetch(tf)]), Pixel);
    }
    let mut tf = TextureFetch::fetch_2d(0, 0, 0);
    tf.op = TextureOp::GetGradients;
    unsupported(assemble(&[exec(0, 1, true)], &[tfetch(tf)]), Pixel);
    let mut tf = TextureFetch::fetch_2d(0, 0, 0);
    tf.use_register_lod = true;
    unsupported(assemble(&[exec(0, 1, true)], &[tfetch(tf)]), Pixel);
    // Kill in a vertex shader.
    let a = AluInstr::scalar(ScalarOp::KillsEq, 0, 0, r(0));
    unsupported(assemble(&[exec(0, 1, true)], &[alu(a)]), Vertex);
}

#[test]
fn point_size_export_is_reported() {
    let words = assemble(
        &[exec(0, 2, true)],
        &[alu(export(63, r(1))), alu(export(62, r(0)))],
    );
    let t = ok(&words, Vertex);
    assert!(t.info.point_size_export_ignored);
    assert!(t.info.writes_position);
}

#[test]
fn malformed_programs_are_errors() {
    let malformed = |words: &[u32]| {
        assert!(
            matches!(translate(words, Pixel), Err(Error::Malformed(_))),
            "{words:08x?}"
        )
    };
    malformed(&[]);
    malformed(&[0, 0]);
    // Exec past the end of the program.
    let mut words = assemble(&[exec(0, 1, true)], &[alu(export(0, r(0)))]);
    words.truncate(3);
    malformed(&words);
    // Invalid vector opcode inside an exec.
    let mut words = assemble(&[exec(0, 1, true)], &[alu(export(0, r(0)))]);
    words[5] |= 31 << 24;
    malformed(&words);
    // Jump past the end.
    let words = assemble(
        &[
            CfInstr::CondJmp {
                address: 40,
                cond: BranchCond::Always,
                direction: false,
            },
            exec(0, 1, true),
        ],
        &[alu(export(0, r(0)))],
    );
    malformed(&words);
    // Invalid export register.
    malformed(&assemble(&[exec(0, 1, true)], &[alu(export(20, r(0)))]));
}

/// Random programs: translation never panics, and every program it accepts
/// is valid WGSL.
#[test]
fn random_programs_never_panic() {
    let mut g = Lcg(99);
    let mut accepted = 0;
    for i in 0..400 {
        let len = 3 * (1 + g.next() as usize % 12);
        let mut words: Vec<u32> = (0..len).map(|_| g.next()).collect();
        // Make most programs at least start with a plausible exec so the
        // translator gets past the control-flow scan.
        if i % 2 == 0 {
            let n = (len / 3) as u32;
            let (lo, hi) = encode_cf(&CfInstr::Exec(Exec {
                sequence: g.next() & 0xFFF,
                ..Exec::new(1, (n - 1).min(6), g.next() & 1 != 0)
            }));
            words[0] = lo;
            words[1] = (words[1] & 0xFFFF_0000) | hi;
        }
        let stage = if g.next() & 1 != 0 { Pixel } else { Vertex };
        if let Ok(t) = translate(&words, stage) {
            common::validate(&t.wgsl);
            accepted += 1;
        }
    }
    assert!(accepted > 0, "fuzzing never produced a valid program");
}

/// Builds well-formed random ALU programs (every opcode mix, export, modifiers)
/// and checks they all validate.
#[test]
fn random_alu_programs_validate() {
    let mut g = Lcg(7);
    for _ in 0..300 {
        let stage = if g.next() & 1 != 0 { Pixel } else { Vertex };
        let mut instrs = Vec::new();
        for _ in 0..6 {
            let vops = VectorOp::ALL;
            let sops = ScalarOp::ALL;
            let mut a = AluInstr::nop();
            a.vector_op = vops[g.next() as usize % vops.len()];
            a.scalar_op = sops[g.next() as usize % sops.len()];
            let is_kill = |a: &AluInstr| {
                matches!(
                    a.vector_op,
                    VectorOp::KillEq | VectorOp::KillGt | VectorOp::KillGe | VectorOp::KillNe
                ) || matches!(
                    a.scalar_op,
                    ScalarOp::KillsEq
                        | ScalarOp::KillsGt
                        | ScalarOp::KillsGe
                        | ScalarOp::KillsNe
                        | ScalarOp::KillsOne
                )
            };
            if stage == Vertex && is_kill(&a) {
                continue;
            }
            for s in &mut a.src {
                *s = AluSrc {
                    reg: g.next() & 0xFF,
                    is_temp: g.next() & 1 != 0,
                    swizzle: g.next() & 0xFF,
                    negate: g.next() & 1 != 0,
                };
            }
            a.vector_dest = g.next() & 63;
            a.scalar_dest = g.next() & 63;
            a.vector_dest_relative = g.next() & 1 != 0;
            a.scalar_dest_relative = g.next() & 1 != 0;
            a.vector_write_mask = g.next() & 15;
            a.scalar_write_mask = g.next() & 15;
            a.vector_saturate = g.next() & 1 != 0;
            a.scalar_saturate = g.next() & 1 != 0;
            a.abs_constants = g.next() & 1 != 0;
            a.const0_relative = g.next() & 1 != 0;
            a.const1_relative = g.next() & 1 != 0;
            a.const_relative_a0 = g.next() & 1 != 0;
            a.predicate = if g.next() & 1 != 0 {
                Some(g.next() & 1 != 0)
            } else {
                None
            };
            if g.next() % 3 == 0 {
                let reg = match stage {
                    Vertex => [0, 5, 15, 62, 63][g.next() as usize % 5],
                    Pixel => g.next() % 4,
                };
                a = a.exported(reg);
            }
            instrs.push(alu(a));
        }
        if instrs.is_empty() {
            continue;
        }
        let words = assemble(&[exec(0, instrs.len() as u32, true)], &instrs);
        ok(&words, stage);
    }
}
