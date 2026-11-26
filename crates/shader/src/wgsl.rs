// ALU, control-flow and fetch semantics ported from Xenia
// src/xenia/gpu/spirv_shader_translator.cc, spirv_shader_translator_alu.cc,
// spirv_shader_translator_fetch.cc, shader_translator.cc and
// shader_interpreter.cc (BSD 3-Clause, see CREDITS.md), cross-checked with
// XenosRecomp shader_recompiler.cpp (MIT), whose pc-switch layout for
// non-structured control flow this follows. Pixel parameter generation
// follows Xenia's PsParamGen in spirv_shader_translator.cc and
// dxbc_shader_translator.cc, with the register layout documented at
// SQ_CONTEXT_MISC::param_gen_pos in registers.h.

//! Microcode to WGSL translation. See the crate docs for the binding contract.

use crate::ucode::*;
use std::fmt::{self, Write as _};

/// Which programmable stage a program is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Pixel,
}

/// Pixel parameter generation (SQ_PROGRAM_CNTL.param_gen): the pixel
/// program finds the pixel's parameters in interpolator `register` instead
/// of a vertex shader output.
///
/// - |x|, |y|: the pixel position in render target pixels, rounded down
///   (0, 1, 2, ..., no half-pixel offset), as the Direct3D 9 VPOS.
/// - sign of x: set for a back-facing polygon.
/// - sign of y: set for a point.
/// - z: -0.0 for a line, else 0.0; w: 0.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamGen {
    /// SQ_CONTEXT_MISC.param_gen_pos; 16 and above write nothing.
    pub register: u32,
    pub primitive: ParamGenPrimitive,
}

/// The primitive class the sign bits of [`ParamGen`] report (Xenia's
/// IsPrimitivePolygonal and IsPrimitiveLine).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamGenPrimitive {
    /// Triangles, quads and fans: the facing goes into the sign of x.
    Polygon,
    Line,
    Point,
    /// Rectangle lists: no flag set.
    Other,
}

/// Translation failure. Nothing is ever silently mistranslated: anything not
/// handled is reported here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The words are not a well-formed program.
    Malformed(String),
    /// A well-formed feature this translator does not handle yet.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed shader: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported shader feature: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        Error::Malformed(e.0)
    }
}

/// What a translated program uses, for building pipelines and bind groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderInfo {
    /// Bit n: texture fetch slot n is sampled (bindings 2n and 2n+1 of
    /// group 1 are declared).
    pub textures_used: u32,
    /// Bit n: slot n's fetch overrides a filter of the fetch constant. The
    /// override is ignored; the drawer's sampler decides the filtering.
    pub texture_filter_overrides: u32,
    /// Bit n: some fetch of slot n picks its LOD itself (a fixed or register
    /// LOD, or register gradients) instead of the hardware-computed one.
    pub texture_explicit_lod: u32,
    /// Bit n: some fetch of slot n adds a non-zero instruction LOD bias.
    pub texture_lod_bias: u32,
    /// Vertex stage: bit i means `@location(i)` is an output.
    pub interpolators_written: u16,
    /// Pixel stage: bit i means `@location(i)` is an input (copied to r<i>).
    /// A superset: every r0..r15 the program reads anywhere.
    pub interpolators_read: u16,
    /// Pixel stage: bit i means `@location(i)` colour output is declared.
    pub colour_targets_written: u8,
    /// Vertex stage: the program exports a position.
    pub writes_position: bool,
    /// Lowest and highest float constant read with an absolute index, in the
    /// stage's own numbering (0..=255; add [`PIXEL_FLOAT_CONST_BASE`] for the
    /// pixel stage's float constant registers).
    pub float_consts_used: Option<(u32, u32)>,
    /// Some float constant is read relative to a0 or aL, so any of the
    /// stage's 256 may be read.
    pub float_consts_dynamic: bool,
    /// The program reads guest memory (binding 0 is declared).
    pub uses_vertex_fetch: bool,
    /// Bit n: vertex fetch constant n (0..=95) is read.
    pub vertex_fetch_consts: u128,
    /// Where the vertex fetches read, one entry per fetch constant, stride
    /// and index kind, so the drawer can upload only those words.
    pub vertex_fetch_reach: Vec<VertexFetchReach>,
    /// The program writes the point size / edge flag / kill-vertex export,
    /// which is dropped.
    pub point_size_export_ignored: bool,
    /// The pixel program can discard.
    pub kills: bool,
}

impl ShaderInfo {
    /// The float constants the stage's `fc` binding holds, as (first, count)
    /// in the stage's own numbering: the span read with an absolute index,
    /// or all 256 when any is read by a dynamic one. `None`: the stage reads
    /// none and declares no `fc`.
    pub fn float_consts_span(&self) -> Option<(u32, u32)> {
        match (self.float_consts_dynamic, self.float_consts_used) {
            (true, _) => Some((0, 256)),
            (false, Some((lo, hi))) => Some((lo, hi - lo + 1)),
            (false, None) => None,
        }
    }
}

/// The guest words some vertex fetches of one fetch constant read: words
/// `[first, end)` past the constant's dword address plus `index * stride`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFetchReach {
    pub fetch_const: u32,
    /// Words per index; 0 means every invocation reads the same words.
    pub stride: u32,
    /// The index is the vertex index the vertex shader starts with in r0.x
    /// (so within the draw's index range), or unused (stride 0). False: the
    /// fetches may read anywhere in the constant's buffer.
    pub bounded: bool,
    pub first: i32,
    pub end: i32,
}

/// Output of [`translate`].
#[derive(Clone, Debug)]
pub struct Translated {
    pub wgsl: String,
    pub info: ShaderInfo,
}

/// `@group(0)` binding of the vertex stage's float constants.
pub const VERTEX_FLOAT_CONSTS_BINDING: u32 = 4;
/// `@group(0)` binding of the pixel stage's float constants.
pub const PIXEL_FLOAT_CONSTS_BINDING: u32 = 5;

/// Number of the first pixel-shader float constant among the 512 float
/// constant registers (the vertex shader's come first).
/// The console's D3D runtime sets SQ_VS_CONST to base 0 and SQ_PS_CONST to
/// base 256, 256 constants each (Xenia asserts this split).
pub const PIXEL_FLOAT_CONST_BASE: u32 = 256;

/// Control-flow steps a single invocation may take before the program is cut
/// short, so a broken loop cannot hang the GPU.
pub const WATCHDOG_STEPS: u32 = 65536;

/// Number of temporary registers addressable by a program (6-bit fields).
pub const TEMP_REGISTERS: u32 = 64;

const FLT_MAX: &str = "3.40282346638528859812e+38f";

const PRELUDE: &str = "struct Consts {
    fetch: array<vec4<u32>, 48>,
    bool_consts: array<vec4<u32>, 2>,
    loop_consts: array<vec4<u32>, 8>,
    window_base: u32,
    vertex_base: u32,
    _pad0: u32,
    _pad1: u32,
    vf_rebase: array<vec4<i32>, 24>,
}

@group(0) @binding(1) var<uniform> consts: Consts;

var<private> r: array<vec4<f32>, 64>;
var<private> a0: i32;
var<private> p0: bool;
var<private> ps: f32;
var<private> loop_depth: u32;
var<private> loop_count: array<u32, 4>;
var<private> loop_al: array<i32, 4>;
var<private> call_depth: u32;
var<private> call_ret: array<u32, 4>;
var<private> vf_addr: i32;
// Register LOD and gradients set by setTexLOD / setGradientH / setGradientV.
var<private> tf_lod: f32;
var<private> tf_grad_h: vec3<f32>;
var<private> tf_grad_v: vec3<f32>;

// aL of the innermost loop, clamped to the sequencer's range.
fn al() -> i32 {
    if (loop_depth == 0u) {
        return 0;
    }
    return clamp(loop_al[loop_depth - 1u], -256, 256);
}

// Shader Model 3 multiply: 0 * anything (even inf or NaN) is 0.
fn mulz(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(a * b, vec4<f32>(0.0), min(abs(a), abs(b)) == vec4<f32>(0.0));
}

fn mulzs(a: f32, b: f32) -> f32 {
    return select(a * b, 0.0, min(abs(a), abs(b)) == 0.0);
}

// Max and min keep the first operand unless the comparison fails, like the
// console (and unlike IEEE max/min) for NaN.
fn vmax(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(b, a, a >= b);
}

fn vmin(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return select(b, a, a < b);
}

fn smax(a: f32, b: f32) -> f32 {
    return select(b, a, a >= b);
}

fn smin(a: f32, b: f32) -> f32 {
    return select(b, a, a < b);
}

// cube: input .z_xy (x = v.z, y = v.w, z = v.x); result is
// (t, s, 2 * major axis, face).
fn cube(v: vec4<f32>) -> vec4<f32> {
    let x = v.z;
    let y = v.w;
    let z = v.x;
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) {
        return vec4<f32>(-y, select(x, -x, z < 0.0), 2.0 * z, select(4.0, 5.0, z < 0.0));
    }
    if (abs(y) >= abs(x)) {
        return vec4<f32>(select(z, -z, y < 0.0), x, 2.0 * y, select(2.0, 3.0, y < 0.0));
    }
    return vec4<f32>(-y, select(-z, z, x < 0.0), 2.0 * x, select(0.0, 1.0, x < 0.0));
}

// Clamps infinities to +-FLT_MAX (rcpc, rsqc).
fn clamp_inf(v: f32) -> f32 {
    return select(select(v, FLT_MAX, v > FLT_MAX), -FLT_MAX, v < -FLT_MAX);
}

// Clamps only -inf to -FLT_MAX (logc).
fn clamp_neg_inf(v: f32) -> f32 {
    return select(v, -FLT_MAX, v < -FLT_MAX);
}

// Turns infinities into signed zeros (the *f variants of rcp/rsq).
fn zero_inf(v: f32) -> f32 {
    return select(select(v, 0.0, v > FLT_MAX), bitcast<f32>(0x80000000u), v < -FLT_MAX);
}
";

const GUEST_PRELUDE: &str = "
@group(0) @binding(0) var<storage, read> guest: array<u32>;

// Xenos endian modes: 0 none, 1 8in16, 2 8in32, 3 16in32.
fn swap32(v: u32, e: u32) -> u32 {
    var x = v;
    if (e == 1u || e == 2u) {
        x = ((x & 0x00FF00FFu) << 8u) | ((x >> 8u) & 0x00FF00FFu);
    }
    if (e == 2u || e == 3u) {
        x = (x << 16u) | (x >> 16u);
    }
    return x;
}

// Console half float: exponent 31 is an ordinary exponent (no inf or NaN).
// Done by hand because unpack2x16float needs an optional capability in naga.
fn half(h: u32) -> f32 {
    let e = (h >> 10u) & 0x1Fu;
    let m = h & 0x3FFu;
    var v = f32(m) * 5.9604645e-8f;
    if (e != 0u) {
        v = ldexp(f32(m | 0x400u), i32(e) - 25);
    }
    return select(v, -v, (h & 0x8000u) != 0u);
}

// Word at guest dword address `addr`, or 0 outside the uploaded window.
fn vword(addr: i32, e: u32) -> u32 {
    let i = addr - i32(consts.window_base >> 2u);
    if (i < 0 || u32(i) >= arrayLength(&guest)) {
        return 0u;
    }
    return swap32(guest[u32(i)], e);
}
";

// Texture fetch result. The host texture already holds the swizzled texels
// with signed components (sign 1) converted at upload, so only the biased
// sign (2: x * 2 - 1) and gamma (3) remain. The sign of an output component
// is the sign of the source component its swizzle select picks (fetch dword
// 0 bits 2..9, swizzle in dword 3 bits 1..12); constant 0 / 1 selects take
// none. The result is then scaled by 2^exp_adjust (dword 3 bits 13..18,
// signed).
//
// Gamma is the console's piecewise-linear 8-bit curve: on the 0..255 input
// scale it has breakpoints at 64, 96 and 192 with slopes 1, 2, 4 and 8 to a
// 10-bit linear value, plus a slope-scaled correction that stretches the
// top of the curve to reach 1023.
const TFETCH_PRELUDE: &str = "
fn xe_pwl_gamma_to_linear(g: f32) -> f32 {
    let x = clamp(g, 0.0, 1.0) * 255.0;
    let seg = u32(x >= 64.0) + u32(x >= 96.0) + u32(x >= 192.0);
    var seg_start = array<f32, 4>(0.0, 64.0, 96.0, 192.0);
    var seg_linear = array<f32, 4>(0.0, 64.0, 128.0, 512.0);
    let slope = f32(1u << seg);
    let l = seg_linear[seg] + slope * (x - seg_start[seg]);
    // Multiply by the reciprocal: WGSL division is allowed 2.5 ulp of error,
    // and the console curve scales by 1/1023 this way.
    return (l + floor(l * slope / 1024.0)) * (1.0 / 1023.0);
}

fn xe_tfetch_sign_bias(v: vec4<f32>, w0: u32, w3: u32) -> vec4<f32> {
    var r = v;
    for (var i = 0u; i < 4u; i++) {
        let s = extractBits(w3, 1u + 3u * i, 3u);
        if (s > 3u) {
            continue;
        }
        let sign = extractBits(w0, 2u + 2u * s, 2u);
        if (sign == 2u) {
            r[i] = v[i] * 2.0 - 1.0;
        } else if (sign == 3u) {
            r[i] = xe_pwl_gamma_to_linear(v[i]);
        }
    }
    return r * ldexp(1.0, extractBits(bitcast<i32>(w3), 13u, 6u));
}
";

/// Translates one vertex or pixel program to a WGSL module with entry point
/// `vs_main` or `ps_main`.
pub fn translate(words: &[u32], stage: ShaderStage) -> Result<Translated, Error> {
    translate_with(words, stage, None)
}

/// [`translate`] with pixel parameter generation (ignored for the vertex
/// stage). The generated register is not an input: its bit is left out of
/// [`ShaderInfo::interpolators_read`].
pub fn translate_with(
    words: &[u32],
    stage: ShaderStage,
    param_gen: Option<ParamGen>,
) -> Result<Translated, Error> {
    let prog = Program::parse(words)?;
    let param_gen = param_gen.filter(|p| stage == ShaderStage::Pixel && p.register < 16);
    let mut g = Gen {
        stage,
        info: ShaderInfo::default(),
        body: String::new(),
        live_fulls: Vec::new(),
        cond_exec: false,
        forward_jump: false,
        r0x_written: false,
        nonlinear: false,
        has_mini: false,
        float_min_max: None,
        param_gen,
    };
    g.info.interpolators_read = prog.interpolators_read(stage);
    if let Some(p) = param_gen {
        g.info.interpolators_read &= !(1 << p.register);
    }
    for (k, cf) in prog.cf.iter().enumerate() {
        g.cf(k as u32, cf, &prog)?;
    }
    g.info.float_consts_used = g.float_min_max;
    // shortcut: with loops, calls or backward jumps a fetch may run after a
    // later r0 write, and a vfetch_mini after another vfetch_full than the
    // program-order one; no index bound then. Track per path if it matters.
    if g.nonlinear && (g.r0x_written || g.has_mini) {
        for r in &mut g.info.vertex_fetch_reach {
            r.bounded &= r.stride == 0 && !g.has_mini;
        }
    }
    Ok(Translated {
        wgsl: g.finish(prog.cf.len() as u32),
        info: g.info,
    })
}

/// The statement writing the pixel parameters to their register. The host
/// fragment position is the pixel centre (x.5), so rounding down gives
/// Xenia's .0 position. The flags are set as sign bits with bitcasts so a
/// zero coordinate keeps its sign.
fn param_gen_wgsl(p: ParamGen) -> String {
    const SIGN: &str = "0x80000000u";
    let x = match p.primitive {
        ParamGenPrimitive::Polygon => {
            format!("bitcast<f32>(bitcast<u32>(xy.x) | select({SIGN}, 0u, input.xe_front_facing))")
        }
        _ => "xy.x".into(),
    };
    let y = match p.primitive {
        // shortcut: points are one host pixel without sprite coordinates
        // (zw stay 0); expand points to sprites when a game draws them.
        ParamGenPrimitive::Point => format!("bitcast<f32>(bitcast<u32>(xy.y) | {SIGN})"),
        _ => "xy.y".into(),
    };
    let z = match p.primitive {
        ParamGenPrimitive::Line => format!("bitcast<f32>({SIGN})"),
        _ => "0.0".into(),
    };
    format!(
        "    {{\n        let xy = floor(input.xe_frag_coord.xy);\n        r[{}] = vec4<f32>({x}, {y}, {z}, 0.0);\n    }}\n",
        p.register
    )
}

struct Program {
    cf: Vec<CfInstr>,
    /// Decoded instructions by slot.
    instrs: Vec<Option<Instr>>,
}

impl Program {
    fn parse(words: &[u32]) -> Result<Self, Error> {
        if words.is_empty() || words.len() % 3 != 0 {
            return Err(Error::Malformed(format!(
                "program length {} is not a positive multiple of 3 words",
                words.len()
            )));
        }
        let slots = (words.len() / 3) as u32;
        let pair = |i: u32| {
            let i = i as usize * 3;
            decode_cf_pair([words[i], words[i + 1], words[i + 2]])
        };
        // Like Xenia (after freedreno): the control-flow area ends at the
        // first slot an exec points to.
        let mut bound = slots;
        let mut i = 0;
        while i < bound {
            for cf in pair(i) {
                if let CfInstr::Exec(e) = cf {
                    if e.count > 0 {
                        bound = bound.min(e.address);
                    }
                }
            }
            i += 1;
        }
        if bound == 0 {
            return Err(Error::Malformed(
                "exec points into the control-flow area at slot 0".into(),
            ));
        }
        let cf: Vec<CfInstr> = (0..bound).flat_map(pair).collect();
        let n = cf.len() as u32;
        let mut instrs = vec![None; slots as usize];
        for (k, c) in cf.iter().enumerate() {
            match *c {
                CfInstr::Exec(e) => {
                    if e.address + e.count > slots {
                        return Err(Error::Malformed(format!(
                            "exec {k} reads slots {}..{} past the end ({slots})",
                            e.address,
                            e.address + e.count
                        )));
                    }
                    for j in 0..e.count {
                        let s = (e.address + j) as usize;
                        let w = [words[s * 3], words[s * 3 + 1], words[s * 3 + 2]];
                        let fetch = (e.sequence >> (2 * j)) & 1 != 0;
                        let ins = decode_instr(w, fetch)?;
                        if instrs[s].is_some_and(|old| old != ins) {
                            return Err(Error::Malformed(format!(
                                "slot {s} used as both ALU and fetch"
                            )));
                        }
                        instrs[s] = Some(ins);
                    }
                }
                CfInstr::LoopStart { address, .. }
                | CfInstr::LoopEnd { address, .. }
                | CfInstr::CondCall { address, .. }
                | CfInstr::CondJmp { address, .. }
                    if address > n =>
                {
                    return Err(Error::Malformed(format!(
                        "control flow {k} targets {address}, past the end ({n})"
                    )));
                }
                _ => {}
            }
        }
        Ok(Program { cf, instrs })
    }

    /// Registers r0..r15 the program reads anywhere (all 16 if any read is
    /// loop-relative). Only meaningful for the pixel stage.
    fn interpolators_read(&self, stage: ShaderStage) -> u16 {
        if stage != ShaderStage::Pixel {
            return 0;
        }
        let mut m = 0u32;
        let mut temp = |reg: u32, rel: bool| {
            if rel {
                m |= 0xFFFF;
            } else if reg < 16 {
                m |= 1 << reg;
            }
        };
        for ins in self.instrs.iter().flatten() {
            match ins {
                Instr::Fetch(FetchInstr::Vertex(v)) => temp(v.src, v.src_relative),
                Instr::Fetch(FetchInstr::Texture(t)) => temp(t.src, t.src_relative),
                Instr::Alu(a) => {
                    let n = a.vector_op.operand_count();
                    for s in &a.src[..n] {
                        if s.is_temp {
                            temp(s.reg & 0x3F, s.reg & 0x40 != 0);
                        }
                    }
                    match a.scalar_op.operand_count() {
                        1 if a.src[2].is_temp => {
                            temp(a.src[2].reg & 0x3F, a.src[2].reg & 0x40 != 0)
                        }
                        2 => temp(a.scalar_const_op_temp_reg(), false),
                        _ => {}
                    }
                }
            }
        }
        m as u16
    }
}

struct Gen {
    stage: ShaderStage,
    info: ShaderInfo,
    body: String,
    /// Fetch constant, stride and index kind of each vfetch_full whose
    /// address a following vfetch_mini may read through, last in program
    /// order last (its constant gives the mini's endianness, as Xenia tracks
    /// it). A full that may be skipped (predicated, in a conditional exec,
    /// or after a forward jump) leaves the earlier ones live.
    live_fulls: Vec<(u32, u32, bool)>,
    /// The exec being translated has a condition.
    cond_exec: bool,
    /// A forward jump came earlier in program order, so later instructions
    /// may be skipped.
    forward_jump: bool,
    /// Some instruction so far in program order may write r0.x.
    r0x_written: bool,
    has_mini: bool,
    /// Control flow can run an instruction after one later in program order
    /// (loops, calls, backward jumps).
    nonlinear: bool,
    float_min_max: Option<(u32, u32)>,
    param_gen: Option<ParamGen>,
}

fn comp(c: u32) -> char {
    ['x', 'y', 'z', 'w'][c as usize & 3]
}

fn flit(v: f32) -> String {
    format!("{v:?}f")
}

fn bool_const(index: u32) -> String {
    format!(
        "(((consts.bool_consts[{}u][{}u] >> {}u) & 1u) != 0u)",
        index >> 7,
        (index >> 5) & 3,
        index & 31
    )
}

fn loop_const(id: u32) -> String {
    format!("consts.loop_consts[{}u][{}u]", id >> 2, id & 3)
}

fn fetch_dword(n: u32) -> String {
    format!("consts.fetch[{}u][{}u]", n / 4, n % 4)
}

fn temp_ref(reg: u32, rel: bool) -> String {
    if rel {
        format!("r[u32({reg} + al()) & 63u]")
    } else {
        format!("r[{reg}]")
    }
}

impl Gen {
    fn line(&mut self, s: &str) {
        self.body.push_str("            ");
        self.body.push_str(s);
        self.body.push('\n');
    }

    fn unsupported<T>(what: impl Into<String>) -> Result<T, Error> {
        Err(Error::Unsupported(what.into()))
    }

    fn cond(c: BranchCond) -> String {
        match c {
            BranchCond::Always => "true".into(),
            BranchCond::Bool { index, value } => format!("({} == {value})", bool_const(index)),
            BranchCond::Predicate(v) => format!("(p0 == {v})"),
        }
    }

    fn cf(&mut self, k: u32, cf: &CfInstr, prog: &Program) -> Result<(), Error> {
        self.nonlinear |= match *cf {
            CfInstr::LoopEnd { .. } | CfInstr::CondCall { .. } | CfInstr::Return => true,
            CfInstr::CondJmp { address, .. } => address <= k,
            _ => false,
        };
        self.forward_jump |= matches!(*cf, CfInstr::CondJmp { address, .. } if address > k + 1);
        let _ = writeln!(self.body, "        case {k}u: {{");
        self.line(&format!("pc = {}u;", k + 1));
        match *cf {
            CfInstr::Nop | CfInstr::Alloc { .. } | CfInstr::MarkVsFetchDone => {}
            CfInstr::Exec(e) => {
                let cond = match e.cond {
                    ExecCond::Always { .. } => "true".to_string(),
                    ExecCond::Bool { index, value, .. } => {
                        format!("({} == {value})", bool_const(index))
                    }
                    ExecCond::Predicate { value, .. } => format!("(p0 == {value})"),
                };
                self.line(&format!("if {cond} {{"));
                self.cond_exec = !matches!(e.cond, ExecCond::Always { .. });
                for j in 0..e.count {
                    let ins = prog.instrs[(e.address + j) as usize].expect("decoded in parse");
                    match ins {
                        Instr::Alu(a) => self.alu(&a)?,
                        Instr::Fetch(FetchInstr::Vertex(v)) => self.vfetch(&v)?,
                        Instr::Fetch(FetchInstr::Texture(t)) => self.tfetch(&t)?,
                    }
                }
                if e.end {
                    self.line("pc = 0xFFFFFFFFu;");
                }
                self.line("}");
            }
            CfInstr::LoopStart {
                address,
                repeat,
                loop_id,
            } => {
                self.line(&format!("let lc = {};", loop_const(loop_id)));
                self.line("if (loop_depth >= 4u || (lc & 0xFFu) == 0u) {");
                self.line(&format!("pc = {address}u;"));
                self.line("} else {");
                let start = if repeat {
                    "al()".to_string()
                } else {
                    "i32((lc >> 8u) & 0xFFu)".to_string()
                };
                self.line(&format!("let start = {start};"));
                self.line("loop_count[loop_depth] = lc & 0xFFu;");
                self.line("loop_al[loop_depth] = start;");
                self.line("loop_depth += 1u;");
                self.line("}");
            }
            CfInstr::LoopEnd {
                address,
                loop_id,
                predicated_break,
            } => {
                let brk = match predicated_break {
                    Some(v) => format!(" && p0 != {v}"),
                    None => String::new(),
                };
                self.line("if (loop_depth > 0u) {");
                self.line("let d = loop_depth - 1u;");
                self.line("loop_count[d] -= 1u;");
                self.line(&format!("if (loop_count[d] != 0u{brk}) {{"));
                self.line(&format!(
                    "loop_al[d] += extractBits(bitcast<i32>({}), 16u, 8u);",
                    loop_const(loop_id)
                ));
                self.line(&format!("pc = {address}u;"));
                self.line("} else {");
                self.line("loop_depth = d;");
                self.line("}");
                self.line("}");
            }
            CfInstr::CondCall { address, cond } => {
                self.line(&format!("if ({} && call_depth < 4u) {{", Self::cond(cond)));
                self.line(&format!("call_ret[call_depth] = {}u;", k + 1));
                self.line("call_depth += 1u;");
                self.line(&format!("pc = {address}u;"));
                self.line("}");
            }
            CfInstr::Return => {
                self.line("if (call_depth > 0u) {");
                self.line("call_depth -= 1u;");
                self.line("pc = call_ret[call_depth];");
                self.line("}");
            }
            CfInstr::CondJmp { address, cond, .. } => {
                self.line(&format!("if {} {{", Self::cond(cond)));
                self.line(&format!("pc = {address}u;"));
                self.line("}");
            }
        }
        self.body.push_str("        }\n");
        Ok(())
    }

    /// Notes a write of temporary `reg` (relative or not) whose x component
    /// is written when `x`.
    fn note_write(&mut self, reg: u32, rel: bool, x: bool) {
        self.r0x_written |= x && (rel || reg == 0);
    }

    fn note_float_const(&mut self, idx: u32) {
        self.float_min_max = Some(match self.float_min_max {
            None => (idx, idx),
            Some((lo, hi)) => (lo.min(idx), hi.max(idx)),
        });
    }

    fn const_ref(&mut self, idx: u32, relative: bool, a0_relative: bool) -> String {
        if relative {
            self.info.float_consts_dynamic = true;
            let off = if a0_relative { "a0" } else { "al()" };
            format!("cnst({idx} + {off})")
        } else {
            self.note_float_const(idx);
            format!("fc[{idx}u - FC_LO]")
        }
    }

    /// Operand `i` (0-based) of a vector operation as a vec4 expression.
    fn vec_src(&mut self, a: &AluInstr, i: usize) -> String {
        let s = a.src[i];
        let (base, abs) = if s.is_temp {
            (temp_ref(s.reg & 0x3F, s.reg & 0x40 != 0), s.reg & 0x80 != 0)
        } else {
            let rel = a.src_const_relative(i + 1);
            (
                self.const_ref(s.reg, rel, a.const_relative_a0),
                a.abs_constants,
            )
        };
        let sw: String = (0..4)
            .map(|c| comp(alu_swizzle_component(s.swizzle, c)))
            .collect();
        let mut e = if sw == "xyzw" {
            base
        } else {
            format!("{base}.{sw}")
        };
        if abs {
            e = format!("abs({e})");
        }
        if s.negate {
            e = format!("(-{e})");
        }
        e
    }

    /// The `a` (W swizzle) and `b` (X swizzle) components of src3.
    fn scalar_src(&mut self, a: &AluInstr) -> (String, String) {
        let s = a.src[2];
        let (base, abs) = if s.is_temp {
            (temp_ref(s.reg & 0x3F, s.reg & 0x40 != 0), s.reg & 0x80 != 0)
        } else {
            let rel = a.src_const_relative(3);
            (
                self.const_ref(s.reg, rel, a.const_relative_a0),
                a.abs_constants,
            )
        };
        let one = |c: u32| {
            let mut e = format!("{base}.{}", comp(alu_swizzle_component(s.swizzle, c)));
            if abs {
                e = format!("abs({e})");
            }
            if s.negate {
                e = format!("(-{e})");
            }
            e
        };
        (one(3), one(0))
    }

    fn export_target(&mut self, reg: u32) -> Result<Option<String>, Error> {
        match (self.stage, reg) {
            (_, 32..=37) => Self::unsupported(format!("memory export (eA/eM{reg})")),
            (ShaderStage::Vertex, 0..=15) => {
                self.info.interpolators_written |= 1 << reg;
                Ok(Some(format!("o_interp[{reg}]")))
            }
            (ShaderStage::Vertex, 62) => {
                self.info.writes_position = true;
                Ok(Some("o_pos".into()))
            }
            (ShaderStage::Vertex, 63) => {
                self.info.point_size_export_ignored = true;
                Ok(None)
            }
            (ShaderStage::Pixel, 0..=3) => {
                self.info.colour_targets_written |= 1 << reg;
                Ok(Some(format!("o_col[{reg}]")))
            }
            (ShaderStage::Pixel, 61) => Self::unsupported("pixel depth export (oDepth)"),
            (stage, _) => Err(Error::Malformed(format!(
                "export register {reg} is invalid for {stage:?}"
            ))),
        }
    }

    fn kill(&mut self, cond: &str) -> Result<(), Error> {
        if self.stage != ShaderStage::Pixel {
            return Self::unsupported("kill in a vertex shader");
        }
        self.info.kills = true;
        self.line(&format!("if {cond} {{ discard; }}"));
        Ok(())
    }

    fn alu(&mut self, a: &AluInstr) -> Result<(), Error> {
        use ScalarOp as S;
        use VectorOp as V;
        let export = a.export;
        let vmask = if export {
            a.vector_write_mask & !a.scalar_write_mask
        } else {
            a.vector_write_mask
        };
        let smask = if export {
            a.scalar_write_mask & !a.vector_write_mask
        } else {
            a.scalar_write_mask
        };
        let zero_mask = if export && a.scalar_dest_relative {
            0xF & !(a.vector_write_mask | a.scalar_write_mask)
        } else {
            0
        };
        let one_mask = if export {
            a.vector_write_mask & a.scalar_write_mask
        } else {
            0
        };
        let vector_side_effect = matches!(
            a.vector_op,
            V::SetpEqPush
                | V::SetpNePush
                | V::SetpGtPush
                | V::SetpGePush
                | V::KillEq
                | V::KillGt
                | V::KillGe
                | V::KillNe
                | V::MaxA
        );
        let do_vector = vmask != 0 || vector_side_effect;
        let do_scalar = a.scalar_op != S::RetainPrev;
        let writes_any = vmask | smask | zero_mask | one_mask != 0;
        if !do_vector && !do_scalar && !writes_any {
            return Ok(());
        }
        // Resolve destinations first so invalid exports fail early.
        let export_target = if export {
            Some(self.export_target(a.vector_dest)?)
        } else {
            None
        };
        if let Some(v) = a.predicate {
            self.line(&format!("if (p0 == {v}) {{"));
        } else {
            self.line("{");
        }
        // All operands are read before either operation writes anything.
        if do_vector {
            for i in 0..a.vector_op.operand_count() {
                let e = self.vec_src(a, i);
                self.line(&format!("let v{i} = {e};"));
            }
        }
        match a.scalar_op.operand_count() {
            1 => {
                let (sa, sb) = self.scalar_src(a);
                self.line(&format!("let sa = {sa};"));
                self.line(&format!("let sb = {sb};"));
            }
            2 => {
                let s = a.src[2];
                let c = self.const_ref(s.reg, a.src_const_relative(3), a.const_relative_a0);
                let fix = |mut e: String| {
                    if a.abs_constants {
                        e = format!("abs({e})");
                    }
                    if s.negate {
                        e = format!("(-{e})");
                    }
                    e
                };
                let sa = fix(format!("{c}.{}", comp(alu_swizzle_component(s.swizzle, 3))));
                let sb = fix(format!(
                    "r[{}].{}",
                    a.scalar_const_op_temp_reg(),
                    comp(alu_swizzle_component(s.swizzle, 0))
                ));
                self.line(&format!("let sa = {sa};"));
                self.line(&format!("let sb = {sb};"));
            }
            _ => {}
        }

        if do_vector {
            let cmp = |op: V| match op {
                V::Seq | V::CndEq | V::SetpEqPush | V::KillEq => "==",
                V::Sgt | V::CndGt | V::SetpGtPush | V::KillGt => ">",
                V::Sge | V::CndGe | V::SetpGePush | V::KillGe => ">=",
                _ => "!=",
            };
            let expr = match a.vector_op {
                V::Add => "v0 + v1".to_string(),
                V::Mul => "mulz(v0, v1)".into(),
                V::Max => "vmax(v0, v1)".into(),
                V::Min => "vmin(v0, v1)".into(),
                V::Seq | V::Sgt | V::Sge | V::Sne => {
                    format!(
                        "select(vec4<f32>(0.0), vec4<f32>(1.0), v0 {} v1)",
                        cmp(a.vector_op)
                    )
                }
                V::Frc => "fract(v0)".into(),
                V::Trunc => "trunc(v0)".into(),
                V::Floor => "floor(v0)".into(),
                V::Mad => "mulz(v0, v1) + v2".into(),
                V::CndEq | V::CndGe | V::CndGt => {
                    format!("select(v2, v1, v0 {} vec4<f32>(0.0))", cmp(a.vector_op))
                }
                V::Dp4 => "vec4<f32>(dot_p.x + dot_p.y + dot_p.z + dot_p.w)".into(),
                V::Dp3 => "vec4<f32>(dot_p.x + dot_p.y + dot_p.z)".into(),
                V::Dp2Add => "vec4<f32>(dot_p.x + dot_p.y + v2.x)".into(),
                V::Cube => "cube(v0)".into(),
                V::Max4 => "vec4<f32>(max(max(max(v0.x, v0.y), v0.z), v0.w))".into(),
                V::SetpEqPush | V::SetpNePush | V::SetpGtPush | V::SetpGePush => {
                    let c = cmp(a.vector_op);
                    self.line(&format!("p0 = v0.w == 0.0 && v1.w {c} 0.0;"));
                    format!("vec4<f32>(select(v0.x, -1.0, v0.x == 0.0 && v1.x {c} 0.0) + 1.0)")
                }
                V::KillEq | V::KillGt | V::KillGe | V::KillNe => {
                    self.kill(&format!("(any(v0 {} v1))", cmp(a.vector_op)))?;
                    "vec4<f32>(0.0)".into()
                }
                V::Dst => "vec4<f32>(1.0, mulzs(v0.y, v1.y), v0.z, v1.w)".into(),
                V::MaxA => {
                    self.line("a0 = i32(clamp(floor(v0.w + 0.5), -256.0, 255.0));");
                    "vmax(v0, v1)".into()
                }
            };
            if matches!(a.vector_op, V::Dp4 | V::Dp3 | V::Dp2Add) {
                self.line("let dot_p = mulz(v0, v1);");
            }
            let expr = if a.vector_saturate {
                format!("saturate({expr})")
            } else {
                expr
            };
            self.line(&format!("let vr = {expr};"));
        }

        if do_scalar {
            let cmp = |op: S| match op {
                S::Seqs | S::SetpEq | S::KillsEq => "==",
                S::Sgts | S::SetpGt | S::KillsGt => ">",
                S::Sges | S::SetpGe | S::KillsGe => ">=",
                _ => "!=",
            };
            let expr: String = match a.scalar_op {
                S::Adds => "sa + sb".into(),
                S::AddsPrev => "sa + ps".into(),
                S::Muls => "mulzs(sa, sb)".into(),
                S::MulsPrev => "mulzs(sa, ps)".into(),
                S::MulsPrev2 => format!(
                    "select(-{FLT_MAX}, mulzs(sa, ps), ps != -{FLT_MAX} && abs(ps) <= {FLT_MAX} && abs(sb) <= {FLT_MAX} && sb > 0.0)"
                ),
                S::Maxs => "smax(sa, sb)".into(),
                S::Mins => "smin(sa, sb)".into(),
                S::MaxAs => {
                    self.line("a0 = i32(clamp(floor(sa + 0.5), -256.0, 255.0));");
                    "smax(sa, sb)".into()
                }
                S::MaxAsf => {
                    self.line("a0 = i32(clamp(floor(sa), -256.0, 255.0));");
                    "smax(sa, sb)".into()
                }
                S::Seqs | S::Sgts | S::Sges | S::Snes => {
                    format!("select(0.0, 1.0, sa {} 0.0)", cmp(a.scalar_op))
                }
                S::Frcs => "fract(sa)".into(),
                S::Truncs => "trunc(sa)".into(),
                S::Floors => "floor(sa)".into(),
                S::Exp => "exp2(sa)".into(),
                S::Log => "log2(sa)".into(),
                // Only -inf is clamped (log of 0); +inf stays (Xenia ucode.h kLogc).
                S::Logc => "clamp_neg_inf(log2(sa))".into(),
                S::Rcp => "1.0 / sa".into(),
                S::Rcpc => "clamp_inf(1.0 / sa)".into(),
                S::Rcpf => "zero_inf(1.0 / sa)".into(),
                S::Rsq => "inverseSqrt(sa)".into(),
                S::Rsqc => "clamp_inf(inverseSqrt(sa))".into(),
                S::Rsqf => "zero_inf(inverseSqrt(sa))".into(),
                S::Subs => "sa - sb".into(),
                S::SubsPrev => "sa - ps".into(),
                S::SetpEq | S::SetpNe | S::SetpGt | S::SetpGe => {
                    self.line(&format!("p0 = sa {} 0.0;", cmp(a.scalar_op)));
                    "select(1.0, 0.0, p0)".into()
                }
                S::SetpInv => {
                    self.line("p0 = sa == 1.0;");
                    "select(select(sa, 1.0, sa == 0.0), 0.0, p0)".into()
                }
                S::SetpPop => {
                    self.line("let pop = sa - 1.0;");
                    self.line("p0 = pop <= 0.0;");
                    "select(pop, 0.0, p0)".into()
                }
                S::SetpClr => {
                    self.line("p0 = false;");
                    FLT_MAX.into()
                }
                S::SetpRstr => {
                    self.line("p0 = sa == 0.0;");
                    "select(sa, 0.0, p0)".into()
                }
                S::KillsEq | S::KillsGt | S::KillsGe | S::KillsNe => {
                    self.kill(&format!("(sa {} 0.0)", cmp(a.scalar_op)))?;
                    "0.0".into()
                }
                S::KillsOne => {
                    self.kill("(sa == 1.0)")?;
                    "0.0".into()
                }
                S::Sqrt => "sqrt(sa)".into(),
                S::Sin => "sin(sa)".into(),
                S::Cos => "cos(sa)".into(),
                S::Mulsc0 | S::Mulsc1 => "mulzs(sa, sb)".into(),
                S::Addsc0 | S::Addsc1 => "sa + sb".into(),
                S::Subsc0 | S::Subsc1 => "sa - sb".into(),
                S::RetainPrev => unreachable!(),
            };
            self.line(&format!("let sr: f32 = {expr};"));
            // ps keeps the unsaturated result, as in Xenia.
            self.line("ps = sr;");
        }

        // Writes.
        if export_target.is_none() {
            self.note_write(a.vector_dest, a.vector_dest_relative, vmask & 1 != 0);
            self.note_write(a.scalar_dest, a.scalar_dest_relative, smask & 1 != 0);
        }
        let (vdst, sdst) = match &export_target {
            Some(None) => (None, None),
            Some(Some(t)) => (Some(t.clone()), Some(t.clone())),
            None => (
                Some(temp_ref(a.vector_dest, a.vector_dest_relative)),
                Some(temp_ref(a.scalar_dest, a.scalar_dest_relative)),
            ),
        };
        if let Some(d) = vdst {
            for c in 0..4 {
                if vmask & (1 << c) != 0 {
                    self.line(&format!("{d}.{0} = vr.{0};", comp(c)));
                }
            }
        }
        if let Some(d) = sdst {
            let val = if a.scalar_saturate {
                "saturate(ps)"
            } else {
                "ps"
            };
            for c in 0..4 {
                if smask & (1 << c) != 0 {
                    self.line(&format!("{d}.{} = {val};", comp(c)));
                }
            }
            if export {
                for c in 0..4 {
                    if zero_mask & (1 << c) != 0 {
                        self.line(&format!("{d}.{} = 0.0;", comp(c)));
                    } else if one_mask & (1 << c) != 0 {
                        self.line(&format!("{d}.{} = 1.0;", comp(c)));
                    }
                }
            }
        }
        self.line("}");
        Ok(())
    }

    /// Stores `v` (a vec4) into a fetch destination by its 3-bit swizzle.
    fn fetch_store(&mut self, dest: u32, rel: bool, swizzle: u32) {
        self.note_write(dest, rel, swizzle & 7 != 7);
        let d = temp_ref(dest, rel);
        for c in 0..4 {
            let src = match (swizzle >> (3 * c)) & 7 {
                s @ 0..=3 => format!("v.{}", comp(s)),
                5 => "1.0".into(),
                7 => continue,
                // 4 is 0.0; Xenia also treats the invalid 6 as 0.0.
                _ => "0.0".into(),
            };
            self.line(&format!("{d}.{} = {src};", comp(c)));
        }
    }

    fn vfetch(&mut self, f: &VertexFetch) -> Result<(), Error> {
        let (fc, _, _) = if f.mini {
            self.has_mini = true;
            *self.live_fulls.last().ok_or_else(|| {
                Error::Malformed("vfetch_mini without a preceding vfetch_full".into())
            })?
        } else {
            let vertex_index = self.stage == ShaderStage::Vertex
                && f.src == 0
                && !f.src_relative
                && f.src_swizzle == 0
                && !self.r0x_written;
            let full = (f.fetch_const, f.stride, f.stride == 0 || vertex_index);
            if f.predicate.is_none() && !self.cond_exec && !self.forward_jump {
                self.live_fulls.clear();
            }
            self.live_fulls.retain(|&l| l != full);
            self.live_fulls.push(full);
            full
        };
        let (words, comps) = vertex_format_decode(f)?;
        self.info.uses_vertex_fetch = true;
        let (first, end) = (f.offset, f.offset + words as i32);
        let sources = if f.mini {
            self.live_fulls.clone()
        } else {
            vec![*self.live_fulls.last().expect("pushed above")]
        };
        for (fc, stride, bounded) in sources {
            self.info.vertex_fetch_consts |= 1u128 << fc;
            let reach = &mut self.info.vertex_fetch_reach;
            match reach
                .iter_mut()
                .find(|r| (r.fetch_const, r.stride, r.bounded) == (fc, stride, bounded))
            {
                Some(r) => (r.first, r.end) = (r.first.min(first), r.end.max(end)),
                None => reach.push(VertexFetchReach {
                    fetch_const: fc,
                    stride,
                    bounded,
                    first,
                    end,
                }),
            }
        }
        if let Some(v) = f.predicate {
            self.line(&format!("if (p0 == {v}) {{"));
        } else {
            self.line("{");
        }
        if !f.mini {
            // Physical dword address, as Xenia masks fetch addresses (0x1FFFFFFF bytes).
            let mut addr = format!("i32(({} >> 2u) & 0x07FFFFFFu)", fetch_dword(fc * 2));
            if f.stride != 0 {
                let round = if f.index_rounded { " + 0.5" } else { "" };
                let src = temp_ref(f.src, f.src_relative);
                self.line(&format!(
                    "let idx = i32(floor({src}.{}{round}));",
                    comp(f.src_swizzle)
                ));
                let _ = write!(addr, " + idx * {}", f.stride);
            }
            let rebase = format!("consts.vf_rebase[{}u][{}u]", fc / 4, fc % 4);
            self.line(&format!("vf_addr = {addr} + {rebase};"));
        }
        self.line(&format!("let e = {} & 3u;", fetch_dword(fc * 2 + 1)));
        for w in 0..words {
            self.line(&format!(
                "let w{w} = vword(vf_addr + {}, e);",
                f.offset + w as i32
            ));
        }
        let mut v: Vec<String> = comps;
        if f.exp_adjust != 0 {
            let scale = flit(2f32.powi(f.exp_adjust));
            v = v.into_iter().map(|c| format!("({c}) * {scale}")).collect();
        }
        v.resize(4, "0.0".into());
        self.line(&format!("let v = vec4<f32>({});", v.join(", ")));
        self.fetch_store(f.dest, f.dest_relative, f.dest_swizzle);
        self.line("}");
        Ok(())
    }

    /// The texture instructions that only move values: setTexLOD,
    /// setGradientH/V and getGradients, as Xenia's
    /// ProcessTextureFetchInstruction handles them.
    fn texture_register_op(&mut self, t: &TextureFetch) -> Result<(), Error> {
        if t.op == TextureOp::GetGradients && self.stage != ShaderStage::Pixel {
            return Self::unsupported("getGradients outside a pixel shader");
        }
        if let Some(v) = t.predicate {
            self.line(&format!("if (p0 == {v}) {{"));
        } else {
            self.line("{");
        }
        let src = temp_ref(t.src, t.src_relative);
        let sw = |n: u32| -> String {
            (0..n)
                .map(|c| comp((t.src_swizzle >> (2 * c)) & 3))
                .collect()
        };
        match t.op {
            TextureOp::SetLod => self.line(&format!("tf_lod = {src}.{};", sw(1))),
            TextureOp::SetGradientsHorz => self.line(&format!("tf_grad_h = {src}.{};", sw(3))),
            TextureOp::SetGradientsVert => self.line(&format!("tf_grad_v = {src}.{};", sw(3))),
            _ => {
                // Coarse derivatives in Xenia's order: x = ddx(s0),
                // y = ddy(s0), z = ddx(s1), w = ddy(s1).
                self.line(&format!("let g = {src}.{};", sw(2)));
                self.line(
                    "let v = vec4<f32>(dpdxCoarse(g.x), dpdyCoarse(g.x), dpdxCoarse(g.y), dpdyCoarse(g.y));",
                );
                self.fetch_store(t.dest, t.dest_relative, t.dest_swizzle);
            }
        }
        self.line("}");
        Ok(())
    }

    fn tfetch(&mut self, t: &TextureFetch) -> Result<(), Error> {
        match t.op {
            TextureOp::Fetch => {}
            TextureOp::SetLod
            | TextureOp::SetGradientsHorz
            | TextureOp::SetGradientsVert
            | TextureOp::GetGradients => return self.texture_register_op(t),
            _ => return Self::unsupported(format!("texture instruction {:?}", t.op)),
        }
        if t.dim != TextureDim::D2 {
            return Self::unsupported(format!("{:?} texture fetch (only 2D is handled)", t.dim));
        }
        // As Xenia: register gradients only count with computed LOD; they
        // also give the vertex stage a computed LOD.
        let grad_lod = t.use_computed_lod && t.use_register_gradients;
        let n = t.slot;
        self.info.textures_used |= 1 << n;
        let overrides = t.mag_filter != 3
            || t.min_filter != 3
            || t.mip_filter != 3
            || t.aniso_filter != 7
            || t.vol_mag_filter != 3
            || t.vol_min_filter != 3;
        if overrides {
            self.info.texture_filter_overrides |= 1 << n;
        }
        if let Some(v) = t.predicate {
            self.line(&format!("if (p0 == {v}) {{"));
        } else {
            self.line("{");
        }
        let src = temp_ref(t.src, t.src_relative);
        let sw: String = (0..2)
            .map(|c| comp((t.src_swizzle >> (2 * c)) & 3))
            .collect();
        let mut coord = format!("{src}.{sw}");
        let needs_size = t.unnormalized || t.offset[0] != 0 || t.offset[1] != 0;
        if needs_size {
            self.line(&format!(
                "let size = vec2<f32>(textureDimensions(tex{n}, 0));"
            ));
        }
        if t.unnormalized {
            coord = format!("({coord}) / size");
        }
        if t.offset[0] != 0 || t.offset[1] != 0 {
            coord = format!(
                "{coord} + vec2<f32>({}, {}) / size",
                flit(t.offset[0] as f32 * 0.5),
                flit(t.offset[1] as f32 * 0.5)
            );
        }
        self.line(&format!("let uv = {coord};"));
        // LOD bias: the fetch constant's (dword 4 bits 12..21, signed, 1/32
        // steps; Xenia xenos.h xe_gpu_texture_fetch_t), the register LOD if
        // used, then the instruction's (1/16 steps), as
        // spirv_shader_translator_fetch.cc adds them.
        self.line(&format!(
            "let lod = f32(extractBits(bitcast<i32>({}), 12u, 10u)) / 32.0{} + {};",
            fetch_dword(n * 6 + 4),
            if t.use_register_lod { " + tf_lod" } else { "" },
            flit(t.lod_bias as f32 / 16.0)
        ));
        if grad_lod
            || t.use_register_lod
            || !(t.use_computed_lod && self.stage == ShaderStage::Pixel)
        {
            self.info.texture_explicit_lod |= 1 << n;
        }
        if t.lod_bias != 0 {
            self.info.texture_lod_bias |= 1 << n;
        }
        let sample = if grad_lod {
            // Register gradients scaled by 2^lod (Xenia's gradient form of
            // the bias), normalized when the coordinates are in texels.
            let norm = if t.unnormalized { " / size" } else { "" };
            format!(
                "textureSampleGrad(tex{n}, samp{n}, uv, tf_grad_h.xy * exp2(lod){norm}, tf_grad_v.xy * exp2(lod){norm})"
            )
        } else if t.use_computed_lod && self.stage == ShaderStage::Pixel {
            format!("textureSampleBias(tex{n}, samp{n}, uv, lod)")
        } else {
            // No derivatives in the vertex stage; without computed LOD the
            // bias is the LOD.
            format!("textureSampleLevel(tex{n}, samp{n}, uv, lod)")
        };
        self.line(&format!(
            "let v = xe_tfetch_sign_bias({sample}, {}, {});",
            fetch_dword(n * 6),
            fetch_dword(n * 6 + 3)
        ));
        self.fetch_store(t.dest, t.dest_relative, t.dest_swizzle);
        self.line("}");
        Ok(())
    }

    fn finish(&self, cf_count: u32) -> String {
        let mut s = String::new();
        if self.stage == ShaderStage::Pixel {
            // The pc-switch dispatch is never uniform control flow as far as
            // WGSL can tell, so implicit-LOD sampling needs this.
            s.push_str("diagnostic(off, derivative_uniformity);\n\n");
        }
        let _ = writeln!(s, "const FLT_MAX: f32 = {FLT_MAX};\n");
        s.push_str(PRELUDE);
        if let Some((first, count)) = self.info.float_consts_span() {
            let binding = match self.stage {
                ShaderStage::Vertex => VERTEX_FLOAT_CONSTS_BINDING,
                ShaderStage::Pixel => PIXEL_FLOAT_CONSTS_BINDING,
            };
            let _ = write!(
                s,
                "
// The stage's float constants c{first}..: only the span it reads.
const FC_LO: u32 = {first}u;
@group(0) @binding({binding}) var<uniform> fc: array<vec4<f32>, {count}>;
"
            );
        }
        if self.info.float_consts_dynamic {
            s.push_str(
                "
// Float constant by dynamic index; out of range reads 0.
fn cnst(i: i32) -> vec4<f32> {
    if (i < 0 || i > 255) {
        return vec4<f32>(0.0);
    }
    return fc[u32(i)];
}
",
            );
        }
        if self.info.uses_vertex_fetch {
            s.push_str(GUEST_PRELUDE);
        }
        if self.info.textures_used != 0 {
            s.push_str(TFETCH_PRELUDE);
        }
        for n in 0..32 {
            if self.info.textures_used & (1 << n) != 0 {
                let _ = write!(
                    s,
                    "\n@group(1) @binding({}) var tex{n}: texture_2d<f32>;\n@group(1) @binding({}) var samp{n}: sampler;\n",
                    2 * n,
                    2 * n + 1
                );
            }
        }
        s.push('\n');
        let (entry, ret) = match self.stage {
            ShaderStage::Vertex => {
                s.push_str("var<private> o_pos: vec4<f32>;\nvar<private> o_interp: array<vec4<f32>, 16>;\n\n");
                s.push_str("struct VsOut {\n    @builtin(position) pos: vec4<f32>,\n");
                for i in 0..16 {
                    if self.info.interpolators_written & (1 << i) != 0 {
                        let _ = writeln!(s, "    @location({i}) o{i}: vec4<f32>,");
                    }
                }
                s.push_str("}\n\n");
                let mut ret = "    var out: VsOut;\n    out.pos = o_pos;\n".to_string();
                for i in 0..16 {
                    if self.info.interpolators_written & (1 << i) != 0 {
                        let _ = writeln!(ret, "    out.o{i} = o_interp[{i}];");
                    }
                }
                ret.push_str("    return out;\n");
                let entry = "@vertex\nfn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {\n    \
                     r[0].x = f32((vertex_index + consts.vertex_base) & 0xFFFFFFu);\n"
                    .to_string();
                (entry, ret)
            }
            ShaderStage::Pixel => {
                s.push_str("var<private> o_col: array<vec4<f32>, 4>;\n\n");
                let reads = self.info.interpolators_read;
                let pg = self.param_gen;
                let has_input = reads != 0 || pg.is_some();
                if has_input {
                    s.push_str("struct PsIn {\n");
                    for i in 0..16 {
                        if reads & (1 << i) != 0 {
                            let _ = writeln!(s, "    @location({i}) i{i}: vec4<f32>,");
                        }
                    }
                    if let Some(p) = pg {
                        s.push_str("    @builtin(position) xe_frag_coord: vec4<f32>,\n");
                        if p.primitive == ParamGenPrimitive::Polygon {
                            s.push_str("    @builtin(front_facing) xe_front_facing: bool,\n");
                        }
                    }
                    s.push_str("}\n\n");
                }
                let cols = self.info.colour_targets_written;
                let mut ret = String::new();
                if cols != 0 {
                    s.push_str("struct PsOut {\n");
                    ret.push_str("    var out: PsOut;\n");
                    for i in 0..4 {
                        if cols & (1 << i) != 0 {
                            let _ = writeln!(s, "    @location({i}) c{i}: vec4<f32>,");
                            let _ = writeln!(ret, "    out.c{i} = o_col[{i}];");
                        }
                    }
                    s.push_str("}\n\n");
                    ret.push_str("    return out;\n");
                }
                let mut entry = format!(
                    "@fragment\nfn ps_main({}){} {{\n",
                    if has_input { "input: PsIn" } else { "" },
                    if cols != 0 { " -> PsOut" } else { "" }
                );
                for i in 0..16 {
                    if reads & (1 << i) != 0 {
                        let _ = writeln!(entry, "    r[{i}] = input.i{i};");
                    }
                }
                if let Some(p) = pg {
                    entry.push_str(&param_gen_wgsl(p));
                }
                (entry, ret)
            }
        };
        s.push_str(&entry);
        let _ = write!(
            s,
            "    var pc: u32 = 0u;
    var steps: u32 = 0u;
    loop {{
        if (pc >= {cf_count}u || steps >= {WATCHDOG_STEPS}u) {{
            break;
        }}
        steps += 1u;
        switch pc {{
{}        default: {{
            pc = 0xFFFFFFFFu;
        }}
        }}
    }}
{ret}}}
",
            self.body
        );
        s
    }
}

/// Words to load and the per-component f32 expressions (over `w0..w3`) for
/// a vertex fetch's format.
fn vertex_format_decode(f: &VertexFetch) -> Result<(u32, Vec<String>), Error> {
    type F = VertexFormat;
    let fields: &[(u32, u32, u32)] = match f.format {
        F::F8_8_8_8 => &[(0, 0, 8), (0, 8, 8), (0, 16, 8), (0, 24, 8)],
        F::F2_10_10_10 => &[(0, 0, 10), (0, 10, 10), (0, 20, 10), (0, 30, 2)],
        F::F10_11_11 => &[(0, 0, 11), (0, 11, 11), (0, 22, 10)],
        F::F11_11_10 => &[(0, 0, 10), (0, 10, 11), (0, 21, 11)],
        F::F16_16 => &[(0, 0, 16), (0, 16, 16)],
        F::F16_16_16_16 => &[(0, 0, 16), (0, 16, 16), (1, 0, 16), (1, 16, 16)],
        F::F16_16_FLOAT | F::F16_16_16_16_FLOAT => {
            let n = f.format.component_count().unwrap_or(0);
            let comps = (0..n)
                .map(|i| format!("half(extractBits(w{}, {}u, 16u))", i / 2, 16 * (i % 2)))
                .collect();
            return Ok((n / 2, comps));
        }
        F::F32_FLOAT | F::F32_32_FLOAT | F::F32_32_32_FLOAT | F::F32_32_32_32_FLOAT => {
            let n = f.format.component_count().unwrap_or(0);
            return Ok((n, (0..n).map(|i| format!("bitcast<f32>(w{i})")).collect()));
        }
        F::F32 | F::F32_32 | F::F32_32_32_32 => {
            let n = f.format.component_count().unwrap_or(0);
            let comps = (0..n)
                .map(|i| {
                    let x = if f.signed {
                        format!("f32(bitcast<i32>(w{i}))")
                    } else {
                        format!("f32(w{i})")
                    };
                    match (f.normalized, f.signed, f.signed_rf_no_zero) {
                        (false, _, _) => x,
                        (true, false, _) => format!("{x} * {}", flit(1.0 / 4294967295.0)),
                        (true, true, false) => format!("{x} * {}", flit(1.0 / 2147483647.0)),
                        (true, true, true) => format!(
                            "{x} * {} + {}",
                            flit(1.0 / 2147483647.5),
                            flit(0.5 / 2147483647.5)
                        ),
                    }
                })
                .collect();
            return Ok((n, comps));
        }
        other => return Err(Error::Unsupported(format!("vertex format {}", other.0))),
    };
    let words = fields.iter().map(|&(w, _, _)| w + 1).max().unwrap_or(1);
    let comps = fields
        .iter()
        .map(|&(w, off, width)| {
            let x = if f.signed {
                format!("f32(extractBits(bitcast<i32>(w{w}), {off}u, {width}u))")
            } else {
                format!("f32(extractBits(w{w}, {off}u, {width}u))")
            };
            if !f.normalized {
                return x;
            }
            if !f.signed {
                return format!("{x} * {}", flit(1.0 / ((1u64 << width) - 1) as f32));
            }
            let max = ((1u32 << (width - 1)) - 1) as f32;
            if f.signed_rf_no_zero {
                let s = 1.0 / (max + 0.5);
                format!("{x} * {} + {}", flit(s), flit(0.5 * s))
            } else {
                format!("max({x} * {}, -1.0)", flit(1.0 / max))
            }
        })
        .collect();
    Ok((words, comps))
}
