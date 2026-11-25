//! Readable listing of a shader program, for diagnostics.
//!
//! The listing walks the control-flow program the way the translator does
//! (the control-flow area ends at the first slot an exec points to) and
//! prints each exec's ALU and fetch instructions under it, with every field
//! that changes results. It never fails: words that do not decode are shown
//! with the error.

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::ucode::*;
use crate::ShaderStage;

/// Output of [`disassemble`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Disassembly {
    pub text: String,
    /// Float constants read with an absolute index (0..=255 in the stage's
    /// own numbering), ascending.
    pub float_consts: Vec<u32>,
    /// Some float constant is read relative to a0 or aL.
    pub float_consts_relative: bool,
    /// Every decoded fetch instruction, in slot order.
    pub fetches: Vec<FetchInstr>,
}

const COMP: [char; 4] = ['x', 'y', 'z', 'w'];

/// Lists `words` as a program of `stage`.
pub fn disassemble(words: &[u32], stage: ShaderStage) -> Disassembly {
    let mut d = Dis {
        stage,
        out: String::new(),
        consts: BTreeSet::new(),
        relative: false,
    };
    let slots = (words.len() / 3) as u32;
    let _ = writeln!(d.out, "; {stage:?} shader, {} words", words.len());
    if words.len() % 3 != 0 {
        let _ = writeln!(d.out, "; warning: length is not a multiple of 3 words");
    }
    let pair = |i: u32| {
        let i = i as usize * 3;
        decode_cf_pair([words[i], words[i + 1], words[i + 2]])
    };
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
    let mut fetches = Vec::new();
    for (k, cf) in (0..bound).flat_map(pair).enumerate() {
        let _ = writeln!(d.out, "{k:4}  {}", cf_text(&cf));
        let CfInstr::Exec(e) = cf else { continue };
        for j in 0..e.count {
            let s = e.address + j;
            if s >= slots {
                let _ = writeln!(d.out, "      {s:4}  <past the end>");
                continue;
            }
            let w = [0, 1, 2].map(|n| words[s as usize * 3 + n]);
            let fetch = (e.sequence >> (2 * j)) & 1 != 0;
            let serialize = if (e.sequence >> (2 * j + 1)) & 1 != 0 {
                " serialize"
            } else {
                ""
            };
            let text = match decode_instr(w, fetch) {
                Ok(Instr::Alu(a)) => d.alu(&a),
                Ok(Instr::Fetch(f)) => {
                    fetches.push((s, f));
                    fetch_text(&f)
                }
                Err(err) => format!("<{err}>"),
            };
            let _ = writeln!(
                d.out,
                "      {s:4}  {text}{serialize}  ; {:08x} {:08x} {:08x}",
                w[0], w[1], w[2]
            );
        }
    }
    let list: Vec<String> = d.consts.iter().map(|c| format!("c{c}")).collect();
    let _ = writeln!(
        d.out,
        "; float constants read: {}{}",
        if list.is_empty() {
            "none".into()
        } else {
            list.join(" ")
        },
        if d.relative {
            " (and relative reads)"
        } else {
            ""
        }
    );
    Disassembly {
        text: d.out,
        float_consts: d.consts.into_iter().collect(),
        float_consts_relative: d.relative,
        fetches: {
            // Slot order; a slot run by two execs is listed once.
            fetches.sort_by_key(|&(s, _)| s);
            fetches.dedup_by_key(|&mut (s, _)| s);
            fetches.into_iter().map(|(_, f)| f).collect()
        },
    }
}

struct Dis {
    stage: ShaderStage,
    out: String,
    consts: BTreeSet<u32>,
    relative: bool,
}

fn pred(p: Option<bool>) -> &'static str {
    match p {
        Some(true) => "(p0) ",
        Some(false) => "(!p0) ",
        None => "",
    }
}

fn branch(c: BranchCond) -> String {
    match c {
        BranchCond::Always => String::new(),
        BranchCond::Bool { index, value } => format!(" if b{index} == {value}"),
        BranchCond::Predicate(v) => format!(" if p0 == {v}"),
    }
}

fn cf_text(cf: &CfInstr) -> String {
    match *cf {
        CfInstr::Nop => "nop".into(),
        CfInstr::Exec(e) => {
            let (name, cond, clean) = match e.cond {
                ExecCond::Always { predicate_clean } => ("exec", String::new(), predicate_clean),
                ExecCond::Bool {
                    index,
                    value,
                    predicate_clean,
                } => ("cexec", format!(" if b{index} == {value}"), predicate_clean),
                ExecCond::Predicate {
                    value,
                    predicate_clean,
                } => ("exec", format!(" if p0 == {value}"), predicate_clean),
            };
            format!(
                "{name}{}{cond} addr {} count {}{}{}",
                if e.end { "e" } else { "" },
                e.address,
                e.count,
                if clean { " predicate_clean" } else { "" },
                if e.is_yield { " yield" } else { "" }
            )
        }
        CfInstr::LoopStart {
            address,
            repeat,
            loop_id,
        } => format!(
            "loop_start i{loop_id} skip_to {address}{}",
            if repeat { " repeat" } else { "" }
        ),
        CfInstr::LoopEnd {
            address,
            loop_id,
            predicated_break,
        } => format!(
            "loop_end i{loop_id} back_to {address}{}",
            predicated_break.map_or(String::new(), |v| format!(" break if p0 == {v}"))
        ),
        CfInstr::CondCall { address, cond } => format!("call {address}{}", branch(cond)),
        CfInstr::Return => "ret".into(),
        CfInstr::CondJmp {
            address,
            cond,
            direction,
        } => format!(
            "jmp {address}{}{}",
            branch(cond),
            if direction { " backward" } else { "" }
        ),
        CfInstr::Alloc {
            size,
            alloc_type,
            unserialized,
        } => format!(
            "alloc type {alloc_type} size {size}{}",
            if unserialized { " unserialized" } else { "" }
        ),
        CfInstr::MarkVsFetchDone => "mark_vs_fetch_done".into(),
    }
}

/// `CamelCase` debug name as `snake_case`.
fn snake(name: String) -> String {
    let mut s = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            s.push('_');
        }
        s.push(c.to_ascii_lowercase());
    }
    s
}

fn mask(m: u32) -> String {
    (0..4)
        .map(|i| if m & (1 << i) != 0 { COMP[i] } else { '_' })
        .collect()
}

/// Fetch destination swizzle: 3 bits per component.
fn fetch_swizzle(s: u32) -> String {
    (0..4)
        .map(|i| b"xyzw01?_"[((s >> (3 * i)) & 7) as usize] as char)
        .collect()
}

fn reg(r: u32, rel: bool) -> String {
    if rel {
        format!("r[{r}+aL]")
    } else {
        format!("r{r}")
    }
}

fn format_name(f: VertexFormat) -> String {
    let name = match f {
        VertexFormat::F8_8_8_8 => "8_8_8_8",
        VertexFormat::F2_10_10_10 => "2_10_10_10",
        VertexFormat::F10_11_11 => "10_11_11",
        VertexFormat::F11_11_10 => "11_11_10",
        VertexFormat::F16_16 => "16_16",
        VertexFormat::F16_16_16_16 => "16_16_16_16",
        VertexFormat::F16_16_FLOAT => "16_16_FLOAT",
        VertexFormat::F16_16_16_16_FLOAT => "16_16_16_16_FLOAT",
        VertexFormat::F32 => "32",
        VertexFormat::F32_32 => "32_32",
        VertexFormat::F32_32_32_32 => "32_32_32_32",
        VertexFormat::F32_FLOAT => "32_FLOAT",
        VertexFormat::F32_32_FLOAT => "32_32_FLOAT",
        VertexFormat::F32_32_32_32_FLOAT => "32_32_32_32_FLOAT",
        VertexFormat::F32_32_32_FLOAT => "32_32_32_FLOAT",
        _ => "unknown",
    };
    format!("{name}({})", f.0)
}

/// One fetch instruction with all its fields.
pub fn fetch_text(f: &FetchInstr) -> String {
    match f {
        FetchInstr::Vertex(v) => format!(
            "{}vfetch{} {}.{}, {}.{}, vf{} format {} {} {}{}{} exp_adjust {} stride {} offset {} prefetch {}",
            pred(v.predicate),
            if v.mini { "_mini" } else { "_full" },
            reg(v.dest, v.dest_relative),
            fetch_swizzle(v.dest_swizzle),
            reg(v.src, v.src_relative),
            COMP[v.src_swizzle as usize & 3],
            v.fetch_const,
            format_name(v.format),
            if v.signed { "signed" } else { "unsigned" },
            if v.normalized { "normalized" } else { "integer" },
            if v.signed_rf_no_zero { " rf_no_zero" } else { "" },
            if v.index_rounded { " index_rounded" } else { "" },
            v.exp_adjust,
            v.stride,
            v.offset,
            v.prefetch_count,
        ),
        FetchInstr::Texture(t) => {
            let (dim, n) = match t.dim {
                TextureDim::D1 => ("1d", 1),
                TextureDim::D2 => ("2d", 2),
                TextureDim::D3OrStacked => ("3d", 3),
                TextureDim::Cube => ("cube", 3),
            };
            let src: String = (0..n)
                .map(|i| COMP[((t.src_swizzle >> (2 * i)) & 3) as usize])
                .collect();
            format!(
                "{}{}{dim} {}.{}, {}.{src}, tf{} filter mag {} min {} mip {} aniso {} arbitrary {} vol_mag {} vol_min {}{}{}{}{}{}{} lod_bias {} offset {} {} {}",
                pred(t.predicate),
                match t.op {
                    TextureOp::Fetch => "tfetch".into(),
                    op => snake(format!("{op:?}")),
                },
                reg(t.dest, t.dest_relative),
                fetch_swizzle(t.dest_swizzle),
                reg(t.src, t.src_relative),
                t.slot,
                t.mag_filter,
                t.min_filter,
                t.mip_filter,
                t.aniso_filter,
                t.arbitrary_filter,
                t.vol_mag_filter,
                t.vol_min_filter,
                if t.use_computed_lod { " computed_lod" } else { "" },
                if t.use_register_lod { " register_lod" } else { "" },
                if t.use_register_gradients { " register_gradients" } else { "" },
                if t.sample_at_center { " at_center" } else { "" },
                if t.unnormalized { " unnormalized" } else { "" },
                if t.fetch_valid_only { " valid_only" } else { "" },
                f32::from(t.lod_bias as i8) / 16.0,
                t.offset[0],
                t.offset[1],
                t.offset[2],
            )
        }
    }
}

impl Dis {
    fn export_name(&self, r: u32) -> String {
        match (self.stage, r) {
            (ShaderStage::Vertex, 62) => "oPos".into(),
            (ShaderStage::Vertex, 63) => "oPts".into(),
            (ShaderStage::Vertex, 0..=15) => format!("o{r}"),
            (ShaderStage::Pixel, 0..=3) => format!("oC{r}"),
            (ShaderStage::Pixel, 61) => "oDepth".into(),
            (_, 32) => "eA".into(),
            (_, 33..=37) => format!("eM{}", r - 33),
            _ => format!("export{r}"),
        }
    }

    /// A constant operand; records it.
    fn constant(&mut self, a: &AluInstr, index: u32, operand: usize) -> String {
        let s = if a.src_const_relative(operand) {
            self.relative = true;
            let off = if a.const_relative_a0 { "a0" } else { "aL" };
            format!("c[{index}+{off}]")
        } else {
            self.consts.insert(index);
            format!("c{index}")
        };
        if a.abs_constants {
            format!("|{s}|")
        } else {
            s
        }
    }

    /// Operand `i` (1-based) with its swizzle, negate and abs.
    fn src(&mut self, a: &AluInstr, i: usize) -> String {
        let s = a.src[i - 1];
        let base = if s.is_temp {
            let r = reg(s.reg & 0x3F, s.reg & 0x40 != 0);
            if s.reg & 0x80 != 0 {
                format!("|{r}|")
            } else {
                r
            }
        } else {
            self.constant(a, s.reg, i)
        };
        let sw: String = (0..4)
            .map(|c| COMP[alu_swizzle_component(s.swizzle, c) as usize])
            .collect();
        format!(
            "{}{base}{}",
            if s.negate { "-" } else { "" },
            if sw == "xyzw" {
                String::new()
            } else {
                format!(".{sw}")
            }
        )
    }

    fn alu(&mut self, a: &AluInstr) -> String {
        use ScalarOp as S;
        use VectorOp as V;
        let dest = |d: &Self, r: u32, rel: bool| {
            if a.export {
                d.export_name(r)
            } else {
                reg(r, rel)
            }
        };
        let side_effect = matches!(
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
        let mut parts = Vec::new();
        if a.vector_write_mask != 0 || side_effect {
            let srcs: Vec<String> = (1..=a.vector_op.operand_count())
                .map(|i| self.src(a, i))
                .collect();
            parts.push(format!(
                "{}{} {}.{}, {}",
                snake(format!("{:?}", a.vector_op)),
                if a.vector_saturate { "_sat" } else { "" },
                dest(self, a.vector_dest, a.vector_dest_relative),
                mask(a.vector_write_mask),
                srcs.join(", ")
            ));
        }
        if a.scalar_op != S::RetainPrev || a.scalar_write_mask != 0 {
            let srcs = match a.scalar_op.operand_count() {
                0 => String::new(),
                1 => self.src(a, 3),
                _ => {
                    let s = a.src[2];
                    let c = self.constant(a, s.reg, 3);
                    let n = if s.negate { "-" } else { "" };
                    format!(
                        "{n}{c}.{}, {n}r{}.{}",
                        COMP[alu_swizzle_component(s.swizzle, 3) as usize],
                        a.scalar_const_op_temp_reg(),
                        COMP[alu_swizzle_component(s.swizzle, 0) as usize]
                    )
                }
            };
            // Exports write the scalar result to the vector destination.
            let (r, rel) = if a.export {
                (a.vector_dest, false)
            } else {
                (a.scalar_dest, a.scalar_dest_relative)
            };
            parts.push(format!(
                "{}{} {}.{}{}{srcs}",
                snake(format!("{:?}", a.scalar_op)),
                if a.scalar_saturate { "_sat" } else { "" },
                dest(self, r, rel),
                mask(a.scalar_write_mask),
                if srcs.is_empty() { "" } else { ", " }
            ));
        }
        if parts.is_empty() {
            parts.push("nop".into());
        }
        let zero = if a.export && a.scalar_dest_relative {
            " zero_unwritten"
        } else {
            ""
        };
        format!("{}{}{zero}", pred(a.predicate), parts.join(" + "))
    }
}
