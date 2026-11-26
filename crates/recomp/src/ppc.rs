//! Xbox 360 CPU instruction -> Rust translation.
//!
//! The CPU is a 64-bit big-endian PowerPC with AltiVec (VMX) and the
//! console's VMX128 extensions (128 vector registers and extra opcodes).
//! Each instruction becomes a few Rust statements that read and write a
//! `Context` (`c`) and guest memory (`rt.mem`), calling small helpers in
//! `xenon_rt::{ops, fpu, vmx}` where the semantics are not a one-liner.
//!
//! Emitted code assumes these names are in scope: `rt: &Runtime`,
//! `c: &mut Context`, a mutable `pc: u32` loop variable, and the modules
//! `ops`, `fpu`, `vmx` from `xenon_rt`. See [`crate::emit::function`].
//!
//! The shape (one statement string per instruction, local branches set `pc`,
//! everything else goes through `rt.call`) is carried over from the authors'
//! earlier 32-bit PowerPC translator.

use std::fmt::Write as _;

/// The function being emitted: local branch targets stay inside it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Func {
    pub start: u32,
    pub end: u32,
}

impl Func {
    pub(crate) fn local(&self, t: u32) -> bool {
        (self.start..self.end).contains(&t)
    }
}

/// One translated instruction.
#[derive(Clone, Debug, Default)]
pub struct Ins {
    /// Mnemonic, for coverage reports and comments.
    pub name: &'static str,
    /// Rust statements.
    pub code: String,
    /// Local branch targets (block starts).
    pub targets: Vec<u32>,
    /// Control never falls through to the next instruction.
    pub ends: bool,
    /// The translator does not know this instruction; `code` stops at run time.
    pub unknown: bool,
}

fn simm(w: u32) -> i64 {
    w as u16 as i16 as i64
}

/// 64-bit effective address of a D-form access, as an expression.
fn ea_d(ra: u32, d: i64) -> String {
    if ra == 0 {
        format!("{:#x}u64", d as u64)
    } else if d == 0 {
        format!("c.r[{ra}]")
    } else {
        format!("c.r[{ra}].wrapping_add({:#x})", d as u64)
    }
}

/// 64-bit effective address of an X-form access.
fn ea_x(ra: u32, rb: u32) -> String {
    if ra == 0 {
        format!("c.r[{rb}]")
    } else {
        format!("c.r[{ra}].wrapping_add(c.r[{rb}])")
    }
}

/// Condition-register field mask for `mtcrf` / `mtfsf` style field masks.
fn field_mask(fm: u32) -> u32 {
    (0..8)
        .filter(|i| fm >> (7 - i) & 1 != 0)
        .fold(0, |m, i| m | 0xF << (28 - 4 * i))
}

/// `bc` condition: (CTR update, condition expression).
fn bc_cond(bo: u32, bi: u32) -> (String, String) {
    let mut pre = String::new();
    let mut parts = Vec::new();
    if bo & 4 == 0 {
        // The test sees only CTR's low word, as on the console: a count made
        // by negating a zero-extended word has its upper half all ones.
        pre.push_str("c.ctr = c.ctr.wrapping_sub(1); ");
        parts.push(if bo & 2 != 0 {
            "c.ctr as u32 == 0".to_string()
        } else {
            "c.ctr as u32 != 0".to_string()
        });
    }
    if bo & 0x10 == 0 {
        parts.push(if bo & 8 != 0 {
            format!("c.bit({bi})")
        } else {
            format!("!c.bit({bi})")
        });
    }
    let cond = if parts.is_empty() {
        "true".to_string()
    } else {
        parts.join(" && ")
    };
    (pre, cond)
}

fn call(target: &str) -> String {
    format!("{{ let t: u32 = {target}; rt.call(c, t); }}")
}

/// Branch to `t` without link: a local jump, or a tail call out of the function.
fn goto(f: &Func, t: u32, targets: &mut Vec<u32>) -> String {
    if f.local(t) || t == f.end {
        targets.push(t);
        format!("pc = {t:#010x}; continue;")
    } else {
        format!("{} return;", call(&format!("{t:#010x}")))
    }
}

/// VMX128 register fields.
fn vx128_d(w: u32) -> u32 {
    (w >> 21 & 31) | (w >> 2 & 3) << 5
}
fn vx128_a(w: u32) -> u32 {
    (w >> 16 & 31) | (w >> 5 & 1) << 5 | (w >> 10 & 1) << 6
}
fn vx128_b(w: u32) -> u32 {
    (w >> 11 & 31) | (w & 3) << 5
}

/// How a vector helper is called.
#[derive(Clone, Copy)]
enum V {
    /// `vD = f(vA, vB)`
    Bin,
    /// `vD = f(vA, vB, &mut vscr)`
    BinSat,
    /// `vD = f(vB)`
    Un,
    /// `vD = f(vB, uimm)` with the immediate in the vA field
    UImm,
    /// `vD = f(vB, uimm, &mut vscr)`
    UImmSat,
    /// `vD = f(simm)` with the signed immediate in the vA field
    SImm,
}

fn vop(name: &'static str, kind: V, d: u32, a: u32, b: u32, imm: u32) -> String {
    let f = name.trim_end_matches("128");
    match kind {
        V::Bin => format!("c.v[{d}] = vmx::{f}(c.v[{a}], c.v[{b}]);"),
        V::BinSat => format!("c.v[{d}] = vmx::{f}(c.v[{a}], c.v[{b}], &mut c.vscr);"),
        V::Un => format!("c.v[{d}] = vmx::{f}(c.v[{b}]);"),
        V::UImm => format!("c.v[{d}] = vmx::{f}(c.v[{b}], {imm});"),
        V::UImmSat => format!("c.v[{d}] = vmx::{f}(c.v[{b}], {imm}, &mut c.vscr);"),
        V::SImm => format!("c.v[{d}] = vmx::{f}({});", ((imm as i32) << 27) >> 27),
    }
}

/// Standard VMX VX-form operations by their 11-bit extended opcode.
fn vx_op(xo: u32) -> Option<(&'static str, V)> {
    use V::*;
    Some(match xo {
        0 => ("vaddubm", Bin),
        2 => ("vmaxub", Bin),
        4 => ("vrlb", Bin),
        8 => ("vmuloub", Bin),
        10 => ("vaddfp", Bin),
        12 => ("vmrghb", Bin),
        14 => ("vpkuhum", Bin),
        64 => ("vadduhm", Bin),
        66 => ("vmaxuh", Bin),
        68 => ("vrlh", Bin),
        72 => ("vmulouh", Bin),
        74 => ("vsubfp", Bin),
        76 => ("vmrghh", Bin),
        78 => ("vpkuwum", Bin),
        128 => ("vadduwm", Bin),
        130 => ("vmaxuw", Bin),
        132 => ("vrlw", Bin),
        140 => ("vmrghw", Bin),
        142 => ("vpkuhus", BinSat),
        206 => ("vpkuwus", BinSat),
        258 => ("vmaxsb", Bin),
        260 => ("vslb", Bin),
        264 => ("vmulosb", Bin),
        266 => ("vrefp", Un),
        268 => ("vmrglb", Bin),
        270 => ("vpkshus", BinSat),
        322 => ("vmaxsh", Bin),
        324 => ("vslh", Bin),
        328 => ("vmulosh", Bin),
        330 => ("vrsqrtefp", Un),
        332 => ("vmrglh", Bin),
        334 => ("vpkswus", BinSat),
        384 => ("vaddcuw", Bin),
        386 => ("vmaxsw", Bin),
        388 => ("vslw", Bin),
        394 => ("vexptefp", Un),
        396 => ("vmrglw", Bin),
        398 => ("vpkshss", BinSat),
        452 => ("vsl", Bin),
        458 => ("vlogefp", Un),
        462 => ("vpkswss", BinSat),
        512 => ("vaddubs", BinSat),
        514 => ("vminub", Bin),
        516 => ("vsrb", Bin),
        520 => ("vmuleub", Bin),
        522 => ("vrfin", Un),
        524 => ("vspltb", UImm),
        526 => ("vupkhsb", Un),
        576 => ("vadduhs", BinSat),
        578 => ("vminuh", Bin),
        580 => ("vsrh", Bin),
        584 => ("vmuleuh", Bin),
        586 => ("vrfiz", Un),
        588 => ("vsplth", UImm),
        590 => ("vupkhsh", Un),
        640 => ("vadduws", BinSat),
        642 => ("vminuw", Bin),
        644 => ("vsrw", Bin),
        650 => ("vrfip", Un),
        652 => ("vspltw", UImm),
        654 => ("vupklsb", Un),
        708 => ("vsr", Bin),
        714 => ("vrfim", Un),
        718 => ("vupklsh", Un),
        768 => ("vaddsbs", BinSat),
        770 => ("vminsb", Bin),
        772 => ("vsrab", Bin),
        776 => ("vmulesb", Bin),
        778 => ("vcfux", UImm),
        780 => ("vspltisb", SImm),
        782 => ("vpkpx", Bin),
        832 => ("vaddshs", BinSat),
        834 => ("vminsh", Bin),
        836 => ("vsrah", Bin),
        840 => ("vmulesh", Bin),
        842 => ("vcfsx", UImm),
        844 => ("vspltish", SImm),
        846 => ("vupkhpx", Un),
        896 => ("vaddsws", BinSat),
        898 => ("vminsw", Bin),
        900 => ("vsraw", Bin),
        906 => ("vctuxs", UImmSat),
        908 => ("vspltisw", SImm),
        970 => ("vctsxs", UImmSat),
        974 => ("vupklpx", Un),
        1024 => ("vsububm", Bin),
        1026 => ("vavgub", Bin),
        1028 => ("vand", Bin),
        1034 => ("vmaxfp", Bin),
        1036 => ("vslo", Bin),
        1088 => ("vsubuhm", Bin),
        1090 => ("vavguh", Bin),
        1092 => ("vandc", Bin),
        1098 => ("vminfp", Bin),
        1100 => ("vsro", Bin),
        1152 => ("vsubuwm", Bin),
        1154 => ("vavguw", Bin),
        1156 => ("vor", Bin),
        1220 => ("vxor", Bin),
        1282 => ("vavgsb", Bin),
        1284 => ("vnor", Bin),
        1346 => ("vavgsh", Bin),
        1408 => ("vsubcuw", Bin),
        1410 => ("vavgsw", Bin),
        1536 => ("vsububs", BinSat),
        1544 => ("vsum4ubs", BinSat),
        1600 => ("vsubuhs", BinSat),
        1608 => ("vsum4shs", BinSat),
        1664 => ("vsubuws", BinSat),
        1672 => ("vsum2sws", BinSat),
        1792 => ("vsubsbs", BinSat),
        1800 => ("vsum4sbs", BinSat),
        1856 => ("vsubshs", BinSat),
        1920 => ("vsubsws", BinSat),
        1928 => ("vsumsws", BinSat),
        _ => return None,
    })
}

/// Vector compares by their 10-bit extended opcode (bit 10 is Rc).
fn vc_op(xo: u32) -> Option<&'static str> {
    Some(match xo {
        6 => "vcmpequb",
        70 => "vcmpequh",
        134 => "vcmpequw",
        198 => "vcmpeqfp",
        454 => "vcmpgefp",
        518 => "vcmpgtub",
        582 => "vcmpgtuh",
        646 => "vcmpgtuw",
        710 => "vcmpgtfp",
        774 => "vcmpgtsb",
        838 => "vcmpgtsh",
        902 => "vcmpgtsw",
        966 => "vcmpbfp",
        _ => return None,
    })
}

fn vcmp(name: &'static str, d: u32, a: u32, b: u32, rc: bool) -> String {
    let f = name.trim_end_matches("128");
    if rc {
        let cr = if f == "vcmpbfp" { "cr6_bounds" } else { "cr6" };
        format!("{{ let r = vmx::{f}(c.v[{a}], c.v[{b}]); c.v[{d}] = r; c.set_cr_field(6, vmx::{cr}(r)); }}")
    } else {
        format!("c.v[{d}] = vmx::{f}(c.v[{a}], c.v[{b}]);")
    }
}

/// Vector loads and stores, shared by the op-31 forms and the VMX128 forms.
fn vmem(name: &'static str, v: u32, ea: &str) -> String {
    let base = name.trim_end_matches("128");
    let op = match base {
        "lvxl" | "lvebx" | "lvehx" | "lvewx" => "lvx",
        "stvxl" => "stvx",
        "lvlxl" => "lvlx",
        "lvrxl" => "lvrx",
        "stvlxl" => "stvlx",
        "stvrxl" => "stvrx",
        other => other,
    };
    match op {
        "lvsl" | "lvsr" => format!("c.v[{v}] = vmx::{op}(({ea}) as u32);"),
        "lvx" | "lvlx" | "lvrx" => format!("c.v[{v}] = vmx::{op}(&rt.mem, ({ea}) as u32);"),
        _ => format!("vmx::{op}(&rt.mem, ({ea}) as u32, c.v[{v}]);"),
    }
}

fn vmem_x(xo: u32) -> Option<&'static str> {
    Some(match xo {
        6 => "lvsl",
        38 => "lvsr",
        7 => "lvebx",
        39 => "lvehx",
        71 => "lvewx",
        103 => "lvx",
        359 => "lvxl",
        135 => "stvebx",
        167 => "stvehx",
        199 => "stvewx",
        231 => "stvx",
        487 => "stvxl",
        519 => "lvlx",
        551 => "lvrx",
        647 => "stvlx",
        679 => "stvrx",
        775 => "lvlxl",
        807 => "lvrxl",
        903 => "stvlxl",
        935 => "stvrxl",
        _ => return None,
    })
}

fn vmem_128(key: u32) -> Option<&'static str> {
    Some(match key {
        0x003 => "lvsl128",
        0x043 => "lvsr128",
        0x083 => "lvewx128",
        0x0C3 => "lvx128",
        0x183 => "stvewx128",
        0x1C3 => "stvx128",
        0x2C3 => "lvxl128",
        0x3C3 => "stvxl128",
        0x403 => "lvlx128",
        0x443 => "lvrx128",
        0x503 => "stvlx128",
        0x543 => "stvrx128",
        0x603 => "lvlxl128",
        0x643 => "lvrxl128",
        0x703 => "stvlxl128",
        0x743 => "stvrxl128",
        _ => return None,
    })
}

/// Opcode 4: standard VMX plus the VMX128 loads, stores and `vsldoi128`.
fn op4(w: u32) -> Option<(&'static str, String)> {
    let (d, a, b) = (w >> 21 & 31, w >> 16 & 31, w >> 11 & 31);
    if w & 0x10 != 0 {
        let (d, a, b) = (vx128_d(w), vx128_a(w), vx128_b(w));
        let sh = w >> 6 & 0xF;
        return Some((
            "vsldoi128",
            format!("c.v[{d}] = vmx::vsldoi(c.v[{a}], c.v[{b}], {sh});"),
        ));
    }
    if w & 0x20 != 0 {
        let cc = w >> 6 & 31;
        let s = |f: &str| format!("c.v[{d}] = vmx::{f}(c.v[{a}], c.v[{b}], c.v[{cc}]);");
        let ss =
            |f: &str| format!("c.v[{d}] = vmx::{f}(c.v[{a}], c.v[{b}], c.v[{cc}], &mut c.vscr);");
        return Some(match w & 0x3F {
            32 => ("vmhaddshs", ss("vmhaddshs")),
            33 => ("vmhraddshs", ss("vmhraddshs")),
            34 => ("vmladduhm", s("vmladduhm")),
            36 => ("vmsumubm", s("vmsumubm")),
            37 => ("vmsummbm", s("vmsummbm")),
            38 => ("vmsumuhm", s("vmsumuhm")),
            39 => ("vmsumuhs", ss("vmsumuhs")),
            40 => ("vmsumshm", s("vmsumshm")),
            41 => ("vmsumshs", ss("vmsumshs")),
            42 => ("vsel", s("vsel")),
            43 => ("vperm", s("vperm")),
            44 => (
                "vsldoi",
                format!("c.v[{d}] = vmx::vsldoi(c.v[{a}], c.v[{b}], {});", cc & 15),
            ),
            46 => (
                "vmaddfp",
                format!("c.v[{d}] = vmx::vmaddfp(c.v[{a}], c.v[{cc}], c.v[{b}]);"),
            ),
            47 => (
                "vnmsubfp",
                format!("c.v[{d}] = vmx::vnmsubfp(c.v[{a}], c.v[{cc}], c.v[{b}]);"),
            ),
            _ => return None,
        });
    }
    if w & 3 == 3 {
        let name = vmem_128(w & 0x7F3)?;
        return Some((name, vmem(name, vx128_d(w), &ea_x(a, b))));
    }
    if let Some(name) = vc_op(w & 0x3FF) {
        return Some((name, vcmp(name, d, a, b, w & 0x400 != 0)));
    }
    match w & 0x7FF {
        1540 => Some(("mfvscr", format!("c.v[{d}] = vmx::mfvscr(c.vscr);"))),
        1604 => Some(("mtvscr", format!("c.vscr = c.v[{b}][3];"))),
        xo => {
            let (name, kind) = vx_op(xo)?;
            Some((name, vop(name, kind, d, a, b, a)))
        }
    }
}

/// Opcode 5: VMX128 arithmetic and logic.
fn op5(w: u32) -> Option<(&'static str, String)> {
    let (d, a, b) = (vx128_d(w), vx128_a(w), vx128_b(w));
    if w & 0x210 == 0 {
        let vc = w >> 6 & 7;
        return Some((
            "vperm128",
            format!("c.v[{d}] = vmx::vperm(c.v[{a}], c.v[{b}], c.v[{vc}]);"),
        ));
    }
    use V::*;
    let (name, kind) = match w & 0x3D0 {
        0x010 => ("vaddfp128", Bin),
        0x050 => ("vsubfp128", Bin),
        0x090 => ("vmulfp128", Bin),
        0x0D0 => {
            return Some((
                "vmaddfp128",
                format!("c.v[{d}] = vmx::vmaddfp(c.v[{a}], c.v[{b}], c.v[{d}]);"),
            ))
        }
        0x110 => {
            return Some((
                "vmaddcfp128",
                format!("c.v[{d}] = vmx::vmaddfp(c.v[{a}], c.v[{d}], c.v[{b}]);"),
            ))
        }
        0x150 => {
            return Some((
                "vnmsubfp128",
                format!("c.v[{d}] = vmx::vnmsubfp(c.v[{a}], c.v[{b}], c.v[{d}]);"),
            ))
        }
        0x190 => ("vmsum3fp128", Bin),
        0x1D0 => ("vmsum4fp128", Bin),
        0x200 => ("vpkshss128", BinSat),
        0x210 => ("vand128", Bin),
        0x240 => ("vpkshus128", BinSat),
        0x250 => ("vandc128", Bin),
        0x280 => ("vpkswss128", BinSat),
        0x290 => ("vnor128", Bin),
        0x2C0 => ("vpkswus128", BinSat),
        0x2D0 => ("vor128", Bin),
        0x300 => ("vpkuhum128", Bin),
        0x310 => ("vxor128", Bin),
        0x340 => ("vpkuhus128", BinSat),
        0x350 => {
            return Some((
                "vsel128",
                format!("c.v[{d}] = vmx::vsel(c.v[{a}], c.v[{b}], c.v[{d}]);"),
            ))
        }
        0x380 => ("vpkuwum128", Bin),
        0x390 => ("vslo128", Bin),
        0x3C0 => ("vpkuwus128", BinSat),
        0x3D0 => ("vsro128", Bin),
        _ => return None,
    };
    Some((name, vop(name, kind, d, a, b, 0)))
}

/// Opcode 6: VMX128 compares, shifts, conversions, permutes and D3D packing.
fn op6(w: u32) -> Option<(&'static str, String)> {
    use V::*;
    let (d, a, b) = (vx128_d(w), vx128_a(w), vx128_b(w));
    let imm = w >> 16 & 31;
    if w & 0x10 == 0 {
        let hit = match w & 0x3D0 {
            0x280 => Some(("vmaxfp128", Bin)),
            0x2C0 => Some(("vminfp128", Bin)),
            0x300 => Some(("vmrghw128", Bin)),
            0x340 => Some(("vmrglw128", Bin)),
            0x380 => Some(("vupkhsb128", Un)),
            0x3C0 => Some(("vupklsb128", Un)),
            _ => None,
        };
        if let Some((name, kind)) = hit {
            return Some((name, vop(name, kind, d, a, b, 0)));
        }
        let name = match w & 0x390 {
            0x000 => "vcmpeqfp128",
            0x080 => "vcmpgefp128",
            0x100 => "vcmpgtfp128",
            0x180 => "vcmpbfp128",
            0x200 => "vcmpequw128",
            _ => return None,
        };
        return Some((name, vcmp(name, d, a, b, w & 0x40 != 0)));
    }
    if w & 0x200 == 0 {
        let name = match w & 0x3D0 {
            0x050 => "vrlw128",
            0x0D0 => "vslw128",
            0x150 => "vsraw128",
            0x1D0 => "vsrw128",
            _ => return None,
        };
        return Some((name, vop(name, Bin, d, a, b, 0)));
    }
    let exact = |key: u32| -> Option<(&'static str, V)> {
        Some(match key {
            0x230 => ("vctsxs", UImmSat),
            0x270 => ("vctuxs", UImmSat),
            0x2B0 => ("vcfsx", UImm),
            0x2F0 => ("vcfux", UImm),
            0x330 => ("vrfim128", Un),
            0x370 => ("vrfin128", Un),
            0x3B0 => ("vrfip128", Un),
            0x3F0 => ("vrfiz128", Un),
            0x630 => ("vrefp128", Un),
            0x670 => ("vrsqrtefp128", Un),
            0x6B0 => ("vexptefp128", Un),
            0x6F0 => ("vlogefp128", Un),
            0x730 => ("vspltw128", UImm),
            0x770 => ("vspltisw128", SImm),
            _ => return None,
        })
    };
    let names128 = |n: &'static str| match n {
        "vctsxs" => "vcfpsxws128",
        "vctuxs" => "vcfpuxws128",
        "vcfsx" => "vcsxwfp128",
        "vcfux" => "vcuxwfp128",
        n => n,
    };
    if w & 0x20 != 0 {
        if w & 0x7F0 == 0x7F0 {
            let typ = imm >> 2;
            return Some((
                "vupkd3d128",
                format!(
                    "c.v[{d}] = vmx::vupkd3d(c.v[{b}], {typ}).unwrap_or_else(|| rt.unknown(c, pc, {w:#010x}));"
                ),
            ));
        }
        let (fname, kind) = exact(w & 0x7F0)?;
        let code = vop(fname, kind, d, 0, b, imm);
        return Some((names128(fname), code));
    }
    if w & 0x400 == 0 {
        let perm = imm | (w >> 6 & 7) << 5;
        return Some((
            "vpermwi128",
            format!("c.v[{d}] = vmx::vpermwi(c.v[{b}], {perm:#x});"),
        ));
    }
    let z = w >> 6 & 3;
    match w & 0x730 {
        0x610 => {
            let (typ, pack) = (imm >> 2, imm & 3);
            Some((
                "vpkd3d128",
                format!(
                    "c.v[{d}] = vmx::vpkd3d(c.v[{d}], c.v[{b}], {typ}, {pack}, {z}).unwrap_or_else(|| rt.unknown(c, pc, {w:#010x}));"
                ),
            ))
        }
        0x710 => Some((
            "vrlimi128",
            format!("c.v[{d}] = vmx::vrlimi(c.v[{d}], c.v[{b}], {imm}, {z});"),
        )),
        _ => None,
    }
}

/// Translates the instruction word `w` at guest address `pc`.
pub fn translate(w: u32, pc: u32, func: &Func) -> Ins {
    let op = w >> 26;
    let d = w >> 21 & 31;
    let a = w >> 16 & 31;
    let b = w >> 11 & 31;
    let cc = w >> 6 & 31;
    let rc = w & 1 != 0;
    let xo10 = w >> 1 & 0x3FF;
    let next = pc.wrapping_add(4);
    let si = simm(w);
    let ui = (w & 0xFFFF) as u64;
    let mut targets = Vec::new();
    let mut ends = false;

    let rec_a = if rc {
        format!(" c.rec(c.r[{a}]);")
    } else {
        String::new()
    };
    let rec_d = if rc {
        format!(" c.rec(c.r[{d}]);")
    } else {
        String::new()
    };
    let rec_f = if rc {
        " c.set_cr_field(1, c.fpscr >> 28);".to_string()
    } else {
        String::new()
    };

    let res: Option<(&'static str, String)> = match op {
        2 => Some((
            "tdi",
            format!("if ops::trap_cond({d}, c.r[{a}] as i64, {si}) {{ rt.trap(c, {pc:#x}); }}"),
        )),
        3 => Some((
            "twi",
            format!(
                "if ops::trap_cond({d}, c.r[{a}] as i32 as i64, {si}) {{ rt.trap(c, {pc:#x}); }}"
            ),
        )),
        4 => op4(w),
        5 => op5(w),
        6 => op6(w),
        7 => Some((
            "mulli",
            format!("c.r[{d}] = (c.r[{a}] as i64).wrapping_mul({si}) as u64;"),
        )),
        8 => Some((
            "subfic",
            format!(
                "{{ let (v, k) = ops::add_ca(!c.r[{a}], {:#x}, true); c.r[{d}] = v; c.ca = k; }}",
                si as u64
            ),
        )),
        10 => {
            let crf = d >> 2;
            Some(if d & 1 != 0 {
                ("cmpldi", format!("c.cmp_u({crf}, c.r[{a}], {ui:#x});"))
            } else {
                (
                    "cmplwi",
                    format!("c.cmp_u({crf}, c.r[{a}] as u32 as u64, {ui:#x});"),
                )
            })
        }
        11 => {
            let crf = d >> 2;
            Some(if d & 1 != 0 {
                ("cmpdi", format!("c.cmp_s({crf}, c.r[{a}] as i64, {si});"))
            } else {
                (
                    "cmpwi",
                    format!("c.cmp_s({crf}, c.r[{a}] as i32 as i64, {si});"),
                )
            })
        }
        12 | 13 => {
            let r = if op == 13 {
                format!(" c.rec(c.r[{d}]);")
            } else {
                String::new()
            };
            Some((
                if op == 13 { "addic." } else { "addic" },
                format!(
                    "{{ let (v, k) = ops::add_ca(c.r[{a}], {:#x}, false); c.r[{d}] = v; c.ca = k; }}{r}",
                    si as u64
                ),
            ))
        }
        14 => Some(("addi", format!("c.r[{d}] = {};", ea_d(a, si)))),
        15 => Some(("addis", format!("c.r[{d}] = {};", ea_d(a, si << 16)))),
        16 => {
            let bd = (w & 0xFFFC) as u16 as i16 as i32;
            let t = if w & 2 != 0 {
                bd as u32
            } else {
                pc.wrapping_add(bd as u32)
            };
            let (pre, cond) = bc_cond(d, a);
            // LK sets LR whether or not the branch is taken.
            let code = if w & 1 != 0 {
                format!(
                    "{pre}c.lr = {next:#x}; if {cond} {{ {} }}",
                    call(&format!("{t:#010x}"))
                )
            } else {
                if cond == "true" {
                    ends = true;
                }
                format!("{pre}if {cond} {{ {} }}", goto(func, t, &mut targets))
            };
            Some((if w & 1 != 0 { "bcl" } else { "bc" }, code))
        }
        17 => Some(("sc", format!("rt.syscall(c, {pc:#x});"))),
        18 => {
            let li = ((w & 0x03FF_FFFC) << 6) as i32 >> 6;
            let t = if w & 2 != 0 {
                li as u32
            } else {
                pc.wrapping_add(li as u32)
            };
            if w & 1 != 0 {
                Some((
                    "bl",
                    format!("c.lr = {next:#x}; {}", call(&format!("{t:#010x}"))),
                ))
            } else {
                ends = true;
                Some(("b", goto(func, t, &mut targets)))
            }
        }
        19 => op19(w, d, a, b, next, &mut ends),
        20 => {
            let m = xenon_mask32(w);
            Some((
                "rlwimi",
                format!(
                    "c.r[{a}] = (ops::rotl32(c.r[{d}], {b}) & {m:#x}) | (c.r[{a}] & {:#x});{rec_a}",
                    !m
                ),
            ))
        }
        21 => {
            let m = xenon_mask32(w);
            Some((
                "rlwinm",
                format!("c.r[{a}] = ops::rotl32(c.r[{d}], {b}) & {m:#x};{rec_a}"),
            ))
        }
        23 => {
            let m = xenon_mask32(w);
            Some((
                "rlwnm",
                format!("c.r[{a}] = ops::rotl32(c.r[{d}], c.r[{b}] as u32) & {m:#x};{rec_a}"),
            ))
        }
        24 => Some((
            "ori",
            if d == a && ui == 0 {
                String::new()
            } else {
                format!("c.r[{a}] = c.r[{d}] | {ui:#x};")
            },
        )),
        25 => Some(("oris", format!("c.r[{a}] = c.r[{d}] | {:#x};", ui << 16))),
        26 => Some(("xori", format!("c.r[{a}] = c.r[{d}] ^ {ui:#x};"))),
        27 => Some(("xoris", format!("c.r[{a}] = c.r[{d}] ^ {:#x};", ui << 16))),
        28 => Some((
            "andi.",
            format!("c.r[{a}] = c.r[{d}] & {ui:#x}; c.rec(c.r[{a}]);"),
        )),
        29 => Some((
            "andis.",
            format!("c.r[{a}] = c.r[{d}] & {:#x}; c.rec(c.r[{a}]);", ui << 16),
        )),
        30 => op30(w, d, a, b, &rec_a),
        31 => op31(w, pc, d, a, b, xo10, &rec_a, &rec_d),
        32..=55 => load_store_d(op, d, a, si),
        58 => {
            let ds = (w & 0xFFFC) as u16 as i16 as i64;
            match w & 3 {
                0 => Some(("ld", format!("{{ let ea = {}; c.r[{d}] = rt.mem.read_u64(ea as u32); }}", ea_d(a, ds)))),
                1 => Some((
                    "ldu",
                    format!("{{ let ea = {}; c.r[{d}] = rt.mem.read_u64(ea as u32); c.r[{a}] = ea; }}", ea_d(a, ds)),
                )),
                2 => Some((
                    "lwa",
                    format!(
                        "{{ let ea = {}; c.r[{d}] = rt.mem.read_u32(ea as u32) as i32 as i64 as u64; }}",
                        ea_d(a, ds)
                    ),
                )),
                _ => None,
            }
        }
        62 => {
            let ds = (w & 0xFFFC) as u16 as i16 as i64;
            match w & 3 {
                0 => Some((
                    "std",
                    format!(
                        "{{ let ea = {}; rt.mem.write_u64(ea as u32, c.r[{d}]); }}",
                        ea_d(a, ds)
                    ),
                )),
                1 => Some((
                    "stdu",
                    format!(
                        "{{ let ea = {}; rt.mem.write_u64(ea as u32, c.r[{d}]); c.r[{a}] = ea; }}",
                        ea_d(a, ds)
                    ),
                )),
                _ => None,
            }
        }
        59 => {
            let xo5 = w >> 1 & 31;
            let s = |e: String| format!("c.f[{d}] = fpu::round_single({e});{rec_f}");
            let (fa, fb, fc) = (
                format!("c.f[{a}]"),
                format!("c.f[{b}]"),
                format!("c.f[{cc}]"),
            );
            match xo5 {
                18 => Some(("fdivs", s(format!("fpu::div({fa}, {fb})")))),
                20 => Some(("fsubs", s(format!("fpu::sub({fa}, {fb})")))),
                21 => Some(("fadds", s(format!("fpu::add({fa}, {fb})")))),
                22 => Some(("fsqrts", s(format!("fpu::sqrt({fb})")))),
                24 => Some(("fres", format!("c.f[{d}] = fpu::fres({fb});{rec_f}"))),
                25 => Some(("fmuls", s(format!("fpu::mul({fa}, {fc})")))),
                28 => Some(("fmsubs", s(format!("fpu::msub({fa}, {fc}, {fb})")))),
                29 => Some(("fmadds", s(format!("fpu::madd({fa}, {fc}, {fb})")))),
                30 => Some((
                    "fnmsubs",
                    s(format!("fpu::neg_num(fpu::msub({fa}, {fc}, {fb}))")),
                )),
                31 => Some((
                    "fnmadds",
                    s(format!("fpu::neg_num(fpu::madd({fa}, {fc}, {fb}))")),
                )),
                _ => None,
            }
        }
        63 => op63(w, d, a, b, cc, xo10, &rec_f),
        _ => None,
    };

    match res {
        Some((name, code)) => Ins {
            name,
            code,
            targets,
            ends,
            unknown: false,
        },
        None => Ins {
            name: "unknown",
            code: format!("rt.unknown(c, pc, {w:#010x});"),
            targets,
            ends: true,
            unknown: true,
        },
    }
}

fn xenon_mask32(w: u32) -> u64 {
    xenon_rt::ops::mask32(w >> 6 & 31, w >> 1 & 31)
}

fn op19(
    w: u32,
    d: u32,
    a: u32,
    b: u32,
    next: u32,
    ends: &mut bool,
) -> Option<(&'static str, String)> {
    let crop = |name: &'static str, e: &str| {
        Some((
            name,
            format!("{{ let a = c.bit({a}); let b = c.bit({b}); c.set_bit({d}, {e}); }}"),
        ))
    };
    match w >> 1 & 0x3FF {
        0 => Some((
            "mcrf",
            format!(
                "{{ let v = c.cr_field({}); c.set_cr_field({}, v); }}",
                a >> 2,
                d >> 2
            ),
        )),
        16 => {
            let (pre, cond) = bc_cond(d, a);
            if w & 1 != 0 {
                Some((
                    "bclrl",
                    format!("{pre}let t = c.lr as u32; c.lr = {next:#x}; if {cond} {{ rt.call(c, t); }}"),
                ))
            } else {
                if cond == "true" {
                    *ends = true;
                }
                Some(("bclr", format!("{pre}if {cond} {{ return; }}")))
            }
        }
        528 => {
            let (_, cond) = bc_cond(d | 4, a);
            if w & 1 != 0 {
                Some((
                    "bcctrl",
                    format!(
                        "let t = c.ctr as u32; c.lr = {next:#x}; if {cond} {{ rt.call(c, t); }}"
                    ),
                ))
            } else {
                if cond == "true" {
                    *ends = true;
                }
                // Jump tables land on local blocks; the function's fallback arm
                // turns anything else into a tail call.
                Some((
                    "bcctr",
                    format!("if {cond} {{ pc = c.ctr as u32; continue; }}"),
                ))
            }
        }
        33 => crop("crnor", "!(a | b)"),
        129 => crop("crandc", "a & !b"),
        150 => Some((
            "isync",
            "std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);".into(),
        )),
        193 => crop("crxor", "a ^ b"),
        225 => crop("crnand", "!(a & b)"),
        257 => crop("crand", "a & b"),
        289 => crop("creqv", "a == b"),
        417 => crop("crorc", "a | !b"),
        449 => crop("cror", "a | b"),
        _ => None,
    }
}

fn op30(w: u32, s: u32, a: u32, b: u32, rec: &str) -> Option<(&'static str, String)> {
    let sh = (w >> 11 & 31) | (w >> 1 & 1) << 5;
    let mb = (w >> 6 & 31) | (w >> 5 & 1) << 5;
    use xenon_rt::ops::mask64;
    match w >> 2 & 7 {
        0 => Some((
            "rldicl",
            format!(
                "c.r[{a}] = c.r[{s}].rotate_left({sh}) & {:#x};{rec}",
                mask64(mb, 63)
            ),
        )),
        1 => Some((
            "rldicr",
            format!(
                "c.r[{a}] = c.r[{s}].rotate_left({sh}) & {:#x};{rec}",
                mask64(0, mb)
            ),
        )),
        2 => Some((
            "rldic",
            format!(
                "c.r[{a}] = c.r[{s}].rotate_left({sh}) & {:#x};{rec}",
                mask64(mb, 63 - sh)
            ),
        )),
        3 => {
            let m = mask64(mb, 63 - sh);
            Some((
                "rldimi",
                format!(
                    "c.r[{a}] = (c.r[{s}].rotate_left({sh}) & {m:#x}) | (c.r[{a}] & {:#x});{rec}",
                    !m
                ),
            ))
        }
        4 => match w >> 1 & 0xF {
            8 => Some((
                "rldcl",
                format!(
                    "c.r[{a}] = c.r[{s}].rotate_left(c.r[{b}] as u32 & 63) & {:#x};{rec}",
                    mask64(mb, 63)
                ),
            )),
            9 => Some((
                "rldcr",
                format!(
                    "c.r[{a}] = c.r[{s}].rotate_left(c.r[{b}] as u32 & 63) & {:#x};{rec}",
                    mask64(0, mb)
                ),
            )),
            _ => None,
        },
        _ => None,
    }
}

/// D-form loads and stores, opcodes 32..55.
fn load_store_d(op: u32, d: u32, a: u32, si: i64) -> Option<(&'static str, String)> {
    let ea = ea_d(a, si);
    let upd = |u: bool| {
        if u {
            format!(" c.r[{a}] = ea;")
        } else {
            String::new()
        }
    };
    let ld = |name: &'static str, e: &str, u: bool| {
        Some((
            name,
            format!("{{ let ea = {ea}; c.r[{d}] = {e};{} }}", upd(u)),
        ))
    };
    let st = |name: &'static str, e: &str, u: bool| {
        Some((name, format!("{{ let ea = {ea}; {e};{} }}", upd(u))))
    };
    match op {
        32 => ld("lwz", "rt.mem.read_u32(ea as u32) as u64", false),
        33 => ld("lwzu", "rt.mem.read_u32(ea as u32) as u64", true),
        34 => ld("lbz", "rt.mem.read_u8(ea as u32) as u64", false),
        35 => ld("lbzu", "rt.mem.read_u8(ea as u32) as u64", true),
        36 => st("stw", &format!("rt.mem.write_u32(ea as u32, c.r[{d}] as u32)"), false),
        37 => st("stwu", &format!("rt.mem.write_u32(ea as u32, c.r[{d}] as u32)"), true),
        38 => st("stb", &format!("rt.mem.write_u8(ea as u32, c.r[{d}] as u8)"), false),
        39 => st("stbu", &format!("rt.mem.write_u8(ea as u32, c.r[{d}] as u8)"), true),
        40 => ld("lhz", "rt.mem.read_u16(ea as u32) as u64", false),
        41 => ld("lhzu", "rt.mem.read_u16(ea as u32) as u64", true),
        42 => ld("lha", "rt.mem.read_u16(ea as u32) as i16 as i64 as u64", false),
        43 => ld("lhau", "rt.mem.read_u16(ea as u32) as i16 as i64 as u64", true),
        44 => st("sth", &format!("rt.mem.write_u16(ea as u32, c.r[{d}] as u16)"), false),
        45 => st("sthu", &format!("rt.mem.write_u16(ea as u32, c.r[{d}] as u16)"), true),
        46 => Some((
            "lmw",
            format!(
                "{{ let ea = {ea} as u32; for (i, r) in ({d}..32).enumerate() {{ c.r[r] = rt.mem.read_u32(ea.wrapping_add(4 * i as u32)) as u64; }} }}"
            ),
        )),
        47 => Some((
            "stmw",
            format!(
                "{{ let ea = {ea} as u32; for (i, r) in ({d}..32).enumerate() {{ rt.mem.write_u32(ea.wrapping_add(4 * i as u32), c.r[r] as u32); }} }}"
            ),
        )),
        48 | 49 => Some((
            if op == 48 { "lfs" } else { "lfsu" },
            format!("{{ let ea = {ea}; c.f[{d}] = fpu::single_to_double(rt.mem.read_u32(ea as u32));{} }}", upd(op == 49)),
        )),
        50 | 51 => Some((
            if op == 50 { "lfd" } else { "lfdu" },
            format!("{{ let ea = {ea}; c.f[{d}] = f64::from_bits(rt.mem.read_u64(ea as u32));{} }}", upd(op == 51)),
        )),
        52 | 53 => Some((
            if op == 52 { "stfs" } else { "stfsu" },
            format!("{{ let ea = {ea}; rt.mem.write_u32(ea as u32, fpu::double_to_single(c.f[{d}]));{} }}", upd(op == 53)),
        )),
        54 | 55 => Some((
            if op == 54 { "stfd" } else { "stfdu" },
            format!("{{ let ea = {ea}; rt.mem.write_u64(ea as u32, c.f[{d}].to_bits());{} }}", upd(op == 55)),
        )),
        _ => None,
    }
}

/// Arithmetic with an OE (overflow enable) bit, by 9-bit extended opcode.
fn arith(x9: u32) -> Option<&'static str> {
    Some(match x9 {
        8 => "subfc",
        10 => "addc",
        40 => "subf",
        104 => "neg",
        136 => "subfe",
        138 => "adde",
        200 => "subfze",
        202 => "addze",
        232 => "subfme",
        233 => "mulld",
        234 => "addme",
        235 => "mullw",
        266 => "add",
        457 => "divdu",
        459 => "divwu",
        489 => "divd",
        491 => "divw",
        _ => return None,
    })
}

#[allow(clippy::too_many_arguments)]
fn op31(
    w: u32,
    pc: u32,
    d: u32,
    a: u32,
    b: u32,
    xo10: u32,
    rec_a: &str,
    rec_d: &str,
) -> Option<(&'static str, String)> {
    let x9 = xo10 & 0x1FF;
    if let Some(name) = arith(x9) {
        let oe = xo10 & 0x200 != 0;
        let (ra, rb) = (format!("c.r[{a}]"), format!("c.r[{b}]"));
        // (x, y, carry-in expression, writes CA)
        let sum = |x: &str, y: &str, cin: &str, ca: bool| {
            let ov = if oe {
                " c.set_ov(ops::add_ov(x, y, v));"
            } else {
                ""
            };
            let k = if ca { " c.ca = k;" } else { "" };
            format!("{{ let (x, y) = ({x}, {y}); let (v, k) = ops::add_ca(x, y, {cin}); let _ = k; c.r[{d}] = v;{k}{ov} }}{rec_d}")
        };
        let mul_div = |f: &str| {
            let ov = if oe { " c.set_ov(o);" } else { "" };
            format!("{{ let (v, o) = ops::{f}({ra}, {rb}); let _ = o; c.r[{d}] = v;{ov} }}{rec_d}")
        };
        let code = match x9 {
            266 => sum(&ra, &rb, "false", false),
            10 => sum(&ra, &rb, "false", true),
            138 => sum(&ra, &rb, "c.ca", true),
            202 => sum(&ra, "0", "c.ca", true),
            234 => sum(&ra, "u64::MAX", "c.ca", true),
            40 => sum(&format!("!{ra}"), &rb, "true", false),
            8 => sum(&format!("!{ra}"), &rb, "true", true),
            136 => sum(&format!("!{ra}"), &rb, "c.ca", true),
            200 => sum(&format!("!{ra}"), "0", "c.ca", true),
            232 => sum(&format!("!{ra}"), "u64::MAX", "c.ca", true),
            104 => sum(&format!("!{ra}"), "0", "true", false),
            235 => mul_div("mullw"),
            233 => mul_div("mulld"),
            491 => mul_div("divw"),
            459 => mul_div("divwu"),
            489 => mul_div("divd"),
            457 => mul_div("divdu"),
            _ => unreachable!(),
        };
        return Some((name, code));
    }
    let (rs, ra, rb) = (
        format!("c.r[{d}]"),
        format!("c.r[{a}]"),
        format!("c.r[{b}]"),
    );
    let logic = |name: &'static str, e: String| Some((name, format!("c.r[{a}] = {e};{rec_a}")));
    let hi = |name: &'static str, f: &str| {
        Some((name, format!("c.r[{d}] = ops::{f}({ra}, {rb});{rec_d}")))
    };
    let ea = ea_x(a, b);
    let ld = |name: &'static str, e: &str, u: bool| {
        let up = if u {
            format!(" c.r[{a}] = ea;")
        } else {
            String::new()
        };
        Some((name, format!("{{ let ea = {ea}; c.r[{d}] = {e};{up} }}")))
    };
    let st = |name: &'static str, e: String, u: bool| {
        let up = if u {
            format!(" c.r[{a}] = ea;")
        } else {
            String::new()
        };
        Some((name, format!("{{ let ea = {ea}; {e};{up} }}")))
    };
    if let Some(name) = vmem_x(xo10) {
        return Some((name, vmem(name, d, &ea)));
    }
    match xo10 {
        0 => {
            let crf = d >> 2;
            Some(if d & 1 != 0 {
                ("cmpd", format!("c.cmp_s({crf}, {ra} as i64, {rb} as i64);"))
            } else {
                ("cmpw", format!("c.cmp_s({crf}, {ra} as i32 as i64, {rb} as i32 as i64);"))
            })
        }
        32 => {
            let crf = d >> 2;
            Some(if d & 1 != 0 {
                ("cmpld", format!("c.cmp_u({crf}, {ra}, {rb});"))
            } else {
                ("cmplw", format!("c.cmp_u({crf}, {ra} as u32 as u64, {rb} as u32 as u64);"))
            })
        }
        4 => Some((
            "tw",
            format!("if ops::trap_cond({d}, {ra} as i32 as i64, {rb} as i32 as i64) {{ rt.trap(c, {pc:#x}); }}"),
        )),
        68 => Some(("td", format!("if ops::trap_cond({d}, {ra} as i64, {rb} as i64) {{ rt.trap(c, {pc:#x}); }}"))),
        9 => hi("mulhdu", "mulhdu"),
        11 => hi("mulhwu", "mulhwu"),
        73 => hi("mulhd", "mulhd"),
        75 => hi("mulhw", "mulhw"),
        19 => {
            if w >> 20 & 1 != 0 {
                let m = field_mask(w >> 12 & 0xFF);
                Some(("mfocrf", format!("c.r[{d}] = (c.cr & {m:#x}) as u64;")))
            } else {
                Some(("mfcr", format!("c.r[{d}] = c.cr as u64;")))
            }
        }
        144 => {
            let fm = w >> 12 & 0xFF;
            Some((if w >> 20 & 1 != 0 { "mtocrf" } else { "mtcrf" }, format!("c.mtcrf({fm:#x}, {rs} as u32);")))
        }
        83 => Some(("mfmsr", format!("c.r[{d}] = c.msr;"))),
        146 => Some(("mtmsr", format!("c.msr = (c.msr & !0xFFFF_FFFF) | ({rs} & 0xFFFF_FFFF);"))),
        178 => Some(("mtmsrd", format!("c.msr = {rs};"))),
        339 | 467 | 371 => {
            let spr = (w >> 16 & 31) | (w >> 11 & 31) << 5;
            let mf = xo10 != 467;
            let (name, code) = match (mf, spr) {
                (true, 1) => ("mfxer", format!("c.r[{d}] = c.xer();")),
                (true, 8) => ("mflr", format!("c.r[{d}] = c.lr;")),
                (true, 9) => ("mfctr", format!("c.r[{d}] = c.ctr;")),
                (true, 256) => ("mfvrsave", format!("c.r[{d}] = c.vrsave as u64;")),
                (true, 268) => ("mftb", format!("c.r[{d}] = rt.timebase();")),
                (true, 269) => ("mftbu", format!("c.r[{d}] = rt.timebase() >> 32;")),
                (false, 1) => ("mtxer", format!("c.set_xer({rs});")),
                (false, 8) => ("mtlr", format!("c.lr = {rs};")),
                (false, 9) => ("mtctr", format!("c.ctr = {rs};")),
                (false, 256) => ("mtvrsave", format!("c.vrsave = {rs} as u32;")),
                _ => return None,
            };
            Some((name, code))
        }
        24 => logic("slw", format!("ops::slw({rs}, {rb})")),
        536 => logic("srw", format!("ops::srw({rs}, {rb})")),
        27 => logic("sld", format!("ops::sld({rs}, {rb})")),
        539 => logic("srd", format!("ops::srd({rs}, {rb})")),
        792 => Some(("sraw", format!("{{ let (v, k) = ops::sraw({rs}, {rb}); c.r[{a}] = v; c.ca = k; }}{rec_a}"))),
        824 => Some(("srawi", format!("{{ let (v, k) = ops::sraw({rs}, {b}); c.r[{a}] = v; c.ca = k; }}{rec_a}"))),
        794 => Some(("srad", format!("{{ let (v, k) = ops::srad({rs}, {rb}); c.r[{a}] = v; c.ca = k; }}{rec_a}"))),
        826 | 827 => {
            let sh = b | (w >> 1 & 1) << 5;
            Some(("sradi", format!("{{ let (v, k) = ops::srad({rs}, {sh}); c.r[{a}] = v; c.ca = k; }}{rec_a}")))
        }
        26 => logic("cntlzw", format!("({rs} as u32).leading_zeros() as u64")),
        58 => logic("cntlzd", format!("{rs}.leading_zeros() as u64")),
        28 => logic("and", format!("{rs} & {rb}")),
        60 => logic("andc", format!("{rs} & !{rb}")),
        124 => logic("nor", format!("!({rs} | {rb})")),
        284 => logic("eqv", format!("!({rs} ^ {rb})")),
        316 => logic("xor", format!("{rs} ^ {rb}")),
        412 => logic("orc", format!("{rs} | !{rb}")),
        444 => logic("or", format!("{rs} | {rb}")),
        476 => logic("nand", format!("!({rs} & {rb})")),
        922 => logic("extsh", format!("{rs} as i16 as i64 as u64")),
        954 => logic("extsb", format!("{rs} as i8 as i64 as u64")),
        986 => logic("extsw", format!("{rs} as i32 as i64 as u64")),
        20 => Some((
            "lwarx",
            format!(
                "{{ let ea = ({ea}) as u32; let v = rt.mem.read_u32(ea) as u64; c.reserve = Some(xenon_rt::Reservation {{ addr: ea, value: v, size: 4 }}); c.r[{d}] = v; }}"
            ),
        )),
        84 => Some((
            "ldarx",
            format!(
                "{{ let ea = ({ea}) as u32; let v = rt.mem.read_u64(ea); c.reserve = Some(xenon_rt::Reservation {{ addr: ea, value: v, size: 8 }}); c.r[{d}] = v; }}"
            ),
        )),
        150 => Some((
            "stwcx.",
            format!(
                "{{ let ea = ({ea}) as u32; let ok = match c.reserve.take() {{ Some(r) if r.addr == ea && r.size == 4 => rt.mem.cas_u32(ea, r.value as u32, {rs} as u32), _ => false }}; c.set_cr_field(0, (ok as u32) << 1 | c.so as u32); }}"
            ),
        )),
        214 => Some((
            "stdcx.",
            format!(
                "{{ let ea = ({ea}) as u32; let ok = match c.reserve.take() {{ Some(r) if r.addr == ea && r.size == 8 => rt.mem.cas_u64(ea, r.value, {rs}), _ => false }}; c.set_cr_field(0, (ok as u32) << 1 | c.so as u32); }}"
            ),
        )),
        21 => ld("ldx", "rt.mem.read_u64(ea as u32)", false),
        53 => ld("ldux", "rt.mem.read_u64(ea as u32)", true),
        23 => ld("lwzx", "rt.mem.read_u32(ea as u32) as u64", false),
        55 => ld("lwzux", "rt.mem.read_u32(ea as u32) as u64", true),
        87 => ld("lbzx", "rt.mem.read_u8(ea as u32) as u64", false),
        119 => ld("lbzux", "rt.mem.read_u8(ea as u32) as u64", true),
        279 => ld("lhzx", "rt.mem.read_u16(ea as u32) as u64", false),
        311 => ld("lhzux", "rt.mem.read_u16(ea as u32) as u64", true),
        343 => ld("lhax", "rt.mem.read_u16(ea as u32) as i16 as i64 as u64", false),
        375 => ld("lhaux", "rt.mem.read_u16(ea as u32) as i16 as i64 as u64", true),
        341 => ld("lwax", "rt.mem.read_u32(ea as u32) as i32 as i64 as u64", false),
        373 => ld("lwaux", "rt.mem.read_u32(ea as u32) as i32 as i64 as u64", true),
        534 => ld("lwbrx", "rt.mem.read_u32(ea as u32).swap_bytes() as u64", false),
        790 => ld("lhbrx", "rt.mem.read_u16(ea as u32).swap_bytes() as u64", false),
        149 => st("stdx", format!("rt.mem.write_u64(ea as u32, {rs})"), false),
        181 => st("stdux", format!("rt.mem.write_u64(ea as u32, {rs})"), true),
        151 => st("stwx", format!("rt.mem.write_u32(ea as u32, {rs} as u32)"), false),
        183 => st("stwux", format!("rt.mem.write_u32(ea as u32, {rs} as u32)"), true),
        215 => st("stbx", format!("rt.mem.write_u8(ea as u32, {rs} as u8)"), false),
        247 => st("stbux", format!("rt.mem.write_u8(ea as u32, {rs} as u8)"), true),
        407 => st("sthx", format!("rt.mem.write_u16(ea as u32, {rs} as u16)"), false),
        439 => st("sthux", format!("rt.mem.write_u16(ea as u32, {rs} as u16)"), true),
        662 => st("stwbrx", format!("rt.mem.write_u32(ea as u32, ({rs} as u32).swap_bytes())"), false),
        918 => st("sthbrx", format!("rt.mem.write_u16(ea as u32, ({rs} as u16).swap_bytes())"), false),
        535 => Some(("lfsx", format!("{{ let ea = {ea}; c.f[{d}] = fpu::single_to_double(rt.mem.read_u32(ea as u32)); }}"))),
        567 => Some((
            "lfsux",
            format!("{{ let ea = {ea}; c.f[{d}] = fpu::single_to_double(rt.mem.read_u32(ea as u32)); c.r[{a}] = ea; }}"),
        )),
        599 => Some(("lfdx", format!("{{ let ea = {ea}; c.f[{d}] = f64::from_bits(rt.mem.read_u64(ea as u32)); }}"))),
        631 => Some((
            "lfdux",
            format!("{{ let ea = {ea}; c.f[{d}] = f64::from_bits(rt.mem.read_u64(ea as u32)); c.r[{a}] = ea; }}"),
        )),
        663 => st("stfsx", format!("rt.mem.write_u32(ea as u32, fpu::double_to_single(c.f[{d}]))"), false),
        695 => st("stfsux", format!("rt.mem.write_u32(ea as u32, fpu::double_to_single(c.f[{d}]))"), true),
        727 => st("stfdx", format!("rt.mem.write_u64(ea as u32, c.f[{d}].to_bits())"), false),
        759 => st("stfdux", format!("rt.mem.write_u64(ea as u32, c.f[{d}].to_bits())"), true),
        983 => st("stfiwx", format!("rt.mem.write_u32(ea as u32, c.f[{d}].to_bits() as u32)"), false),
        1014 => {
            let size = if d == 1 { 128 } else { 32 };
            Some((
                if d == 1 { "dcbz128" } else { "dcbz" },
                format!("rt.mem.zero_block(({ea}) as u32, {size});"),
            ))
        }
        // Cache hints, barriers and data-stream touches: nothing to do.
        54 => Some(("dcbst", String::new())),
        86 => Some(("dcbf", String::new())),
        246 => Some(("dcbtst", String::new())),
        278 => Some(("dcbt", String::new())),
        342 => Some(("dst", String::new())),
        374 => Some(("dstst", String::new())),
        470 => Some(("dcbi", String::new())),
        // sync (L=0) is a full barrier; lwsync (L=1) orders everything but
        // store-then-load, which is what acquire/release fences give.
        598 if d & 3 == 1 => Some((
            "lwsync",
            "std::sync::atomic::fence(std::sync::atomic::Ordering::AcqRel);".into(),
        )),
        598 => Some((
            "sync",
            "std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);".into(),
        )),
        822 => Some(("dss", String::new())),
        854 => Some((
            "eieio",
            "std::sync::atomic::fence(std::sync::atomic::Ordering::Release);".into(),
        )),
        982 => Some(("icbi", String::new())),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn op63(
    w: u32,
    d: u32,
    a: u32,
    b: u32,
    cc: u32,
    xo10: u32,
    rec: &str,
) -> Option<(&'static str, String)> {
    let (fa, fb, fc) = (
        format!("c.f[{a}]"),
        format!("c.f[{b}]"),
        format!("c.f[{cc}]"),
    );
    let set = |name: &'static str, e: String| Some((name, format!("c.f[{d}] = {e};{rec}")));
    match w >> 1 & 31 {
        18 => return set("fdiv", format!("fpu::div({fa}, {fb})")),
        20 => return set("fsub", format!("fpu::sub({fa}, {fb})")),
        21 => return set("fadd", format!("fpu::add({fa}, {fb})")),
        22 => return set("fsqrt", format!("fpu::sqrt({fb})")),
        23 => return set("fsel", format!("fpu::fsel({fa}, {fc}, {fb})")),
        25 => return set("fmul", format!("fpu::mul({fa}, {fc})")),
        26 => return set("frsqrte", format!("fpu::frsqrte({fb})")),
        28 => return set("fmsub", format!("fpu::msub({fa}, {fc}, {fb})")),
        29 => return set("fmadd", format!("fpu::madd({fa}, {fc}, {fb})")),
        30 => {
            return set(
                "fnmsub",
                format!("fpu::neg_num(fpu::msub({fa}, {fc}, {fb}))"),
            )
        }
        31 => {
            return set(
                "fnmadd",
                format!("fpu::neg_num(fpu::madd({fa}, {fc}, {fb}))"),
            )
        }
        _ => {}
    }
    match xo10 {
        0 | 32 => {
            let crf = d >> 2;
            Some((
                if xo10 == 0 { "fcmpu" } else { "fcmpo" },
                format!(
                    "{{ let v = fpu::cmp({fa}, {fb}); c.set_cr_field({crf}, v); c.fpscr = (c.fpscr & !fpu::FPCC_MASK) | v << 12; }}"
                ),
            ))
        }
        12 => set("frsp", format!("fpu::frsp({fb})")),
        14 => set("fctiw", format!("fpu::fctiw({fb}, c.fpscr & 3)")),
        15 => set("fctiwz", format!("fpu::fctiw({fb}, 1)")),
        814 => set("fctid", format!("fpu::fctid({fb}, c.fpscr & 3)")),
        815 => set("fctidz", format!("fpu::fctid({fb}, 1)")),
        846 => set("fcfid", format!("fpu::fcfid({fb})")),
        40 => set(
            "fneg",
            format!("f64::from_bits({fb}.to_bits() ^ 0x8000_0000_0000_0000)"),
        ),
        72 => set("fmr", fb.clone()),
        136 => set(
            "fnabs",
            format!("f64::from_bits({fb}.to_bits() | 0x8000_0000_0000_0000)"),
        ),
        264 => set(
            "fabs",
            format!("f64::from_bits({fb}.to_bits() & !0x8000_0000_0000_0000)"),
        ),
        583 => set("mffs", "f64::from_bits(c.fpscr as u64)".to_string()),
        711 => {
            let m = field_mask(w >> 17 & 0xFF);
            Some((
                "mtfsf",
                format!("c.fpscr = (c.fpscr & !{m:#x}) | ({fb}.to_bits() as u32 & {m:#x});{rec}"),
            ))
        }
        134 => {
            let crf = d >> 2;
            let imm = w >> 12 & 0xF;
            let sh = 28 - 4 * crf;
            Some((
                "mtfsfi",
                format!("c.fpscr = (c.fpscr & !(0xF << {sh})) | ({imm} << {sh});{rec}"),
            ))
        }
        38 => Some(("mtfsb1", format!("c.fpscr |= 1 << {};{rec}", 31 - d))),
        70 => Some(("mtfsb0", format!("c.fpscr &= !(1 << {});{rec}", 31 - d))),
        64 => Some((
            "mcrfs",
            format!(
                "{{ let v = c.fpscr >> {} & 0xF; c.set_cr_field({}, v); }}",
                28 - 4 * (a >> 2),
                d >> 2
            ),
        )),
        _ => None,
    }
}

/// A short listing of a translated instruction for comments in generated code.
pub fn describe(w: u32, pc: u32) -> String {
    let ins = translate(w, pc, &Func::default());
    let mut s = String::new();
    let _ = write!(s, "{pc:08X}: {w:08X} {}", ins.name);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(w: u32) -> &'static str {
        translate(w, 0x8200_0000, &Func::default()).name
    }

    /// VMX128 encodings as published by an independent source (Xenia's
    /// instruction tests, which hand-code them because binutils cannot).
    #[test]
    fn vmx128_published_encodings() {
        assert_eq!(name(0x189B_1A10), "vpermwi128");
        assert_eq!(name(0x1884_1BD0), "vpermwi128");
        assert_eq!(name(0x188F_1F50), "vrlimi128");
        assert_eq!(name(0x1860_1FF0), "vupkd3d128");
        assert_eq!(name(0x1881_1E10), "vpkd3d128");
        let t = translate(0x189B_1A10, 0, &Func::default());
        assert!(
            t.code.contains("c.v[4] = vmx::vpermwi(c.v[3], 0x1b)"),
            "{}",
            t.code
        );
        let t = translate(0x188F_1F50, 0, &Func::default());
        assert!(
            t.code.contains("vmx::vrlimi(c.v[4], c.v[3], 15, 1)"),
            "{}",
            t.code
        );
    }

    #[test]
    fn unknown_words_compile_to_a_runtime_stop() {
        let t = translate(0, 0x8200_0010, &Func::default());
        assert!(t.unknown && t.ends);
        assert_eq!(t.code, "rt.unknown(c, pc, 0x00000000);");
    }
}
