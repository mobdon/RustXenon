//! Function and jump-table discovery for an executable with no symbols.
//!
//! Phases (order and idea from ReXGlue's analysis; the extent walk, helper
//! layouts and the four jump-table shapes from XenonRecomp's XenonAnalyse and
//! recompiler, MIT; see CREDITS.md):
//!
//! 1. Register starts we are told about: per-game overrides and hook roles,
//!    register save/restore helpers (by byte pattern), import thunks, the
//!    exception table, `mflr r12` prologues after a return, the entry point.
//! 2. Jump tables: `mtctr`/`bctr` sequences of four known shapes, with the
//!    case count from the bounds check before them, or, for an absolute table
//!    stored right after its `bctr` with no check, from the entries up to the
//!    first case.
//! 3. Every `bl` target in code is a function start; a plain `b` is not.
//! 4. Each start without a size is walked block by block up to the next
//!    start; the function ends at the end of its contiguous blocks.
//! 5. Gap fill: code no function covers starts a new function.
//! 6. Validate: overlaps, jump targets outside their function, calls outside
//!    code, a missing entry point.

use crate::hooks::Hooks;
use crate::xex::Image;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    Override,
    Helper,
    Import,
    ExceptionTable,
    Prologue,
    Entry,
    Call,
    Gap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub start: u32,
    /// Bytes; 0 only before phase 4.
    pub size: u32,
    pub source: Source,
    /// Helper, import or hook role name.
    pub name: Option<String>,
}

impl Function {
    pub fn end(&self) -> u32 {
        self.start + self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    /// 32-bit absolute targets (`lwzx`).
    Absolute,
    /// Byte offsets shifted left, added to a base (`lbzx`, `rlwinm`).
    Computed,
    /// Byte offsets added to a base (`lbzx`).
    ByteOffset,
    /// 16-bit offsets added to a base (`lhzx`).
    ShortOffset,
    /// From the per-game file.
    Override,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpTable {
    pub bctr: u32,
    pub kind: TableKind,
    /// Register holding the case index, when known.
    pub index_reg: Option<u8>,
    pub default: Option<u32>,
    pub targets: Vec<u32>,
    /// Where the table's bytes are, when read from the image.
    pub data: Option<std::ops::Range<u32>>,
}

#[derive(Debug, Default)]
pub struct Analysis {
    pub functions: BTreeMap<u32, Function>,
    /// Keyed by the `bctr` address.
    pub jump_tables: BTreeMap<u32, JumpTable>,
    pub problems: Vec<String>,
}

impl Analysis {
    /// The function covering `addr`, if any.
    pub fn function_at(&self, addr: u32) -> Option<&Function> {
        self.functions
            .range(..=addr)
            .next_back()
            .map(|(_, f)| f)
            .filter(|f| addr < f.end())
    }
}

// ---- instruction fields ----------------------------------------------------

const BLR: u32 = 0x4E80_0020;
const BCTR: u32 = 0x4E80_0420;
const NOP: u32 = 0x6000_0000;
const MFLR_R12: u32 = 0x7D88_02A6;

fn op(w: u32) -> u32 {
    w >> 26
}
fn rt(w: u32) -> u32 {
    (w >> 21) & 31
}
fn ra(w: u32) -> u32 {
    (w >> 16) & 31
}
fn simm(w: u32) -> u32 {
    w as i16 as i32 as u32
}
fn xo(w: u32) -> u32 {
    (w >> 1) & 0x3FF
}
fn link(w: u32) -> bool {
    w & 1 != 0
}
/// Target of `b`/`bl` at `pc` (relative forms only).
fn b_target(w: u32, pc: u32) -> Option<u32> {
    (w & 2 == 0).then(|| pc.wrapping_add((((w & 0x03FF_FFFC) << 6) as i32 >> 6) as u32))
}
/// Target of a conditional branch at `pc` (relative forms only).
fn bc_target(w: u32, pc: u32) -> Option<u32> {
    (w & 2 == 0).then(|| pc.wrapping_add((w & 0xFFFC) as i16 as i32 as u32))
}
/// The BO field says "branch always".
fn always(w: u32) -> bool {
    rt(w) & 0x14 == 0x14
}
fn is_lis(w: u32) -> bool {
    op(w) == 15 && ra(w) == 0
}
fn is_x(w: u32, x: u32) -> bool {
    op(w) == 31 && xo(w) == x
}
fn is_mtctr(w: u32) -> bool {
    w & 0xFC1F_FFFF == 0x7C09_03A6
}

// ---- phase 1: helpers --------------------------------------------------------

struct HelperShape {
    name: &'static str,
    /// First register and how many the helper family covers.
    first: u32,
    count: u32,
    /// Words for register `first + i`.
    body: fn(u32) -> Vec<u32>,
    tail: &'static [u32],
}

fn ds(base: u32, reg: u32, ra: u32, disp: i32) -> u32 {
    base | reg << 21 | ra << 16 | (disp as u32 & 0xFFFF)
}

/// `li r11, disp` then the vector store/load `op` of `vr` at `r11 + r12`.
fn vmx(op: u32, vr: u32, disp: i32) -> Vec<u32> {
    vec![
        0x3960_0000 | (disp as u32 & 0xFFFF),
        op | vr << 21 | 11 << 16 | 12 << 11,
    ]
}

/// `li r11, disp` then `stvx128`/`lvx128` (`op`) of v64+i at `r11 + r12`.
fn vmx128(op: u32, i: u32) -> Vec<u32> {
    let r = 64 + i;
    vec![
        0x3960_0000 | ((-0x400 + 16 * i as i32) as u32 & 0xFFFF),
        op | (r & 31) << 21 | (r >> 5) << 2 | 11 << 16 | 12 << 11,
    ]
}

const HELPERS: [HelperShape; 8] = [
    HelperShape {
        name: "savegprlr",
        first: 14,
        count: 18,
        body: |i| vec![ds(0xF800_0000, 14 + i, 1, -0x98 + 8 * i as i32)],
        tail: &[0x9181_FFF8, BLR],
    },
    HelperShape {
        name: "restgprlr",
        first: 14,
        count: 18,
        body: |i| vec![ds(0xE800_0000, 14 + i, 1, -0x98 + 8 * i as i32)],
        tail: &[0x8181_FFF8, 0x7D88_03A6, BLR],
    },
    HelperShape {
        name: "savefpr",
        first: 14,
        count: 18,
        body: |i| vec![ds(0xD800_0000, 14 + i, 12, -0x90 + 8 * i as i32)],
        tail: &[BLR],
    },
    HelperShape {
        name: "restfpr",
        first: 14,
        count: 18,
        body: |i| vec![ds(0xC800_0000, 14 + i, 12, -0x90 + 8 * i as i32)],
        tail: &[BLR],
    },
    HelperShape {
        name: "savevmx",
        first: 14,
        count: 18,
        body: |i| vmx(0x7C00_01CE, 14 + i, -0x120 + 16 * i as i32),
        tail: &[BLR],
    },
    HelperShape {
        name: "restvmx",
        first: 14,
        count: 18,
        body: |i| vmx(0x7C00_00CE, 14 + i, -0x120 + 16 * i as i32),
        tail: &[BLR],
    },
    HelperShape {
        name: "savevmx",
        first: 64,
        count: 64,
        body: |i| vmx128(0x1000_01C3, i),
        tail: &[BLR],
    },
    HelperShape {
        name: "restvmx",
        first: 64,
        count: 64,
        body: |i| vmx128(0x1000_00C3, i),
        tail: &[BLR],
    },
];

fn code_words(img: &Image) -> impl Iterator<Item = (u32, u32)> + '_ {
    img.sections.iter().filter(|s| s.code).flat_map(move |s| {
        (s.start..s.end() & !3)
            .step_by(4)
            .filter_map(move |a| img.u32(a).map(|w| (a, w)))
    })
}

fn find_helpers(img: &Image, out: &mut BTreeMap<u32, Function>) {
    for shape in &HELPERS {
        let n = shape.count;
        let mut seq: Vec<u32> = (0..n).flat_map(shape.body).collect();
        seq.extend(shape.tail);
        let stride = 4 * (shape.body)(0).len() as u32;
        for (a, w) in code_words(img) {
            if w != seq[0] || !img.bytes(a, seq.len() * 4).is_some_and(|b| words(b) == seq) {
                continue;
            }
            for i in 0..n {
                let start = a + i * stride;
                out.entry(start).or_insert(Function {
                    start,
                    size: (n - i) * stride + 4 * shape.tail.len() as u32,
                    source: Source::Helper,
                    name: Some(format!("{}_{}", shape.name, shape.first + i)),
                });
            }
        }
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
        .collect()
}

// ---- phase 2: jump tables ---------------------------------------------------

/// `lis rX, hi` then `addi rX, rX, lo` at `a`: the 32-bit value.
fn lis_addi(img: &Image, a: u32) -> Option<u32> {
    let (hi, lo) = (img.u32(a)?, img.u32(a + 4)?);
    (is_lis(hi) && op(lo) == 14).then(|| (simm(hi) << 16).wrapping_add(simm(lo)))
}

fn match_table(img: &Image, mtctr: u32) -> Option<(TableKind, u32)> {
    let at = |k: u32| img.u32(mtctr.wrapping_sub(4 * k)).unwrap_or(0);
    if is_x(at(1), 23) && op(at(2)) == 21 && is_lis(at(4)) {
        return Some((TableKind::Absolute, mtctr - 16));
    }
    if !is_x(at(1), 266) || op(at(2)) != 14 || !is_lis(at(3)) {
        return None;
    }
    if is_x(at(4), 87) && is_lis(at(6)) {
        return Some((TableKind::ByteOffset, mtctr - 24));
    }
    if op(at(4)) == 21 && is_x(at(5), 87) && is_lis(at(7)) {
        return Some((TableKind::Computed, mtctr - 28));
    }
    if is_x(at(4), 279) && op(at(5)) == 21 && is_lis(at(7)) {
        return Some((TableKind::ShortOffset, mtctr - 28));
    }
    None
}

/// Case count, index register and default from the bounds check before
/// `pattern`: `cmplwi crN, rI, max` then `bgt crN, default` (or `ble`).
fn bounds(img: &Image, pattern: u32) -> Option<(usize, u8, Option<u32>)> {
    let mut cr = None;
    let mut default = None;
    for k in 1..=32 {
        let a = pattern.checked_sub(4 * k)?;
        let w = img.u32(a)?;
        match cr {
            None if (op(w) == 16 || (op(w) == 19 && xo(w) == 16)) && !link(w) && ra(w) & 3 == 1 => {
                let bo = rt(w) & 0x1C;
                if bo == 0x0C || bo == 0x04 {
                    cr = Some(ra(w) >> 2);
                    if op(w) == 16 && bo == 0x0C {
                        default = bc_target(w, a);
                    }
                }
            }
            Some(c) if op(w) == 10 && rt(w) == c << 2 => {
                return Some(((w & 0xFFFF) as usize + 1, ra(w) as u8, default));
            }
            _ => {}
        }
    }
    None
}

/// Address of the table bytes and the size of one entry.
fn table_data(img: &Image, kind: TableKind, start: u32) -> Option<(u32, u32)> {
    let entry = match kind {
        TableKind::Absolute => 4,
        TableKind::ShortOffset => 2,
        _ => 1,
    };
    Some((lis_addi(img, start)?, entry))
}

/// Case count of an absolute table stored right after its `bctr` with no
/// bounds check: entries run while they are code addresses and the table has
/// not reached the first case.
fn inline_count(img: &Image, bctr: u32) -> Option<usize> {
    let sec = img.section_at(bctr)?;
    let mut first = u32::MAX;
    let mut a = bctr + 4;
    while a < first {
        match img.u32(a) {
            Some(t) if t & 3 == 0 && t > bctr && (sec.start..sec.end()).contains(&t) => {
                first = first.min(t)
            }
            _ => break,
        }
        a += 4;
    }
    let n = ((a - bctr - 4) / 4) as usize;
    (n > 0).then_some(n)
}

fn read_table(img: &Image, kind: TableKind, start: u32, count: usize) -> Option<Vec<u32>> {
    let table = lis_addi(img, start)?;
    let (rest, base) = match kind {
        TableKind::Absolute => return (0..count as u32).map(|i| img.u32(table + 4 * i)).collect(),
        TableKind::ByteOffset => (0, lis_addi(img, start + 12)?),
        TableKind::Computed => (img.u32(start + 12)? >> 11 & 31, lis_addi(img, start + 16)?),
        TableKind::ShortOffset => (0, lis_addi(img, start + 16)?),
        TableKind::Override => return None,
    };
    (0..count as u32)
        .map(|i| match kind {
            TableKind::ShortOffset => img
                .bytes(table + 2 * i, 2)
                .map(|b| base.wrapping_add(u32::from(u16::from_be_bytes([b[0], b[1]])))),
            _ => img
                .bytes(table + i, 1)
                .map(|b| base.wrapping_add(u32::from(b[0]) << rest)),
        })
        .collect()
}

fn find_tables(img: &Image, hooks: &Hooks, a: &mut Analysis) {
    for (pc, w) in code_words(img) {
        if w != BCTR || hooks.is_data(pc) {
            continue;
        }
        // The compiler sometimes puts a nop between mtctr and bctr.
        let mtctr = if img.u32(pc - 4) == Some(NOP) {
            pc - 8
        } else {
            pc - 4
        };
        if !img.u32(mtctr).is_some_and(is_mtctr) {
            continue;
        }
        let Some((kind, start)) = match_table(img, mtctr) else {
            continue;
        };
        let data = table_data(img, kind, start);
        let (count, index_reg, default) = match bounds(img, start) {
            Some((c, r, d)) => (c, Some(r), d),
            None => match data
                .filter(|&(t, _)| kind == TableKind::Absolute && t == pc + 4)
                .and_then(|_| inline_count(img, pc))
            {
                Some(c) => (c, None, None),
                None => {
                    a.problems
                        .push(format!("{pc:#x}: jump table without a bounds check"));
                    continue;
                }
            },
        };
        match read_table(img, kind, start, count) {
            Some(targets) => {
                a.jump_tables.insert(
                    pc,
                    JumpTable {
                        bctr: pc,
                        kind,
                        index_reg,
                        default,
                        targets,
                        data: data.map(|(t, e)| t..t + e * count as u32),
                    },
                );
            }
            None => a
                .problems
                .push(format!("{pc:#x}: jump table data outside the image")),
        }
    }
    for t in &hooks.jump_tables {
        a.jump_tables.insert(
            t.bctr,
            JumpTable {
                bctr: t.bctr,
                kind: TableKind::Override,
                index_reg: None,
                default: t.default,
                targets: t.targets.clone(),
                data: None,
            },
        );
    }
}

// ---- phase 4: extent ----------------------------------------------------------

/// Size of the function at `start`, walking blocks below `limit`.
fn extent(img: &Image, start: u32, limit: u32, tables: &BTreeMap<u32, JumpTable>) -> u32 {
    let mut seen = BTreeSet::new();
    let mut work = vec![start];
    let local = |t: u32| (start..limit).contains(&t);
    while let Some(mut a) = work.pop() {
        while local(a) && img.is_code(a) && !seen.contains(&a) {
            let Some(w) = img.u32(a).filter(|&w| w != 0) else {
                break; // zero padding ends a block
            };
            seen.insert(a);
            match op(w) {
                16 if !link(w) => {
                    work.extend(bc_target(w, a).filter(|&t| local(t)));
                    if always(w) {
                        break;
                    }
                }
                18 if !link(w) => {
                    work.extend(b_target(w, a).filter(|&t| local(t)));
                    break;
                }
                19 if !link(w) && (xo(w) == 16 || xo(w) == 528) => {
                    if xo(w) == 528 {
                        if let Some(t) = tables.get(&a) {
                            work.extend(t.targets.iter().copied().filter(|&t| local(t)));
                        }
                    }
                    if always(w) {
                        break;
                    }
                }
                _ => {}
            }
            a += 4;
        }
    }
    // Keep the contiguous run from the start; later islands are dropped.
    // Table bytes stored in code right after one of our `bctr`s are part of the run.
    let inline: Vec<_> = tables
        .values()
        .filter(|t| seen.contains(&t.bctr))
        .filter_map(|t| t.data.clone())
        .collect();
    let mut end = start;
    loop {
        if seen.contains(&end) {
            end += 4;
        } else if let Some(r) = inline.iter().find(|r| r.contains(&end)) {
            end = (r.end + 3) & !3;
        } else {
            break;
        }
    }
    (end - start).max(4)
}

fn next_start(fns: &BTreeMap<u32, Function>, img: &Image, start: u32) -> u32 {
    let section_end = img.section_at(start).map_or(start + 4, |s| s.end());
    fns.range(start + 1..)
        .next()
        .map_or(section_end, |(&a, _)| a.min(section_end))
}

// ---- driver --------------------------------------------------------------------

pub fn analyze(img: &Image, hooks: &Hooks) -> Analysis {
    let mut a = Analysis::default();
    let fns = &mut a.functions;
    let add = |fns: &mut BTreeMap<u32, Function>,
               start: u32,
               size: u32,
               source: Source,
               name: Option<String>| {
        fns.entry(start).or_insert(Function {
            start,
            size,
            source,
            name,
        });
    };

    // 1. Registered starts. Overrides first so they win.
    for r in &hooks.functions.bounds {
        add(fns, r.start, r.end - r.start, Source::Override, None);
    }
    for &s in &hooks.functions.extra {
        add(fns, s, 0, Source::Override, None);
    }
    for (role, &s) in &hooks.hooks {
        add(fns, s, 0, Source::Override, Some(role.clone()));
    }
    find_helpers(img, fns);
    for i in img
        .imports
        .iter()
        .filter(|i| i.thunk && img.is_code(i.address))
    {
        add(
            fns,
            i.address,
            16,
            Source::Import,
            Some(format!("{}#{}", i.library, i.ordinal)),
        );
    }
    for f in img.exception_table.iter().filter(|f| f.size > 0) {
        add(fns, f.start, f.size, Source::ExceptionTable, None);
    }
    let mut prev = 0;
    for (pc, w) in code_words(img) {
        if w == MFLR_R12 && (prev == BLR || prev == 0) && !hooks.is_data(pc) {
            add(fns, pc, 0, Source::Prologue, None);
        }
        prev = w;
    }
    add(fns, img.entry, 0, Source::Entry, None);

    // 2. Jump tables.
    find_tables(img, hooks, &mut a);
    let fns = &mut a.functions;

    // 3. Call targets.
    let mut outside = BTreeSet::new();
    for (pc, w) in code_words(img) {
        if op(w) != 18 || !link(w) || hooks.is_data(pc) {
            continue;
        }
        if let Some(t) = b_target(w, pc) {
            if img.is_code(t) && !hooks.is_data(t) {
                add(fns, t, 0, Source::Call, None);
            } else {
                outside.insert(pc);
            }
        }
    }

    // 4. Extents.
    let pending: Vec<u32> = fns
        .values()
        .filter(|f| f.size == 0)
        .map(|f| f.start)
        .collect();
    for s in pending {
        let size = extent(img, s, next_start(fns, img, s), &a.jump_tables);
        fns.get_mut(&s).unwrap().size = size;
    }

    // 5. Gap fill.
    for sec in img.sections.iter().filter(|s| s.code) {
        let mut pc = sec.start;
        while pc < sec.end() {
            if let Some(f) = fns
                .range(..=pc)
                .next_back()
                .map(|(_, f)| f)
                .filter(|f| pc < f.end())
            {
                pc = f.end();
                continue;
            }
            if hooks.is_data(pc) || img.u32(pc).unwrap_or(0) == 0 {
                pc += 4;
                continue;
            }
            let size = extent(img, pc, next_start(fns, img, pc), &a.jump_tables);
            add(fns, pc, size, Source::Gap, None);
            pc += size;
        }
    }

    // 6. Validate.
    let p = &mut a.problems;
    if !fns.contains_key(&img.entry) || !img.is_code(img.entry) {
        p.push(format!("entry point {:#x} is not in code", img.entry));
    }
    for pc in outside {
        p.push(format!("{pc:#x}: call to an address outside code"));
    }
    let mut last: Option<&Function> = None;
    for f in fns.values() {
        if let Some(l) = last.filter(|l| l.end() > f.start) {
            if !(l.source == Source::Helper && f.source == Source::Helper) {
                p.push(format!(
                    "functions {:#x} and {:#x} overlap",
                    l.start, f.start
                ));
            }
        }
        if last.map_or(true, |l| f.end() > l.end()) {
            last = Some(f);
        }
    }
    for t in a.jump_tables.values() {
        let home = fns
            .range(..=t.bctr)
            .next_back()
            .map(|(_, f)| f)
            .filter(|f| t.bctr < f.end());
        match home {
            None => p.push(format!("{:#x}: jump table outside any function", t.bctr)),
            Some(f) => {
                for &x in t.targets.iter().filter(|&&x| x < f.start || x >= f.end()) {
                    p.push(format!(
                        "{:#x}: jump target {x:#x} outside function {:#x}",
                        t.bctr, f.start
                    ));
                }
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::Range;
    use crate::xex::{Import, RuntimeFunction, Section};

    const BASE: u32 = 0x8200_0000;
    const RDATA: u32 = BASE + 0x100;
    const TEXT: u32 = BASE + 0x1000;

    fn image(code: &[u32], rdata: &[u8]) -> Image {
        let mut data = vec![0u8; 0x1000 + code.len() * 4];
        data[0x100..0x100 + rdata.len()].copy_from_slice(rdata);
        for (i, w) in code.iter().enumerate() {
            data[0x1000 + i * 4..0x1004 + i * 4].copy_from_slice(&w.to_be_bytes());
        }
        Image {
            base: BASE,
            entry: TEXT,
            title_id: None,
            version: None,
            data,
            sections: vec![
                Section {
                    name: ".rdata".into(),
                    start: RDATA,
                    size: 0xF00,
                    code: false,
                },
                Section {
                    name: ".text".into(),
                    start: TEXT,
                    size: code.len() as u32 * 4,
                    code: true,
                },
            ],
            imports: vec![],
            exception_table: vec![],
        }
    }

    fn at(i: usize) -> u32 {
        TEXT + 4 * i as u32
    }
    fn b(from: usize, to: usize, lk: bool) -> u32 {
        0x4800_0000 | (at(to).wrapping_sub(at(from)) & 0x03FF_FFFC) | u32::from(lk)
    }
    /// `bc` with BO/BI.
    fn bc(bo: u32, bi: u32, from: usize, to: usize) -> u32 {
        0x4000_0000 | bo << 21 | bi << 16 | (at(to).wrapping_sub(at(from)) & 0xFFFC)
    }
    fn lis(r: u32, v: u32) -> u32 {
        0x3C00_0000 | r << 21 | (v.wrapping_add(0x8000) >> 16)
    }
    fn addi(r: u32, v: u32) -> u32 {
        0x3800_0000 | r << 21 | r << 16 | (v & 0xFFFF)
    }
    fn x(xo: u32, d: u32, a: u32, b: u32) -> u32 {
        0x7C00_0000 | d << 21 | a << 16 | b << 11 | xo << 1
    }
    fn rlwinm(a: u32, s: u32, sh: u32) -> u32 {
        0x5400_0000 | s << 21 | a << 16 | sh << 11
    }
    fn cmplwi(cr: u32, r: u32, v: u32) -> u32 {
        0x2800_0000 | cr << 23 | r << 16 | v
    }
    const MTCTR_R0: u32 = 0x7C09_03A6;

    fn starts(a: &Analysis) -> Vec<(u32, u32, Source)> {
        a.functions
            .values()
            .map(|f| (f.start, f.size, f.source))
            .collect()
    }

    #[test]
    fn calls_branches_tail_calls_and_gaps() {
        let code = [
            /* 0 entry */ b(0, 4, true),
            /* 1 */ bc(12, 2, 1, 3), // beq +8 (local)
            /* 2 */ NOP,
            /* 3 */ b(3, 8, false), // tail call to a later function
            /* 4 callee */ bc(12, 0, 4, 6),
            /* 5 */ BLR,
            /* 6 */ NOP,
            /* 7 */ BLR,
            /* 8 tail target, reached only by b */ BLR,
            /* 9 padding */ 0,
            /* 10 gap */ NOP,
            /* 11 */ BLR,
        ];
        let img = image(&code, &[]);
        let a = analyze(&img, &Hooks::default());
        assert_eq!(
            starts(&a),
            vec![
                (at(0), 16, Source::Entry),
                (at(4), 16, Source::Call),
                (at(8), 4, Source::Gap),
                (at(10), 8, Source::Gap),
            ]
        );
        assert!(a.problems.is_empty(), "{:?}", a.problems);
        assert_eq!(a.function_at(at(6)).unwrap().start, at(4));
    }

    #[test]
    fn exception_table_imports_prologues_and_helpers() {
        let mut code = vec![MFLR_R12, NOP, BLR, MFLR_R12, BLR];
        let helper = code.len();
        for i in 0..18 {
            code.push(ds(0xF800_0000, 14 + i, 1, -0x98 + 8 * i as i32));
        }
        code.extend([0x9181_FFF8, BLR]);
        let thunk = code.len();
        code.extend([0, 0, 0, 0]);
        code.push(b(code.len(), helper + 4, true)); // bl savegprlr_18
        code.push(BLR);
        let mut img = image(&code, &[]);
        img.exception_table = vec![RuntimeFunction {
            start: at(0),
            size: 12,
            prolog_words: 1,
        }];
        img.imports = vec![Import {
            library: "k".into(),
            ordinal: 7,
            address: at(thunk),
            thunk: true,
        }];
        let a = analyze(&img, &Hooks::default());
        let f = |i| a.functions[&at(i)].clone();
        assert_eq!(f(0).source, Source::ExceptionTable);
        assert_eq!((f(3).source, f(3).size), (Source::Prologue, 8));
        assert_eq!(f(helper).name.as_deref(), Some("savegprlr_14"));
        assert_eq!(f(helper + 4).size, 14 * 4 + 8);
        assert_eq!(f(helper + 17).name.as_deref(), Some("savegprlr_31"));
        assert_eq!(
            (f(thunk).source, f(thunk).name.as_deref()),
            (Source::Import, Some("k#7"))
        );
        assert_eq!(f(thunk + 4).source, Source::Gap);
        assert!(a.problems.is_empty(), "{:?}", a.problems);
    }

    /// A switch on r11 with 3 cases; returns (code, rdata, bctr index).
    fn switch(kind: TableKind) -> (Vec<u32>, Vec<u8>, usize) {
        switch_with(kind, false)
    }

    /// `nop_before_bctr` puts a nop between mtctr and bctr.
    fn switch_with(kind: TableKind, nop_before_bctr: bool) -> (Vec<u32>, Vec<u8>, usize) {
        let table = RDATA;
        let mut code = vec![cmplwi(6, 11, 2), bc(12, 25, 1, 0)]; // bgt cr6, default (patched)
        let tbl = match kind {
            TableKind::Absolute => vec![
                lis(12, table),
                addi(12, table),
                rlwinm(0, 11, 2),
                x(23, 0, 12, 0),
            ],
            TableKind::Computed => vec![
                lis(12, table),
                addi(12, table),
                x(87, 0, 12, 11),
                rlwinm(0, 0, 2),
                lis(12, TEXT),
                addi(12, TEXT),
                x(266, 0, 12, 0),
            ],
            TableKind::ByteOffset => vec![
                lis(12, table),
                addi(12, table),
                x(87, 0, 12, 11),
                lis(12, TEXT),
                addi(12, TEXT),
                x(266, 0, 12, 0),
            ],
            TableKind::ShortOffset => vec![
                lis(12, table),
                addi(12, table),
                rlwinm(0, 11, 1),
                x(279, 0, 12, 0),
                lis(12, TEXT),
                addi(12, TEXT),
                x(266, 0, 12, 0),
            ],
            TableKind::Override => vec![],
        };
        code.extend(tbl);
        code.push(MTCTR_R0);
        if nop_before_bctr {
            code.push(NOP);
        }
        code.push(BCTR);
        let bctr = code.len() - 1;
        let cases: Vec<usize> = (0..3).map(|i| code.len() + i * 2).collect();
        for _ in 0..3 {
            code.extend([NOP, BLR]);
        }
        let default = code.len();
        code.push(BLR);
        code[1] = bc(12, 25, 1, default);
        let rdata = match kind {
            TableKind::Absolute => cases.iter().flat_map(|&c| at(c).to_be_bytes()).collect(),
            TableKind::Computed => cases.iter().map(|&c| ((c * 4) >> 2) as u8).collect(),
            TableKind::ByteOffset => cases.iter().map(|&c| (c * 4) as u8).collect(),
            _ => cases
                .iter()
                .flat_map(|&c| ((c * 4) as u16).to_be_bytes())
                .collect(),
        };
        (code, rdata, bctr)
    }

    #[test]
    fn four_jump_table_shapes() {
        for (kind, nop) in [
            (TableKind::Absolute, false),
            (TableKind::Computed, false),
            (TableKind::ByteOffset, false),
            (TableKind::ShortOffset, false),
            (TableKind::ShortOffset, true),
            (TableKind::ByteOffset, true),
        ] {
            let (code, rdata, bctr) = switch_with(kind, nop);
            let img = image(&code, &rdata);
            let a = analyze(&img, &Hooks::default());
            let t = &a.jump_tables[&at(bctr)];
            assert_eq!(t.kind, kind);
            assert_eq!(t.index_reg, Some(11));
            assert_eq!(
                t.targets,
                vec![at(bctr + 1), at(bctr + 3), at(bctr + 5)],
                "{kind:?}"
            );
            assert_eq!(t.default, Some(at(bctr + 7)));
            // Cases reached only through the table belong to the switch's function.
            assert_eq!(
                starts(&a),
                vec![(at(0), code.len() as u32 * 4, Source::Entry)],
                "{kind:?}"
            );
            assert!(a.problems.is_empty(), "{kind:?}: {:?}", a.problems);
        }
    }

    #[test]
    fn overrides_and_validation() {
        let (mut code, _, bctr) = switch(TableKind::Absolute);
        code.extend([0xDEAD_BEEF, 0xFFFF_FFFF, NOP, BLR, NOP, NOP, BLR]);
        let n = code.len();
        let img = image(&code, &[0; 12]); // table data all zero: targets outside
        let auto = analyze(&img, &Hooks::default());
        assert!(
            auto.problems.iter().any(|p| p.contains("jump target")),
            "{:?}",
            auto.problems
        );

        let hooks = Hooks {
            jump_tables: vec![crate::hooks::JumpTableOverride {
                bctr: at(bctr),
                targets: vec![at(bctr + 1), at(bctr + 3), at(bctr + 5)],
                default: None,
            }],
            not_code: vec![Range {
                start: at(n - 7),
                end: at(n - 5),
            }],
            functions: crate::hooks::Functions {
                extra: vec![at(n - 3)],
                bounds: vec![Range {
                    start: at(n - 5),
                    end: at(n - 3),
                }],
            },
            hooks: [("frame_end".to_string(), at(n - 1))].into(),
            ..Hooks::default()
        };
        let a = analyze(&img, &hooks);
        assert!(a.problems.is_empty(), "{:?}", a.problems);
        assert_eq!(a.jump_tables[&at(bctr)].kind, TableKind::Override);
        let s = starts(&a);
        assert_eq!(
            &s[1..],
            [
                (at(n - 5), 8, Source::Override),
                (at(n - 3), 8, Source::Override),
                (at(n - 1), 4, Source::Override),
            ]
        );
        assert!(
            a.function_at(at(n - 7)).is_none(),
            "data range not gap-filled"
        );
    }

    /// A switch whose absolute table sits in code right after the `bctr`,
    /// optionally with no bounds check; returns (code, bctr index, case indices).
    fn inline_switch(prefix: &[u32], checked: bool) -> (Vec<u32>, usize, Vec<usize>) {
        let mut code = prefix.to_vec();
        if checked {
            code.extend([cmplwi(6, 11, 2), 0x4D99_0020]); // bgtlr cr6
        }
        let table = code.len() + 6;
        code.extend([
            lis(12, at(table)),
            addi(12, at(table)),
            rlwinm(0, 11, 2),
            x(23, 0, 12, 0),
            MTCTR_R0,
            BCTR,
        ]);
        let bctr = code.len() - 1;
        let cases: Vec<usize> = (0..3).map(|i| table + 3 + i * 2).collect();
        code.extend(cases.iter().map(|&c| at(c)));
        for _ in 0..3 {
            code.extend([NOP, BLR]);
        }
        (code, bctr, cases)
    }

    #[test]
    fn inline_tables_stay_inside_their_function() {
        // Reached by a call, then by gap fill (no call, no prologue).
        for prefix in [vec![b(0, 2, true), BLR], vec![BLR, 0]] {
            let (code, bctr, cases) = inline_switch(&prefix, true);
            let img = image(&code, &[]);
            let a = analyze(&img, &Hooks::default());
            let t = &a.jump_tables[&at(bctr)];
            assert_eq!(t.targets, cases.iter().map(|&c| at(c)).collect::<Vec<_>>());
            let f = a.function_at(at(bctr)).unwrap();
            assert_eq!((f.start, f.end()), (at(2), at(code.len())), "{prefix:x?}");
            assert_eq!(a.functions.len(), 2, "no functions in the table or cases");
            assert!(a.problems.is_empty(), "{:?}", a.problems);
        }
    }

    #[test]
    fn inline_table_without_bounds_check() {
        let (code, bctr, cases) = inline_switch(&[], false);
        let img = image(&code, &[]);
        let a = analyze(&img, &Hooks::default());
        let t = &a.jump_tables[&at(bctr)];
        assert_eq!(t.targets, cases.iter().map(|&c| at(c)).collect::<Vec<_>>());
        assert_eq!((t.index_reg, t.default), (None, None));
        assert_eq!(
            starts(&a),
            vec![(at(0), code.len() as u32 * 4, Source::Entry)]
        );
        assert!(a.problems.is_empty(), "{:?}", a.problems);
    }

    #[test]
    fn float_and_vector_helpers_use_r12() {
        let mut code = vec![BLR];
        let fpr = code.len();
        code.extend((0..18).map(|i| ds(0xD800_0000, 14 + i, 12, -0x90 + 8 * i as i32)));
        code.push(BLR);
        let v128 = code.len();
        for i in 0..64 {
            code.extend(vmx128(0x1000_00C3, i));
        }
        code.push(BLR);
        let caller = code.len();
        code.extend([
            b(caller, fpr + 6, true),
            b(caller + 1, v128 + 2 * 40, true),
            BLR,
        ]);
        let mut img = image(&code, &[]);
        // The exception table often lists the helper block as one function.
        img.exception_table = vec![RuntimeFunction {
            start: at(fpr),
            size: 19 * 4,
            prolog_words: 0,
        }];
        let a = analyze(&img, &Hooks::default());
        let f = |i| a.functions[&at(i)].clone();
        assert_eq!(f(fpr + 6).name.as_deref(), Some("savefpr_20"));
        assert_eq!((f(fpr).source, f(fpr).size), (Source::Helper, 19 * 4));
        assert_eq!(f(v128).name.as_deref(), Some("restvmx_64"));
        assert_eq!(f(v128 + 2 * 63).name.as_deref(), Some("restvmx_127"));
        assert_eq!(f(v128 + 2 * 40).size, 24 * 8 + 4);
        assert_eq!(code[v128 + 2 * 32 + 1], 0x100B_60CF, "lvx128 v96, r11, r12");
        assert!(a.problems.is_empty(), "{:?}", a.problems);
    }
}
