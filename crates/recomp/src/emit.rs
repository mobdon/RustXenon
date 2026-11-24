//! Emits a guest function as one Rust function: a `loop { match pc { .. } }`
//! with one arm per instruction. Local branches set `pc`; a jump to an
//! address outside the function (a tail call, or a jump table landing
//! elsewhere) goes through `Runtime::call`.
//!
//! This is the simple layout used by the per-instruction tests. Step 2 adds
//! basic-block grouping, function discovery and splitting of huge functions.

use std::fmt::Write as _;

use crate::ppc::{translate, Func};

/// Prelude a generated file needs before any emitted function.
pub const PRELUDE: &str =
    "#[allow(unused_imports)]\nuse xenon_rt::{fpu, ops, vmx, Context, Runtime};\n";

/// Emits `words` (starting at guest address `start`) as `pub fn {name}`.
/// Falling off the end returns to the caller.
pub fn function(name: &str, start: u32, words: &[u32]) -> String {
    let end = start + 4 * words.len() as u32;
    let func = Func { start, end };
    let mut s = String::new();
    let _ = writeln!(
        s,
        "#[allow(unused_mut, unused_variables, unreachable_code, unused_braces, clippy::all)]"
    );
    let _ = writeln!(
        s,
        "#[allow(unused_mut, unused_variables, unreachable_code, unused_braces, clippy::all)]
pub fn {name}(rt: &Runtime, c: &mut Context) {{"
    );
    let _ = writeln!(s, "    let mut pc: u32 = {start:#010x};");
    let _ = writeln!(s, "    loop {{");
    let _ = writeln!(s, "        match pc {{");
    for (i, &w) in words.iter().enumerate() {
        let pc = start + 4 * i as u32;
        let ins = translate(w, pc, &func);
        let _ = writeln!(s, "            // {w:08X} {}", ins.name);
        let fall = if ins.ends {
            String::new()
        } else {
            format!(" pc = {:#010x};", pc + 4)
        };
        let _ = writeln!(s, "            {pc:#010x} => {{ {}{fall} }}", ins.code);
    }
    let _ = writeln!(s, "            {end:#010x} => return,");
    let _ = writeln!(s, "            _ => {{ rt.call(c, pc); return; }}");
    let _ = writeln!(s, "        }}");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s
}

/// Emits one guest function of a whole executable: one match arm per basic
/// block. `extra` adds block starts the instructions do not show (jump-table
/// targets). Falling off the end is a tail call into the next function.
/// A function longer than `max_part` instructions becomes several Rust
/// functions over consecutive block ranges plus a dispatcher, so the host
/// compiler never sees one enormous function.
pub fn block_function(
    name: &str,
    start: u32,
    words: &[u32],
    extra: &[u32],
    max_part: usize,
) -> String {
    let end = start + 4 * words.len() as u32;
    let func = Func { start, end };
    let ins: Vec<_> = words
        .iter()
        .enumerate()
        .map(|(i, &w)| translate(w, start + 4 * i as u32, &func))
        .collect();
    let mut starts = std::collections::BTreeSet::from([start]);
    for (i, x) in ins.iter().enumerate() {
        starts.extend(x.targets.iter().copied().filter(|t| func.local(*t)));
        if x.ends && i + 1 < ins.len() {
            starts.insert(start + 4 * (i as u32 + 1));
        }
    }
    starts.extend(
        extra
            .iter()
            .copied()
            .filter(|&t| func.local(t) && t % 4 == 0),
    );

    // Blocks as (first instruction index, one-past-last), cut into parts.
    let idx: Vec<usize> = starts.iter().map(|&a| ((a - start) / 4) as usize).collect();
    let mut parts: Vec<Vec<(usize, usize)>> = vec![vec![]];
    let mut size = 0;
    for (k, &b) in idx.iter().enumerate() {
        let e = idx.get(k + 1).copied().unwrap_or(ins.len());
        if size >= max_part {
            parts.push(vec![]);
            size = 0;
        }
        parts.last_mut().unwrap().push((b, e));
        size += e - b;
    }

    let arms = |s: &mut String, blocks: &[(usize, usize)]| {
        for &(b, e) in blocks {
            let _ = writeln!(s, "            {:#010x} => {{", start + 4 * b as u32);
            for (i, x) in ins.iter().enumerate().take(e).skip(b) {
                let _ = writeln!(s, "                // {:08X} {}", words[i], x.name);
                let _ = writeln!(s, "                {}", x.code);
            }
            if !ins[e - 1].ends {
                let _ = writeln!(s, "                pc = {:#010x};", start + 4 * e as u32);
            }
            let _ = writeln!(s, "            }}");
        }
        let _ = writeln!(
            s,
            "            {end:#010x} => {{ rt.call(c, {end:#010x}); return; }}"
        );
    };
    let range = |p: &[(usize, usize)]| {
        let a = start + 4 * p[0].0 as u32;
        let b = start + 4 * p.last().unwrap().1 as u32;
        (a, b)
    };

    let mut s = String::new();
    if parts.len() == 1 {
        let _ = writeln!(
            s,
            "#[allow(unused_mut, unused_variables, unreachable_code, unused_braces, clippy::all)]
pub fn {name}(rt: &Runtime, c: &mut Context) {{"
        );
        let _ = writeln!(s, "    let mut pc: u32 = {start:#010x};");
        let _ = writeln!(s, "    loop {{\n        match pc {{");
        arms(&mut s, &parts[0]);
        let _ = writeln!(s, "            _ => {{ rt.call(c, pc); return; }}");
        let _ = writeln!(s, "        }}\n    }}\n}}");
        return s;
    }
    // Parts leave through `next`: another part's address, or u32::MAX for a
    // guest return (every `return;` in translated code).
    let _ = writeln!(
        s,
        "#[allow(unused_mut, unused_variables, unreachable_code, unused_braces, clippy::all)]
pub fn {name}(rt: &Runtime, c: &mut Context) {{"
    );
    let _ = writeln!(s, "    let mut next: u32 = {start:#010x};");
    let _ = writeln!(s, "    loop {{\n        match next {{");
    for (k, p) in parts.iter().enumerate() {
        let (a, b) = range(p);
        let _ = writeln!(
            s,
            "            {a:#010x}..={:#010x} => {name}_p{k}(rt, c, &mut next),",
            b - 1
        );
    }
    let _ = writeln!(s, "            _ => return,\n        }}\n    }}\n}}");
    for (k, p) in parts.iter().enumerate() {
        let (a, b) = range(p);
        let _ = writeln!(
            s,
            "#[allow(unused_mut, unused_variables, unreachable_code, unused_braces, clippy::all)]
fn {name}_p{k}(rt: &Runtime, c: &mut Context, next: &mut u32) {{"
        );
        let _ = writeln!(s, "    let mut pc: u32 = *next;\n    *next = u32::MAX;");
        let _ = writeln!(s, "    loop {{\n        match pc {{");
        arms(&mut s, p);
        let _ = writeln!(
            s,
            "            x if ({start:#010x}..{end:#010x}).contains(&x) && !({a:#010x}..{b:#010x}).contains(&x) => {{ *next = x; return; }}"
        );
        let _ = writeln!(s, "            _ => {{ rt.call(c, pc); return; }}");
        let _ = writeln!(s, "        }}\n    }}\n}}");
    }
    s
}
