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
    let _ = writeln!(s, "pub fn {name}(rt: &mut Runtime, c: &mut Context) {{");
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
