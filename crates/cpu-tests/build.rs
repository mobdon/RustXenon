//! Translates every case in `data/*.txt` with `xenon-recomp` and writes one
//! Rust function plus one `#[test]` per case into `OUT_DIR/cases.rs`.

use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut out = String::from(xenon_recomp::emit::PRELUDE);
    let mut n = 0;
    for file in ["oracle.txt", "hand.txt"] {
        let path = dir.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let stem = file.trim_end_matches(".txt");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('|');
            let name = parts.next().unwrap().trim();
            let words: Vec<u32> = parts
                .next()
                .unwrap()
                .split_whitespace()
                .map(|w| u32::from_str_radix(w, 16).unwrap())
                .collect();
            let f = format!("exec_{n}");
            out.push_str(&xenon_recomp::emit::function(
                &f,
                xenon_cpu_tests_start(),
                &words,
            ));
            let test = format!("{stem}_{name}").replace(['.', '-'], "_");
            let _ = writeln!(
                out,
                "#[test]\nfn {test}() {{ xenon_cpu_tests::check({line:?}, {f}); }}\n"
            );
            n += 1;
        }
    }
    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("cases.rs"),
        out,
    )
    .unwrap();

    // Small functions for the call tests in tests/calls.rs.
    let funcs: &[(&str, &[u32])] = &[
        // bl 0x82100000; addi r3, r3, 1; blr
        ("call_bl", &[0x4810_0001, 0x3863_0001, 0x4E80_0020]),
        // b 0x82100000 (tail call)
        ("call_tail", &[0x4810_0000]),
        // mtctr r4; bctrl; blr
        ("call_bctrl", &[0x7C89_03A6, 0x4E80_0421, 0x4E80_0020]),
        // mtlr r4; blrl; blr
        ("call_blrl", &[0x7C88_03A6, 0x4E80_0021, 0x4E80_0020]),
        // mtctr r4; bctr; li r3, 1; li r3, 2; blr   (jump table landing inside)
        (
            "jump_local",
            &[
                0x7C89_03A6,
                0x4E80_0420,
                0x3860_0001,
                0x3860_0002,
                0x4E80_0020,
            ],
        ),
        // twi 31, r0, 0
        ("trap", &[0x0FE0_0000]),
        // tw 4, r3, r4 (trap if equal)
        ("trap_eq", &[0x7C83_2008]),
        // an all-zero word is not an instruction
        ("unknown", &[0x0000_0000]),
        // sc
        ("syscall", &[0x4400_0002]),
    ];
    let mut calls = String::from(xenon_recomp::emit::PRELUDE);
    for (name, words) in funcs {
        calls.push_str(&xenon_recomp::emit::function(
            name,
            xenon_cpu_tests_start(),
            words,
        ));
    }
    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("calls.rs"),
        calls,
    )
    .unwrap();
    // Guest programs for the multi-thread tests in tests/threads.rs.
    // Assembled with llvm-mc (branch offsets filled in by hand).
    let funcs: &[(&str, &[u32])] = &[
        // Adds 1 to the word at r3, r4 times:
        // mtctr r4; 1: lwarx r5,0,r3; addi r5,r5,1; stwcx. r5,0,r3; bne- 1b;
        // bdnz 1b; blr
        (
            "atomic_add32",
            &[
                0x7C89_03A6,
                0x7CA0_1828,
                0x38A5_0001,
                0x7CA0_192D,
                0x40C2_FFF4,
                0x4200_FFF0,
                0x4E80_0020,
            ],
        ),
        // The same on a doubleword with ldarx / stdcx.
        (
            "atomic_add64",
            &[
                0x7C89_03A6,
                0x7CA0_18A8,
                0x38A5_0001,
                0x7CA0_19AD,
                0x40C2_FFF4,
                0x4200_FFF0,
                0x4E80_0020,
            ],
        ),
        // Spin lock at r3 guarding a plain counter at r3+4, r4 times:
        // mtctr r4
        // 1: lwarx r5,0,r3; cmpwi r5,0; bne- 1b; li r5,1; stwcx. r5,0,r3;
        //    bne- 1b; isync
        //    lwz r6,4(r3); addi r6,r6,1; stw r6,4(r3)
        //    lwsync; li r5,0; stw r5,0(r3)
        //    bdnz 1b
        // blr
        (
            "spin_lock_count",
            &[
                0x7C89_03A6,
                0x7CA0_1828,
                0x2C05_0000,
                0x40C2_FFF8,
                0x38A0_0001,
                0x7CA0_192D,
                0x40C2_FFEC,
                0x4C00_012C,
                0x80C3_0004,
                0x38C6_0001,
                0x90C3_0004,
                0x7C20_04AC,
                0x38A0_0000,
                0x90A3_0000,
                0x4200_FFCC,
                0x4E80_0020,
            ],
        ),
    ];
    let mut threads = String::from(xenon_recomp::emit::PRELUDE);
    for (name, words) in funcs {
        threads.push_str(&xenon_recomp::emit::function(
            name,
            xenon_cpu_tests_start(),
            words,
        ));
    }
    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("threads.rs"),
        threads,
    )
    .unwrap();
    // Whole-executable layout (tests/blocks.rs): basic blocks, split into
    // parts of at most two instructions so every part boundary is crossed.
    let start = xenon_cpu_tests_start();
    let funcs: &[(&str, &[u32], &[u32])] = &[
        // li r3,0; li r4,10; mtctr r4; 1: addi r3,r3,1; bdnz 1b; blr
        (
            "count_loop",
            &[
                0x3860_0000,
                0x3880_000A,
                0x7C89_03A6,
                0x3863_0001,
                0x4200_FFFC,
                0x4E80_0020,
            ],
            &[],
        ),
        // li r3,5, then falls into the next function
        ("fall_off", &[0x3860_0005], &[]),
        // mtctr r3; bctr; li r4,1; blr; li r4,2; blr (table targets +8, +16)
        (
            "table",
            &[
                0x7C69_03A6,
                0x4E80_0420,
                0x3880_0001,
                0x4E80_0020,
                0x3880_0002,
                0x4E80_0020,
            ],
            &[start + 8, start + 16],
        ),
    ];
    let mut blocks = String::from(xenon_recomp::emit::PRELUDE);
    for (name, words, extra) in funcs {
        blocks.push_str(&xenon_recomp::emit::block_function(
            name, start, words, extra, 2,
        ));
    }
    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("blocks.rs"),
        blocks,
    )
    .unwrap();
    // Keep the build script's view of the translator fresh.
    println!("cargo:rerun-if-changed=../recomp/src");
}

/// Guest address of the first instruction of every case (matches `lib.rs`).
fn xenon_cpu_tests_start() -> u32 {
    0x8200_0000
}
