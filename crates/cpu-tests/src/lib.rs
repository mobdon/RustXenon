//! Per-instruction tests for the translator, with no game bytes.
//!
//! A case is one line of `data/*.txt`:
//!
//! ```text
//! name | words | inputs | expected changes | flags
//! ```
//!
//! - `words`: the instruction words (hex), run as one small guest function.
//! - `inputs`: `key=hex` pairs; everything not listed starts at zero (VSCR
//!   starts with NJ set, as on the console).
//! - `expected changes`: every register or memory range that the reference
//!   says changed. Anything not listed must be unchanged, so a translation
//!   that clobbers an unrelated register fails.
//! - `flags`: `ignore_fpscr` for floating point cases, because the FPSCR
//!   exception and class bits are not modelled.
//!
//! Keys: `r0..r31`, `f0..f31` (bits), `v0..v127` (32 hex digits, element 0
//! first), `cr`, `xer`, `ctr`, `lr`, `fpscr`, `vscr`, and `m<addr>=bytes`.

use std::collections::BTreeMap;

use xenon_rt::{Context, GuestFn, GuestMemory, Runtime};

/// Guest address of the case's first instruction.
pub const START: u32 = 0x8200_0000;
/// Scratch memory the cases read and write.
pub const SCRATCH: u32 = 0x0010_0000;
pub const SCRATCH_LEN: u32 = 512;

fn parse_kv(s: &str) -> Vec<(String, String)> {
    s.split_whitespace()
        .map(|kv| {
            let (k, v) = kv
                .split_once('=')
                .unwrap_or_else(|| panic!("bad field {kv:?}"));
            (k.to_string(), v.to_string())
        })
        .collect()
}

fn hex64(v: &str) -> u64 {
    u64::from_str_radix(v, 16).unwrap()
}

fn hex_bytes(v: &str) -> Vec<u8> {
    (0..v.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&v[i..i + 2], 16).unwrap())
        .collect()
}

fn set(c: &mut Context, m: &mut GuestMemory, k: &str, v: &str) {
    if let Some(addr) = k.strip_prefix('m') {
        let addr = u32::from_str_radix(addr, 16).unwrap();
        for (i, b) in hex_bytes(v).into_iter().enumerate() {
            m.write_u8(addr + i as u32, b);
        }
        return;
    }
    match k {
        "cr" => c.cr = hex64(v) as u32,
        "xer" => c.set_xer(hex64(v)),
        "ctr" => c.ctr = hex64(v),
        "lr" => c.lr = hex64(v),
        "fpscr" => c.fpscr = hex64(v) as u32,
        "vscr" => c.vscr = hex64(v) as u32,
        _ => {
            let (kind, n) = k.split_at(1);
            let n: usize = n.parse().unwrap_or_else(|_| panic!("bad key {k}"));
            match kind {
                "r" => c.r[n] = hex64(v),
                "f" => c.f[n] = f64::from_bits(hex64(v)),
                "v" => {
                    let x = u128::from_str_radix(v, 16).unwrap();
                    c.v[n] = [
                        (x >> 96) as u32,
                        (x >> 64) as u32,
                        (x >> 32) as u32,
                        x as u32,
                    ];
                }
                _ => panic!("bad key {k}"),
            }
        }
    }
}

/// Every architecturally visible value, as text for readable diffs.
fn snapshot(c: &Context, m: &GuestMemory) -> BTreeMap<String, String> {
    let mut s = BTreeMap::new();
    for i in 0..32 {
        s.insert(format!("r{i}"), format!("{:x}", c.r[i]));
        s.insert(format!("f{i}"), format!("{:x}", c.f[i].to_bits()));
    }
    for i in 0..128 {
        let v = c.v[i];
        let x = (v[0] as u128) << 96 | (v[1] as u128) << 64 | (v[2] as u128) << 32 | v[3] as u128;
        s.insert(format!("v{i}"), format!("{x:032x}"));
    }
    s.insert("cr".into(), format!("{:x}", c.cr));
    s.insert("xer".into(), format!("{:x}", c.xer()));
    s.insert("ctr".into(), format!("{:x}", c.ctr));
    s.insert("lr".into(), format!("{:x}", c.lr));
    s.insert("fpscr".into(), format!("{:x}", c.fpscr));
    s.insert("vscr".into(), format!("{:x}", c.vscr));
    let mem = &m.bytes()[SCRATCH as usize..(SCRATCH + SCRATCH_LEN) as usize];
    for (i, b) in mem.iter().enumerate() {
        s.insert(format!("m{:x}", SCRATCH as usize + i), format!("{b:02x}"));
    }
    s
}

/// Runs one case and panics with a readable diff if any value is wrong.
pub fn check(line: &str, f: GuestFn) {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    assert!(parts.len() >= 4, "bad case line: {line}");
    let (name, inputs, expect) = (parts[0], parse_kv(parts[2]), parse_kv(parts[3]));
    let flags = parts.get(4).copied().unwrap_or("");

    let mut c = Context::new();
    let mut m = GuestMemory::new((SCRATCH + 0x1000) as usize);
    for (k, v) in &inputs {
        set(&mut c, &mut m, k, v);
    }
    let before = snapshot(&c, &m);

    // Expected state: inputs with the reference's changes applied.
    let (mut ec, mut em) = (c.clone(), GuestMemory::new((SCRATCH + 0x1000) as usize));
    em.bytes_mut().copy_from_slice(m.bytes());
    for (k, v) in &expect {
        set(&mut ec, &mut em, k, v);
    }
    let want = snapshot(&ec, &em);

    let mut rt = Runtime::new(m);
    f(&mut rt, &mut c);
    let got = snapshot(&c, &rt.mem);

    let mut bad = Vec::new();
    for (k, w) in &want {
        if k == "fpscr" && flags.contains("ignore_fpscr") {
            continue;
        }
        let g = &got[k];
        if g != w {
            bad.push(format!("  {k}: got {g}, want {w} (was {})", before[k]));
        }
    }
    if !bad.is_empty() {
        panic!("{name} [{}]:\n{}", parts[1], bad.join("\n"));
    }
}
