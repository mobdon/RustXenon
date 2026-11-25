//! The opt-in shader dump writes each drawn shader once, with its listing,
//! WGSL and first draw state, and honours the hash filter. Needs a wgpu
//! adapter like the other drawer tests.

mod common;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use common::*;
use xenon_draw::ShaderDump;
use xenon_gpu::regs::*;

const PRIM_TRI_LIST: u32 = 4;

/// The program hash the drawer names files by.
fn hash(words: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    words.hash(&mut h);
    h.finish()
}

fn fresh_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xenon-shader-dump-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn files(dir: &Path) -> HashSet<String> {
    std::fs::read_dir(dir)
        .map(|r| {
            r.map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect()
        })
        .unwrap_or_default()
}

/// Two draws with one vertex shader and two pixel shaders, the second
/// pixel shader drawn twice.
fn draw_twice(dump: ShaderDump) -> Option<()> {
    let mut d = drawer()?;
    d.shader_dump = Some(dump);
    let mut mem = memory();
    let white = [255; 4];
    mem.write_words(
        VERTS,
        &color_verts(&[
            (0.0, 0.0, 0.0, white),
            (64.0, 0.0, 0.0, white),
            (0.0, 64.0, 0.0, white),
        ]),
    );
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, VERTS, 12)
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[f(0.25), f(0.5), f(1.0), f(2.0)],
        )
        .draw_auto(PRIM_TRI_LIST, 3)
        .shader(true, &ps_const())
        .draw_auto(PRIM_TRI_LIST, 3)
        .regs(
            SHADER_CONSTANT_FLOAT + 256 * 4,
            &[f(9.0), f(9.0), f(9.0), f(9.0)],
        )
        .draw_auto(PRIM_TRI_LIST, 3);
    let report = run(&mut d, &s, &mut mem);
    assert_eq!(report.draws, 3, "{report:?}");
    Some(())
}

#[test]
fn dumps_each_shader_once_with_draw_state() {
    let dir = fresh_dir("all");
    if draw_twice(ShaderDump::new(&dir, None)).is_none() {
        return;
    }
    let (vs, pi, pc) = (hash(&vs_color()), hash(&ps_interp()), hash(&ps_const()));
    let mut want = HashSet::new();
    for (h, stage) in [(vs, "vs"), (pi, "ps"), (pc, "ps")] {
        for ext in ["ucode", "txt", "wgsl"] {
            want.insert(format!("{h:016x}.{stage}.{ext}"));
        }
    }
    want.insert(format!("{pi:016x}.ps.draw.txt"));
    want.insert(format!("{pc:016x}.ps.draw.txt"));
    assert_eq!(files(&dir), want);

    let read = |name: String| std::fs::read_to_string(dir.join(name)).unwrap();
    let ucode = std::fs::read(dir.join(format!("{vs:016x}.vs.ucode"))).unwrap();
    let words: Vec<u32> = ucode
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(words, vs_color());
    let txt = read(format!("{vs:016x}.vs.txt"));
    assert!(
        txt.contains("vfetch_full r1.xyz1, r0.x, vf0 format 32_32_32_FLOAT(57)"),
        "{txt}"
    );
    assert!(txt.contains("max oPos.xyzw, r1, r1"), "{txt}");
    assert!(read(format!("{vs:016x}.vs.wgsl")).contains("fn vs_main"));
    assert!(read(format!("{pc:016x}.ps.wgsl")).contains("fn ps_main"));

    // The first draw's state: c0 as set before it, not the later 9s.
    let state = read(format!("{pc:016x}.ps.draw.txt"));
    for want in [
        &format!("vs {vs:016x} ps {pc:016x}"),
        "vfetch_mini r2.xyzw format 8_8_8_8(6)",
        "vf0 VertexFetch {",
        "RB_COLOR_INFO[0]",
        "ps c0 [0.25, 0.5, 1.0, 2.0] | 3e800000 3f000000 3f800000 40000000",
    ] {
        assert!(state.contains(want), "missing {want:?} in\n{state}");
    }
    assert!(!state.contains("ps c1 "), "{state}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hash_filter_keeps_listed_pixel_shader_and_its_vertex_shader() {
    let dir = fresh_dir("filter");
    let pc = hash(&ps_const());
    let dump = ShaderDump::parse(&format!("{},{pc:x}", dir.display())).unwrap();
    if draw_twice(dump).is_none() {
        return;
    }
    let vs = hash(&vs_color());
    let mut want: HashSet<String> = ["ucode", "txt", "wgsl"]
        .iter()
        .flat_map(|ext| [format!("{vs:016x}.vs.{ext}"), format!("{pc:016x}.ps.{ext}")])
        .collect();
    want.insert(format!("{pc:016x}.ps.draw.txt"));
    assert_eq!(files(&dir), want);
    let _ = std::fs::remove_dir_all(&dir);

    // A filter matching nothing writes nothing.
    let dir = fresh_dir("none");
    let dump = ShaderDump::parse(&format!("{},0x1", dir.display())).unwrap();
    if draw_twice(dump).is_some() {
        assert!(files(&dir).is_empty());
    }
}

#[test]
fn parse_rejects_bad_values() {
    assert!(ShaderDump::parse("").is_err());
    assert!(ShaderDump::parse("/tmp/x,nothex").is_err());
    let d = ShaderDump::parse("/tmp/x, 0xAB ,cd").unwrap();
    assert_eq!(d.only, Some(HashSet::from([0xAB, 0xCD])));
    assert!(ShaderDump::parse("/tmp/x").unwrap().only.is_none());
}
