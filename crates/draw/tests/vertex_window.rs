//! Vertex windows: each draw uploads only the guest vertex words its
//! fetches read for its index range, not every fetch constant's whole
//! buffer, and still draws the same.

mod common;

use common::*;
use xenon_gpu::regs::*;

const PRIM_RECT_LIST: u32 = 8;
const RED: [u8; 4] = [255, 0, 0, 255];
/// A 1 MiB vertex buffer: 65536 `vs_color` vertices of 16 bytes.
const BIG: u32 = 0x10_0000;
const BIG_DWORDS: u32 = 0x4_0000;

fn big_memory() -> xenon_gpu::SliceMemory {
    xenon_gpu::SliceMemory {
        base: 0,
        data: vec![0; 0x40_0000],
    }
}

/// Writes `vs_color` vertex `i` of the buffer at BIG.
fn vertex(mem: &mut xenon_gpu::SliceMemory, i: u32, x: f32, y: f32) {
    mem.write_words(BIG + 16 * i, &color_verts(&[(x, y, 0.0, RED)]));
}

/// A full-target rectangle from vertices `i`, `j` and `k` of the buffer.
fn full_rect(mem: &mut xenon_gpu::SliceMemory, [i, j, k]: [u32; 3]) {
    vertex(mem, i, 0.0, 0.0);
    vertex(mem, j, W as f32, 0.0);
    vertex(mem, k, 0.0, H as f32);
}

/// Draws `s` and checks it filled the target red; returns the guest vertex
/// bytes uploaded, or None without an adapter.
fn draw(mem: &mut xenon_gpu::SliceMemory, s: &Stream) -> Option<u64> {
    let mut d = drawer()?;
    let report = run(&mut d, s, mem);
    assert_eq!(report.draws, 1, "{report:?}");
    assert!(report.unsupported.is_empty(), "{report:?}");
    let img = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    assert_eq!(px(&img, W, 40, 40), RED);
    Some(report.timings.guest_bytes)
}

fn stream() -> Stream {
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, BIG, BIG_DWORDS);
    s
}

/// Three vertices of a 1 MiB buffer upload their 48 bytes.
#[test]
fn three_vertices_of_a_big_buffer() {
    let mut mem = big_memory();
    full_rect(&mut mem, [0, 1, 2]);
    let mut s = stream();
    s.draw_auto(PRIM_RECT_LIST, 3);
    if let Some(bytes) = draw(&mut mem, &s) {
        assert_eq!(bytes, 48);
    }
}

/// Indices 1000, 1003 and 1010 upload vertices 1000..=1010.
#[test]
fn indexed_draw_uploads_its_index_range() {
    let mut mem = big_memory();
    full_rect(&mut mem, [1010, 1000, 1003]);
    // Big-endian 16-bit indices (k8in16 swaps them back).
    mem.write_words(INDICES, &[(1010 << 16) | 1000, 1003 << 16]);
    let mut s = stream();
    s.draw_dma16(PRIM_RECT_LIST, 3);
    if let Some(bytes) = draw(&mut mem, &s) {
        assert_eq!(bytes, 11 * 16);
    }
}

/// VGT_INDX_OFFSET moves the auto indices 0..3 to vertices 5000..5003.
#[test]
fn index_offset_moves_the_window() {
    let mut mem = big_memory();
    full_rect(&mut mem, [5000, 5001, 5002]);
    let mut s = stream();
    s.regs(VGT_INDX_OFFSET, &[5000])
        .draw_auto(PRIM_RECT_LIST, 3);
    if let Some(bytes) = draw(&mut mem, &s) {
        assert_eq!(bytes, 48);
    }
}

/// Positions (3 floats) through vertex fetch constant 0 and colours
/// (8_8_8_8) through constant 1, each with its own stride.
fn vs_two_buffers() -> Vec<u32> {
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let mut pos = VertexFetch::full(1, 0, 0, VertexFormat::F32_32_32_FLOAT, 3, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let col = VertexFetch::full(2, 0, 1, VertexFormat::F8_8_8_8, 1, 0);
    let fetch = |v| Instr::Fetch(FetchInstr::Vertex(v));
    let r = |n| AluSrc::temp(n, SWZ_XYZW);
    assemble(
        &[exec(0, 4, true)],
        &[fetch(pos), fetch(col), export(62, r(1)), export(0, r(2))],
    )
}

/// Two buffers megabytes apart upload their vertices, not the gap, in
/// either address order.
#[test]
fn far_apart_buffers_upload_only_their_vertices() {
    for colours in [0x3F_0000, 0x8_0000] {
        let mut mem = big_memory();
        let (wf, hf) = (W as f32, H as f32);
        let pos = [0.0, 0.0, 0.0, wf, 0.0, 0.0, 0.0, hf, 0.0].map(f);
        mem.write_words(BIG, &pos);
        mem.write_words(colours, &[rgba(RED); 3]);
        let mut s = Stream::default();
        s.setup()
            .shader(false, &vs_two_buffers())
            .shader(true, &ps_interp())
            .vertex_fetch(0, BIG, BIG_DWORDS)
            .vertex_fetch(1, colours, 0x1000)
            .draw_auto(PRIM_RECT_LIST, 3);
        if let Some(bytes) = draw(&mut mem, &s) {
            assert_eq!(bytes, 36 + 12, "colours at 0x{colours:X}");
        }
    }
}

/// A colour constant two words long read for three vertices: the third
/// read passes its buffer, so the window is one span that holds the
/// guest's word there instead of a part next to the positions.
#[test]
fn read_past_a_buffer_finds_the_guest_word() {
    let colours = 0x8_0000;
    let mut mem = big_memory();
    let (wf, hf) = (W as f32, H as f32);
    let pos = [0.0, 0.0, 0.0, wf, 0.0, 0.0, 0.0, hf, 0.0].map(f);
    mem.write_words(BIG, &pos);
    mem.write_words(colours, &[rgba(RED); 3]);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_two_buffers())
        .shader(true, &ps_interp())
        .vertex_fetch(0, BIG, BIG_DWORDS)
        .vertex_fetch(1, colours, 2)
        .draw_auto(PRIM_RECT_LIST, 3);
    if let Some(bytes) = draw(&mut mem, &s) {
        assert_eq!(bytes, u64::from(BIG - colours) + 36);
    }
}

/// A colour fetch indexed by a copy of the vertex index is unbounded: its
/// buffer is one word, so vertices 1 and 2 read past it. Those reads find
/// the guest's red words after it, not the positions staged next to it.
#[test]
fn unbounded_read_past_a_buffer_finds_the_guest_word() {
    use xenon_shader::asm::*;
    use xenon_shader::ucode::*;
    let mut pos = VertexFetch::full(1, 0, 0, VertexFormat::F32_32_32_FLOAT, 3, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let col = VertexFetch::full(2, 3, 1, VertexFormat::F8_8_8_8, 1, 0);
    let fetch = |v| Instr::Fetch(FetchInstr::Vertex(v));
    let r = |n| AluSrc::temp(n, SWZ_XYZW);
    let copy = AluInstr::vector(VectorOp::Max, 3, mask("x"), &[r(0), r(0)]);
    let vs = assemble(
        &[exec(0, 5, true)],
        &[
            Instr::Alu(copy),
            fetch(pos),
            fetch(col),
            export(62, r(1)),
            export(0, r(2)),
        ],
    );
    let colours = 0x8_0000;
    let mut mem = big_memory();
    let (wf, hf) = (W as f32, H as f32);
    let pos = [0.0, 0.0, 0.0, wf, 0.0, 0.0, 0.0, hf, 0.0].map(f);
    mem.write_words(BIG, &pos);
    mem.write_words(colours, &[rgba(RED); 3]);
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs)
        .shader(true, &ps_interp())
        .vertex_fetch(0, BIG, BIG_DWORDS)
        .vertex_fetch(1, colours, 1)
        .draw_auto(PRIM_RECT_LIST, 3);
    if let Some(bytes) = draw(&mut mem, &s) {
        assert_eq!(bytes, u64::from(BIG - colours) + 36);
    }
}
