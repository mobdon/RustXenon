//! Resolves kept on the GPU: a texture fetched from a resolve destination
//! is decoded on the GPU from the bytes the resolve wrote there, and draws
//! exactly what the guest path (read guest memory, decode, upload) draws.
//! Each test compares with a fresh drawer that only sees guest memory.

mod common;

use common::*;
use xenon_draw::RenderTargetKey;
use xenon_gpu::regs::*;
use xenon_gpu::SliceMemory;

const PRIM_RECT_LIST: u32 = 8;
const PRIM_QUAD_LIST: u32 = 13;
const PAINT: u32 = VERTS;
const SMALL: u32 = VERTS + 0x100;
const SCREEN: u32 = VERTS + 0x200;
/// The target `sample` draws into.
const OUT: RenderTargetKey = RenderTargetKey {
    base_tiles: 256,
    ..COLOR_KEY
};
const IDENTITY: u16 = (1 << 3) | (2 << 6) | (3 << 9);
/// B, G, R, constant 1.
const BGR1: u16 = 2 | (1 << 3) | (5 << 9);

/// Fetch constant of a tiled 64x64 texture at DEST, point sampled.
fn fetch(format: u32, endian: u32, swizzle: u16) -> [u32; 6] {
    [
        2 | (2 << 10) | (2 << 13) | ((W >> 5) << 22) | (1 << 31),
        format | (endian << 6) | DEST,
        (W - 1) | ((H - 1) << 13),
        u32::from(swizzle) << 1,
        0,
        1 << 9,
    ]
}

/// Vertex data: a four-colour gradient quad over the whole target, a blue
/// 32x32 rectangle, and a whole-target quad with texture coordinates.
fn vertices(mem: &mut SliceMemory) {
    let (w, h) = (W as f32, H as f32);
    mem.write_words(
        PAINT,
        &color_verts(&[
            (0.0, 0.0, 0.0, [255, 0, 0, 255]),
            (w, 0.0, 0.0, [0, 255, 0, 128]),
            (w, h, 0.0, [0, 0, 255, 0]),
            (0.0, h, 0.0, [255, 255, 255, 64]),
        ]),
    );
    let blue = [10, 20, 230, 255];
    mem.write_words(
        SMALL,
        &color_verts(&[
            (0.0, 0.0, 0.0, blue),
            (32.0, 0.0, 0.0, blue),
            (0.0, 32.0, 0.0, blue),
        ]),
    );
    let quad: Vec<u32> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        .iter()
        .flat_map(|&(u, v): &(f32, f32)| [f(u * w), f(v * h), f(0.0), f(u), f(v)])
        .collect();
    mem.write_words(SCREEN, &quad);
}

/// Draws the gradient into the COLOR_KEY target.
fn paint(s: &mut Stream) -> &mut Stream {
    s.shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, PAINT, 16)
        .draw_auto(PRIM_QUAD_LIST, 4)
}

/// Draws the blue rectangle into the COLOR_KEY target.
fn paint_small(s: &mut Stream) -> &mut Stream {
    s.shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, SMALL, 12)
        .draw_auto(PRIM_RECT_LIST, 3)
}

/// Resolves (0,0)-(w,h) of the COLOR_KEY target to DEST as 8_8_8_8, with
/// its rectangle at RECT + `w` (so resolves of other widths in
/// one stream keep their own).
fn resolve(s: &mut Stream, mem: &mut SliceMemory, w: u32, h: u32) {
    resolve_to(s, mem, w, h, DEST);
}

/// [`resolve`] to `dest`.
fn resolve_to(s: &mut Stream, mem: &mut SliceMemory, w: u32, h: u32, dest: u32) {
    let at = RECT + w;
    let (wf, hf) = (w as f32, h as f32);
    mem.write_words(at, &[f(0.0), f(0.0), f(wf), f(0.0), f(wf), f(hf)]);
    s.vertex_fetch(0, at, 6)
        .regs(RB_MODECONTROL, &[6])
        .regs(
            RB_COPY_CONTROL,
            &[COPY_CONVERT, dest, W | (H << 16), dest_info(6, 0, false)],
        )
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_MODECONTROL, &[4]);
}

/// Draws texture slot 1 = `fetch` texel for texel into OUT, then makes
/// COLOR_KEY current again.
fn sample(s: &mut Stream, fetch: [u32; 6]) -> &mut Stream {
    s.regs(RB_COLOR_INFO, &[OUT.base_tiles])
        .shader(false, &vs_uv())
        .shader(true, &ps_texture(1))
        .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
        .vertex_fetch(0, SCREEN, 20)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .regs(RB_COLOR_INFO, &[COLOR_BASE])
}

/// What a drawer that never resolved draws from `mem` with `fetch`.
fn guest_path(mem: &SliceMemory, fetch: [u32; 6]) -> Vec<u8> {
    let mut d = drawer().expect("a drawer");
    let mut mem = mem.clone();
    let mut s = Stream::default();
    sample(s.setup(), fetch);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.draws, 1, "{r:?}");
    assert_eq!(r.timings.textures_kept, 0, "{r:?}");
    d.read_render_target(OUT, 0, 0, W, H).unwrap()
}

/// The resolved gradient sampled in the same replay, at several endians and
/// swizzles: decoded on the GPU (nothing read, hashed or uploaded), the same
/// pixels as the guest path.
#[test]
fn same_replay_matches_the_guest_path() {
    for (endian, swizzle) in [(2, IDENTITY), (1, BGR1), (3, IDENTITY), (0, BGR1)] {
        let f = fetch(6, endian, swizzle);
        let Some(mut d) = drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        paint(s.setup());
        resolve(&mut s, &mut mem, W, H);
        sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.kept_bytes),
            (1, u64::from(W * H * 4)),
            "{r:?}"
        );
        assert_eq!((t.textures_uploaded, t.hashed_bytes), (0, 0), "{r:?}");
        let kept = d.read_render_target(OUT, 0, 0, W, H).unwrap();
        drop(d);
        // Guest memory has the resolve: the guest path draws from it.
        assert_eq!(kept, guest_path(&mem, f), "endian {endian}");
        assert_ne!(px(&kept, W, 1, 1), px(&kept, W, 62, 62));
    }
}

/// A resolve sampled by the next replay is still on the GPU: guest memory
/// under it is only hashed to check the CPU did not write it.
#[test]
fn next_replay_matches_the_guest_path() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    run(&mut d, &s, &mut mem);
    let mut s = Stream::default();
    sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    let t = &r.timings;
    assert_eq!(
        (t.textures_kept, t.textures_uploaded, t.hashed_bytes),
        (1, 0, 0),
        "{r:?}"
    );
    assert_eq!(t.kept_hashed_bytes, u64::from(W * H * 4), "{r:?}");
    let kept = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(kept, guest_path(&mem, f));
}

/// A CPU write to the destination between replays: the next replay reads
/// guest memory, with the CPU's bytes.
#[test]
fn cpu_write_between_replays_takes_the_guest_path() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    run(&mut d, &s, &mut mem);
    mem.data[DEST as usize..][..256].fill(0x55);
    let mut s = Stream::default();
    sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (0, 1),
        "{r:?}"
    );
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert!(img.chunks(4).any(|p| p == [0x55; 4]));
}

/// A MEM_WRITE over the destination between the resolve and the fetch:
/// the fetch reads guest memory, which has the resolve and then the
/// MEM_WRITE's words (the write-back does not undo them).
#[test]
fn mem_write_after_the_resolve_takes_the_guest_path() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    let mut packet = vec![DEST];
    packet.extend([0x1122_3344; 16]);
    s.0.extend(xenon_gpu::pm4::build::type3(
        xenon_gpu::pm4::PM4_MEM_WRITE,
        &packet,
    ));
    sample(&mut s, f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (0, 1),
        "{r:?}"
    );
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(mem.data[DEST as usize..][..4], [0x44, 0x33, 0x22, 0x11]);
    assert_ne!(mem.data[DEST as usize + 64..][..4], [0; 4]);
    assert_eq!(img, guest_path(&mem, f));
}

/// A fetch of the destination as another format (k_2_10_10_10) is not
/// decoded on the GPU; the same-bits k_8_8_8_8_A view is.
#[test]
fn other_fetch_formats_take_the_guest_path() {
    for (format, kept) in [(7, 0), (14, 1)] {
        let f = fetch(format, 2, IDENTITY);
        let Some(mut d) = drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        paint(s.setup());
        resolve(&mut s, &mut mem, W, H);
        sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert_eq!(r.timings.textures_kept, kept, "{r:?}");
        let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
        drop(d);
        assert_eq!(img, guest_path(&mem, f), "format {format}");
    }
}

/// A resolve of part of a kept texture updates that part: the next fetch
/// decodes again only the rows it wrote, and shows both resolves.
#[test]
fn partial_resolve_updates_the_kept_texture() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    sample(&mut s, f);
    paint_small(&mut s);
    resolve(&mut s, &mut mem, 32, 32);
    sample(&mut s, f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (2, 0),
        "{r:?}"
    );
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert_eq!(px(&img, W, 5, 5), [10, 20, 230, 255]);
    assert_ne!(px(&img, W, 40, 5), [10, 20, 230, 255]);
}

/// Two resolves one under the other, as tiles of one surface (the second
/// at the address of row 32): one texture over both is decoded on the GPU.
#[test]
fn texture_over_two_resolves_matches_the_guest_path() {
    let f = fetch(6, 2, BGR1);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve_to(&mut s, &mut mem, W, 32, DEST);
    paint_small(&mut s);
    resolve_to(&mut s, &mut mem, W, 32, DEST + W * 32 * 4);
    sample(&mut s, f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (1, 0),
        "{r:?}"
    );
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert_ne!(px(&img, W, 5, 5), px(&img, W, 5, 37));
}

/// A texture larger than what the resolve wrote (the rest is older guest
/// memory) is read from guest memory.
#[test]
fn texture_past_the_resolve_takes_the_guest_path() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    mem.data[DEST as usize..][..(W * H * 4) as usize].fill(0x77);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, 32);
    sample(&mut s, f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (0, 1),
        "{r:?}"
    );
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert_eq!(px(&img, W, 5, 50), [0x77; 4]);
}

/// Without write-back the resolve stays on the GPU only: guest memory is
/// untouched after the replay, the fetch still draws it, and a later swap
/// of the destination (a read by the drawer) writes it back first.
#[test]
fn skipped_write_back_happens_when_the_drawer_reads() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    d.skip_resolve_writeback = true;
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    sample(&mut s, f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.timings.textures_kept, 1, "{r:?}");
    let kept = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    assert!(mem.data[DEST as usize..][..(W * H * 4) as usize]
        .iter()
        .all(|&b| b == 0));
    // A replay that only samples it: still kept, still not written.
    let mut s = Stream::default();
    sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.timings.textures_kept, 1, "{r:?}");
    assert_eq!(mem.data[DEST as usize], 0);
    let mut s = Stream::default();
    s.swap_at(DEST, W, H);
    let r = run(&mut d, &s, &mut mem);
    drop(d);
    assert_eq!(r.frames.len(), 1, "{r:?}");
    assert_eq!(kept, guest_path(&mem, f));
    assert_eq!(r.frames[0].rgba8, detiled(&mem, DEST, W, H, W, 4));
    assert_ne!(r.frames[0].rgba8[..4], [0; 4]);
}
