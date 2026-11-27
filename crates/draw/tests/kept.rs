//! Resolves kept on the GPU: a texture fetched from a resolve destination
//! is decoded on the GPU from the bytes the resolve wrote there, and draws
//! exactly what the guest path (read guest memory, decode, upload) draws.
//! Each test compares with a fresh drawer that only sees guest memory.

mod common;

use common::*;
use xenon_draw::{RenderTargetKey, TargetFormat};
use xenon_gpu::regs::*;
use xenon_gpu::SliceMemory;

const PRIM_RECT_LIST: u32 = 8;
const PRIM_QUAD_LIST: u32 = 13;
const PAINT: u32 = VERTS;
const SMALL: u32 = VERTS + 0x100;
const SCREEN: u32 = VERTS + 0x200;
/// Four 8x8 rectangles in a row from the top-left corner, 0x40 apart.
const SPECIALS: u32 = VERTS + 0x300;
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
    for i in 0..4 {
        let x = 8.0 * i as f32;
        mem.write_words(
            SPECIALS + 0x40 * i,
            &color_verts(&[
                (x, 0.0, 0.0, blue),
                (x + 8.0, 0.0, 0.0, blue),
                (x, 8.0, 0.0, blue),
            ]),
        );
    }
}

/// Draws the gradient into the COLOR_KEY target.
fn paint(s: &mut Stream) -> &mut Stream {
    s.shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, PAINT, 16)
        .draw_auto(PRIM_QUAD_LIST, 4)
}

/// The gradient, then rectangles of NaNs of both signs, infinities, signed
/// zeros, negatives and denormals in half precision.
fn paint_specials(s: &mut Stream) -> &mut Stream {
    let consts: [[u32; 4]; 4] = [
        [0xFFC0_0000, 0x7FC0_0000, 0xFF80_0000, 0x8000_0000],
        [f(-1.5), f(-65504.0), f(-(2f32).powi(-24)), f(-0.25)],
        [0xFFD0_0000, f(0.5), f(-3.0), 0x7F80_0000],
        [f(-(2f32).powi(-20)), f(1.0), 0xFFC0_0001, f(-100.0)],
    ];
    paint(s).shader(true, &ps_const());
    for (i, c) in (0..).zip(consts) {
        // Pixel c0 is float constant 256.
        s.regs(SHADER_CONSTANT_FLOAT + 256 * 4, &c)
            .vertex_fetch(0, SPECIALS + 0x40 * i, 12)
            .draw_auto(PRIM_RECT_LIST, 3);
    }
    s
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

/// Kept-texture decodes are recorded in the batch after the resolves they
/// read, not flushed: the same pixels as the guest path, in one batch.
/// (`XENON_DECODE_IN_BATCH=0` flushes before each instead.)
#[test]
fn decodes_join_the_batch() {
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
    let t = &r.timings;
    assert_eq!((t.textures_kept, t.textures_uploaded), (2, 0), "{r:?}");
    // The copies and the decodes after them are recorded in one batch:
    // that batch and the end-of-replay write-back.
    assert_eq!(t.submits_by_site, [0, 0, 0, 2], "{t:?}");
    assert_eq!(t.submits, 2, "{t:?}");
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert_eq!(px(&img, W, 5, 5), [10, 20, 230, 255]);
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

/// Guest memory whose CPU writes `fill` over `at` while the drawer reads
/// the gradient's vertices (as a guest CPU thread running beside the
/// replay may).
struct Racing {
    mem: std::cell::RefCell<SliceMemory>,
    at: std::ops::Range<usize>,
    armed: std::cell::Cell<bool>,
}

impl xenon_gpu::GuestRead for Racing {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        self.mem.borrow().read_u32(addr)
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        if addr == PAINT && self.armed.replace(false) {
            self.mem.borrow_mut().data[self.at.clone()].fill(0x55);
        }
        self.mem.borrow().read_bytes(addr, len)
    }
}

impl xenon_draw::GuestMemory for Racing {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        self.mem.get_mut().write_bytes(addr, bytes)
    }
}

/// A CPU write during a replay to the part of a kept buffer that the
/// replay did not resolve: the next replay sees it in guest memory and
/// takes the guest path, instead of decoding the buffer's older bytes.
#[test]
fn cpu_write_during_a_replay_takes_the_guest_path() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let bottom = DEST + W * 32 * 4;
    // Two halves of one surface, kept in one buffer.
    let mut s = Stream::default();
    paint(s.setup());
    resolve_to(&mut s, &mut mem, W, 32, DEST);
    resolve_to(&mut s, &mut mem, W, 32, bottom);
    run(&mut d, &s, &mut mem);
    // The top half again; the CPU writes the bottom half meanwhile.
    let mut s = Stream::default();
    paint_small(s.setup());
    resolve_to(&mut s, &mut mem, W, 32, DEST);
    paint(&mut s);
    mem.write_words(CMD, &s.0);
    let mut rec = xenon_gpu::Recorder::new();
    rec.record_buffer(&mem, CMD, s.0.len() as u32).unwrap();
    let mut racing = Racing {
        mem: std::cell::RefCell::new(mem),
        at: bottom as usize..(bottom + 256) as usize,
        armed: std::cell::Cell::new(true),
    };
    d.replay(&rec.commands, &mut racing).unwrap();
    assert!(!racing.armed.get());
    let mut mem = racing.mem.into_inner();
    let mut s = Stream::default();
    sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.timings.textures_kept, 0, "{r:?}");
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(img, guest_path(&mem, f));
    assert!(img.chunks(4).any(|p| p == [0x55; 4]));
}

/// A kept buffer over three 64x32 tiles, resolved whole by one replay; the
/// next replays (two per frame) resolve the first tile, then the first and
/// last: each reads back and hashes only the tiles it resolved, not the
/// middle one or the whole buffer, and writes back exactly those to guest
/// memory. The CPU wrote the middle tile, which no replay uses, so a
/// write-back of the buffer's older copy of it would show.
#[test]
fn write_back_reads_back_only_resolved_bytes() {
    let tile = W * 32 * 4;
    let tiles = [DEST, DEST + tile, DEST + 2 * tile];
    let bytes = |mem: &SliceMemory, t: u32| mem.data[t as usize..][..tile as usize].to_vec();
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    for t in tiles {
        resolve_to(&mut s, &mut mem, W, 32, t);
    }
    run(&mut d, &s, &mut mem);
    mem.data[tiles[1] as usize..][..tile as usize].fill(0x55);
    for (resolved, read) in [(&tiles[..1], tile), (&[tiles[0], tiles[2]][..], 2 * tile)] {
        let mut s = Stream::default();
        paint_small(s.setup());
        for &t in resolved {
            resolve_to(&mut s, &mut mem, W, 32, t);
        }
        let before = mem.clone();
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        assert_eq!(r.timings.readback_bytes, u64::from(read), "{r:?}");
        // Checked before the resolves, hashed again at write-back.
        assert_eq!(r.timings.kept_hashed_bytes, u64::from(2 * read), "{r:?}");
        let top = d.read_render_target(COLOR_KEY, 0, 0, W, 32).unwrap();
        assert_eq!(px(&top, W, 5, 5), [10, 20, 230, 255]);
        for t in tiles {
            if resolved.contains(&t) {
                assert_eq!(detiled(&mem, t, W, 32, W, 4), top, "tile 0x{t:X}");
            } else {
                assert_eq!(bytes(&mem, t), bytes(&before, t), "tile 0x{t:X}");
            }
        }
    }
}

/// A resolve into a kept buffer at a word aligned destination that is not
/// block aligned: the guest bytes under it are the target's, and the bytes
/// beside it in the same blocks keep what guest memory held.
#[test]
fn write_back_mid_block_keeps_the_bytes_beside_it() {
    let tile = W * 32 * 4;
    let at = DEST + 0x100;
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve_to(&mut s, &mut mem, W, 32, DEST);
    resolve_to(&mut s, &mut mem, W, 32, DEST + tile);
    run(&mut d, &s, &mut mem);
    let mut s = Stream::default();
    paint_small(s.setup());
    resolve_to(&mut s, &mut mem, W, 32, at);
    let before = mem.clone();
    let r = run(&mut d, &s, &mut mem);
    assert!(r.unsupported.is_empty(), "{r:?}");
    // Still the kept buffer: two blocks of the first tile and one of the
    // second come back, not the whole buffer.
    assert_eq!(r.timings.readback_bytes, u64::from(tile + 4096), "{r:?}");
    let top = d.read_render_target(COLOR_KEY, 0, 0, W, 32).unwrap();
    assert_eq!(detiled(&mem, at, W, 32, W, 4), top);
    let around = |m: &SliceMemory| {
        let (lo, hi) = (DEST as usize, (at + tile) as usize);
        (
            m.data[lo..lo + 0x100].to_vec(),
            m.data[hi..hi + 4096 - 0x100].to_vec(),
        )
    };
    assert_eq!(around(&mem), around(&before));
}

/// A resolve into a destination twice as wide as the rectangle, sampled
/// with that pitch: the kept texture is the target's pixels column for
/// column, as the guest path draws them.
#[test]
fn wider_destination_pitch_keeps_the_columns() {
    let mut wide = fetch(6, 2, IDENTITY);
    wide[0] = (wide[0] & !(0x1FF << 22)) | (((2 * W) >> 5) << 22);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    let (wf, hf) = (W as f32, H as f32);
    mem.write_words(RECT, &[f(0.0), f(0.0), f(wf), f(0.0), f(wf), f(hf)]);
    s.vertex_fetch(0, RECT, 6)
        .regs(RB_MODECONTROL, &[6])
        .regs(
            RB_COPY_CONTROL,
            &[
                COPY_CONVERT,
                DEST,
                (2 * W) | (H << 16),
                dest_info(6, 0, false),
            ],
        )
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_MODECONTROL, &[4]);
    sample(&mut s, wide);
    let r = run(&mut d, &s, &mut mem);
    assert!(r.unsupported.is_empty(), "{r:?}");
    assert_eq!(r.timings.textures_kept, 1, "{r:?}");
    let painted = d.read_render_target(COLOR_KEY, 0, 0, W, H).unwrap();
    let kept = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    drop(d);
    assert_eq!(kept, painted);
    assert_eq!(kept, guest_path(&mem, wide));
}

const RT_2_10_10_10_FLOAT: u32 = 3;
const RT_16_16_16_16_FLOAT: u32 = 7;
const RT_32_FLOAT: u32 = 14;
const FETCH_16_16_16_16_FLOAT: u32 = 32;
const FETCH_32_FLOAT: u32 = 36;

/// A float resolve and the texture fetched from it: source target, its
/// resolve destination format, the fetch format, and the target (and its
/// format) the fetch is sampled into, which holds the sampled floats
/// exactly.
struct FloatCase {
    rt: u32,
    dest_format: u32,
    fetch_format: u32,
    out_rt: u32,
    out: TargetFormat,
    paint: fn(&mut Stream) -> &mut Stream,
    /// Whether drawers may upload 32-bit float textures as `Rgba32Float`.
    float32: bool,
}

const HALF4: FloatCase = FloatCase {
    rt: RT_2_10_10_10_FLOAT,
    dest_format: FETCH_16_16_16_16_FLOAT,
    fetch_format: FETCH_16_16_16_16_FLOAT,
    out_rt: RT_16_16_16_16_FLOAT,
    out: TargetFormat::Rgba16Float,
    paint,
    float32: true,
};

/// [`HALF4`] from a half target holding NaNs, infinities and negatives.
const HALF4_SPECIALS: FloatCase = FloatCase {
    rt: RT_16_16_16_16_FLOAT,
    paint: paint_specials,
    ..HALF4
};

const FLOAT32: FloatCase = FloatCase {
    rt: RT_32_FLOAT,
    dest_format: FETCH_32_FLOAT,
    fetch_format: FETCH_32_FLOAT,
    out_rt: RT_32_FLOAT,
    out: TargetFormat::Float32,
    paint,
    float32: true,
};

/// [`FLOAT32`] on a host without filterable `Rgba32Float`: k_32_FLOAT is
/// kept and uploaded as `Rgba16Float`.
const FLOAT32_AS_HALF: FloatCase = FloatCase {
    float32: false,
    ..FLOAT32
};

impl FloatCase {
    fn drawer(&self) -> Option<TestDrawer> {
        let mut d = drawer()?;
        if !self.float32 {
            d.disable_float32_textures();
        }
        Some(d)
    }

    fn out_key(&self) -> RenderTargetKey {
        RenderTargetKey {
            format: self.out,
            ..OUT
        }
    }

    /// Paints the gradient into a target of `rt` and resolves it whole to
    /// DEST with `endian` and exponent bias 2.
    fn paint_and_resolve(&self, s: &mut Stream, mem: &mut SliceMemory, endian: u32) {
        let (wf, hf) = (W as f32, H as f32);
        mem.write_words(RECT, &[f(0.0), f(0.0), f(wf), f(0.0), f(wf), f(hf)]);
        s.regs(RB_COLOR_INFO, &[COLOR_BASE | (self.rt << 16)]);
        (self.paint)(s)
            .vertex_fetch(0, RECT, 6)
            .regs(RB_MODECONTROL, &[6])
            .regs(
                RB_COPY_CONTROL,
                &[
                    COPY_CONVERT,
                    DEST,
                    W | (H << 16),
                    (dest_info(self.dest_format, 2, false) & !7) | endian,
                ],
            )
            .draw_auto(PRIM_RECT_LIST, 3)
            .regs(RB_MODECONTROL, &[4]);
    }

    /// Samples `fetch` texel for texel into the output target.
    fn sample(&self, s: &mut Stream, fetch: [u32; 6]) {
        s.regs(RB_COLOR_INFO, &[OUT.base_tiles | (self.out_rt << 16)])
            .shader(false, &vs_uv())
            .shader(true, &ps_texture(1))
            .regs(SHADER_CONSTANT_FETCH + 6, &fetch)
            .vertex_fetch(0, SCREEN, 20)
            .draw_auto(PRIM_QUAD_LIST, 4)
            .regs(RB_COLOR_INFO, &[COLOR_BASE]);
    }

    /// What a drawer that never resolved samples from `mem` with `fetch`.
    fn guest_path(&self, mem: &SliceMemory, fetch: [u32; 6]) -> Vec<u8> {
        let mut d = self.drawer().expect("a drawer");
        let mut mem = mem.clone();
        let mut s = Stream::default();
        self.sample(s.setup(), fetch);
        let r = run(&mut d, &s, &mut mem);
        assert_eq!((r.draws, r.timings.textures_kept), (1, 0), "{r:?}");
        d.read_render_target_raw(self.out_key(), 0, 0, W, H)
            .unwrap()
    }
}

/// 7e3 and half NaNs, infinities and negatives resolved to
/// k_16_16_16_16_FLOAT and k_32_FLOAT resolved to k_32_FLOAT (uploaded as
/// `Rgba32Float` or `Rgba16Float`), fetched in the same replay at several
/// endians (8in64 for 64bpp) and swizzles: decoded on the GPU, nothing read
/// back or uploaded, and bit for bit the floats the guest path samples.
#[test]
fn float_resolves_are_kept_and_match_the_guest_path() {
    for (case, endian, swizzle, bytes) in [
        (&HALF4, 2, IDENTITY, 8),
        (&HALF4, 4, BGR1, 8),
        (&HALF4, 1, IDENTITY, 8),
        (&HALF4_SPECIALS, 2, IDENTITY, 8),
        (&HALF4_SPECIALS, 4, BGR1, 8),
        (&FLOAT32, 2, IDENTITY, 4),
        (&FLOAT32, 0, IDENTITY, 4),
        (&FLOAT32, 3, BGR1, 4),
        (&FLOAT32_AS_HALF, 2, IDENTITY, 4),
        (&FLOAT32_AS_HALF, 3, BGR1, 4),
    ] {
        // The fetch endian undoes the resolve's within each word; 8in64
        // also swaps the words, which a fetch cannot undo.
        let fetch_endian = if endian == 4 { 2 } else { endian };
        let f = fetch(case.fetch_format, fetch_endian, swizzle);
        let Some(mut d) = case.drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        case.paint_and_resolve(s.setup(), &mut mem, endian);
        case.sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        if case.rt == RT_16_16_16_16_FLOAT {
            // The guest bytes hold negative NaN halves (8in32 or 8in64
            // leaves each half big-endian), which the guest path decodes
            // as positive NaNs.
            let dest = &mem.data[DEST as usize..(DEST + W * H * 8) as usize];
            assert!(
                dest.chunks(2)
                    .any(|h| h[0] >= 0xFC && (h[0] & 3, h[1]) != (0, 0)),
                "no negative NaN"
            );
        }
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.kept_bytes),
            (1, u64::from(W * H * bytes)),
            "{r:?}"
        );
        assert_eq!((t.textures_uploaded, t.hashed_bytes), (0, 0), "{r:?}");
        let kept = d
            .read_render_target_raw(case.out_key(), 0, 0, W, H)
            .unwrap();
        drop(d);
        let n = kept.len() / (W * H) as usize;
        let distinct: std::collections::HashSet<_> = kept.chunks(n).collect();
        // Red, all a k_32_FLOAT target keeps, changes only along x.
        assert!(
            distinct.len() >= 64,
            "{} {}",
            case.fetch_format,
            distinct.len()
        );
        assert!(
            kept == case.guest_path(&mem, f),
            "format {} endian {endian}",
            case.fetch_format
        );
    }
}

/// A float resolve sampled by the next replay is still on the GPU.
#[test]
fn float_resolve_is_kept_for_the_next_replay() {
    for (case, bytes) in [(&HALF4, 8), (&FLOAT32, 4)] {
        let f = fetch(case.fetch_format, 2, IDENTITY);
        let Some(mut d) = drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        case.paint_and_resolve(s.setup(), &mut mem, 2);
        run(&mut d, &s, &mut mem);
        let mut s = Stream::default();
        case.sample(s.setup(), f);
        let r = run(&mut d, &s, &mut mem);
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.textures_uploaded, t.hashed_bytes),
            (1, 0, 0),
            "{r:?}"
        );
        assert_eq!(t.kept_hashed_bytes, u64::from(W * H * bytes), "{r:?}");
        let kept = d
            .read_render_target_raw(case.out_key(), 0, 0, W, H)
            .unwrap();
        drop(d);
        assert!(kept == case.guest_path(&mem, f), "{}", case.fetch_format);
    }
}

const RT_16_16_FLOAT: u32 = 6;
const FETCH_16_16_FLOAT: u32 = 31;
/// Z, W, constant 0, constant 1: k_16_16_FLOAT's Z and W repeat its Y.
const ZW01: u16 = 2 | (3 << 3) | (4 << 6) | (5 << 9);

/// A 7e3 target resolved to k_16_16_FLOAT and fetched as k_16_16_FLOAT.
const HALF2: FloatCase = FloatCase {
    dest_format: FETCH_16_16_FLOAT,
    fetch_format: FETCH_16_16_FLOAT,
    ..HALF4
};

/// [`HALF2`] from a half target holding NaNs, infinities and negatives.
const HALF2_SPECIALS: FloatCase = FloatCase {
    rt: RT_16_16_16_16_FLOAT,
    paint: paint_specials,
    ..HALF2
};

/// [`HALF2_SPECIALS`] from a k_16_16_FLOAT target.
const HALF2_FROM_RG16F: FloatCase = FloatCase {
    rt: RT_16_16_FLOAT,
    ..HALF2_SPECIALS
};

/// 7e3 and half NaNs, infinities, negatives and denormals resolved to
/// k_16_16_FLOAT and fetched as k_16_16_FLOAT in the same replay at every
/// 32bpp endian and several swizzles (Z and W included): decoded on the
/// GPU, nothing read back or uploaded, and bit for bit the halves the guest
/// path samples. The next replay still samples it from the GPU.
#[test]
fn half2_resolves_are_kept_and_match_the_guest_path() {
    for (case, endian, swizzle) in [
        (&HALF2, 2, IDENTITY),
        (&HALF2, 0, BGR1),
        (&HALF2, 1, ZW01),
        (&HALF2, 3, IDENTITY),
        (&HALF2_SPECIALS, 2, IDENTITY),
        (&HALF2_SPECIALS, 3, ZW01),
        (&HALF2_FROM_RG16F, 2, BGR1),
        (&HALF2_FROM_RG16F, 1, IDENTITY),
    ] {
        let f = fetch(case.fetch_format, endian, swizzle);
        let Some(mut d) = case.drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        case.paint_and_resolve(s.setup(), &mut mem, endian);
        case.sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        if case.rt != HALF2.rt && endian == 2 {
            // 8in32 leaves each half big-endian: negative NaNs, which the
            // guest path decodes as positive NaNs.
            let dest = &mem.data[DEST as usize..(DEST + W * H * 4) as usize];
            assert!(
                dest.chunks(2)
                    .any(|h| h[0] >= 0xFC && (h[0] & 3, h[1]) != (0, 0)),
                "no negative NaN"
            );
        }
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.kept_bytes),
            (1, u64::from(W * H * 4)),
            "{r:?}"
        );
        assert_eq!((t.textures_uploaded, t.hashed_bytes), (0, 0), "{r:?}");
        let kept = d
            .read_render_target_raw(case.out_key(), 0, 0, W, H)
            .unwrap();
        let distinct: std::collections::HashSet<_> = kept.chunks(8).collect();
        assert!(distinct.len() >= 64, "{}", distinct.len());
        // The next replay samples the same texture, still kept.
        let mut s = Stream::default();
        case.sample(s.setup(), f);
        let r = run(&mut d, &s, &mut mem);
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.textures_uploaded, t.hashed_bytes),
            (1, 0, 0),
            "{r:?}"
        );
        let next = d
            .read_render_target_raw(case.out_key(), 0, 0, W, H)
            .unwrap();
        drop(d);
        let guest = case.guest_path(&mem, f);
        assert!(kept == guest, "rt {} endian {endian}", case.rt);
        assert!(next == guest, "next replay: rt {} endian {endian}", case.rt);
    }
}

const FETCH_24_8: u32 = 22;
const FETCH_24_8_FLOAT: u32 = 23;
/// The depth gradient quad of [`depth_resolve`], after the other vertex data.
const DEPTH_QUAD: u32 = VERTS + 0x500;

/// The depth formats sampled from a depth resolve into a k_32_FLOAT target
/// (the depth is RRRR), with `Rgba32Float` or as `Rgba16Float`.
fn depth_case(fetch_format: u32, float32: bool) -> FloatCase {
    FloatCase {
        fetch_format,
        float32,
        ..FLOAT32
    }
}

/// Draws a depth gradient (z 0 to `top`) into the depth target (kD24FS8
/// when `float`), colour writes off, and resolves the depth whole to DEST
/// as raw 32-bit words with `endian`.
fn depth_resolve(s: &mut Stream, mem: &mut SliceMemory, float: bool, endian: u32, top: f32) {
    let (w, h) = (W as f32, H as f32);
    let c = [0; 4];
    mem.write_words(
        DEPTH_QUAD,
        &color_verts(&[
            (0.0, 0.0, 0.0, c),
            (w, 0.0, 0.7 * top, c),
            (w, h, top, c),
            (0.0, h, 0.001 * top, c),
        ]),
    );
    let depth_info = DEPTH_BASE | (u32::from(float) << 16);
    s.regs(RB_DEPTH_INFO, &[depth_info])
        .regs(RB_COLOR_MASK, &[0])
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        // z ALWAYS with z write.
        .regs(RB_DEPTHCONTROL, &[2 | 4 | (7 << 4)])
        .vertex_fetch(0, DEPTH_QUAD, 16)
        .draw_auto(PRIM_QUAD_LIST, 4)
        .regs(RB_DEPTHCONTROL, &[0])
        .regs(RB_COLOR_MASK, &[0xF]);
    s.resolve_with(
        mem,
        (W, H),
        4 | COPY_RAW,
        (DEST, W | (H << 16), (dest_info(6, 0, false) & !7) | endian),
        [0; 3],
    );
}

/// kD24S8 and kD24FS8 depth resolved (kept on the GPU) and fetched as
/// k_24_8 and k_24_8_FLOAT in the same replay, at several endians and
/// swizzles, with depths up to 1 and up to 1e-4 (20e4 denormals below
/// 2^-15): decoded on the GPU from the kept words, nothing hashed or
/// uploaded (only the end of the replay writes the words back), and bit
/// for bit the floats the guest path samples (`Rgba32Float`, or
/// `Rgba16Float` without filterable 32-bit floats).
#[test]
fn depth_resolves_are_kept_and_match_the_guest_path() {
    for (fetch_format, float, float32, endian, swizzle, top) in [
        (FETCH_24_8, false, true, 2, IDENTITY, 1.0),
        (FETCH_24_8, false, true, 0, BGR1, 1.0),
        (FETCH_24_8, false, false, 2, IDENTITY, 1.0),
        (FETCH_24_8, false, true, 2, IDENTITY, 1e-4),
        (FETCH_24_8_FLOAT, true, true, 2, IDENTITY, 1.0),
        (FETCH_24_8_FLOAT, true, true, 1, BGR1, 1.0),
        (FETCH_24_8_FLOAT, true, false, 3, IDENTITY, 1.0),
        (FETCH_24_8_FLOAT, true, true, 2, IDENTITY, 1e-4),
        (FETCH_24_8_FLOAT, true, false, 2, IDENTITY, 1e-4),
        // The other format's reading of the same words.
        (FETCH_24_8_FLOAT, false, true, 2, IDENTITY, 1.0),
        (FETCH_24_8, true, true, 2, IDENTITY, 1.0),
    ] {
        let case = depth_case(fetch_format, float32);
        let f = fetch(fetch_format, endian, swizzle);
        let Some(mut d) = case.drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        depth_resolve(s.setup(), &mut mem, float, endian, top);
        case.sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        if float && top < 1.0 {
            // Little-endian words: 20e4 exponent 0 in the top byte's high bits.
            let words = detiled(&mem, DEST, W, H, W, 4);
            assert!(
                words.chunks(4).any(|b| b[3] < 0x10 && b[1..] != [0; 3]),
                "no 20e4 denormal"
            );
        }
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.kept_bytes),
            (1, u64::from(W * H * 4)),
            "{r:?}"
        );
        assert_eq!((t.textures_uploaded, t.hashed_bytes), (0, 0), "{r:?}");
        let kept = d
            .read_render_target_raw(case.out_key(), 0, 0, W, H)
            .unwrap();
        drop(d);
        let distinct: std::collections::HashSet<_> = kept.chunks(4).collect();
        assert!(distinct.len() >= 64, "{fetch_format} {}", distinct.len());
        assert!(
            kept == case.guest_path(&mem, f),
            "format {fetch_format} float {float} float32 {float32} endian {endian} top {top}"
        );
    }
}

/// A depth resolve sampled by the next replay is still on the GPU.
#[test]
fn depth_resolve_is_kept_for_the_next_replay() {
    let case = depth_case(FETCH_24_8_FLOAT, true);
    let f = fetch(FETCH_24_8_FLOAT, 2, IDENTITY);
    let Some(mut d) = case.drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    depth_resolve(s.setup(), &mut mem, true, 2, 1.0);
    run(&mut d, &s, &mut mem);
    let mut s = Stream::default();
    case.sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    let t = &r.timings;
    assert_eq!(
        (t.textures_kept, t.textures_uploaded, t.hashed_bytes),
        (1, 0, 0),
        "{r:?}"
    );
    assert_eq!(t.kept_hashed_bytes, u64::from(W * H * 4), "{r:?}");
    let kept = d
        .read_render_target_raw(case.out_key(), 0, 0, W, H)
        .unwrap();
    drop(d);
    assert!(kept == case.guest_path(&mem, f));
}

const FETCH_16_16_16_16_EXPAND: u32 = 29;
/// Mip level 1 of a 64x64 8-byte texture at DEST: right after the base.
const MIP: u32 = DEST + W * H * 8;

/// Fetch constant of the 64x64 k_16_16_16_16_EXPAND chain at DEST (signed,
/// integer number format, as games fetch 16_16_16_16_FLOAT resolves) with
/// one mip at MIP, sampling levels `min`..=1, all filters linear.
fn mip_fetch(min: u32) -> [u32; 6] {
    let mut f = fetch(FETCH_16_16_16_16_EXPAND, 2, IDENTITY);
    f[0] |= 0x55 << 2;
    f[3] |= 1 | (1 << 19) | (1 << 21) | (1 << 23);
    f[4] = (min << 2) | (1 << 6);
    f[5] |= MIP;
    f
}

/// Draws `paint` and resolves its top-left 32x32 to MIP as
/// k_16_16_16_16_FLOAT.
fn paint_mip(s: &mut Stream, mem: &mut SliceMemory, paint: fn(&mut Stream) -> &mut Stream) {
    let at = RECT + 32;
    mem.write_words(at, &[f(0.0), f(0.0), f(32.0), f(0.0), f(32.0), f(32.0)]);
    paint(s)
        .vertex_fetch(0, at, 6)
        .regs(RB_MODECONTROL, &[6])
        .regs(
            RB_COPY_CONTROL,
            &[
                COPY_CONVERT,
                MIP,
                32 | (32 << 16),
                (dest_info(FETCH_16_16_16_16_FLOAT, 0, false) & !7) | 2,
            ],
        )
        .draw_auto(PRIM_RECT_LIST, 3)
        .regs(RB_MODECONTROL, &[4]);
}

/// A 7e3 gradient resolved to k_16_16_16_16_FLOAT as the base level and a
/// blue rectangle resolved as its 32x32 mip right after it, the way games
/// build blurred chains. The mip lives in the same kept resolve and is
/// decoded with the base: a fetch that starts at level 1 samples only
/// blue, and one that starts at level 0 (texel for texel, so LOD 0) still
/// samples the base exactly as the guest path does. The
/// packed mips bit changes nothing: a 64x64 chain packs from level 2.
#[test]
fn kept_mip_level_is_sampled() {
    let case = &HALF4;
    let mut outputs = Vec::new();
    for (min, packed) in [(0, 0), (1, 0), (1, 1 << 11)] {
        let Some(mut d) = case.drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        case.paint_and_resolve(s.setup(), &mut mem, 2);
        paint_mip(&mut s, &mut mem, paint_small);
        let mut f = mip_fetch(min);
        f[5] |= packed;
        case.sample(&mut s, f);
        let r = run(&mut d, &s, &mut mem);
        assert!(r.unsupported.is_empty(), "{r:?}");
        let t = &r.timings;
        assert_eq!(
            (t.textures_kept, t.kept_bytes, t.textures_uploaded),
            (1, u64::from(W * H * 8 + 32 * 32 * 8), 0),
            "{r:?}"
        );
        outputs.push(
            d.read_render_target_raw(case.out_key(), 0, 0, W, H)
                .unwrap(),
        );
    }
    let half = |t: &[u8], c: usize| u16::from_le_bytes([t[2 * c], t[2 * c + 1]]);
    assert!(outputs[2] == outputs[1], "packed bit");
    let mip: std::collections::HashSet<_> = outputs[1].chunks(8).collect();
    assert_eq!(mip.len(), 1, "level 1 is one blue");
    let blue = mip.into_iter().next().unwrap();
    assert!(
        half(blue, 2) > half(blue, 0) && half(blue, 0) > 0,
        "{blue:?}"
    );
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    case.paint_and_resolve(s.setup(), &mut mem, 2);
    let Some(mut d) = case.drawer() else { return };
    run(&mut d, &s, &mut mem);
    drop(d);
    assert!(outputs[0] == case.guest_path(&mem, mip_fetch(0)));
}

/// A resolve into the mip of a kept chain after it was sampled decodes the
/// chain again: the second fetch of level 1 sees the new gradient.
#[test]
fn resolve_into_a_kept_mip_decodes_it_again() {
    let case = &HALF4;
    let Some(mut d) = case.drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    case.paint_and_resolve(s.setup(), &mut mem, 2);
    paint_mip(&mut s, &mut mem, paint_small);
    case.sample(&mut s, mip_fetch(1));
    s.regs(RB_COLOR_INFO, &[COLOR_BASE | (case.rt << 16)]);
    paint_mip(&mut s, &mut mem, paint);
    case.sample(&mut s, mip_fetch(1));
    let r = run(&mut d, &s, &mut mem);
    assert!(r.unsupported.is_empty(), "{r:?}");
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (2, 0),
        "{r:?}"
    );
    let out = d
        .read_render_target_raw(case.out_key(), 0, 0, W, H)
        .unwrap();
    let distinct: std::collections::HashSet<_> = out.chunks(8).collect();
    assert!(distinct.len() > 16, "{}", distinct.len());
}

/// Late write-back: guest memory gets the resolve when the next replay
/// starts, not at the end of its own, and then matches the exact mode.
#[test]
fn late_write_back_lands_at_the_next_replay() {
    let exact = {
        let Some(mut d) = drawer() else { return };
        let mut mem = memory();
        vertices(&mut mem);
        let mut s = Stream::default();
        paint(s.setup());
        resolve(&mut s, &mut mem, W, H);
        run(&mut d, &s, &mut mem);
        mem.data[DEST as usize..][..(W * H * 4) as usize].to_vec()
    };
    let Some(mut d) = drawer() else { return };
    d.late_writeback = true;
    let mut mem = memory();
    vertices(&mut mem);
    let before = mem.data[DEST as usize..][..(W * H * 4) as usize].to_vec();
    assert_ne!(before, exact);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    let r = run(&mut d, &s, &mut mem);
    assert_eq!(r.timings.writebacks, 1, "{r:?}");
    assert_eq!(mem.data[DEST as usize..][..before.len()], before[..]);
    run(&mut d, Stream::default().setup(), &mut mem);
    assert_eq!(mem.data[DEST as usize..][..exact.len()], exact[..]);
}

/// Late write-back after a CPU write between the replays: the CPU's bytes
/// stay and the kept resolve is dropped.
#[test]
fn late_write_back_keeps_a_cpu_write() {
    let Some(mut d) = drawer() else { return };
    d.late_writeback = true;
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    run(&mut d, &s, &mut mem);
    mem.data[DEST as usize..][..256].fill(0x55);
    let f = fetch(6, 2, IDENTITY);
    let mut s = Stream::default();
    sample(s.setup(), f);
    let r = run(&mut d, &s, &mut mem);
    assert!(mem.data[DEST as usize..][..256].iter().all(|&b| b == 0x55));
    assert_eq!(
        (r.timings.textures_kept, r.timings.textures_uploaded),
        (0, 1),
        "{r:?}"
    );
}

/// Guest memory that lists the writes the test says were made.
struct Watched {
    mem: SliceMemory,
    written: Option<Vec<(u64, u64)>>,
}

impl xenon_gpu::GuestRead for Watched {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        self.mem.read_u32(addr)
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        self.mem.read_bytes(addr, len)
    }
}

impl xenon_draw::GuestMemory for Watched {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        self.mem.write_bytes(addr, bytes)
    }
    fn take_written(&mut self) -> Option<Vec<(u64, u64)>> {
        self.written.replace(Vec::new())
    }
}

/// With guest writes listed, a kept resolve checked against guest memory
/// is not hashed again until a listed write covers it: the write-back's
/// own write, then nothing, then a CPU write, which takes the guest path.
#[test]
fn listed_writes_skip_hashing_unwritten_kept_resolves() {
    let f = fetch(6, 2, IDENTITY);
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    vertices(&mut mem);
    let mut s = Stream::default();
    paint(s.setup());
    resolve(&mut s, &mut mem, W, H);
    let mut sampler = Stream::default();
    sample(sampler.setup(), f);
    // Another fetch constant over the same bytes: not in the texture memo.
    let mut other = Stream::default();
    sample(other.setup(), fetch(6, 2, BGR1));
    let commands = |mem: &mut SliceMemory, s: &Stream| {
        mem.write_words(CMD, &s.0);
        let mut rec = xenon_gpu::Recorder::new();
        rec.record_buffer(&*mem, CMD, s.0.len() as u32).unwrap();
        rec.commands
    };
    let paint_cmds = commands(&mut mem, &s);
    let sample_cmds = commands(&mut mem, &sampler);
    let other_cmds = commands(&mut mem, &other);
    let mut w = Watched {
        mem,
        written: Some(vec![]),
    };
    let replay = |d: &mut xenon_draw::Drawer, w: &mut Watched, cmds| {
        let r = d
            .replay(cmds, w as &mut dyn xenon_draw::GuestMemory)
            .unwrap();
        let t = &r.timings;
        (t.textures_kept, t.kept_hashed_bytes)
    };
    replay(&mut d, &mut w, &paint_cmds);
    // The write-back wrote the destination: hashed again.
    let dest = u64::from(DEST);
    let all = u64::from(W * H * 4);
    w.written = Some(vec![(dest, dest + all)]);
    assert_eq!(replay(&mut d, &mut w, &sample_cmds), (1, all));
    // Nothing written since: kept, not hashed.
    assert_eq!(replay(&mut d, &mut w, &other_cmds), (1, 0));
    // A CPU write: hashed, does not match, guest path.
    w.mem.data[DEST as usize..][..256].fill(0x55);
    w.written = Some(vec![(dest, dest + 256)]);
    assert_eq!(replay(&mut d, &mut w, &sample_cmds).0, 0);
    let img = d.read_render_target(OUT, 0, 0, W, H).unwrap();
    assert!(img.chunks(4).any(|p| p == [0x55; 4]));
}
