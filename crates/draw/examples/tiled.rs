//! Times a synthetic frame shaped like a predicated-tiling world frame:
//!
//! - a 256x256 7e3 pass of 330 textured quads resolved to four mip sizes
//!   (16_16_16_16_FLOAT, the CPU resolve path), then 94 depth-only draws
//!   into a 128x128 depth target and its raw depth resolve;
//! - three 1280x256 tiles, each starting with a depth restore (a k_8_8_8_8
//!   rectangle at the depth buffer's EDRAM base and pitch, which the drawer
//!   transfers into the D24FS8 target; a frame that restores
//!   at another pitch and MSAA mode is not transferred today, so this
//!   overstates the transfer cost), 117 + 30 quads into three k_8_8_8_8
//!   targets with depth, 24 point draws, then five resolves: raw depth,
//!   colour 0 twice (the second clearing colour and depth), colours 1 and 2
//!   with clears;
//! - a 1280x720 7e3 pass of 70 quads with depth test, a raw 1280x720 depth
//!   resolve, 7e3 resolves to 16_16_16_16_FLOAT at 1280x720 and four
//!   smaller sizes, a last 7e3 -> 8_8_8_8 resolve and a swap.
//!
//! Every textured draw samples 11 slots: three from a pool of 48 256x256
//! textures, four 512x512 ones and four 1280x720 resolve destinations, so
//! textures rewritten each frame are decoded each frame (one pool texture
//! changes every frame, so the resolved pixels do too). Nothing here comes
//! from a game.
//!
//! cargo run --release -p xenon-draw --example tiled -- [frames] [skip]
//!
//! `skip` sets `Drawer::skip_resolve_writeback`.

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use std::time::{Duration, Instant};
use xenon_draw::Timings;
use xenon_gpu::regs::*;
use xenon_gpu::{Recorder, SliceMemory};
use xenon_shader::asm::*;
use xenon_shader::ucode::*;
use xenon_texture::{endian, tile};

const P: u32 = 1280;
const TILE_H: u32 = 256;
const FH: u32 = 720;
const TILES: u32 = 3;
const POOL: u32 = 0x100_0000;
const POOL_N: u32 = 48;
const SHARED: u32 = 0x1C0_0000;
/// Resolve destinations, 1280 pixels wide, one region per resolve kind.
const DEST_RT0: u32 = 0x200_0000;
const DEST_RT0_B: u32 = 0x240_0000;
const DEST_RT1: u32 = 0x280_0000;
const DEST_RT2: u32 = 0x2C0_0000;
const DEST_TILE_Z: u32 = 0x300_0000;
const DEST_FULL_Z: u32 = 0x340_0000;
const DEST_POST16: u32 = 0x380_0000;
const DEST_SMALL: u32 = 0x400_0000;
const FRONT: u32 = 0x440_0000;
const QUADS: u32 = 0x480_0000;
const STREAM: u32 = 0x4A0_0000;
const MEM: usize = 0x500_0000;

const PRIM_POINTS: u32 = 1;
const PRIM_RECTS: u32 = 8;
const PRIM_QUADS: u32 = 13;
/// RB_COLOR_INFO format fields.
const RT_8888: u32 = 0;
const RT_7E3: u32 = 12;
/// RB_DEPTH_INFO: kD24FS8 at tile 0.
const DEPTH_FS8: u32 = 1 << 16;
/// RB_DEPTHCONTROL: z test GEQUAL (reversed depth, cleared to 0) with write.
const Z_WRITE: u32 = 2 | 4 | (6 << 4);
const Z_TEST: u32 = 2 | (6 << 4);
const DEST_16F4: u32 = 32;

/// Texture fetch constant of a tiled k_8_8_8_8 `w` x `h` texture at `addr`.
fn fetch(addr: u32, w: u32, h: u32) -> [u32; 6] {
    [
        2 | (2 << 10) | (2 << 13) | ((w.next_multiple_of(32) >> 5) << 22) | (1 << 31),
        6 | (2 << 6) | addr,
        (w - 1) | ((h - 1) << 13),
        ((1 << 3) | (2 << 6) | (3 << 9)) << 1,
        0,
        1 << 9,
    ]
}

/// Pixel shader: fetches slots 0..11 at interpolator 0 and writes the first
/// three to oC0..oC2.
fn ps_tile() -> Vec<u32> {
    let mut instrs: Vec<Instr> = (0..11)
        .map(|slot| {
            Instr::Fetch(FetchInstr::Texture(TextureFetch::fetch_2d(
                1 + slot,
                0,
                slot,
            )))
        })
        .collect();
    for rt in 0..3 {
        instrs.push(export(rt, AluSrc::temp(1 + rt, SWZ_XYZW)));
    }
    assemble(
        &[exec(0, 6, false), exec(6, 6, false), exec(12, 2, true)],
        &instrs,
    )
}

fn quad(x: f32, y: f32, w: f32, h: f32) -> Vec<u32> {
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        .into_iter()
        .flat_map(|(u, v)| [f(x + u * w), f(y + v * h), f(0.5), f(u), f(v)])
        .collect()
}

struct Builder {
    s: Stream,
    quads: Vec<u32>,
    draws: u32,
}

impl Builder {
    /// A quad draw with vs_uv; slots 0..3 get pool textures.
    fn quad(&mut self, x: f32, y: f32, w: f32, h: f32) {
        let i = self.draws;
        let at = QUADS + 4 * self.quads.len() as u32;
        self.quads.extend(quad(x, y, w, h));
        let mut fetches = Vec::new();
        for slot in 0..3 {
            let k = (i * 3 + slot) % POOL_N;
            fetches.extend(fetch(POOL + k * 0x4_0000, 256, 256));
        }
        self.s
            .regs(SHADER_CONSTANT_FETCH, &fetches)
            .vertex_fetch(95, at, 20)
            .draw_auto(PRIM_QUADS, 4);
        self.draws += 1;
    }

    /// Resolve of the (0,0)-(w,h) rectangle, as Stream::resolve_with but
    /// with its own rectangle vertices, since the stream is replayed whole.
    fn resolve(
        &mut self,
        (w, h): (u32, u32),
        control: u32,
        dest: (u32, u32, u32),
        clear: [u32; 3],
    ) {
        let at = QUADS + 4 * self.quads.len() as u32;
        let (wf, hf) = (w as f32, h as f32);
        self.quads
            .extend([f(0.0), f(0.0), f(wf), f(0.0), f(wf), f(hf)]);
        self.s
            .vertex_fetch(0, at, 6)
            .regs(RB_MODECONTROL, &[6])
            .regs(RB_COPY_CONTROL, &[control, dest.0, dest.1, dest.2])
            .regs(RB_DEPTH_CLEAR, &clear)
            .draw_auto(PRIM_RECTS, 3)
            .regs(RB_MODECONTROL, &[4]);
    }

    /// `n` single point draws with the slots as they are.
    fn points(&mut self, n: u32) {
        let at = QUADS + 4 * self.quads.len() as u32;
        self.quads.extend([f(4.0), f(4.0), f(0.5), f(0.0), f(0.0)]);
        for _ in 0..n {
            self.s.vertex_fetch(95, at, 5).draw_auto(PRIM_POINTS, 1);
            self.draws += 1;
        }
    }
}

/// vs_uv reading vertex fetch constant 95 (slot 0's dwords hold textures).
fn vs_uv95() -> Vec<u32> {
    let mut pos = VertexFetch::full(1, 0, 95, VertexFormat::F32_32_32_FLOAT, 5, 0);
    pos.dest_swizzle = fetch_dest_swz("xyz1");
    let mut uv = VertexFetch::full(2, 0, 95, VertexFormat::F32_32_FLOAT, 5, 3);
    uv.mini = true;
    uv.dest_swizzle = fetch_dest_swz("xy01");
    assemble(
        &[exec(0, 4, true)],
        &[
            Instr::Fetch(FetchInstr::Vertex(pos)),
            Instr::Fetch(FetchInstr::Vertex(uv)),
            export(62, AluSrc::temp(1, SWZ_XYZW)),
            export(0, AluSrc::temp(2, SWZ_XYZW)),
        ],
    )
}

fn build(mem: &mut SliceMemory) -> (Stream, u32) {
    // Static textures: a gradient per pool slot.
    for k in 0..POOL_N + 4 {
        let size = if k < POOL_N { 256 } else { 512 };
        let linear: Vec<u8> = (0..size * size)
            .flat_map(|i| [(i % size) as u8, (i / size) as u8, k as u8, 255])
            .collect();
        let mut tiled = tile::tile_2d(&linear, size, size, size, 4).unwrap();
        endian::swap_in_place(&mut tiled, xenon_texture::Endian::K8in32);
        let at = if k < POOL_N {
            POOL + k * 0x4_0000
        } else {
            SHARED + (k - POOL_N) * 0x10_0000
        };
        mem.data[at as usize..][..tiled.len()].copy_from_slice(&tiled);
    }
    let mut b = Builder {
        s: Stream::default(),
        quads: Vec::new(),
        draws: 0,
    };
    let mut frame_slots = Vec::new();
    for k in 0..4 {
        frame_slots.extend(fetch(SHARED + k * 0x10_0000, 512, 512));
    }
    for dest in [DEST_TILE_Z, DEST_FULL_Z, DEST_RT0, DEST_RT1] {
        frame_slots.extend(fetch(dest, P, FH));
    }
    b.s.setup()
        .regs(SHADER_CONSTANT_FETCH + 3 * 6, &frame_slots)
        .regs(RB_BLENDCONTROL1, &[0x0001_0001; 3])
        .shader(false, &vs_uv95())
        .shader(true, &ps_tile());

    // Pre-pass: 256x256 7e3 with depth, four mip resolves.
    b.s.regs(
        RB_SURFACE_INFO,
        &[256 | (2 << 16), 224 | (RT_7E3 << 16), DEPTH_FS8],
    )
    .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, 256 | (256 << 16)])
    .regs(RB_COLOR_MASK, &[0xF])
    .regs(RB_DEPTHCONTROL, &[Z_WRITE]);
    for i in 0..330 {
        b.quad(
            (i * 32 % 256) as f32,
            (i * 32 / 256 * 32 % 256) as f32,
            64.0,
            64.0,
        );
    }
    let mut small = DEST_SMALL;
    for size in [256, 128, 64, 32] {
        b.resolve(
            (size, size),
            COPY_CONVERT,
            (small, size | (size << 16), dest_info(DEST_16F4, 0, false)),
            [0; 3],
        );
        small += size * size * 8;
    }
    // Depth-only 128x128 pass and its raw depth resolve.
    b.s.regs(RB_SURFACE_INFO, &[160, 0, DEPTH_FS8])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, 128 | (128 << 16)])
        .regs(RB_MODECONTROL, &[5])
        .regs(RB_DEPTHCONTROL, &[Z_WRITE]);
    for i in 0..94 {
        b.quad(
            (i * 16 % 128) as f32,
            (i * 16 / 128 * 16 % 128) as f32,
            32.0,
            32.0,
        );
    }
    b.s.regs(RB_MODECONTROL, &[4]);
    b.resolve(
        (128, 128),
        4 | COPY_RAW,
        (small, 128 | (128 << 16), dest_info(6, 0, false)),
        [0; 3],
    );
    small += 128 * 128 * 4;

    // Tiles.
    let tile_bytes = P * TILE_H * 4;
    let restore = QUADS + 4 * b.quads.len() as u32;
    b.quads.extend(
        [(0.0, 0.0), (P as f32, 0.0), (0.0, TILE_H as f32)]
            .into_iter()
            .flat_map(|(x, y)| [f(x), f(y), f(0.5), f(x / P as f32), f(y / FH as f32)]),
    );
    b.s.regs(RB_SURFACE_INFO, &[P | (1 << 16), 0, DEPTH_FS8])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, P | (TILE_H << 16)]);
    for t in 0..TILES {
        // Depth restore: the tile depth texture drawn as colour at the
        // depth base and pitch.
        b.s.regs(RB_COLOR_INFO, &[RT_8888 << 16])
            .regs(RB_COLOR_MASK, &[0xF])
            .regs(RB_DEPTHCONTROL, &[0])
            .shader(true, &ps_texture(7))
            .vertex_fetch(95, restore, 15)
            .draw_auto(PRIM_RECTS, 3);
        b.draws += 1;
        b.s.regs(RB_COLOR_INFO, &[512 | (RT_8888 << 16)])
            .regs(RB_COLOR_INFO_RT[1], &[1024 | (RT_8888 << 16)])
            .regs(RB_COLOR_INFO_RT[2], &[1536 | (RT_8888 << 16)])
            .regs(RB_COLOR_MASK, &[0xFFF])
            .regs(RB_DEPTHCONTROL, &[Z_WRITE])
            .shader(true, &ps_tile());
        for i in 0..147 {
            b.quad(
                (i * 80 % P) as f32,
                (i * 80 / P * 64 % TILE_H) as f32,
                160.0,
                128.0,
            );
        }
        b.points(24);
        let dest = |base: u32| {
            (
                base + t * tile_bytes,
                P | (FH << 16),
                dest_info(6, 0, false),
            )
        };
        for (control, base) in [
            (4 | COPY_RAW, DEST_TILE_Z),
            (COPY_CONVERT, DEST_RT0),
            (COPY_CONVERT | CLEAR_COLOR | CLEAR_DEPTH, DEST_RT0_B),
            (COPY_CONVERT | CLEAR_COLOR | 1, DEST_RT1),
            (COPY_CONVERT | CLEAR_COLOR | 2, DEST_RT2),
        ] {
            b.resolve((P, TILE_H), control, dest(base), [0; 3]);
        }
    }

    // Post: 1280x720 7e3 with a depth test, its resolves and the swap.
    b.s.regs(RB_SURFACE_INFO, &[P, 720 | (RT_7E3 << 16), DEPTH_FS8])
        .regs(PA_SC_WINDOW_SCISSOR_TL, &[1 << 31, P | (FH << 16)])
        .regs(RB_COLOR_MASK, &[0xF])
        .regs(RB_DEPTHCONTROL, &[Z_TEST]);
    for i in 0..70 {
        b.quad((i * 64 % 640) as f32, (i * 36 % 360) as f32, 640.0, 360.0);
    }
    b.points(144);
    b.resolve(
        (P, FH),
        4 | COPY_RAW,
        (DEST_FULL_Z, P | (FH << 16), dest_info(6, 0, false)),
        [0; 3],
    );
    b.resolve(
        (P, FH),
        COPY_CONVERT,
        (DEST_POST16, P | (FH << 16), dest_info(DEST_16F4, 0, false)),
        [0; 3],
    );
    for (w, h) in [(640, 360), (320, 180), (320, 180), (320, 180), (160, 90)] {
        b.resolve(
            (w, h),
            COPY_CONVERT,
            (small, w | (h << 16), dest_info(DEST_16F4, 0, false)),
            [0; 3],
        );
        small += (w * h * 8).next_multiple_of(0x1000);
    }
    assert!(small < FRONT);
    b.resolve(
        (P, FH),
        COPY_CONVERT,
        (FRONT, P | (FH << 16), dest_info(6, 0, false)),
        [0; 3],
    );
    b.s.swap_at(FRONT, P, FH);
    mem.write_words(QUADS, &b.quads);
    assert!(QUADS as usize + 4 * b.quads.len() < STREAM as usize);
    (b.s, b.draws)
}

fn main() {
    let frames: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4)
        .max(2);
    let mut d = match xenon_draw::Drawer::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    d.skip_resolve_writeback = std::env::args().nth(2).as_deref() == Some("skip");
    println!("adapter: {}", d.adapter);
    let mut mem = SliceMemory {
        base: 0,
        data: vec![0; MEM],
    };
    let (s, draws) = build(&mut mem);
    assert!(STREAM as usize + 4 * s.0.len() < MEM);
    mem.write_words(STREAM, &s.0);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, STREAM, s.0.len() as u32).unwrap();
    assert!(rec.warnings.is_empty(), "{:?}", rec.warnings);
    let cmds = rec.commands;

    let mut total = Timings::default();
    let mut wall = Duration::ZERO;
    for i in 0..frames {
        // A new frame's content: one pool texture changes, so every resolve
        // destination differs from the last frame's, as in a moving scene.
        mem.data[POOL as usize..][..0x4_0000].fill(i as u8 * 40);
        let t = Instant::now();
        let r = d.replay(&cmds, &mut mem).unwrap();
        let dt = t.elapsed();
        assert_eq!(r.frames.len(), 1, "{r:?}");
        if i == 0 {
            println!(
                "frame 0 (cold caches): {dt:?}, {} draws, {} resolves",
                r.draws, r.resolves
            );
            println!(
                "unsupported: {:?}\nwarnings: {:?}",
                r.unsupported, r.warnings
            );
            assert_eq!(r.draws, draws, "{r:?}");
            continue;
        }
        wall += dt;
        total += &r.timings;
    }
    let n = frames - 1;
    println!("warm frames: {n}, mean {:?} per frame", wall / n);
    let t = &total;
    for (name, v) in [
        ("shaders+pipelines", t.shaders),
        ("guest data upload", t.guest_data),
        ("texture decode+upload", t.textures),
        ("encode+submit", t.submit),
        ("GPU wait (rendering)", t.gpu_wait),
        ("resolve readback", t.readback),
        ("resolve CPU convert+tile", t.resolve_cpu),
        ("clears", t.clears),
        ("swap read+detile", t.swap),
        ("(EDRAM transfers)", t.transfer),
    ] {
        println!("  {name:26} {:>10.2?} per frame", v / n);
    }
    let mb = |b: u64| b as f64 / n as f64 / 1e6;
    println!(
        "  per frame: textures uploaded {} ({:.1} MB), hashed {:.1} MB, guest data {:.1} MB, readback {:.1} MB",
        t.textures_uploaded / n,
        mb(t.texture_bytes),
        mb(t.hashed_bytes),
        mb(t.guest_bytes),
        mb(t.readback_bytes)
    );
    println!(
        "  per frame: textures from kept resolves {} ({:.1} MB not read), kept checks hashed {:.1} MB",
        t.textures_kept / n,
        mb(t.kept_bytes),
        mb(t.kept_hashed_bytes)
    );
    println!(
        "  per frame: pipelines created {}, submits {}, transfers {} ({} passes)",
        t.pipelines_created / n,
        t.submits / n,
        t.transfers / n,
        t.transfer_passes / n
    );
}
