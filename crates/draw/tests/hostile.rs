//! Hostile synthetic command lists: random register state, shader words and
//! packets must never panic the drawer (errors and skips are fine).

mod common;

use common::*;
use xenon_draw::GuestMemory;
use xenon_gpu::regs::*;
use xenon_gpu::{Command, Recorder};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// Registers the drawer reads, with random values, then draws, resolves and
/// a swap; some shaders are valid, some random words.
#[test]
fn random_state_never_panics() {
    let Some(mut d) = drawer() else { return };
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let interesting = [
        RB_MODECONTROL,
        RB_SURFACE_INFO,
        RB_COLOR_INFO,
        RB_DEPTH_INFO,
        RB_COLOR_MASK,
        RB_BLENDCONTROL0,
        RB_DEPTHCONTROL,
        PA_SU_SC_MODE_CNTL,
        PA_SC_WINDOW_SCISSOR_TL,
        PA_SC_WINDOW_SCISSOR_BR,
        PA_SC_WINDOW_OFFSET,
        PA_SC_SCREEN_SCISSOR_BR,
        PA_CL_VTE_CNTL,
        PA_SU_VTX_CNTL,
        PA_CL_VPORT_XSCALE,
        VGT_INDX_OFFSET,
        RB_COPY_CONTROL,
        RB_COPY_DEST_INFO,
        RB_COPY_DEST_PITCH,
        RB_COPY_DEST_BASE,
        RB_DEPTH_CLEAR,
        RB_COLOR_CLEAR,
        RB_COLOR_CLEAR_LO,
        SHADER_CONSTANT_FETCH,
        SHADER_CONSTANT_FETCH + 1,
        SHADER_CONSTANT_FETCH + 6,
        SHADER_CONSTANT_FETCH + 7,
    ];
    let mut mem = memory();
    for i in 0..mem.data.len() / 4 {
        let v = rng.next();
        mem.data[i * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    let mut stats = (0, 0, 0);
    let mut reasons: Vec<String> = Vec::new();
    for round in 0..120 {
        let mut s = Stream::default();
        s.setup().vertex_fetch(0, VERTS, 64);
        if round % 3 != 0 {
            s.shader(false, &vs_color()).shader(true, &ps_interp());
        } else {
            let n = 3 + rng.next() % 12;
            let words: Vec<u32> = (0..n).map(|_| rng.next()).collect();
            s.shader(round % 2 == 0, &words);
        }
        for _ in 0..1 + rng.next() % 6 {
            let reg = interesting[(rng.next() as usize) % interesting.len()];
            // Mostly small values so some draws get far enough to matter.
            let v = match rng.next() % 3 {
                0 => rng.next(),
                1 => rng.next() & 0xFF,
                _ => rng.next() & 0xFFFF,
            };
            s.regs(reg, &[v]);
        }
        // Rectangle lists (8) included: counts not a multiple of 3 too.
        let prims = [1, 2, 3, 4, 5, 6, 8, 13, rng.next() % 64];
        let prim = prims[(rng.next() % 9) as usize];
        let count = rng.next() % 300;
        match rng.next() % 3 {
            0 => s.draw_auto(prim, count),
            1 => s.draw_dma16(prim, count),
            _ => s.regs(RB_MODECONTROL, &[6]).draw_auto(8, 3),
        };
        if round % 10 == 9 {
            s.swap(rng.next() % 300, rng.next() % 300);
        }
        mem.write_words(CMD, &s.0);
        let mut rec = Recorder::new();
        if let Err(e) = rec.record_buffer(&mem, CMD, s.0.len() as u32) {
            eprintln!("record: {e:?}");
            continue;
        }
        let cmds: Vec<Command> = rec.commands;
        // Any Ok or Err is fine; a panic fails the test.
        match d.replay(&cmds, &mut mem as &mut dyn GuestMemory) {
            Ok(r) => {
                stats.0 += r.draws;
                stats.1 += r.resolves;
                for u in r.unsupported {
                    if !reasons.contains(&u) {
                        reasons.push(u);
                    }
                }
            }
            Err(_) => stats.2 += 1,
        }
    }
    eprintln!("draws, resolves, errors: {stats:?}; skipped: {reasons:?}");
}

/// Hand-made commands with extreme fields, outside anything the recorder
/// would produce.
#[test]
fn extreme_commands_never_panic() {
    use xenon_gpu::{Draw, Endian, IndexFormat, IndexSource, PrimitiveType, Resolve};
    let Some(mut d) = drawer() else { return };
    let mut mem = memory();
    let mut s = Stream::default();
    s.setup()
        .shader(false, &vs_color())
        .shader(true, &ps_interp())
        .vertex_fetch(0, 0xFFFF_FFFC, 0xFF_FFFF);
    mem.write_words(CMD, &s.0);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, CMD, s.0.len() as u32).unwrap();
    let mut cmds = rec.commands;
    let draw = |index_count, source| {
        Command::Draw(Draw {
            primitive: PrimitiveType::TriangleList,
            index_count,
            source,
            initiator: 0,
            viz_query: None,
        })
    };
    cmds.push(draw(u32::MAX, IndexSource::Auto));
    cmds.push(draw(
        300,
        IndexSource::Dma {
            address: u32::MAX - 1,
            buffer_indices: u32::MAX,
            format: IndexFormat::Int16,
            endian: Endian::E8in32,
        },
    ));
    let mut regs = xenon_gpu::RegisterFile::default();
    for (reg, v) in [
        (RB_COPY_CONTROL, u32::MAX),
        (RB_COPY_DEST_INFO, u32::MAX),
        (RB_COPY_DEST_PITCH, u32::MAX),
        (RB_COPY_DEST_BASE, u32::MAX),
        (RB_SURFACE_INFO, 0x3FFF),
        (PA_SC_WINDOW_OFFSET, 0x4000_4000),
        (PA_SC_WINDOW_SCISSOR_BR, u32::MAX),
        (PA_SC_SCREEN_SCISSOR_BR, 0x3FFF_3FFF),
    ] {
        regs.set(reg, v);
    }
    let mut r = Resolve::from_registers(&regs);
    cmds.push(Command::Resolve(Box::new(r.clone())));
    r.copy_command = xenon_gpu::CopyCommand::Raw;
    r.msaa_samples = 0;
    r.is_depth = false;
    r.src_format = 0;
    r.dest_format = 6;
    r.dest_endian = 2;
    r.dest_array = false;
    r.surface_pitch = u32::MAX;
    r.dest_pitch = u32::MAX;
    r.dest_base = u32::MAX;
    cmds.push(Command::Resolve(Box::new(r)));
    for (fb, w, h) in [(u32::MAX, 8192, 8192), (0, u32::MAX, 1), (0, 1280, 720)] {
        cmds.push(Command::Swap {
            signature_ok: false,
            frontbuffer: fb,
            width: w,
            height: h,
        });
    }
    cmds.push(Command::WriteRegisters {
        base: u32::MAX,
        values: vec![1, 2, 3],
    });
    cmds.push(Command::MemWrite {
        addr: u32::MAX - 2,
        endian: Endian::E8in32,
        values: vec![1, 2, 3],
    });
    // Each bad command is skipped or reported; replay goes on to the end.
    let report = d.replay(&cmds, &mut mem as &mut dyn GuestMemory).unwrap();
    eprintln!("{report:?}");
    assert_eq!(report.draws, 0);
}
