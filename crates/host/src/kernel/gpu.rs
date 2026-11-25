//! Headless command processor on top of `xenon-gpu`'s recorder: the
//! recorder parses the command ring, and this module applies what the host
//! must do itself (register side effects, memory and fence writes,
//! conditional writes, waits and interrupts). Draws are not rendered here;
//! the commands of one frame can be kept for a drawer or for the texture
//! dump in [`capture`].

use super::video::GPU_REGS;
use super::*;
use xenon_gpu::{Command, Endian, EventKind, GuestRead, Recorder, WaitRegMem};

pub const CP_RB_RPTR: u32 = 0x01C4;
pub const CP_RB_WPTR: u32 = 0x01C5;
const SCRATCH_UMSK: u32 = 0x01DC;
const SCRATCH_ADDR: u32 = 0x01DD;
const SCRATCH_REG0: u32 = 0x0578;
const COHER_STATUS_HOST: u32 = 0x0A31;

/// What the command processor is doing, for stall reports: 0 idle, 1 in
/// the ring, 2 waiting, 3 in an interrupt handler; and the address of the
/// last wait that did not hold.
pub static STATE: AtomicU32 = AtomicU32::new(0);
pub static WAIT_ON: AtomicU32 = AtomicU32::new(0);
/// Rounds of the processor loop and interrupts raised.
pub static ROUNDS: AtomicU32 = AtomicU32::new(0);
pub static RAISED: AtomicU32 = AtomicU32::new(0);

/// GPU (physical) address to guest address, in the 0xA0000000 view.
pub fn gpu_to_cpu(a: u32) -> u32 {
    0xA000_0000 | (a & 0x1FFF_FFFF)
}

/// Guest memory as the GPU reads it (physical addresses).
pub struct Physical<'a>(pub &'a Runtime);

impl GuestRead for Physical<'_> {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        Some(self.0.mem.read_u32(gpu_to_cpu(addr)))
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        let a = gpu_to_cpu(addr);
        if a as u64 + len as u64 > 0xC000_0000 {
            return None;
        }
        let mut b = vec![0; len];
        self.0.mem.read_bytes(a, &mut b);
        Some(b)
    }
}

impl xenon_draw::GuestMemory for Physical<'_> {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        let a = gpu_to_cpu(addr);
        if a as u64 + bytes.len() as u64 > 0xC000_0000 {
            return false;
        }
        self.0.mem.write_bytes(a, bytes);
        true
    }
}

/// A GPU write of `v` at physical `addr` in byte order `e`.
fn store(rt: &Runtime, addr: u32, e: Endian, v: u32) {
    rt.mem
        .write_bytes(gpu_to_cpu(addr & !3), &e.to_guest_bytes(v));
}

fn load(rt: &Runtime, addr: u32, e: Endian) -> u32 {
    let mut b = [0u8; 4];
    rt.mem.read_bytes(gpu_to_cpu(addr & !3), &mut b);
    e.from_guest_bytes(b)
}

#[derive(Default)]
pub struct Gpu {
    rec: Recorder,
    /// Processor masks of interrupt packets applied, for the caller to raise.
    pub interrupts: Vec<u32>,
    /// Vertical blanks so far: the value counter fences write.
    pub vblanks: u32,
    /// When set, applied commands are also kept here (see [`capture`]).
    pub keep: Option<Vec<Command>>,
    /// Renders every command when set (`--render`).
    pub drawer: Option<xenon_draw::Drawer>,
    /// The newest frame the drawer produced, and how many it produced.
    pub last_frame: Option<xenon_draw::Frame>,
    pub frames_drawn: u32,
    stats: DrawStats,
}

/// Drawer replays summed over a window, logged every [`DrawStats::EVERY`].
#[derive(Default)]
struct DrawStats {
    since: Option<std::time::Instant>,
    replays: u32,
    frames: u32,
    draws: u32,
    resolves: u32,
    /// Wall time inside the drawer.
    busy: std::time::Duration,
    t: xenon_draw::Timings,
}

impl DrawStats {
    const EVERY: std::time::Duration = std::time::Duration::from_secs(5);

    fn add(&mut self, r: &xenon_draw::Report, busy: std::time::Duration) {
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        self.replays += 1;
        self.frames += r.frames.len() as u32;
        self.draws += r.draws;
        self.resolves += r.resolves;
        self.busy += busy;
        self.t += &r.timings;
        let window = since.elapsed();
        if window >= Self::EVERY {
            crate::log(&self.line(window));
            *self = DrawStats::default();
        }
    }

    /// Per frame (or per window when no frame was swapped): milliseconds by
    /// phase, then counts.
    fn line(&self, window: std::time::Duration) -> String {
        let (n, per) = match self.frames {
            0 => (1, "window"),
            f => (f, "frame"),
        };
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / f64::from(n);
        let mb = |b: u64| b as f64 / 1e6 / f64::from(n);
        let t = &self.t;
        format!(
            "[draw] {:.1} s: {} frames, {} replays; ms per {per}: drawer {:.0} = shaders {:.0}, \
             guest data {:.0}, textures {:.0}, submit {:.0}, gpu wait {:.0}, readback {:.0}, \
             resolve cpu {:.0}, clears {:.0}, swap {:.0} (transfers {:.0}); per {per}: {} draws, \
             {} resolves, {} submits, {} transfers ({} passes), readback {:.1} MB, \
             hashed {:.1} MB, {} textures uploaded ({:.1} MB, {} evicted before), \
             {} evicted, {} from kept resolves, {} pipelines created",
            window.as_secs_f64(),
            self.frames,
            self.replays,
            ms(self.busy),
            ms(t.shaders),
            ms(t.guest_data),
            ms(t.textures),
            ms(t.submit),
            ms(t.gpu_wait),
            ms(t.readback),
            ms(t.resolve_cpu),
            ms(t.clears),
            ms(t.swap),
            ms(t.transfer),
            self.draws / n,
            self.resolves / n,
            t.submits / n,
            t.transfers / n,
            t.transfer_passes / n,
            mb(t.readback_bytes),
            mb(t.hashed_bytes),
            t.textures_uploaded / n,
            mb(t.texture_bytes),
            t.textures_reuploaded / n,
            t.textures_evicted / n,
            t.textures_kept / n,
            t.pipelines_created / n,
        )
    }
}

impl Gpu {
    fn write_reg(&mut self, rt: &Runtime, i: u32, v: u32) {
        let mut v = v;
        if (SCRATCH_REG0..SCRATCH_REG0 + 8).contains(&i) {
            let n = i - SCRATCH_REG0;
            if self.rec.regs.get(SCRATCH_UMSK) >> n & 1 != 0 {
                rt.mem
                    .write_u32(gpu_to_cpu(self.rec.regs.get(SCRATCH_ADDR) + 4 * n), v);
            }
        }
        // Cache flushes finish at once: the busy bit reads back clear.
        if i == COHER_STATUS_HOST {
            v &= 0x7FFF_FFFF;
            self.rec.regs.set(i, v);
        }
        if i < 0x6000 {
            rt.mem.write_u32(GPU_REGS + 4 * i, v);
        }
    }

    fn holds(&self, rt: &Runtime, w: &WaitRegMem) -> bool {
        let v = if w.memory {
            load(rt, w.poll_addr, w.endian)
        } else {
            self.rec.regs.get(w.poll_addr)
        };
        w.matches(v)
    }

    /// Runs the ring from `rptr` to `wptr`; returns the new read pointer.
    /// A wait blocks, calling `idle` between checks.
    pub fn run_ring(
        &mut self,
        rt: &Runtime,
        ring: u32,
        words: u32,
        rptr: u32,
        wptr: u32,
        idle: &mut dyn FnMut(&mut Gpu),
    ) -> u32 {
        let r = self
            .rec
            .record_ring(&Physical(rt), ring, words, rptr, wptr)
            .unwrap_or(wptr);
        let cmds = std::mem::take(&mut self.rec.commands);
        self.rec.warnings.clear();
        for c in &cmds {
            self.apply(rt, c, idle);
        }
        if let Some(d) = self.drawer.as_mut() {
            // Every 30 s, what the drawer holds, to tell cache growth from a leak.
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let now = crate::ms();
            if now >= NEXT.load(Ordering::Relaxed) {
                NEXT.store(now + 30_000, Ordering::Relaxed);
                crate::log(&d.memory_line());
            }
            let start = std::time::Instant::now();
            match d.replay(&cmds, &mut Physical(rt)) {
                Ok(r) => {
                    self.stats.add(&r, start.elapsed());
                    for u in r.unsupported.iter().chain(&r.warnings) {
                        capture::note(u);
                    }
                    for f in r.frames {
                        self.frames_drawn += 1;
                        capture::frame_done(self.frames_drawn, &f);
                        self.last_frame = Some(f);
                    }
                }
                Err(e) => {
                    crate::log(&format!("[draw] stopped: {e}"));
                    self.drawer = None;
                }
            }
        }
        if let Some(k) = self.keep.as_mut() {
            k.extend(cmds);
        }
        r
    }

    fn apply(&mut self, rt: &Runtime, c: &Command, idle: &mut dyn FnMut(&mut Gpu)) {
        match c {
            Command::WriteRegisters { base, values } => {
                for (i, v) in values.iter().enumerate() {
                    self.write_reg(rt, base + i as u32, *v);
                }
            }
            Command::MemWrite {
                addr,
                endian,
                values,
            } => {
                for (i, v) in values.iter().enumerate() {
                    store(rt, addr + 4 * i as u32, *endian, *v);
                }
            }
            Command::RegToMem {
                register,
                addr,
                endian,
            } => store(rt, *addr, *endian, self.rec.regs.get(*register)),
            Command::EventWrite(e) => match (e.kind, e.address) {
                (EventKind::Shader, Some(a)) => {
                    let v = if e.use_counter {
                        self.vblanks
                    } else {
                        e.value.unwrap_or(0)
                    };
                    store(rt, a, e.endian, v);
                }
                // Screen extents: min/max x, y, z as 16-bit values.
                (EventKind::ScreenExtent, Some(a)) => {
                    for (k, v) in [0u16, 8192 >> 3, 0, 8192 >> 3, 0, 1]
                        .into_iter()
                        .enumerate()
                    {
                        rt.mem.write_u16(gpu_to_cpu(a) + 2 * k as u32, v);
                    }
                }
                _ => {}
            },
            Command::CondWrite {
                wait,
                write_memory,
                write_addr,
                write_endian,
                write_data,
            } => {
                if self.holds(rt, wait) {
                    if *write_memory {
                        store(rt, *write_addr, *write_endian, *write_data);
                    } else {
                        self.rec.regs.set(*write_addr, *write_data);
                        self.write_reg(rt, *write_addr, *write_data);
                    }
                }
            }
            Command::WaitRegMem(w) => {
                while !self.holds(rt, w) {
                    WAIT_ON.store(w.poll_addr, Ordering::Relaxed);
                    STATE.store(2, Ordering::Relaxed);
                    idle(self);
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                STATE.store(1, Ordering::Relaxed);
            }
            Command::Interrupt { cpu_mask } => self.interrupts.push(*cpu_mask),
            _ => {}
        }
    }
}

/// The packets `VdSwap` leaves in the ring: the front buffer's fetch
/// constant (with its physical address) as a register write, then the swap
/// with the front buffer, width and height.
pub fn swap_packets(fetch: [u32; 6], frontbuffer: u32, width: u32, height: u32) -> Vec<u32> {
    let mut f = fetch;
    f[1] = (f[1] & 0xFFF) | (frontbuffer & 0x1FFF_F000);
    let mut v = vec![(6 - 1) << 16 | 0x4800];
    v.extend(f);
    v.extend([
        0xC000_0000 | (4 - 1) << 16 | 0x64 << 8,
        u32::from_be_bytes(*b"SWAP"),
        frontbuffer & 0x1FFF_FFFF,
        width,
        height,
    ]);
    v
}

pub mod capture {
    //! Writes what one frame used, to find out which screen a title shows
    //! before frames are rendered: every texture bound at a draw, decoded
    //! to a BMP, and a list of the draws. Local files only: they are
    //! derived from the game.

    use super::*;
    use std::collections::BTreeSet;
    use std::fmt::Write as _;
    use std::path::Path;
    use xenon_gpu::RegisterFile;
    use xenon_texture::{bytes_needed, decode_base_level, TextureFetch};

    /// When to keep a frame: `--capture <dir>` with `--capture-at <s>,<s>...`
    /// (seconds since start; default 20).
    pub static SETTINGS: std::sync::OnceLock<(std::path::PathBuf, Vec<u32>)> =
        std::sync::OnceLock::new();

    /// `--render`: draw every frame with the headless drawer.
    pub static RENDER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// `--capture-frames N,M --capture-dir DIR`: frame numbers (counted
    /// from 1, one per swap) to write as `DIR/frame-N.png`.
    pub static FRAMES: std::sync::OnceLock<(std::path::PathBuf, Vec<u32>)> =
        std::sync::OnceLock::new();

    /// Writes the drawer's frame number `n` when it was asked for.
    pub fn frame_done(n: u32, f: &xenon_draw::Frame) {
        if let Some(w) = super::super::input::WINDOW.get() {
            w.show(f.width, f.height, f.rgba8.clone());
        }
        let Some((dir, want)) = FRAMES.get() else {
            return;
        };
        if !want.contains(&n) {
            return;
        }
        let path = dir.join(format!("frame-{n}.png"));
        let r = std::fs::create_dir_all(dir)
            .map_err(|e| e.to_string())
            .and_then(|_| f.write_png(&path).map_err(|e| e.to_string()));
        match r {
            Ok(()) => crate::log(&format!(
                "[capture] {} ({}x{})",
                path.display(),
                f.width,
                f.height
            )),
            Err(e) => crate::log(&format!("[capture] {}: {e}", path.display())),
        }
    }

    /// Logs each drawer note once.
    pub fn note(s: &str) {
        static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let mut v = SEEN.lock().unwrap();
        if !v.iter().any(|x| x == s) {
            crate::log(&format!("[draw] {s}"));
            v.push(s.to_string());
        }
    }

    /// The capture schedule, run by the GPU thread after each ring span.
    pub struct Plan {
        dir: Option<std::path::PathBuf>,
        /// Capture times left, in milliseconds since start.
        at: Vec<u32>,
    }

    impl Plan {
        pub fn from_settings() -> Self {
            match SETTINGS.get() {
                Some((d, at)) => Plan {
                    dir: Some(d.clone()),
                    at: at.iter().rev().map(|s| s * 1000).collect(),
                },
                None => Plan {
                    dir: None,
                    at: vec![],
                },
            }
        }

        /// Starts keeping commands when a capture time has come, and once a
        /// whole frame (swap to swap) is kept, writes it out.
        pub fn after_ring(&mut self, rt: &Runtime, g: &mut Gpu) {
            let Some(dir) = &self.dir else { return };
            let Some(&t) = self.at.last() else { return };
            if g.keep.is_none() {
                if crate::ms() >= t {
                    g.keep = Some(Vec::new());
                }
                return;
            }
            let kept = g.keep.as_ref().unwrap();
            let swaps: Vec<usize> = kept
                .iter()
                .enumerate()
                .filter(|(_, c)| matches!(c, Command::Swap { .. }))
                .map(|(i, _)| i)
                .collect();
            if swaps.len() < 2 {
                return;
            }
            let frame = &kept[swaps[0] + 1..=swaps[1]];
            let out = dir.join(format!("t{}", t / 1000));
            if let Some(f) = &g.last_frame {
                let _ = std::fs::create_dir_all(&out);
                match f.write_png(out.join("frame.png")) {
                    Ok(()) => crate::log(&format!(
                        "[capture] frame image {}x{} written",
                        f.width, f.height
                    )),
                    Err(e) => crate::log(&format!("[capture] frame image: {e}")),
                }
            }
            match dump(rt, frame, &out) {
                Ok((d, n)) => crate::log(&format!(
                    "[capture] frame at {} s: {d} draws, {n} textures in {}",
                    t / 1000,
                    out.display()
                )),
                Err(e) => crate::log(&format!("[capture] {}: {e}", out.display())),
            }
            g.keep = None;
            self.at.pop();
        }
    }

    /// Returns (draws, textures written).
    pub fn dump(rt: &Runtime, cmds: &[Command], dir: &Path) -> std::io::Result<(usize, usize)> {
        std::fs::create_dir_all(dir)?;
        let mut regs = RegisterFile::default();
        let mut seen = BTreeSet::new();
        let mut list = String::new();
        let (mut draws, mut written) = (0, 0);
        for c in cmds {
            match c {
                Command::WriteRegisters { base, values } => {
                    for (i, v) in values.iter().enumerate() {
                        regs.set(base + i as u32, *v);
                    }
                }
                Command::Draw(d) => {
                    draws += 1;
                    let mut used = Vec::new();
                    for n in 0..32 {
                        let f = TextureFetch::decode(regs.fetch_constant(n));
                        if f.fetch_type != 2 || f.base_address == 0 {
                            continue;
                        }
                        used.push(format!("t{n}={:08x}", f.base_address));
                        if !seen.insert((f.base_address, f.width(), f.height())) {
                            continue;
                        }
                        let Ok(need) = bytes_needed(&f) else { continue };
                        let Some(bytes) = Physical(rt).read_bytes(f.base_address, need) else {
                            continue;
                        };
                        if let Ok(img) = decode_base_level(&f, &bytes) {
                            let name = format!(
                                "tex_{:08x}_{}x{}_{:?}.bmp",
                                f.base_address, img.width, img.height, f.format
                            );
                            write_bmp(&dir.join(name), img.width, img.height, &img.rgba8)?;
                            written += 1;
                        }
                    }
                    let _ = writeln!(
                        list,
                        "draw {draws}: {:?} x{} {} | {}",
                        d.primitive,
                        d.index_count,
                        used.join(" "),
                        state(&regs)
                    );
                }
                Command::Resolve(r) => {
                    // The rectangle's three corners at vertex fetch 0 (six
                    // floats, read big-endian as Direct3D writes them).
                    let f0 = regs.fetch_constant(0);
                    let rect: Vec<String> = Physical(rt)
                        .read_bytes(f0[0] & 0x1FFF_FFFC, 24)
                        .map(|b| {
                            b.chunks_exact(4)
                                .map(|c| {
                                    f32::from_be_bytes(c.try_into().unwrap_or([0; 4])).to_string()
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let _ = writeln!(
                        list,
                        "resolve: source {} format {}, destination format {} pitch {} height {}, {:?} | edram base {} pitch {} msaa {} dest 0x{:08x} rect [{}] copy_control {:08x} dest_info {:08x} | {}",
                        if r.is_depth { "depth" } else { "colour" },
                        r.src_format,
                        r.dest_format,
                        r.dest_pitch,
                        r.dest_height,
                        r.copy_command,
                        r.src_edram_base,
                        r.surface_pitch,
                        r.msaa_samples,
                        r.dest_base,
                        rect.join(" "),
                        r.copy_control,
                        r.dest_info,
                        state(&regs)
                    );
                }
                Command::Swap { width, height, .. } => {
                    let _ = writeln!(list, "swap {width}x{height}");
                }
                _ => {}
            }
        }
        std::fs::write(dir.join("draws.txt"), list)?;
        Ok((draws, written))
    }

    /// Render state that places a draw or resolve in EDRAM and on screen,
    /// raw register values in hex: surface, colour targets 0..3, depth,
    /// mode, window offset, window and screen scissors, mode control,
    /// viewport transform control, depth control and colour mask.
    fn state(regs: &RegisterFile) -> String {
        use xenon_gpu::regs::*;
        let r = |a: u32| regs.get(a);
        format!(
            "surf {:x} rt {:x} {:x} {:x} {:x} depth {:x} mode {:x} woff {:x} wsc {:x}-{:x} ssc {:x}-{:x} sc_mode {:x} vte {:x} zc {:x} mask {:x}",
            r(RB_SURFACE_INFO),
            r(RB_COLOR_INFO_RT[0]),
            r(RB_COLOR_INFO_RT[1]),
            r(RB_COLOR_INFO_RT[2]),
            r(RB_COLOR_INFO_RT[3]),
            r(RB_DEPTH_INFO),
            r(RB_MODECONTROL),
            r(PA_SC_WINDOW_OFFSET),
            r(PA_SC_WINDOW_SCISSOR_TL),
            r(PA_SC_WINDOW_SCISSOR_TL + 1),
            r(PA_SC_SCREEN_SCISSOR_TL),
            r(PA_SC_SCREEN_SCISSOR_TL + 1),
            r(PA_SU_SC_MODE_CNTL),
            r(PA_CL_VTE_CNTL),
            r(RB_DEPTHCONTROL),
            r(RB_COLOR_MASK),
        )
    }

    /// An uncompressed 32-bit BMP, rows top to bottom.
    pub fn write_bmp(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
        let size = 54 + 4 * w * h;
        let mut b = Vec::with_capacity(size as usize);
        b.extend(b"BM");
        b.extend(size.to_le_bytes());
        b.extend([0; 4]);
        b.extend(54u32.to_le_bytes());
        b.extend(40u32.to_le_bytes());
        b.extend((w as i32).to_le_bytes());
        b.extend((-(h as i32)).to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(32u16.to_le_bytes());
        b.extend([0; 24]);
        for p in rgba.chunks_exact(4) {
            b.extend([p[2], p[1], p[0], p[3]]);
        }
        std::fs::write(path, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stats line divides by the frames of the window, or by one window
    /// when none was swapped.
    #[test]
    fn draw_stats_line_is_per_frame() {
        let ms = std::time::Duration::from_millis;
        let mut s = DrawStats {
            frames: 2,
            replays: 7,
            draws: 2760,
            resolves: 84,
            busy: ms(2000),
            ..Default::default()
        };
        s.t.readback = ms(300);
        s.t.transfers = 12;
        s.t.transfer_passes = 60;
        s.t.readback_bytes = 80_000_000;
        let line = s.line(ms(5000));
        assert!(
            line.starts_with("[draw] 5.0 s: 2 frames, 7 replays; ms per frame: drawer 1000 ="),
            "{line}"
        );
        for part in [
            "readback 150,",
            "1380 draws",
            "42 resolves",
            "6 transfers (30 passes)",
            "readback 40.0 MB",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
        s.frames = 0;
        assert!(s.line(ms(5000)).contains("ms per window: drawer 2000 ="));
    }

    #[test]
    fn swap_packets_are_what_the_recorder_reads() {
        let words = swap_packets([2, 0x1234_5000 | 0x86, 0, 0, 0, 0], 0xA012_3000, 1280, 720);
        let mut r = Recorder::new();
        r.record_words(&xenon_gpu::SliceMemory::default(), &words);
        assert!(r.commands.iter().any(|c| matches!(
            c,
            Command::Swap {
                signature_ok: true,
                frontbuffer: 0x0012_3000,
                width: 1280,
                height: 720
            }
        )));
        assert_eq!(r.regs.fetch_constant(0)[1], 0x0012_3086);
    }

    #[test]
    fn bmp_header_and_pixel_order() {
        let p = std::env::temp_dir().join(format!("xenon-bmp-{}.bmp", std::process::id()));
        capture::write_bmp(&p, 2, 1, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..2], b"BM");
        assert_eq!(b.len(), 54 + 8);
        assert_eq!(&b[54..58], &[3, 2, 1, 4]);
        std::fs::remove_file(p).unwrap();
    }
}
