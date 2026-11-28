//! Headless command processor on top of `xenon-gpu`'s recorder: the
//! recorder parses the command ring, and this module applies what the host
//! must do itself (register side effects, memory and fence writes,
//! conditional writes, waits and interrupts). Draws are not rendered here;
//! the commands of one frame can be kept for a drawer or for the texture
//! dump in [`capture`].

use super::video::GPU_REGS;
use super::*;
use std::sync::Arc;
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
/// Vertical blanks so far: the value counter fences write.
pub static VBLANKS: AtomicU32 = AtomicU32::new(0);

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
        let mut b = vec![0; len];
        self.read_into(addr, &mut b).then_some(b)
    }
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        let a = gpu_to_cpu(addr);
        if a as u64 + out.len() as u64 > 0xC000_0000 {
            return false;
        }
        self.0.mem.read_bytes(a, out);
        true
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

    fn take_written(&mut self) -> Option<Vec<(u64, u64)>> {
        // Only the physical view: the OS walks its pages alone, an eighth
        // of the whole guest range, and writes elsewhere were dropped.
        let (lo, hi) = (PHYSICAL_VIEW.0 as usize, PHYSICAL_VIEW.1 as usize);
        Some(physical_ranges(&self.0.mem.take_written_in(lo, hi)?))
    }
}

/// [`Physical`] memory whose written ranges were taken on the command
/// processor thread when its ring span was queued, so the draw thread does
/// not walk the OS write record itself.
struct Taken<'a>(&'a Runtime, Option<Vec<(u64, u64)>>);

impl GuestRead for Taken<'_> {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        Physical(self.0).read_u32(addr)
    }
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        Physical(self.0).read_bytes(addr, len)
    }
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        Physical(self.0).read_into(addr, out)
    }
}

impl xenon_draw::GuestMemory for Taken<'_> {
    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> bool {
        Physical(self.0).write_bytes(addr, bytes)
    }

    fn take_written(&mut self) -> Option<Vec<(u64, u64)>> {
        self.1.take()
    }
}

/// One ring span for the drawer: its commands, the guest ranges written
/// since the span before ([`Physical`]'s `take_written`) and the time
/// taking them cost.
///
/// With the draw thread the ranges are taken when the span is queued, not
/// when it is replayed: a CPU write made while the span waits in the queue
/// is listed only with the next span queued. Until that replay, memo hits
/// (textures, indices, windows, kept resolves) give the bytes from before
/// the write while memo misses read the new ones. The drawer's memos thus
/// follow memory as the command processor saw it; nothing stays stale past
/// the next span. Inline drawing takes the ranges just before replaying,
/// as before.
struct Span {
    cmds: Vec<Command>,
    written: Option<Vec<(u64, u64)>>,
    watch: std::time::Duration,
}

impl Span {
    /// Takes the written ranges now, after the span's own memory writes.
    fn new(rt: &Runtime, cmds: Vec<Command>) -> Span {
        let start = std::time::Instant::now();
        let written = xenon_draw::GuestMemory::take_written(&mut Physical(rt));
        Span {
            cmds,
            written,
            watch: start.elapsed(),
        }
    }
}

/// The guest range every physical view aliases to (see [`physical_ranges`]).
const PHYSICAL_VIEW: (u64, u64) = (0xA000_0000, 0xC000_0000);

/// Written guest memory ranges ([`xenon_rt::mem::GuestMemory::take_written`]
/// offsets) as the physical ranges the GPU reads them at: the
/// `0xA000_0000` view, which every physical view aliases to.
fn physical_ranges(written: &[(u64, u64)]) -> Vec<(u64, u64)> {
    const LO: u64 = PHYSICAL_VIEW.0;
    const HI: u64 = PHYSICAL_VIEW.1;
    written
        .iter()
        .filter(|&&(a, b)| a < HI && LO < b)
        .map(|&(a, b)| (a.max(LO) - LO, b.min(HI) - LO))
        .collect()
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
    /// When set, applied commands are also kept here (see [`capture`]).
    pub keep: Option<Vec<Command>>,
    /// The last vertex and pixel shader loads, kept for a frame bundle
    /// (`--capture-bundle`) so that it starts with the shaders bound.
    shaders: [Option<Command>; 2],
    /// Renders every command when set (`--render`), see [`Gpu::set_drawer`].
    draw: Option<Draw>,
    /// The newest frame the drawer produced.
    pub last_frame: Arc<Mutex<Option<Arc<xenon_draw::Frame>>>>,
}

enum Draw {
    Inline(Box<DrawSide>),
    /// Batches go to the draw thread through this queue.
    Thread(Arc<xenon_gpu::ahead::Ahead<Span>>),
}

/// The drawer and what it reports, on the GPU thread or the draw thread.
struct DrawSide {
    drawer: xenon_draw::Drawer,
    stats: DrawStats,
    frames_drawn: u32,
    last_frame: Arc<Mutex<Option<Arc<xenon_draw::Frame>>>>,
}

impl DrawSide {
    /// Replays one ring span; false once the drawer stopped.
    fn replay(&mut self, rt: &Runtime, span: &mut Span) -> bool {
        let cmds = &span.cmds;
        // Every 30 s, what the drawer holds, to tell cache growth from a leak.
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let now = crate::ms();
        if now >= NEXT.load(Ordering::Relaxed) {
            NEXT.store(now + 30_000, Ordering::Relaxed);
            crate::log(&self.drawer.memory_line());
        }
        let start = std::time::Instant::now();
        let replayed = self
            .drawer
            .replay(cmds, &mut Taken(rt, span.written.take()));
        let busy = start.elapsed();
        // The drawer is never dropped at exit (its thread is left running),
        // so new pipelines reach the cache file here at most 5 s late.
        static SAVE: AtomicU32 = AtomicU32::new(0);
        if crate::ms() >= SAVE.load(Ordering::Relaxed) {
            SAVE.store(crate::ms() + 5_000, Ordering::Relaxed);
            let t = std::time::Instant::now();
            if let Err(e) = self.drawer.save_pipeline_cache() {
                crate::log(&format!("[draw] pipeline cache not saved: {e}"));
            }
            let t = t.elapsed();
            if t.as_millis() >= 2 {
                crate::log(&format!(
                    "[draw] pipeline cache save took {t:?} on the draw thread"
                ));
            }
        }
        match replayed {
            Ok(r) => {
                self.stats.add(&r, busy, span.watch);
                for u in r.unsupported.iter().chain(&r.warnings) {
                    capture::note(u);
                }
                for f in r.frames {
                    self.frames_drawn += 1;
                    let f = Arc::new(f);
                    capture::frame_done(self.frames_drawn, &f);
                    *self.last_frame.lock().unwrap() = Some(f);
                }
                true
            }
            Err(e) => {
                crate::log(&format!("[draw] stopped: {e}"));
                false
            }
        }
    }
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
    /// The longest replay: where a frame stalled.
    longest: std::time::Duration,
    /// Taking the written ranges, on the command processor thread.
    watch: std::time::Duration,
    t: xenon_draw::Timings,
}

impl DrawStats {
    const EVERY: std::time::Duration = std::time::Duration::from_secs(5);

    fn add(
        &mut self,
        r: &xenon_draw::Report,
        busy: std::time::Duration,
        watch: std::time::Duration,
    ) {
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        self.replays += 1;
        self.frames += r.frames.len() as u32;
        self.draws += r.draws;
        self.resolves += r.resolves;
        self.busy += busy;
        self.longest = self.longest.max(busy);
        self.watch += watch;
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
        let by = |i: usize| {
            let (count, bytes) = t.uploads_by_format[i];
            format!("{} ({:.1} MB)", count / n, mb(bytes))
        };
        // The drawer's time in none of the phases (they do not overlap but
        // for transfers, which are also counted where they were needed).
        let phases = [
            t.shaders,
            t.guest_data,
            t.textures,
            t.submit,
            t.gpu_wait,
            t.readback,
            t.resolve_cpu,
            t.clears,
            t.swap,
            t.commands,
            t.edges,
        ];
        let other = self.busy.saturating_sub(phases.iter().sum());
        let percent = |a: u32, b: u32| f64::from(a) * 100.0 / f64::from(b.max(1));
        format!(
            "[draw] {:.1} s: {} frames, {} replays; ms per {per}: drawer {:.0} = shaders {:.0} (state {:.0}, indices {:.0}, targets {:.0}, pipeline {:.0}), \
             guest data {:.0}, textures {:.0}, submit {:.0} (groups {:.1}, pass {:.1}, record {:.1}, queue {:.0}, upload {:.0}; finish {:.0} and submit {:.0} on the submit thread alongside), gpu wait {:.0}, readback {:.0}, \
             resolve cpu {:.0}, clears {:.0}, swap {:.0} (transfers {:.0}), commands {:.1}, edges {:.1}, other {:.1}; of those, creating \
             pipelines {:.0} (and {:.0} on worker threads); per {per}: {} draws ({} more skipped waiting for pipelines, {} consts reused, {} indices reused, {} texture sets reused = {:.0}%), {} shader loads, \
             {} resolves, {} submits ({} resolve, {} transfer, {} decode, {} flush/other), {} transfers ({} passes, {} set up), {} render passes, {} write-backs, readback {:.1} MB, \
             hashed {:.1} MB (kept resolves {:.1} MB), {} textures uploaded ({:.1} MB, {} evicted before, {:.1} decoded on the gpu), \
             {} evicted, {} from kept resolves, {} pipelines created; guest data {:.1} MB ({} windows reused, {:.1} MB); \
             uploads by format: rgba8 {}, bc {}, f16 {}, f32 {}, other {}; \
             write watch {:.1} ms per {per} on the gpu thread;              texture memo: {} re-hashes found the cached texture ({:.1} MB), {} writes only between levels, {} cap hits, {} restored by hashing written pages",
            window.as_secs_f64(),
            self.frames,
            self.replays,
            ms(self.busy),
            ms(t.shaders),
            ms(t.setup_parts[0]),
            ms(t.setup_parts[1]),
            ms(t.setup_parts[2]),
            ms(t.setup_parts[3]),
            ms(t.guest_data),
            ms(t.textures),
            ms(t.submit),
            ms(t.submit_parts[0]),
            ms(t.submit_parts[1]),
            ms(t.submit_parts[2]),
            ms(t.queue),
            ms(t.upload),
            ms(t.finish_thread),
            ms(t.submit_thread),
            ms(t.gpu_wait),
            ms(t.readback),
            ms(t.resolve_cpu),
            ms(t.clears),
            ms(t.swap),
            ms(t.transfer),
            ms(t.commands),
            ms(t.edges),
            ms(other),
            ms(t.pipeline),
            ms(t.pipeline_workers),
            self.draws / n,
            t.draws_waiting / n,
            t.consts_reused / n,
            t.indices_reused / n,
            t.texture_sets_reused / n,
            percent(t.texture_sets_reused, self.draws),
            t.shader_loads / n,
            self.resolves / n,
            t.submits / n,
            t.submits_by_site[0] / n,
            t.submits_by_site[1] / n,
            t.submits_by_site[2] / n,
            t.submits_by_site[3] / n,
            t.transfers / n,
            t.transfer_passes / n,
            t.transfer_setups / n,
            t.render_passes / n,
            t.writebacks / n,
            mb(t.readback_bytes),
            mb(t.hashed_bytes),
            mb(t.kept_hashed_bytes),
            t.textures_uploaded / n,
            mb(t.texture_bytes),
            t.textures_reuploaded / n,
            f64::from(t.textures_gpu_decoded) / f64::from(n),
            t.textures_evicted / n,
            t.textures_kept / n,
            t.pipelines_created / n,
            mb(t.guest_bytes),
            t.windows_reused / n,
            mb(t.reused_bytes),
            by(0),
            by(1),
            by(2),
            by(3),
            by(4),
            ms(self.watch),
            t.rehash_hits / n,
            mb(t.rehash_hit_bytes),
            t.gap_writes / n,
            t.memo_cap_hits,
            t.textures_restored / n,
        ) + &format!(
            "; longest replay {:.0} ms",
            self.longest.as_secs_f64() * 1e3
        ) + &if t.passes_timed > 0 {
            format!(
                "; GPU ms per {per} (timestamps, {} passes): draw {:.1}, clear {:.1}, resolve {:.1}",
                t.passes_timed / n,
                ms(t.gpu_passes[0]),
                ms(t.gpu_passes[1]),
                ms(t.gpu_passes[2]),
            )
        } else {
            String::new()
        }
    }
}

impl Gpu {
    /// Draws with `drawer` from now on, on a thread of its own: the command
    /// processor goes on up to `XENON_DRAW_AHEAD=<frames>` swaps ahead of
    /// it (default 1, so guest code and drawing overlap); 0 draws inline. Ahead, a
    /// title may reuse a buffer the drawer has not read yet, since fences
    /// are written when the command processor passes them.
    pub fn set_drawer(&mut self, rt: &Arc<Runtime>, drawer: xenon_draw::Drawer) {
        let mut side = DrawSide {
            drawer,
            stats: DrawStats::default(),
            frames_drawn: 0,
            last_frame: self.last_frame.clone(),
        };
        let ahead = std::env::var("XENON_DRAW_AHEAD")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(1);
        if ahead == 0 {
            self.draw = Some(Draw::Inline(Box::new(side)));
            return;
        }
        // 256 ring spans between swaps bound what the queue holds.
        let q = Arc::new(xenon_gpu::ahead::Ahead::<Span>::new(ahead, 256));
        let (q2, rt) = (q.clone(), rt.clone());
        // Replayed commands are freed on a thread of their own: thousands of
        // small allocations a frame, made on the command thread, were a
        // twentieth of the draw thread to free there.
        let (spent, freer) = std::sync::mpsc::channel::<Vec<Command>>();
        let _ = std::thread::Builder::new()
            .name("draw free".into())
            .spawn(move || freer.into_iter().for_each(drop));
        let spawned = std::thread::Builder::new()
            .name("draw".into())
            .spawn(move || {
                while let Some((mut span, swaps)) = q2.pop() {
                    let ok = side.replay(&rt, &mut span);
                    q2.done(swaps);
                    if let Err(e) = spent.send(span.cmds) {
                        drop(e.0);
                    }
                    if !ok {
                        q2.close();
                    }
                }
            });
        match spawned {
            Ok(_) => {
                crate::log(&format!(
                    "[draw] on its own thread, up to {ahead} frames behind"
                ));
                self.draw = Some(Draw::Thread(q));
            }
            Err(e) => crate::log(&format!("[draw] no draw thread: {e}")),
        }
    }

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
        if capture::BUNDLE.load(std::sync::atomic::Ordering::Relaxed) {
            for c in &cmds {
                if let Command::LoadShader { stage, .. } = c {
                    self.shaders[*stage as usize] = Some(c.clone());
                }
            }
        }
        for c in &cmds {
            self.apply(rt, c, idle);
        }
        let mut cmds = cmds;
        let going = match &mut self.draw {
            Some(Draw::Inline(d)) => {
                let mut span = Span::new(rt, std::mem::take(&mut cmds));
                let going = d.replay(rt, &mut span);
                cmds = span.cmds;
                going
            }
            Some(Draw::Thread(q)) => {
                let swaps = cmds
                    .iter()
                    .filter(|c| matches!(c, Command::Swap { .. }))
                    .count() as u32;
                let batch = match self.keep {
                    Some(_) => cmds.clone(),
                    None => std::mem::take(&mut cmds),
                };
                // Per span, in ring order: each replay sees the writes made
                // before its span was queued (see [`Span`]).
                q.push(Span::new(rt, batch), swaps)
            }
            None => true,
        };
        if !going {
            self.draw = None;
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
                        VBLANKS.load(Ordering::Relaxed)
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
    use xenon_gpu::regs::REGISTER_COUNT;
    use xenon_gpu::RegisterFile;
    use xenon_texture::{bytes_needed, decode_base_level, TextureFetch};

    /// When to keep a frame: `--capture <dir>` with `--capture-at <s>,<s>...`
    /// (seconds since start; default 20).
    pub static SETTINGS: std::sync::OnceLock<(std::path::PathBuf, Vec<u32>)> =
        std::sync::OnceLock::new();

    /// `--render`: draw every frame with the headless drawer.
    pub static RENDER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// `--capture-bundle`: each capture also writes a frame bundle (see
    /// [`xenon_gpu::bundle`]): the register file and shaders at the start,
    /// a warm-up frame and the captured frame, and guest memory after it.
    pub static BUNDLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// `--capture-frames N,M --capture-dir DIR`: frame numbers (counted
    /// from 1, one per swap) to write as `DIR/frame-N.png`.
    pub static FRAMES: std::sync::OnceLock<(std::path::PathBuf, Vec<u32>)> =
        std::sync::OnceLock::new();

    /// Writes the drawer's frame number `n` when it was asked for.
    pub fn frame_done(n: u32, f: &Arc<xenon_draw::Frame>) {
        // A frame with no pixels went to the window on the GPU already.
        if let Some(w) = super::super::input::WINDOW
            .get()
            .filter(|_| !f.rgba8.is_empty())
        {
            // Shared, not copied: a copy of every frame was 3.7 MB at 720p.
            w.show(f.width, f.height, f.clone());
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

    /// Logs each drawer note once (again after 4096 distinct notes, when
    /// the set is cleared so that per-texture log lines stay bounded).
    pub fn note(s: &str) {
        static SEEN: std::sync::LazyLock<Mutex<std::collections::HashSet<String>>> =
            std::sync::LazyLock::new(Default::default);
        let mut v = SEEN.lock().unwrap();
        if !v.contains(s) {
            if v.len() >= 4096 {
                v.clear();
            }
            crate::log(&format!("[draw] {s}"));
            v.insert(s.to_string());
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
                    g.keep = Some(if bundling() {
                        let mut k = vec![Command::WriteRegisters {
                            base: 0,
                            values: g.rec.regs.slice(0, REGISTER_COUNT as u32).to_vec(),
                        }];
                        k.extend(g.shaders.iter().flatten().cloned());
                        k
                    } else {
                        Vec::new()
                    });
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
            // A bundle keeps a warm-up frame before the captured one.
            let n = swaps.len();
            if n < 2 + usize::from(bundling()) {
                return;
            }
            let frame = &kept[swaps[n - 2] + 1..=swaps[n - 1]];
            let out = dir.join(format!("t{}", t / 1000));
            // With the draw thread this frame can be up to its frames behind.
            if let Some(f) = &*g.last_frame.lock().unwrap() {
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
            if bundling() {
                let dir = out.join("bundle");
                match write_bundle(rt, kept, &dir) {
                    Ok(()) => crate::log(&format!("[capture] frame bundle in {}", dir.display())),
                    Err(e) => crate::log(&format!("[capture] bundle {}: {e}", dir.display())),
                }
            }
            g.keep = None;
            self.at.pop();
        }
    }

    fn bundling() -> bool {
        BUNDLE.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// GPU physical memory the bundle saves: 512 MB from address 0.
    const BUNDLE_MEMORY: u32 = 0x2000_0000;

    fn write_bundle(rt: &Runtime, cmds: &[Command], dir: &Path) -> std::io::Result<()> {
        use std::io::Write as _;
        xenon_gpu::bundle::write_commands(dir, cmds)?;
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(dir.join(xenon_gpu::bundle::MEMORY))?);
        let mut buf = vec![0u8; 16 << 20];
        for a in (0..BUNDLE_MEMORY).step_by(buf.len()) {
            if !Physical(rt).read_into(a, &mut buf) {
                buf.fill(0);
            }
            f.write_all(&buf)?;
        }
        f.flush()
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
    /// viewport transform control, depth control, colour mask, colour
    /// control (alpha test), alpha reference and target 0 blend.
    fn state(regs: &RegisterFile) -> String {
        use xenon_gpu::regs::*;
        let r = |a: u32| regs.get(a);
        format!(
            "surf {:x} rt {:x} {:x} {:x} {:x} depth {:x} mode {:x} woff {:x} wsc {:x}-{:x} ssc {:x}-{:x} sc_mode {:x} vte {:x} zc {:x} mask {:x} cc {:x} aref {:x} blend {:x}",
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
            r(RB_COLORCONTROL),
            r(RB_ALPHA_REF),
            r(RB_BLENDCONTROL0),
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

    #[test]
    fn written_ranges_map_to_physical() {
        assert_eq!(
            physical_ranges(&[
                (0x4000_0000, 0x4000_1000),
                (0x9FFF_F000, 0xA000_2000),
                (0xA100_0000, 0xA100_3000),
                (0xBFFF_F000, 0xC000_1000),
            ]),
            [
                (0, 0x2000),
                (0x100_0000, 0x100_3000),
                (0x1FFF_F000, 0x2000_0000)
            ]
        );
    }

    /// A span lists the physical writes made before it was made; a write
    /// made while it waits for the drawer is not given to its replay but
    /// to the next span's.
    #[test]
    fn span_takes_writes_made_before_it() {
        let mem = xenon_rt::GuestMemory::new_watched(PHYSICAL_VIEW.0 as usize + (1 << 20));
        if !mem.watched() {
            return;
        }
        let rt = Runtime::new(mem);
        let _ = Span::new(&rt, Vec::new());
        rt.mem.write_u32(0xA000_1000, 1);
        let mut span = Span::new(&rt, Vec::new());
        rt.mem.write_u32(0xA000_3000, 2);
        let mut taken = Taken(&rt, span.written.take());
        assert_eq!(
            xenon_draw::GuestMemory::take_written(&mut taken),
            Some(vec![(0x1000, 0x2000)])
        );
        assert_eq!(xenon_draw::GuestMemory::take_written(&mut taken), None);
        let next = Span::new(&rt, Vec::new());
        assert_eq!(next.written, Some(vec![(0x3000, 0x4000)]));
    }

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
            watch: ms(3),
            ..Default::default()
        };
        s.t.readback = ms(300);
        s.t.submit = ms(160);
        s.t.queue = ms(60);
        s.t.upload = ms(20);
        s.t.submit_parts = [ms(20), ms(10), ms(40)];
        s.t.commands = ms(40);
        s.t.edges = ms(20);
        s.t.consts_reused = 600;
        s.t.texture_sets_reused = 690;
        s.t.shader_loads = 10;
        s.t.finish_thread = ms(30);
        s.t.submit_thread = ms(50);
        s.t.writebacks = 14;
        s.t.render_passes = 500;
        s.t.transfers = 12;
        s.t.transfer_passes = 60;
        s.t.readback_bytes = 80_000_000;
        s.t.guest_bytes = 9_000_000;
        s.t.submits = 280;
        s.t.submits_by_site = [88, 38, 34, 120];
        s.t.windows_reused = 900;
        s.t.reused_bytes = 3_000_000;
        s.t.uploads_by_format = [(600, 200_000_000), (40, 8_000_000), (6, 0), (0, 0), (0, 0)];
        s.t.textures_gpu_decoded = 5;
        let line = s.line(ms(5000));
        assert!(
            line.starts_with("[draw] 5.0 s: 2 frames, 7 replays; ms per frame: drawer 1000 ="),
            "{line}"
        );
        for part in [
            "readback 150,",
            "1380 draws",
            "42 resolves",
            "submit 80 (groups 10.0, pass 5.0, record 20.0, queue 30, upload 10; finish 15 and submit 25 on the submit thread alongside),",
            // 2000 ms busy less 300 readback, 160 submit, 40 commands and
            // 20 edges, per frame.
            "commands 20.0, edges 10.0, other 740.0;",
            "1380 draws (0 more skipped waiting for pipelines, 300 consts reused, 0 indices reused, 345 texture sets reused = 25%), 5 shader loads,",
            "6 transfers (30 passes, 0 set up), 250 render passes, 7 write-backs,",
            "readback 40.0 MB",
            "140 submits (44 resolve, 19 transfer, 17 decode, 60 flush/other)",
            "guest data 4.5 MB (450 windows reused, 1.5 MB)",
            "write watch 1.5 ms per frame on the gpu thread",
            "evicted before, 2.5 decoded on the gpu),",
            "rgba8 300 (100.0 MB), bc 20 (4.0 MB), f16 3 (0.0 MB), f32 0 (0.0 MB), other 0",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
        assert!(!line.contains("GPU ms"), "{line}");
        assert!(line.ends_with("; longest replay 0 ms"), "{line}");
        s.t.passes_timed = 400;
        s.t.gpu_passes = [ms(100), ms(3), ms(9)];
        let line = s.line(ms(5000));
        assert!(
            line.ends_with(
                "; GPU ms per frame (timestamps, 200 passes): draw 50.0, clear 1.5, resolve 4.5"
            ),
            "{line}"
        );
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
