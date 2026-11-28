//! Video calls, headless. No GPU yet: the command ring is consumed as fast
//! as the title writes it (the read pointer write-back follows the write
//! pointer register), and the vertical-blank interrupt is raised at 60 Hz
//! so titles that pace themselves on it keep running. Each `VdSwap` is one
//! presented frame; the host counts them to see that the main loop runs.

use super::*;
use std::time::{Duration, Instant};
use xenon_window::pace::{Pacing, Spread};

/// The guest's swaps and vertical blanks as they happened, summed up each
/// second in the run log (see [`pace_line`]).
static SWAP_PACE: Mutex<Pacing> = Mutex::new(Pacing::new());
static VBLANK_PACE: Mutex<Pacing> = Mutex::new(Pacing::new());
/// Vertical blanks skipped because their thread woke a whole period late.
static VBLANKS_SKIPPED: AtomicU32 = AtomicU32::new(0);

const VBLANK_PERIOD: Duration = Duration::from_micros(16_667);

/// One line on how evenly the title swapped and the vertical blank came
/// over the last second.
pub fn pace_line() -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let f = |s: Option<Spread>| match s {
        Some(s) => format!(
            "{} intervals, ms p50 {:.1} p99 {:.1} max {:.1} min {:.1}, over 25 ms: {}",
            s.frames,
            ms(s.p50),
            ms(s.p99),
            ms(s.max),
            ms(s.min),
            s.hitches
        ),
        None => "none".into(),
    };
    format!(
        "[pace] swaps: {}; vertical blanks: {}, skipped {}",
        f(SWAP_PACE.lock().unwrap().take()),
        f(VBLANK_PACE.lock().unwrap().take()),
        VBLANKS_SKIPPED.swap(0, Ordering::Relaxed)
    )
}

#[derive(Default)]
pub struct Video {
    pub ring: u32,
    pub ring_log2: u32,
    pub writeback: u32,
    pub interrupt: Option<(u32, u32)>,
}

/// GPU register file in guest memory, and the ring write pointer register.
pub const GPU_REGS: u32 = 0x7FC8_0000;

fn mode(rt: &Runtime, p: u32) {
    // display width, height, interlaced, widescreen, high definition,
    // refresh rate, video standard (NTSC), two words seen on hardware.
    fill(rt, p, 0x30, 0);
    for (i, v) in [1280u32, 720, 0, 1, 1, 60f32.to_bits(), 1, 0x4A, 1]
        .into_iter()
        .enumerate()
    {
        rt.mem.write_u32(p + 4 * i as u32, v);
    }
}

pub fn vd_query_video_mode(rt: &Runtime, c: &mut Context) {
    mode(rt, arg(rt, c, 0));
}

pub fn vd_query_video_flags(_: &Runtime, c: &mut Context) {
    ret(c, 3); // widescreen, at least 1024 wide
}

/// Display information (0x58 bytes): front buffer size, scaler source and
/// output, display size and refresh rate.
pub fn vd_get_current_display_information(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 0);
    fill(rt, p, 0x58, 0);
    rt.mem.write_u16(p, 1280);
    rt.mem.write_u16(p + 2, 720);
    rt.mem.write_u32(p + 0x10, 1280);
    rt.mem.write_u32(p + 0x14, 720);
    rt.mem.write_u32(p + 0x18, 1280);
    rt.mem.write_u32(p + 0x1C, 720);
    rt.mem.write_u16(p + 0x48, 1280);
    rt.mem.write_u16(p + 0x4A, 720);
    rt.mem.write_u32(p + 0x4C, 60f32.to_bits());
    rt.mem.write_u16(p + 0x56, 1280);
}

pub fn vd_get_current_display_gamma(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 0), 2);
    rt.mem.write_u32(arg(rt, c, 1), 2.22222f32.to_bits());
}

pub fn vd_initialize_ring_buffer(rt: &Runtime, c: &mut Context) {
    let mut v = k().video.lock().unwrap();
    v.ring = arg(rt, c, 0);
    v.ring_log2 = arg(rt, c, 1);
}

pub fn vd_enable_ring_buffer_r_ptr_write_back(rt: &Runtime, c: &mut Context) {
    k().video.lock().unwrap().writeback = arg(rt, c, 0);
}

pub fn vd_set_graphics_interrupt_callback(rt: &Runtime, c: &mut Context) {
    let (f, user) = (arg(rt, c, 0), arg(rt, c, 1));
    k().video.lock().unwrap().interrupt = (f != 0).then_some((f, user));
}

pub fn vd_get_system_command_buffer(rt: &Runtime, c: &mut Context) {
    let (p0, p1) = (arg(rt, c, 0), arg(rt, c, 1));
    fill(rt, p0, 0x94, 0);
    rt.mem.write_u32(p0, 0xBEEF_0000);
    rt.mem.write_u32(p1, 0xBEEF_0001);
}

/// The title leaves 64 words in the ring for the swap; fill them with
/// type-2 no-op packets.
/// (ring buffer space, front buffer fetch constant, ..., front buffer
/// address pointer, format, color space, width, height). The title leaves
/// 64 words in the ring; they get the swap packets (see
/// [`super::gpu::swap_packets`]) and type-2 no-ops after them.
pub fn vd_swap(rt: &Runtime, c: &mut Context) {
    let (buf, fetch_p, fb_p) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 5));
    let (w_p, h_p) = (arg(rt, c, 8), arg(rt, c, 9));
    let fetch: [u32; 6] = std::array::from_fn(|i| rt.mem.read_u32(fetch_p + 4 * i as u32));
    let read = |p: u32| if p != 0 { rt.mem.read_u32(p) } else { 0 };
    let words = super::gpu::swap_packets(fetch, read(fb_p), read(w_p), read(h_p));
    for i in 0..64 {
        rt.mem.write_u32(
            buf + 4 * i,
            words.get(i as usize).copied().unwrap_or(0x8000_0000),
        );
    }
    k().swaps.fetch_add(1, Ordering::Relaxed);
    SWAP_PACE.lock().unwrap().mark(Instant::now());
    ret(c, 0);
}

/// (type, out): the title later frees `*out` with MmFreePhysicalMemory
/// (type 1), so it must be physical memory, not pool.
pub fn vd_persist_display(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 1);
    if p != 0 {
        let a = super::memory::physical(rt, 64, 32);
        rt.mem.write_u32(p, a);
    }
    ret(c, 1);
}

pub fn vd_initialize_engines(_: &Runtime, c: &mut Context) {
    ret(c, 1);
}
pub fn vd_shutdown_engines(_: &Runtime, _: &mut Context) {}
pub fn vd_set_display_mode(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn vd_enable_disable_clock_gating(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn vd_is_hsio_training_succeeded(_: &Runtime, c: &mut Context) {
    ret(c, 1);
}
pub fn vd_retrain_edram(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn vd_retrain_edram_worker(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn vd_call_graphics_notification_routines(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn vd_set_system_command_buffer_gpu_identifier_address(_: &Runtime, _: &mut Context) {}

/// Fills the scaler command buffer with no-op packets.
pub fn vd_initialize_scaler_command_buffer(rt: &Runtime, c: &mut Context) {
    let (dest, n) = (arg(rt, c, 10), arg(rt, c, 11));
    for i in 0..n.min(0x1000) {
        rt.mem.write_u32(dest + 4 * i, 0x8000_0000);
    }
    ret(c, 0);
}

/// Runs the title's graphics interrupt callback, if it set one, for
/// `source` as processor `cpu`. The command processor's and the vertical
/// blank's interrupts take turns on the one context `irq`.
fn raise(rt: &Runtime, irq: &Mutex<Context>, source: u32, cpu: u32) {
    let callback = k().video.lock().unwrap().interrupt;
    if let Some((f, user)) = callback {
        let held = irq.lock().unwrap();
        let mut c = held.clone();
        rt.mem.write_u8(c.r[13] as u32 + 0x10C, cpu as u8);
        c.r[3] = source as u64;
        c.r[4] = user as u64;
        super::gpu::RAISED.fetch_add(1, Ordering::Relaxed);
        crate::cpu_profile::interrupt(|| rt.call(&mut c, f));
    }
}

/// One vertical blank: counted, and raised as source 0 on processor 2.
fn vblank(rt: &Runtime, irq: &Mutex<Context>) {
    super::gpu::VBLANKS.fetch_add(1, Ordering::Relaxed);
    VBLANK_PACE.lock().unwrap().mark(Instant::now());
    raise(rt, irq, 0, 2);
}

/// When the vertical blank after one due at `due` is, woken at `now`: a
/// period on, or a period from `now` (the missed ones skipped, not raised
/// in a burst) when `now` is a whole period late.
fn next_vblank(due: Instant, now: Instant) -> Instant {
    if now.saturating_duration_since(due) >= VBLANK_PERIOD {
        VBLANKS_SKIPPED.fetch_add(1, Ordering::Relaxed);
        now + VBLANK_PERIOD
    } else {
        due + VBLANK_PERIOD
    }
}

/// Raises the vertical blank at 60 Hz on a thread of its own, so it comes
/// on time while the command processor waits on the drawer or runs a long
/// span: raised from the processor's loop, it could come late and then
/// several at once, and the title (which retires a swap a vertical blank)
/// made its frames in bursts that the window could not show evenly.
fn start_vblank(rt: std::sync::Arc<Runtime>, irq: std::sync::Arc<Mutex<Context>>) {
    let spawned = std::thread::Builder::new()
        .name("vblank".into())
        .spawn(move || {
            let mut due = Instant::now() + VBLANK_PERIOD;
            loop {
                let now = Instant::now();
                if now < due {
                    std::thread::sleep(due - now);
                    continue;
                }
                due = next_vblank(due, now);
                vblank(&rt, &irq);
            }
        });
    if let Err(e) = spawned {
        crate::log(&format!("[video] no vertical blank thread: {e}"));
    }
}

/// Host side of the GPU: runs the command ring and raises interrupts
/// (command-processor interrupts as source 1, the vertical blank at 60 Hz
/// as source 0, on a thread of its own unless `XENON_VBLANK=cp` raises it
/// from the processor's loop as before) on a context of its own. Runs
/// until the process ends.
pub fn start(rt: std::sync::Arc<Runtime>) {
    use super::gpu::{gpu_to_cpu, Gpu, CP_RB_RPTR, CP_RB_WPTR, STATE};
    let _ = gpu_to_cpu;
    std::thread::Builder::new()
        .name("gpu".into())
        .spawn(move || {
            let (ctx, _) = super::threads::new_context(&rt, 64 << 10);
            let irq = std::sync::Arc::new(Mutex::new(ctx));
            let cp_vblank = std::env::var("XENON_VBLANK").as_deref() == Ok("cp");
            if cp_vblank {
                crate::log("[video] vertical blank raised from the command processor's loop");
            } else {
                start_vblank(rt.clone(), irq.clone());
            }
            // The callback runs as processor 2 for a vertical blank and as
            // the packet's processor for command interrupts.
            let mut due = Instant::now();
            let rt2 = rt.clone();
            let mut idle = move |g: &mut Gpu| {
                for mask in std::mem::take(&mut g.interrupts) {
                    for cpu in (0..6).filter(|n| mask >> n & 1 != 0) {
                        let prev = STATE.swap(3, Ordering::Relaxed);
                        raise(&rt2, &irq, 1, cpu);
                        STATE.store(prev, Ordering::Relaxed);
                    }
                }
                if cp_vblank && Instant::now() >= due {
                    due += VBLANK_PERIOD;
                    let prev = STATE.swap(3, Ordering::Relaxed);
                    vblank(&rt2, &irq);
                    STATE.store(prev, Ordering::Relaxed);
                }
            };
            let mut gpu = Gpu::default();
            if super::gpu::capture::RENDER.load(Ordering::Relaxed) {
                match xenon_draw::Drawer::new() {
                    Ok(mut d) => {
                        crate::log(&format!("[draw] adapter: {}", d.adapter));
                        // Skipped unless asked for: in play it showed no
                        // visual change and saved about 23 ms a frame.
                        let mode = std::env::var("XENON_RESOLVE_WRITEBACK");
                        d.skip_resolve_writeback = !matches!(mode.as_deref(), Ok("on" | "late"));
                        // Frame images come one swap late, so a swap never
                        // waits for the GPU (XENON_LATE_FRAME=0 turns it off).
                        d.late_front_buffer = std::env::var("XENON_LATE_FRAME").as_deref() != Ok("0");
                        // New shaders and pipelines are made on worker threads and
                        // their draws skipped until then, so a frame never stalls
                        // on a compile (XENON_ASYNC_PIPELINES=0 makes them on the
                        // spot, as captures need for frames that match).
                        d.async_pipelines = std::env::var("XENON_ASYNC_PIPELINES").as_deref() != Ok("0")
                            && super::gpu::capture::SETTINGS.get().is_none()
                            && super::gpu::capture::FRAMES.get().is_none();
                        // Frames at the drawing scale (XENON_SCALE above 1),
                        // but captures keep guest size, which they compare.
                        // Only with a window: nothing else shows the frame,
                        // and the copies cost GPU memory and readback.
                        use super::gpu::capture::{FRAMES, SETTINGS};
                        d.sharp_front = super::input::WINDOW.get().is_some()
                            && SETTINGS.get().is_none()
                            && FRAMES.get().is_none();
                        // Frames go to the window on the GPU, with no
                        // read-back, unless captures need their pixels
                        // (XENON_GPU_PRESENT=0 reads them back as before).
                        let window = (super::input::WINDOW.get()).filter(|_| {
                            SETTINGS.get().is_none()
                                && FRAMES.get().is_none()
                                && std::env::var("XENON_GPU_PRESENT").as_deref() != Ok("0")
                        });
                        if let Some(w) = window {
                            let (instance, adapter, device, queue) = d.gpu();
                            let gpu = xenon_window::Gpu {
                                instance,
                                adapter,
                                device,
                                queue,
                            };
                            match w.attach_gpu(gpu) {
                                Ok(()) => {
                                    let w = w.clone();
                                    d.present = Some(std::sync::Arc::new(move |t| w.show_texture(t)));
                                    crate::log("[draw] frames go to the window on the GPU (XENON_GPU_PRESENT=0 reads them back)");
                                }
                                Err(e) => crate::log(&format!("[draw] frames read back for the window: {e}")),
                            }
                        }
                        if d.skip_resolve_writeback {
                            crate::log("[draw] resolve write-back: skipped (the CPU never sees resolve output; XENON_RESOLVE_WRITEBACK=on writes it back)");
                        }
                        if d.late_writeback {
                            crate::log("[draw] resolve write-back: late (the CPU sees resolves a frame later)");
                        }
                        // Compiled pipelines persist across runs, beside the
                        // title's saves unless XENON_PIPELINE_CACHE names a
                        // directory (0 turns it off). It spares the seconds-long
                        // pipeline stall of a cold driver cache only when the
                        // shaders match the file (a copied exe, an evicted
                        // driver cache), not after a rebuild that changes them.
                        let dir = match std::env::var_os("XENON_PIPELINE_CACHE") {
                            Some(v) if v == "0" => None,
                            Some(v) => Some(std::path::PathBuf::from(v)),
                            None => Some(k().data_dir.join("pipeline-cache")),
                        };
                        if let Some(dir) = dir {
                            match d.set_pipeline_cache_dir(&dir) {
                                Ok(Some(f)) => {
                                    crate::log(&format!("[draw] pipeline cache: {}", f.display()))
                                }
                                Ok(None) => crate::log("[draw] no pipeline cache on this device"),
                                Err(e) => crate::log(&format!(
                                    "[draw] no pipeline cache in {}: {e}",
                                    dir.display()
                                )),
                            }
                        }
                        gpu.set_drawer(&rt, d);
                    }
                    Err(e) => crate::log(&format!("[draw] no drawer: {e}")),
                }
            }
            // Registers the title reads and the hardware keeps current:
            // EDRAM timing, cache control, the scan-out line counter, the
            // display interrupt status (vertical blank pending) and the
            // viewport size.
            let fixed = [
                (0x0F00u32, 0x0810_0748u32),
                (0x0F01, 0x200E),
                (0x194C, 0x2D0),
                (0x1951, 1),
                (0x1961, 0x0500_02D0),
            ];
            let mut rptr = 0u32;
            let mut capture = super::gpu::capture::Plan::from_settings();
            loop {
                let (ring, log2, wb) = {
                    let v = k().video.lock().unwrap();
                    (v.ring, v.ring_log2, v.writeback)
                };
                if ring != 0 {
                    let words = ((1u32 << (log2 + 3)) / 4).max(1);
                    let w = rt.mem.read_u32(GPU_REGS + 4 * CP_RB_WPTR) % words;
                    super::gpu::STATE.store(1, Ordering::Relaxed);
                    if rptr != w {
                        rptr = gpu.run_ring(&rt, ring, words, rptr, w, &mut idle);
                        capture.after_ring(&rt, &mut gpu);
                    }
                    rt.mem.write_u32(GPU_REGS + 4 * CP_RB_RPTR, rptr);
                    if wb != 0 {
                        rt.mem.write_u32(gpu_to_cpu(wb), rptr);
                    }
                }
                for (r, v) in fixed {
                    rt.mem.write_u32(GPU_REGS + 4 * r, v);
                }
                super::gpu::STATE.store(0, Ordering::Relaxed);
                super::gpu::ROUNDS.fetch_add(1, Ordering::Relaxed);
                idle(&mut gpu);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        })
        .expect("cannot start a host thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_vertical_blank_skips_rather_than_bursts() {
        let t = Instant::now();
        let ms = Duration::from_millis;
        // On time or a little late: the next is a period after the due one.
        assert_eq!(next_vblank(t, t), t + VBLANK_PERIOD);
        assert_eq!(next_vblank(t, t + ms(5)), t + VBLANK_PERIOD);
        // A whole period late: the next is a period from now, not due.
        let now = t + ms(50);
        assert_eq!(next_vblank(t, now), now + VBLANK_PERIOD);
    }
}
