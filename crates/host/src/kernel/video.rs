//! Video calls, headless. No GPU yet: the command ring is consumed as fast
//! as the title writes it (the read pointer write-back follows the write
//! pointer register), and the vertical-blank interrupt is raised at 60 Hz
//! so titles that pace themselves on it keep running. Each `VdSwap` is one
//! presented frame; the host counts them to see that the main loop runs.

use super::*;

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

/// Host side of the GPU: runs the command ring and raises interrupts
/// (command-processor interrupts as source 1, the vertical blank at 60 Hz
/// as source 0) on a context of its own. Runs until the process ends.
pub fn start(rt: std::sync::Arc<Runtime>) {
    use super::gpu::{gpu_to_cpu, Gpu, CP_RB_RPTR, CP_RB_WPTR};
    let _ = gpu_to_cpu;
    std::thread::Builder::new()
        .name("gpu".into())
        .spawn(move || {
            let (ctx, _) = super::threads::new_context(&rt, 64 << 10);
            // The callback runs as processor 2 for a vertical blank and as
            // the packet's processor for command interrupts.
            let raise = move |rt: &Runtime, source: u32, cpu: u32| {
                let irq = k().video.lock().unwrap().interrupt;
                if let Some((f, user)) = irq {
                    let mut c = ctx.clone();
                    rt.mem.write_u8(c.r[13] as u32 + 0x10C, cpu as u8);
                    c.r[3] = source as u64;
                    c.r[4] = user as u64;
                    let prev = super::gpu::STATE.swap(3, Ordering::Relaxed);
                    super::gpu::RAISED.fetch_add(1, Ordering::Relaxed);
                    crate::cpu_profile::interrupt(|| rt.call(&mut c, f));
                    super::gpu::STATE.store(prev, Ordering::Relaxed);
                }
            };
            let mut next_vblank = std::time::Instant::now();
            let rt2 = rt.clone();
            let mut idle = move |g: &mut Gpu| {
                for mask in std::mem::take(&mut g.interrupts) {
                    for cpu in (0..6).filter(|n| mask >> n & 1 != 0) {
                        raise(&rt2, 1, cpu);
                    }
                }
                if std::time::Instant::now() >= next_vblank {
                    next_vblank += std::time::Duration::from_micros(16_667);
                    g.vblanks += 1;
                    raise(&rt2, 0, 2);
                }
            };
            let mut gpu = Gpu::default();
            if super::gpu::capture::RENDER.load(Ordering::Relaxed) {
                match xenon_draw::Drawer::new() {
                    Ok(mut d) => {
                        crate::log(&format!("[draw] adapter: {}", d.adapter));
                        if d.skip_resolve_writeback {
                            crate::log("[draw] resolve write-back: skipped (the CPU never sees resolve output)");
                        }
                        if d.late_writeback {
                            crate::log("[draw] resolve write-back: late (the CPU sees resolves a frame later)");
                        }
                        // Opt-in: loading the cache cost about 900 MB of host
                        // memory in a warm run and gained no speed over the
                        // driver's own shader cache.
                        if let Some(dir) = std::env::var_os("XENON_PIPELINE_CACHE") {
                            let dir = std::path::PathBuf::from(dir);
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
