//! Headless command processor: consumes the command ring as fast as the
//! title writes it. Register writes, memory writes, fence and event writes,
//! conditional writes and interrupts run; draws and render state are
//! skipped, so nothing is rendered. Packet layouts follow the GPU's
//! published PM4 format (facts checked against Xenia, BSD-3, no code).

use super::video::GPU_REGS;
use super::*;

pub const CP_RB_RPTR: u32 = 0x01C4;
pub const CP_RB_WPTR: u32 = 0x01C5;
const SCRATCH_UMSK: u32 = 0x01DC;
const SCRATCH_ADDR: u32 = 0x01DD;
const SCRATCH_REG0: u32 = 0x0578;
const COHER_STATUS_HOST: u32 = 0x0A31;
const REGS: usize = 0x6000;

/// GPU (physical) address to guest address, in the 0xA0000000 view.
pub fn gpu_to_cpu(a: u32) -> u32 {
    0xA000_0000 | (a & 0x1FFF_FFFF)
}

/// Byte order of a GPU memory access: the low two address bits. The GPU
/// reads memory little-endian, then applies this swap; [`load`] and
/// [`store`] give the value as the guest's big-endian word.
pub fn swap(v: u32, endian: u32) -> u32 {
    match endian & 3 {
        1 => (v << 8 & 0xFF00_FF00) | (v >> 8 & 0x00FF_00FF),
        2 => v.swap_bytes(),
        3 => v.rotate_left(16),
        _ => v,
    }
}

/// A GPU read of the word at physical `addr` (endian in the low bits).
fn load(rt: &Runtime, addr: u32) -> u32 {
    swap(rt.mem.read_u32(gpu_to_cpu(addr & !3)).swap_bytes(), addr)
}

/// A GPU write of `v` to physical `addr` (endian in the low bits).
fn store(rt: &Runtime, addr: u32, v: u32) {
    rt.mem
        .write_u32(gpu_to_cpu(addr & !3), swap(v, addr).swap_bytes());
}

fn compare(info: u32, v: u32, r: u32) -> bool {
    match info & 7 {
        1 => v < r,
        2 => v <= r,
        3 => v == r,
        4 => v != r,
        5 => v >= r,
        6 => v > r,
        7 => true,
        _ => false,
    }
}

/// What the command processor is doing, for stall reports: 0 idle, 1 in
/// the ring, 2 waiting in an indirect buffer, 3 in an interrupt handler;
/// and the address of the last wait that did not hold.
pub static STATE: AtomicU32 = AtomicU32::new(0);
pub static WAIT_ON: AtomicU32 = AtomicU32::new(0);
/// Rounds of the processor loop and interrupts raised.
pub static ROUNDS: AtomicU32 = AtomicU32::new(0);
pub static RAISED: AtomicU32 = AtomicU32::new(0);

pub struct Gpu {
    regs: Vec<u32>,
    /// Processor masks of interrupt packets seen, for the caller to raise.
    pub interrupts: Vec<u32>,
    /// Vertical blanks so far: the value counter fences write.
    pub vblanks: u32,
}

impl Default for Gpu {
    fn default() -> Self {
        Gpu {
            regs: vec![0; REGS],
            interrupts: Vec::new(),
            vblanks: 0,
        }
    }
}

impl Gpu {
    fn write_reg(&mut self, rt: &Runtime, i: u32, v: u32) {
        let Some(r) = self.regs.get_mut(i as usize) else {
            return;
        };
        *r = v;
        if (SCRATCH_REG0..SCRATCH_REG0 + 8).contains(&i) {
            let n = i - SCRATCH_REG0;
            if self.regs[SCRATCH_UMSK as usize] >> n & 1 != 0 {
                rt.mem
                    .write_u32(gpu_to_cpu(self.regs[SCRATCH_ADDR as usize] + 4 * n), v);
            }
        }
        // Cache flushes finish at once: the busy bit reads back clear.
        if i == COHER_STATUS_HOST {
            self.regs[i as usize] = v & 0x7FFF_FFFF;
        }
        rt.mem.write_u32(GPU_REGS + 4 * i, self.regs[i as usize]);
    }

    fn value(&self, rt: &Runtime, memory: bool, addr: u32) -> u32 {
        if memory {
            load(rt, addr)
        } else {
            self.regs.get(addr as usize).copied().unwrap_or(0)
        }
    }

    /// Executes `n` words at guest address `p` (an indirect buffer). A wait
    /// inside one blocks, calling `idle` between checks.
    pub fn run(
        &mut self,
        rt: &Runtime,
        p: u32,
        n: u32,
        depth: u32,
        idle: &mut dyn FnMut(&mut Gpu),
    ) {
        let mut i = 0;
        while i < n {
            let at = i;
            match self.packet(
                rt,
                |k| rt.mem.read_u32(p + 4 * (at + k)),
                n - i,
                depth,
                idle,
            ) {
                0 => {
                    STATE.store(2, Ordering::Relaxed);
                    idle(self);
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                len => i += len,
            }
        }
    }

    /// One packet; `word(k)` reads the k-th word from its header and `left`
    /// is how many words are there. Returns the packet length in words, or
    /// 0 for a wait whose condition does not hold yet (run it again later).
    pub fn packet(
        &mut self,
        rt: &Runtime,
        word: impl Fn(u32) -> u32,
        left: u32,
        depth: u32,
        idle: &mut dyn FnMut(&mut Gpu),
    ) -> u32 {
        let h = word(0);
        let count = (h >> 16 & 0x3FFF) + 1;
        if 1 + count > left && h >> 30 != 2 && h != 0 {
            return left; // truncated packet: drop the rest
        }
        match h >> 30 {
            0 if h == 0 => 1,
            0 => {
                let (base, one) = (h & 0x7FFF, h >> 15 & 1 != 0);
                for m in 0..count {
                    self.write_reg(rt, if one { base } else { base + m }, word(1 + m));
                }
                1 + count
            }
            1 => {
                self.write_reg(rt, h & 0x7FF, word(1));
                self.write_reg(rt, h >> 11 & 0x7FF, word(2));
                3
            }
            2 => 1,
            _ => {
                let d = |k: u32| word(1 + k);
                if std::env::var_os("XENON_TRACE_GPU").is_some() {
                    let words: Vec<String> =
                        (0..count.min(6)).map(|k| format!("{:x}", d(k))).collect();
                    crate::log(&format!(
                        "[gpu] op {:#04x} x{count} {}",
                        h >> 8 & 0x7F,
                        words.join(" ")
                    ));
                }
                match h >> 8 & 0x7F {
                    // INDIRECT_BUFFER, INDIRECT_BUFFER_PFD
                    0x3F | 0x37 if depth < 4 => {
                        self.run(rt, gpu_to_cpu(d(0)), d(1) & 0xF_FFFF, depth + 1, idle)
                    }
                    // WAIT_REG_MEM: until a register or memory word compares
                    // true against the reference.
                    0x3C => {
                        let (info, poll, rf, mask) = (d(0), d(1), d(2), d(3));
                        if !compare(info, self.value(rt, info & 0x10 != 0, poll) & mask, rf) {
                            WAIT_ON.store(poll, Ordering::Relaxed);
                            return 0;
                        }
                    }
                    // MEM_WRITE
                    0x3D => {
                        let a = d(0);
                        for k in 0..count - 1 {
                            store(rt, ((a & !3) + 4 * k) | (a & 3), d(1 + k));
                        }
                    }
                    // COND_WRITE
                    0x45 => {
                        let (info, poll, rf, mask, dst, data) =
                            (d(0), d(1), d(2), d(3), d(4), d(5));
                        if compare(info, self.value(rt, info & 0x10 != 0, poll) & mask, rf) {
                            if info & 0x100 != 0 {
                                store(rt, dst, data);
                            } else {
                                self.write_reg(rt, dst, data);
                            }
                        }
                    }
                    // EVENT_WRITE_SHD: a fence value, or the frame counter.
                    0x58 => {
                        let (init, a, v) = (d(0), d(1), d(2));
                        let v = if init >> 31 != 0 { self.vblanks } else { v };
                        store(rt, a, v);
                    }
                    // EVENT_WRITE_EXT: screen extents (min/max x, y, z).
                    0x59 => {
                        let a = gpu_to_cpu(d(1) & !3);
                        for (k, v) in [0u16, 8192 >> 3, 0, 8192 >> 3, 0, 1]
                            .into_iter()
                            .enumerate()
                        {
                            rt.mem.write_u16(a + 2 * k as u32, v);
                        }
                    }
                    // REG_RMW
                    0x21 => {
                        let (info, and, or) = (d(0), d(1), d(2));
                        let r = info & 0x1FFF;
                        let mut v = self.value(rt, false, r);
                        v &= if info >> 31 != 0 {
                            self.value(rt, false, and & 0x1FFF)
                        } else {
                            and
                        };
                        v |= if info >> 30 & 1 != 0 {
                            self.value(rt, false, or & 0x1FFF)
                        } else {
                            or
                        };
                        self.write_reg(rt, r, v);
                    }
                    // REG_TO_MEM
                    0x3E => {
                        let v = self.value(rt, false, d(0));
                        store(rt, d(1), v);
                    }
                    // INTERRUPT
                    0x54 => self.interrupts.push(d(0)),
                    // Draws and render state.
                    _ => {}
                }
                1 + count
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endian_swaps() {
        assert_eq!(swap(0x1122_3344, 0), 0x1122_3344);
        assert_eq!(swap(0x1122_3344, 1), 0x2211_4433);
        assert_eq!(swap(0x1122_3344, 2), 0x4433_2211);
        assert_eq!(swap(0x1122_3344, 3), 0x3344_1122);
        assert_eq!(gpu_to_cpu(0x0012_3000), 0xA012_3000);
        assert_eq!(gpu_to_cpu(0xA012_3000), 0xA012_3000);
        // Endian mode 2 (8 in 32) is the guest's own byte order.
        let rt = Runtime::new(xenon_rt::GuestMemory::new(1 << 32));
        store(&rt, 0x10 | 2, 0x1234_5678);
        assert_eq!(rt.mem.read_u32(0xA000_0010), 0x1234_5678);
        assert_eq!(load(&rt, 0x10 | 2), 0x1234_5678);
        assert_eq!(load(&rt, 0x10), 0x7856_3412);
    }
}
