//! Guest address-range allocators for the kernel replacements: one per
//! region of the guest address space (virtual memory, kernel pool, physical
//! memory). Bookkeeping only; the bytes live in `GuestMemory`.

use std::collections::BTreeMap;

pub struct Heap {
    pub name: &'static str,
    lo: u32,
    hi: u32,
    /// Allocation granularity (a power of two).
    page: u32,
    /// Allocations by start address: size in bytes.
    map: BTreeMap<u32, u32>,
    /// Lowest address never handed out; memory above it is still zero.
    fresh: u32,
    used: u64,
    peak: u64,
}

fn round(x: u32, a: u32) -> Option<u32> {
    x.checked_add(a - 1).map(|v| v & !(a - 1))
}

impl Heap {
    pub fn new(name: &'static str, lo: u32, hi: u32, page: u32) -> Self {
        assert!(page.is_power_of_two() && lo % page == 0 && lo < hi);
        Heap {
            name,
            lo,
            hi,
            page,
            map: BTreeMap::new(),
            fresh: lo,
            used: 0,
            peak: 0,
        }
    }

    pub fn contains(&self, addr: u32) -> bool {
        (self.lo..self.hi).contains(&addr)
    }

    /// First fit of `size` bytes aligned to `align` (at least a page).
    /// Returns the address and whether the range was used before (so the
    /// caller must zero it).
    pub fn alloc(&mut self, size: u32, align: u32) -> Option<(u32, bool)> {
        let align = align.max(self.page).next_power_of_two();
        let size = round(size.max(1), self.page)?;
        let mut at = round(self.lo, align)?;
        for (&s, &n) in &self.map {
            if at.checked_add(size)? <= s {
                break;
            }
            at = at.max(round(s + n, align)?);
        }
        if at.checked_add(size)? > self.hi {
            return None;
        }
        Some((at, self.take(at, size)))
    }

    /// Allocates exactly `[addr, addr + size)` if it is free.
    pub fn alloc_at(&mut self, addr: u32, size: u32) -> Option<bool> {
        let addr = addr & !(self.page - 1);
        let size = round(size.max(1), self.page)?;
        let end = addr.checked_add(size)?;
        if addr < self.lo || end > self.hi {
            return None;
        }
        if let Some((&s, &n)) = self.map.range(..end).next_back() {
            if s + n > addr {
                return None;
            }
        }
        Some(self.take(addr, size))
    }

    fn take(&mut self, at: u32, size: u32) -> bool {
        // Only does something with guarded guest memory (--guard-memory).
        xenon_rt::guard::commit(at, size);
        self.map.insert(at, size);
        self.used += size as u64;
        self.peak = self.peak.max(self.used);
        let reused = at < self.fresh;
        self.fresh = self.fresh.max(at + size);
        reused
    }

    /// Frees the allocation starting at `addr`; returns its size.
    pub fn free(&mut self, addr: u32) -> Option<u32> {
        let n = self.map.remove(&addr)?;
        self.used -= n as u64;
        Some(n)
    }

    /// The allocation holding `addr`: (start, size).
    pub fn find(&self, addr: u32) -> Option<(u32, u32)> {
        let (&s, &n) = self.map.range(..=addr).next_back()?;
        (addr < s + n).then_some((s, n))
    }

    /// (bytes in use, peak bytes in use, bytes in the region).
    pub fn usage(&self) -> (u64, u64, u64) {
        (self.used, self.peak, (self.hi - self.lo) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_alignment_reuse_and_headroom() {
        let mut h = Heap::new("t", 0x1_0000, 0x10_0000, 0x1000);
        let (a, r) = h.alloc(0x1800, 0).unwrap();
        assert_eq!((a, r), (0x1_0000, false));
        let (b, _) = h.alloc(0x1000, 0x1_0000).unwrap();
        assert_eq!(b, 0x2_0000);
        assert_eq!(h.find(0x1_1FFF), Some((0x1_0000, 0x2000)));
        assert_eq!(h.free(a), Some(0x2000));
        // The freed range is handed out again and must be zeroed.
        assert_eq!(h.alloc(0x1000, 0), Some((0x1_0000, true)));
        assert_eq!(h.alloc_at(0x2_0000, 0x1000), None);
        assert_eq!(h.alloc_at(0x3_0000, 0x1000), Some(false));
        assert_eq!(h.usage().0, 0x3000);
        // Headroom: a request larger than what is left fails, never wraps.
        assert_eq!(h.alloc(0x10_0000, 0), None);
        assert_eq!(h.alloc(u32::MAX, 0), None);
    }
}
