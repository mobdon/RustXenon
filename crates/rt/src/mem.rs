//! Big-endian guest memory addressed by 32-bit guest pointers, shared by all
//! guest threads.
//!
//! One allocation made up front and never moved: guest address `a` lives at
//! `base + a`. Large allocations come straight from the OS as zero pages, so
//! untouched guest RAM costs nothing.
//!
//! Guest threads run at the same time on host threads, so every access is a
//! host atomic: aligned loads and stores are relaxed atomics of their own
//! width (plain moves on x86 and ARM), unaligned ones go byte by byte, and the
//! reserved stores (`stwcx.`, `stdcx.`) are compare-and-swap. Ordering
//! between threads comes from the guest's own barriers (`sync`, `lwsync`,
//! `isync`, `eieio`), which translate to host fences.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8};

pub struct GuestMemory {
    base: NonNull<u8>,
    len: usize,
    /// Reserved with no access ([`crate::guard`]) instead of allocated.
    guarded: bool,
    /// Allocated with the OS recording writes ([`crate::watch`]).
    watched: bool,
}

// SAFETY: every access through `&self` is atomic (see the module notes);
// `bytes_mut` needs `&mut self`.
unsafe impl Send for GuestMemory {}
unsafe impl Sync for GuestMemory {}

/// Base alignment. Kept at the allocator's natural alignment so a large
/// zeroed allocation stays lazy (a larger one makes the allocator zero it
/// by hand, touching every page). Covers the widest atomic access.
const ALIGN: usize = 16;

macro_rules! access {
    ($read:ident, $write:ident, $t:ty, $atomic:ty) => {
        #[inline]
        pub fn $read(&self, addr: u32) -> $t {
            const N: usize = size_of::<$t>();
            if addr as usize % N == 0 {
                let p = self.at(addr, N);
                // SAFETY: in bounds and aligned (the base is 16-byte aligned).
                <$t>::from_be(unsafe { <$atomic>::from_ptr(p.cast()) }.load(Relaxed))
            } else if crosses_view(addr, N) {
                let mut b = [0u8; N];
                for (i, x) in b.iter_mut().enumerate() {
                    *x = self.read_u8(addr.wrapping_add(i as u32));
                }
                <$t>::from_be_bytes(b)
            } else {
                let p = self.at(addr, N);
                let mut b = [0u8; N];
                for (i, x) in b.iter_mut().enumerate() {
                    *x = byte(p, i).load(Relaxed);
                }
                <$t>::from_be_bytes(b)
            }
        }

        #[inline]
        pub fn $write(&self, addr: u32, v: $t) {
            const N: usize = size_of::<$t>();
            if addr as usize % N == 0 {
                let p = self.at(addr, N);
                // SAFETY: in bounds and aligned.
                unsafe { <$atomic>::from_ptr(p.cast()) }.store(v.to_be(), Relaxed);
            } else if crosses_view(addr, N) {
                for (i, x) in v.to_be_bytes().into_iter().enumerate() {
                    self.write_u8(addr.wrapping_add(i as u32), x);
                }
            } else {
                let p = self.at(addr, N);
                for (i, x) in v.to_be_bytes().into_iter().enumerate() {
                    byte(p, i).store(x, Relaxed);
                }
            }
        }
    };
}

/// The console maps its 512 MB of physical memory three times: at
/// 0xA0000000 (64 KB pages), 0xC0000000 (16 MB pages) and 0xE0000000 (4 KB
/// pages, offset by 4 KB). The upper two views are folded onto the first, so
/// a buffer written through one view reads back through another.
#[inline]
pub(crate) fn alias(addr: u32) -> u32 {
    if addr < 0xC000_0000 {
        return addr;
    }
    let off = if addr >= 0xE000_0000 { 0x1000 } else { 0 };
    0xA000_0000 | ((addr & 0x1FFF_FFFF) + off) & 0x1FFF_FFFF
}

/// An access (unaligned scalar or byte range; aligned scalars never cross)
/// whose bytes are not contiguous after `alias`: it runs
/// off the end of a mirrored view and wraps, so it goes byte by byte.
#[inline]
fn crosses_view(addr: u32, n: usize) -> bool {
    let last = n as u32 - 1;
    alias(addr.wrapping_add(last)) != alias(addr).wrapping_add(last)
}

/// Byte `i` after `p`, which `at` has bounds checked.
#[inline]
fn byte<'a>(p: *mut u8, i: usize) -> &'a AtomicU8 {
    // SAFETY: callers only pass offsets inside a checked range.
    unsafe { AtomicU8::from_ptr(p.add(i)) }
}

/// For `n` bytes at `p`: the bytes before the first 8-byte aligned one
/// (all of them when none is in range), and the whole aligned words after.
#[inline]
fn split(p: *mut u8, n: usize) -> (usize, usize) {
    let head = p.align_offset(8).min(n);
    (head, (n - head) / 8)
}

impl GuestMemory {
    /// Allocates `size` bytes of zeroed guest memory.
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size.max(1), ALIGN).expect("guest memory size");
        // SAFETY: the layout has a non-zero size.
        let base = NonNull::new(unsafe { alloc_zeroed(layout) })
            .unwrap_or_else(|| panic!("cannot reserve {size} bytes of guest memory"));
        Self {
            base,
            len: size,
            guarded: false,
            watched: false,
        }
    }

    /// Like [`Self::new`], with the OS recording the pages written (see
    /// [`Self::take_written`]); plain [`Self::new`] memory where the OS
    /// cannot.
    pub fn new_watched(size: usize) -> Self {
        match NonNull::new(crate::watch::allocate(size.max(1))) {
            Some(base) => Self {
                base,
                len: size,
                guarded: false,
                watched: true,
            },
            None => Self::new(size),
        }
    }

    /// Whether [`Self::take_written`] can list writes.
    pub fn watched(&self) -> bool {
        self.watched
    }

    /// The `[start, end)` offsets (the addresses [`alias`] maps to) of the
    /// pages written since the last call, merged and sorted, with the
    /// record reset; `None` when writes are not recorded or were too many
    /// to list: then any byte may have changed.
    pub fn take_written(&self) -> Option<Vec<(u64, u64)>> {
        self.take_written_in(0, self.len)
    }

    /// [`Self::take_written`] for the offsets `[lo, hi)` alone (page
    /// aligned): the OS walks only those pages, and writes elsewhere stay
    /// recorded. The offsets it gives are still from the start.
    pub fn take_written_in(&self, lo: usize, hi: usize) -> Option<Vec<(u64, u64)>> {
        let hi = hi.min(self.len);
        if !self.watched || lo >= hi {
            return self.watched.then(Vec::new);
        }
        // SAFETY: lo < hi <= len, inside the allocation.
        let p = unsafe { self.base.as_ptr().add(lo) };
        let (pages, size) = crate::watch::take_written(p, hi - lo)?;
        let mut out = merge_pages(&pages, size);
        for r in &mut out {
            *r = (r.0 + lo as u64, r.1 + lo as u64);
        }
        Some(out)
    }

    /// Reserves `size` bytes with no access: only ranges passed to
    /// [`crate::guard::commit`] can be used, and touching any other address
    /// faults into the [`crate::guard::on_wild_access`] handler. One per
    /// process.
    pub fn new_guarded(size: usize) -> Self {
        let base = NonNull::new(crate::guard::reserve(size.max(1))).expect("reserved base");
        Self {
            base,
            len: size,
            guarded: true,
            watched: false,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host pointer for `n` bytes at `addr`; stops with the address when the
    /// range is outside guest memory (one compare, and it keeps the pointer
    /// math sound).
    #[inline]
    fn at(&self, addr: u32, n: usize) -> *mut u8 {
        let a = alias(addr) as usize;
        if a + n > self.len {
            out_of_range(addr, n, self.len);
        }
        // SAFETY: just checked.
        unsafe { self.base.as_ptr().add(a) }
    }

    #[inline]
    pub fn read_u8(&self, addr: u32) -> u8 {
        byte(self.at(addr, 1), 0).load(Relaxed)
    }
    #[inline]
    pub fn write_u8(&self, addr: u32, v: u8) {
        byte(self.at(addr, 1), 0).store(v, Relaxed)
    }

    access!(read_u16, write_u16, u16, AtomicU16);
    access!(read_u32, write_u32, u32, AtomicU32);
    access!(read_u64, write_u64, u64, AtomicU64);

    /// Reads 16 bytes at `addr` as a vector (element 0 at the lowest address).
    pub fn read_v128(&self, addr: u32) -> crate::V128 {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&self.read_u64(addr).to_be_bytes());
        b[8..].copy_from_slice(&self.read_u64(addr.wrapping_add(8)).to_be_bytes());
        crate::vmx::from_bytes(b)
    }

    pub fn write_v128(&self, addr: u32, v: crate::V128) {
        let b = crate::vmx::bytes(v);
        self.write_u64(addr, u64::from_be_bytes(b[..8].try_into().unwrap()));
        self.write_u64(
            addr.wrapping_add(8),
            u64::from_be_bytes(b[8..].try_into().unwrap()),
        );
    }

    /// Zeroes the `size`-byte cache block holding `addr` (`dcbz`, `dcbz128`).
    pub fn zero_block(&self, addr: u32, size: u32) {
        let a = addr & !(size - 1);
        for i in (0..size).step_by(8) {
            self.write_u64(a + i, 0);
        }
    }

    /// Conditional store for `stwcx.`: writes `new` only if the word still
    /// holds `old`, as one host compare-and-swap.
    pub fn cas_u32(&self, addr: u32, old: u32, new: u32) -> bool {
        let p = self.reserved(addr, 4);
        // SAFETY: in bounds and aligned.
        unsafe { AtomicU32::from_ptr(p.cast()) }
            .compare_exchange(old.to_be(), new.to_be(), SeqCst, SeqCst)
            .is_ok()
    }

    /// Conditional store for `stdcx.`.
    pub fn cas_u64(&self, addr: u32, old: u64, new: u64) -> bool {
        let p = self.reserved(addr, 8);
        // SAFETY: in bounds and aligned.
        unsafe { AtomicU64::from_ptr(p.cast()) }
            .compare_exchange(old.to_be(), new.to_be(), SeqCst, SeqCst)
            .is_ok()
    }

    /// The console raises an alignment interrupt for an unaligned reserved
    /// access; stop with the address instead.
    fn reserved(&self, addr: u32, n: usize) -> *mut u8 {
        if addr as usize % n != 0 {
            panic!("unaligned {n}-byte reserved store at {addr:#010x}");
        }
        self.at(addr, n)
    }

    /// Copies guest bytes at `addr` into `out`.
    pub fn read_bytes(&self, addr: u32, out: &mut [u8]) {
        if !out.is_empty() && crosses_view(addr, out.len()) {
            for (i, x) in out.iter_mut().enumerate() {
                *x = self.read_u8(addr.wrapping_add(i as u32));
            }
            return;
        }
        let p = self.at(addr, out.len());
        // Bytewise up to the first 8-byte boundary and after the last one,
        // relaxed 8-byte loads between: the bytes are the same, only
        // tearing against a concurrent store is per word.
        let (head, words) = split(p, out.len());
        for (i, x) in out[..head].iter_mut().enumerate() {
            *x = byte(p, i).load(Relaxed);
        }
        let (mid, tail) = out[head..].split_at_mut(words * 8);
        for (i, x) in mid.chunks_exact_mut(8).enumerate() {
            // SAFETY: in bounds and 8-byte aligned (see `split`).
            let w = unsafe { AtomicU64::from_ptr(p.add(head + 8 * i).cast()) };
            x.copy_from_slice(&w.load(Relaxed).to_ne_bytes());
        }
        let at = head + words * 8;
        for (i, x) in tail.iter_mut().enumerate() {
            *x = byte(p, at + i).load(Relaxed);
        }
    }

    /// Copies `data` into guest memory at `addr`.
    pub fn write_bytes(&self, addr: u32, data: &[u8]) {
        if !data.is_empty() && crosses_view(addr, data.len()) {
            for (i, &x) in data.iter().enumerate() {
                self.write_u8(addr.wrapping_add(i as u32), x);
            }
            return;
        }
        let p = self.at(addr, data.len());
        // As in `read_bytes`.
        let (head, words) = split(p, data.len());
        for (i, &x) in data[..head].iter().enumerate() {
            byte(p, i).store(x, Relaxed);
        }
        let (mid, tail) = data[head..].split_at(words * 8);
        for (i, x) in mid.chunks_exact(8).enumerate() {
            // SAFETY: in bounds and 8-byte aligned (see `split`).
            let w = unsafe { AtomicU64::from_ptr(p.add(head + 8 * i).cast()) };
            w.store(u64::from_ne_bytes(x.try_into().unwrap()), Relaxed);
        }
        let at = head + words * 8;
        for (i, &x) in tail.iter().enumerate() {
            byte(p, at + i).store(x, Relaxed);
        }
    }

    /// All of guest memory as plain bytes, for loaders before any guest
    /// thread runs (needs exclusive access).
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // A slice must be readable throughout; reserved pages are not.
        assert!(!self.guarded, "bytes_mut on guarded guest memory");
        // SAFETY: `&mut self` rules out concurrent access.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), self.len) }
    }
}

impl Drop for GuestMemory {
    fn drop(&mut self) {
        if self.guarded {
            crate::guard::release(self.base.as_ptr(), self.len.max(1));
            return;
        }
        if self.watched {
            crate::watch::release(self.base.as_ptr());
            return;
        }
        let layout = Layout::from_size_align(self.len.max(1), ALIGN).unwrap();
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.base.as_ptr(), layout) }
    }
}

/// Sorted page offsets of `size` bytes each as merged `[start, end)` ranges.
fn merge_pages(pages: &[usize], size: usize) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &p in pages {
        let (a, b) = (p as u64, (p + size) as u64);
        match out.last_mut() {
            Some(last) if last.1 == a => last.1 = b,
            _ => out.push((a, b)),
        }
    }
    out
}

#[cold]
#[inline(never)]
fn out_of_range(addr: u32, n: usize, len: usize) -> ! {
    panic!("guest access of {n} bytes at {addr:#010x} is outside the {len}-byte guest memory");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_pages_merge_into_ranges() {
        assert_eq!(
            merge_pages(&[0, 4096, 12288, 16384, 32768], 4096),
            [(0, 8192), (12288, 20480), (32768, 36864)]
        );
        assert!(merge_pages(&[], 4096).is_empty());
    }

    /// Watched memory reads and writes like any other; where the OS cannot
    /// watch, `take_written` says so (`None`) rather than listing nothing.
    #[test]
    fn watched_memory_lists_writes_or_says_it_cannot() {
        let m = GuestMemory::new_watched(1 << 20);
        let _ = m.take_written();
        m.write_u32(0x1_2344, 7);
        assert_eq!(m.read_u32(0x1_2344), 7);
        match m.take_written() {
            Some(w) => {
                assert!(
                    w.iter().any(|&(a, b)| a <= 0x1_2344 && 0x1_2348 <= b),
                    "{w:?}"
                );
                assert_eq!(m.take_written(), Some(vec![]));
            }
            None => assert!(!m.watched()),
        }
    }

    /// A query of part of the memory lists the writes there, at offsets
    /// from the start, and leaves those elsewhere for a later query.
    #[test]
    fn written_in_a_range_lists_that_range_alone() {
        let m = GuestMemory::new_watched(1 << 20);
        if !m.watched() {
            return;
        }
        let _ = m.take_written();
        m.write_u32(0x1000, 1);
        m.write_u32(0x8_0000, 2);
        let inside = m.take_written_in(0x4_0000, 0xC_0000).unwrap();
        // The OS page: 4 KB, or more on some Linux and Android machines.
        let pg = inside[0].1 - inside[0].0;
        assert_eq!(inside, [(0x8_0000, 0x8_0000 + pg)]);
        assert_eq!(m.take_written_in(0x4_0000, 0xC_0000), Some(vec![]));
        // The write outside the range is still listed.
        let at = 0x1000 / pg * pg;
        assert_eq!(m.take_written(), Some(vec![(at, at + pg)]));
        assert_eq!(m.take_written_in(0x4_0000, 0x4_0000), Some(vec![]));
    }

    #[test]
    fn big_endian_round_trip() {
        let m = GuestMemory::new(64);
        m.write_u32(8, 0x1234_5678);
        assert_eq!(m.read_u8(8), 0x12);
        assert_eq!(m.read_u16(10), 0x5678);
        assert_eq!(m.read_u32(8), 0x1234_5678);
        m.write_u64(16, 0x0102_0304_0506_0708);
        assert_eq!(m.read_u64(16), 0x0102_0304_0506_0708);
        assert_eq!(m.read_u8(23), 0x08);
    }

    #[test]
    fn unaligned_accesses_are_big_endian_too() {
        let m = GuestMemory::new(64);
        m.write_u32(9, 0xA1B2_C3D4);
        assert_eq!(m.read_u8(9), 0xA1);
        assert_eq!(m.read_u32(9), 0xA1B2_C3D4);
        m.write_u64(3, 0x0102_0304_0506_0708);
        assert_eq!(m.read_u64(3), 0x0102_0304_0506_0708);
        assert_eq!(m.read_u16(5), 0x0304);
    }

    #[test]
    fn cas_checks_the_old_value() {
        let m = GuestMemory::new(64);
        m.write_u32(4, 7);
        assert!(!m.cas_u32(4, 6, 1));
        assert!(m.cas_u32(4, 7, 1));
        assert_eq!(m.read_u32(4), 1);
        assert!(m.cas_u64(8, 0, u64::MAX));
        assert_eq!(m.read_u64(8), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "unaligned 4-byte reserved store at 0x00000002")]
    fn unaligned_reserved_store_stops() {
        GuestMemory::new(64).cas_u32(2, 0, 0);
    }

    #[test]
    #[should_panic(expected = "at 0x0000003e is outside the 64-byte guest memory")]
    fn out_of_range_reports_the_address() {
        GuestMemory::new(64).read_u32(62);
    }

    #[test]
    fn unaligned_access_wraps_at_the_end_of_a_view() {
        let m = GuestMemory::new(0xC000_0000);
        // 0xFFFFEFFE is the last physical bytes; the next ones wrap to 0.
        m.write_u32(0xFFFF_EFFE, 0x1122_3344);
        assert_eq!(m.read_u16(0xBFFF_FFFE), 0x1122);
        assert_eq!(m.read_u16(0xA000_0000), 0x3344);
        assert_eq!(m.read_u32(0xFFFF_EFFE), 0x1122_3344);
        m.write_bytes(0xFFFF_EFFC, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(m.read_u16(0xA000_0000), 0x0506);
        let mut b = [0; 6];
        m.read_bytes(0xFFFF_EFFC, &mut b);
        assert_eq!(b, [1, 2, 3, 4, 5, 6]);
        // A longer odd range over the wrap, written through the folded view
        // and read back through the first.
        let fill: Vec<u8> = (1..=45).collect();
        m.write_bytes(0xFFFF_EFE3, &fill);
        let mut b = [0; 45];
        m.read_bytes(0xBFFF_FFE3, &mut b[..29]);
        m.read_bytes(0xA000_0000, &mut b[29..]);
        assert_eq!(b[..], fill[..]);
        m.read_bytes(0xFFFF_EFE3, &mut b);
        assert_eq!(b[..], fill[..]);
    }

    /// Byte ranges of every alignment and length up to a few words copy
    /// the same bytes as single-byte accesses, in and out.
    #[test]
    fn byte_ranges_match_single_bytes() {
        let m = GuestMemory::new(128);
        let fill: Vec<u8> = (0..128u32).map(|i| (i * 37 + 11) as u8).collect();
        for addr in 0..24u32 {
            for len in 0..40usize {
                m.write_bytes(0, &[0; 128]);
                m.write_bytes(addr, &fill[..len]);
                for i in 0..128u32 {
                    let want = if (addr..addr + len as u32).contains(&i) {
                        fill[(i - addr) as usize]
                    } else {
                        0
                    };
                    assert_eq!(m.read_u8(i), want, "write at {addr} len {len} byte {i}");
                }
                for i in 0..128u32 {
                    m.write_u8(i, fill[i as usize]);
                }
                let mut out = vec![0xEE; len];
                m.read_bytes(addr, &mut out);
                assert_eq!(
                    out,
                    fill[addr as usize..][..len],
                    "read at {addr} len {len}"
                );
            }
        }
    }

    #[test]
    fn physical_views_alias() {
        assert_eq!(alias(0x8200_0000), 0x8200_0000);
        assert_eq!(alias(0xA123_4560), 0xA123_4560);
        assert_eq!(alias(0xC123_4560), 0xA123_4560);
        assert_eq!(alias(0xE123_3560), 0xA123_4560);
        assert_eq!(alias(0xFFFF_F000), 0xA000_0000);
    }
}
