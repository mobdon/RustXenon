//! Guest memory whose writes the OS records page by page (Windows
//! `MEM_WRITE_WATCH`), so the drawer can skip re-reading guest data and
//! textures that nothing wrote since the last frame. Stores cost nothing
//! extra: the OS sets the bits. Linux and Android build the same from page
//! protection and a fault handler (the first write to a page after a
//! `take_written` costs a fault, and some pages are listed as written
//! without one). Elsewhere it is not available and callers fall back to
//! reading everything.

/// Allocates `len` bytes of zeroed, writable memory whose writes are
/// watched; null where that is not available.
pub(crate) fn allocate(len: usize) -> *mut u8 {
    os::allocate(len)
}

pub(crate) fn release(p: *mut u8) {
    os::release(p)
}

/// Offsets from `p` of the pages written since the last call, sorted, and
/// the page size, with the record reset; `None` when there were too many
/// to list (then all of `len` counts as written). On Linux and Android it
/// also lists pages it cannot watch yet (see `os` there), all of them on
/// the first call.
pub(crate) fn take_written(p: *mut u8, len: usize) -> Option<(Vec<usize>, usize)> {
    os::take_written(p, len)
}

#[cfg(windows)]
mod os {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, ResetWriteWatch, VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE,
        MEM_RESERVE, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemServices::{MEM_WRITE_WATCH, WRITE_WATCH_FLAG_RESET};

    /// Pages listed per call; past this everything counts as written.
    const MAX_PAGES: usize = 1 << 16;

    pub fn allocate(len: usize) -> *mut u8 {
        // SAFETY: a fresh allocation, no existing memory involved.
        unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast()
    }

    pub fn release(p: *mut u8) {
        // SAFETY: p is the allocation made by `allocate`.
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }

    pub fn take_written(p: *mut u8, len: usize) -> Option<(Vec<usize>, usize)> {
        // Kept per thread: at 512 KiB the heap hands a new one to the OS
        // and back on every call, a frame's worth of page faults each time.
        thread_local! {
            static ADDRS: std::cell::RefCell<Vec<*mut core::ffi::c_void>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        ADDRS.with_borrow_mut(|addrs| take_written_into(addrs, p, len))
    }

    fn take_written_into(
        addrs: &mut Vec<*mut core::ffi::c_void>,
        p: *mut u8,
        len: usize,
    ) -> Option<(Vec<usize>, usize)> {
        addrs.resize(MAX_PAGES, std::ptr::null_mut());
        let mut count = addrs.len();
        let mut page = 0u32;
        // SAFETY: p..p+len is the watched allocation; `addrs` holds `count`.
        let r = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                p.cast(),
                len,
                addrs.as_mut_ptr(),
                &mut count,
                &mut page,
            )
        };
        if r != 0 || count >= MAX_PAGES {
            // SAFETY: as above.
            unsafe { ResetWriteWatch(p.cast(), len) };
            return None;
        }
        let base = p as usize;
        let mut pages: Vec<usize> = addrs[..count].iter().map(|&a| a as usize - base).collect();
        pages.sort_unstable();
        Some((pages, page as usize))
    }
}

/// Linux and Android have no write watch, so it is built from page
/// protection: a page `take_written` has covered is read-only, the first
/// write to it faults, and the SIGSEGV handler records the page in a bitmap
/// and makes it writable so the store runs again. `take_written` clears the
/// bits it collects and makes those pages read-only again ("arms" them).
///
/// Some pages are left writable ("open") and listed at every take instead,
/// since nothing would see their writes:
/// - every page until a take first covers it, so memory nothing takes
///   (the guest's virtual range, say) never faults or splits the mapping;
/// - hot pages, written again right after the take that armed them. Takes
///   run once per ring span, many times a frame, and a page the guest
///   writes all frame would otherwise fault again after each one. Listing
///   it costs the caller a re-read of one page; a fault and the two
///   `mprotect` calls cost more.
///
/// Open pages are armed again by the first take [`REARM`] after the last
/// time that happened.
///
/// Only stores from user space are seen. A kernel write into a read-only
/// page (`read(2)` straight into guest memory, say) fails with `EFAULT`
/// instead of faulting, so guest memory is only ever written through
/// [`crate::GuestMemory`]'s own stores, never handed to a syscall.
///
/// Cost on a Snapdragon 8 Gen 3 phone (Android 17, 4 KB pages), 512 MB
/// watched with a quarter in use, 3000 scattered pages written per round:
/// 2.4 to 8 us for the first write to a page after `take_written` (a plain
/// store is 4 to 10 ns), 4 to 20 us for the very first touch (the kernel
/// zero fills it too), and 2.8 to 13 ms for `take_written` itself, nearly
/// all of it the 3000 `mprotect` calls; 4 to 13 us when nothing was written.
/// The spread is between runs (clocks and load), not within one (crate test
/// `cost`, run with `--ignored`). These were measured before hot pages were
/// left open; the cost of a span that rewrites hot pages (the last line of
/// `cost`) is not measured on the phone yet.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod os {
    use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
    use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize};
    use std::sync::{Mutex, Once, OnceLock};
    use std::time::{Duration, Instant};

    /// Pages listed per call; past this everything counts as written.
    const MAX_PAGES: usize = 1 << 16;

    /// How long open pages stay open before a take arms them again.
    pub(super) const REARM: Duration = Duration::from_millis(100);

    /// One watched mapping and its bitmaps, a bit per page (`dirty` and
    /// `opening` are per 64-page word). Never freed: `release` only unhooks
    /// it, so a handler or take that found it just before reads stale
    /// bits, never freed memory.
    struct Map {
        base: usize,
        len: usize,
        /// Written since the last take; set by the handler.
        dirty: Box<[AtomicU64]>,
        /// Handlers between opening a page of this word and setting its
        /// dirty bit; a take lists the whole word meanwhile.
        opening: Box<[AtomicU32]>,
        /// Handlers opening the whole mapping (out of mappings to split);
        /// a take lists everything meanwhile.
        opening_all: AtomicU32,
        /// Writable without a fault, so listed at every take. Takes only.
        open: Box<[AtomicU64]>,
        /// Armed by the last take covering them because they were written,
        /// so a write by the next is a hot page. Takes only.
        armed: Box<[AtomicU64]>,
        /// Serialises takes; when open pages were last armed.
        take: Mutex<Option<Instant>>,
    }

    /// A few at once, so tests running side by side each get their own.
    static MAPS: [AtomicPtr<Map>; 8] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 8];
    /// Serialises `allocate` and `release`; never taken in the handler.
    static LOCK: Mutex<()> = Mutex::new(());
    static PAGE: AtomicUsize = AtomicUsize::new(0);
    static INSTALL: Once = Once::new();
    static PREV: OnceLock<libc::sigaction> = OnceLock::new();

    fn words(len: usize, page: usize) -> usize {
        len.div_ceil(page).div_ceil(64)
    }

    /// The watched mapping holding `addr`.
    fn find(addr: usize) -> Option<&'static Map> {
        MAPS.iter()
            .map(|m| m.load(Acquire))
            .filter(|m| !m.is_null())
            // SAFETY: published by `allocate`, never freed.
            .map(|m| unsafe { &*m })
            .find(|m| addr >= m.base && addr - m.base < m.len)
    }

    fn protect(p: usize, n: usize, write: bool) -> bool {
        let prot = libc::PROT_READ | if write { libc::PROT_WRITE } else { 0 };
        // SAFETY: callers pass page ranges inside a watched mapping.
        unsafe { libc::mprotect(p as *mut libc::c_void, n, prot) == 0 }
    }

    pub fn allocate(len: usize) -> *mut u8 {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        PAGE.store(page, Relaxed);
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slot) = MAPS.iter().find(|m| m.load(Relaxed).is_null()) else {
            return std::ptr::null_mut();
        };
        // SAFETY: a fresh anonymous mapping, no existing memory involved.
        // Writable: pages are armed by the first take that covers them.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        let n = words(len, page);
        let bits = |v: u64| -> Box<[AtomicU64]> { (0..n).map(|_| AtomicU64::new(v)).collect() };
        let map = Box::new(Map {
            base: p as usize,
            len,
            dirty: bits(0),
            opening: (0..n).map(|_| AtomicU32::new(0)).collect(),
            opening_all: AtomicU32::new(0),
            open: bits(!0),
            armed: bits(0),
            take: Mutex::new(None),
        });
        INSTALL.call_once(install);
        slot.store(Box::into_raw(map), Release);
        p.cast()
    }

    pub fn release(p: *mut u8) {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let found = MAPS.iter().find(|m| {
            let m = m.load(Relaxed);
            // SAFETY: published by `allocate`, never freed.
            !m.is_null() && unsafe { (*m).base } == p as usize
        });
        let Some(slot) = found else { return };
        let map = slot.swap(std::ptr::null_mut(), Relaxed);
        // SAFETY: p and len are the mapping made by `allocate`. The Map is
        // leaked, not freed: see `Map`.
        unsafe { libc::munmap(p.cast(), (*map).len) };
    }

    /// From the handler: a fault at `addr` by the instruction at `pc`.
    /// Pages here are always readable, so a fault is a write, unless the
    /// instruction itself is here (a jump into guest memory): that is no
    /// write, opening the page would not get past it, so it is not ours.
    fn fault(addr: usize, pc: usize) -> bool {
        let Some(m) = find(addr) else { return false };
        if pc.wrapping_sub(m.base) < m.len {
            return false;
        }
        let page = PAGE.load(Relaxed);
        let i = (addr - m.base) / page;
        let (w, b) = (i / 64, 1u64 << (i % 64));
        // Between opening the page and setting its bit, another thread can
        // write it without a fault; `opening` makes a take list it then.
        m.opening[w].fetch_add(1, SeqCst);
        let ok = if protect(m.base + i * page, page, true) {
            m.dirty[w].fetch_or(b, SeqCst);
            true
        } else {
            // Out of mappings to split (vm.max_map_count): open it all and
            // count every page as written; one mapping again.
            m.opening_all.fetch_add(1, SeqCst);
            let ok = protect(m.base, m.len, true);
            if ok {
                for d in &m.dirty {
                    d.store(!0, SeqCst);
                }
            }
            m.opening_all.fetch_sub(1, SeqCst);
            ok
        };
        m.opening[w].fetch_sub(1, SeqCst);
        ok
    }

    /// Arms pages `start..start + n` of `m`; where it cannot (out of
    /// mappings) they stay open, listed until a later take arms them.
    fn arm(m: &Map, page: usize, start: usize, n: usize) {
        if n == 0 || protect(m.base + start * page, n * page, false) {
            return;
        }
        for i in start..start + n {
            m.open[i / 64].fetch_or(1 << (i % 64), Relaxed);
            m.armed[i / 64].fetch_and(!(1 << (i % 64)), Relaxed);
        }
    }

    pub fn take_written(p: *mut u8, len: usize) -> Option<(Vec<usize>, usize)> {
        let page = PAGE.load(Relaxed);
        let p = p as usize;
        let m = find(p)?;
        let mut last = m.take.lock().unwrap_or_else(|e| e.into_inner());
        let due = last.is_none_or(|t| t.elapsed() >= REARM);
        if due {
            *last = Some(Instant::now());
        }
        let off = p - m.base;
        debug_assert_eq!(off % page, 0, "take_written from a page boundary");
        let (first, end) = (off / page, (off + len).min(m.len).div_ceil(page));
        let mut all = m.opening_all.load(SeqCst) != 0;
        let mut pages = Vec::new();
        // The run of pages to arm, as (first page, count).
        let mut run = (0, 0);
        for w in first / 64..end.div_ceil(64) {
            let lo = (w * 64).max(first);
            let hi = (w * 64 + 64).min(end);
            let mask = (!0u64 >> (64 - (hi - lo))) << (lo - w * 64);
            // `opening` before `dirty`: a handler that finished in between
            // set its bit first.
            let opening = if m.opening[w].load(SeqCst) != 0 {
                mask
            } else {
                0
            };
            let dirty = if m.dirty[w].load(Acquire) & mask != 0 {
                m.dirty[w].fetch_and(!mask, SeqCst) & mask
            } else {
                0
            };
            let open = m.open[w].load(Relaxed) & mask;
            let hot = dirty & m.armed[w].load(Relaxed);
            let keep = if due { hot } else { open | hot };
            let to_arm = (dirty | open) & !keep;
            // Marked before they are armed: a write in between lands on a
            // page still writable, listed here, before the caller reads.
            m.open[w].store((m.open[w].load(Relaxed) & !mask) | keep, Relaxed);
            m.armed[w].store(
                (m.armed[w].load(Relaxed) & !mask) | (dirty & !keep),
                Relaxed,
            );
            let mut listed = dirty | open | opening;
            while listed != 0 && !all {
                pages.push((w * 64 + listed.trailing_zeros() as usize) * page - off);
                listed &= listed - 1;
                all = pages.len() >= MAX_PAGES;
            }
            // Contiguous runs, one mprotect each (no faster as one call
            // over the whole range, measured on the phone).
            let mut a = to_arm;
            while a != 0 {
                let i = w * 64 + a.trailing_zeros() as usize;
                a &= a - 1;
                if run.0 + run.1 == i {
                    run.1 += 1;
                } else {
                    arm(m, page, run.0, run.1);
                    run = (i, 1);
                }
            }
        }
        arm(m, page, run.0, run.1);
        (!all).then_some((pages, page))
    }

    /// The faulting instruction's address; 0 where not known.
    #[cfg(target_arch = "aarch64")]
    fn pc(ctx: *mut libc::c_void) -> usize {
        // SAFETY: the kernel passes a valid ucontext for SA_SIGINFO handlers.
        unsafe { (*ctx.cast::<libc::ucontext_t>()).uc_mcontext.pc as usize }
    }

    #[cfg(target_arch = "x86_64")]
    fn pc(ctx: *mut libc::c_void) -> usize {
        // SAFETY: the kernel passes a valid ucontext for SA_SIGINFO handlers.
        unsafe {
            (*ctx.cast::<libc::ucontext_t>()).uc_mcontext.gregs[libc::REG_RIP as usize] as usize
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn pc(_: *mut libc::c_void) -> usize {
        0
    }

    extern "C" fn handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
        let addr = unsafe { (*info).si_addr() } as usize;
        if fault(addr, pc(ctx)) {
            return;
        }
        // Not ours: hand it on to whatever handled it before (guard's
        // handler, Rust's stack-overflow check, or the default).
        let Some(prev) = PREV.get() else { return };
        let h = prev.sa_sigaction;
        // SAFETY: calling or restoring an action the OS gave us, with the
        // arguments the kernel gave us.
        unsafe {
            if h == libc::SIG_DFL || h == libc::SIG_IGN {
                // Put it back; the access faults again into it.
                libc::sigaction(sig, prev, std::ptr::null_mut());
            } else if prev.sa_flags & libc::SA_SIGINFO != 0 {
                let f: extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) =
                    std::mem::transmute(h);
                f(sig, info, ctx);
            } else {
                let f: extern "C" fn(i32) = std::mem::transmute(h);
                f(sig);
            }
        }
    }

    fn install() {
        // SAFETY: plain sigaction calls with zeroed, then filled, structs.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
            let mut prev: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGSEGV, &sa, &mut prev);
            let _ = PREV.set(prev);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
mod os {
    pub fn allocate(_: usize) -> *mut u8 {
        std::ptr::null_mut()
    }
    pub fn release(_: *mut u8) {}
    pub fn take_written(_: *mut u8, _: usize) -> Option<(Vec<usize>, usize)> {
        None
    }
}

/// The Linux and Android watch, through the crate calls. Each test has its
/// own region.
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
mod tests {
    use super::os::REARM;
    use super::{allocate, release, take_written};
    use std::collections::HashSet;
    use std::sync::atomic::Ordering::{Relaxed, SeqCst};
    use std::sync::atomic::{AtomicBool, AtomicU32};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    fn page() -> usize {
        // SAFETY: sysconf has no preconditions.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    fn store(p: *mut u8, off: usize, v: u32) {
        // SAFETY: callers pass 4-byte aligned offsets inside the region.
        unsafe { AtomicU32::from_ptr(p.add(off).cast()) }.store(v, Relaxed)
    }

    fn load(p: *mut u8, off: usize) -> u32 {
        // SAFETY: as in `store`.
        unsafe { AtomicU32::from_ptr(p.add(off).cast()) }.load(Relaxed)
    }

    fn written(p: *mut u8, len: usize) -> Vec<usize> {
        let (pages, size) = take_written(p, len).expect("listed");
        assert_eq!(size, page());
        pages
    }

    /// Distinct pages out of `n`, in a scattered order.
    fn scattered(n: usize, count: usize, seed: u32) -> Vec<usize> {
        let mut x = seed;
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        while out.len() < count {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if seen.insert(x as usize % n) {
                out.push(x as usize % n);
            }
        }
        out
    }

    #[test]
    fn first_write_is_listed_once_and_rearmed() {
        let pg = page();
        let len = 256 * pg;
        let p = allocate(len);
        assert!(!p.is_null());
        assert_eq!(written(p, len).len(), 256, "all written until first taken");
        assert!(written(p, len).is_empty(), "nothing written since");
        assert_eq!(load(p, 5 * pg), 0, "zeroed, and reading is not writing");
        store(p, 3 * pg, 1);
        store(p, 3 * pg + 8, 2);
        store(p, 7 * pg + 4, 3);
        store(p, 8 * pg, 4);
        assert_eq!(written(p, len), [3 * pg, 7 * pg, 8 * pg]);
        assert!(written(p, len).is_empty(), "listed once");
        assert_eq!(load(p, 3 * pg + 8), 2);
        // Read-only again: the next write is seen.
        store(p, 3 * pg, 5);
        assert_eq!(written(p, len), [3 * pg]);
        // Part of the region lists its own pages and leaves the rest.
        store(p, pg, 1);
        store(p, 200 * pg, 1);
        // SAFETY: inside the region.
        let half = unsafe { p.add(128 * pg) };
        assert_eq!(take_written(half, 128 * pg).unwrap().0, [72 * pg]);
        assert_eq!(written(p, len), [pg]);
        release(p);
    }

    /// A page written again right after the take that armed it stays
    /// writable and is listed at every take, until one `REARM` later arms
    /// it again.
    #[test]
    fn hot_pages_stay_open_until_rearmed() {
        let pg = page();
        let len = 16 * pg;
        let p = allocate(len);
        assert!(!p.is_null());
        let _ = written(p, len);
        store(p, 5 * pg, 1);
        assert_eq!(written(p, len), [5 * pg]);
        store(p, 5 * pg, 2);
        assert_eq!(written(p, len), [5 * pg], "hot");
        assert_eq!(written(p, len), [5 * pg], "open, so listed");
        store(p, 9 * pg, 3);
        assert_eq!(written(p, len), [5 * pg, 9 * pg]);
        std::thread::sleep(REARM);
        assert_eq!(written(p, len), [5 * pg], "listed as it is armed");
        assert!(written(p, len).is_empty());
        store(p, 5 * pg, 4);
        assert_eq!(written(p, len), [5 * pg]);
        assert!(written(p, len).is_empty(), "not hot again");
        assert_eq!(load(p, 5 * pg), 4);
        release(p);
    }

    #[test]
    fn released_memory_comes_back_fresh() {
        let pg = page();
        let a = allocate(16 * pg);
        let b = allocate(16 * pg);
        assert!(!a.is_null() && !b.is_null() && a != b);
        let _ = (written(a, 16 * pg), written(b, 16 * pg));
        store(a, 0, 1);
        store(b, pg, 1);
        assert_eq!(written(a, 16 * pg), [0]);
        assert_eq!(written(b, 16 * pg), [pg]);
        release(a);
        for _ in 0..3 {
            let c = allocate(16 * pg);
            assert!(!c.is_null());
            assert_eq!(
                written(c, 16 * pg).len(),
                16,
                "all written until first taken"
            );
            assert_eq!(load(c, 0), 0);
            store(c, 2 * pg, 9);
            assert_eq!(written(c, 16 * pg), [2 * pg]);
            release(c);
        }
        release(b);
    }

    /// Threads write pages at random while `take_written` runs. A write
    /// that began after take `d` returned is listed by a later take; none
    /// is missed by the take after it ended. Nothing is listed that was
    /// never written, but for a take that runs while a fault is being
    /// handled: it lists the faulting page's 64-page group.
    #[test]
    fn concurrent_writes_are_never_lost() {
        const THREADS: usize = 8;
        const ROUNDS: u32 = 600;
        let pg = page();
        let n = 8192;
        let len = n * pg;
        let p = allocate(len) as usize;
        assert_ne!(p, 0);
        let _ = written(p as *mut u8, len);
        let finished = AtomicU32::new(0);
        let stop = AtomicBool::new(false);
        let logs: Vec<Mutex<Vec<(usize, u32)>>> =
            (0..THREADS).map(|_| Mutex::new(Vec::new())).collect();
        let mut reported = vec![0u32; n];
        let mut checked = [0usize; THREADS];
        // Every write logged before `ends` was taken must have been listed
        // in a round after the last one finished before it began.
        let mut check = |reported: &[u32], ends: &[usize]| {
            for ((log, done), &end) in logs.iter().zip(&mut checked).zip(ends) {
                for &(page, d) in &log.lock().unwrap()[*done..end] {
                    assert!(
                        reported[page] > d,
                        "page {page} written after round {d}, last listed in {}",
                        reported[page]
                    );
                }
                *done = end;
            }
        };
        std::thread::scope(|s| {
            for (t, log) in logs.iter().enumerate() {
                let (finished, stop) = (&finished, &stop);
                s.spawn(move || {
                    let mut x = 0x9E37_79B9u32.wrapping_mul(t as u32 + 1);
                    let mut i = 0;
                    while !stop.load(Relaxed) {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let page = x as usize % n;
                        let d = finished.load(SeqCst);
                        store(p as *mut u8, page * pg + 4 * t, i);
                        log.lock().unwrap().push((page, d));
                        i += 1;
                    }
                });
            }
            for r in 1..=ROUNDS {
                // These writes ended before this round's take began.
                let ends: Vec<usize> = logs.iter().map(|l| l.lock().unwrap().len()).collect();
                for off in written(p as *mut u8, len) {
                    reported[off / pg] = r;
                }
                finished.store(r, SeqCst);
                check(&reported, &ends);
                std::thread::sleep(Duration::from_micros(500));
            }
            stop.store(true, Relaxed);
        });
        let ends: Vec<usize> = logs.iter().map(|l| l.lock().unwrap().len()).collect();
        for off in written(p as *mut u8, len) {
            reported[off / pg] = ROUNDS + 1;
        }
        check(&reported, &ends);
        let truth: HashSet<usize> = logs
            .iter()
            .flat_map(|l| l.lock().unwrap().iter().map(|e| e.0).collect::<Vec<_>>())
            .collect();
        let groups: HashSet<usize> = truth.iter().map(|p| p / 64).collect();
        let listed = reported.iter().filter(|&&r| r != 0).count();
        for (page, &r) in reported.iter().enumerate() {
            assert!(
                r == 0 || groups.contains(&(page / 64)),
                "page {page} listed, nothing near it written"
            );
        }
        let writes: usize = ends.iter().sum();
        eprintln!(
            "{writes} writes, {} pages written, {listed} listed",
            truth.len()
        );
        assert!(writes > 10_000, "the writers ran");
        release(p as *mut u8);
    }

    /// A fault outside every watched region goes on to the handler before
    /// (here Rust's own, then the default), so a stray write still kills the
    /// process with SIGSEGV; so does a jump into watched memory, which is
    /// readable but not executable. Runs this test binary again as a child.
    #[test]
    fn stray_accesses_still_crash() {
        use std::os::unix::process::ExitStatusExt;
        if let Some(how) = std::env::var_os("XENON_WATCH_CHILD") {
            let pg = page();
            let w = allocate(pg);
            store(w, 0, 1);
            let _ = take_written(w, pg);
            if how == "exec" {
                // SAFETY: none; it faults on purpose.
                let f: extern "C" fn() = unsafe { std::mem::transmute(w) };
                f();
                eprintln!("ran guest memory without a fault");
                std::process::exit(0);
            }
            // SAFETY: a fresh read-only mapping, written to fault on purpose.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    pg,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            store(p.cast(), 0, 1);
            eprintln!("wrote a read-only page without a fault");
            std::process::exit(0);
        }
        for how in ["write", "exec"] {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "watch::tests::stray_accesses_still_crash"])
                .env("XENON_WATCH_CHILD", how)
                .output()
                .unwrap();
            let err = String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.signal(), Some(libc::SIGSEGV), "{how}: {err}");
        }
    }

    /// The cost on this machine: `cargo test -p xenon-rt -- --ignored
    /// --nocapture watch::tests::cost`.
    #[test]
    #[ignore]
    fn cost() {
        const DIRTY: usize = 3000;
        let pg = page();
        let len = 512 << 20;
        let p = allocate(len);
        assert!(!p.is_null());
        let pages = scattered(len / pg, DIRTY, 0x1234_5678);
        // A quarter of it in use, as guest memory in a game is.
        for i in (0..len / 4).step_by(pg) {
            store(p, i + 8, 1);
        }
        let t = Instant::now();
        let _ = take_written(p, len);
        eprintln!("first take_written, arming 512 MB: {:?}", t.elapsed());
        let t = Instant::now();
        for &i in &pages {
            store(p, i * pg, 1);
        }
        let populate = t.elapsed();
        let t = Instant::now();
        assert_eq!(written(p, len).len(), DIRTY);
        eprintln!(
            "first touch (fault + kernel zero fill): {} ns a page; take_written {:?}",
            populate.as_nanos() / DIRTY as u128,
            t.elapsed()
        );
        let t = Instant::now();
        assert!(written(p, len).is_empty());
        eprintln!("take_written of 512 MB, nothing written: {:?}", t.elapsed());
        for round in 0..5 {
            // A take between rounds, so last round's pages are not hot.
            assert!(written(p, len).is_empty());
            let t = Instant::now();
            for &i in &pages {
                store(p, i * pg, round);
            }
            let faults = t.elapsed();
            let t = Instant::now();
            for &i in &pages {
                store(p, i * pg + 4, round);
            }
            let plain = t.elapsed();
            let t = Instant::now();
            assert_eq!(written(p, len).len(), DIRTY);
            let take = t.elapsed();
            eprintln!(
                "round {round}: first write {} ns a page (plain store {} ns), take_written of {DIRTY} pages {take:?}",
                faults.saturating_sub(plain).as_nanos() / DIRTY as u128,
                plain.as_nanos() / DIRTY as u128,
            );
        }
        // Ring spans: the same pages written between every take, as the
        // guest writes its command buffers and constants all frame.
        let spans = 200;
        let t = Instant::now();
        for span in 0..spans {
            for &i in &pages {
                store(p, i * pg, span);
            }
            assert_eq!(written(p, len).len(), DIRTY);
        }
        eprintln!(
            "{spans} spans rewriting {DIRTY} pages: {:?} a span, writes and take_written",
            t.elapsed() / spans
        );
        release(p);
    }
}
