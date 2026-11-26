//! Guest memory whose writes the OS records page by page (Windows
//! `MEM_WRITE_WATCH`), so the drawer can skip re-reading guest data and
//! textures that nothing wrote since the last frame. Stores cost nothing
//! extra: the OS sets the bits. Linux and Android build the same from page
//! protection and a fault handler (the first write to a page after each
//! `take_written` costs a fault). Elsewhere it is not available and callers
//! fall back to reading everything.

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
/// to list (then all of `len` counts as written).
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
/// protection: watched memory is mapped read-only, the first write to a
/// page faults, and the SIGSEGV handler records the page in a bitmap and
/// makes it writable so the store runs again. `take_written` clears the
/// bits it collects and makes those pages read-only again.
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
/// `cost`, run with `--ignored`).
#[cfg(any(target_os = "linux", target_os = "android"))]
mod os {
    use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
    use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize};
    use std::sync::{Mutex, Once, OnceLock};

    /// Pages listed per call; past this everything counts as written.
    const MAX_PAGES: usize = 1 << 16;

    /// One watched mapping and its bitmap of written pages. A few at once,
    /// so tests running side by side each get their own.
    struct Region {
        base: AtomicUsize,
        len: AtomicUsize,
        bits: AtomicPtr<AtomicU64>,
    }

    static REGIONS: [Region; 8] = [const {
        Region {
            base: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            bits: AtomicPtr::new(std::ptr::null_mut()),
        }
    }; 8];
    /// Serialises `allocate` and `release`; never taken in the handler.
    static LOCK: Mutex<()> = Mutex::new(());
    static PAGE: AtomicUsize = AtomicUsize::new(0);
    static INSTALL: Once = Once::new();
    static PREV: OnceLock<libc::sigaction> = OnceLock::new();

    fn words(len: usize, page: usize) -> usize {
        len.div_ceil(page).div_ceil(64)
    }

    impl Region {
        /// Base, length and bitmap, when this region is in use.
        fn get(&self) -> Option<(usize, usize, &'static [AtomicU64])> {
            let base = self.base.load(Acquire);
            if base == 0 {
                return None;
            }
            let len = self.len.load(Relaxed);
            let bits = self.bits.load(Relaxed);
            // SAFETY: published by `allocate` before `base`, freed by
            // `release` only after `base` is cleared.
            let bits = unsafe { std::slice::from_raw_parts(bits, words(len, PAGE.load(Relaxed))) };
            Some((base, len, bits))
        }
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
        let Some(r) = REGIONS.iter().find(|r| r.base.load(Relaxed) == 0) else {
            return std::ptr::null_mut();
        };
        // SAFETY: a fresh anonymous mapping, no existing memory involved.
        // Read-only from the start, so the first write to every page faults.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        let bits: Box<[AtomicU64]> = (0..words(len, page)).map(|_| AtomicU64::new(0)).collect();
        r.len.store(len, Relaxed);
        r.bits.store(Box::into_raw(bits).cast(), Relaxed);
        r.base.store(p as usize, Release);
        INSTALL.call_once(install);
        p.cast()
    }

    pub fn release(p: *mut u8) {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(r) = REGIONS.iter().find(|r| r.base.load(Relaxed) == p as usize) else {
            return;
        };
        let len = r.len.load(Relaxed);
        r.base.store(0, Release);
        // SAFETY: p and len are the mapping made by `allocate`.
        unsafe { libc::munmap(p.cast(), len) };
        let bits = r.bits.swap(std::ptr::null_mut(), Relaxed);
        // SAFETY: made by `Box::into_raw` in `allocate` with this length.
        drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                bits,
                words(len, PAGE.load(Relaxed)),
            ))
        });
    }

    /// From the handler: a fault at `addr`. Inside a watched region it can
    /// only be a write (the pages are always readable): open the page,
    /// THEN set its bit. The other order loses writes: `take_written`
    /// could clear the bit and re-protect between the two, and the open
    /// would leave a writable page with a clear bit.
    fn fault(addr: usize) -> bool {
        let page = PAGE.load(Relaxed);
        for r in &REGIONS {
            let Some((base, len, bits)) = r.get() else {
                continue;
            };
            if addr < base || addr - base >= len {
                continue;
            }
            let i = (addr - base) / page;
            if protect(base + i * page, page, true) {
                bits[i / 64].fetch_or(1 << (i % 64), Release);
            } else {
                // Out of mappings to split (vm.max_map_count): open it all
                // and count every page as written; one mapping again.
                if !protect(base, len, true) {
                    return false;
                }
                for w in bits {
                    w.store(!0, Release);
                }
            }
            return true;
        }
        false
    }

    pub fn take_written(p: *mut u8, len: usize) -> Option<(Vec<usize>, usize)> {
        let page = PAGE.load(Relaxed);
        let p = p as usize;
        let (base, rlen, bits) = REGIONS
            .iter()
            .filter_map(Region::get)
            .find(|&(b, l, _)| p >= b && p - b < l)?;
        let off = p - base;
        debug_assert_eq!(off % page, 0, "take_written from a page boundary");
        let (first, end) = (off / page, (off + len).min(rlen).div_ceil(page));
        // Clear the bits first, re-protect after: a write in between lands
        // on a page still writable, before the caller reads memory, so it
        // is seen; the other order would lose it.
        let mut pages = Vec::new();
        let mut many = false;
        for (w, word) in bits
            .iter()
            .enumerate()
            .take(end.div_ceil(64))
            .skip(first / 64)
        {
            let lo = (w * 64).max(first);
            let hi = (w * 64 + 64).min(end);
            let mask = (!0u64 >> (64 - (hi - lo))) << (lo - w * 64);
            if word.load(Relaxed) & mask == 0 {
                continue;
            }
            let mut got = word.fetch_and(!mask, AcqRel) & mask;
            while got != 0 && !many {
                pages.push((w * 64 + got.trailing_zeros() as usize) * page - off);
                got &= got - 1;
                many = pages.len() >= MAX_PAGES;
            }
        }
        let end_byte = (off + len).min(rlen);
        let all = || {
            assert!(
                protect(p, end_byte - off, false),
                "cannot re-protect watched memory"
            )
        };
        if many {
            all();
            return None;
        }
        // Contiguous runs, one mprotect each (no faster as one call over
        // the whole range, measured on the phone).
        let mut i = 0;
        while i < pages.len() {
            let mut j = i + 1;
            while j < pages.len() && pages[j] == pages[j - 1] + page {
                j += 1;
            }
            if !protect(p + pages[i], (j - i) * page, false) {
                // Out of mappings: the whole range read-only is one, and
                // pages with their bit still set stay listed.
                all();
                break;
            }
            i = j;
        }
        Some((pages, page))
    }

    extern "C" fn handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
        let addr = unsafe { (*info).si_addr() } as usize;
        if fault(addr) {
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
        assert!(written(p, len).is_empty(), "nothing written yet");
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

    #[test]
    fn released_memory_comes_back_fresh() {
        let pg = page();
        let a = allocate(16 * pg);
        let b = allocate(16 * pg);
        assert!(!a.is_null() && !b.is_null() && a != b);
        store(a, 0, 1);
        store(b, pg, 1);
        assert_eq!(written(a, 16 * pg), [0]);
        assert_eq!(written(b, 16 * pg), [pg]);
        release(a);
        for _ in 0..3 {
            let c = allocate(16 * pg);
            assert!(!c.is_null());
            assert!(written(c, 16 * pg).is_empty());
            assert_eq!(load(c, 0), 0);
            store(c, 2 * pg, 9);
            assert_eq!(written(c, 16 * pg), [2 * pg]);
            release(c);
        }
        release(b);
    }

    /// Threads write pages at random while `take_written` runs. A write
    /// that began after round `k` started is listed in round `k` or later,
    /// and by the first round that starts after it ended; nothing that was
    /// never written is listed.
    #[test]
    fn concurrent_writes_are_never_lost() {
        const THREADS: usize = 8;
        const ROUNDS: u32 = 300;
        let pg = page();
        let n = 8192;
        let len = n * pg;
        let p = allocate(len) as usize;
        assert_ne!(p, 0);
        let round = AtomicU32::new(0);
        let stop = AtomicBool::new(false);
        let logs: Vec<Mutex<Vec<(usize, u32)>>> =
            (0..THREADS).map(|_| Mutex::new(Vec::new())).collect();
        let mut reported = vec![0u32; n];
        let mut checked = [0usize; THREADS];
        // Every write logged before `ends` was taken must have been listed
        // in a round no earlier than the one it began in.
        let mut check = |reported: &[u32], ends: &[usize]| {
            for ((log, done), &end) in logs.iter().zip(&mut checked).zip(ends) {
                for &(page, k) in &log.lock().unwrap()[*done..end] {
                    assert!(
                        reported[page] >= k.max(1),
                        "page {page} written in round {k}, last listed in {}",
                        reported[page]
                    );
                }
                *done = end;
            }
        };
        std::thread::scope(|s| {
            for (t, log) in logs.iter().enumerate() {
                let (round, stop) = (&round, &stop);
                s.spawn(move || {
                    let mut x = 0x9E37_79B9u32.wrapping_mul(t as u32 + 1);
                    let mut i = 0;
                    while !stop.load(Relaxed) {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let page = x as usize % n;
                        let k = round.load(SeqCst);
                        store(p as *mut u8, page * pg + 4 * t, i);
                        log.lock().unwrap().push((page, k));
                        i += 1;
                    }
                });
            }
            for r in 1..=ROUNDS {
                // These writes ended before this round's take began.
                let ends: Vec<usize> = logs.iter().map(|l| l.lock().unwrap().len()).collect();
                round.store(r, SeqCst);
                for off in written(p as *mut u8, len) {
                    reported[off / pg] = r;
                }
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
        let listed = reported.iter().filter(|&&r| r != 0).count();
        for (page, &r) in reported.iter().enumerate() {
            assert!(
                r == 0 || truth.contains(&page),
                "page {page} listed, never written"
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
    /// process with SIGSEGV. Runs this test binary again as a child.
    #[test]
    fn writes_outside_still_crash() {
        use std::os::unix::process::ExitStatusExt;
        if std::env::var_os("XENON_WATCH_CHILD").is_some() {
            let pg = page();
            let w = allocate(pg);
            store(w, 0, 1);
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
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "watch::tests::writes_outside_still_crash"])
            .env("XENON_WATCH_CHILD", "1")
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.signal(), Some(libc::SIGSEGV), "{err}");
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
        let _ = take_written(p, len);
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
        release(p);
    }
}
