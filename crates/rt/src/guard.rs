//! Guarded guest memory (opt-in): the whole guest range is reserved with no
//! access and pages are committed only when the kernel hands them out
//! ([`commit`]). A wild pointer then faults on the host instead of reading
//! zeros, and the handler set with [`on_wild_access`] gets the guest address.
//!
//! One guarded memory per process. Pages are 4 KB; the bitmap of committed
//! pages makes [`commit`] cheap when called again for the same range.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::OnceLock;

const PAGE: usize = 0x1000;

static BASE: AtomicUsize = AtomicUsize::new(0);
static LEN: AtomicUsize = AtomicUsize::new(0);
/// One bit per page of the 4 GB range.
static BITS: [AtomicU64; (1 << 20) / 64] = [const { AtomicU64::new(0) }; (1 << 20) / 64];
static HOOK: OnceLock<fn(u32) -> !> = OnceLock::new();
static IN_FAULT: AtomicBool = AtomicBool::new(false);

/// Reserves `len` bytes with no access; the base pointer.
pub(crate) fn reserve(len: usize) -> *mut u8 {
    assert!(len <= 1 << 32, "guest memory is at most 4 GB");
    let p = os::reserve(len);
    assert!(
        !p.is_null(),
        "cannot reserve {len} bytes of guarded guest memory"
    );
    if BASE
        .compare_exchange(0, p as usize, Relaxed, Relaxed)
        .is_err()
    {
        os::release(p, len);
        panic!("only one guarded guest memory per process");
    }
    LEN.store(len, Relaxed);
    p
}

pub(crate) fn release(p: *mut u8, len: usize) {
    os::release(p, len);
    for b in &BITS {
        b.store(0, Relaxed);
    }
    LEN.store(0, Relaxed);
    BASE.store(0, Relaxed);
}

/// Whether a guarded guest memory exists in this process.
pub fn active() -> bool {
    BASE.load(Relaxed) != 0
}

/// Makes `[addr, addr + len)` readable and writable (zero on first use).
/// Does nothing without a guarded memory. Addresses in the mirrored
/// physical views commit the pages they fold onto.
pub fn commit(addr: u32, len: u32) {
    let base = BASE.load(Relaxed);
    if base == 0 || len == 0 {
        return;
    }
    let first = addr as usize / PAGE;
    let last = (addr as usize + len as usize - 1).min(u32::MAX as usize) / PAGE;
    // Runs of host pages not committed yet, committed with one call each.
    let mut run: Option<(usize, usize)> = None;
    let flush = |r: Option<(usize, usize)>| {
        if let Some((s, n)) = r {
            assert!(
                os::commit((base + s * PAGE) as *mut u8, n * PAGE),
                "cannot commit guest memory at {:#010x}",
                s * PAGE
            );
            for p in s..s + n {
                BITS[p / 64].fetch_or(1 << (p % 64), Relaxed);
            }
        }
    };
    for g in first..=last {
        let h = crate::mem::alias((g * PAGE) as u32) as usize / PAGE;
        if (h + 1) * PAGE > LEN.load(Relaxed) || is_page_committed(h) {
            flush(run.take());
            continue;
        }
        run = match run {
            Some((s, n)) if s + n == h => Some((s, n + 1)),
            r => {
                flush(r);
                Some((h, 1))
            }
        };
    }
    flush(run);
}

fn is_page_committed(h: usize) -> bool {
    BITS[h / 64].load(Relaxed) & (1 << (h % 64)) != 0
}

/// Whether reading `n` bytes at `addr` is safe: always without a guarded
/// memory, else only when every page is committed. For reports that walk
/// guest data they cannot trust (a stack pointer from a stalled thread).
pub fn readable(addr: u32, n: u32) -> bool {
    if !active() {
        return true;
    }
    let end = addr as u64 + n.max(1) as u64 - 1;
    if end > u32::MAX as u64 {
        return false;
    }
    (addr as usize / PAGE..=end as usize / PAGE).all(|g| {
        let h = crate::mem::alias((g * PAGE) as u32) as usize / PAGE;
        (h + 1) * PAGE <= LEN.load(Relaxed) && is_page_committed(h)
    })
}

/// The guest address of host address `host`, if it lies in the guarded memory.
pub fn guest_address(host: usize) -> Option<u32> {
    let base = BASE.load(Relaxed);
    (base != 0 && host >= base && host - base < LEN.load(Relaxed)).then(|| (host - base) as u32)
}

/// Calls `f` with the guest address when guest code (or the host) touches
/// guarded guest memory that was never committed. `f` must not return.
/// Other faults go to whatever handled them before. Set once per process.
///
/// shortcut: `f` runs inside the fault handler, so it may deadlock if the
/// faulting thread holds a lock `f` takes; fine for a stop-and-report hook.
pub fn on_wild_access(f: fn(u32) -> !) {
    if HOOK.set(f).is_ok() {
        os::install();
    }
}

/// Entered from the OS fault handler with the faulting host address.
fn fault(host: usize) -> bool {
    let Some(g) = guest_address(host) else {
        return false;
    };
    let Some(f) = HOOK.get() else {
        return false;
    };
    if IN_FAULT.swap(true, Relaxed) {
        // Another thread is already reporting and will end the process.
        loop {
            std::thread::park();
        }
    }
    f(g)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod os {
    use std::sync::OnceLock;

    pub fn reserve(len: usize) -> *mut u8 {
        // SAFETY: a fresh anonymous mapping, no existing memory involved.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p.cast()
        }
    }

    pub fn release(p: *mut u8, len: usize) {
        // SAFETY: p and len are the mapping made by `reserve`.
        unsafe { libc::munmap(p.cast(), len) };
    }

    pub fn commit(p: *mut u8, n: usize) -> bool {
        // SAFETY: inside the reserved mapping (callers check the length).
        unsafe { libc::mprotect(p.cast(), n, libc::PROT_READ | libc::PROT_WRITE) == 0 }
    }

    static PREV: OnceLock<[libc::sigaction; 2]> = OnceLock::new();

    extern "C" fn handler(sig: i32, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
        // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
        let addr = unsafe { (*info).si_addr() } as usize;
        if super::fault(addr) {
            return;
        }
        // Not ours: put the previous handler back; the access faults again into it.
        if let Some(prev) = PREV.get() {
            let i = (sig == libc::SIGBUS) as usize;
            // SAFETY: restoring an action the OS gave us.
            unsafe { libc::sigaction(sig, &prev[i], std::ptr::null_mut()) };
        }
    }

    pub fn install() {
        // SAFETY: plain sigaction calls with zeroed, then filled, structs.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            let mut prev: [libc::sigaction; 2] = std::mem::zeroed();
            libc::sigaction(libc::SIGSEGV, &sa, &mut prev[0]);
            libc::sigaction(libc::SIGBUS, &sa, &mut prev[1]);
            let _ = PREV.set(prev);
        }
    }
}

#[cfg(windows)]
mod os {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS,
        PAGE_READWRITE,
    };

    pub fn reserve(len: usize) -> *mut u8 {
        // SAFETY: a fresh reservation, no existing memory involved.
        unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS).cast() }
    }

    pub fn release(p: *mut u8, _len: usize) {
        // SAFETY: p is the reservation made by `reserve`.
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }

    pub fn commit(p: *mut u8, n: usize) -> bool {
        // SAFETY: inside the reservation (callers check the length).
        !unsafe { VirtualAlloc(p.cast(), n, MEM_COMMIT, PAGE_READWRITE) }.is_null()
    }

    unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the OS passes valid exception pointers.
        let rec = unsafe { &*(*info).ExceptionRecord };
        let code: NTSTATUS = rec.ExceptionCode;
        if code == EXCEPTION_ACCESS_VIOLATION && rec.NumberParameters >= 2 {
            super::fault(rec.ExceptionInformation[1]);
        }
        0 // EXCEPTION_CONTINUE_SEARCH
    }

    pub fn install() {
        // SAFETY: registers a handler that only reads the exception record.
        unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
mod os {
    pub fn reserve(_: usize) -> *mut u8 {
        panic!("guarded guest memory is only built for Linux and Windows")
    }
    pub fn release(_: *mut u8, _: usize) {}
    pub fn commit(_: *mut u8, _: usize) -> bool {
        false
    }
    pub fn install() {}
}
