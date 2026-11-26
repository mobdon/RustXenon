//! Guest memory whose writes the OS records page by page (Windows
//! `MEM_WRITE_WATCH`), so the drawer can skip re-reading guest data and
//! textures that nothing wrote since the last frame. Stores cost nothing
//! extra: the OS sets the bits. Elsewhere it is not available and callers
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

#[cfg(not(windows))]
mod os {
    pub fn allocate(_: usize) -> *mut u8 {
        std::ptr::null_mut()
    }
    pub fn release(_: *mut u8) {}
    pub fn take_written(_: *mut u8, _: usize) -> Option<(Vec<usize>, usize)> {
        None
    }
}
