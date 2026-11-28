//! Waits for a monitor's vertical blank, so the host can raise the guest's
//! vertical blank on the display's own clock (Windows: the kernel display
//! driver's `D3DKMTWaitForVerticalBlankEvent`, by the monitor's GDI name).

/// A monitor's vertical blank to wait on.
pub struct Vsync {
    #[cfg(windows)]
    adapter: u32,
    #[cfg(windows)]
    source: u32,
}

#[cfg(windows)]
mod sys {
    #[repr(C)]
    pub struct OpenFromName {
        pub name: [u16; 32],
        pub adapter: u32,
        pub luid: [u32; 2],
        pub source: u32,
    }
    #[repr(C)]
    pub struct Wait {
        pub adapter: u32,
        pub device: u32,
        pub source: u32,
    }
    #[link(name = "gdi32")]
    extern "system" {
        pub fn D3DKMTOpenAdapterFromGdiDisplayName(p: *mut OpenFromName) -> i32;
        pub fn D3DKMTWaitForVerticalBlankEvent(p: *const Wait) -> i32;
        pub fn D3DKMTCloseAdapter(p: *const u32) -> i32;
    }
}

impl Vsync {
    /// The vertical blank of the monitor GDI names `name` (as
    /// `\\.\DISPLAY1`); none off Windows or when the driver will not say.
    pub fn open(name: &str) -> Option<Vsync> {
        #[cfg(windows)]
        {
            let mut p = sys::OpenFromName {
                name: [0; 32],
                adapter: 0,
                luid: [0; 2],
                source: 0,
            };
            let wide: Vec<u16> = name.encode_utf16().collect();
            if wide.len() >= p.name.len() {
                return None;
            }
            p.name[..wide.len()].copy_from_slice(&wide);
            // SAFETY: `p` is a valid D3DKMT_OPENADAPTERFROMGDIDISPLAYNAME.
            (unsafe { sys::D3DKMTOpenAdapterFromGdiDisplayName(&mut p) } == 0).then_some(Vsync {
                adapter: p.adapter,
                source: p.source,
            })
        }
        #[cfg(not(windows))]
        {
            let _ = name;
            None
        }
    }

    /// Blocks until the next vertical blank; false when the driver failed.
    pub fn wait(&self) -> bool {
        #[cfg(windows)]
        {
            let w = sys::Wait {
                adapter: self.adapter,
                device: 0,
                source: self.source,
            };
            // SAFETY: `w` names an adapter this handle opened.
            unsafe { sys::D3DKMTWaitForVerticalBlankEvent(&w) == 0 }
        }
        #[cfg(not(windows))]
        false
    }
}

impl Drop for Vsync {
    fn drop(&mut self) {
        // SAFETY: the adapter was opened by `open` and is closed once.
        #[cfg(windows)]
        unsafe {
            sys::D3DKMTCloseAdapter(&self.adapter);
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// Needs a display: passes with none attached (nothing to open).
    #[test]
    fn a_monitor_vertical_blanks_come_at_its_refresh() {
        assert!(Vsync::open("not a display").is_none());
        let Some(v) = (1..64).find_map(|n| Vsync::open(&format!(r"\\.\DISPLAY{n}"))) else {
            return;
        };
        assert!(v.wait());
        let t = std::time::Instant::now();
        for _ in 0..10 {
            assert!(v.wait());
        }
        let ms = t.elapsed().as_secs_f64() * 100.0;
        eprintln!("vertical blank every {ms:.2} ms");
        // Any refresh from 24 to 360 Hz.
        assert!((2.5..45.0).contains(&ms), "{ms}");
    }
}
