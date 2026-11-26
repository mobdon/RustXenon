//! `xenon-window`: a window that shows the latest frame and turns an XInput
//! controller (Windows) plus the keyboard into one pad reading.
//!
//! The event loop runs on its own thread so the host's main thread keeps its
//! headless watch loop. Frames are copied to the window with the CPU
//! (bilinear scaling, letterboxed, rows split over a few threads); no GPU
//! work happens here.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};
use xenon_input::{filtered, merge, Keyboard, Pad};

/// One frame to show: rows top to bottom, 8-bit RGBA.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Shared between the host and the window thread.
#[derive(Default)]
pub struct Link {
    frame: Mutex<Option<Image>>,
    pad: Mutex<Pad>,
}

impl Link {
    /// Hands the window a new frame; an older one not yet shown is dropped.
    pub fn show(&self, width: u32, height: u32, rgba8: Vec<u8>) {
        *self.frame.lock().unwrap() = Some(Image {
            width,
            height,
            rgba8,
        });
    }

    /// The controller and keyboard together, as last read.
    pub fn pad(&self) -> Pad {
        *self.pad.lock().unwrap()
    }
}

/// Gives a window the program's icon. On Windows the title bar and taskbar
/// load the exe's icon resource at their own sizes, so the small one is the
/// image drawn for 16 px; elsewhere, or with no resource, drawn images are used.
pub fn with_icon(a: winit::window::WindowAttributes) -> winit::window::WindowAttributes {
    use winit::window::Icon;
    let drawn = |s: u32| Icon::from_rgba(xenon_icon::rgba(s), s, s).ok();
    #[cfg(windows)]
    {
        use winit::platform::windows::{IconExtWindows, WindowAttributesExtWindows};
        let res = |s: Option<u32>| {
            Icon::from_resource(1, s.map(|s| winit::dpi::PhysicalSize::new(s, s))).ok()
        };
        // shortcut: the small icon is 16 px at any DPI; ask SM_CXSMICON if it looks soft above 100%.
        a.with_window_icon(res(Some(16)).or_else(|| drawn(16)))
            .with_taskbar_icon(res(None).or_else(|| drawn(32)))
    }
    #[cfg(not(windows))]
    a.with_window_icon(drawn(64))
}

/// Opens the window on a new thread. `on_close` runs on that thread when the
/// player closes it, or after printing why no window could be opened.
pub fn spawn(title: String, on_close: impl FnOnce() + Send + 'static) -> Arc<Link> {
    let link = Arc::new(Link::default());
    let l = link.clone();
    std::thread::Builder::new()
        .name("window".into())
        .spawn(move || {
            match event_loop() {
                Ok(ev) => {
                    if let Err(e) = ev.run_app(&mut App::new(title, l)) {
                        eprintln!("window: {e}");
                    }
                }
                Err(e) => eprintln!("window: cannot open one: {e}"),
            }
            on_close();
        })
        .expect("window thread");
    link
}

fn event_loop() -> Result<EventLoop<()>, winit::error::EventLoopError> {
    let mut b = EventLoop::builder();
    #[cfg(windows)]
    winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut b, true);
    #[cfg(all(unix, not(target_vendor = "apple")))]
    {
        winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut b, true);
        winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(&mut b, true);
    }
    b.build()
}

type Surface = softbuffer::Surface<Rc<Window>, Rc<Window>>;

struct App {
    title: String,
    link: Arc<Link>,
    keys: Keyboard,
    pad: xinput::Reader,
    window: Option<(Rc<Window>, Surface)>,
    /// Kept so a resize can redraw it.
    last: Option<Image>,
}

impl App {
    fn new(title: String, link: Arc<Link>) -> Self {
        App {
            title,
            link,
            keys: Keyboard::default(),
            pad: xinput::Reader::default(),
            window: None,
            last: None,
        }
    }

    fn draw(&mut self) {
        let (Some((w, s)), Some(img)) = (self.window.as_mut(), self.last.as_ref()) else {
            return;
        };
        let size = w.inner_size();
        let (Some(dw), Some(dh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if s.resize(dw, dh).is_err() {
            return;
        }
        let Ok(mut buf) = s.buffer_mut() else { return };
        blit(img, &mut buf, size.width, size.height);
        let _ = buf.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, ev: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = with_icon(Window::default_attributes())
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let made = ev
            .create_window(attrs)
            .map_err(|e| e.to_string())
            .and_then(|w| {
                let w = Rc::new(w);
                let cx = softbuffer::Context::new(w.clone()).map_err(|e| e.to_string())?;
                let s = Surface::new(&cx, w.clone()).map_err(|e| e.to_string())?;
                Ok((w, s))
            });
        match made {
            Ok(x) => self.window = Some(x),
            Err(e) => {
                eprintln!("window: {e}");
                ev.exit();
            }
        }
    }

    fn window_event(&mut self, ev: &ActiveEventLoop, _: WindowId, e: WindowEvent) {
        match e {
            WindowEvent::CloseRequested => ev.exit(),
            WindowEvent::Focused(false) => self.keys.release_all(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    self.keys.key(&format!("{code:?}"), down);
                }
            }
            WindowEvent::Resized(_) | WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, ev: &ActiveEventLoop) {
        let pad = merge(filtered(self.pad.read()), self.keys.pad());
        *self.link.pad.lock().unwrap() = pad;
        let fresh = self.link.frame.lock().unwrap().take();
        if let Some(img) = fresh {
            self.last = Some(img);
            self.draw();
        }
        // Read input at about 250 Hz; frames arrive far slower.
        ev.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(4),
        ));
    }
}

/// Scales `img` to fit `w` x `h` keeping its shape, black bars around it,
/// into 0RGB pixels. Bilinear: a 720p frame stretched to a large window with
/// nearest-neighbour scaling showed every frame pixel as a hard block.
fn blit(img: &Image, out: &mut [u32], w: u32, h: u32) {
    out.fill(0);
    if img.width == 0 || img.height == 0 || img.rgba8.len() < (img.width * img.height * 4) as usize
    {
        return;
    }
    // Largest size with the frame's aspect that fits.
    let (fw, fh) = if w as u64 * img.height as u64 > h as u64 * img.width as u64 {
        ((h as u64 * img.width as u64 / img.height as u64) as u32, h)
    } else {
        (w, (w as u64 * img.height as u64 / img.width as u64) as u32)
    };
    if fw == 0 || fh == 0 {
        return;
    }
    let (ox, oy) = ((w - fw) / 2, (h - fh) / 2);
    let xs = taps(img.width, fw);
    let ys = taps(img.height, fh);
    let rows = &mut out[(oy * w) as usize..][..(fh * w) as usize];
    // A 4K window is 8 million pixels a frame: rows in a few threads.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let per = (fh as usize).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in rows.chunks_mut(per * w as usize).enumerate() {
            let (xs, ys) = (&xs, &ys[i * per..]);
            scope.spawn(move || {
                for (row, &(y0, y1, wy)) in chunk.chunks_mut(w as usize).zip(ys) {
                    let row = &mut row[ox as usize..][..fw as usize];
                    let (r0, r1) = (y0 * img.width as usize, y1 * img.width as usize);
                    for (px, &(x0, x1, wx)) in row.iter_mut().zip(xs) {
                        let at = |r: usize, x: usize| &img.rgba8[(r + x) * 4..][..3];
                        let (a, b, c, d) = (at(r0, x0), at(r0, x1), at(r1, x0), at(r1, x1));
                        let mut v = 0;
                        for k in 0..3 {
                            let top = a[k] as u32 * (256 - wx) + b[k] as u32 * wx;
                            let bottom = c[k] as u32 * (256 - wx) + d[k] as u32 * wx;
                            let mix = (top * (256 - wy) + bottom * wy + (1 << 15)) >> 16;
                            v |= mix << (16 - 8 * k);
                        }
                        *px = v;
                    }
                }
            });
        }
    });
}

/// For each of `dst` output pixels across `src` source ones: the two
/// source pixels around its centre and the second's weight in 256ths.
fn taps(src: u32, dst: u32) -> Vec<(usize, usize, u32)> {
    (0..dst)
        .map(|i| {
            // Centre of output pixel i in source pixels, in 256ths, less half
            // a pixel so whole positions land on source pixel centres.
            let c = ((2 * i as u64 + 1) * src as u64 * 128 / dst as u64) as i64 - 128;
            let c = c.clamp(0, (src as i64 - 1) * 256);
            let x0 = (c >> 8) as usize;
            (x0, (x0 + 1).min(src as usize - 1), (c & 255) as u32)
        })
        .collect()
}

#[cfg(windows)]
mod xinput {
    use std::time::{Duration, Instant};
    use xenon_input::Pad;

    #[repr(C)]
    #[derive(Default)]
    struct State {
        packet: u32,
        buttons: u16,
        lt: u8,
        rt: u8,
        lx: i16,
        ly: i16,
        rx: i16,
        ry: i16,
    }

    #[link(name = "xinput")]
    extern "system" {
        fn XInputGetState(user: u32, state: *mut State) -> u32;
    }

    fn get(user: u32) -> Option<Pad> {
        let mut s = State::default();
        // SAFETY: `s` is a valid XINPUT_STATE for the call to fill.
        (unsafe { XInputGetState(user, &mut s) } == 0).then_some(Pad {
            // Bit for bit the console's layout; 0x0400 (guide) is not reported.
            buttons: s.buttons & !0x0400,
            lt: s.lt,
            rt: s.rt,
            lx: s.lx,
            ly: s.ly,
            rx: s.rx,
            ry: s.ry,
        })
    }

    /// The first connected controller. Asking an empty port is slow, so
    /// ports are searched again at most once a second.
    #[derive(Default)]
    pub struct Reader {
        port: Option<u32>,
        searched: Option<Instant>,
    }

    impl Reader {
        pub fn read(&mut self) -> Pad {
            if let Some(p) = self.port.and_then(get) {
                return p;
            }
            self.port = None;
            if self
                .searched
                .is_some_and(|t| t.elapsed() < Duration::from_secs(1))
            {
                return Pad::default();
            }
            self.searched = Some(Instant::now());
            for user in 0..4 {
                if let Some(p) = get(user) {
                    eprintln!("window: controller on XInput port {user}");
                    self.port = Some(user);
                    return p;
                }
            }
            Pad::default()
        }
    }
}

#[cfg(not(windows))]
mod xinput {
    /// shortcut: no controller off Windows (keyboard only), add one if the
    /// host is ever played on another system.
    #[derive(Default)]
    pub struct Reader(());

    impl Reader {
        pub fn read(&mut self) -> xenon_input::Pad {
            xenon_input::Pad::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_and_scale() {
        // 2x1 frame, red then blue, into a 4x4 window: a black row on
        // top, the 4x2 picture, a black row below. Bilinear: the outer
        // pixels are the frame's, the inner two a quarter and three
        // quarters of the way from red to blue.
        let img = Image {
            width: 2,
            height: 1,
            rgba8: vec![255, 0, 0, 255, 0, 0, 255, 255],
        };
        let mut out = vec![7; 16];
        blit(&img, &mut out, 4, 4);
        let (r, b) = (0xFF0000, 0x0000FF);
        let (rb, br) = (0xBF0040, 0x4000BF);
        assert_eq!(out, [0, 0, 0, 0, r, rb, br, b, r, rb, br, b, 0, 0, 0, 0]);
        // At its own size the frame is copied exactly.
        let mut out = vec![7; 2];
        blit(&img, &mut out, 2, 1);
        assert_eq!(out, [r, b]);
        // A 720p frame into a 1440p window stays well inside a frame's time
        // (checked here in release builds only).
        let big = Image {
            width: 1280,
            height: 720,
            rgba8: vec![128; 1280 * 720 * 4],
        };
        let mut out = vec![0; 2560 * 1440];
        let t = std::time::Instant::now();
        for _ in 0..10 {
            blit(&big, &mut out, 2560, 1440);
        }
        let per = t.elapsed() / 10;
        eprintln!("blit 720p to 1440p: {per:?}");
        if !cfg!(debug_assertions) {
            assert!(per < std::time::Duration::from_millis(8), "{per:?}");
        }
        // Wide window, side bars: 1x1 frame into 3x1.
        let one = Image {
            width: 1,
            height: 1,
            rgba8: vec![0, 255, 0, 255],
        };
        let mut out = vec![7; 3];
        blit(&one, &mut out, 3, 1);
        assert_eq!(out, [0, 0x00FF00, 0]);
        // A short buffer draws nothing rather than panicking.
        let bad = Image {
            width: 4,
            height: 4,
            rgba8: vec![0; 4],
        };
        blit(&bad, &mut out, 3, 1);
        assert_eq!(out, [0, 0, 0]);
    }
}
