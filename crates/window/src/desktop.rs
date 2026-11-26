//! The desktop window: winit and softbuffer on a thread of its own.

use crate::{blit, text, Config, Image, Link, Px};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Fullscreen, Window, WindowId};
use xenon_input::{filtered, merge, Keyboard};

/// Opens the window on a new thread. `on_close` runs on that thread when the
/// player closes it, or after printing why no window could be opened.
pub fn spawn(title: String, cfg: Config, on_close: impl FnOnce() + Send + 'static) -> Arc<Link> {
    let link = Arc::new(Link::default());
    let l = link.clone();
    std::thread::Builder::new()
        .name("window".into())
        .spawn(move || {
            match event_loop() {
                Ok(ev) => {
                    if let Err(e) = ev.run_app(&mut App::new(title, cfg, l)) {
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
    cfg: Config,
    link: Arc<Link>,
    keys: Keyboard,
    pad: xinput::Reader,
    window: Option<(Rc<Window>, Surface)>,
    /// Kept so a resize can redraw it.
    last: Option<Image>,
    /// Text shown instead of frames, and whether the close buttons have
    /// been seen up since (so a button held at the stop does not close it).
    text: Option<(Vec<String>, bool)>,
}

impl App {
    fn new(title: String, cfg: Config, link: Arc<Link>) -> Self {
        App {
            title,
            cfg,
            link,
            keys: Keyboard::default(),
            pad: xinput::Reader::default(),
            window: None,
            last: None,
            text: None,
        }
    }

    fn draw(&mut self) {
        let Some((w, s)) = self.window.as_mut() else {
            return;
        };
        if self.last.is_none() && self.text.is_none() {
            return;
        }
        let size = w.inner_size();
        let (Some(dw), Some(dh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if s.resize(dw, dh).is_err() {
            return;
        }
        let Ok(mut buf) = s.buffer_mut() else { return };
        match (&self.text, &self.last) {
            (Some((lines, _)), _) => text(
                lines,
                &mut buf,
                size.width,
                size.height,
                size.width as usize,
                Px::Xrgb,
            ),
            (None, Some(img)) => blit(
                img,
                &mut buf,
                size.width,
                size.height,
                size.width as usize,
                Px::Xrgb,
            ),
            (None, None) => {}
        }
        let _ = buf.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, ev: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let c = self.cfg;
        let fullscreen = c.fullscreen.then(|| {
            let mon = ev
                .primary_monitor()
                .or_else(|| ev.available_monitors().next());
            let mode = mon.as_ref().and_then(|m| {
                m.video_modes().find(|v| {
                    v.size() == winit::dpi::PhysicalSize::new(c.width, c.height)
                        && (v.refresh_rate_millihertz() + 500) / 1000 == c.refresh
                })
            });
            mode.map_or(Fullscreen::Borderless(mon), Fullscreen::Exclusive)
        });
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                c.width as f64,
                c.height as f64,
            ))
            .with_fullscreen(fullscreen);
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
                    if down && self.text.is_some() && code == winit::keyboard::KeyCode::Escape {
                        ev.exit();
                    }
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
        let lines = self.link.text.lock().unwrap().take();
        if let Some(lines) = lines {
            self.text = Some((lines, false));
            self.draw();
        }
        if let Some((_, armed)) = self.text.as_mut() {
            // B or Back.
            let held = pad.buttons & 0x2020 != 0;
            if *armed && held {
                ev.exit();
            }
            *armed |= !held;
        }
        let fresh = self.link.frame.lock().unwrap().take();
        if let Some(img) = fresh.filter(|_| self.text.is_none()) {
            self.last = Some(img);
            self.draw();
        }
        // Read input at about 250 Hz; frames arrive far slower.
        ev.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(4),
        ));
    }
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
