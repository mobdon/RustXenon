//! `xenon-window`: a window that shows the latest frame and turns an XInput
//! controller (Windows) plus the keyboard into one pad reading.
//!
//! The event loop runs on its own thread so the host's main thread keeps its
//! headless watch loop. Once the drawer hands over its device
//! ([`Link::attach_gpu`]), frames are presented through a swapchain on it,
//! scaled on the GPU at the monitor's refresh (see [`gpu`]). Until then, or
//! without one, frames are copied to the window with the CPU (bilinear
//! scaling, letterboxed, rows split over a few threads).

use std::num::NonZeroU32;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};
use xenon_input::{filtered, merge, Keyboard, Mouse, Pad};

mod gpu;
mod pace;

pub use gpu::Gpu;

/// Pixels shared with whoever made them, so a frame reaches the window
/// without being copied.
pub type Pixels = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// One frame to show: rows top to bottom, 8-bit RGBA.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Pixels,
}

/// How the window opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    /// Exclusive full screen at width x height and `refresh` Hz when the
    /// monitor has that mode, else borderless full screen.
    pub fullscreen: bool,
    pub refresh: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            width: 1280,
            height: 720,
            fullscreen: false,
            refresh: 60,
        }
    }
}

/// A frame to show: pixels, or a texture on the device of
/// [`Link::attach_gpu`].
enum Shown {
    Cpu(Image),
    Gpu(wgpu::Texture),
}

/// Shared between the host and the window thread.
#[derive(Default)]
pub struct Link {
    frame: Mutex<Option<Shown>>,
    text: Mutex<Option<Vec<String>>>,
    pad: Mutex<Pad>,
    /// Wakes the event loop, once it runs, so a frame is shown when it
    /// arrives rather than at the next input poll.
    wake: Mutex<Option<EventLoopProxy<()>>>,
    /// A device to present with, and where to answer whether it can.
    attach: Mutex<Option<(Gpu, Answer)>>,
}

/// Whether the window presents with the device it was given.
type Answer = mpsc::Sender<Result<(), String>>;

impl Link {
    /// Hands the window a new frame; an older one not yet shown is dropped.
    pub fn show(&self, width: u32, height: u32, rgba8: Pixels) {
        *self.frame.lock().unwrap() = Some(Shown::Cpu(Image {
            width,
            height,
            rgba8,
        }));
        self.wake();
    }

    /// Hands the window a frame texture on the device given to
    /// [`Link::attach_gpu`], whose writes are submitted already; an older
    /// one not yet shown is dropped.
    pub fn show_texture(&self, tex: wgpu::Texture) {
        *self.frame.lock().unwrap() = Some(Shown::Gpu(tex));
        self.wake();
    }

    /// Has the window present through a swapchain on `gpu`'s device from
    /// now on, and waits for its answer: an error leaves it copying
    /// frames with the CPU, and [`Link::show_texture`] unused.
    pub fn attach_gpu(&self, gpu: Gpu) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        *self.attach.lock().unwrap() = Some((gpu, tx));
        self.wake();
        (rx.recv_timeout(Duration::from_secs(10)))
            .map_err(|_| "the window did not take the device".to_string())?
    }

    fn wake(&self) {
        if let Some(p) = &*self.wake.lock().unwrap() {
            let _ = p.send_event(());
        }
    }

    /// Replaces the picture with these lines of text (green on dark, the
    /// newest at the bottom) for good: later frames are ignored, and Esc, B
    /// or Back closes the window.
    pub fn show_text(&self, lines: Vec<String>) {
        *self.text.lock().unwrap() = Some(lines);
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
pub fn spawn(title: String, cfg: Config, on_close: impl FnOnce() + Send + 'static) -> Arc<Link> {
    let link = Arc::new(Link::default());
    let l = link.clone();
    std::thread::Builder::new()
        .name("window".into())
        .spawn(move || {
            match event_loop() {
                Ok(ev) => {
                    *l.wake.lock().unwrap() = Some(ev.create_proxy());
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

type Surface = softbuffer::Surface<Arc<Window>, Arc<Window>>;

struct App {
    title: String,
    cfg: Config,
    link: Arc<Link>,
    keys: Keyboard,
    mouse: Mouse,
    /// The cursor is hidden and held in the window: mouse motion is the camera.
    captured: bool,
    /// A d-pad press from the mouse wheel (weapon switch) and until when.
    wheel: Option<(u16, Instant)>,
    last_poll: Instant,
    pad: xinput::Reader,
    window: Option<Arc<Window>>,
    /// The CPU path's surface, made when first drawn to.
    soft: Option<Surface>,
    /// The GPU path, once the drawer handed over its device.
    gpu: Option<gpu::Presenter>,
    /// Kept so a resize can redraw it.
    last: Option<Shown>,
    /// Text shown instead of frames, and whether the close buttons have
    /// been seen up since (so a button held at the stop does not close it).
    text: Option<(Vec<String>, bool)>,
    /// Intervals between new frames as they are presented.
    pace: pace::Pacing,
}

impl App {
    fn new(title: String, cfg: Config, link: Arc<Link>) -> Self {
        App {
            title,
            cfg,
            link,
            keys: Keyboard::default(),
            mouse: Mouse::default(),
            captured: false,
            wheel: None,
            last_poll: Instant::now(),
            pad: xinput::Reader::default(),
            window: None,
            soft: None,
            gpu: None,
            last: None,
            text: None,
            pace: pace::Pacing::default(),
        }
    }

    /// Takes or gives back the mouse: hidden and held in the window while
    /// taken. Alt-Tab (focus lost) gives it back; a click takes it again.
    fn capture(&mut self, on: bool) {
        let Some(w) = self.window.as_ref() else {
            return;
        };
        if on {
            let held = (w.set_cursor_grab(CursorGrabMode::Locked))
                .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
            if held.is_err() {
                return;
            }
        } else {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
        }
        w.set_cursor_visible(!on);
        self.captured = on;
    }

    /// Shows the text or the last frame; whether it reached the screen.
    fn draw(&mut self) -> bool {
        let Some(w) = self.window.clone() else {
            return false;
        };
        if self.last.is_none() && self.text.is_none() {
            return false;
        }
        let size = w.inner_size();
        let (Some(dw), Some(dh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return false;
        };
        let (ww, wh) = (size.width, size.height);
        if let Some(g) = self.gpu.as_mut() {
            let tex = match (&self.text, &self.last) {
                (Some((lines, _)), _) => {
                    let mut buf = vec![0; ww as usize * wh as usize];
                    text(lines, &mut buf, ww, wh);
                    let rgba: Vec<u8> = (buf.iter())
                        .flat_map(|&p| [(p >> 16) as u8, (p >> 8) as u8, p as u8, 255])
                        .collect();
                    g.upload(ww, wh, &rgba)
                }
                (None, Some(Shown::Gpu(t))) => t.clone(),
                (None, Some(Shown::Cpu(img))) => {
                    let px: &[u8] = (*img.rgba8).as_ref();
                    let n = img.width as usize * img.height as usize * 4;
                    if n == 0 || px.len() < n {
                        return false;
                    }
                    g.upload(img.width, img.height, &px[..n])
                }
                (None, None) => return false,
            };
            return match g.present(&tex, ww, wh) {
                Ok(shown) => shown,
                Err(e) => {
                    eprintln!("window: present: {e}");
                    false
                }
            };
        }
        if self.soft.is_none() {
            let made = softbuffer::Context::new(w.clone())
                .and_then(|cx| Surface::new(&cx, w.clone()))
                .map_err(|e| eprintln!("window: {e}"));
            self.soft = made.ok();
        }
        let Some(s) = self.soft.as_mut() else {
            return false;
        };
        if s.resize(dw, dh).is_err() {
            return false;
        }
        let Ok(mut buf) = s.buffer_mut() else {
            return false;
        };
        match (&self.text, &self.last) {
            (Some((lines, _)), _) => text(lines, &mut buf, ww, wh),
            (None, Some(Shown::Cpu(img))) => blit(img, &mut buf, ww, wh),
            // Textures come only with a presenter.
            (None, Some(Shown::Gpu(_))) | (None, None) => return false,
        }
        buf.present().is_ok()
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
        let attrs = with_icon(Window::default_attributes())
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                c.width as f64,
                c.height as f64,
            ))
            .with_fullscreen(fullscreen);
        match ev.create_window(attrs) {
            Ok(w) => self.window = Some(Arc::new(w)),
            Err(e) => {
                eprintln!("window: {e}");
                ev.exit();
            }
        }
    }

    fn window_event(&mut self, ev: &ActiveEventLoop, _: WindowId, e: WindowEvent) {
        match e {
            WindowEvent::CloseRequested => ev.exit(),
            WindowEvent::Focused(false) => {
                self.keys.release_all();
                self.mouse.release();
                self.capture(false);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                // The first click only takes the mouse.
                if down && !self.captured {
                    self.capture(true);
                    return;
                }
                match button {
                    MouseButton::Left => self.mouse.left = down,
                    MouseButton::Right => self.mouse.right = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if self.captured => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                // Up is the next weapon (d-pad right), down the previous one.
                let b = if y > 0.0 { 0x0008 } else { 0x0004 };
                self.wheel = Some((b, Instant::now() + Duration::from_millis(80)));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    // Alt gives the mouse back; a click takes it again.
                    if down
                        && matches!(
                            code,
                            winit::keyboard::KeyCode::AltLeft | winit::keyboard::KeyCode::AltRight
                        )
                    {
                        self.mouse.release();
                        self.capture(false);
                    }
                    if down && self.text.is_some() && code == winit::keyboard::KeyCode::Escape {
                        ev.exit();
                    }
                    self.keys.key(&format!("{code:?}"), down);
                }
            }
            WindowEvent::Resized(_) | WindowEvent::RedrawRequested => {
                self.draw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, e: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = e {
            if self.captured {
                self.mouse.motion(delta.0, delta.1);
            }
        }
    }

    fn about_to_wait(&mut self, ev: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_poll).as_secs_f32();
        self.last_poll = now;
        self.keys.step(dt);
        let mouse = self.mouse.pad_in(dt, self.keys.in_vehicle());
        let mut pad = merge(merge(filtered(self.pad.read()), self.keys.pad()), mouse);
        match self.wheel {
            Some((b, until)) if now < until => pad.buttons |= b,
            _ => self.wheel = None,
        }

        *self.link.pad.lock().unwrap() = pad;
        let asked = self.link.attach.lock().unwrap().take();
        match (asked, self.window.clone()) {
            (Some((gpu, answer)), Some(w)) => {
                // The swapchain takes the window over from the CPU path.
                self.soft = None;
                let made = gpu::Presenter::new(gpu, w).map(|p| self.gpu = Some(p));
                if let Err(e) = &made {
                    eprintln!("window: frames stay on the CPU: {e}");
                }
                let _ = answer.send(made);
                self.draw();
            }
            // No window yet: asked again next time.
            (Some(a), None) => *self.link.attach.lock().unwrap() = Some(a),
            (None, _) => {}
        }
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
            if self.draw() {
                if let Some(s) = self.pace.frame(Instant::now()) {
                    eprintln!("window: {s}");
                }
            }
        }
        // Read input at about 250 Hz; frames wake the loop when they come.
        ev.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(4),
        ));
    }
}

/// Scales `img` to fit `w` x `h` keeping its shape, black bars around it,
/// into 0RGB pixels. Bilinear: a 720p frame stretched to a large window with
/// nearest-neighbour scaling showed every frame pixel as a hard block.
fn blit(img: &Image, out: &mut [u32], w: u32, h: u32) {
    blit_with(img, out, w, h, true)
}

/// [`blit`]; with `copy`, a frame shown at its own size (bars or not) is
/// copied straight, the same pixels the bilinear path gives without its
/// arithmetic.
fn blit_with(img: &Image, out: &mut [u32], w: u32, h: u32, copy: bool) {
    out.fill(0);
    let src: &[u8] = (*img.rgba8).as_ref();
    if img.width == 0 || img.height == 0 || src.len() < (img.width * img.height * 4) as usize {
        return;
    }
    let Some((ox, oy, fw, fh)) = fit(img.width, img.height, w, h) else {
        return;
    };
    let same = copy && (fw, fh) == (img.width, img.height);
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
                    if same {
                        // At its own size every tap is source pixel y0, x.
                        let line = src[r0 * 4..][..fw as usize * 4].chunks_exact(4);
                        for (px, p) in row.iter_mut().zip(line) {
                            *px = u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2]);
                        }
                        continue;
                    }
                    for (px, &(x0, x1, wx)) in row.iter_mut().zip(xs) {
                        let at = |r: usize, x: usize| &src[(r + x) * 4..][..3];
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

/// Where an `iw` x `ih` frame goes in a `w` x `h` window: the largest
/// rectangle with its shape that fits, centred (x, y, width, height); none
/// when nothing of it would show.
fn fit(iw: u32, ih: u32, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    if iw == 0 || ih == 0 {
        return None;
    }
    let (fw, fh) = if w as u64 * ih as u64 > h as u64 * iw as u64 {
        ((h as u64 * iw as u64 / ih as u64) as u32, h)
    } else {
        (w, (w as u64 * ih as u64 / iw as u64) as u32)
    };
    (fw != 0 && fh != 0).then_some(((w - fw) / 2, (h - fh) / 2, fw, fh))
}

/// The last `rows` screen lines of `text`, each line wrapped at `cols`
/// characters.
pub fn tail(text: &str, rows: usize, cols: usize) -> Vec<String> {
    let cols = cols.max(1);
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        if chars.is_empty() {
            out.push(String::new());
        }
        out.extend(chars.chunks(cols).map(|c| c.iter().collect()));
    }
    let skip = out.len().saturating_sub(rows);
    out.split_off(skip)
}

/// Draws `lines` (wrapped to fit, the newest at the bottom kept) in an 8x8
/// pixel font, green on dark, at twice the size when the window is wide.
fn text(lines: &[String], out: &mut [u32], w: u32, h: u32) {
    use font8x8::legacy::BASIC_LEGACY;
    out.fill(BG);
    let px = if w >= 1600 { 2 } else { 1 };
    let (cw, ch) = (8 * px, 10 * px);
    let (cols, rows) = ((w.saturating_sub(2 * cw) / cw) as usize, (h / ch) as usize);
    if cols == 0 {
        return;
    }
    for (r, line) in tail(&lines.join("\n"), rows, cols).iter().enumerate() {
        for (c, chr) in line.chars().enumerate() {
            let glyph = BASIC_LEGACY
                .get(chr as usize)
                .unwrap_or(&BASIC_LEGACY[b'?' as usize]);
            let (x0, y0) = (cw + c as u32 * cw, r as u32 * ch + px);
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in (0..8).filter(|gx| bits >> gx & 1 != 0) {
                    let x = (x0 + gx * px) as usize;
                    for dy in 0..px {
                        let y = (y0 + gy as u32 * px + dy) as usize;
                        out[y * w as usize + x..][..px as usize].fill(FG);
                    }
                }
            }
        }
    }
}

const BG: u32 = 0x0010_1410;
const FG: u32 = 0x0040_FF60;

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
    fn tail_wraps_and_keeps_the_newest() {
        let log = "one\n\nabcdefg\nlast";
        assert_eq!(tail(log, 10, 3), ["one", "", "abc", "def", "g", "las", "t"]);
        assert_eq!(tail(log, 3, 3), ["g", "las", "t"]);
        assert_eq!(tail(log, 2, 100), ["abcdefg", "last"]);
        assert!(tail("", 5, 5).is_empty());
        assert!(tail(log, 0, 0).is_empty());
        // Characters, not bytes: never split inside one.
        assert_eq!(tail("\u{e9}\u{e9}\u{e9}", 5, 2), ["\u{e9}\u{e9}", "\u{e9}"]);
    }

    #[test]
    fn text_stays_inside_the_buffer() {
        for (w, h) in [(40, 25), (41, 9), (1600, 23), (17, 3), (2, 2), (0, 0)] {
            let mut out = vec![7; (w * h) as usize];
            text(&["A\u{e9}xyz, far too long to fit".into()], &mut out, w, h);
            assert!(out.iter().all(|&p| p == BG || p == FG), "{w}x{h}");
        }
        let mut out = vec![0; 40 * 25];
        text(&["A".into()], &mut out, 40, 25);
        assert!(out.contains(&FG));
    }

    #[test]
    fn letterbox_and_scale() {
        // 2x1 frame, red then blue, into a 4x4 window: a black row on
        // top, the 4x2 picture, a black row below. Bilinear: the outer
        // pixels are the frame's, the inner two a quarter and three
        // quarters of the way from red to blue.
        let img = Image {
            width: 2,
            height: 1,
            rgba8: Arc::new(vec![255, 0, 0, 255, 0, 0, 255, 255]),
        };
        // Where the GPU path draws it too.
        assert_eq!(fit(2, 1, 4, 4), Some((0, 1, 4, 2)));
        assert_eq!(fit(1280, 720, 2560, 1600), Some((0, 80, 2560, 1440)));
        assert_eq!(fit(1280, 720, 3440, 1440), Some((440, 0, 2560, 1440)));
        assert_eq!(fit(1280, 720, 1, 1), None);
        assert_eq!(fit(0, 720, 100, 100), None);
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
            rgba8: Arc::new(vec![128; 1280 * 720 * 4]),
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
            rgba8: Arc::new(vec![0, 255, 0, 255]),
        };
        let mut out = vec![7; 3];
        blit(&one, &mut out, 3, 1);
        assert_eq!(out, [0, 0x00FF00, 0]);
        // A short buffer draws nothing rather than panicking.
        let bad = Image {
            width: 4,
            height: 4,
            rgba8: Arc::new(vec![0; 4]),
        };
        blit(&bad, &mut out, 3, 1);
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn copy_at_own_size_matches_the_bilinear_blit() {
        // Every byte value in every channel, at a size that splits rows
        // over the threads unevenly; then with bars above and below and at
        // the sides, still at the frame's own size.
        let (fw, fh) = (37, 23);
        let img = Image {
            width: fw,
            height: fh,
            rgba8: Arc::new(
                (0..fw * fh * 4)
                    .map(|i: u32| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
                    .collect::<Vec<_>>(),
            ),
        };
        for (w, h) in [(fw, fh), (fw, fh + 9), (fw + 6, fh)] {
            let (mut fast, mut slow) = (vec![7; (w * h) as usize], vec![9; (w * h) as usize]);
            blit_with(&img, &mut fast, w, h, true);
            blit_with(&img, &mut slow, w, h, false);
            assert_eq!(fast, slow, "{w}x{h}");
            assert!(fast.iter().any(|&p| p != 0), "{w}x{h}");
        }
    }
}
