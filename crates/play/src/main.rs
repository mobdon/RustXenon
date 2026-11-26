//! `rustxenon-play`: the double-click launcher. Lists the built games under
//! its home folder, starts one as a separate process and waits for it.
//!
//! ```text
//! rustxenon-play [--home <dir>] [<built game folder>...]
//! rustxenon-play --screenshot <out.png> [--page games|settings] [--size <w>x<h>] [--home <dir>]
//! ```
//!
//! Home is the launcher's own folder unless `--home` says otherwise. A folder
//! on the command line (a folder dropped on the exe in Explorer) is added to
//! the recent list. It writes its log to `rustxenon-play-launcher.log`
//! beside itself.

#![windows_subsystem = "windows"]

use std::fs::File;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};
use xenon_play::draw::{self, Hit, Page, View, ROWS};
use xenon_play::settings::{Settings, FIELDS};
use xenon_play::Game;

static LOG: Mutex<Option<File>> = Mutex::new(None);

fn log(msg: &str) {
    if let Some(f) = LOG.lock().unwrap().as_mut() {
        let _ = writeln!(f, "{msg}");
    }
}

fn main() {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    *LOG.lock().unwrap() = File::create(exe_dir.join("rustxenon-play-launcher.log")).ok();
    // No console in the GUI subsystem, so panics go to the log file.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut l) = LOG.lock() {
            if let Some(f) = l.as_mut() {
                let _ = writeln!(f, "panic: {info}");
            }
        }
        default_hook(info);
    }));
    let code = match run(exe_dir) {
        Ok(()) => 0,
        Err(e) => {
            log(&format!("error: {e}"));
            eprintln!("rustxenon-play: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn run(exe_dir: PathBuf) -> Result<(), String> {
    let mut home = exe_dir;
    let mut shot = None;
    let mut page = Page::Games;
    let mut size = (1280, 720);
    let mut folders = Vec::new();
    let mut it = std::env::args_os().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match a.to_str() {
            Some("--home") => home = val("--home")?.into(),
            Some("--screenshot") => shot = Some(PathBuf::from(val("--screenshot")?)),
            Some("--page") => {
                page = match val("--page")?.to_str() {
                    Some("games") => Page::Games,
                    Some("settings") => Page::Settings,
                    _ => return Err("--page is games or settings".into()),
                }
            }
            Some("--size") => {
                let v = val("--size")?;
                size = v
                    .to_str()
                    .and_then(|s| s.split_once('x'))
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .filter(|&(w, h)| w > 0 && h > 0)
                    .ok_or("--size is <width>x<height>")?;
            }
            Some(s) if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => folders.push(PathBuf::from(a)),
        }
    }
    // The game runs with its own folder as working folder, so a relative home
    // would point somewhere else for it.
    let home = std::path::absolute(&home).map_err(|e| format!("--home: {e}"))?;
    log(&format!("home {}", home.display()));
    let mut app = App::new(home);
    app.page = page;
    for f in folders {
        app.add_folder(&f);
    }
    if let Some(out) = shot {
        let (w, h) = size;
        let mut px = vec![0; w * h];
        let t0 = Instant::now();
        let back = draw::backdrop(w, h, app.page);
        let t1 = Instant::now();
        // The overlay is drawn every frame: time a few.
        for _ in 0..10 {
            px.copy_from_slice(&back);
            draw::overlay(&mut px, w, h, &app.view(), 0.0);
        }
        let frame = t1.elapsed() / 10;
        log(&format!(
            "screenshot {w}x{h}: backdrop {:?}, frame {frame:?}",
            t1 - t0
        ));
        return write_png(&out, w, h, &px);
    }
    let ev = EventLoop::new().map_err(|e| e.to_string())?;
    ev.run_app(&mut app).map_err(|e| e.to_string())?;
    log("closed");
    Ok(())
}

fn write_png(path: &Path, w: usize, h: usize, px: &[u32]) -> Result<(), String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let file = File::create(path).map_err(|e| err(&e))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let rgb: Vec<u8> = px
        .iter()
        .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8])
        .collect();
    let mut wr = enc.write_header().map_err(|e| err(&e))?;
    wr.write_image_data(&rgb).map_err(|e| err(&e))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Up,
    Down,
    Left,
    Right,
    Go,
    Back,
    Refresh,
    Page,
    Full,
}

type Surface = softbuffer::Surface<Rc<Window>, Rc<Window>>;

struct App {
    home: PathBuf,
    games: Vec<Game>,
    settings: Settings,
    /// Why settings.toml cannot be read, while it cannot: then it is not
    /// overwritten, so a hand edit with a typo keeps its other keys.
    settings_err: Option<String>,
    page: Page,
    /// Selection and first row shown, per page.
    sel: [usize; 2],
    scroll: [usize; 2],
    /// A message about the last thing done; it goes when the player moves on.
    note: Option<String>,
    running: Option<(Game, Child)>,
    window: Option<(Rc<Window>, Surface)>,
    /// The backdrop of each page seen at this size, so a page switch is free.
    back: Vec<(usize, usize, Page, Vec<u32>)>,
    pad: xinput::Reader,
    buttons: u16,
    held: Option<(Act, Instant)>,
    /// The controller is read only while the window has focus.
    focused: bool,
    cursor: (f32, f32),
    /// Wheel or touchpad scroll not yet turned into a whole row, in rows.
    wheel: f32,
    started: Instant,
}

impl App {
    fn new(home: PathBuf) -> App {
        let mut app = App {
            home,
            games: Vec::new(),
            settings: Settings::default(),
            settings_err: None,
            page: Page::Games,
            sel: [0; 2],
            scroll: [0; 2],
            note: None,
            running: None,
            window: None,
            back: Vec::new(),
            pad: xinput::Reader::default(),
            buttons: 0,
            held: None,
            focused: false,
            cursor: (-1.0, -1.0),
            wheel: 0.0,
            started: Instant::now(),
        };
        app.refresh();
        app
    }

    fn view(&self) -> View<'_> {
        let p = self.page as usize;
        View {
            page: self.page,
            games: &self.games,
            settings: &self.settings,
            sel: self.sel[p],
            scroll: self.scroll[p],
            running: self.running.as_ref().map(|r| &r.0),
            note: self.note.as_deref(),
            settings_err: self.settings_err.as_deref(),
            home: &self.home,
        }
    }

    fn len(&self) -> usize {
        match self.page {
            Page::Games => self.games.len(),
            Page::Settings => FIELDS.len(),
        }
    }

    /// Keeps the selection in the list and on screen.
    fn select(&mut self, sel: usize) {
        let p = self.page as usize;
        let len = self.len();
        let sel = sel.min(len.saturating_sub(1));
        self.sel[p] = sel;
        let s = &mut self.scroll[p];
        if sel < *s {
            *s = sel;
        } else if sel >= *s + ROWS {
            *s = sel + 1 - ROWS;
        }
        *s = (*s).min(len.saturating_sub(ROWS));
    }

    /// Selects row `sel` for the player: a note about the row left goes.
    fn pick(&mut self, sel: usize) {
        if sel != self.sel[self.page as usize] {
            self.note = None;
        }
        self.select(sel);
    }

    fn set_page(&mut self, p: Page) {
        if p != self.page {
            self.note = None;
        }
        self.page = p;
    }

    /// Reads the games and settings again, keeping the selected game, or
    /// the place in the list if it is gone.
    fn refresh(&mut self) {
        let keep = self.games.get(self.sel[0]).map(|g| g.dir.clone());
        self.games = xenon_play::merged(&self.home);
        self.load_settings();
        let at = keep.and_then(|d| self.games.iter().position(|g| g.dir == d));
        let page = std::mem::replace(&mut self.page, Page::Games);
        self.select(at.unwrap_or(self.sel[0]));
        self.page = page;
        log(&format!("{} games", self.games.len()));
    }

    /// Reads settings.toml; false if it cannot be read.
    fn load_settings(&mut self) -> bool {
        match Settings::load(&self.home.join("settings.toml")) {
            Ok(s) => {
                self.settings = s;
                self.settings_err = None;
                true
            }
            Err(e) => {
                log(&format!("settings.toml: {e}"));
                self.settings_err = Some(format!("settings.toml not read: {e}"));
                false
            }
        }
    }

    /// Adds a dropped folder (or its game.exe or run.toml) to the recent list.
    fn add_folder(&mut self, p: &Path) {
        let dir = xenon_play::dropped_folder(p);
        let dir = std::path::absolute(&dir).unwrap_or(dir);
        match xenon_play::load_game(&dir) {
            Ok(_) => {
                let recent = xenon_play::add_recent(&xenon_play::load_recent(&self.home), &dir);
                if let Err(e) = xenon_play::save_recent(&self.home, &recent) {
                    self.note = Some(format!("recent.txt not saved: {e}"));
                } else {
                    self.note = None;
                }
                log(&format!("added {}", dir.display()));
                self.refresh();
                self.page = Page::Games;
                self.select(0);
            }
            Err(e) => {
                log(&format!("not added: {}: {e}", dir.display()));
                // The folder's name: the whole path would not fit.
                let name = dir.file_name().unwrap_or(dir.as_os_str());
                let name = name.to_string_lossy();
                self.note = Some(format!("{name} is not a built game folder: {e}"));
            }
        }
    }

    fn start(&mut self) {
        let Some(g) = self.games.get(self.sel[0]).cloned() else {
            return;
        };
        self.note = None;
        // Its folder may have gone since the list was read.
        let spawned = xenon_play::load_game(&g.dir).and_then(|_| {
            let c = xenon_play::command(&self.home, &g).spawn();
            c.map_err(|e| format!("game.exe: {e}"))
        });
        match spawned {
            Ok(child) => {
                log(&format!("started {} (pid {})", g.dir.display(), child.id()));
                // Only a game that started goes to the top of the list.
                let recent = xenon_play::add_recent(&xenon_play::load_recent(&self.home), &g.dir);
                if let Err(e) = xenon_play::save_recent(&self.home, &recent) {
                    log(&format!("recent.txt: {e}"));
                }
                self.running = Some((g, child));
            }
            Err(e) => {
                log(&format!("cannot start {}: {e}", g.dir.display()));
                self.note = Some(format!("Cannot start {}: {e}", g.title));
            }
        }
        self.refresh();
    }

    fn act(&mut self, a: Act) {
        // Full screen is the launcher window's own: it works while a game runs.
        if self.running.is_some() && a != Act::Full {
            return;
        }
        let p = self.page as usize;
        match a {
            Act::Up => self.pick(self.sel[p].saturating_sub(1)),
            Act::Down => self.pick(self.sel[p] + 1),
            Act::Left | Act::Right if self.page == Page::Settings => {
                self.change(if a == Act::Left { -1 } else { 1 })
            }
            Act::Left | Act::Right => {}
            Act::Go => match self.page {
                Page::Games => self.start(),
                Page::Settings => self.change(1),
            },
            Act::Back => {
                self.note = None;
                self.page = Page::Games;
            }
            Act::Refresh => {
                self.note = None;
                self.refresh();
            }
            Act::Page => self.set_page(match self.page {
                Page::Games => Page::Settings,
                Page::Settings => Page::Games,
            }),
            Act::Full => {
                if let Some((w, _)) = &self.window {
                    let full = w.fullscreen().is_some();
                    w.set_fullscreen((!full).then_some(Fullscreen::Borderless(None)));
                }
            }
        }
    }

    /// Steps the selected setting in settings.toml as it is on disk now (it
    /// may have been edited since it was read), so only that key changes.
    /// The page shows what the file holds, saved or not.
    fn change(&mut self, dir: i32) {
        self.note = None;
        if !self.load_settings() {
            return;
        }
        let mut next = self.settings.clone();
        next.step(self.sel[1], dir);
        match next.save(&self.home.join("settings.toml")) {
            Ok(()) => self.settings = next,
            Err(e) => {
                log(&format!("settings.toml not saved: {e}"));
                self.note = Some(format!("settings.toml not saved: {e}"));
            }
        }
    }

    fn draw(&mut self) {
        let t = self.started.elapsed().as_secs_f32();
        let Some((win, s)) = self.window.as_mut() else {
            return;
        };
        let size = win.inner_size();
        let (Some(dw), Some(dh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        let (w, h) = (size.width as usize, size.height as usize);
        if s.resize(dw, dh).is_err() {
            return;
        }
        let page = self.page;
        if !self.back.iter().any(|b| (b.0, b.1, b.2) == (w, h, page)) {
            self.back.retain(|b| (b.0, b.1) == (w, h));
            self.back.push((w, h, page, draw::backdrop(w, h, page)));
        }
        let Some((_, s)) = self.window.as_mut() else {
            return;
        };
        let Ok(mut buf) = s.buffer_mut() else { return };
        if let Some((.., back)) = self.back.iter().find(|b| b.2 == page) {
            buf.copy_from_slice(back);
        }
        // Not self.view(): self.window is borrowed mutably for the buffer.
        let v = View {
            page: self.page,
            games: &self.games,
            settings: &self.settings,
            sel: self.sel[page as usize],
            scroll: self.scroll[page as usize],
            running: self.running.as_ref().map(|r| &r.0),
            note: self.note.as_deref(),
            settings_err: self.settings_err.as_deref(),
            home: &self.home,
        };
        draw::overlay(&mut buf, w, h, &v, t);
        let _ = buf.present();
    }

    /// The controller: buttons on press, directions repeating while held.
    fn read_pad(&mut self) {
        let p = xenon_input::filtered(self.pad.read());
        let pressed = p.buttons & !self.buttons;
        self.buttons = p.buttons;
        for (bit, a) in [
            (0x1000, Act::Go),
            (0x0010, Act::Go),
            (0x2000, Act::Back),
            (0x0020, Act::Back),
            (0x8000, Act::Refresh),
            (0x0100, Act::Page),
            (0x0200, Act::Page),
        ] {
            if pressed & bit != 0 {
                self.act(a);
            }
        }
        let dir = pad_dir(&p, self.held.map(|h| h.0));
        let now = Instant::now();
        match (dir, self.held) {
            (Some(d), Some((h, next))) if d == h => {
                if now >= next {
                    self.act(d);
                    self.held = Some((d, now + Duration::from_millis(110)));
                }
            }
            (Some(d), _) => {
                self.act(d);
                self.held = Some((d, now + Duration::from_millis(380)));
            }
            (None, _) => self.held = None,
        }
    }

    /// Takes the pad as it is now without acting, so a button still held
    /// when a game exits (Start, say) does not count as a new press, and a
    /// direction still held waits for its release.
    fn pad_settle(&mut self) {
        let p = xenon_input::filtered(self.pad.read());
        self.buttons = p.buttons;
        let never = Instant::now() + Duration::from_secs(3600);
        self.held = pad_dir(&p, None).map(|d| (d, never));
    }

    fn size(&self) -> (usize, usize) {
        self.window.as_ref().map_or((0, 0), |(w, _)| {
            let s = w.inner_size();
            (s.width as usize, s.height as usize)
        })
    }
}

/// The direction the pad points. A direction held lets go only below a
/// lower threshold, so a stick resting near the edge does not flicker.
fn pad_dir(p: &xenon_input::Pad, held: Option<Act>) -> Option<Act> {
    let b = p.buttons;
    let past = |a: Act, v: i32| v > if held == Some(a) { 12000 } else { 16000 };
    let (lx, ly) = (i32::from(p.lx), i32::from(p.ly));
    if b & 0x0001 != 0 || past(Act::Up, ly) {
        Some(Act::Up)
    } else if b & 0x0002 != 0 || past(Act::Down, -ly) {
        Some(Act::Down)
    } else if b & 0x0004 != 0 || past(Act::Left, -lx) {
        Some(Act::Left)
    } else if b & 0x0008 != 0 || past(Act::Right, lx) {
        Some(Act::Right)
    } else {
        None
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, ev: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // Fit the screen: a small or high-scale one is under 1280x720 logical.
        let (w, h) = ev.primary_monitor().map_or((1280.0, 720.0), |m| {
            let s = m.size().to_logical::<f64>(m.scale_factor());
            xenon_play::window_size((s.width, s.height))
        });
        let (mw, mh) = xenon_play::MIN_WINDOW;
        let attrs = Window::default_attributes()
            .with_title("RustXenon")
            .with_inner_size(winit::dpi::LogicalSize::new(w, h))
            .with_min_inner_size(winit::dpi::LogicalSize::new(mw, mh));
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
            Ok(x) => {
                log("window open");
                self.focused = x.0.has_focus();
                self.window = Some(x);
            }
            Err(e) => {
                log(&format!("error: cannot open a window: {e}"));
                ev.exit();
            }
        }
    }

    fn window_event(&mut self, ev: &ActiveEventLoop, _: WindowId, e: WindowEvent) {
        match e {
            WindowEvent::CloseRequested => ev.exit(),
            WindowEvent::Focused(f) => {
                self.focused = f;
                if f {
                    self.pad_settle();
                }
            }
            // A synthetic press is a key already down when focus came back
            // (the Enter that closed a game, say): not a new press.
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } if event.state == ElementState::Pressed => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                use KeyCode as K;
                let a = match code {
                    K::KeyW | K::ArrowUp => Act::Up,
                    K::KeyS | K::ArrowDown => Act::Down,
                    K::KeyA | K::ArrowLeft => Act::Left,
                    K::KeyD | K::ArrowRight => Act::Right,
                    K::Enter | K::NumpadEnter | K::Space => Act::Go,
                    K::Escape | K::Backspace => Act::Back,
                    K::F5 | K::KeyR | K::KeyY => Act::Refresh,
                    K::Tab | K::KeyQ | K::KeyE => Act::Page,
                    K::F11 => Act::Full,
                    _ => return,
                };
                // A held key repeats moves and steps, nothing else.
                if event.repeat && !matches!(a, Act::Up | Act::Down | Act::Left | Act::Right) {
                    return;
                }
                self.act(a);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (w, h) = self.size();
                self.cursor = (position.x as f32, position.y as f32);
                if let Some(Hit::Row(r)) = draw::hit(w, h, self.page, self.cursor.0, self.cursor.1)
                {
                    let i = self.scroll[self.page as usize] + r;
                    if i < self.len() && self.running.is_none() {
                        self.pick(i);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let (w, h) = self.size();
                match draw::hit(w, h, self.page, self.cursor.0, self.cursor.1) {
                    Some(Hit::Tab(p)) if self.running.is_none() => self.set_page(p),
                    Some(Hit::Row(r))
                        if self.running.is_none()
                            && self.scroll[self.page as usize] + r < self.len() =>
                    {
                        self.pick(self.scroll[self.page as usize] + r);
                        // A settings row changes only through its arrows.
                        if self.page == Page::Games {
                            self.act(Act::Go);
                        }
                    }
                    Some(Hit::Step(r, dir))
                        if self.running.is_none()
                            && self.scroll[self.page as usize] + r < self.len() =>
                    {
                        self.pick(self.scroll[self.page as usize] + r);
                        self.change(dir);
                    }
                    Some(Hit::Info) if self.running.is_none() => {
                        self.act(if self.games.is_empty() {
                            Act::Refresh
                        } else {
                            Act::Go
                        })
                    }
                    Some(Hit::Hint(i)) => {
                        let a = [Act::Refresh, Act::Back, Act::Go, Act::Page];
                        self.act(a[i.min(3)]);
                    }
                    _ => {}
                }
            }
            // A right click on a settings row, arrows too, steps its value down.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } if self.page == Page::Settings && self.running.is_none() => {
                let (w, h) = self.size();
                if let Some(Hit::Row(r) | Hit::Step(r, _)) =
                    draw::hit(w, h, self.page, self.cursor.0, self.cursor.1)
                {
                    if self.scroll[self.page as usize] + r < self.len() {
                        self.pick(self.scroll[self.page as usize] + r);
                        self.change(-1);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // A touchpad sends many small steps: add them up, one row per 60 px.
                self.wheel += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let rows = self.wheel.trunc();
                self.wheel -= rows;
                let a = if rows > 0.0 { Act::Up } else { Act::Down };
                for _ in 0..rows.abs() as usize {
                    self.act(a);
                }
            }
            WindowEvent::DroppedFile(p) => {
                if self.running.is_none() {
                    self.add_folder(&p);
                } else {
                    self.note = Some("Close the game first, then drop the folder again.".into());
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, ev: &ActiveEventLoop) {
        if let Some((_, child)) = self.running.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    log(&format!("game exited: {status}"));
                    let title = self.running.take().map(|r| r.0.title).unwrap_or_default();
                    self.pad_settle();
                    if !status.success() {
                        self.note = Some(format!(
                            "{title} stopped ({status}) - see rustxenon-play.log in its folder."
                        ));
                    }
                    self.refresh();
                }
                Ok(None) => {}
                Err(e) => {
                    log(&format!("waiting for the game: {e}"));
                    self.running = None;
                    self.pad_settle();
                }
            }
        } else if self.focused {
            self.read_pad();
        }
        if let Some((w, _)) = &self.window {
            w.request_redraw();
        }
        // About 30 frames a second for the pulse; slower while a game runs.
        let wait = if self.running.is_some() { 100 } else { 33 };
        ev.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(wait),
        ));
    }
}

#[cfg(windows)]
mod xinput {
    //! The first connected XInput controller (the same reader as the game
    //! window's).
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
            buttons: s.buttons,
            lt: s.lt,
            rt: s.rt,
            lx: s.lx,
            ly: s.ly,
            rx: s.rx,
            ry: s.ry,
        })
    }

    /// Asking an empty port is slow, so ports are searched at most once a second.
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
                    super::log(&format!("controller on XInput port {user}"));
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
    /// shortcut: no controller off Windows (keyboard and mouse only).
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
    use xenon_input::Pad;

    #[test]
    fn pad_direction_lets_go_lower() {
        let stick = |lx: i16, ly: i16| Pad {
            lx,
            ly,
            ..Pad::default()
        };
        assert_eq!(pad_dir(&stick(0, 0), None), None);
        assert_eq!(pad_dir(&stick(0, 17000), None), Some(Act::Up));
        assert_eq!(pad_dir(&stick(0, -17000), None), Some(Act::Down));
        assert_eq!(pad_dir(&stick(-17000, 0), None), Some(Act::Left));
        assert_eq!(pad_dir(&stick(17000, 0), None), Some(Act::Right));
        // Resting at 14000: not a press, but a held Up stays held.
        assert_eq!(pad_dir(&stick(0, 14000), None), None);
        assert_eq!(pad_dir(&stick(0, 14000), Some(Act::Up)), Some(Act::Up));
        assert_eq!(pad_dir(&stick(0, 11000), Some(Act::Up)), None);
        assert_eq!(pad_dir(&stick(0, i16::MIN), None), Some(Act::Down));
        let dpad = Pad {
            buttons: 0x0008,
            ..Pad::default()
        };
        assert_eq!(pad_dir(&dpad, None), Some(Act::Right));
    }
}
