//! The Android window: a NativeActivity's native window, filled with the
//! CPU, and a Bluetooth controller read from the activity's input events.
//!
//! [`run`] is the event pump and runs on the `android_main` thread; the game
//! runs elsewhere and talks to it through the [`Link`] that [`spawn`] made.

use crate::{blit, buffer_size, text, Config, Image, Link, Px};
use android_activity::input::{Axis, InputEvent, KeyAction, Source};
use android_activity::ndk::hardware_buffer_format::HardwareBufferFormat;
use android_activity::ndk::native_window::NativeWindow;
use android_activity::{InputStatus, MainEvent, PollEvent, WindowManagerFlags};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use xenon_input::android::Gamepad;
use xenon_input::filtered;

pub use android_activity::AndroidApp;

static LINK: OnceLock<Arc<Link>> = OnceLock::new();
type OnClose = Box<dyn FnOnce() + Send>;
static ON_CLOSE: Mutex<Option<OnClose>> = Mutex::new(None);

/// The size text screens are drawn at before the compositor scales them up,
/// so the 8x8 font is readable on a phone.
const TEXT_SIZE: (u32, u32) = (960, 540);

/// Starts no thread: [`run`] on the `android_main` thread already pumps the
/// window. Keeps `on_close` for when the activity goes away or the player
/// leaves a stop screen. The title and size are the activity's, not ours.
pub fn spawn(_title: String, _cfg: Config, on_close: impl FnOnce() + Send + 'static) -> Arc<Link> {
    *ON_CLOSE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(on_close));
    LINK.get_or_init(Default::default).clone()
}

/// Runs `on_close` (which normally exits), else exits.
fn close() -> ! {
    let f = ON_CLOSE.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(f) = f {
        f();
    }
    std::process::exit(0)
}

fn set_hidden(on: bool) {
    if let Some(l) = LINK.get() {
        l.hidden.store(on, Ordering::Relaxed);
    }
}

/// Whether an input source is a controller: the gamepad, joystick or D-pad
/// source bits all set (a keyboard shares their class bit only).
fn from_pad(s: Source) -> bool {
    let s: u32 = s.into();
    [0x0000_0401, 0x0100_0010, 0x0000_0201]
        .iter()
        .any(|&m| s & m == m)
}

/// Feeds one input event to the pad.
fn input(pad: &mut Gamepad, ev: &InputEvent) -> InputStatus {
    let used = match ev {
        InputEvent::KeyEvent(k) if from_pad(k.source()) => {
            let code = u32::from(k.key_code()) as i32;
            match k.action() {
                KeyAction::Down => pad.key(code, true),
                KeyAction::Up => pad.key(code, false),
                _ => false,
            }
        }
        InputEvent::MotionEvent(m) if from_pad(m.source()) && m.pointer_count() > 0 => {
            let p = m.pointer_at_index(0);
            pad.motion(|id| p.axis_value(Axis::from(id as u32)));
            true
        }
        _ => false,
    };
    if used {
        InputStatus::Handled
    } else {
        InputStatus::Unhandled
    }
}

/// The native window with its own size (read before any buffer size is
/// set: after that, width() and height() report the buffer's) and the
/// buffer size last set.
struct Win {
    win: NativeWindow,
    size: (u32, u32),
    buf: (u32, u32),
}

const FMT: Option<HardwareBufferFormat> = Some(HardwareBufferFormat::R8G8B8X8_UNORM);

impl Win {
    /// Takes a new or resized window. 0 x 0 drops any buffer size set
    /// earlier, so its own (new) shape is seen.
    fn new(win: NativeWindow) -> Option<Win> {
        win.set_buffers_geometry(0, 0, FMT).ok()?;
        let size = (win.width().max(0) as u32, win.height().max(0) as u32);
        Some(Win {
            win,
            size,
            buf: (0, 0),
        })
    }
}

/// Fills the window with the stop screen's text, else the newest frame:
/// a buffer with the window's shape at about the picture's size, which the
/// compositor scales to the screen. The buffer size is set only when it
/// changes, so frames do not reallocate the window's buffers.
fn draw(w: &mut Win, lines: Option<&[String]>, last: Option<&Image>) {
    let (pw, ph) = match (lines, last) {
        (Some(_), _) => TEXT_SIZE,
        (None, Some(img)) => (img.width, img.height),
        (None, None) => return,
    };
    let (bw, bh) = buffer_size(w.size.0, w.size.1, pw, ph);
    if bw == 0 {
        return;
    }
    if w.buf != (bw, bh) {
        if w.win
            .set_buffers_geometry(bw as i32, bh as i32, FMT)
            .is_err()
        {
            return;
        }
        w.buf = (bw, bh);
    }
    let win = &w.win;
    // Checked before locking: a locked buffer is posted even if unwritten.
    if !matches!(
        win.format(),
        HardwareBufferFormat::R8G8B8X8_UNORM | HardwareBufferFormat::R8G8B8A8_UNORM
    ) {
        return;
    }
    let Ok(mut g) = win.lock(None) else { return };
    let (w, h, stride) = (g.width() as u32, g.height() as u32, g.stride());
    // SAFETY: a locked 4-byte-per-pixel buffer is stride x height pixels,
    // aligned for u32, and ours until the guard drops (which posts it).
    let out =
        unsafe { std::slice::from_raw_parts_mut(g.bits().cast::<u32>(), stride * h as usize) };
    match (lines, last) {
        (Some(lines), _) => text(lines, out, w, h, stride, Px::Rgbx),
        (None, Some(img)) => blit(img, out, w, h, stride, Px::Rgbx),
        (None, None) => {}
    }
}

/// The event pump, on the `android_main` thread, for good: window made,
/// resized or gone (folding and unfolding a phone does all three, and the
/// game keeps running meanwhile), controller input to the [`Link`] pad,
/// frames and stop screens to the window. The activity's end exits.
pub fn run(app: AndroidApp) -> ! {
    crate::mark_pump_thread();
    // shortcut: FULLSCREEN hides the status bar only; hiding the navigation
    // bar needs JNI (WindowInsetsController), add it if the bar gets in the way.
    app.set_window_flags(
        WindowManagerFlags::KEEP_SCREEN_ON | WindowManagerFlags::FULLSCREEN,
        WindowManagerFlags::empty(),
    );
    let mut pad = Gamepad::default();
    let mut window: Option<Win> = None;
    let mut last: Option<Image> = None;
    // Text shown instead of frames, and whether B and Back have been seen
    // up since (so a button held at the stop does not close it).
    let mut shown: Option<(Vec<String>, bool)> = None;
    loop {
        let (mut redraw, mut gone) = (false, false);
        // Read input at about 250 Hz; frames arrive far slower.
        app.poll_events(Some(Duration::from_millis(4)), |e| {
            if let PollEvent::Main(m) = e {
                match m {
                    MainEvent::InitWindow { .. }
                    | MainEvent::WindowResized { .. }
                    | MainEvent::ConfigChanged { .. }
                    | MainEvent::RedrawNeeded { .. } => {
                        window = app.native_window().and_then(Win::new);
                        redraw = true;
                    }
                    MainEvent::TerminateWindow { .. } => window = None,
                    MainEvent::LostFocus => pad.release_all(),
                    // Home, the app switcher or the screen going off: the
                    // game runs on, unheard and with the pad let go.
                    MainEvent::Pause => {
                        pad.release_all();
                        set_hidden(true);
                    }
                    MainEvent::Resume { .. } => set_hidden(false),
                    MainEvent::Destroy => gone = true,
                    _ => {}
                }
            }
        });
        if gone {
            close();
        }
        if let Ok(mut it) = app.input_events_iter() {
            while it.next(|ev| input(&mut pad, ev)) {}
        }
        let Some(link) = LINK.get() else { continue };
        let now = filtered(pad.pad());
        *link.pad.lock().unwrap() = now;
        if let Some(lines) = link.text.lock().unwrap().take() {
            shown = Some((lines, false));
            redraw = true;
        }
        if let Some((_, armed)) = shown.as_mut() {
            // B or Back.
            let held = now.buttons & 0x2020 != 0;
            if *armed && held {
                close();
            }
            *armed |= !held;
        }
        let fresh = link.frame.lock().unwrap().take();
        if let Some(img) = fresh.filter(|_| shown.is_none()) {
            last = Some(img);
            redraw = true;
        }
        if let (true, Some(w)) = (redraw, window.as_mut()) {
            draw(w, shown.as_ref().map(|s| s.0.as_slice()), last.as_ref());
        }
    }
}
