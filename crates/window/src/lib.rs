//! `xenon-window`: a window that shows the latest frame and turns a
//! controller plus the keyboard into one pad reading.
//!
//! On desktops the event loop runs on its own thread so the host's main
//! thread keeps its headless watch loop; an XInput controller is read on
//! Windows. On Android, [`android::run`] is the event pump on the
//! `android_main` thread and the pad comes from the activity's input events.
//! Frames are copied to the window with the CPU (bilinear scaling,
//! letterboxed, rows split over a few threads); no GPU work happens here.

use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;
use xenon_input::Pad;

/// The thread that pumps the window (there is only ever one).
static PUMP: OnceLock<ThreadId> = OnceLock::new();

/// Marks this thread as the one that pumps the window.
fn mark_pump_thread() {
    let _ = PUMP.set(std::thread::current().id());
}

/// Whether this thread pumps the window (and so must never park): the
/// desktop's window thread or Android's `android_main` thread. Android
/// starts that one unnamed, so a thread name cannot tell.
pub fn is_pump_thread() -> bool {
    PUMP.get() == Some(&std::thread::current().id())
}

#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "android")]
pub use android::spawn;
#[cfg(not(target_os = "android"))]
mod desktop;
#[cfg(not(target_os = "android"))]
pub use desktop::spawn;

/// One frame to show: rows top to bottom, 8-bit RGBA.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
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

/// Shared between the host and the window thread.
#[derive(Default)]
pub struct Link {
    frame: Mutex<Option<Image>>,
    text: Mutex<Option<Vec<String>>>,
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

/// How a pixel's colour sits in the `u32`s of a window buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Each platform uses one layout; the tests use both.
#[allow(dead_code)]
pub(crate) enum Px {
    /// 0x00RRGGBB (softbuffer).
    Xrgb,
    /// Bytes R, G, B, A in memory with A opaque: Android's RGBA_8888 and
    /// RGBX_8888 on a little-endian CPU.
    Rgbx,
}

impl Px {
    /// Where red, green and blue go, and the bits always set.
    fn layout(self) -> ([u32; 3], u32) {
        match self {
            Px::Xrgb => ([16, 8, 0], 0),
            Px::Rgbx => ([0, 8, 16], 0xFF00_0000),
        }
    }

    /// A 0x00RRGGBB colour in this layout.
    fn colour(self, c: u32) -> u32 {
        let (at, base) = self.layout();
        (0..3).fold(base, |v, k| v | (c >> (16 - 8 * k) & 0xFF) << at[k])
    }
}

/// Scales `img` to fit `w` x `h` keeping its shape, black bars around it.
/// `out` holds `h` rows of `stride` pixels laid out as `px` says. Bilinear: a
/// 720p frame stretched to a large window with nearest-neighbour scaling
/// showed every frame pixel as a hard block.
pub(crate) fn blit(img: &Image, out: &mut [u32], w: u32, h: u32, stride: usize, px: Px) {
    let (shift, base) = px.layout();
    out.fill(base);
    if img.width == 0
        || img.height == 0
        || img.rgba8.len() < (img.width * img.height * 4) as usize
        || stride < w as usize
        || out.len() < stride * h as usize
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
    let rows = &mut out[oy as usize * stride..][..fh as usize * stride];
    // A 4K window is 8 million pixels a frame: rows in a few threads.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let per = (fh as usize).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in rows.chunks_mut(per * stride).enumerate() {
            let (xs, ys) = (&xs, &ys[i * per..]);
            scope.spawn(move || {
                for (row, &(y0, y1, wy)) in chunk.chunks_mut(stride).zip(ys) {
                    let row = &mut row[ox as usize..][..fw as usize];
                    let (r0, r1) = (y0 * img.width as usize, y1 * img.width as usize);
                    for (px, &(x0, x1, wx)) in row.iter_mut().zip(xs) {
                        let at = |r: usize, x: usize| &img.rgba8[(r + x) * 4..][..3];
                        let (a, b, c, d) = (at(r0, x0), at(r0, x1), at(r1, x0), at(r1, x1));
                        let mut v = base;
                        for k in 0..3 {
                            let top = a[k] as u32 * (256 - wx) + b[k] as u32 * wx;
                            let bottom = c[k] as u32 * (256 - wx) + d[k] as u32 * wx;
                            let mix = (top * (256 - wy) + bottom * wy + (1 << 15)) >> 16;
                            v |= mix << shift[k];
                        }
                        *px = v;
                    }
                }
            });
        }
    });
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
/// `out` is laid out as for [`blit`].
pub(crate) fn text(lines: &[String], out: &mut [u32], w: u32, h: u32, stride: usize, px: Px) {
    use font8x8::legacy::BASIC_LEGACY;
    let (bg, fg) = (px.colour(BG), px.colour(FG));
    out.fill(bg);
    if stride < w as usize || out.len() < stride * h as usize {
        return;
    }
    let scale = if w >= 1600 { 2 } else { 1 };
    let (cw, ch) = (8 * scale, 10 * scale);
    let (cols, rows) = ((w.saturating_sub(2 * cw) / cw) as usize, (h / ch) as usize);
    if cols == 0 {
        return;
    }
    for (r, line) in tail(&lines.join("\n"), rows, cols).iter().enumerate() {
        for (c, chr) in line.chars().enumerate() {
            let glyph = BASIC_LEGACY
                .get(chr as usize)
                .unwrap_or(&BASIC_LEGACY[b'?' as usize]);
            let (x0, y0) = (cw + c as u32 * cw, r as u32 * ch + scale);
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in (0..8).filter(|gx| bits >> gx & 1 != 0) {
                    let x = (x0 + gx * scale) as usize;
                    for dy in 0..scale {
                        let y = (y0 + gy as u32 * scale + dy) as usize;
                        out[y * stride + x..][..scale as usize].fill(fg);
                    }
                }
            }
        }
    }
}

const BG: u32 = 0x0010_1410;
const FG: u32 = 0x0040_FF60;

/// A buffer size with the shape of a `ww` x `wh` window that a `fw` x `fh`
/// frame fills one way at its own size: the compositor then scales the
/// buffer to the window, and the frame is letterboxed inside it.
pub fn buffer_size(ww: u32, wh: u32, fw: u32, fh: u32) -> (u32, u32) {
    if ww == 0 || wh == 0 || fw == 0 || fh == 0 {
        return (0, 0);
    }
    let along = |a: u32, b: u32, c: u32| ((a as u64 * b as u64 + c as u64 / 2) / c as u64) as u32;
    if ww as u64 * fh as u64 > wh as u64 * fw as u64 {
        (along(fh, ww, wh).max(1), fh)
    } else {
        (fw, along(fw, wh, ww).max(1))
    }
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

#[cfg(test)]
mod tests {
    #[test]
    fn pump_thread_is_only_the_marked_one() {
        std::thread::spawn(|| {
            assert!(!super::is_pump_thread());
            super::mark_pump_thread();
            assert!(super::is_pump_thread());
        })
        .join()
        .unwrap();
        assert!(!super::is_pump_thread());
    }

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
            text(
                &["A\u{e9}xyz, far too long to fit".into()],
                &mut out,
                w,
                h,
                w as usize,
                Px::Xrgb,
            );
            assert!(out.iter().all(|&p| p == BG || p == FG), "{w}x{h}");
        }
        let mut out = vec![0; 40 * 25];
        text(&["A".into()], &mut out, 40, 25, 40, Px::Xrgb);
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
            rgba8: vec![255, 0, 0, 255, 0, 0, 255, 255],
        };
        let mut out = vec![7; 16];
        blit(&img, &mut out, 4, 4, 4, Px::Xrgb);
        let (r, b) = (0xFF0000, 0x0000FF);
        let (rb, br) = (0xBF0040, 0x4000BF);
        assert_eq!(out, [0, 0, 0, 0, r, rb, br, b, r, rb, br, b, 0, 0, 0, 0]);
        // At its own size the frame is copied exactly.
        let mut out = vec![7; 2];
        blit(&img, &mut out, 2, 1, 2, Px::Xrgb);
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
            blit(&big, &mut out, 2560, 1440, 2560, Px::Xrgb);
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
        blit(&one, &mut out, 3, 1, 3, Px::Xrgb);
        assert_eq!(out, [0, 0x00FF00, 0]);
        // A short buffer draws nothing rather than panicking.
        let bad = Image {
            width: 4,
            height: 4,
            rgba8: vec![0; 4],
        };
        blit(&bad, &mut out, 3, 1, 3, Px::Xrgb);
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn rgbx_with_stride() {
        // Red then blue into a 2x1 window whose rows are 3 pixels apart:
        // bytes R, G, B, A in memory, the padding filled opaque black.
        let img = Image {
            width: 2,
            height: 1,
            rgba8: vec![255, 0, 0, 255, 0, 0, 255, 255],
        };
        let mut out = vec![7; 6];
        blit(&img, &mut out, 2, 2, 3, Px::Rgbx);
        let bytes: Vec<u8> = out[..2].iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(bytes, [255, 0, 0, 255, 0, 0, 255, 255][..]);
        assert_eq!(out[2..], [0xFF00_0000; 4]);
        // Letterboxed: 2x1 into 2x3 puts the picture on the middle row.
        let mut out = vec![7; 9];
        blit(&img, &mut out, 2, 3, 3, Px::Rgbx);
        assert_eq!(out[3..5], [0xFF00_00FF, 0xFFFF_0000]);
        assert!(out
            .iter()
            .enumerate()
            .all(|(i, &p)| (3..5).contains(&i) || p == 0xFF00_0000));
        // A buffer shorter than stride x height draws nothing.
        let mut out = vec![7; 5];
        blit(&img, &mut out, 2, 2, 3, Px::Rgbx);
        assert_eq!(out, [0xFF00_0000; 5]);
        // Text colours move the same way.
        assert_eq!(Px::Rgbx.colour(0x0011_2233), 0xFF33_2211);
        assert_eq!(Px::Xrgb.colour(0x0011_2233), 0x0011_2233);
        let mut out = vec![7; 41 * 25];
        text(&["A".into()], &mut out, 40, 25, 41, Px::Rgbx);
        assert!(out.contains(&Px::Rgbx.colour(FG)));
        assert!(out
            .iter()
            .all(|&p| p == Px::Rgbx.colour(FG) || p == Px::Rgbx.colour(BG)));
    }

    #[test]
    fn buffer_has_the_window_shape() {
        // Wide phone screen: frame height kept, wider buffer (side bars).
        assert_eq!(buffer_size(2400, 1080, 1280, 720), (1600, 720));
        // Nearly square inner screen: frame width kept, taller buffer.
        assert_eq!(buffer_size(2176, 1812, 1280, 720), (1280, 1066));
        // Same shape: the frame's own size.
        assert_eq!(buffer_size(1920, 1080, 1280, 720), (1280, 720));
        // Tall portrait window.
        assert_eq!(buffer_size(1080, 2400, 1280, 720), (1280, 2844));
        assert_eq!(buffer_size(0, 1080, 1280, 720), (0, 0));
        assert_eq!(buffer_size(1920, 1080, 0, 720), (0, 0));
        assert_eq!(buffer_size(100_000, 1, 1, 1), (100_000, 1));
        assert_eq!(buffer_size(1, 100_000, 1, 1), (1, 100_000));
    }
}
