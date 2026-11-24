//! `xenon-compare`: compares two frames.
//!
//! - [`diff`] counts pixels whose colour differs by more than a per-channel
//!   tolerance and draws a difference image.
//! - [`side_by_side`] lays images out in a row (expected, actual, diff).
//! - [`save_blink`] writes an animated PNG that flips between two images, so
//!   a small shift shows up as flicker.
//!
//! Pure logic and PNG files only: no window, sound or GPU.
//!
//! Alpha is ignored when comparing: frames read back from a GPU often carry
//! an undefined alpha channel, and the screen never shows it.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// An RGBA8 image, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Image {
    /// A `width` x `height` image filled with one colour.
    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let n = width as usize * height as usize;
        Image {
            width,
            height,
            rgba: rgba.repeat(n),
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.rgba[i..i + 4].try_into().unwrap()
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, p: [u8; 4]) {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.rgba[i..i + 4].copy_from_slice(&p);
    }
}

/// How far an actual frame may be from the expected one and still pass.
///
/// Runs with several guest threads are not exactly repeatable, so a test
/// always needs some tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// A pixel is "bad" when any of R, G, B differs by more than this.
    pub channel: u8,
    /// The frame passes when at most this fraction (0 to 1) of pixels is bad.
    pub max_bad_fraction: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance {
            channel: 8,
            max_bad_fraction: 0.001,
        }
    }
}

/// Result of comparing two frames of the same size.
#[derive(Clone, Debug)]
pub struct Diff {
    pub bad_pixels: u64,
    pub total_pixels: u64,
    /// Largest per-channel difference seen anywhere.
    pub max_channel_delta: u8,
    /// Dim grey copy of the actual frame, with bad pixels in red.
    pub image: Image,
}

impl Diff {
    pub fn bad_fraction(&self) -> f64 {
        if self.total_pixels == 0 {
            0.0
        } else {
            self.bad_pixels as f64 / self.total_pixels as f64
        }
    }

    pub fn passes(&self, tol: &Tolerance) -> bool {
        self.bad_fraction() <= tol.max_bad_fraction
    }
}

/// Compares `actual` with `expected`. Fails if their sizes differ.
pub fn diff(expected: &Image, actual: &Image, channel: u8) -> Result<Diff, String> {
    if (expected.width, expected.height) != (actual.width, actual.height) {
        return Err(format!(
            "size differs: expected {}x{}, actual {}x{}",
            expected.width, expected.height, actual.width, actual.height
        ));
    }
    let mut out = Vec::with_capacity(actual.rgba.len());
    let mut bad = 0u64;
    let mut max_delta = 0u8;
    for (e, a) in expected
        .rgba
        .chunks_exact(4)
        .zip(actual.rgba.chunks_exact(4))
    {
        let d = (0..3).map(|c| e[c].abs_diff(a[c])).max().unwrap();
        max_delta = max_delta.max(d);
        if d > channel {
            bad += 1;
            out.extend_from_slice(&[128 + d / 2, 0, 0, 255]);
        } else {
            let grey = ((a[0] as u16 + a[1] as u16 + a[2] as u16) / 9) as u8;
            out.extend_from_slice(&[grey, grey, grey, 255]);
        }
    }
    Ok(Diff {
        bad_pixels: bad,
        total_pixels: expected.width as u64 * expected.height as u64,
        max_channel_delta: max_delta,
        image: Image {
            width: actual.width,
            height: actual.height,
            rgba: out,
        },
    })
}

/// Gap between images in [`side_by_side`], in pixels.
pub const GAP: u32 = 4;

/// Lays images out left to right on a dark background, tops aligned.
pub fn side_by_side(images: &[&Image]) -> Image {
    let width =
        images.iter().map(|i| i.width).sum::<u32>() + GAP * images.len().saturating_sub(1) as u32;
    let height = images.iter().map(|i| i.height).max().unwrap_or(0);
    let mut sheet = Image::filled(width, height, [32, 32, 32, 255]);
    let mut x0 = 0;
    for img in images {
        for y in 0..img.height {
            let src = (y * img.width * 4) as usize..((y + 1) * img.width * 4) as usize;
            let dst = ((y * width + x0) * 4) as usize;
            sheet.rgba[dst..dst + src.len()].copy_from_slice(&img.rgba[src]);
        }
        x0 += img.width + GAP;
    }
    sheet
}

/// Reads a PNG of any colour type as RGBA8.
pub fn load_png(path: &Path) -> Result<Image, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let file = File::open(path).map_err(|e| err(&e))?;
    let mut dec = png::Decoder::new(std::io::BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| err(&e))?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| err(&e))?;
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        png::ColorType::Indexed => return Err(err(&"indexed colour not expanded")),
    };
    Ok(Image {
        width: info.width,
        height: info.height,
        rgba,
    })
}

fn encoder(path: &Path, img: &Image) -> Result<png::Encoder<'static, BufWriter<File>>, String> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    Ok(enc)
}

/// Writes an RGBA8 PNG.
pub fn save_png(path: &Path, img: &Image) -> Result<(), String> {
    let err = |e: png::EncodingError| format!("{}: {e}", path.display());
    let mut w = encoder(path, img)?.write_header().map_err(err)?;
    w.write_image_data(&img.rgba).map_err(err)
}

/// Writes an animated PNG that shows `a`, then `b`, each for `ms`
/// milliseconds, looping forever. Browsers play it; a still viewer shows `a`.
pub fn save_blink(path: &Path, a: &Image, b: &Image, ms: u16) -> Result<(), String> {
    if (a.width, a.height) != (b.width, b.height) {
        return Err("blink needs two images of the same size".into());
    }
    let err = |e: png::EncodingError| format!("{}: {e}", path.display());
    let mut enc = encoder(path, a)?;
    enc.set_animated(2, 0).map_err(err)?;
    enc.set_frame_delay(ms, 1000).map_err(err)?;
    let mut w = enc.write_header().map_err(err)?;
    w.write_image_data(&a.rgba).map_err(err)?;
    w.write_image_data(&b.rgba).map_err(err)?;
    w.finish().map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Image {
        let mut img = Image::filled(w, h, [0, 0, 0, 255]);
        for y in 0..h {
            for x in 0..w {
                img.set_pixel(x, y, [(x * 8) as u8, (y * 8) as u8, 100, 255]);
            }
        }
        img
    }

    #[test]
    fn identical_frames_have_no_bad_pixels() {
        let a = gradient(16, 8);
        let d = diff(&a, &a, 0).unwrap();
        assert_eq!(
            (d.bad_pixels, d.total_pixels, d.max_channel_delta),
            (0, 128, 0)
        );
        assert!(d.passes(&Tolerance {
            channel: 0,
            max_bad_fraction: 0.0
        }));
    }

    #[test]
    fn channel_tolerance_and_fraction() {
        let a = gradient(10, 10);
        let mut b = a.clone();
        // Small change: within channel tolerance 8.
        b.set_pixel(0, 0, [8, 0, 100, 255]);
        // Two large changes.
        b.set_pixel(5, 5, [255, 0, 0, 255]);
        b.set_pixel(6, 5, [0, 255, 0, 255]);
        let d = diff(&a, &b, 8).unwrap();
        assert_eq!(d.bad_pixels, 2);
        assert_eq!(d.max_channel_delta, 255 - 40);
        assert!(d.passes(&Tolerance {
            channel: 8,
            max_bad_fraction: 0.02
        }));
        assert!(!d.passes(&Tolerance {
            channel: 8,
            max_bad_fraction: 0.01
        }));
        // Bad pixels are red in the diff image, good ones grey.
        assert_eq!(d.image.pixel(5, 5)[1..], [0, 0, 255]);
        let g = d.image.pixel(0, 0);
        assert_eq!((g[0], g[0]), (g[1], g[2]));
    }

    #[test]
    fn alpha_is_ignored() {
        let a = gradient(4, 4);
        let mut b = a.clone();
        b.set_pixel(1, 1, [8, 8, 100, 0]);
        assert_eq!(diff(&a, &b, 0).unwrap().bad_pixels, 0);
    }

    #[test]
    fn size_mismatch_is_an_error() {
        assert!(diff(&gradient(4, 4), &gradient(4, 5), 8).is_err());
    }

    #[test]
    fn side_by_side_places_images() {
        let a = Image::filled(3, 2, [255, 0, 0, 255]);
        let b = Image::filled(2, 4, [0, 255, 0, 255]);
        let s = side_by_side(&[&a, &b]);
        assert_eq!((s.width, s.height), (3 + GAP + 2, 4));
        assert_eq!(s.pixel(2, 1), [255, 0, 0, 255]);
        assert_eq!(s.pixel(0, 3), [32, 32, 32, 255]);
        assert_eq!(s.pixel(3 + GAP, 3), [0, 255, 0, 255]);
    }

    #[test]
    fn png_round_trip_and_blink() {
        let dir = std::env::temp_dir().join(format!("xenon-compare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = gradient(9, 7);
        let p = dir.join("a.png");
        save_png(&p, &a).unwrap();
        assert_eq!(load_png(&p).unwrap(), a);

        let b = Image::filled(9, 7, [1, 2, 3, 255]);
        let blink = dir.join("blink.png");
        save_blink(&blink, &a, &b, 500).unwrap();
        // A still decoder reads the first frame.
        assert_eq!(load_png(&blink).unwrap(), a);
        let dec = png::Decoder::new(File::open(&blink).unwrap());
        let info = dec.read_info().unwrap();
        assert_eq!(info.info().animation_control.unwrap().num_frames, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
