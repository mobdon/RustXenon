//! Writes previews of the icon into a folder you name:
//! `cargo run -p xenon-icon --example preview -- <dir>` gives icon-256.png,
//! icon-48.png, icon-32.png, icon-16.png and icon-sheet.png (every size, at
//! 1x and enlarged, on dark and light backgrounds).

use std::path::Path;

fn save(path: &Path, w: u32, h: u32, rgba: &[u8]) {
    let f = std::fs::File::create(path).expect("create png");
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(rgba).unwrap();
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: preview <dir>");
    let dir = Path::new(&dir);
    for s in [256, 48, 32, 16] {
        save(
            &dir.join(format!("icon-{s}.png")),
            s,
            s,
            &xenon_icon::rgba(s),
        );
    }
    // Two rows per background: every size at 1x, then the small ones
    // enlarged with nearest-neighbour so single pixels show.
    let (w, h) = (1100u32, 2 * (256 + 16 + 8 * 32 + 32));
    let mut sheet = vec![0u8; (w * h * 4) as usize];
    for (row, bg) in [[18u8, 20, 16], [236, 232, 224]].into_iter().enumerate() {
        for y in row as u32 * (h / 2)..(row as u32 + 1) * (h / 2) {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                sheet[o..o + 4].copy_from_slice(&[bg[0], bg[1], bg[2], 255]);
            }
        }
    }
    let mut put = |x0: u32, y0: u32, s: u32, zoom: u32, bg: [u8; 3]| {
        let px = xenon_icon::rgba(s);
        for y in 0..s * zoom {
            for x in 0..s * zoom {
                let i = (((y / zoom) * s + x / zoom) * 4) as usize;
                let a = px[i + 3] as u32;
                let o = (((y0 + y) * w + x0 + x) * 4) as usize;
                for c in 0..3 {
                    sheet[o + c] = ((px[i + c] as u32 * a + bg[c] as u32 * (255 - a)) / 255) as u8;
                }
            }
        }
    };
    for (row, bg) in [[18u8, 20, 16], [236, 232, 224]].into_iter().enumerate() {
        let y0 = row as u32 * (h / 2);
        let mut x = 16;
        for s in xenon_icon::SIZES {
            put(x, y0 + 16, s, 1, bg);
            x += s + 16;
        }
        let mut x = 16;
        for s in [16, 24, 32] {
            put(x, y0 + 16 + 256 + 16, s, 8, bg);
            x += s * 8 + 16;
        }
    }
    save(&dir.join("icon-sheet.png"), w, h, &sheet);
}
