//! The programs' icon, drawn by code so no picture file lives in git: a disc
//! of corroded metal inside a glowing green ring, crossed by a bold X of two
//! bevelled steel bars, with "360" in green on a plate under it at larger
//! sizes. [`rgba`] draws it at any size, [`ico`] packs several sizes into an
//! .ico file and [`res`] wraps an .ico in a Win32 .res file that the MSVC
//! linker takes as an input.

use std::f32::consts::FRAC_1_SQRT_2;

/// The sizes Windows asks an executable for, smallest first.
pub const SIZES: [u32; 6] = [16, 24, 32, 48, 64, 256];

const GREEN: u32 = 0x5ad41e;
const GREEN_HI: u32 = 0xc8ff96;
const GREEN_DEEP: u32 = 0x173f06;
const RUST_DARK: u32 = 0x4a2410;
const RUST_MID: u32 = 0x7a3a12;
const RUST_HI: u32 = 0xb7541a;
const PIT: u32 = 0x170904;
const STEEL: u32 = 0xc4bcae;
const STEEL_DARK: u32 = 0x4a4440;
const BONE: u32 = 0xf2e8d4;
/// Light comes from above and a little to the left (y grows downwards).
const LIGHT: [f32; 3] = [-0.38, -0.7, 0.6];

type Rgb = [f32; 3];

fn rgb(c: u32) -> Rgb {
    let f = |s: u32| (c >> s & 255) as f32 / 255.0;
    [f(16), f(8), f(0)]
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn scale(a: Rgb, s: f32) -> Rgb {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Smooth value noise in 0..1.
fn vnoise(x: f32, y: f32, seed: u32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let (tx, ty) = (x - fx, y - fy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let n = |dx, dy| unit(hash(ix + dx, iy + dy, seed));
    let top = n(0, 0) + (n(1, 0) - n(0, 0)) * sx;
    let bot = n(0, 1) + (n(1, 1) - n(0, 1)) * sx;
    top + (bot - top) * sy
}

fn oxide(x: f32, y: f32, seed: u32) -> Rgb {
    let n = 0.55 * vnoise(x * 3.0 + 7.0, y * 3.0, seed)
        + 0.3 * vnoise(x * 8.0, y * 8.0, seed + 1)
        + 0.15 * vnoise(x * 21.0, y * 21.0, seed + 2);
    if n < 0.5 {
        mix(rgb(RUST_DARK), rgb(RUST_MID), smooth(0.25, 0.5, n))
    } else {
        mix(rgb(RUST_MID), rgb(RUST_HI), smooth(0.5, 0.72, n))
    }
}

/// Shapes per size: small icons get a thicker ring and bars and no fine detail.
#[derive(Clone, Copy)]
struct Look {
    /// Ring outer and inner radius; a soft shadow fills from `ro` to the edge.
    ro: f32,
    ri: f32,
    /// Bar half width and half length.
    w: f32,
    l: f32,
    /// Width of the bars' dark outline and green rim.
    outline: f32,
    rim: f32,
    pits: bool,
    /// The "360" plate under the X; its digits blur below 48 px.
    label: bool,
}

fn look(size: u32) -> Look {
    let px = 2.0 / size as f32;
    let mut k = match size {
        0..=24 => Look {
            ro: 0.97,
            ri: 0.72,
            // Short, narrow arms leave rust showing round their ends.
            w: 0.21,
            l: 0.62,
            outline: 0.0,
            rim: 0.0,
            pits: false,
            label: false,
        },
        // The owner's pick: the simple one-piece X at every larger size too.
        _ => Look {
            ro: 0.93,
            ri: 0.74,
            w: 0.19,
            l: 0.64,
            outline: 0.0,
            rim: 0.0,
            pits: true,
            label: size >= 48,
        },
    };
    // About a pixel each at small sizes, so the X keeps its shape.
    k.outline = (1.1 * px).max(0.035);
    k.rim = (0.8 * px).max(0.16 * k.w);
    k
}

/// Pitting: dark holes with a bright rim on the lit side.
fn pits(mut c: Rgb, x: f32, y: f32, cells: f32, seed: u32) -> Rgb {
    let (gx, gy) = (x * cells, y * cells);
    let (cx, cy) = (gx.floor(), gy.floor());
    let h = hash(cx as i32, cy as i32, seed);
    if unit(h) > 0.5 {
        return c;
    }
    let px = cx + 0.25 + 0.5 * unit(h.rotate_left(8));
    let py = cy + 0.25 + 0.5 * unit(h.rotate_left(16));
    let rad = 0.12 + 0.16 * unit(h.rotate_left(24));
    let (dx, dy) = (gx - px, gy - py);
    let d = (dx * dx + dy * dy).sqrt();
    // The rim catches the light on the side away from it.
    let rim = smooth(rad, rad * 1.15, d) * (1.0 - smooth(rad * 1.25, rad * 1.6, d));
    let away = ((dx * -LIGHT[0] + dy * -LIGHT[1]) / d.max(1e-4)).max(0.0);
    c = mix(c, rgb(0xe08a3c), 0.5 * rim * away);
    mix(c, rgb(PIT), 0.9 * (1.0 - smooth(rad * 0.6, rad, d)))
}

/// The disc at `q` = radius / inner ring radius.
fn disc(x: f32, y: f32, q: f32, k: &Look) -> Rgb {
    let mut c = oxide(x, y, 11);
    if k.pits {
        c = pits(c, x, y, 6.0, 21);
    }
    // A shallow dome, lit from the top left.
    let z = (1.0 - q * q).max(0.0).sqrt();
    let (nx, ny) = (x / k.ri * 0.6, y / k.ri * 0.6);
    let len = (nx * nx + ny * ny + z * z).sqrt().max(1e-4);
    let lit = (nx * LIGHT[0] + ny * LIGHT[1] + z * LIGHT[2]) / len;
    c = scale(c, 0.3 + 0.65 * lit.max(0.0));
    // A dark groove where the disc meets the ring.
    mix(c, rgb(PIT), 0.85 * smooth(0.88, 1.0, q))
}

/// Coverage of a bar along (`dx`, `dy`) at a point, as distances to its
/// edges: `None` outside, else (distance to nearest edge, that edge's outward
/// normal, along, across).
fn bar_at(x: f32, y: f32, dx: f32, dy: f32, k: &Look) -> Option<(f32, [f32; 2], f32, f32)> {
    let a = (x * dx + y * dy) * FRAC_1_SQRT_2;
    let n = (x * dy - y * dx) * FRAC_1_SQRT_2;
    let side = k.w - n.abs();
    // Ends cut on a slant so the tips point outwards.
    let end = k.l - (a.abs() + 0.45 * n.abs());
    if side < 0.0 || end < 0.0 {
        return None;
    }
    let s = FRAC_1_SQRT_2;
    let normal = if side < end {
        [n.signum() * dy * s, -n.signum() * dx * s]
    } else {
        [a.signum() * dx * s, a.signum() * dy * s]
    };
    Some((side.min(end), normal, a, n))
}

/// The colour of a bar where it covers the point.
/// `edges` false leaves out the bevel and rim (where a small X is one shape).
fn bar(x: f32, y: f32, dx: f32, dy: f32, top: bool, edges: bool, k: &Look) -> Option<Rgb> {
    let (dist, normal, a, n) = bar_at(x, y, dx, dy, k)?;
    // Brushed steel along the bar, bright so the X stands out from the rust.
    let brush = vnoise(a * 2.5, n * 40.0, if top { 31 } else { 41 });
    let mut c = mix(rgb(STEEL_DARK), rgb(STEEL), 0.6 + 0.4 * brush);
    // Bevelled edges: lit faces towards the light, dark ones away.
    let bev = 0.34 * k.w;
    if edges && dist < bev {
        let lit = (normal[0] * LIGHT[0] + normal[1] * LIGHT[1]) / 0.8;
        c = if lit > 0.0 {
            mix(c, rgb(BONE), 0.6 * lit.min(1.0))
        } else {
            scale(c, 1.0 + 0.65 * lit.max(-1.0))
        };
    }
    // A green rim, as if lit by the ring.
    let rim = if edges {
        1.0 - smooth(0.0, k.rim, dist)
    } else {
        0.0
    };
    c = mix(c, rgb(GREEN), 0.9 * rim);
    Some(c)
}

/// "360" in a 3x5 pixel font, one row of bits per line, high bit left.
const DIGITS: [[u8; 5]; 3] = [
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b101, 0b101, 0b111],
];
/// The plate's centre height and one font cell, in -1..1 units.
const PLATE_Y: f32 = 0.55;
const CELL: f32 = 0.04;

/// Whether font cell (`col`, `row`) of the "360" is lit.
fn lit_cell(col: i32, row: i32) -> bool {
    let (d, c) = (col / 4, col % 4);
    (0..3).contains(&d)
        && c < 3
        && (0..5).contains(&row)
        && DIGITS[d as usize][row as usize] >> (2 - c) & 1 == 1
}

/// The "360" plate over the disc: a dark recessed plate with glowing digits.
fn label(c: Rgb, x: f32, y: f32) -> Rgb {
    // 11 by 5 cells, plus a one-cell margin round them.
    let (w, h) = (6.5 * CELL, 3.5 * CELL);
    let (ux, uy) = (x, y - PLATE_Y);
    if ux.abs() > w || uy.abs() > h {
        return c;
    }
    let edge = (w - ux.abs()).min(h - uy.abs());
    let mut c = mix(c, rgb(PIT), 0.8);
    // A thin steel lip round the plate.
    c = mix(c, rgb(STEEL), 0.7 * (1.0 - smooth(0.0, 0.3 * CELL, edge)));
    let (gx, gy) = (ux / CELL + 5.5, uy / CELL + 2.5);
    let (col, row) = (gx.floor() as i32, gy.floor() as i32);
    // Soft glow from lit cells nearby, then the lit cells themselves.
    let mut glow: f32 = 0.0;
    for dc in -1..=1 {
        for dr in -1..=1 {
            if lit_cell(col + dc, row + dr) {
                let (cx, cy) = ((col + dc) as f32 + 0.5, (row + dr) as f32 + 0.5);
                let d = ((gx - cx).abs().max((gy - cy).abs()) - 0.5).max(0.0);
                glow = glow.max(1.0 - smooth(0.0, 0.9, d));
            }
        }
    }
    c = mix(c, rgb(GREEN_DEEP), 0.8 * glow);
    if lit_cell(col, row) {
        let (fx, fy) = (gx - col as f32, gy - row as f32);
        let core = 1.0 - (2.0 * fx - 1.0).abs().max((2.0 * fy - 1.0).abs());
        c = mix(rgb(GREEN), rgb(GREEN_HI), smooth(0.2, 0.9, core));
    }
    c
}

/// One premultiplied RGBA sample at (`x`, `y`) in -1..1.
fn sample(x: f32, y: f32, k: &Look) -> [f32; 4] {
    let over = |d: [f32; 4], c: Rgb, a: f32| {
        let m = 1.0 - a;
        [
            c[0] * a + d[0] * m,
            c[1] * a + d[1] * m,
            c[2] * a + d[2] * m,
            a + d[3] * m,
        ]
    };
    let r = (x * x + y * y).sqrt();
    let mut px = [0.0; 4];
    if r >= 1.0 {
        return px;
    }
    if r >= k.ro - 0.02 {
        // A soft green halo outside the ring.
        let t = (1.0 - r) / (1.0 - k.ro + 0.02);
        px = over(px, rgb(GREEN), 0.65 * t * t);
        if r >= k.ro {
            return px;
        }
    }
    if r >= k.ri {
        // The ring: a glowing tube, brightest along its middle and top left.
        let s = (r - k.ri) / (k.ro - k.ri);
        let tube = 1.0 - (2.0 * s - 1.0).powi(2);
        let lit = 0.5 + 0.5 * (x * LIGHT[0] + y * LIGHT[1]) / (r * 0.8);
        let c = mix(rgb(GREEN_DEEP), rgb(GREEN), smooth(0.0, 0.55, tube));
        let c = mix(
            c,
            rgb(GREEN_HI),
            smooth(0.5, 1.0, tube) * (0.35 + 0.65 * lit),
        );
        return over(px, c, 1.0);
    }
    let mut c = disc(x, y, r / k.ri, k);
    // Each bar over a dark outline and a shadow: lower bar first.
    let sh = 0.05;
    let wide = Look {
        w: k.w + k.outline,
        l: k.l + k.outline,
        ..*k
    };
    // The X is one shape: an edge across the crossing would cut the lower
    // bar in two.
    let under = bar_at(x, y, 1.0, -1.0, k).is_some();
    let edge = |mut c: Rgb, dy: f32| {
        if bar_at(x - sh, y - sh, 1.0, dy, k).is_some() {
            c = scale(c, 0.4);
        }
        if bar_at(x, y, 1.0, dy, &wide).is_some() {
            c = mix(c, rgb(PIT), 0.85);
        }
        c
    };
    c = edge(edge(c, -1.0), 1.0);
    for (dy, top) in [(-1.0, false), (1.0, true)] {
        if let Some(b) = bar(x, y, 1.0, dy, top, !(top && under), k) {
            c = b;
        }
    }
    if k.label {
        c = label(c, x, y);
    }
    over(px, c, 1.0)
}

/// The icon at `size` by `size` pixels, straight (not premultiplied) RGBA8,
/// rows top down. The same size always gives the same bytes.
pub fn rgba(size: u32) -> Vec<u8> {
    const SS: u32 = 4;
    let k = look(size);
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    let n = (SS * SS) as f32;
    for py in 0..size {
        for px in 0..size {
            let mut acc = [0.0f32; 4];
            for sy in 0..SS {
                for sx in 0..SS {
                    let f = |p: u32, s: u32| {
                        (p as f32 + (s as f32 + 0.5) / SS as f32) / size as f32 * 2.0 - 1.0
                    };
                    let s = sample(f(px, sx), f(py, sy), &k);
                    for i in 0..4 {
                        acc[i] += s[i] / n;
                    }
                }
            }
            let a = acc[3];
            let b = |v: f32| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            for v in &acc[..3] {
                out.push(if a > 0.0 { b(v / a) } else { 0 });
            }
            out.push(b(a));
        }
    }
    out
}

/// One .ico image: a 32-bit bottom-up DIB with its AND mask.
fn dib(size: u32) -> Vec<u8> {
    let px = rgba(size);
    let mask_row = size.div_ceil(32) * 4;
    let mut d = Vec::new();
    for v in [40, size, size * 2] {
        d.extend_from_slice(&v.to_le_bytes());
    }
    d.extend_from_slice(&1u16.to_le_bytes());
    d.extend_from_slice(&32u16.to_le_bytes());
    let len = size * size * 4 + mask_row * size;
    for v in [0, len, 0, 0, 0, 0] {
        d.extend_from_slice(&v.to_le_bytes());
    }
    for y in (0..size).rev() {
        for x in 0..size {
            let i = ((y * size + x) * 4) as usize;
            d.extend_from_slice(&[px[i + 2], px[i + 1], px[i], px[i + 3]]);
        }
    }
    for y in (0..size).rev() {
        let mut row = vec![0u8; mask_row as usize];
        for x in 0..size {
            if px[((y * size + x) * 4 + 3) as usize] == 0 {
                row[(x / 8) as usize] |= 0x80 >> (x % 8);
            }
        }
        d.extend_from_slice(&row);
    }
    d
}

/// An .ico file with one 32-bit image per size (each 1..=256).
pub fn ico(sizes: &[u32]) -> Vec<u8> {
    let mut f = Vec::new();
    for v in [0u16, 1, sizes.len() as u16] {
        f.extend_from_slice(&v.to_le_bytes());
    }
    let images: Vec<Vec<u8>> = sizes.iter().map(|&s| dib(s)).collect();
    let mut offset = 6 + 16 * sizes.len() as u32;
    for (&s, img) in sizes.iter().zip(&images) {
        assert!((1..=256).contains(&s), "icon size {s}");
        let b = (s % 256) as u8; // 256 is written as 0
        f.extend_from_slice(&[b, b, 0, 0]);
        f.extend_from_slice(&1u16.to_le_bytes());
        f.extend_from_slice(&32u16.to_le_bytes());
        f.extend_from_slice(&(img.len() as u32).to_le_bytes());
        f.extend_from_slice(&offset.to_le_bytes());
        offset += img.len() as u32;
    }
    for img in &images {
        f.extend_from_slice(img);
    }
    f
}

const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;
const LANG_EN_US: u16 = 0x0409;

/// One .res entry: a 32-byte header with ordinal type and name, then the data
/// padded to four bytes.
fn res_entry(out: &mut Vec<u8>, ty: u16, id: u16, flags: u16, lang: u16, data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    for v in [0xffff, ty, 0xffff, id] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // data version
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&lang.to_le_bytes());
    out.extend_from_slice(&[0; 8]); // version, characteristics
    out.extend_from_slice(data);
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

/// A Win32 .res file holding the images of `ico` (as made by [`ico`]) as
/// icons 1..=n and one icon group with id 1, which Explorer shows as the
/// executable's icon. Panics on a malformed .ico.
pub fn res(ico: &[u8]) -> Vec<u8> {
    let u16_at = |i: usize| u16::from_le_bytes([ico[i], ico[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([ico[i], ico[i + 1], ico[i + 2], ico[i + 3]]);
    assert!(u16_at(0) == 0 && u16_at(2) == 1, "not an .ico file");
    let n = u16_at(4) as usize;
    let mut out = Vec::new();
    res_entry(&mut out, 0, 0, 0, 0, &[]);
    let mut group = ico[..6].to_vec();
    for i in 0..n {
        let e = 6 + 16 * i;
        let (len, off) = (u32_at(e + 8) as usize, u32_at(e + 12) as usize);
        let id = i as u16 + 1;
        // MOVEABLE | DISCARDABLE, as the resource compiler writes them.
        res_entry(
            &mut out,
            RT_ICON,
            id,
            0x1010,
            LANG_EN_US,
            &ico[off..off + len],
        );
        group.extend_from_slice(&ico[e..e + 12]);
        group.extend_from_slice(&id.to_le_bytes());
    }
    res_entry(&mut out, RT_GROUP_ICON, 1, 0x1030, LANG_EN_US, &group);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(b: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([b[i], b[i + 1]])
    }
    fn u32_at(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
    }

    #[test]
    fn pixels_have_the_right_size_and_shape() {
        for s in SIZES {
            let p = rgba(s);
            assert_eq!(p.len(), (s * s * 4) as usize);
            let alpha = |x: u32, y: u32| p[((y * s + x) * 4 + 3) as usize];
            for (x, y) in [(0, 0), (s - 1, 0), (0, s - 1), (s - 1, s - 1)] {
                assert_eq!(alpha(x, y), 0, "corner of {s}");
            }
            assert_eq!(alpha(s / 2, s / 2), 255, "middle of {s}");
        }
    }

    #[test]
    fn pixels_are_deterministic() {
        assert_eq!(rgba(48), rgba(48));
    }

    #[test]
    fn ico_header_and_images() {
        let f = ico(&SIZES);
        assert_eq!((u16_at(&f, 0), u16_at(&f, 2)), (0, 1));
        assert_eq!(u16_at(&f, 4) as usize, SIZES.len());
        let mut end = 6 + 16 * SIZES.len();
        for (i, s) in SIZES.iter().enumerate() {
            let e = 6 + 16 * i;
            assert_eq!(f[e] as u32, s % 256);
            assert_eq!((u16_at(&f, e + 4), u16_at(&f, e + 6)), (1, 32));
            let (len, off) = (u32_at(&f, e + 8) as usize, u32_at(&f, e + 12) as usize);
            assert_eq!(off, end);
            assert_eq!(u32_at(&f, off), 40);
            assert_eq!((u32_at(&f, off + 4), u32_at(&f, off + 8)), (*s, s * 2));
            assert_eq!(len as u32, 40 + s * s * 4 + s.div_ceil(32) * 4 * s);
            end += len;
        }
        assert_eq!(end, f.len());
    }

    #[test]
    fn res_parses_back() {
        let f = ico(&[16, 32, 256]);
        let r = res(&f);
        let mut at = 0;
        let mut entries = Vec::new();
        while at < r.len() {
            assert_eq!(at % 4, 0);
            let (len, head) = (u32_at(&r, at) as usize, u32_at(&r, at + 4) as usize);
            assert_eq!(head, 32);
            assert_eq!((u16_at(&r, at + 8), u16_at(&r, at + 12)), (0xffff, 0xffff));
            let (ty, id) = (u16_at(&r, at + 10), u16_at(&r, at + 14));
            entries.push((ty, id, r[at + 32..at + 32 + len].to_vec()));
            at += (32 + len).div_ceil(4) * 4;
        }
        assert_eq!(at, r.len());
        assert_eq!((entries[0].0, entries[0].1, entries[0].2.len()), (0, 0, 0));
        let icons: Vec<_> = entries.iter().filter(|e| e.0 == RT_ICON).collect();
        assert_eq!(icons.len(), 3);
        let groups: Vec<_> = entries.iter().filter(|e| e.0 == RT_GROUP_ICON).collect();
        assert_eq!(groups.len(), 1);
        let (_, gid, g) = groups[0];
        assert_eq!(*gid, 1);
        assert_eq!((u16_at(g, 0), u16_at(g, 2), u16_at(g, 4)), (0, 1, 3));
        assert_eq!(g.len(), 6 + 14 * 3);
        for (i, icon) in icons.iter().enumerate() {
            let e = 6 + 14 * i;
            assert_eq!(u32_at(g, e + 8) as usize, icon.2.len());
            assert_eq!(u16_at(g, e + 12), icon.1);
            // The image is the .ico's image, byte for byte.
            let off = u32_at(&f, 6 + 16 * i + 12) as usize;
            assert_eq!(&f[off..off + icon.2.len()], &icon.2[..]);
        }
    }
}
