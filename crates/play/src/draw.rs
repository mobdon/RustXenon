//! The launcher screen, drawn with the CPU into 0RGB pixels of any size.
//!
//! The design is laid out on a 1280x720 board, scaled to fit the window and
//! centred; the background fills the whole window. Drawing is split in two:
//! [`backdrop`] (background, title, tabs, rusted panels, footer: slow, only
//! redrawn when the size or page changes) and [`overlay`] (rows, details,
//! notes: every frame). [`render`] does both.

use crate::settings::{Settings, FIELDS};
use crate::{font, Game};
use std::path::Path;

/// The design board.
pub const BOARD_W: f32 = 1280.0;
pub const BOARD_H: f32 = 720.0;
/// Rows the list shows at once.
pub const ROWS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Games,
    Settings,
}

/// Everything the screen shows.
pub struct View<'a> {
    pub page: Page,
    pub games: &'a [Game],
    pub settings: &'a Settings,
    /// Selected row of the page's list.
    pub sel: usize,
    /// First row shown.
    pub scroll: usize,
    /// Title of the game running now.
    pub running: Option<&'a str>,
    /// A short message for the player, such as why a dropped folder is not a game.
    pub note: Option<&'a str>,
    pub home: &'a Path,
}

/// What a point of the window is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Tab(Page),
    /// A list row, counted from the first row shown.
    Row(usize),
}

// Layout on the board.
const M: f32 = 56.0;
const TAB_Y: f32 = 92.0;
const TAB_W: f32 = 200.0;
const TAB_H: f32 = 38.0;
const TAB_SLANT: f32 = 16.0;
const TAB_GAP: f32 = 14.0;
const PANEL_Y: f32 = 160.0;
const PANEL_H: f32 = 476.0;
const LIST_W: f32 = 736.0;
const DETAIL_X: f32 = 812.0;
const DETAIL_W: f32 = BOARD_W - M - DETAIL_X;
const FRAME: f32 = 16.0;
const ROW_Y: f32 = PANEL_Y + FRAME + 12.0;
const ROW_H: f32 = 60.0;
const ROW_X: f32 = M + FRAME + 10.0;
const ROW_W: f32 = LIST_W - 2.0 * FRAME - 34.0;
const FOOT_Y: f32 = 652.0;
const FOOT_H: f32 = 50.0;

const GREEN: u32 = 0x5ad41e;
const GREEN_HI: u32 = 0xc8ff96;
const BONE: u32 = 0xe8dcc4;
const DIM: u32 = 0x9a8670;
const INK: u32 = 0x061004;
const WARN: u32 = 0xff8a3c;
const RUST_DARK: u32 = 0x4a2410;
const RUST_MID: u32 = 0x7a3a12;
const RUST_HI: u32 = 0xb7541a;

/// Board scale and offset for a `w` x `h` window.
fn fit(w: usize, h: usize) -> (f32, f32, f32) {
    let k = (w as f32 / BOARD_W).min(h as f32 / BOARD_H).max(0.01);
    (
        k,
        (w as f32 - BOARD_W * k) / 2.0,
        (h as f32 - BOARD_H * k) / 2.0,
    )
}

/// What window pixel (`x`, `y`) is over.
pub fn hit(w: usize, h: usize, x: f32, y: f32) -> Option<Hit> {
    let (k, ox, oy) = fit(w, h);
    let (x, y) = ((x - ox) / k, (y - oy) / k);
    for (i, p) in [Page::Games, Page::Settings].into_iter().enumerate() {
        let tx = M + i as f32 * (TAB_W + TAB_GAP);
        if (TAB_Y..TAB_Y + TAB_H).contains(&y) && (tx..tx + TAB_W + TAB_SLANT).contains(&x) {
            return Some(Hit::Tab(p));
        }
    }
    let r = ((y - ROW_Y) / ROW_H).floor();
    ((ROW_X..ROW_X + ROW_W).contains(&x) && (0.0..ROWS as f32).contains(&r))
        .then_some(Hit::Row(r as usize))
}

/// Draws the whole screen.
pub fn render(px: &mut [u32], w: usize, h: usize, v: &View, t: f32) {
    px.copy_from_slice(&backdrop(w, h, v.page));
    overlay(px, w, h, v, t);
}

// ---------------------------------------------------------------- colour

type Rgb = [f32; 3];

fn rgb(c: u32) -> Rgb {
    [
        (c >> 16 & 255) as f32,
        (c >> 8 & 255) as f32,
        (c & 255) as f32,
    ]
}

fn pack(c: Rgb) -> u32 {
    let b = |v: f32| v.clamp(0.0, 255.0) as u32;
    b(c[0]) << 16 | b(c[1]) << 8 | b(c[2])
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

// ---------------------------------------------------------------- noise

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

/// Corroded metal at board point (`u`, `v`), screen pixel (`px`, `py`).
fn rust(u: f32, v: f32, px: i32, py: i32, seed: u32) -> Rgb {
    let n = 0.5 * vnoise(u / 52.0, v / 52.0, seed)
        + 0.32 * vnoise(u / 19.0, v / 19.0, seed + 1)
        + 0.18 * vnoise(u / 6.0, v / 6.0, seed + 2);
    let mut c = if n < 0.5 {
        mix(rgb(RUST_DARK), rgb(RUST_MID), smooth(0.2, 0.5, n))
    } else {
        mix(rgb(RUST_MID), rgb(RUST_HI), smooth(0.5, 0.78, n))
    };
    // Run-off streaks, long and thin downwards.
    let streak = vnoise(u / 4.0, v / 70.0, seed + 3);
    c = scale(c, 1.0 - 0.38 * smooth(0.55, 0.85, streak));
    // Pitting: dark holes with a raised bright rim.
    let p = vnoise(u / 5.5, v / 5.5, seed + 4);
    c = mix(
        c,
        scale(rgb(0xd9772e), 1.05),
        0.35 * smooth(0.68, 0.76, p) * (1.0 - smooth(0.78, 0.8, p)),
    );
    c = mix(c, rgb(0x170904), 0.85 * smooth(0.78, 0.86, p));
    // Speckle and grain, per screen pixel.
    let h = unit(hash(px, py, seed + 5));
    if h < 0.03 {
        c = mix(c, rgb(0xe08a3c), 0.55);
    } else if h > 0.975 {
        c = scale(c, 0.45);
    }
    scale(c, 0.9 + 0.2 * unit(hash(px, py, seed + 6)))
}

// ---------------------------------------------------------------- canvas

struct Cv<'a> {
    px: &'a mut [u32],
    w: i32,
    h: i32,
    k: f32,
    ox: f32,
    oy: f32,
}

// Drawing calls take a position, a size, a colour and a strength.
#[allow(clippy::too_many_arguments)]
impl Cv<'_> {
    fn new(px: &mut [u32], w: usize, h: usize) -> Cv<'_> {
        let (k, ox, oy) = fit(w, h);
        Cv {
            px,
            w: w as i32,
            h: h as i32,
            k,
            ox,
            oy,
        }
    }

    fn sx(&self, x: f32) -> i32 {
        (self.ox + x * self.k).round() as i32
    }

    fn sy(&self, y: f32) -> i32 {
        (self.oy + y * self.k).round() as i32
    }

    /// Screen pixel back to board units.
    fn board(&self, x: i32, y: i32) -> (f32, f32) {
        (
            (x as f32 + 0.5 - self.ox) / self.k,
            (y as f32 + 0.5 - self.oy) / self.k,
        )
    }

    fn get(&self, x: i32, y: i32) -> Rgb {
        rgb(self.px[(y * self.w + x) as usize])
    }

    fn set(&mut self, x: i32, y: i32, c: Rgb) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.px[(y * self.w + x) as usize] = pack(c);
        }
    }

    fn blend(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h && a > 0.0 {
            let old = self.get(x, y);
            self.set(x, y, mix(old, c, a.min(1.0)));
        }
    }

    fn add(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h && a > 0.0 {
            let o = self.get(x, y);
            self.set(x, y, [o[0] + c[0] * a, o[1] + c[1] * a, o[2] + c[2] * a]);
        }
    }

    /// Screen pixel rectangle of a board rectangle, clipped.
    fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> (i32, i32, i32, i32) {
        (
            self.sx(x).max(0),
            self.sy(y).max(0),
            self.sx(x + w).min(self.w),
            self.sy(y + h).min(self.h),
        )
    }

    fn fill(&mut self, x: f32, y: f32, w: f32, h: f32, c: u32, a: f32) {
        let (x0, y0, x1, y1) = self.rect(x, y, w, h);
        let c = rgb(c);
        for py in y0..y1 {
            for px in x0..x1 {
                self.blend(px, py, c, a);
            }
        }
    }

    /// Screen size of one font pixel at board size `s`.
    fn font_px(&self, s: f32) -> i32 {
        (s * self.k).round().max(1.0) as i32
    }

    /// Width of `s` in board units at font size `size`.
    fn text_w(&self, size: f32, s: &str) -> f32 {
        font::width(s) as f32 * self.font_px(size) as f32 / self.k
    }

    fn text(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str) {
        let (p, x0, y0) = (self.font_px(size), self.sx(x), self.sy(y));
        let c = rgb(c);
        font::each(s, |fx, fy| {
            for dy in 0..p {
                for dx in 0..p {
                    self.set(x0 + fx * p + dx, y0 + fy * p + dy, c);
                }
            }
        });
    }

    fn text_right(&mut self, right: f32, y: f32, size: f32, c: u32, s: &str) {
        let w = self.text_w(size, s);
        self.text(right - w, y, size, c, s);
    }

    /// A soft halo around `s`: its pixels blurred and added in `c`.
    fn text_glow(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str, radius: f32, amount: f32) {
        let p = self.font_px(size);
        let r = (radius * self.k).max(1.0) as i32;
        let pad = r * 3;
        let (tw, th) = (font::width(s) * p, 7 * p);
        let (mw, mh) = (tw + 2 * pad, th + 2 * pad);
        let mut m = vec![0f32; (mw * mh) as usize];
        font::each(s, |fx, fy| {
            for dy in 0..p {
                for dx in 0..p {
                    m[((pad + fy * p + dy) * mw + pad + fx * p + dx) as usize] = 1.0;
                }
            }
        });
        blur(&mut m, mw as usize, mh as usize, r as usize);
        let (x0, y0, c) = (self.sx(x) - pad, self.sy(y) - pad, rgb(c));
        for my in 0..mh {
            for mx in 0..mw {
                let a = m[(my * mw + mx) as usize] * amount;
                self.add(x0 + mx, y0 + my, c, a);
            }
        }
    }

    /// A soft halo around a board rectangle.
    fn rect_glow(&mut self, x: f32, y: f32, w: f32, h: f32, radius: f32, c: u32, amount: f32) {
        let (x0, y0, x1, y1) =
            self.rect(x - radius, y - radius, w + 2.0 * radius, h + 2.0 * radius);
        let c = rgb(c);
        for py in y0..y1 {
            for px in x0..x1 {
                let (u, v) = self.board(px, py);
                let dx = (x - u).max(u - (x + w)).max(0.0);
                let dy = (y - v).max(v - (y + h)).max(0.0);
                let d = (dx * dx + dy * dy).sqrt() / radius;
                if d < 1.0 && (dx > 0.0 || dy > 0.0) {
                    self.add(px, py, c, (1.0 - d).powi(2) * amount);
                }
            }
        }
    }

    /// A round button face with a letter, centred on (`cx`, `cy`).
    fn button(&mut self, cx: f32, cy: f32, r: f32, c: u32, letter: &str) {
        let (x0, y0, x1, y1) = self.rect(cx - r - 1.0, cy - r - 1.0, 2.0 * r + 2.0, 2.0 * r + 2.0);
        let base = rgb(c);
        for py in y0..y1 {
            for px in x0..x1 {
                let (u, v) = self.board(px, py);
                let (dx, dy) = ((u - cx) / r, (v - cy) / r);
                let d = (dx * dx + dy * dy).sqrt();
                let cover = ((1.0 - d) * r * self.k + 0.5).clamp(0.0, 1.0);
                let shade = 1.15 - 0.45 * (dy + 1.0) / 2.0;
                let ring = smooth(0.78, 0.95, d) * 0.35;
                self.blend(px, py, scale(base, shade * (1.0 - ring)), cover);
            }
        }
        let tw = self.text_w(2.0, letter);
        let th = 7.0 * self.font_px(2.0) as f32 / self.k;
        self.text(cx - tw / 2.0, cy - th / 2.0, 2.0, INK, letter);
    }

    /// A keyboard key cap with a label; returns its width.
    fn keycap(&mut self, x: f32, cy: f32, label: &str) -> f32 {
        let w = self.text_w(2.0, label) + 14.0;
        let (y, h) = (cy - 12.0, 24.0);
        self.fill(x, y, w, h, 0x1a120c, 0.9);
        self.fill(x, y, w, 1.5, BONE, 0.7);
        self.fill(x, y + h - 2.0, w, 2.0, 0x000000, 0.6);
        self.fill(x, y, 1.5, h, BONE, 0.5);
        self.fill(x + w - 1.5, y, 1.5, h, BONE, 0.3);
        let th = 7.0 * self.font_px(2.0) as f32 / self.k;
        self.text(x + 7.0, cy - th / 2.0, 2.0, BONE, label);
        w
    }
}

/// Three box blurs (close to a Gaussian) of radius `r`, in place.
fn blur(m: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0f32; m.len()];
    let pass = |src: &[f32], dst: &mut [f32], n: usize, len: usize, stride: usize, step: usize| {
        let norm = 1.0 / (2 * r + 1) as f32;
        for line in 0..n {
            let at = |i: isize| -> f32 {
                if i < 0 || i >= len as isize {
                    0.0
                } else {
                    src[line * stride + i as usize * step]
                }
            };
            let mut sum: f32 = (-(r as isize)..=r as isize).map(at).sum();
            for i in 0..len {
                dst[line * stride + i * step] = sum * norm;
                sum += at(i as isize + r as isize + 1) - at(i as isize - r as isize);
            }
        }
    };
    for _ in 0..3 {
        pass(m, &mut tmp, h, w, w, 1);
        pass(&tmp, m, w, h, 1, w);
    }
}

// ---------------------------------------------------------------- pieces

fn background(cv: &mut Cv) {
    let (w, h) = (cv.w as f32, cv.h as f32);
    let (cx, cy) = (w * 0.5, h * 0.4);
    let reach = w.max(h) * 0.72;
    let base = rgb(0x050705);
    let green = rgb(GREEN);
    let line = (cv.k * 2.0).round().max(1.0) as i32;
    for py in 0..cv.h {
        for px in 0..cv.w {
            let d = ((px as f32 - cx).powi(2) + (py as f32 - cy).powi(2)).sqrt() / reach;
            let glow = (1.0 - d).max(0.0).powf(2.2) * 0.42;
            let mut c = mix(base, green, glow);
            c = scale(c, 1.0 - 0.55 * (d * d).min(1.0));
            // A faint dot grid and scan lines.
            let (u, v) = cv.board(px, py);
            if (u.rem_euclid(32.0) - 16.0).abs() < 1.0 && (v.rem_euclid(32.0) - 16.0).abs() < 1.0 {
                c = mix(c, green, 0.08);
            }
            if (py / line) % 2 == 1 {
                c = scale(c, 0.9);
            }
            cv.set(px, py, c);
        }
    }
}

/// A plate of rusted metal: a frame `frame` thick round a dark recessed
/// middle, two corners cut, rivets at the corners.
fn panel(cv: &mut Cv, x: f32, y: f32, w: f32, h: f32, frame: f32, seed: u32) {
    let cut = frame * 1.9;
    let inner_cut = cut - frame * 0.9;
    let (x0, y0, x1, y1) = cv.rect(x, y, w, h);
    let bevel = 2.2;
    for py in y0..y1 {
        for px in x0..x1 {
            let (bu, bv) = cv.board(px, py);
            let (u, v) = (bu - x, bv - y);
            let (ru, rv) = (w - u, h - v);
            // Cut corners top left and bottom right.
            let diag = (u + v).min(ru + rv);
            if diag < cut {
                continue;
            }
            let edge = u
                .min(v)
                .min(ru)
                .min(rv)
                .min((diag - cut) / std::f32::consts::SQRT_2);
            let mut c = rust(bu, bv, px, py, seed);
            let (iu, iv, iru, irv) = (u - frame, v - frame, ru - frame, rv - frame);
            let idiag = (iu + iv).min(iru + irv);
            let inside = iu > 0.0 && iv > 0.0 && iru > 0.0 && irv > 0.0 && idiag > inner_cut;
            if inside {
                // Recessed: dark oxide, deeper towards the rim.
                let ie = iu
                    .min(iv)
                    .min(iru)
                    .min(irv)
                    .min((idiag - inner_cut) / std::f32::consts::SQRT_2);
                c = scale(c, 0.2 + 0.08 * smooth(0.0, 14.0, ie));
                c = mix(c, rgb(0x0a0604), 0.35);
                if ie < bevel {
                    c = if iu.min(iv) < bevel {
                        scale(c, 0.4)
                    } else {
                        scale(c, 1.9)
                    };
                }
            } else if edge < bevel {
                // Raised outer edge: lit from the top left.
                c = if u.min(v) < bevel || (u + v) - cut < bevel * 1.5 {
                    mix(c, rgb(0xf0a060), 0.45)
                } else {
                    scale(c, 0.45)
                };
            }
            let a = (edge * cv.k).clamp(0.0, 1.0);
            cv.blend(px, py, c, a);
        }
    }
    let r = frame * 0.28;
    let m = frame * 0.5;
    let d = (cut + inner_cut + 2.0 * frame) / 4.0;
    for (cx, cy) in [
        (x + d, y + d),
        (x + w - m, y + m),
        (x + m, y + h - m),
        (x + w - d, y + h - d),
    ] {
        rivet(cv, cx, cy, r);
    }
}

fn rivet(cv: &mut Cv, cx: f32, cy: f32, r: f32) {
    let (x0, y0, x1, y1) = cv.rect(cx - r - 2.0, cy - r - 2.0, 2.0 * r + 4.0, 2.0 * r + 4.0);
    let light = [-0.45f32, -0.6, 0.66];
    for py in y0..y1 {
        for px in x0..x1 {
            let (u, v) = cv.board(px, py);
            // Shadow below right.
            let (sx, sy) = ((u - cx - 1.0) / r, (v - cy - 1.4) / r);
            let sd = (sx * sx + sy * sy).sqrt();
            cv.blend(px, py, rgb(0x000000), (1.0 - sd).clamp(0.0, 1.0) * 0.6);
            let (dx, dy) = ((u - cx) / r, (v - cy) / r);
            let d2 = dx * dx + dy * dy;
            if d2 >= 1.0 {
                continue;
            }
            let dz = (1.0 - d2).sqrt();
            let l = (dx * light[0] + dy * light[1] + dz * light[2]).max(0.0);
            let mut c = mix(rgb(0x2a1206), rgb(0xd88848), l);
            c = mix(c, rgb(0xfff0d8), l.powi(14) * 0.8);
            let cover = ((1.0 - d2.sqrt()) * r * cv.k + 0.5).clamp(0.0, 1.0);
            cv.blend(px, py, c, cover);
        }
    }
}

/// A slanted tab blade; returns nothing, the page decides the look.
fn tab(cv: &mut Cv, x: f32, label: &str, active: bool, seed: u32) {
    let (y, h) = (TAB_Y, TAB_H);
    if active {
        cv.rect_glow(x + TAB_SLANT / 2.0, y, TAB_W, h, 18.0, GREEN, 0.45);
    }
    let (x0, y0, x1, y1) = cv.rect(x, y, TAB_W + TAB_SLANT, h);
    for py in y0..y1 {
        for px in x0..x1 {
            let (u, v) = cv.board(px, py);
            let f = (v - y) / h;
            let left = x + TAB_SLANT * (1.0 - f);
            let d = (u - left).min(left + TAB_W - u);
            if d <= -0.5 / cv.k {
                continue;
            }
            let a = (d * cv.k + 0.5).clamp(0.0, 1.0);
            let c = if active {
                let mut c = mix(rgb(0x9cff5a), rgb(0x2f8a0c), f);
                if v - y < 2.0 {
                    c = rgb(0xe6ffcc);
                }
                c
            } else {
                let mut c = scale(rust(u, v, px, py, seed), 0.55);
                if v - y < 1.6 {
                    c = mix(c, rgb(0xf0a060), 0.5);
                } else if y + h - v < 1.6 {
                    c = scale(c, 0.4);
                }
                c
            };
            cv.blend(px, py, c, a);
        }
    }
    let tw = cv.text_w(3.0, label);
    let tx = x + TAB_SLANT / 2.0 + (TAB_W - tw) / 2.0;
    let ty = y + (h - 21.0) / 2.0;
    if active {
        cv.text(tx, ty, 3.0, INK, label);
    } else {
        cv.text(tx + 1.0, ty + 1.0, 3.0, 0x000000, label);
        cv.text(tx, ty, 3.0, BONE, label);
    }
}

/// Button hints, right-aligned in the footer.
fn hints(cv: &mut Cv, page: Page) {
    let first = if page == Page::Games {
        "START"
    } else {
        "CHANGE"
    };
    let list: [(&str, Option<u32>, &str); 5] = [
        ("A", Some(GREEN), first),
        ("B", Some(0xe0402a), "BACK"),
        ("Y", Some(0xf0b428), "REFRESH"),
        ("ENTER", None, first),
        ("ESC", None, "BACK"),
    ];
    let cy = FOOT_Y + FOOT_H / 2.0;
    let gap = 26.0;
    let mut total = 0.0;
    for (key, btn, label) in list {
        let kw = if btn.is_some() {
            24.0
        } else {
            cv.text_w(2.0, key) + 14.0
        };
        total += kw + 10.0 + cv.text_w(2.0, label) + gap;
    }
    let mut x = BOARD_W - M - 26.0 - (total - gap);
    for (key, btn, label) in list {
        let kw = match btn {
            Some(c) => {
                cv.button(x + 12.0, cy, 12.0, c, key);
                24.0
            }
            None => cv.keycap(x, cy, key),
        };
        x += kw + 10.0;
        cv.text(x, cy - 7.0, 2.0, BONE, label);
        x += cv.text_w(2.0, label) + gap;
    }
}

/// The slow part: everything that changes only with the size or page.
pub fn backdrop(w: usize, h: usize, page: Page) -> Vec<u32> {
    let mut px = vec![0u32; w * h];
    let mut cv = Cv::new(&mut px, w, h);
    background(&mut cv);

    // RUST in oxide orange, XENON in green, one halo behind both.
    let (tx, ty, ts) = (M, 26.0, 6.0);
    cv.text_glow(tx, ty, ts, GREEN, "RUSTXENON", 12.0, 1.9);
    cv.text(tx + 3.0, ty + 3.0, ts, 0x000000, "RUSTXENON");
    cv.text(tx, ty, ts, 0xe07a32, "RUST");
    // Four glyphs of six font pixels each.
    let rw = 24.0 * cv.font_px(ts) as f32 / cv.k;
    cv.text(tx + rw, ty, ts, GREEN_HI, "XENON");
    cv.text_right(BOARD_W - M, 32.0, 2.0, DIM, "RECOMPILED GAME LAUNCHER");

    for (i, (p, label)) in [(Page::Games, "GAMES"), (Page::Settings, "SETTINGS")]
        .into_iter()
        .enumerate()
    {
        tab(
            &mut cv,
            M + i as f32 * (TAB_W + TAB_GAP),
            label,
            p == page,
            40 + i as u32,
        );
    }
    // A glowing rule under the tabs.
    cv.fill(M, TAB_Y + TAB_H + 10.0, BOARD_W - 2.0 * M, 1.5, GREEN, 0.8);
    cv.rect_glow(
        M,
        TAB_Y + TAB_H + 10.0,
        BOARD_W - 2.0 * M,
        1.5,
        8.0,
        GREEN,
        0.35,
    );

    panel(&mut cv, M, PANEL_Y, LIST_W, PANEL_H, FRAME, 7);
    panel(&mut cv, DETAIL_X, PANEL_Y, DETAIL_W, PANEL_H, FRAME, 11);
    panel(&mut cv, M, FOOT_Y, BOARD_W - 2.0 * M, FOOT_H, 8.0, 23);
    hints(&mut cv, page);
    cv.text(
        M + 26.0,
        FOOT_Y + FOOT_H / 2.0 - 7.0,
        2.0,
        DIM,
        "DROP A GAME FOLDER HERE TO ADD IT",
    );
    px
}

// ---------------------------------------------------------------- overlay

/// `s` cut to `max` characters, keeping the end (paths end in what matters).
fn tail(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.into();
    }
    let keep: String = s.chars().skip(n + 2 - max).collect();
    format!("..{keep}")
}

/// `s` cut to `max` characters, keeping the start.
fn head(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.into();
    }
    let keep: String = s.chars().take(max.saturating_sub(2)).collect();
    format!("{keep}..")
}

/// Words of `s` in lines of at most `max` characters, at most `lines` lines.
fn wrap(s: &str, max: usize, lines: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let mut word: String = word.into();
        while word.chars().count() > max {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let first: String = word.chars().take(max).collect();
            word = word.chars().skip(max).collect();
            out.push(first);
        }
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.len() > lines {
        out.truncate(lines);
        let last = out.last_mut().unwrap();
        *last = head(&format!("{last}   "), max);
    }
    out
}

fn chars_in(cv: &Cv, width: f32, size: f32) -> usize {
    (width / (6.0 * cv.font_px(size) as f32 / cv.k))
        .floor()
        .max(1.0) as usize
}

/// The rows and the details panel.
pub fn overlay(px: &mut [u32], w: usize, h: usize, v: &View, t: f32) {
    let mut cv = Cv::new(px, w, h);
    let pulse = 0.5 + 0.5 * (t * 3.2).sin();

    // Status at the top right.
    let (status, sc) = match v.running {
        Some(_) => ("RUNNING", 0xffd040),
        None => ("READY", GREEN),
    };
    cv.text_right(BOARD_W - M, 56.0, 2.0, sc, status);
    let sw = cv.text_w(2.0, status);
    cv.button(BOARD_W - M - sw - 14.0, 63.0, 5.0, sc, "");
    let count = match v.page {
        Page::Games => format!(
            "{} GAME{}",
            v.games.len(),
            if v.games.len() == 1 { "" } else { "S" }
        ),
        Page::Settings => "PLAYER DEFAULTS".into(),
    };
    cv.text_right(BOARD_W - M - sw - 32.0, 56.0, 2.0, DIM, &count);

    let rows: Vec<(String, String, Option<&str>)> = match v.page {
        Page::Games => v
            .games
            .iter()
            .map(|g| {
                let tag = if v.running.is_some_and(|r| r == g.title) {
                    Some("RUNNING")
                } else if g.recent {
                    Some("RECENT")
                } else {
                    None
                };
                (g.title.clone(), g.game.clone(), tag)
            })
            .collect(),
        Page::Settings => (0..FIELDS.len())
            .map(|i| (FIELDS[i].label.into(), v.settings.show(i), None))
            .collect(),
    };

    if rows.is_empty() {
        empty(&mut cv, v.home);
    }
    for (i, (title, sub, tag)) in rows.iter().enumerate().skip(v.scroll).take(ROWS) {
        let y = ROW_Y + (i - v.scroll) as f32 * ROW_H;
        let sel = i == v.sel;
        let (rx, rh) = (ROW_X, ROW_H - 6.0);
        if sel {
            cv.rect_glow(rx, y, ROW_W, rh, 16.0, GREEN, 0.3 + 0.25 * pulse);
            let (x0, y0, x1, y1) = cv.rect(rx, y, ROW_W, rh);
            for py in y0..y1 {
                let f = (py - y0) as f32 / (y1 - y0).max(1) as f32;
                let c = mix(rgb(0x8ef04c), rgb(0x3c9e12), f);
                for px in x0..x1 {
                    cv.blend(px, py, c, 0.86 + 0.1 * pulse);
                }
            }
            cv.fill(rx, y, ROW_W, 2.0, 0xeaffd4, 0.9);
            cv.fill(rx, y, 6.0, rh, 0xeaffd4, 0.9);
        } else {
            cv.fill(rx, y + rh + 2.0, ROW_W, 1.0, BONE, 0.08);
        }
        let (tc, subc) = if sel { (INK, 0x1e4a0a) } else { (BONE, DIM) };
        cv.text(
            rx + 20.0,
            y + 8.0,
            2.0,
            if sel { 0x1e4a0a } else { 0x5a8a3a },
            &format!("{:02}", i + 1),
        );
        let tx = rx + 56.0;
        match v.page {
            Page::Games => {
                let room = ROW_W - 56.0 - 20.0 - tag.map_or(0.0, |_| 110.0);
                cv.text(tx, y + 8.0, 3.0, tc, &head(title, chars_in(&cv, room, 3.0)));
                cv.text(
                    tx,
                    y + 34.0,
                    2.0,
                    subc,
                    &tail(sub, chars_in(&cv, room, 2.0)),
                );
                if let Some(tag) = tag {
                    let tw = cv.text_w(2.0, tag);
                    let (bx, by) = (rx + ROW_W - 16.0 - tw - 12.0, y + 16.0);
                    let (bg, fg) = if sel {
                        (INK, GREEN_HI)
                    } else {
                        (0x1c2a10, GREEN)
                    };
                    cv.fill(bx, by, tw + 12.0, 22.0, bg, 0.9);
                    cv.text(bx + 6.0, by + 4.0, 2.0, fg, tag);
                }
            }
            Page::Settings => {
                cv.text(tx, y + 16.0, 3.0, tc, title);
                let val = if sel {
                    format!("< {sub} >")
                } else {
                    sub.clone()
                };
                cv.text_right(rx + ROW_W - 20.0, y + 16.0, 3.0, tc, &val);
            }
        }
    }
    // Scroll bar.
    if rows.len() > ROWS {
        let (bx, by, bh) = (ROW_X + ROW_W + 10.0, ROW_Y, ROWS as f32 * ROW_H - 6.0);
        cv.fill(bx, by, 4.0, bh, 0x000000, 0.5);
        let th = bh * ROWS as f32 / rows.len() as f32;
        let ty = by + (bh - th) * v.scroll as f32 / (rows.len() - ROWS) as f32;
        cv.fill(bx, ty, 4.0, th, GREEN, 0.9);
    }

    details(&mut cv, v, pulse);
}

fn empty(cv: &mut Cv, home: &Path) {
    let (x, mut y) = (ROW_X + 20.0, ROW_Y + 14.0);
    cv.text_glow(x, y, 4.0, GREEN, "NO GAMES YET", 6.0, 0.8);
    cv.text(x, y, 4.0, GREEN_HI, "NO GAMES YET");
    y += 56.0;
    let n = chars_in(cv, ROW_W - 40.0, 2.0);
    let games = home.join("games");
    let lines = [
        ("Put each built game folder in:", BONE),
        (
            &*tail(&format!("{}\\<sha256>\\", games.display()), n),
            GREEN,
        ),
        ("", BONE),
        ("A built game folder holds game.exe and run.toml.", BONE),
        ("", BONE),
        ("Or drop a built game folder on this window,", BONE),
        ("or on rustxenon-play.exe in Explorer.", BONE),
        ("", BONE),
        ("Press Y to look again.", DIM),
    ];
    for (s, c) in lines {
        cv.text(x, y, 2.0, c, s);
        y += 26.0;
    }
}

/// The right-hand panel: the selected row in full, notes, the prompt.
fn details(cv: &mut Cv, v: &View, pulse: f32) {
    let x = DETAIL_X + FRAME + 18.0;
    let iw = DETAIL_W - 2.0 * FRAME - 36.0;
    let mut y = PANEL_Y + FRAME + 18.0;
    let n2 = chars_in(cv, iw, 2.0);
    let n4 = chars_in(cv, iw, 4.0);
    let label = |cv: &mut Cv, y: &mut f32, s: &str| {
        cv.text(x, *y, 2.0, 0x5a8a3a, s);
        *y += 22.0;
    };
    let lines = |cv: &mut Cv, y: &mut f32, ls: Vec<String>, size: f32, c: u32, step: f32| {
        for l in ls {
            cv.text(x, *y, size, c, &l);
            *y += step;
        }
    };
    match v.page {
        Page::Games => match v.games.get(v.sel) {
            Some(g) => {
                label(cv, &mut y, "SELECTED");
                lines(cv, &mut y, wrap(&g.title, n4, 2), 4.0, BONE, 36.0);
                y += 10.0;
                for (name, value) in [
                    ("FOLDER", &g.dir.to_string_lossy().into_owned()),
                    ("EXECUTABLE", &g.exe),
                    ("GAME FILES", &g.game),
                ] {
                    label(cv, &mut y, name);
                    lines(cv, &mut y, vec![tail(value, n2)], 2.0, DIM, 26.0);
                }
            }
            None => {
                label(cv, &mut y, "SELECTED");
                cv.text(x, y, 3.0, DIM, "NOTHING");
            }
        },
        Page::Settings => {
            let f = &FIELDS[v.sel.min(FIELDS.len() - 1)];
            label(cv, &mut y, "SETTING");
            lines(cv, &mut y, wrap(f.label, n4, 2), 4.0, BONE, 36.0);
            y += 6.0;
            lines(cv, &mut y, wrap(f.help, n2, 4), 2.0, DIM, 24.0);
            y += 12.0;
            label(cv, &mut y, "SAVED IN");
            lines(
                cv,
                &mut y,
                vec![tail(&v.home.join("settings.toml").to_string_lossy(), n2)],
                2.0,
                DIM,
                26.0,
            );
            lines(
                cv,
                &mut y,
                wrap("A game's own run.toml overrides these.", n2, 2),
                2.0,
                DIM,
                24.0,
            );
        }
    }

    // From the bottom up: the prompt, then any note above it.
    let bottom = PANEL_Y + PANEL_H - FRAME - 18.0;
    let py = bottom - 28.0;
    if let Some(title) = v.running {
        let a = 0.6 + 0.4 * pulse;
        cv.text_glow(x, py - 34.0, 4.0, 0xffd040, "RUNNING", 6.0, 0.5 * a);
        cv.text(x, py - 34.0, 4.0, 0xffe080, "RUNNING");
        cv.text(x, py + 4.0, 2.0, DIM, &head(title, n2));
        cv.text(x, py + 24.0, 2.0, DIM, "back here when it closes");
    } else if v.page == Page::Games && !v.games.is_empty() {
        cv.rect_glow(x, py - 6.0, iw, 40.0, 12.0, GREEN, 0.18 + 0.2 * pulse);
        cv.fill(x, py - 6.0, iw, 40.0, 0x0c1a06, 0.9);
        cv.fill(x, py - 6.0, iw, 1.5, GREEN, 0.9);
        cv.button(x + 26.0, py + 14.0, 13.0, GREEN, "A");
        cv.text(x + 52.0, py + 3.0, 3.0, GREEN_HI, "START GAME");
    } else if v.page == Page::Settings {
        cv.text(x, py + 6.0, 2.0, 0x5a8a3a, "LEFT / RIGHT TO CHANGE");
    }
    if let Some(note) = v.note {
        let ls = wrap(note, n2, 4);
        let mut ny = py - 64.0 - (ls.len() as f32 - 1.0) * 24.0;
        if v.running.is_some() {
            ny -= 20.0;
        }
        cv.fill(
            x - 8.0,
            ny - 10.0,
            3.0,
            ls.len() as f32 * 24.0 + 6.0,
            WARN,
            0.9,
        );
        for l in ls {
            cv.text(x + 4.0, ny, 2.0, WARN, &l);
            ny += 24.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(games: &'a [Game], s: &'a Settings) -> View<'a> {
        View {
            page: Page::Games,
            games,
            settings: s,
            sel: 0,
            scroll: 0,
            running: None,
            note: Some("not a built game folder"),
            home: Path::new("C:\\home"),
        }
    }

    #[test]
    fn renders_any_size() {
        let s = Settings::default();
        let g = Game {
            dir: "C:\\home\\games\\ab".into(),
            title: "Test Game One".into(),
            exe: "e".into(),
            game: "g".into(),
            update: None,
            data: None,
            recent: true,
        };
        let games = vec![g; 9];
        for (w, h) in [(1280, 720), (333, 777), (1, 1), (64, 16)] {
            let mut px = vec![0; w * h];
            let mut v = view(&games, &s);
            v.scroll = 2;
            v.sel = 3;
            render(&mut px, w, h, &v, 0.3);
            v.games = &[];
            v.page = Page::Settings;
            render(&mut px, w, h, &v, 0.0);
        }
        // Deterministic, and not blank.
        let mut a = vec![0; 320 * 180];
        let mut b = vec![1; 320 * 180];
        render(&mut a, 320, 180, &view(&games, &s), 1.0);
        render(&mut b, 320, 180, &view(&games, &s), 1.0);
        assert_eq!(a, b);
        assert!(a.iter().any(|&p| p != a[0]));
    }

    #[test]
    fn hits() {
        // Centre of the first row and of the settings tab, at 1x and 2x.
        for k in [1.0, 2.0] {
            let (w, h) = ((BOARD_W * k) as usize, (BOARD_H * k) as usize);
            let at = |x: f32, y: f32| hit(w, h, x * k, y * k);
            assert_eq!(at(ROW_X + 100.0, ROW_Y + 20.0), Some(Hit::Row(0)));
            assert_eq!(at(ROW_X + 100.0, ROW_Y + 2.5 * ROW_H), Some(Hit::Row(2)));
            assert_eq!(
                at(M + TAB_W + TAB_GAP + 50.0, TAB_Y + 10.0),
                Some(Hit::Tab(Page::Settings))
            );
            assert_eq!(at(5.0, 5.0), None);
        }
    }

    #[test]
    fn text_fitting() {
        assert_eq!(tail("abcdefgh", 5), "..fgh");
        assert_eq!(head("abcdefgh", 5), "abc..");
        assert_eq!(wrap("one two three", 7, 3), ["one two", "three"]);
        assert_eq!(wrap("abcdefghij", 4, 3), ["abcd", "efgh", "ij"]);
        assert_eq!(wrap("a b c d e f", 3, 2), ["a b", "c.."]);
    }
}
