//! The launcher screen, drawn with the CPU into 0RGB pixels of any size.
//!
//! The screen is a stack of blades: the page shown is one tall glossy green
//! blade over rusted metal, the other pages are narrow rusted steel strips at
//! the left and right edges with their names written up the strip. Inside the
//! blade: the page title, a list of rows on the left and a details panel on
//! the right. A slim bar of button hints sits below.
//!
//! The design is laid out on a 1280x720 board, scaled to fit the window and
//! centred; the background fills the whole window. Drawing is split in two:
//! [`backdrop`] (background, title, blades, preview plate, footer: slow,
//! only redrawn when the size or page changes) and [`overlay`] (rows,
//! details, notes: every frame). [`render`] does both. The rusted thumbnail
//! tiles the overlay draws are made once per size and kept.

use crate::settings::{Settings, FIELDS};
use crate::{font, Game};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// The design board.
pub const BOARD_W: f32 = 1280.0;
pub const BOARD_H: f32 = 720.0;
/// Rows the list shows at once.
pub const ROWS: usize = 7;

/// The pages, in blade order from left to right.
pub const PAGES: [Page; 2] = [Page::Games, Page::Settings];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Games,
    Settings,
}

impl Page {
    /// The name written up its blade.
    fn name(self) -> &'static str {
        match self {
            Page::Games => "games",
            Page::Settings => "settings",
        }
    }
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
    /// The game running now.
    pub running: Option<&'a Game>,
    /// A short message for the player, such as why a dropped folder is not a game.
    pub note: Option<&'a str>,
    /// Why settings.toml cannot be read: then its values are not known.
    pub settings_err: Option<&'a str>,
    pub home: &'a Path,
}

/// What a point of the window is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// The side blade of another page.
    Tab(Page),
    /// A list row, counted from the first row shown.
    Row(usize),
    /// The Games page's prompt box under the details (start, or look again).
    Info,
    /// A settings row's `<` (-1) or `>` (+1) arrow.
    Step(usize, i32),
    /// A footer hint: 0 Refresh, 1 Back, 2 Start or Change, 3 Page, 4 Full
    /// screen.
    Hint(usize),
    /// The list's scroll bar (there only while the list is longer than
    /// [`ROWS`]): [`bar_grab`] and [`bar_scroll`] say where along it.
    Bar,
}

// Layout on the board.
const M: f32 = 26.0;
const BLADE_Y: f32 = 86.0;
const BLADE_H: f32 = 548.0;
const BLADE_R: f32 = 26.0;
/// Rusted rim round the active blade.
const RIM: f32 = 7.0;
/// Visible width of a side blade, and how far it tucks under its neighbour.
const STRIP_W: f32 = 58.0;
const STRIP_TUCK: f32 = 18.0;
const STRIP_INSET: f32 = 12.0;
const PAD: f32 = 40.0;
const LIST_Y: f32 = BLADE_Y + 96.0;
const ROW_H: f32 = 60.0;
const ROW_GAP: f32 = 6.0;
const ROW_R: f32 = 12.0;
const TILE: f32 = 44.0;
const PREVIEW_H: f32 = 164.0;
const INFO_H: f32 = 56.0;
const FOOT_Y: f32 = 648.0;
const FOOT_H: f32 = 48.0;

const GREEN: u32 = 0x5ad41e;
const GREEN_HI: u32 = 0xc8ff96;
const GREEN_DEEP: u32 = 0x173f06;
const PALE: u32 = 0xe6ffd2;
const SUB: u32 = 0xd2efbe;
const WHITE: u32 = 0xffffff;
const BLACK: u32 = 0x000000;
const BONE: u32 = 0xe8dcc4;
const DIM: u32 = 0x9a8670;
const INK: u32 = 0x0e2a06;
const WARN: u32 = 0xff8a3c;
const GOLD: u32 = 0xffd040;
const RUST_DARK: u32 = 0x4a2410;
const RUST_MID: u32 = 0x7a3a12;
const RUST_HI: u32 = 0xb7541a;

/// Where the active blade and its insides go for a page.
#[derive(Clone, Copy, Debug)]
struct Lay {
    bx: f32,
    bw: f32,
    list_x: f32,
    list_w: f32,
    /// The details panel.
    dx: f32,
    dw: f32,
}

fn index(page: Page) -> usize {
    PAGES.iter().position(|&p| p == page).unwrap_or(0)
}

fn lay(page: Page) -> Lay {
    let i = index(page);
    let after = PAGES.len() - 1 - i;
    let bx = M + i as f32 * STRIP_W;
    let bw = BOARD_W - M - after as f32 * STRIP_W - bx;
    let list_w = ((bw - 2.0 * PAD - 30.0) * 0.585).round();
    let dx = bx + PAD + list_w + 30.0;
    Lay {
        bx,
        bw,
        list_x: bx + PAD,
        list_w,
        dx,
        dw: bx + bw - PAD - dx,
    }
}

type Rect = (f32, f32, f32, f32);

/// Board rectangle of the side blade of `other` while `page` is shown,
/// including the part tucked under the active blade.
fn strip_rect(page: Page, other: Page) -> Option<Rect> {
    let (i, j) = (index(page), index(other));
    let (y, h, w) = (
        BLADE_Y + STRIP_INSET,
        BLADE_H - 2.0 * STRIP_INSET,
        STRIP_W + STRIP_TUCK,
    );
    match j.cmp(&i) {
        std::cmp::Ordering::Less => Some((M + j as f32 * STRIP_W, y, w, h)),
        std::cmp::Ordering::Greater => {
            let right = BOARD_W - M - (PAGES.len() - 1 - j) as f32 * STRIP_W;
            Some((right - w, y, w, h))
        }
        std::cmp::Ordering::Equal => None,
    }
}

/// Board rectangle of list row `r` (counted from the first row shown).
fn row_rect(l: &Lay, r: usize) -> Rect {
    (
        l.list_x,
        LIST_Y + r as f32 * ROW_H,
        l.list_w,
        ROW_H - ROW_GAP,
    )
}

/// Width of the settings value column, wide enough for `< 3840 >`.
const VAL_COL: f32 = 190.0;

/// A settings row's `<` and `>` arrows: what is drawn and what is clicked
/// (the arrow plus some room round it, the full row height).
fn arrow_rects(l: &Lay, r: usize) -> (Rect, Rect) {
    let (rx, ry, rw, rh) = row_rect(l, r);
    let aw = text_w(3.0, ">");
    let right = rx + rw - 20.0 - aw;
    let left = rx + rw - 20.0 - VAL_COL;
    let pad = 12.0;
    (
        (left - pad, ry, aw + 2.0 * pad, rh),
        (right - pad, ry, aw + 2.0 * pad, rh),
    )
}

fn preview_rect(l: &Lay) -> Rect {
    (l.dx, LIST_Y, l.dw, PREVIEW_H)
}

/// The scroll bar's track, as drawn.
fn bar_rect(l: &Lay) -> Rect {
    (
        l.list_x + l.list_w + 10.0,
        LIST_Y,
        6.0,
        ROWS as f32 * ROW_H - ROW_GAP,
    )
}

/// The scroll bar's thumb for `len` rows from `scroll`: at least 24 board
/// units tall, so a long list's thumb is not a dot.
fn thumb_rect(l: &Lay, len: usize, scroll: usize) -> Rect {
    let (bx, by, bw, bh) = bar_rect(l);
    let th = (bh * ROWS as f32 / len.max(1) as f32).clamp(24.0, bh);
    let ty = by + (bh - th) * scroll as f32 / len.saturating_sub(ROWS).max(1) as f32;
    (bx, ty, bw, th)
}

/// Where a press at window height `y` holds the thumb of `page`'s scroll
/// bar, from the thumb's top in board units: where it is on the thumb, or
/// its middle for a press on the track (the thumb jumps there).
pub fn bar_grab(w: usize, h: usize, page: Page, len: usize, scroll: usize, y: f32) -> f32 {
    let (k, _, oy) = fit(w, h);
    let (_, ty, _, th) = thumb_rect(&lay(page), len, scroll);
    let at = (y - oy) / k - ty;
    if (0.0..th).contains(&at) {
        at
    } else {
        th / 2.0
    }
}

/// The first row shown when the thumb is held at `grab` ([`bar_grab`]) with
/// the pointer at window height `y`, clamped (a drag can leave the bar).
pub fn bar_scroll(w: usize, h: usize, page: Page, len: usize, y: f32, grab: f32) -> usize {
    let (k, _, oy) = fit(w, h);
    let l = lay(page);
    let (_, by, _, bh) = bar_rect(&l);
    let th = thumb_rect(&l, len, 0).3;
    let along = (((y - oy) / k - grab - by) / (bh - th).max(1.0)).clamp(0.0, 1.0);
    (along * len.saturating_sub(ROWS) as f32).round() as usize
}

fn info_rect(l: &Lay) -> Rect {
    let bottom = LIST_Y + ROWS as f32 * ROW_H - ROW_GAP;
    (l.dx, bottom - INFO_H, l.dw, INFO_H)
}

/// Board scale and offset for a `w` x `h` window.
fn fit(w: usize, h: usize) -> (f32, f32, f32) {
    let k = (w as f32 / BOARD_W).min(h as f32 / BOARD_H).max(0.01);
    (
        k,
        (w as f32 - BOARD_W * k) / 2.0,
        (h as f32 - BOARD_H * k) / 2.0,
    )
}

/// What window pixel (`x`, `y`) is over while `page` is shown.
pub fn hit(w: usize, h: usize, page: Page, x: f32, y: f32) -> Option<Hit> {
    let (k, ox, oy) = fit(w, h);
    let (x, y) = ((x - ox) / k, (y - oy) / k);
    let inside = |(rx, ry, rw, rh): Rect| (rx..rx + rw).contains(&x) && (ry..ry + rh).contains(&y);
    let l = lay(page);
    // Only the part of a side blade that shows: the active blade covers its tuck.
    if !(l.bx..l.bx + l.bw).contains(&x) {
        if let Some(p) = PAGES
            .into_iter()
            .find(|&p| strip_rect(page, p).is_some_and(inside))
        {
            return Some(Hit::Tab(p));
        }
    }
    if page == Page::Games && inside(info_rect(&l)) {
        return Some(Hit::Info);
    }
    if let Some(i) = hint_rects(page).into_iter().position(inside) {
        return Some(Hit::Hint(i));
    }
    // The thin bar with room round it to click.
    let (bx, by, bw, bh) = bar_rect(&l);
    if inside((bx - 5.0, by, bw + 10.0, bh)) {
        return Some(Hit::Bar);
    }
    let r = (0..ROWS).find(|&r| inside(row_rect(&l, r)))?;
    if page == Page::Settings {
        let (down, up) = arrow_rects(&l, r);
        if inside(down) {
            return Some(Hit::Step(r, -1));
        }
        if inside(up) {
            return Some(Hit::Step(r, 1));
        }
    }
    Some(Hit::Row(r))
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

/// A colour ramp through evenly spaced stops, `t` in 0..1.
fn ramp(stops: &[u32], t: f32) -> Rgb {
    let n = stops.len() - 1;
    let f = t.clamp(0.0, 1.0) * n as f32;
    let i = (f.floor() as usize).min(n - 1);
    mix(rgb(stops[i]), rgb(stops[i + 1]), f - i as f32)
}

/// Signed distance from (`u`, `v`) to a rounded rectangle, negative inside.
fn rr(u: f32, v: f32, (x, y, w, h): Rect, r: f32) -> f32 {
    let r = r.min(w / 2.0).min(h / 2.0);
    let qx = (u - x - w / 2.0).abs() - (w / 2.0 - r);
    let qy = (v - y - h / 2.0).abs() - (h / 2.0 - r);
    let (ox, oy) = (qx.max(0.0), qy.max(0.0));
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
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

/// Rust textures by pixel size, scale and seed.
type Plates = HashMap<(i32, i32, u32, u32), Rc<Vec<u32>>>;

thread_local! {
    static PLATES: RefCell<Plates> = RefCell::new(HashMap::new());
}

/// A `w` x `h` pixel rust texture at scale `k`, made once and kept.
fn rust_tex(w: i32, h: i32, k: f32, seed: u32) -> Rc<Vec<u32>> {
    PLATES.with(|m| {
        let mut m = m.borrow_mut();
        // A resize leaves old sizes behind; drop them all now and then.
        if m.len() > 48 {
            m.clear();
        }
        m.entry((w, h, k.to_bits(), seed))
            .or_insert_with(|| {
                let mut t = Vec::with_capacity((w.max(0) * h.max(0)) as usize);
                for y in 0..h {
                    for x in 0..w {
                        let (u, v) = ((x as f32 + 0.5) / k, (y as f32 + 0.5) / k);
                        t.push(pack(rust(u, v, x, y, seed)));
                    }
                }
                Rc::new(t)
            })
            .clone()
    })
}

// ---------------------------------------------------------------- text

/// How much of each screen pixel a piece of text covers.
struct Mask {
    x0: i32,
    y0: i32,
    w: i32,
    h: i32,
    a: Vec<f32>,
}

/// The coverage of `s` drawn with its cap top left at board (`x`, `y`), one
/// font pixel `size` board units. `up` turns it a quarter turn anticlockwise
/// to read bottom to top, its first letter at the bottom: then (`x`, `y`) is
/// the top left of its box. The font bitmap is sampled smoothly and cut at
/// `thr`, so the letters are rounded and smooth at any size; a lower `thr`
/// is bolder.
fn text_mask(fit: (f32, f32, f32), x: f32, y: f32, size: f32, s: &str, up: bool) -> Mask {
    let (k, ox, oy) = fit;
    let bw = font::width(s).max(0) as usize;
    let bh = font::ROWS;
    let p = (size * k).max(0.05);
    if bw == 0 {
        return Mask {
            x0: 0,
            y0: 0,
            w: 0,
            h: 0,
            a: Vec::new(),
        };
    }
    // The bitmap with a blank border, so sampling needs no bounds checks.
    let bm = font::bitmap(s);
    let pw = bw + 2;
    let mut pad = vec![0f32; pw * (bh + 2)];
    for j in 0..bh {
        for i in 0..bw {
            pad[(j + 1) * pw + i + 1] = bm[j * bw + i] as f32;
        }
    }
    let (sx0, sy0) = (ox + x * k, oy + y * k);
    let (ew, eh) = if up {
        (bh as f32 * p, bw as f32 * p)
    } else {
        (bw as f32 * p, bh as f32 * p)
    };
    let (x0, y0) = (sx0.floor() as i32 - 1, sy0.floor() as i32 - 1);
    let (mw, mh) = (
        (sx0 + ew).ceil() as i32 + 1 - x0,
        (sy0 + eh).ceil() as i32 + 1 - y0,
    );
    // Each screen column or row: the padded font cell before it and the weight
    // of the one after, sampling at pixel centres.
    let axis = |start: i32, n: i32, origin: f32, flip: Option<f32>, cells: usize| {
        (0..n)
            .map(|i| {
                let s = (start + i) as f32 + 0.5 - origin;
                let f = flip.map_or(s, |e| e - s) / p - 0.5;
                let f = f.clamp(-1.0, cells as f32 - 0.001);
                let fi = f.floor();
                ((fi + 1.0) as usize, f - fi)
            })
            .collect::<Vec<_>>()
    };
    let (cols, rows) = if up {
        (axis(x0, mw, sx0, None, bh), axis(y0, mh, sy0, Some(eh), bw))
    } else {
        (axis(x0, mw, sx0, None, bw), axis(y0, mh, sy0, None, bh))
    };
    let thr = if p >= 4.0 { 0.44 } else { 0.5 };
    // About one screen pixel of edge ramp.
    let sharp = p.max(1.0) * 1.25;
    let mut a = vec![0f32; (mw * mh) as usize];
    let at = |i: usize, j: usize| pad[j * pw + i];
    for (r, &(rj, rt)) in rows.iter().enumerate() {
        let line = &mut a[r * mw as usize..(r + 1) * mw as usize];
        for (o, &(cj, ct)) in line.iter_mut().zip(&cols) {
            // Font column and row, whichever way the text runs.
            let ((i, tu), (j, tv)) = if up {
                ((rj, rt), (cj, ct))
            } else {
                ((cj, ct), (rj, rt))
            };
            let f = (at(i, j) * (1.0 - tu) + at(i + 1, j) * tu) * (1.0 - tv)
                + (at(i, j + 1) * (1.0 - tu) + at(i + 1, j + 1) * tu) * tv;
            *o = ((f - thr) * sharp + 0.5).clamp(0.0, 1.0);
        }
    }
    Mask {
        x0,
        y0,
        w: mw,
        h: mh,
        a,
    }
}

/// Width of `s` in board units at font size `size`.
fn text_w(size: f32, s: &str) -> f32 {
    font::width(s) as f32 * size
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

    fn fit(&self) -> (f32, f32, f32) {
        (self.k, self.ox, self.oy)
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

    fn blend(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if x < 0 || y < 0 || x >= self.w || y >= self.h || a <= 0.0 {
            return;
        }
        let p = &mut self.px[(y * self.w + x) as usize];
        *p = if a >= 1.0 {
            pack(c)
        } else {
            pack(mix(rgb(*p), c, a))
        };
    }

    fn add(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if x < 0 || y < 0 || x >= self.w || y >= self.h || a <= 0.0 {
            return;
        }
        let p = &mut self.px[(y * self.w + x) as usize];
        let o = rgb(*p);
        *p = pack([o[0] + c[0] * a, o[1] + c[1] * a, o[2] + c[2] * a]);
    }

    /// Screen pixel rectangle of a board rectangle, clipped.
    fn rect(&self, (x, y, w, h): Rect) -> (i32, i32, i32, i32) {
        (
            self.sx(x).max(0),
            self.sy(y).max(0),
            self.sx(x + w).min(self.w),
            self.sy(y + h).min(self.h),
        )
    }

    /// Shades a rounded rectangle: `f(u, v, depth, px, py)` gives the colour
    /// and strength at board point (`u`, `v`) relative to its top left,
    /// `depth` board units in from its edge. The edge is smoothed.
    fn shape(
        &mut self,
        (x, y, w, h): Rect,
        r: f32,
        mut f: impl FnMut(f32, f32, f32, i32, i32) -> (Rgb, f32),
    ) {
        let (x0, y0, x1, y1) = self.rect((x - 1.0, y - 1.0, w + 2.0, h + 2.0));
        for py in y0..y1 {
            for px in x0..x1 {
                let (u, v) = self.board(px, py);
                let d = rr(u, v, (x, y, w, h), r);
                let cover = (0.5 - d * self.k).clamp(0.0, 1.0);
                if cover <= 0.0 {
                    continue;
                }
                let (c, a) = f(u - x, v - y, -d, px, py);
                self.blend(px, py, c, a * cover);
            }
        }
    }

    /// A flat rounded rectangle, filled a span at a time.
    fn fill(&mut self, (x, y, w, h): Rect, r: f32, c: u32, a: f32) {
        let (c, k) = (rgb(c), self.k);
        let r = r.min(w / 2.0).min(h / 2.0);
        let (_, y0, _, y1) = self.rect((x, y - 1.0, w, h + 2.0));
        for py in y0..y1 {
            let (_, v) = self.board(0, py);
            let vy = ((v - y).min(y + h - v) * k + 0.5).clamp(0.0, 1.0);
            if vy <= 0.0 {
                continue;
            }
            // How far the corner rounding pulls this line in.
            let dv = (y + r - v).max(v - (y + h - r)).max(0.0).min(r);
            let inset = r - (r * r - dv * dv).sqrt();
            let (sl, sr) = (self.ox + (x + inset) * k, self.ox + (x + w - inset) * k);
            for px in (sl.floor() as i32).max(0)..(sr.ceil() as i32).min(self.w) {
                let cov = ((px as f32 + 1.0).min(sr) - (px as f32).max(sl)).clamp(0.0, 1.0);
                self.blend(px, py, c, a * cov * vy);
            }
        }
    }

    /// A soft shadow (black) or glow (added light) round a rounded rectangle.
    fn glow(&mut self, (x, y, w, h): Rect, r: f32, reach: f32, c: u32, a: f32) {
        let (x0, y0, x1, y1) = self.rect((x - reach, y - reach, w + 2.0 * reach, h + 2.0 * reach));
        let (c, add) = (rgb(c), c != BLACK);
        // Lines beside the straight sides only need the bands left and right.
        let (il, ir) = (self.sx(x + r).max(x0), self.sx(x + w - r).min(x1));
        for py in y0..y1 {
            let (_, v) = self.board(0, py);
            let side = v > y + r && v < y + h - r && il < ir;
            let mut px = x0;
            while px < x1 {
                if side && px == il {
                    px = ir;
                    continue;
                }
                let (u, _) = self.board(px, py);
                let d = rr(u, v, (x, y, w, h), r);
                if d > 0.0 && d < reach {
                    let s = (1.0 - d / reach).powi(2) * a;
                    if add {
                        self.add(px, py, c, s);
                    } else {
                        self.blend(px, py, c, s);
                    }
                }
                px += 1;
            }
        }
    }

    fn mask(&mut self, m: &Mask, dx: i32, dy: i32, c: Rgb, a: f32) {
        for my in 0..m.h {
            for mx in 0..m.w {
                let cov = m.a[(my * m.w + mx) as usize];
                if cov > 0.0 {
                    self.blend(m.x0 + mx + dx, m.y0 + my + dy, c, cov * a);
                }
            }
        }
    }

    fn text_ex(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str, a: f32, up: bool) {
        let m = text_mask(self.fit(), x, y, size, s, up);
        self.mask(&m, 0, 0, rgb(c), a);
    }

    fn text(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str) {
        self.text_ex(x, y, size, c, s, 1.0, false);
    }

    /// Text with a soft dark drop shadow, for light text on the gloss.
    fn text_sh(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str) {
        let m = text_mask(self.fit(), x, y, size, s, false);
        let o = (size * 0.45 * self.k).round().max(1.0) as i32;
        self.mask(&m, (o + 1) / 2, o, [0.0; 3], 0.4);
        self.mask(&m, 0, 0, rgb(c), 1.0);
    }

    fn text_right(&mut self, right: f32, y: f32, size: f32, c: u32, s: &str) {
        self.text(right - text_w(size, s), y, size, c, s);
    }

    /// A soft glow round `s`: its pixels blurred and added in `c`.
    fn text_glow(&mut self, x: f32, y: f32, size: f32, c: u32, s: &str, radius: f32, amount: f32) {
        let m = text_mask(self.fit(), x, y, size, s, false);
        let r = (radius * self.k).max(1.0) as i32;
        let pad = r * 3 + 2;
        let (mw, mh) = (m.w + 2 * pad, m.h + 2 * pad);
        let mut g = vec![0f32; (mw * mh) as usize];
        for my in 0..m.h {
            for mx in 0..m.w {
                g[((my + pad) * mw + mx + pad) as usize] = m.a[(my * m.w + mx) as usize];
            }
        }
        blur(&mut g, mw as usize, mh as usize, r as usize);
        let c = rgb(c);
        for gy in 0..mh {
            for gx in 0..mw {
                let a = g[(gy * mw + gx) as usize] * amount;
                self.add(m.x0 - pad + gx, m.y0 - pad + gy, c, a);
            }
        }
    }

    /// A glossy round controller button with a letter, centred on (`cx`, `cy`).
    fn button(&mut self, cx: f32, cy: f32, r: f32, c: u32, letter: &str) {
        let base = rgb(c);
        let disc = (cx - r, cy - r, 2.0 * r, 2.0 * r);
        self.glow((cx - r, cy - r + 1.5, 2.0 * r, 2.0 * r), r, 4.0, BLACK, 0.5);
        self.shape(disc, r, |u, v, depth, _, _| {
            let (dx, dy) = ((u - r) / r, (v - r) / r);
            let mut col = scale(base, 1.15 - 0.5 * (dy + 1.0) / 2.0);
            // Dark outer ring, a sheen cap on top.
            col = scale(col, 0.55 + 0.45 * smooth(0.0, 2.2, depth));
            let cap = smooth(0.05, -0.15, dy + 0.35 * dx * dx) * smooth(-1.0, -0.6, dy);
            (mix(col, [255.0; 3], cap * 0.55), 1.0)
        });
        let size = r * 0.12;
        let tw = text_w(size, letter);
        let (lx, ly) = (cx - tw / 2.0, cy - 3.5 * size);
        self.text_sh(lx, ly, size, WHITE, letter);
    }

    /// A rounded label pill; returns its width.
    fn pill(&mut self, x: f32, y: f32, h: f32, bg: u32, fg: u32, label: &str) -> f32 {
        let size = (h - 8.0) / 7.0;
        let w = pill_w(h, label);
        let base = rgb(bg);
        self.shape((x, y, w, h), h / 2.0, |_, v, depth, _, _| {
            let mut c = scale(base, 1.12 - 0.3 * v / h);
            c = mix(c, [255.0; 3], 0.25 * smooth(1.6, 0.4, depth));
            (c, 0.95)
        });
        self.text(x + h * 0.45, y + (h - 7.0 * size) / 2.0, size, fg, label);
        w
    }

    /// A bevelled rusted plate from a kept texture.
    fn plate(&mut self, (x, y, w, h): Rect, r: f32, seed: u32) {
        let (sx0, sy0) = (self.sx(x), self.sy(y));
        let (tw, th) = (self.sx(x + w) - sx0 + 2, self.sy(y + h) - sy0 + 2);
        let tex = rust_tex(tw, th, self.k, seed);
        self.shape((x, y, w, h), r, |u, v, depth, px, py| {
            let (i, j) = ((px - sx0).clamp(0, tw - 1), (py - sy0).clamp(0, th - 1));
            let mut c = rgb(tex[(j * tw + i) as usize]);
            if depth < 2.0 {
                c = if u.min(v) < 2.5 {
                    mix(c, rgb(0xf0a060), 0.45)
                } else {
                    scale(c, 0.45)
                };
            }
            (c, 1.0)
        });
    }
}

fn pill_w(h: f32, label: &str) -> f32 {
    text_w((h - 8.0) / 7.0, label) + h * 0.9
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

// ---------------------------------------------------------------- backdrop

fn background(cv: &mut Cv) {
    let (w, h) = (cv.w as f32, cv.h as f32);
    let (cx, cy) = (w * 0.5, h * 0.42);
    let reach = w.max(h) * 0.75;
    let base = rgb(0x050805);
    let green = rgb(GREEN);
    for py in 0..cv.h {
        for px in 0..cv.w {
            let d = ((px as f32 - cx).powi(2) + (py as f32 - cy).powi(2)).sqrt() / reach;
            let glow = (1.0 - d).max(0.0).powf(2.2) * 0.34;
            let mut c = mix(base, green, glow);
            c = scale(c, 1.0 - 0.55 * (d * d).min(1.0));
            // A faint dot grid.
            let (u, v) = cv.board(px, py);
            if (u.rem_euclid(32.0) - 16.0).abs() < 1.0 && (v.rem_euclid(32.0) - 16.0).abs() < 1.0 {
                c = mix(c, green, 0.07);
            }
            cv.blend(px, py, c, 1.0);
        }
    }
}

fn rivet(cv: &mut Cv, cx: f32, cy: f32, r: f32) {
    let (x0, y0, x1, y1) = cv.rect((cx - r - 2.0, cy - r - 2.0, 2.0 * r + 4.0, 2.0 * r + 4.0));
    let light = [-0.45f32, -0.6, 0.66];
    for py in y0..y1 {
        for px in x0..x1 {
            let (u, v) = cv.board(px, py);
            // Shadow below right.
            let (sx, sy) = ((u - cx - 1.0) / r, (v - cy - 1.4) / r);
            let sd = (sx * sx + sy * sy).sqrt();
            cv.blend(px, py, [0.0; 3], (1.0 - sd).clamp(0.0, 1.0) * 0.6);
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

/// A side blade: a curved strip of brushed steel, rusting from its edges,
/// riveted top and bottom, its page's name written up it. `left` says which
/// edge of the screen it is at; its name sits in the part that shows. With
/// no name it is a bare edge of the stack: no rivets, no label.
fn strip(cv: &mut Cv, (x, y, w, h): Rect, name: &str, left: bool, seed: u32) {
    let r = 16.0;
    cv.glow((x, y + 4.0, w, h), r, 14.0, BLACK, 0.55);
    let shown_x = if left { x } else { x + STRIP_TUCK };
    cv.shape((x, y, w, h), r, |u, v, depth, px, py| {
        let (bu, bv) = (u + x, v + y);
        // A cylinder lit from the upper left: one bright band down the strip.
        let s = (bu - shown_x) / STRIP_W;
        let band = (-((s - 0.32) / 0.2).powi(2)).exp();
        let mut c = scale(rgb(0xc9cdd2), 0.5 + 0.48 * band + 0.12 * (1.0 - v / h));
        // Brushed along its length.
        c = scale(c, 0.9 + 0.16 * vnoise(bu / 1.3, bv / 46.0, seed + 20));
        // Rust creeping in from the edges and in blotches.
        let n =
            0.6 * vnoise(bu / 22.0, bv / 30.0, seed) + 0.4 * vnoise(bu / 7.0, bv / 9.0, seed + 1);
        // Less in the middle, where the name is written.
        let mid = (-((s - 0.5) / 0.22).powi(2)).exp() * smooth(30.0, 50.0, v);
        let amt = smooth(
            0.54,
            0.72,
            n + 0.35 * (1.0 - smooth(0.0, 12.0, depth)) - 0.22 * mid,
        );
        if amt > 0.0 {
            c = mix(c, rust(bu, bv, px, py, seed), amt);
        }
        // A bevelled rim.
        if depth < 1.8 {
            c = if v < h / 2.0 {
                mix(c, [255.0; 3], 0.35)
            } else {
                scale(c, 0.55)
            };
        }
        (c, 1.0)
    });
    let cx = shown_x + STRIP_W / 2.0;
    if name.is_empty() {
        return;
    }
    for cy in [y + 20.0, y + h - 20.0] {
        rivet(cv, cx, cy, 4.2);
    }
    // The name reads up the strip from a little below its top, embossed.
    let size = 3.4;
    let (tx, ty) = (cx - 3.5 * size, y + 42.0);
    let m = text_mask(cv.fit(), tx, ty, size, name, true);
    cv.mask(&m, 1, 1, [255.0; 3], 0.55);
    cv.mask(&m, 0, 0, rgb(0x1c1e20), 0.92);
}

/// The active blade: green gloss over rusted metal, a rusted rim, rust
/// creeping in and pitting the paint near its edge.
fn blade(cv: &mut Cv, l: &Lay) {
    let (x, y, w, h) = (l.bx, BLADE_Y, l.bw, BLADE_H);
    cv.glow((x, y + 8.0, w, h), BLADE_R, 22.0, BLACK, 0.7);
    cv.shape((x, y, w, h), BLADE_R, |u, v, depth, px, py| {
        let (bu, bv) = (u + x, v + y);
        if depth < RIM {
            let mut c = rust(bu, bv, px, py, 3);
            if depth < 1.8 {
                // Raised outer edge, lit from the top.
                c = if v < h * 0.5 {
                    mix(c, rgb(0xf0a060), 0.5)
                } else {
                    scale(c, 0.5)
                };
            } else if depth > RIM - 1.6 {
                // Shadow where the paint starts.
                c = scale(c, 0.45);
            }
            return (c, 1.0);
        }
        let t = v / h;
        // The gloss: light at the top, saturated, dark at the bottom.
        let mut c = ramp(&[0xa8f46c, GREEN, 0x3a9a12, GREEN_DEEP], t.powf(0.85));
        // A soft sheen band across the top, curving down at the sides.
        let s = u / w - 0.5;
        let tb = t - 0.24 * s * s;
        let band = smooth(0.012, 0.05, tb) * (1.0 - smooth(0.11, 0.24, tb));
        c = mix(c, [255.0; 3], 0.5 * band);
        // Light pooling at the upper left, falling off to the right.
        c = scale(c, 1.06 - 0.14 * (u / w));
        // Faint brushed metal under the paint.
        c = scale(c, 0.95 + 0.07 * vnoise(bu / 70.0, bv / 1.6, 9));
        // A thin light rim inside the rust.
        let inward = depth - RIM;
        c = mix(c, rgb(GREEN_HI), 0.6 * (1.0 - smooth(0.0, 2.2, inward)));
        if inward < 40.0 {
            // Rust creeping in from the rim in blotches.
            let n = 0.65 * vnoise(bu / 16.0, bv / 16.0, 5) + 0.35 * vnoise(bu / 5.0, bv / 5.0, 6);
            let amt = smooth(0.62, 0.7, n + 0.4 - inward / 30.0);
            if amt > 0.0 {
                c = mix(c, scale(rust(bu, bv, px, py, 3), 0.85), amt);
            }
        }
        // Pits in the paint near the rim only, rarer further in.
        let near = 1.0 - smooth(4.0, 36.0, inward);
        let (pit, core) = pit(bu, bv, 0.22 * near, 8);
        if pit > 0.0 {
            c = mix(c, rgb(RUST_HI), pit * 0.9);
            c = mix(c, rgb(0x1a0a04), core * 0.85);
        }
        (c, 1.0)
    });
}

/// A round pit at board point (`u`, `v`): at most one in each 10-unit cell,
/// in `density` of the cells. Gives how much of the pit and of its dark
/// core the point is in.
fn pit(u: f32, v: f32, density: f32, seed: u32) -> (f32, f32) {
    const CELL: f32 = 10.0;
    let (cx, cy) = ((u / CELL).floor(), (v / CELL).floor());
    let (ix, iy) = (cx as i32, cy as i32);
    if unit(hash(ix, iy, seed)) >= density {
        return (0.0, 0.0);
    }
    let r = 0.7 + 1.8 * unit(hash(ix, iy, seed + 1));
    let px = cx * CELL + r + 0.5 + (CELL - 2.0 * r - 1.0) * unit(hash(ix, iy, seed + 2));
    let py = cy * CELL + r + 0.5 + (CELL - 2.0 * r - 1.0) * unit(hash(ix, iy, seed + 3));
    let d = ((u - px).powi(2) + (v - py).powi(2)).sqrt();
    (
        smooth(r + 0.5, r - 0.5, d),
        smooth(r * 0.55 + 0.4, r * 0.55 - 0.4, d),
    )
}

/// The preview box's rusted plate, riveted, darker in the middle.
fn preview_plate(cv: &mut Cv, l: &Lay, seed: u32) {
    let (x, y, w, h) = preview_rect(l);
    cv.glow((x, y + 3.0, w, h), 12.0, 12.0, BLACK, 0.5);
    cv.plate((x, y, w, h), 12.0, seed);
    cv.shape((x, y, w, h), 12.0, |u, v, _, _, _| {
        let (dx, dy) = (u / w - 0.5, v / h - 0.5);
        ([0.0; 3], 0.55 * (dx * dx * 2.4 + dy * dy * 2.6).min(1.0))
    });
    for (cx, cy) in [
        (x + 12.0, y + 12.0),
        (x + w - 12.0, y + 12.0),
        (x + 12.0, y + h - 12.0),
        (x + w - 12.0, y + h - 12.0),
    ] {
        rivet(cv, cx, cy, 4.0);
    }
}

const HINT_GAP: f32 = 24.0;

/// The footer's button hints (Y, B, A), each with the board x it starts at
/// and the keyboard key that does the same, and the x where the LB/RB page
/// hint starts.
#[allow(clippy::type_complexity)]
fn hints(
    page: Page,
) -> (
    [(f32, &'static str, u32, &'static str, &'static str); 3],
    f32,
) {
    let first = if page == Page::Games {
        "Start"
    } else {
        "Change"
    };
    let mut hints = [
        (0.0, "Y", 0xf0b428, "Refresh", "F5"),
        (0.0, "B", 0xe0402a, "Back", "Esc"),
        (0.0, "A", GREEN, first, "Enter"),
    ];
    let mut total = text_w(2.0, "Page") + pill_w(22.0, "LB") + pill_w(22.0, "RB") + 12.0;
    for (_, _, _, label, _) in hints {
        total += 32.0 + text_w(2.0, label) + HINT_GAP;
    }
    let mut hx = BOARD_W - M - 26.0 - total;
    for (x, _, _, label, _) in &mut hints {
        *x = hx;
        hx += 32.0 + text_w(2.0, label) + HINT_GAP;
    }
    (hints, hx)
}

/// Width of the LB/RB page hint.
fn page_hint_w() -> f32 {
    pill_w(22.0, "LB") + pill_w(22.0, "RB") + 12.0 + text_w(2.0, "Page")
}

/// The footer's full screen caption, under the drop line.
const FULL_HINT: &str = "F11 full screen";

/// What is clicked for each footer hint: Refresh, Back, Start, Page, and
/// the full screen caption (its line only: the drop line above is text).
fn hint_rects(page: Page) -> [Rect; 5] {
    let (hs, end) = hints(page);
    let r = |x: f32, w: f32| (x - 6.0, FOOT_Y, w + 12.0, FOOT_H);
    let h = |i: usize| r(hs[i].0, 32.0 + text_w(2.0, hs[i].3));
    let ky = FOOT_Y + FOOT_H / 2.0 + 5.0;
    let full = (
        M + 22.0,
        ky - 2.0,
        text_w(1.4, FULL_HINT) + 12.0,
        7.0 * 1.4 + 6.0,
    );
    [h(0), h(1), h(2), r(end, page_hint_w()), full]
}

/// The slim bar of button hints at the bottom.
fn footer(cv: &mut Cv, page: Page) {
    let (x, y, w, h) = (M, FOOT_Y, BOARD_W - 2.0 * M, FOOT_H);
    cv.glow((x, y + 3.0, w, h), h / 2.0, 10.0, BLACK, 0.5);
    cv.shape((x, y, w, h), h / 2.0, |u, v, depth, px, py| {
        let mut c = mix(rgb(0x2a2e30), rgb(0x0c0e0e), v / h);
        if depth < 2.5 {
            c = scale(rust(u + x, v + y, px, py, 31), 0.9);
        }
        c = mix(c, [255.0; 3], 0.12 * smooth(0.5, 0.0, v / h - 0.1));
        (c, 0.94)
    });
    // Each label with its keyboard key under it.
    let cy = y + h / 2.0;
    let (ly, ky) = (cy - 12.0, cy + 5.0);
    cv.text(x + 28.0, ly, 2.0, DIM, "Drop a game folder here to add it");
    cv.text(x + 28.0, ky, 1.4, DIM, FULL_HINT);
    let (hints, mut hx) = hints(page);
    for (hx, button, c, label, key) in hints {
        cv.button(hx + 13.0, cy, 13.0, c, button);
        cv.text(hx + 32.0, ly, 2.0, BONE, label);
        cv.text(hx + 32.0, ky, 1.4, DIM, key);
    }
    for b in ["LB", "RB"] {
        hx += cv.pill(hx, cy - 11.0, 22.0, 0x50565a, WHITE, b) + 4.0;
    }
    cv.text(hx + 4.0, ly, 2.0, BONE, "Page");
    cv.text(hx + 4.0, ky, 1.4, DIM, "Tab");
}

/// The header: the name, the 360 badge, the launcher's line.
fn header(cv: &mut Cv) {
    let (tx, ty, ts) = (M + 8.0, 22.0, 5.6);
    cv.text_glow(tx, ty, ts, GREEN, "RUSTXENON", 10.0, 1.4);
    cv.text_ex(tx + 2.5, ty + 3.0, ts, BLACK, "RUSTXENON", 0.7, false);
    // RUST in oxide, XENON in pale green.
    cv.text(tx, ty, ts, 0xe8823a, "RUST");
    cv.text(tx + text_w(ts, "RUST ") - 2.0 * ts, ty, ts, PALE, "XENON");
    // The badge: a glowing green pill with 360 in it.
    let bx = tx + text_w(ts, "RUSTXENON") + 18.0;
    let badge = (bx, ty - 0.5, 86.0, 38.0);
    let (_, by, bw, bh) = badge;
    cv.glow(badge, bh / 2.0, 22.0, GREEN, 0.55);
    cv.shape(badge, bh / 2.0, |_, v, depth, _, _| {
        let t = v / bh;
        let mut c = ramp(&[0xe4ffc8, 0x8af04a, GREEN, 0x2c8a0a], t);
        let sheen = smooth(0.0, 0.1, t) * (1.0 - smooth(0.38, 0.5, t));
        c = mix(c, [255.0; 3], 0.35 * sheen);
        if depth < 1.6 {
            c = mix(c, rgb(0x123a06), 0.6);
        }
        (c, 1.0)
    });
    let s = 3.6;
    let nx = bx + (bw - text_w(s, "360")) / 2.0;
    let ny = by + (bh - 7.0 * s) / 2.0;
    cv.text_ex(nx, ny + 1.5, s, WHITE, "360", 0.5, false);
    cv.text(nx, ny, s, INK, "360");
    cv.text_right(
        BOARD_W - M - 8.0,
        26.0,
        2.0,
        DIM,
        "recompiled game launcher",
    );
}

/// The slow part: everything that changes only with the size or page.
pub fn backdrop(w: usize, h: usize, page: Page) -> Vec<u32> {
    let mut px = vec![0u32; w * h];
    let mut cv = Cv::new(&mut px, w, h);
    background(&mut cv);
    header(&mut cv);
    let l = lay(page);
    let i = index(page);
    // Bare edges of more blades behind, at both ends, so they read as a stack.
    for (n, off) in [(0u32, 16.0), (1, 8.0)] {
        let inset = STRIP_INSET + 10.0 - off * 0.5;
        let (y, h, w) = (BLADE_Y + inset, BLADE_H - 2.0 * inset, STRIP_W);
        strip(&mut cv, (M - off, y, w, h), "", true, 60 + n);
        let x = BOARD_W - M + off - w - STRIP_TUCK;
        strip(&mut cv, (x, y, w + STRIP_TUCK, h), "", false, 62 + n);
    }
    // Outermost strips first, so each tucks under its inner neighbour.
    let mut order: Vec<usize> = (0..PAGES.len()).filter(|&j| j != i).collect();
    order.sort_by_key(|&j| std::cmp::Reverse(j.abs_diff(i)));
    for j in order {
        if let Some(r) = strip_rect(page, PAGES[j]) {
            strip(&mut cv, r, PAGES[j].name(), j < i, 40 + j as u32);
        }
    }
    blade(&mut cv, &l);
    let title = match page {
        Page::Games => "Select a Game",
        Page::Settings => "Player Settings",
    };
    let (tx, ty) = (l.bx + PAD + 4.0, BLADE_Y + 30.0);
    // White on the sheen needs a deep green edge and drop to stand out.
    let m = text_mask(cv.fit(), tx, ty, 5.0, title, false);
    let o = (2.0 * cv.k).round().max(1.0) as i32;
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        cv.mask(&m, dx, dy, rgb(GREEN_DEEP), 0.45);
    }
    cv.mask(&m, o, o + 1, rgb(GREEN_DEEP), 0.7);
    cv.mask(&m, 0, 0, [255.0; 3], 1.0);
    preview_plate(&mut cv, &l, 150 + i as u32);
    // The info box under the details.
    let (ix, iy, iw, ih) = info_rect(&l);
    cv.fill((ix, iy, iw, ih), 14.0, 0x9ae06a, 0.8);
    cv.shape(
        (ix + 1.5, iy + 1.5, iw - 3.0, ih - 3.0),
        12.5,
        |_, v, _, _, _| (mix(rgb(0x0c2004), rgb(0x020800), v / ih), 0.75),
    );
    footer(&mut cv, page);
    px
}

// ---------------------------------------------------------------- overlay

/// `s` cut from the end to fit `room` board units at `size`, ending "..".
fn head(s: &str, size: f32, room: f32) -> String {
    if text_w(size, s) <= room {
        return s.into();
    }
    let mut c: Vec<char> = s.chars().collect();
    while c.pop().is_some() {
        let t: String = c.iter().chain(['.', '.'].iter()).collect();
        if text_w(size, &t) <= room {
            return t;
        }
    }
    String::new()
}

/// `s` cut from the start to fit, starting ".." (paths end in what matters).
fn tail(s: &str, size: f32, room: f32) -> String {
    if text_w(size, s) <= room {
        return s.into();
    }
    let c: Vec<char> = s.chars().collect();
    for skip in 1..=c.len() {
        let t: String = ['.', '.'].iter().chain(&c[skip..]).collect();
        if text_w(size, &t) <= room {
            return t;
        }
    }
    String::new()
}

/// Words of `s` in lines that fit `room` at `size`, at most `lines` lines.
fn wrap(s: &str, size: f32, room: f32, lines: usize) -> Vec<String> {
    let fits = |t: &str| text_w(size, t) <= room;
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let mut word: String = word.into();
        while !fits(&word) && word.chars().count() > 1 {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let mut n = word.chars().count() - 1;
            while n > 1 && !fits(&word.chars().take(n).collect::<String>()) {
                n -= 1;
            }
            out.push(word.chars().take(n).collect());
            word = word.chars().skip(n).collect();
        }
        if !cur.is_empty() && !fits(&format!("{cur} {word}")) {
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
        let t = format!("{last}..");
        *last = if fits(&t) {
            t
        } else {
            head(&format!("{t}...."), size, room)
        };
    }
    out
}

/// Up to two capitals for a placeholder picture: the first letters of the
/// first two words.
fn initials(title: &str) -> String {
    let s: String = title
        .split_whitespace()
        .filter_map(|w| w.chars().find(|c| c.is_alphanumeric()))
        .take(2)
        .flat_map(char::to_uppercase)
        .collect();
    if s.is_empty() {
        "?".into()
    } else {
        s
    }
}

/// A small number from a title, to vary the rust of its tile.
fn seed_of(title: &str) -> u32 {
    100 + title
        .bytes()
        .fold(7u32, |h, b| h.wrapping_mul(31) ^ b as u32)
        % 6
}

/// One list row.
struct Row {
    title: String,
    sub: String,
    /// Label, background and text colour.
    tag: Option<(&'static str, u32, u32)>,
    tile: String,
    seed: u32,
}

fn rows(v: &View) -> Vec<Row> {
    match v.page {
        Page::Games => v
            .games
            .iter()
            .map(|g| Row {
                title: g.title.clone(),
                sub: g.game.clone(),
                // By folder: two games can have the same title.
                tag: if v.running.is_some_and(|r| r.dir == g.dir) {
                    Some(("running", GOLD, 0x2a1c00))
                } else if g.recent {
                    Some(("recent", 0x1c4a0c, GREEN_HI))
                } else {
                    None
                },
                tile: initials(&g.title),
                seed: seed_of(&g.title),
            })
            .collect(),
        Page::Settings => FIELDS
            .iter()
            .enumerate()
            .map(|(i, f)| Row {
                title: f.label.into(),
                sub: if v.settings_err.is_none() && !v.settings.valid(i) {
                    format!("[{}] {}: not valid", f.table, f.key)
                } else {
                    format!("[{}] {}", f.table, f.key)
                },
                tag: None,
                tile: format!("{:02}", i + 1),
                seed: 100 + i as u32 % 6,
            })
            .collect(),
    }
}

/// Settings field `i` as shown: "?" while settings.toml cannot be read.
fn setting(v: &View, i: usize) -> String {
    match v.settings_err {
        Some(_) => "?".into(),
        None => v.settings.show(i),
    }
}

/// Lines a note may take: it goes over the preview plate, clear of the text.
const NOTE_LINES: usize = 6;

/// The note box for a note of `lines` lines, at the top of the preview.
fn note_rect(l: &Lay, lines: usize) -> Rect {
    let (x, y, w, _) = preview_rect(l);
    (x, y, w, lines as f32 * 22.0 + 18.0)
}

/// The rows and the details panel.
pub fn overlay(px: &mut [u32], w: usize, h: usize, v: &View, t: f32) {
    let mut cv = Cv::new(px, w, h);
    let pulse = 0.5 + 0.5 * (t * 3.2).sin();
    let l = lay(v.page);

    // Status at the top right.
    let (status, sc) = match v.running {
        Some(_) => ("running", GOLD),
        None => ("ready", GREEN),
    };
    let right = BOARD_W - M - 8.0;
    cv.text_right(right, 52.0, 2.0, sc, status);
    let dot = (right - text_w(2.0, status) - 18.0, 55.0, 8.0, 8.0);
    cv.glow(dot, 4.0, 6.0, sc, 0.3 + 0.3 * pulse);
    cv.fill(dot, 4.0, sc, 1.0);
    let count = match v.page {
        Page::Games => format!(
            "{} game{}",
            v.games.len(),
            if v.games.len() == 1 { "" } else { "s" }
        ),
        Page::Settings => "player defaults".into(),
    };
    let cx = l.bx + l.bw - PAD - text_w(2.4, &count);
    cv.text_sh(cx, BLADE_Y + 44.0, 2.4, WHITE, &count);

    let rows = rows(v);
    if rows.is_empty() {
        empty(&mut cv, &l, v.home);
    }
    for (i, row) in rows.iter().enumerate().skip(v.scroll).take(ROWS) {
        let r = row_rect(&l, i - v.scroll);
        let (rx, y, rw, rh) = r;
        let sel = i == v.sel;
        if sel {
            cv.glow(r, ROW_R, 16.0, 0xeaffd4, 0.22 + 0.18 * pulse);
            cv.shape(r, ROW_R, |_, rv, depth, _, _| {
                let f = rv / rh;
                let mut c = ramp(&[0xf6ffec, 0xd4f7b4, 0xa6e676], f);
                // Glossy top half.
                c = mix(c, [255.0; 3], 0.4 * (1.0 - smooth(0.42, 0.5, f)));
                if depth < 1.4 {
                    c = if f < 0.5 { [255.0; 3] } else { rgb(0x5a9a2a) };
                }
                (c, 0.97)
            });
        } else {
            cv.fill(r, ROW_R, 0x000c00, 0.2);
            cv.fill((rx + ROW_R, y, rw - 2.0 * ROW_R, 1.2), 0.0, WHITE, 0.16);
        }
        // The thumbnail tile.
        let tile = (rx + 5.0, y + (rh - TILE) / 2.0, TILE, TILE);
        let (ix, iy, ..) = tile;
        cv.glow((ix, iy + 1.5, TILE, TILE), 8.0, 5.0, BLACK, 0.45);
        cv.plate(tile, 8.0, row.seed);
        let ts = if row.tile.chars().count() > 1 {
            2.6
        } else {
            3.2
        };
        let tw = text_w(ts, &row.tile);
        cv.text_sh(
            ix + (TILE - tw) / 2.0,
            iy + (TILE - 7.0 * ts) / 2.0,
            ts,
            BONE,
            &row.tile,
        );
        let (tc, subc) = if sel { (INK, 0x2f6a14) } else { (WHITE, SUB) };
        let tx = ix + TILE + 16.0;
        let text = |cv: &mut Cv, x: f32, y: f32, size: f32, c: u32, s: &str| {
            if sel {
                cv.text(x, y, size, c, s)
            } else {
                cv.text_sh(x, y, size, c, s)
            }
        };
        match v.page {
            Page::Games => {
                let tag_w = row
                    .tag
                    .map_or(0.0, |(label, ..)| pill_w(22.0, label) + 16.0);
                let room = rx + rw - tx - 16.0 - tag_w;
                text(&mut cv, tx, y + 9.0, 3.0, tc, &head(&row.title, 3.0, room));
                text(&mut cv, tx, y + 35.0, 1.8, subc, &tail(&row.sub, 1.8, room));
                if let Some((label, bg, fg)) = row.tag {
                    let (bg, fg) = if sel && bg != GOLD {
                        (INK, GREEN_HI)
                    } else {
                        (bg, fg)
                    };
                    let px = rx + rw - 14.0 - pill_w(22.0, label);
                    cv.pill(px, y + (rh - 22.0) / 2.0, 22.0, bg, fg, label);
                }
            }
            Page::Settings => {
                text(&mut cv, tx, y + 9.0, 3.0, tc, &row.title);
                text(&mut cv, tx, y + 35.0, 1.8, subc, &row.sub);
                // Fixed arrows (they are click targets), the value centred
                // between them.
                let (down, up) = arrow_rects(&l, i - v.scroll);
                let ax = |r: Rect| r.0 + 12.0;
                text(&mut cv, ax(down), y + 17.0, 3.0, tc, "<");
                text(&mut cv, ax(up), y + 17.0, 3.0, tc, ">");
                let val = setting(v, i);
                let mid = (ax(down) + ax(up) + text_w(3.0, ">")) / 2.0;
                // A long value a player typed (4294967295) is drawn smaller,
                // clear of the arrows.
                let room = ax(up) - ax(down) - text_w(3.0, "<") - 24.0;
                let size = (3.0 * room / text_w(3.0, &val).max(1.0)).min(3.0);
                text(
                    &mut cv,
                    mid - text_w(size, &val) / 2.0,
                    y + 17.0 + 3.5 * (3.0 - size),
                    size,
                    tc,
                    &val,
                );
            }
        }
    }
    // Scroll bar.
    if rows.len() > ROWS {
        cv.fill(bar_rect(&l), 3.0, BLACK, 0.3);
        cv.fill(thumb_rect(&l, rows.len(), v.scroll), 3.0, 0xeaffd4, 0.9);
    }

    details(&mut cv, &l, v, pulse);

    // Dim the footer hints that do nothing now: all of them while a game
    // runs, A when there is no game to start.
    let (hs, end) = hints(v.page);
    let (a_x, _, _, a_label, _) = hs[2];
    let dim = if v.running.is_some() {
        Some((hs[0].0, end + page_hint_w()))
    } else if v.page == Page::Games && v.games.is_empty() {
        Some((a_x, a_x + 32.0 + text_w(2.0, a_label)))
    } else {
        None
    };
    if let Some((x0, x1)) = dim {
        let r = (x0 - 4.0, FOOT_Y + 8.0, x1 - x0 + 8.0, FOOT_H - 16.0);
        cv.fill(r, 12.0, 0x16191a, 0.72);
    }
}

fn empty(cv: &mut Cv, l: &Lay, home: &Path) {
    let (x, y0, w, h) = (l.list_x, LIST_Y, l.list_w, ROWS as f32 * ROW_H - ROW_GAP);
    cv.fill((x, y0, w, h), 14.0, BLACK, 0.22);
    let (x, mut y) = (x + 26.0, y0 + 26.0);
    cv.text_glow(x, y, 4.4, GREEN_HI, "No games yet", 8.0, 0.5);
    cv.text_sh(x, y, 4.4, WHITE, "No games yet");
    y += 58.0;
    let games = home.join("games");
    let path = tail(&format!("{}\\<sha256>\\", games.display()), 2.0, w - 52.0);
    let lines = [
        ("Put each built game folder in:", WHITE),
        (&*path, GREEN_HI),
        ("", WHITE),
        ("A built game folder holds game.exe and run.toml.", WHITE),
        ("", WHITE),
        ("Or drop a built game folder on this window,", WHITE),
        ("or on rustxenon-play.exe in Explorer.", WHITE),
        ("", WHITE),
        ("Press Y (F5 on a keyboard) to look again.", SUB),
    ];
    for (s, c) in lines {
        cv.text_sh(x, y, 2.0, c, s);
        y += 26.0;
    }
}

/// Big letters on the preview plate.
fn preview_text(cv: &mut Cv, l: &Lay, big: &str) {
    let (x, y, w, h) = preview_rect(l);
    let size = (h * 0.5 / 7.0).min((w - 60.0) / (font::width(big).max(1) as f32));
    let (tx, ty) = (
        x + (w - text_w(size, big)) / 2.0,
        y + (h - 7.0 * size) / 2.0,
    );
    let m = text_mask(cv.fit(), tx, ty, size, big, false);
    let o = (2.5 * cv.k).round().max(1.0) as i32;
    cv.mask(&m, o, o + 1, [0.0; 3], 0.6);
    cv.mask(&m, 0, 0, rgb(0xf2e8d2), 1.0);
}

/// The right-hand panel: a preview, the selected row in full, notes, the prompt.
fn details(cv: &mut Cv, l: &Lay, v: &View, pulse: f32) {
    let (x, iw) = (l.dx, l.dw);
    let mut y = LIST_Y + PREVIEW_H + 16.0;
    let pair = |cv: &mut Cv, y: &mut f32, name: &str, value: &str| {
        cv.text_sh(x, *y, 1.7, GREEN_HI, name);
        *y += 16.0;
        cv.text_sh(x, *y, 2.0, WHITE, &tail(value, 2.0, iw));
        *y += 24.0;
    };
    // A note goes over the preview plate, so it never hides the text.
    let note = v.note.or(v.settings_err);
    let preview = |cv: &mut Cv, big: &str| {
        if note.is_none() {
            preview_text(cv, l, big)
        }
    };
    match v.page {
        Page::Games => match v.games.get(v.sel) {
            Some(g) => {
                preview(cv, &initials(&g.title));
                for s in wrap(&g.title, 3.0, iw, 2) {
                    cv.text_sh(x, y, 3.0, WHITE, &s);
                    y += 28.0;
                }
                y += 4.0;
                pair(cv, &mut y, "Folder", &g.dir.to_string_lossy());
                pair(cv, &mut y, "Executable", &g.exe);
                pair(cv, &mut y, "Game files", &g.game);
            }
            None => {
                preview(cv, "?");
                cv.text_sh(x, y, 3.0, WHITE, "Nothing selected");
            }
        },
        Page::Settings => {
            let i = v.sel.min(FIELDS.len() - 1);
            let f = &FIELDS[i];
            preview(cv, &setting(v, i));
            cv.text_sh(x, y, 3.0, WHITE, &head(f.label, 3.0, iw));
            y += 32.0;
            let bad = v.settings_err.is_none() && !v.settings.valid(i);
            let why = "The value in settings.toml is not one the game takes: it uses this default.";
            let lines = wrap(f.help, 2.0, iw, 3)
                .into_iter()
                .map(|s| (s, PALE))
                .chain(
                    bad.then(|| wrap(why, 2.0, iw, 2))
                        .into_iter()
                        .flatten()
                        .map(|s| (s, WARN)),
                );
            for (s, c) in lines {
                cv.text_sh(x, y, 2.0, c, &s);
                y += 22.0;
            }
        }
    }

    // The info box, then any note above it.
    let info = info_rect(l);
    let (ix, iy) = (info.0 + 16.0, info.1);
    if let Some(Game { title, .. }) = v.running {
        let a = 0.6 + 0.4 * pulse;
        cv.glow(info, 14.0, 12.0, GOLD, 0.25 * a);
        cv.text_glow(ix, iy + 9.0, 3.0, GOLD, "Running", 6.0, 0.5 * a);
        cv.text(ix, iy + 9.0, 3.0, 0xffe080, "Running");
        let rx = ix + text_w(3.0, "Running") + 14.0;
        cv.text(
            rx,
            iy + 13.0,
            2.0,
            BONE,
            &head(title, 2.0, x + iw - rx - 16.0),
        );
        cv.text(ix, iy + 35.0, 1.8, DIM, "back here when it closes");
    } else if v.page == Page::Games && !v.games.is_empty() {
        cv.button(ix + 15.0, iy + INFO_H / 2.0, 15.0, GREEN, "A");
        cv.text_glow(
            ix + 42.0,
            iy + 18.0,
            3.0,
            GREEN,
            "Start game",
            6.0,
            0.2 + 0.25 * pulse,
        );
        cv.text(ix + 42.0, iy + 18.0, 3.0, GREEN_HI, "Start game");
        let kx = ix + 42.0 + text_w(3.0, "Start game") + 14.0;
        cv.text(kx, iy + 22.0, 1.8, DIM, "or Enter");
    } else if v.page == Page::Games {
        cv.button(ix + 15.0, iy + INFO_H / 2.0, 15.0, 0xf0b428, "Y");
        cv.text(ix + 42.0, iy + 21.0, 2.0, GREEN_HI, "Look again");
        let kx = ix + 42.0 + text_w(2.0, "Look again") + 14.0;
        cv.text(kx, iy + 22.0, 1.8, DIM, "or F5");
    } else {
        // The label stays; the path loses its start if it is long.
        let label = "Saved in ";
        cv.text(ix, iy + 12.0, 1.8, GREEN_HI, label);
        let path = v.home.join("settings.toml").display().to_string();
        let lw = text_w(1.8, label) + 4.0;
        let path = tail(&path, 1.8, iw - 32.0 - lw);
        cv.text(ix + lw, iy + 12.0, 1.8, GREEN_HI, &path);
        cv.text(
            ix,
            iy + 32.0,
            1.8,
            DIM,
            "A game's own run.toml overrides these.",
        );
    }
    if let Some(note) = note {
        let ls = wrap(note, 2.0, iw - 32.0, NOTE_LINES);
        let (_, ny, _, nh) = note_rect(l, ls.len());
        cv.fill((x, ny, iw, nh), 12.0, WARN, 0.9);
        cv.fill(
            (x + 1.5, ny + 1.5, iw - 3.0, nh - 3.0),
            10.5,
            0x2a1004,
            0.95,
        );
        let mut ly = ny + 10.0;
        for s in ls {
            cv.text(x + 16.0, ly, 2.0, WARN, &s);
            ly += 22.0;
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
            settings_err: None,
            home: Path::new("C:\\home"),
        }
    }

    fn games() -> Vec<Game> {
        let g = Game {
            dir: "C:\\home\\games\\ab".into(),
            title: "Test Game One".into(),
            exe: "e".into(),
            game: "g".into(),
            update: None,
            data: None,
            recent: true,
        };
        vec![g; 9]
    }

    #[test]
    fn renders_any_size() {
        let s = Settings::default();
        let games = games();
        for (w, h) in [(1280, 720), (333, 777), (1, 1), (64, 16), (800, 600)] {
            let mut px = vec![0; w * h];
            let mut v = view(&games, &s);
            v.scroll = 2;
            v.sel = 3;
            v.running = Some(&games[0]);
            render(&mut px, w, h, &v, 0.3);
            v.settings_err = Some("TOML parse error at line 3, column 10: invalid string");
            v.page = Page::Settings;
            render(&mut px, w, h, &v, 0.3);
            v.settings_err = None;
            v.games = &[];
            v.page = Page::Settings;
            render(&mut px, w, h, &v, 0.0);
            v.page = Page::Games;
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
        // Row corners and centres and the other page's side blade, at 1x and
        // 2x, on both pages.
        for k in [1.0, 2.0] {
            let (w, h) = ((BOARD_W * k) as usize, (BOARD_H * k) as usize);
            for page in PAGES {
                let at = |x: f32, y: f32| hit(w, h, page, x * k, y * k);
                let l = lay(page);
                for r in 0..ROWS {
                    let (x, y, rw, rh) = row_rect(&l, r);
                    assert_eq!(at(x + rw / 2.0, y + rh / 2.0), Some(Hit::Row(r)));
                    assert_eq!(at(x + 3.0, y + 3.0), Some(Hit::Row(r)));
                    // The arrows only step on the Settings page.
                    let (down, up) = arrow_rects(&l, r);
                    let mid = |r: Rect| (r.0 + r.2 / 2.0, r.1 + r.3 / 2.0);
                    let (dx, dy) = mid(down);
                    let (ux, uy) = mid(up);
                    if page == Page::Settings {
                        assert_eq!(at(dx, dy), Some(Hit::Step(r, -1)));
                        assert_eq!(at(ux, uy), Some(Hit::Step(r, 1)));
                        // Between the arrows (on the value) only selects.
                        assert_eq!(at((dx + ux) / 2.0, dy), Some(Hit::Row(r)));
                    } else {
                        assert_eq!(at(dx, dy), Some(Hit::Row(r)));
                        assert_eq!(at(x + rw - 3.0, y + rh - 3.0), Some(Hit::Row(r)));
                    }
                }
                for other in PAGES.into_iter().filter(|&p| p != page) {
                    let (x, y, sw, sh) = strip_rect(page, other).unwrap();
                    // The middle of the part that shows.
                    let cx = if x < l.bx {
                        x + STRIP_W / 2.0
                    } else {
                        x + sw - STRIP_W / 2.0
                    };
                    assert_eq!(at(cx, y + sh / 2.0), Some(Hit::Tab(other)));
                    // The tucked part is the active blade's.
                    let tuck = if x < l.bx {
                        l.bx + 4.0
                    } else {
                        l.bx + l.bw - 4.0
                    };
                    assert_eq!(at(tuck, y + sh / 2.0), None);
                }
                // The footer hints are buttons.
                for (i, (x, y, hw, hh)) in hint_rects(page).into_iter().enumerate() {
                    assert_eq!(at(x + hw / 2.0, y + hh / 2.0), Some(Hit::Hint(i)));
                }
                assert_eq!(at(M + 40.0, FOOT_Y + FOOT_H / 2.0), None);
                // The scroll bar, and where along it.
                let (bx, by, bw, bh) = bar_rect(&l);
                assert_eq!(at(bx + bw / 2.0, by + bh / 4.0), Some(Hit::Bar));
                let (_, ty, _, th) = thumb_rect(&l, 300, 293);
                assert!(th >= 24.0 && (ty + th - (by + bh)).abs() < 0.01);
                // A press on the thumb holds it where it is: no scroll until
                // a drag. One on the track centres the thumb there.
                for (scroll, at_y) in [(293, ty + th / 2.0), (293, ty + 1.0), (0, by + 1.0)] {
                    let y = at_y * k;
                    let grab = bar_grab(w, h, page, 300, scroll, y);
                    assert_eq!(bar_scroll(w, h, page, 300, y, grab), scroll);
                }
                let y = (by + bh / 2.0) * k;
                let grab = bar_grab(w, h, page, 300, 0, y);
                assert_eq!(grab, th / 2.0);
                assert_eq!(bar_scroll(w, h, page, 300, y, grab), 293 / 2 + 1);
                // The last pixel of the track reaches the end.
                let y = (by + bh) * k - 1.0;
                assert_eq!(bar_scroll(w, h, page, 300, y, grab), 293);
                assert_eq!(bar_scroll(w, h, page, 300, 0.0, grab), 0);
                let (x, y, iw, ih) = info_rect(&l);
                let info = (page == Page::Games).then_some(Hit::Info);
                assert_eq!(at(x + iw / 2.0, y + ih / 2.0), info);
                assert_eq!(at(x + 3.0, y + ih - 3.0), info);
                assert_eq!(at(l.bx + PAD + 10.0, BLADE_Y + 40.0), None);
                assert_eq!(at(5.0, 5.0), None);
            }
        }
    }

    /// The rows and blades are drawn where the hit test finds them: the
    /// selected row's glossy bar is lighter than the rows round it, and a
    /// side blade's steel is grey where the active blade is green.
    #[test]
    fn drawn_where_hit() {
        let s = Settings::default();
        let games = games();
        let (w, h) = (1280, 720);
        let px_at = |px: &[u32], x: f32, y: f32| rgb(px[y as usize * w + x as usize]);
        for (page, sel) in [(Page::Games, 2), (Page::Settings, 4)] {
            let mut v = view(&games, &s);
            v.page = page;
            v.sel = sel;
            v.note = None;
            let mut px = vec![0; w * h];
            render(&mut px, w, h, &v, 0.0);
            let l = lay(page);
            let luma = |r: usize| {
                let (x, y, rw, rh) = row_rect(&l, r);
                // A blank spot of the row between the text and the end.
                let c = px_at(&px, x + rw * 0.7, y + rh - 4.0);
                c[0] + c[1] + c[2]
            };
            for r in (0..ROWS).filter(|&r| r != sel) {
                assert!(luma(sel) > luma(r) + 150.0, "{page:?} row {r}");
            }
            let other = PAGES.into_iter().find(|&p| p != page).unwrap();
            let (sx, sy, sw, sh) = strip_rect(page, other).unwrap();
            let cx = if sx < l.bx { sx + 8.0 } else { sx + sw - 8.0 };
            let steel = px_at(&px, cx, sy + sh / 2.0);
            assert!(
                (steel[1] - steel[0]).abs() < 60.0,
                "{page:?} strip {steel:?}"
            );
            // In the gap between the list and the details, below the sheen.
            let green = px_at(&px, l.dx - 8.0, LIST_Y + 250.0);
            assert!(green[1] > green[0] + 60.0, "{page:?} blade {green:?}");
        }
    }

    /// The longest note stays on the preview plate, above the details text.
    #[test]
    fn note_clear_of_details() {
        for page in PAGES {
            let l = lay(page);
            let (_, py, _, ph) = preview_rect(&l);
            let (_, ny, _, nh) = note_rect(&l, NOTE_LINES);
            assert!(ny >= py && ny + nh <= py + ph, "{page:?}");
        }
    }

    /// Only the game started is tagged running, not another with its title;
    /// settings show "?" while the file cannot be read.
    #[test]
    fn running_tag_and_unknown_settings() {
        let s = Settings::default();
        let mut games = games();
        games[1].dir = "C:\\home\\games\\cd".into();
        let mut v = view(&games, &s);
        v.running = Some(&games[1]);
        let tags: Vec<_> = rows(&v).iter().map(|r| r.tag.map(|t| t.0)).collect();
        assert_eq!(tags[..3], [Some("recent"), Some("running"), Some("recent")]);
        assert_eq!(setting(&v, 0), "1280");
        v.settings_err = Some("broken");
        assert_eq!(setting(&v, 0), "?");
    }

    #[test]
    fn text_fitting() {
        let w = |s: &str| text_w(1.0, s);
        assert_eq!(head("abcdefgh", 1.0, w("abcdefgh")), "abcdefgh");
        let h = head("abcdefgh", 1.0, w("abcd.."));
        assert_eq!(h, "abcd..");
        let t = tail("C:\\games\\folder", 1.0, w("..folder"));
        assert_eq!(t, "..folder");
        assert_eq!(
            wrap("one two three", 1.0, w("one two"), 3),
            ["one two", "three"]
        );
        // A word too long for a line is split.
        assert_eq!(wrap("abcdefgh", 1.0, w("abcd"), 3), ["abcd", "efgh"]);
        // Too many lines: the last one says so.
        let ls = wrap("a b c d e f", 1.0, w("a b"), 2);
        assert_eq!(ls.len(), 2);
        assert!(ls[1].ends_with(".."), "{ls:?}");
        assert!(ls.iter().all(|l| w(l) <= w("a b")));
        assert_eq!(initials("Test Game One"), "TG");
        assert_eq!(initials("  "), "?");
    }

    /// Frame time at 1920x1080:
    /// `cargo test --release -p xenon-play frame_time -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn frame_time() {
        let s = Settings::default();
        let mut games = games();
        games[1].exe = "C:\\x\\default.xex".into();
        let (w, h) = (1920, 1080);
        for page in PAGES {
            let mut v = view(&games, &s);
            v.page = page;
            v.sel = 1;
            let back = backdrop(w, h, page);
            let mut px = back.clone();
            overlay(&mut px, w, h, &v, 0.0);
            let t = std::time::Instant::now();
            for i in 0..20 {
                px.copy_from_slice(&back);
                overlay(&mut px, w, h, &v, i as f32 * 0.03);
            }
            eprintln!("{page:?}: {:?} a frame", t.elapsed() / 20);
        }
    }
}
