//! Vector unit helpers called by translated code: AltiVec (VMX) plus the
//! console's VMX128 extensions.
//!
//! Every helper takes its source vectors by value and returns the result, so
//! translated code reads all sources before the destination is written. That
//! makes `vD == vA` or `vD == vB` safe by construction; the per-instruction
//! tests check it for every instruction anyway.
//!
//! Lanes are numbered in big-endian order: element 0 is the leftmost
//! (lowest address in memory).
//!
//! Floating point follows the console: denormal inputs and results are
//! flushed to zero (VSCR[NJ] behaviour, always on), NaN operands propagate
//! quieted, and invalid operations give the default NaN 0x7FC00000. The
//! estimate instructions (`vrefp`, `vrsqrtefp`, `vexptefp`, `vlogefp`) return
//! the exact value rather than the hardware's estimate.

use crate::{GuestMemory, V128};

pub const VSCR_NJ: u32 = 0x1_0000;
pub const VSCR_SAT: u32 = 0x1;
const DEFAULT_NAN: u32 = 0x7FC0_0000;

// ---------------------------------------------------------------- lanes

#[inline]
pub fn bytes(v: V128) -> [u8; 16] {
    let mut o = [0u8; 16];
    for i in 0..4 {
        o[4 * i..4 * i + 4].copy_from_slice(&v[i].to_be_bytes());
    }
    o
}

#[inline]
pub fn from_bytes(b: [u8; 16]) -> V128 {
    let mut v = [0u32; 4];
    for (i, w) in v.iter_mut().enumerate() {
        *w = u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
    }
    v
}

#[inline]
pub fn halves(v: V128) -> [u16; 8] {
    let mut o = [0u16; 8];
    for i in 0..4 {
        o[2 * i] = (v[i] >> 16) as u16;
        o[2 * i + 1] = v[i] as u16;
    }
    o
}

#[inline]
pub fn from_halves(h: [u16; 8]) -> V128 {
    let mut v = [0u32; 4];
    for (i, w) in v.iter_mut().enumerate() {
        *w = (h[2 * i] as u32) << 16 | h[2 * i + 1] as u32;
    }
    v
}

#[inline]
fn to_u128(v: V128) -> u128 {
    (v[0] as u128) << 96 | (v[1] as u128) << 64 | (v[2] as u128) << 32 | v[3] as u128
}

#[inline]
fn from_u128(x: u128) -> V128 {
    [
        (x >> 96) as u32,
        (x >> 64) as u32,
        (x >> 32) as u32,
        x as u32,
    ]
}

#[inline]
fn map_b(a: V128, b: V128, mut f: impl FnMut(u8, u8) -> u8) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = f(a[i], b[i]);
    }
    from_bytes(o)
}

#[inline]
fn map_h(a: V128, b: V128, mut f: impl FnMut(u16, u16) -> u16) -> V128 {
    let (a, b) = (halves(a), halves(b));
    let mut o = [0u16; 8];
    for i in 0..8 {
        o[i] = f(a[i], b[i]);
    }
    from_halves(o)
}

#[inline]
fn map_w(a: V128, b: V128, mut f: impl FnMut(u32, u32) -> u32) -> V128 {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

#[inline]
fn sat(vscr: &mut u32, hit: bool) {
    if hit {
        *vscr |= VSCR_SAT;
    }
}

fn clamp_i(x: i64, lo: i64, hi: i64, s: &mut bool) -> i64 {
    if x < lo {
        *s = true;
        lo
    } else if x > hi {
        *s = true;
        hi
    } else {
        x
    }
}

// ---------------------------------------------------------------- integer add / subtract

pub fn vaddubm(a: V128, b: V128) -> V128 {
    map_b(a, b, u8::wrapping_add)
}
pub fn vadduhm(a: V128, b: V128) -> V128 {
    map_h(a, b, u16::wrapping_add)
}
pub fn vadduwm(a: V128, b: V128) -> V128 {
    map_w(a, b, u32::wrapping_add)
}
pub fn vsububm(a: V128, b: V128) -> V128 {
    map_b(a, b, u8::wrapping_sub)
}
pub fn vsubuhm(a: V128, b: V128) -> V128 {
    map_h(a, b, u16::wrapping_sub)
}
pub fn vsubuwm(a: V128, b: V128) -> V128 {
    map_w(a, b, u32::wrapping_sub)
}
pub fn vaddcuw(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x.overflowing_add(y).1 as u32)
}
/// Carry out of `a - b`: 1 when no borrow.
pub fn vsubcuw(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| (x >= y) as u32)
}

macro_rules! sat_op {
    ($name:ident, $map:ident, $t:ty, $wide:ty, $lo:expr, $hi:expr, $op:tt) => {
        pub fn $name(a: V128, b: V128, vscr: &mut u32) -> V128 {
            let mut s = false;
            let r = $map(a, b, |x, y| {
                let v = (x as $t as i64) $op (y as $t as i64);
                clamp_i(v, $lo, $hi, &mut s) as $wide
            });
            sat(vscr, s);
            r
        }
    };
}
sat_op!(vaddubs, map_b, u8, u8, 0, 255, +);
sat_op!(vadduhs, map_h, u16, u16, 0, 65535, +);
sat_op!(vadduws, map_w, u32, u32, 0, u32::MAX as i64, +);
sat_op!(vaddsbs, map_b, i8, u8, -128, 127, +);
sat_op!(vaddshs, map_h, i16, u16, -32768, 32767, +);
sat_op!(vaddsws, map_w, i32, u32, i32::MIN as i64, i32::MAX as i64, +);
sat_op!(vsububs, map_b, u8, u8, 0, 255, -);
sat_op!(vsubuhs, map_h, u16, u16, 0, 65535, -);
sat_op!(vsubuws, map_w, u32, u32, 0, u32::MAX as i64, -);
sat_op!(vsubsbs, map_b, i8, u8, -128, 127, -);
sat_op!(vsubshs, map_h, i16, u16, -32768, 32767, -);
sat_op!(vsubsws, map_w, i32, u32, i32::MIN as i64, i32::MAX as i64, -);

// ---------------------------------------------------------------- max / min / average

macro_rules! lane_op {
    ($name:ident, $map:ident, $t:ty, $w:ty, |$x:ident, $y:ident| $e:expr) => {
        pub fn $name(a: V128, b: V128) -> V128 {
            $map(a, b, |p, q| {
                let ($x, $y) = (p as $t, q as $t);
                ($e) as $w
            })
        }
    };
}
lane_op!(vmaxub, map_b, u8, u8, |x, y| x.max(y));
lane_op!(vmaxuh, map_h, u16, u16, |x, y| x.max(y));
lane_op!(vmaxuw, map_w, u32, u32, |x, y| x.max(y));
lane_op!(vmaxsb, map_b, i8, u8, |x, y| x.max(y));
lane_op!(vmaxsh, map_h, i16, u16, |x, y| x.max(y));
lane_op!(vmaxsw, map_w, i32, u32, |x, y| x.max(y));
lane_op!(vminub, map_b, u8, u8, |x, y| x.min(y));
lane_op!(vminuh, map_h, u16, u16, |x, y| x.min(y));
lane_op!(vminuw, map_w, u32, u32, |x, y| x.min(y));
lane_op!(vminsb, map_b, i8, u8, |x, y| x.min(y));
lane_op!(vminsh, map_h, i16, u16, |x, y| x.min(y));
lane_op!(vminsw, map_w, i32, u32, |x, y| x.min(y));
lane_op!(vavgub, map_b, u8, u8, |x, y| (x as i64 + y as i64 + 1) >> 1);
lane_op!(vavguh, map_h, u16, u16, |x, y| (x as i64 + y as i64 + 1)
    >> 1);
lane_op!(vavguw, map_w, u32, u32, |x, y| (x as i64 + y as i64 + 1)
    >> 1);
lane_op!(vavgsb, map_b, i8, u8, |x, y| (x as i64 + y as i64 + 1) >> 1);
lane_op!(vavgsh, map_h, i16, u16, |x, y| (x as i64 + y as i64 + 1)
    >> 1);
lane_op!(vavgsw, map_w, i32, u32, |x, y| (x as i64 + y as i64 + 1)
    >> 1);

// ---------------------------------------------------------------- shifts and rotates

lane_op!(vrlb, map_b, u8, u8, |x, y| x.rotate_left(y as u32 & 7));
lane_op!(vrlh, map_h, u16, u16, |x, y| x.rotate_left(y as u32 & 15));
lane_op!(vrlw, map_w, u32, u32, |x, y| x.rotate_left(y & 31));
lane_op!(vslb, map_b, u8, u8, |x, y| x << (y & 7));
lane_op!(vslh, map_h, u16, u16, |x, y| x << (y & 15));
lane_op!(vslw, map_w, u32, u32, |x, y| x << (y & 31));
lane_op!(vsrb, map_b, u8, u8, |x, y| x >> (y & 7));
lane_op!(vsrh, map_h, u16, u16, |x, y| x >> (y & 15));
lane_op!(vsrw, map_w, u32, u32, |x, y| x >> (y & 31));
lane_op!(vsrab, map_b, i8, u8, |x, y| x >> (y as u8 & 7));
lane_op!(vsrah, map_h, i16, u16, |x, y| x >> (y as u16 & 15));
lane_op!(vsraw, map_w, i32, u32, |x, y| x >> (y as u32 & 31));

/// `vsl`: whole vector left by the low 3 bits of the last byte of `b`.
pub fn vsl(a: V128, b: V128) -> V128 {
    from_u128(to_u128(a) << (b[3] & 7))
}
pub fn vsr(a: V128, b: V128) -> V128 {
    from_u128(to_u128(a) >> (b[3] & 7))
}
/// `vslo`: whole vector left by whole bytes, count in bits 1..4 of the last byte of `b`.
pub fn vslo(a: V128, b: V128) -> V128 {
    let n = (b[3] >> 3) & 0xF;
    from_u128(if n == 0 {
        to_u128(a)
    } else {
        to_u128(a) << (8 * n)
    })
}
pub fn vsro(a: V128, b: V128) -> V128 {
    let n = (b[3] >> 3) & 0xF;
    from_u128(if n == 0 {
        to_u128(a)
    } else {
        to_u128(a) >> (8 * n)
    })
}

/// `vsldoi`: bytes `sh..sh+16` of `a || b`.
pub fn vsldoi(a: V128, b: V128, sh: u32) -> V128 {
    let sh = sh & 15;
    if sh == 0 {
        a
    } else {
        from_u128(to_u128(a) << (8 * sh) | to_u128(b) >> (128 - 8 * sh))
    }
}

// ---------------------------------------------------------------- logical, select, permute

pub fn vand(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x & y)
}
pub fn vandc(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x & !y)
}
pub fn vor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x | y)
}
pub fn vnor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| !(x | y))
}
pub fn vxor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x ^ y)
}
/// `vsel`: bits of `b` where `c` is set, else bits of `a`.
pub fn vsel(a: V128, b: V128, c: V128) -> V128 {
    [0, 1, 2, 3].map(|i| (a[i] & !c[i]) | (b[i] & c[i]))
}
/// `vperm`: byte `i` is byte `c[i] & 31` of `a || b`.
pub fn vperm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b, c) = (bytes(a), bytes(b), bytes(c));
    let mut o = [0u8; 16];
    for i in 0..16 {
        let k = (c[i] & 31) as usize;
        o[i] = if k < 16 { a[k] } else { b[k - 16] };
    }
    from_bytes(o)
}

// ---------------------------------------------------------------- merge, splat

pub fn vmrghb(a: V128, b: V128) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    from_bytes(core::array::from_fn(|i| {
        if i % 2 == 0 {
            a[i / 2]
        } else {
            b[i / 2]
        }
    }))
}
pub fn vmrglb(a: V128, b: V128) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    from_bytes(core::array::from_fn(|i| {
        if i % 2 == 0 {
            a[8 + i / 2]
        } else {
            b[8 + i / 2]
        }
    }))
}
pub fn vmrghh(a: V128, b: V128) -> V128 {
    let (a, b) = (halves(a), halves(b));
    from_halves(core::array::from_fn(|i| {
        if i % 2 == 0 {
            a[i / 2]
        } else {
            b[i / 2]
        }
    }))
}
pub fn vmrglh(a: V128, b: V128) -> V128 {
    let (a, b) = (halves(a), halves(b));
    from_halves(core::array::from_fn(|i| {
        if i % 2 == 0 {
            a[4 + i / 2]
        } else {
            b[4 + i / 2]
        }
    }))
}
pub fn vmrghw(a: V128, b: V128) -> V128 {
    [a[0], b[0], a[1], b[1]]
}
pub fn vmrglw(a: V128, b: V128) -> V128 {
    [a[2], b[2], a[3], b[3]]
}
pub fn vspltb(b: V128, i: u32) -> V128 {
    let x = bytes(b)[(i & 15) as usize] as u32;
    [x * 0x0101_0101; 4]
}
pub fn vsplth(b: V128, i: u32) -> V128 {
    let x = halves(b)[(i & 7) as usize] as u32;
    [x << 16 | x; 4]
}
pub fn vspltw(b: V128, i: u32) -> V128 {
    [b[(i & 3) as usize]; 4]
}
/// Splat of a 5-bit signed immediate (already sign-extended by the caller).
pub fn vspltisb(simm: i32) -> V128 {
    [(simm as u8 as u32) * 0x0101_0101; 4]
}
pub fn vspltish(simm: i32) -> V128 {
    let x = simm as u16 as u32;
    [x << 16 | x; 4]
}
pub fn vspltisw(simm: i32) -> V128 {
    [simm as u32; 4]
}

// ---------------------------------------------------------------- pack / unpack

fn pack_h(a: V128, b: V128, mut f: impl FnMut(u16) -> u8) -> V128 {
    let (a, b) = (halves(a), halves(b));
    from_bytes(core::array::from_fn(|i| {
        if i < 8 {
            f(a[i])
        } else {
            f(b[i - 8])
        }
    }))
}
fn pack_w(a: V128, b: V128, mut f: impl FnMut(u32) -> u16) -> V128 {
    from_halves(core::array::from_fn(|i| {
        if i < 4 {
            f(a[i])
        } else {
            f(b[i - 4])
        }
    }))
}

pub fn vpkuhum(a: V128, b: V128) -> V128 {
    pack_h(a, b, |x| x as u8)
}
pub fn vpkuwum(a: V128, b: V128) -> V128 {
    pack_w(a, b, |x| x as u16)
}
pub fn vpkuhus(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_h(a, b, |x| clamp_i(x as i64, 0, 255, &mut s) as u8);
    sat(vscr, s);
    r
}
pub fn vpkuwus(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_w(a, b, |x| clamp_i(x as i64, 0, 65535, &mut s) as u16);
    sat(vscr, s);
    r
}
pub fn vpkshus(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_h(a, b, |x| clamp_i(x as i16 as i64, 0, 255, &mut s) as u8);
    sat(vscr, s);
    r
}
pub fn vpkswus(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_w(a, b, |x| clamp_i(x as i32 as i64, 0, 65535, &mut s) as u16);
    sat(vscr, s);
    r
}
pub fn vpkshss(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_h(a, b, |x| clamp_i(x as i16 as i64, -128, 127, &mut s) as u8);
    sat(vscr, s);
    r
}
pub fn vpkswss(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = pack_w(a, b, |x| {
        clamp_i(x as i32 as i64, -32768, 32767, &mut s) as u16
    });
    sat(vscr, s);
    r
}
/// `vpkpx`: 8:8:8:8 pixels to 1:5:5:5.
pub fn vpkpx(a: V128, b: V128) -> V128 {
    pack_w(a, b, |w| {
        ((w >> 24 & 1) << 15 | (w >> 19 & 0x1F) << 10 | (w >> 11 & 0x1F) << 5 | (w >> 3 & 0x1F))
            as u16
    })
}

fn unpk_b(b: V128, off: usize) -> V128 {
    let b = bytes(b);
    from_halves(core::array::from_fn(|i| b[off + i] as i8 as i16 as u16))
}
fn unpk_h(b: V128, off: usize) -> V128 {
    let h = halves(b);
    core::array::from_fn(|i| h[off + i] as i16 as i32 as u32)
}
fn unpk_px(b: V128, off: usize) -> V128 {
    let h = halves(b);
    core::array::from_fn(|i| {
        let p = h[off + i] as u32;
        let a = if p & 0x8000 != 0 { 0xFF } else { 0 };
        a << 24 | (p >> 10 & 0x1F) << 16 | (p >> 5 & 0x1F) << 8 | (p & 0x1F)
    })
}
pub fn vupkhsb(b: V128) -> V128 {
    unpk_b(b, 0)
}
pub fn vupklsb(b: V128) -> V128 {
    unpk_b(b, 8)
}
pub fn vupkhsh(b: V128) -> V128 {
    unpk_h(b, 0)
}
pub fn vupklsh(b: V128) -> V128 {
    unpk_h(b, 4)
}
pub fn vupkhpx(b: V128) -> V128 {
    unpk_px(b, 0)
}
pub fn vupklpx(b: V128) -> V128 {
    unpk_px(b, 4)
}

// ---------------------------------------------------------------- multiply, multiply-add, sums

fn mul_b(a: V128, b: V128, odd: usize, signed: bool) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    from_halves(core::array::from_fn(|i| {
        let (x, y) = (a[2 * i + odd], b[2 * i + odd]);
        if signed {
            (x as i8 as i16 * y as i8 as i16) as u16
        } else {
            x as u16 * y as u16
        }
    }))
}
fn mul_h(a: V128, b: V128, odd: usize, signed: bool) -> V128 {
    let (a, b) = (halves(a), halves(b));
    core::array::from_fn(|i| {
        let (x, y) = (a[2 * i + odd], b[2 * i + odd]);
        if signed {
            (x as i16 as i32 * y as i16 as i32) as u32
        } else {
            x as u32 * y as u32
        }
    })
}
pub fn vmuleub(a: V128, b: V128) -> V128 {
    mul_b(a, b, 0, false)
}
pub fn vmuloub(a: V128, b: V128) -> V128 {
    mul_b(a, b, 1, false)
}
pub fn vmulesb(a: V128, b: V128) -> V128 {
    mul_b(a, b, 0, true)
}
pub fn vmulosb(a: V128, b: V128) -> V128 {
    mul_b(a, b, 1, true)
}
pub fn vmuleuh(a: V128, b: V128) -> V128 {
    mul_h(a, b, 0, false)
}
pub fn vmulouh(a: V128, b: V128) -> V128 {
    mul_h(a, b, 1, false)
}
pub fn vmulesh(a: V128, b: V128) -> V128 {
    mul_h(a, b, 0, true)
}
pub fn vmulosh(a: V128, b: V128) -> V128 {
    mul_h(a, b, 1, true)
}

pub fn vmhaddshs(a: V128, b: V128, c: V128, vscr: &mut u32) -> V128 {
    let (a, b, c) = (halves(a), halves(b), halves(c));
    let mut s = false;
    let r = from_halves(core::array::from_fn(|i| {
        let p = (a[i] as i16 as i64 * b[i] as i16 as i64) >> 15;
        clamp_i(p + c[i] as i16 as i64, -32768, 32767, &mut s) as u16
    }));
    sat(vscr, s);
    r
}
pub fn vmhraddshs(a: V128, b: V128, c: V128, vscr: &mut u32) -> V128 {
    let (a, b, c) = (halves(a), halves(b), halves(c));
    let mut s = false;
    let r = from_halves(core::array::from_fn(|i| {
        let p = (a[i] as i16 as i64 * b[i] as i16 as i64 + 0x4000) >> 15;
        clamp_i(p + c[i] as i16 as i64, -32768, 32767, &mut s) as u16
    }));
    sat(vscr, s);
    r
}
pub fn vmladduhm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b, c) = (halves(a), halves(b), halves(c));
    from_halves(core::array::from_fn(|i| {
        a[i].wrapping_mul(b[i]).wrapping_add(c[i])
    }))
}
pub fn vmsumubm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    core::array::from_fn(|i| {
        (0..4).fold(c[i], |s, j| {
            s.wrapping_add(a[4 * i + j] as u32 * b[4 * i + j] as u32)
        })
    })
}
pub fn vmsummbm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b) = (bytes(a), bytes(b));
    core::array::from_fn(|i| {
        (0..4).fold(c[i], |s, j| {
            s.wrapping_add((a[4 * i + j] as i8 as i32 * b[4 * i + j] as i32) as u32)
        })
    })
}
pub fn vmsumuhm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b) = (halves(a), halves(b));
    core::array::from_fn(|i| {
        (0..2).fold(c[i], |s, j| {
            s.wrapping_add(a[2 * i + j] as u32 * b[2 * i + j] as u32)
        })
    })
}
pub fn vmsumshm(a: V128, b: V128, c: V128) -> V128 {
    let (a, b) = (halves(a), halves(b));
    core::array::from_fn(|i| {
        (0..2).fold(c[i], |s, j| {
            s.wrapping_add((a[2 * i + j] as i16 as i32 * b[2 * i + j] as i16 as i32) as u32)
        })
    })
}
pub fn vmsumuhs(a: V128, b: V128, c: V128, vscr: &mut u32) -> V128 {
    let (a, b) = (halves(a), halves(b));
    let mut s = false;
    let r = core::array::from_fn(|i| {
        let t = (0..2).fold(c[i] as i64, |t, j| {
            t + a[2 * i + j] as i64 * b[2 * i + j] as i64
        });
        clamp_i(t, 0, u32::MAX as i64, &mut s) as u32
    });
    sat(vscr, s);
    r
}
pub fn vmsumshs(a: V128, b: V128, c: V128, vscr: &mut u32) -> V128 {
    let (a, b) = (halves(a), halves(b));
    let mut s = false;
    let r = core::array::from_fn(|i| {
        let t = (0..2).fold(c[i] as i32 as i64, |t, j| {
            t + a[2 * i + j] as i16 as i64 * b[2 * i + j] as i16 as i64
        });
        clamp_i(t, i32::MIN as i64, i32::MAX as i64, &mut s) as u32
    });
    sat(vscr, s);
    r
}
pub fn vsum4ubs(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let a = bytes(a);
    let mut s = false;
    let r = core::array::from_fn(|i| {
        let t = (0..4).fold(b[i] as i64, |t, j| t + a[4 * i + j] as i64);
        clamp_i(t, 0, u32::MAX as i64, &mut s) as u32
    });
    sat(vscr, s);
    r
}
pub fn vsum4sbs(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let a = bytes(a);
    let mut s = false;
    let r = core::array::from_fn(|i| {
        let t = (0..4).fold(b[i] as i32 as i64, |t, j| t + a[4 * i + j] as i8 as i64);
        clamp_i(t, i32::MIN as i64, i32::MAX as i64, &mut s) as u32
    });
    sat(vscr, s);
    r
}
pub fn vsum4shs(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let a = halves(a);
    let mut s = false;
    let r = core::array::from_fn(|i| {
        let t = b[i] as i32 as i64 + a[2 * i] as i16 as i64 + a[2 * i + 1] as i16 as i64;
        clamp_i(t, i32::MIN as i64, i32::MAX as i64, &mut s) as u32
    });
    sat(vscr, s);
    r
}
pub fn vsum2sws(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let mut r = [0u32; 4];
    for k in [1usize, 3] {
        let t = a[k - 1] as i32 as i64 + a[k] as i32 as i64 + b[k] as i32 as i64;
        r[k] = clamp_i(t, i32::MIN as i64, i32::MAX as i64, &mut s) as u32;
    }
    sat(vscr, s);
    r
}
pub fn vsumsws(a: V128, b: V128, vscr: &mut u32) -> V128 {
    let mut s = false;
    let t = a.iter().map(|&x| x as i32 as i64).sum::<i64>() + b[3] as i32 as i64;
    let r = [
        0,
        0,
        0,
        clamp_i(t, i32::MIN as i64, i32::MAX as i64, &mut s) as u32,
    ];
    sat(vscr, s);
    r
}

// ---------------------------------------------------------------- integer compares

/// CR6 value for a record-form compare: all lanes true (LT bit) or all false (EQ bit).
pub fn cr6(v: V128) -> u32 {
    if v == [u32::MAX; 4] {
        0b1000
    } else if v == [0; 4] {
        0b0010
    } else {
        0
    }
}

lane_op!(vcmpequb, map_b, u8, u8, |x, y| if x == y {
    0xFFu8
} else {
    0
});
lane_op!(vcmpequh, map_h, u16, u16, |x, y| if x == y {
    0xFFFFu16
} else {
    0
});
lane_op!(vcmpequw, map_w, u32, u32, |x, y| if x == y {
    u32::MAX
} else {
    0
});
lane_op!(vcmpgtub, map_b, u8, u8, |x, y| if x > y {
    0xFFu8
} else {
    0
});
lane_op!(vcmpgtuh, map_h, u16, u16, |x, y| if x > y {
    0xFFFFu16
} else {
    0
});
lane_op!(vcmpgtuw, map_w, u32, u32, |x, y| if x > y {
    u32::MAX
} else {
    0
});
lane_op!(vcmpgtsb, map_b, i8, u8, |x, y| if x > y {
    0xFFu8
} else {
    0
});
lane_op!(vcmpgtsh, map_h, i16, u16, |x, y| if x > y {
    0xFFFFu16
} else {
    0
});
lane_op!(vcmpgtsw, map_w, i32, u32, |x, y| if x > y {
    u32::MAX
} else {
    0
});

// ---------------------------------------------------------------- floating point

/// Flushes a denormal to a zero of the same sign.
#[inline]
pub fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

#[inline]
fn fl(w: u32) -> f32 {
    ftz(f32::from_bits(w))
}

#[inline]
fn quiet(w: u32) -> u32 {
    w | 0x0040_0000
}

#[inline]
fn out(r: f32) -> u32 {
    if r.is_nan() {
        DEFAULT_NAN
    } else {
        ftz(r).to_bits()
    }
}

fn f2(a: V128, b: V128, f: impl Fn(f32, f32) -> f32) -> V128 {
    map_w(a, b, |x, y| {
        let (p, q) = (fl(x), fl(y));
        if p.is_nan() {
            quiet(x)
        } else if q.is_nan() {
            quiet(y)
        } else {
            out(f(p, q))
        }
    })
}

fn f1(b: V128, f: impl Fn(f32) -> f32) -> V128 {
    b.map(|x| {
        let p = fl(x);
        if p.is_nan() {
            quiet(x)
        } else {
            out(f(p))
        }
    })
}

pub fn vaddfp(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| x + y)
}
pub fn vsubfp(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| x - y)
}
/// VMX128 `vmulfp128` (standard VMX has no plain multiply).
pub fn vmulfp(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| x * y)
}

fn f3(a: V128, b: V128, c: V128, f: impl Fn(f32, f32, f32) -> f32) -> V128 {
    core::array::from_fn(|i| {
        let (p, q, r) = (fl(a[i]), fl(b[i]), fl(c[i]));
        if p.is_nan() {
            quiet(a[i])
        } else if q.is_nan() {
            quiet(b[i])
        } else if r.is_nan() {
            quiet(c[i])
        } else {
            out(f(p, q, r))
        }
    })
}

/// `vmaddfp vD,vA,vC,vB`: `a * c + b`, fused. NaN priority A, B, C.
pub fn vmaddfp(a: V128, c: V128, b: V128) -> V128 {
    f3(a, b, c, |x, y, z| x.mul_add(z, y))
}
/// `vnmsubfp vD,vA,vC,vB`: `-(a * c - b)`, fused.
pub fn vnmsubfp(a: V128, c: V128, b: V128) -> V128 {
    f3(a, b, c, |x, y, z| -(x.mul_add(z, -y)))
}

pub fn vmaxfp(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| {
        if x == y {
            // +0 beats -0.
            f32::from_bits(x.to_bits() & y.to_bits())
        } else if x > y {
            x
        } else {
            y
        }
    })
}
pub fn vminfp(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| {
        if x == y {
            f32::from_bits(x.to_bits() | y.to_bits())
        } else if x < y {
            x
        } else {
            y
        }
    })
}

pub fn vrefp(b: V128) -> V128 {
    f1(b, |x| 1.0 / x)
}
// Negative inputs are checked explicitly: the optimiser assumes `sqrt` of a
// non-NaN is never NaN, so testing the result afterwards is not reliable.
pub fn vrsqrtefp(b: V128) -> V128 {
    f1(b, |x| if x < 0.0 { f32::NAN } else { 1.0 / x.sqrt() })
}
pub fn vexptefp(b: V128) -> V128 {
    f1(b, f32::exp2)
}
pub fn vlogefp(b: V128) -> V128 {
    f1(b, |x| if x < 0.0 { f32::NAN } else { x.log2() })
}
pub fn vrfin(b: V128) -> V128 {
    f1(b, f32::round_ties_even)
}
pub fn vrfiz(b: V128) -> V128 {
    f1(b, f32::trunc)
}
pub fn vrfip(b: V128) -> V128 {
    f1(b, f32::ceil)
}
pub fn vrfim(b: V128) -> V128 {
    f1(b, f32::floor)
}

/// `vcfux`: unsigned words to float, divided by 2^uimm.
pub fn vcfux(b: V128, uimm: u32) -> V128 {
    b.map(|x| out((x as f64 / (1u64 << uimm) as f64) as f32))
}
pub fn vcfsx(b: V128, uimm: u32) -> V128 {
    b.map(|x| out((x as i32 as f64 / (1u64 << uimm) as f64) as f32))
}
/// `vctuxs`: float times 2^uimm to unsigned words, toward zero, saturating.
/// A NaN gives 0 without setting SAT.
pub fn vctuxs(b: V128, uimm: u32, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = b.map(|x| {
        let f = fl(x);
        if f.is_nan() {
            0
        } else {
            let v = (f as f64 * (1u64 << uimm) as f64).trunc();
            if v < 0.0 {
                s = true;
                0
            } else if v > u32::MAX as f64 {
                s = true;
                u32::MAX
            } else {
                v as u32
            }
        }
    });
    sat(vscr, s);
    r
}
pub fn vctsxs(b: V128, uimm: u32, vscr: &mut u32) -> V128 {
    let mut s = false;
    let r = b.map(|x| {
        let f = fl(x);
        if f.is_nan() {
            0
        } else {
            let v = (f as f64 * (1u64 << uimm) as f64).trunc();
            if v < i32::MIN as f64 {
                s = true;
                i32::MIN as u32
            } else if v > i32::MAX as f64 {
                s = true;
                i32::MAX as u32
            } else {
                v as i32 as u32
            }
        }
    });
    sat(vscr, s);
    r
}

fn fcmp(a: V128, b: V128, f: impl Fn(f32, f32) -> bool) -> V128 {
    map_w(a, b, |x, y| if f(fl(x), fl(y)) { u32::MAX } else { 0 })
}
pub fn vcmpeqfp(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x == y)
}
pub fn vcmpgefp(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x >= y)
}
pub fn vcmpgtfp(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x > y)
}
/// `vcmpbfp`: bit 0 set when `a > b` fails `a <= b`, bit 1 when `a >= -b` fails.
pub fn vcmpbfp(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| {
        let (p, q) = (fl(x), fl(y));
        let le = p <= q;
        let ge = p >= -q;
        (!le as u32) << 31 | (!ge as u32) << 30
    })
}
/// CR6 for `vcmpbfp.`: EQ when every element is within bounds.
pub fn cr6_bounds(v: V128) -> u32 {
    if v == [0; 4] {
        0b0010
    } else {
        0
    }
}

// ---------------------------------------------------------------- VMX128 extras

/// `vmsum3fp128`: dot product of elements 0..3, in every element.
pub fn vmsum3fp(a: V128, b: V128) -> V128 {
    dot(a, b, 3)
}
/// `vmsum4fp128`: dot product of all four elements, in every element.
pub fn vmsum4fp(a: V128, b: V128) -> V128 {
    dot(a, b, 4)
}
fn dot(a: V128, b: V128, n: usize) -> V128 {
    let mut nan = None;
    for i in 0..n {
        for w in [a[i], b[i]] {
            if nan.is_none() && fl(w).is_nan() {
                nan = Some(quiet(w));
            }
        }
    }
    if let Some(q) = nan {
        return [q; 4];
    }
    let mut s = 0.0f32;
    for i in 0..n {
        s = ftz(s + ftz(fl(a[i]) * fl(b[i])));
    }
    [out(s); 4]
}

/// `vpermwi128`: element `i` is `b[(imm >> (6 - 2i)) & 3]`.
pub fn vpermwi(b: V128, imm: u32) -> V128 {
    core::array::from_fn(|i| b[(imm >> (6 - 2 * i) & 3) as usize])
}

/// `vrlimi128`: rotate `b` left by `rot` elements, then insert into `d` the
/// elements whose mask bit is set (mask bit 8 is element 0).
pub fn vrlimi(d: V128, b: V128, mask: u32, rot: u32) -> V128 {
    core::array::from_fn(|i| {
        if mask & (8 >> i) != 0 {
            b[(i + rot as usize) & 3]
        } else {
            d[i]
        }
    })
}

/// Half precision bits to single, IEEE rules.
fn half_to_f32(h: u16) -> u32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = (h >> 10 & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e: i32 = -14;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e + 127) as u32) << 23 | (m & 0x3FF) << 13
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | frac << 13
    } else {
        sign | (exp + 127 - 15) << 23 | frac << 13
    }
}

/// Single to the console's half format: no infinities (saturates at 0x7FFF
/// with the sign), mantissa truncated. Ported from XenonRecomp (MIT); see
/// CREDITS.md.
fn f32_to_half(w: u32) -> u16 {
    let mag = w & 0x7FFF_FFFF;
    let f = f32::from_bits(mag);
    let sign = ((w & 0x8000_0000) >> 16) as u16;
    let e8: u32 = if f.is_nan() || f > 65504.0 {
        0xFF
    } else {
        (w & 0x7F80_0000) >> 23
    };
    let m = if e8 != 0xFF {
        ((w & 0x7F_E000) >> 13) as u16
    } else {
        0
    };
    let h = if e8 == 0xFF {
        0x7FFF
    } else if e8 > 0x70 {
        (((e8 - 0x70) << 10) as u16).wrapping_add(m)
    } else if 0x71 - e8 > 31 {
        0
    } else {
        ((0x400 + m as u32) >> (0x71 - e8)) as u16
    };
    h | sign
}

/// Normalised short as the console stores it: `3.0 + s * 2^-22`, i.e. the
/// bits of 3.0 plus the short; -32768 unpacks to NaN.
fn short_to_f32(s: u16) -> u32 {
    if s == 0x8000 {
        DEFAULT_NAN
    } else {
        0x4040_0000u32.wrapping_add(s as i16 as i32 as u32)
    }
}

fn f32_to_short(w: u32) -> u16 {
    let f = f32::from_bits(w);
    let (lo, hi) = (f32::from_bits(0x403F_8001), f32::from_bits(0x4040_7FFF));
    let c = if f >= hi {
        hi
    } else if f >= lo {
        f
    } else {
        lo
    };
    c.to_bits() as u16
}

/// Low bits of `w` saturated to the floats with bits `lo..=hi`, as
/// [`f32_to_short`] does (NaN takes `lo`).
fn f32_to_bits_in(w: u32, lo: u32, hi: u32, mask: u32) -> u32 {
    let (f, l, h) = (f32::from_bits(w), f32::from_bits(lo), f32::from_bits(hi));
    let c = if f >= h { hi } else if f >= l { w } else { lo };
    c & mask
}

fn f32_to_color(w: u32) -> u32 {
    let f = f32::from_bits(w);
    if f >= f32::from_bits(0x4040_00FF) {
        0xFF
    } else if f >= 3.0 {
        w & 0xFF
    } else {
        0
    }
}

/// Signed 10-bit field as `3.0 + v * 2^-22`, like [`short_to_f32`]; -512
/// unpacks to NaN.
fn ten_to_f32(v: u32) -> u32 {
    let v = v & 0x3FF;
    if v == 0x200 {
        DEFAULT_NAN
    } else {
        0x4040_0000u32.wrapping_add(((v << 22) as i32 >> 22) as u32)
    }
}

/// `vupkd3d128`. Types: 0 colour (ARGB to x=R y=G z=B w=A), 1 two
/// normalised shorts, 2 packed 2-10-10-10 (x bits 0-9, y 10-19, z 20-29
/// signed, w 30-31 unsigned on 1.0), 3 two halves, 4 four normalised shorts,
/// 5 four halves. Returns `None` for types not implemented yet (6).
pub fn vupkd3d(b: V128, typ: u32) -> Option<V128> {
    let one = 0x3F80_0000;
    Some(match typ {
        0 => {
            let w = b[3];
            [
                one | (w >> 16 & 0xFF),
                one | (w >> 8 & 0xFF),
                one | (w & 0xFF),
                one | (w >> 24),
            ]
        }
        1 => [
            short_to_f32((b[3] >> 16) as u16),
            short_to_f32(b[3] as u16),
            0,
            one,
        ],
        2 => {
            let w = b[3];
            [ten_to_f32(w), ten_to_f32(w >> 10), ten_to_f32(w >> 20), one | w >> 30]
        }
        3 => [
            half_to_f32((b[3] >> 16) as u16),
            half_to_f32(b[3] as u16),
            0,
            one,
        ],
        4 => [
            short_to_f32((b[2] >> 16) as u16),
            short_to_f32(b[2] as u16),
            short_to_f32((b[3] >> 16) as u16),
            short_to_f32(b[3] as u16),
        ],
        5 => [
            half_to_f32((b[2] >> 16) as u16),
            half_to_f32(b[2] as u16),
            half_to_f32((b[3] >> 16) as u16),
            half_to_f32(b[3] as u16),
        ],
        _ => return None,
    })
}

/// `vpkd3d128`: packs `b` by `typ` (as for [`vupkd3d`]) and merges the packed
/// word(s) into `d` at the slot chosen by `pack` (1 = one word, 2 or 3 = two
/// words) and `shift`. Returns `None` for combinations not implemented yet.
pub fn vpkd3d(d: V128, b: V128, typ: u32, pack: u32, shift: u32) -> Option<V128> {
    let p: V128 = match typ {
        0 => {
            let [x, y, z, w] = b.map(f32_to_color);
            [0, 0, 0, w << 24 | x << 16 | y << 8 | z]
        }
        1 => [
            0,
            0,
            0,
            (f32_to_short(b[0]) as u32) << 16 | f32_to_short(b[1]) as u32,
        ],
        2 => {
            let ten = |w| f32_to_bits_in(w, 0x403F_FE01, 0x4040_01FF, 0x3FF);
            let two = f32_to_bits_in(b[3], 0x3F80_0000, 0x3F80_0003, 3);
            [0, 0, 0, two << 30 | ten(b[2]) << 20 | ten(b[1]) << 10 | ten(b[0])]
        }
        3 => [
            0,
            0,
            0,
            (f32_to_half(b[0]) as u32) << 16 | f32_to_half(b[1]) as u32,
        ],
        4 => {
            let s = b.map(|w| f32_to_short(w) as u32);
            [0, 0, s[0] << 16 | s[1], s[2] << 16 | s[3]]
        }
        5 => {
            let h = b.map(|w| f32_to_half(w) as u32);
            [0, 0, h[0] << 16 | h[1], h[2] << 16 | h[3]]
        }
        _ => return None,
    };
    let [d0, d1, d2, d3] = d;
    Some(match (pack, shift & 3) {
        (1, 0) => [d0, d1, d2, p[3]],
        (1, 1) => [d0, d1, p[3], d3],
        (1, 2) => [d0, p[3], d2, d3],
        (1, _) => [p[3], d1, d2, d3],
        (2 | 3, 0) => [d0, d1, p[2], p[3]],
        (2 | 3, 1) => [d0, p[2], p[3], d3],
        (2 | 3, 2) => [p[2], p[3], d2, d3],
        (2, _) => [p[3], d1, d2, d3],
        (3, _) => [d0, d1, d2, p[2]],
        _ => return None,
    })
}

// ---------------------------------------------------------------- loads, stores, VSCR

/// `lvsl`: permute control for a left shift by `ea & 15`.
pub fn lvsl(ea: u32) -> V128 {
    let sh = (ea & 15) as u8;
    from_bytes(core::array::from_fn(|i| sh + i as u8))
}
/// `lvsr`: permute control for a right shift by `ea & 15`.
pub fn lvsr(ea: u32) -> V128 {
    let sh = (ea & 15) as u8;
    from_bytes(core::array::from_fn(|i| 16 - sh + i as u8))
}

/// `lvx` and the element loads (`lvebx`, `lvehx`, `lvewx`), which load the
/// whole aligned quadword so the addressed element lands in its slot.
pub fn lvx(m: &GuestMemory, ea: u32) -> V128 {
    m.read_v128(ea & !15)
}
pub fn stvx(m: &GuestMemory, ea: u32, v: V128) {
    m.write_v128(ea & !15, v)
}
pub fn stvebx(m: &GuestMemory, ea: u32, v: V128) {
    m.write_u8(ea, bytes(v)[(ea & 15) as usize]);
}
pub fn stvehx(m: &GuestMemory, ea: u32, v: V128) {
    let ea = ea & !1;
    m.write_u16(ea, halves(v)[((ea & 15) >> 1) as usize]);
}
pub fn stvewx(m: &GuestMemory, ea: u32, v: V128) {
    let ea = ea & !3;
    m.write_u32(ea, v[((ea & 15) >> 2) as usize]);
}

/// `lvlx`: bytes from `ea` to the end of its quadword, left-justified, zero filled.
pub fn lvlx(m: &GuestMemory, ea: u32) -> V128 {
    let k = (ea & 15) as usize;
    let mut o = [0u8; 16];
    for (i, b) in o.iter_mut().enumerate().take(16 - k) {
        *b = m.read_u8(ea + i as u32);
    }
    from_bytes(o)
}
/// `lvrx`: bytes from the start of the quadword up to `ea`, right-justified.
pub fn lvrx(m: &GuestMemory, ea: u32) -> V128 {
    let k = (ea & 15) as usize;
    let base = ea & !15;
    let mut o = [0u8; 16];
    for i in 0..k {
        o[16 - k + i] = m.read_u8(base + i as u32);
    }
    from_bytes(o)
}
/// `stvlx`: the leftmost `16 - (ea & 15)` bytes to `ea` onwards.
pub fn stvlx(m: &GuestMemory, ea: u32, v: V128) {
    let k = (ea & 15) as usize;
    let b = bytes(v);
    for (i, &x) in b.iter().enumerate().take(16 - k) {
        m.write_u8(ea + i as u32, x);
    }
}
/// `stvrx`: the rightmost `ea & 15` bytes to the start of the quadword.
pub fn stvrx(m: &GuestMemory, ea: u32, v: V128) {
    let k = (ea & 15) as usize;
    let base = ea & !15;
    let b = bytes(v);
    for i in 0..k {
        m.write_u8(base + i as u32, b[16 - k + i]);
    }
}

pub fn mfvscr(vscr: u32) -> V128 {
    [0, 0, 0, vscr]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_order_is_big_endian() {
        let v = [0x0001_0203, 0x0405_0607, 0x0809_0A0B, 0x0C0D_0E0F];
        assert_eq!(bytes(v)[0], 0x00);
        assert_eq!(bytes(v)[15], 0x0F);
        assert_eq!(halves(v)[1], 0x0203);
        assert_eq!(from_bytes(bytes(v)), v);
        assert_eq!(from_halves(halves(v)), v);
    }

    #[test]
    fn half_round_trip() {
        for h in [0x3800u16, 0xB800, 0x3C00, 0x0001, 0x7BFF] {
            assert_eq!(f32_to_half(half_to_f32(h)), h, "{h:#x}");
        }
    }
}
