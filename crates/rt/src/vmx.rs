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
#[inline]
pub fn vadduwm(a: V128, b: V128) -> V128 {
    map_w(a, b, u32::wrapping_add)
}
pub fn vsububm(a: V128, b: V128) -> V128 {
    map_b(a, b, u8::wrapping_sub)
}
pub fn vsubuhm(a: V128, b: V128) -> V128 {
    map_h(a, b, u16::wrapping_sub)
}
#[inline]
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
#[inline]
pub fn vsl(a: V128, b: V128) -> V128 {
    from_u128(to_u128(a) << (b[3] & 7))
}
#[inline]
pub fn vsr(a: V128, b: V128) -> V128 {
    from_u128(to_u128(a) >> (b[3] & 7))
}
/// `vslo`: whole vector left by whole bytes, count in bits 1..4 of the last byte of `b`.
#[inline]
pub fn vslo(a: V128, b: V128) -> V128 {
    let n = (b[3] >> 3) & 0xF;
    from_u128(if n == 0 {
        to_u128(a)
    } else {
        to_u128(a) << (8 * n)
    })
}
#[inline]
pub fn vsro(a: V128, b: V128) -> V128 {
    let n = (b[3] >> 3) & 0xF;
    from_u128(if n == 0 {
        to_u128(a)
    } else {
        to_u128(a) >> (8 * n)
    })
}

/// `vsldoi`: bytes `sh..sh+16` of `a || b`.
#[inline]
pub fn vsldoi(a: V128, b: V128, sh: u32) -> V128 {
    let sh = sh & 15;
    if sh == 0 {
        a
    } else {
        from_u128(to_u128(a) << (8 * sh) | to_u128(b) >> (128 - 8 * sh))
    }
}

// ---------------------------------------------------------------- logical, select, permute

#[inline]
pub fn vand(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x & y)
}
#[inline]
pub fn vandc(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x & !y)
}
#[inline]
pub fn vor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x | y)
}
#[inline]
pub fn vnor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| !(x | y))
}
#[inline]
pub fn vxor(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| x ^ y)
}
/// `vsel`: bits of `b` where `c` is set, else bits of `a`.
#[inline]
pub fn vsel(a: V128, b: V128, c: V128) -> V128 {
    [0, 1, 2, 3].map(|i| (a[i] & !c[i]) | (b[i] & c[i]))
}
/// `vperm`: byte `i` is byte `c[i] & 31` of `a || b`.
#[inline]
pub fn vperm(a: V128, b: V128, c: V128) -> V128 {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64"
    ))]
    {
        simd::perm(a, b, c)
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64"
    )))]
    {
        vperm_scalar(a, b, c)
    }
}
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "ssse3"),
        target_arch = "aarch64"
    ),
    allow(dead_code)
)]
fn vperm_scalar(a: V128, b: V128, c: V128) -> V128 {
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
#[inline]
pub fn vmrghw(a: V128, b: V128) -> V128 {
    [a[0], b[0], a[1], b[1]]
}
#[inline]
pub fn vmrglw(a: V128, b: V128) -> V128 {
    [a[2], b[2], a[3], b[3]]
}
#[inline]
pub fn vspltb(b: V128, i: u32) -> V128 {
    let x = bytes(b)[(i & 15) as usize] as u32;
    [x * 0x0101_0101; 4]
}
#[inline]
pub fn vsplth(b: V128, i: u32) -> V128 {
    let x = halves(b)[(i & 7) as usize] as u32;
    [x << 16 | x; 4]
}
#[inline]
pub fn vspltw(b: V128, i: u32) -> V128 {
    [b[(i & 3) as usize]; 4]
}
/// Splat of a 5-bit signed immediate (already sign-extended by the caller).
#[inline]
pub fn vspltisb(simm: i32) -> V128 {
    [(simm as u8 as u32) * 0x0101_0101; 4]
}
#[inline]
pub fn vspltish(simm: i32) -> V128 {
    let x = simm as u16 as u32;
    [x << 16 | x; 4]
}
#[inline]
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
#[inline]
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

// The float helpers below that translated code calls most have an SSE path
// (module `sse`) on x86-64 and a NEON path (module `neon`) on AArch64, each
// giving bit for bit the scalar result. The scalar code stays the definition:
// it is the only path elsewhere, and with a fast path it runs out of line
// (cold) for the cases the fast path hands back, a NaN operand.
#[cfg(target_arch = "aarch64")]
use neon as simd;
#[cfg(target_arch = "x86_64")]
use sse as simd;

#[inline]
pub fn vaddfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::add(a, b) {
        return r;
    }
    vaddfp_scalar(a, b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vaddfp_scalar(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| x + y)
}
#[inline]
pub fn vsubfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::sub(a, b) {
        return r;
    }
    vsubfp_scalar(a, b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vsubfp_scalar(a: V128, b: V128) -> V128 {
    f2(a, b, |x, y| x - y)
}
/// VMX128 `vmulfp128` (standard VMX has no plain multiply).
#[inline]
pub fn vmulfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::mul(a, b) {
        return r;
    }
    vmulfp_scalar(a, b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vmulfp_scalar(a: V128, b: V128) -> V128 {
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
#[inline]
pub fn vmaddfp(a: V128, c: V128, b: V128) -> V128 {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "fma"),
        target_arch = "aarch64"
    ))]
    if let Some(r) = simd::madd(a, c, b) {
        return r;
    }
    vmaddfp_scalar(a, c, b)
}
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "fma"),
        target_arch = "aarch64"
    ),
    cold,
    inline(never)
)]
fn vmaddfp_scalar(a: V128, c: V128, b: V128) -> V128 {
    f3(a, b, c, |x, y, z| x.mul_add(z, y))
}
/// `vnmsubfp vD,vA,vC,vB`: `-(a * c - b)`, fused.
#[inline]
pub fn vnmsubfp(a: V128, c: V128, b: V128) -> V128 {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "fma"),
        target_arch = "aarch64"
    ))]
    if let Some(r) = simd::nmsub(a, c, b) {
        return r;
    }
    vnmsubfp_scalar(a, c, b)
}
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "fma"),
        target_arch = "aarch64"
    ),
    cold,
    inline(never)
)]
fn vnmsubfp_scalar(a: V128, c: V128, b: V128) -> V128 {
    f3(a, b, c, |x, y, z| -(x.mul_add(z, -y)))
}

#[inline]
pub fn vmaxfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::max(a, b) {
        return r;
    }
    vmaxfp_scalar(a, b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vmaxfp_scalar(a: V128, b: V128) -> V128 {
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
#[inline]
pub fn vminfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::min(a, b) {
        return r;
    }
    vminfp_scalar(a, b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vminfp_scalar(a: V128, b: V128) -> V128 {
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

#[inline]
pub fn vrefp(b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::refp(b) {
        return r;
    }
    vrefp_scalar(b)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vrefp_scalar(b: V128) -> V128 {
    f1(b, |x| 1.0 / x)
}
#[inline]
pub fn vrsqrtefp(b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::rsqrte(b) {
        return r;
    }
    vrsqrtefp_scalar(b)
}
// Negative inputs are checked explicitly: the optimiser assumes `sqrt` of a
// non-NaN is never NaN, so testing the result afterwards is not reliable.
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
fn vrsqrtefp_scalar(b: V128) -> V128 {
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
// The SSE compares need no scalar fallback: a NaN compares false either way.
#[inline]
pub fn vcmpeqfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        simd::cmpeq(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vcmpeqfp_scalar(a, b)
    }
}
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn vcmpeqfp_scalar(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x == y)
}
#[inline]
pub fn vcmpgefp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        simd::cmpge(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vcmpgefp_scalar(a, b)
    }
}
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn vcmpgefp_scalar(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x >= y)
}
#[inline]
pub fn vcmpgtfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        simd::cmpgt(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vcmpgtfp_scalar(a, b)
    }
}
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn vcmpgtfp_scalar(a: V128, b: V128) -> V128 {
    fcmp(a, b, |x, y| x > y)
}
/// `vcmpbfp`: bit 0 set when `a > b` fails `a <= b`, bit 1 when `a >= -b` fails.
#[inline]
pub fn vcmpbfp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        simd::cmpb(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vcmpbfp_scalar(a, b)
    }
}
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn vcmpbfp_scalar(a: V128, b: V128) -> V128 {
    map_w(a, b, |x, y| {
        let (p, q) = (fl(x), fl(y));
        let le = p <= q;
        let ge = p >= -q;
        (!le as u32) << 31 | (!ge as u32) << 30
    })
}
/// CR6 for `vcmpbfp.`: EQ when every element is within bounds.
#[inline]
pub fn cr6_bounds(v: V128) -> u32 {
    if v == [0; 4] {
        0b0010
    } else {
        0
    }
}

// ---------------------------------------------------------------- VMX128 extras

/// `vmsum3fp128`: dot product of elements 0..3, in every element.
#[inline]
pub fn vmsum3fp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::dot(a, b, 3) {
        return r;
    }
    dot(a, b, 3)
}
/// `vmsum4fp128`: dot product of all four elements, in every element.
#[inline]
pub fn vmsum4fp(a: V128, b: V128) -> V128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(r) = simd::dot(a, b, 4) {
        return r;
    }
    dot(a, b, 4)
}
#[cfg_attr(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    cold,
    inline(never)
)]
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

/// SSE paths for the hot helpers, bit for bit the scalar results: inputs
/// and results are flushed and a NaN result becomes the default NaN without
/// a branch. A NaN operand, which must come out quieted in operand order,
/// returns `None` so the caller runs the scalar code. Lane `i` of an SSE
/// register is word `i` of a [`V128`].
///
/// The `unsafe` blocks only call intrinsics: SSE and SSE2 are part of every
/// x86-64 CPU, and the FMA and SSSE3 paths are compiled only when the build
/// enables those features.
#[cfg(target_arch = "x86_64")]
mod sse {
    use super::{DEFAULT_NAN, V128};
    use core::arch::x86_64::*;

    #[inline(always)]
    fn ps(v: V128) -> __m128 {
        // SAFETY: both are 16 bytes of plain data.
        unsafe { core::mem::transmute::<V128, __m128>(v) }
    }
    #[inline(always)]
    fn words(x: __m128) -> V128 {
        // SAFETY: as in `ps`.
        unsafe { core::mem::transmute::<__m128, V128>(x) }
    }
    #[inline(always)]
    fn splat(w: u32) -> __m128 {
        unsafe { _mm_castsi128_ps(_mm_set1_epi32(w as i32)) }
    }
    /// Bits of `x` where `m` is set, else bits of `y`.
    #[inline(always)]
    fn select(m: __m128, x: __m128, y: __m128) -> __m128 {
        unsafe { _mm_or_ps(_mm_and_ps(m, x), _mm_andnot_ps(m, y)) }
    }
    /// [`super::ftz`] per lane.
    #[inline(always)]
    fn ftz(x: __m128) -> __m128 {
        unsafe {
            let mag = _mm_and_si128(_mm_castps_si128(x), _mm_set1_epi32(0x7FFF_FFFF));
            let small = _mm_castsi128_ps(_mm_cmplt_epi32(mag, _mm_set1_epi32(0x0080_0000)));
            _mm_andnot_ps(_mm_and_ps(small, splat(0x7FFF_FFFF)), x)
        }
    }
    /// Lanes where `x` or `y` is a NaN, one bit per lane.
    #[inline(always)]
    fn nans(x: __m128, y: __m128) -> i32 {
        unsafe { _mm_movemask_ps(_mm_cmpunord_ps(x, y)) }
    }
    /// [`super::out`] per lane.
    #[inline(always)]
    fn out(r: __m128) -> V128 {
        unsafe {
            let r = ftz(r);
            words(select(_mm_cmpunord_ps(r, r), splat(DEFAULT_NAN), r))
        }
    }
    #[inline(always)]
    fn bin(a: V128, b: V128, f: impl FnOnce(__m128, __m128) -> __m128) -> Option<V128> {
        let (x, y) = (ps(a), ps(b));
        if nans(x, y) != 0 {
            return None;
        }
        Some(out(f(ftz(x), ftz(y))))
    }
    #[inline(always)]
    fn un(b: V128, f: impl FnOnce(__m128) -> __m128) -> Option<V128> {
        let x = ps(b);
        if nans(x, x) != 0 {
            return None;
        }
        Some(out(f(ftz(x))))
    }
    #[inline(always)]
    fn cmp(a: V128, b: V128, f: impl FnOnce(__m128, __m128) -> __m128) -> V128 {
        words(f(ftz(ps(a)), ftz(ps(b))))
    }

    #[inline(always)]
    pub fn add(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| _mm_add_ps(x, y)) }
    }
    #[inline(always)]
    pub fn sub(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| _mm_sub_ps(x, y)) }
    }
    #[inline(always)]
    pub fn mul(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| _mm_mul_ps(x, y)) }
    }
    // `maxps` / `minps` give the second operand on equality; the scalar code
    // picks +0 (max) or -0 (min) there, which is the AND / OR of the bits.
    #[inline(always)]
    pub fn max(a: V128, b: V128) -> Option<V128> {
        unsafe {
            bin(a, b, |x, y| {
                select(_mm_cmpeq_ps(x, y), _mm_and_ps(x, y), _mm_max_ps(x, y))
            })
        }
    }
    #[inline(always)]
    pub fn min(a: V128, b: V128) -> Option<V128> {
        unsafe {
            bin(a, b, |x, y| {
                select(_mm_cmpeq_ps(x, y), _mm_or_ps(x, y), _mm_min_ps(x, y))
            })
        }
    }
    #[inline(always)]
    pub fn refp(b: V128) -> Option<V128> {
        unsafe { un(b, |x| _mm_div_ps(_mm_set1_ps(1.0), x)) }
    }
    /// `sqrtps` of a negative lane gives a NaN, which `out` makes the default NaN.
    #[inline(always)]
    pub fn rsqrte(b: V128) -> Option<V128> {
        unsafe { un(b, |x| _mm_div_ps(_mm_set1_ps(1.0), _mm_sqrt_ps(x))) }
    }
    /// `a * c + b`, one rounding.
    #[cfg(target_feature = "fma")]
    #[inline(always)]
    pub fn madd(a: V128, c: V128, b: V128) -> Option<V128> {
        unsafe {
            let (x, y, z) = (ps(a), ps(b), ps(c));
            if nans(x, y) | nans(z, z) != 0 {
                return None;
            }
            Some(out(_mm_fmadd_ps(ftz(x), ftz(z), ftz(y))))
        }
    }
    /// `-(a * c - b)`: the sign flip comes after rounding, so an exact zero
    /// is -0 as in the scalar code (`fnmadd` would give +0).
    #[cfg(target_feature = "fma")]
    #[inline(always)]
    pub fn nmsub(a: V128, c: V128, b: V128) -> Option<V128> {
        unsafe {
            let (x, y, z) = (ps(a), ps(b), ps(c));
            if nans(x, y) | nans(z, z) != 0 {
                return None;
            }
            let r = _mm_fmsub_ps(ftz(x), ftz(z), ftz(y));
            Some(out(_mm_xor_ps(r, splat(0x8000_0000))))
        }
    }
    /// Products in parallel, then the scalar code's left-to-right sum, each
    /// step flushed, in lane 0.
    #[inline(always)]
    pub fn dot(a: V128, b: V128, n: usize) -> Option<V128> {
        unsafe {
            let (x, y) = (ps(a), ps(b));
            if nans(x, y) & ((1 << n) - 1) != 0 {
                return None;
            }
            let p = ftz(_mm_mul_ps(ftz(x), ftz(y)));
            let mut s = ftz(_mm_add_ss(_mm_setzero_ps(), p));
            s = ftz(_mm_add_ss(s, _mm_shuffle_ps::<0x55>(p, p)));
            s = ftz(_mm_add_ss(s, _mm_shuffle_ps::<0xAA>(p, p)));
            if n == 4 {
                s = ftz(_mm_add_ss(s, _mm_shuffle_ps::<0xFF>(p, p)));
            }
            Some(out(_mm_shuffle_ps::<0>(s, s)))
        }
    }
    #[inline(always)]
    pub fn cmpeq(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| _mm_cmpeq_ps(x, y)) }
    }
    #[inline(always)]
    pub fn cmpge(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| _mm_cmpge_ps(x, y)) }
    }
    #[inline(always)]
    pub fn cmpgt(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| _mm_cmpgt_ps(x, y)) }
    }
    #[inline(always)]
    pub fn cmpb(a: V128, b: V128) -> V128 {
        unsafe {
            cmp(a, b, |p, q| {
                let le = _mm_cmple_ps(p, q);
                let ge = _mm_cmpge_ps(p, _mm_xor_ps(q, splat(0x8000_0000)));
                _mm_or_ps(
                    _mm_andnot_ps(le, splat(0x8000_0000)),
                    _mm_andnot_ps(ge, splat(0x4000_0000)),
                )
            })
        }
    }

    /// `vperm` with `pshufb`: bytes swapped to big-endian order within each
    /// word, picked from `a` or `b` by bit 4 of the index, swapped back.
    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub fn perm(a: V128, b: V128, c: V128) -> V128 {
        unsafe {
            let swap = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            let be = |v: V128| _mm_shuffle_epi8(_mm_castps_si128(ps(v)), swap);
            let k = _mm_and_si128(be(c), _mm_set1_epi8(31));
            let hi = _mm_cmpgt_epi8(k, _mm_set1_epi8(15));
            let r = _mm_or_si128(
                _mm_and_si128(hi, _mm_shuffle_epi8(be(b), k)),
                _mm_andnot_si128(hi, _mm_shuffle_epi8(be(a), k)),
            );
            words(_mm_castsi128_ps(_mm_shuffle_epi8(r, swap)))
        }
    }
}

/// NEON paths for the same helpers, built the same way as module `sse`:
/// inputs and results are flushed explicitly (FPCR.FZ is not set on Android),
/// a NaN result becomes the default NaN, and a NaN operand returns `None` so
/// the caller runs the scalar code. Lane `i` is word `i` of a [`V128`].
///
/// The `unsafe` blocks only call intrinsics: NEON, FMA included, is part of
/// every AArch64 CPU.
#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
mod neon {
    use super::{DEFAULT_NAN, V128};
    use core::arch::aarch64::*;

    #[inline(always)]
    fn u(v: V128) -> uint32x4_t {
        // SAFETY: both are 16 bytes of plain data.
        unsafe { core::mem::transmute::<V128, uint32x4_t>(v) }
    }
    #[inline(always)]
    fn words(x: uint32x4_t) -> V128 {
        // SAFETY: as in `u`.
        unsafe { core::mem::transmute::<uint32x4_t, V128>(x) }
    }
    #[inline(always)]
    fn f(x: uint32x4_t) -> float32x4_t {
        unsafe { vreinterpretq_f32_u32(x) }
    }
    #[inline(always)]
    fn w(x: float32x4_t) -> uint32x4_t {
        unsafe { vreinterpretq_u32_f32(x) }
    }
    /// [`super::ftz`] per lane.
    #[inline(always)]
    fn ftz(x: float32x4_t) -> float32x4_t {
        unsafe {
            let x = w(x);
            let mag = vandq_u32(x, vdupq_n_u32(0x7FFF_FFFF));
            let small = vcltq_u32(mag, vdupq_n_u32(0x0080_0000));
            f(vbicq_u32(x, vandq_u32(small, vdupq_n_u32(0x7FFF_FFFF))))
        }
    }
    /// All ones in the lanes of `x` that are not NaNs.
    #[inline(always)]
    fn ordered(x: uint32x4_t) -> uint32x4_t {
        unsafe { vceqq_f32(f(x), f(x)) }
    }
    /// True when any lane of `x` or `y` is a NaN.
    #[inline(always)]
    fn nans(x: uint32x4_t, y: uint32x4_t) -> bool {
        unsafe { vminvq_u32(vandq_u32(ordered(x), ordered(y))) == 0 }
    }
    /// [`super::out`] per lane.
    #[inline(always)]
    fn out(r: float32x4_t) -> V128 {
        unsafe {
            let r = ftz(r);
            words(vbslq_u32(vceqq_f32(r, r), w(r), vdupq_n_u32(DEFAULT_NAN)))
        }
    }
    #[inline(always)]
    fn bin(
        a: V128,
        b: V128,
        op: impl FnOnce(float32x4_t, float32x4_t) -> float32x4_t,
    ) -> Option<V128> {
        let (x, y) = (u(a), u(b));
        if nans(x, y) {
            return None;
        }
        Some(out(op(ftz(f(x)), ftz(f(y)))))
    }
    #[inline(always)]
    fn un(b: V128, op: impl FnOnce(float32x4_t) -> float32x4_t) -> Option<V128> {
        let x = u(b);
        if nans(x, x) {
            return None;
        }
        Some(out(op(ftz(f(x)))))
    }
    #[inline(always)]
    fn cmp(a: V128, b: V128, op: impl FnOnce(float32x4_t, float32x4_t) -> uint32x4_t) -> V128 {
        words(op(ftz(f(u(a))), ftz(f(u(b)))))
    }

    #[inline(always)]
    pub fn add(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| vaddq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn sub(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| vsubq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn mul(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| vmulq_f32(x, y)) }
    }
    // `fmax` / `fmin` order +0 above -0, the scalar code's choice for equal zeros.
    #[inline(always)]
    pub fn max(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| vmaxq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn min(a: V128, b: V128) -> Option<V128> {
        unsafe { bin(a, b, |x, y| vminq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn refp(b: V128) -> Option<V128> {
        unsafe { un(b, |x| vdivq_f32(vdupq_n_f32(1.0), x)) }
    }
    /// `fsqrt` of a negative lane gives a NaN, which `out` makes the default NaN.
    #[inline(always)]
    pub fn rsqrte(b: V128) -> Option<V128> {
        unsafe { un(b, |x| vdivq_f32(vdupq_n_f32(1.0), vsqrtq_f32(x))) }
    }
    /// `a * c + b`, one rounding.
    #[inline(always)]
    pub fn madd(a: V128, c: V128, b: V128) -> Option<V128> {
        unsafe {
            let (x, y, z) = (u(a), u(b), u(c));
            if nans(x, y) || nans(z, z) {
                return None;
            }
            Some(out(vfmaq_f32(ftz(f(y)), ftz(f(x)), ftz(f(z)))))
        }
    }
    /// `-(a * c - b)`: rounded as `a * c + -b`, then negated, so an exact
    /// zero is -0 as in the scalar code (`fmls`, `b - a * c`, would give +0).
    #[inline(always)]
    pub fn nmsub(a: V128, c: V128, b: V128) -> Option<V128> {
        unsafe {
            let (x, y, z) = (u(a), u(b), u(c));
            if nans(x, y) || nans(z, z) {
                return None;
            }
            let r = vfmaq_f32(vnegq_f32(ftz(f(y))), ftz(f(x)), ftz(f(z)));
            Some(out(vnegq_f32(r)))
        }
    }
    /// Products in parallel, then the scalar code's left-to-right sum, each
    /// step flushed. Only the first `n` lanes are checked for NaNs.
    #[inline(always)]
    pub fn dot(a: V128, b: V128, n: usize) -> Option<V128> {
        unsafe {
            let (x, y) = (u(a), u(b));
            if words(vandq_u32(ordered(x), ordered(y)))[..n].contains(&0) {
                return None;
            }
            let p = words(w(ftz(vmulq_f32(ftz(f(x)), ftz(f(y))))));
            let mut s = 0.0f32;
            for &q in &p[..n] {
                s = super::ftz(s + f32::from_bits(q));
            }
            Some(out(vdupq_n_f32(s)))
        }
    }
    #[inline(always)]
    pub fn cmpeq(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| vceqq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn cmpge(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| vcgeq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn cmpgt(a: V128, b: V128) -> V128 {
        unsafe { cmp(a, b, |x, y| vcgtq_f32(x, y)) }
    }
    #[inline(always)]
    pub fn cmpb(a: V128, b: V128) -> V128 {
        unsafe {
            cmp(a, b, |p, q| {
                let le = vcleq_f32(p, q);
                let ge = vcgeq_f32(p, vnegq_f32(q));
                vorrq_u32(
                    vbicq_u32(vdupq_n_u32(0x8000_0000), le),
                    vbicq_u32(vdupq_n_u32(0x4000_0000), ge),
                )
            })
        }
    }

    /// `vperm` with `tbl` over `a || b`: bytes swapped to big-endian order
    /// within each word, looked up by the low five bits, swapped back.
    #[inline(always)]
    pub fn perm(a: V128, b: V128, c: V128) -> V128 {
        unsafe {
            let be = |v: V128| vrev32q_u8(vreinterpretq_u8_u32(u(v)));
            let k = vandq_u8(be(c), vdupq_n_u8(31));
            let r = vqtbl2q_u8(uint8x16x2_t(be(a), be(b)), k);
            words(vreinterpretq_u32_u8(vrev32q_u8(r)))
        }
    }
}

/// `vpermwi128`: element `i` is `b[(imm >> (6 - 2i)) & 3]`.
#[inline]
pub fn vpermwi(b: V128, imm: u32) -> V128 {
    core::array::from_fn(|i| b[(imm >> (6 - 2 * i) & 3) as usize])
}

/// `vrlimi128`: rotate `b` left by `rot` elements, then insert into `d` the
/// elements whose mask bit is set (mask bit 8 is element 0).
#[inline]
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
    let c = if f >= h {
        hi
    } else if f >= l {
        w
    } else {
        lo
    };
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
            [
                ten_to_f32(w),
                ten_to_f32(w >> 10),
                ten_to_f32(w >> 20),
                one | w >> 30,
            ]
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
            [
                0,
                0,
                0,
                two << 30 | ten(b[2]) << 20 | ten(b[1]) << 10 | ten(b[0]),
            ]
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
#[inline]
pub fn lvsl(ea: u32) -> V128 {
    // Word-wise so inlined unoptimised callers stay small; bytes never carry.
    let sh = (ea & 15) * 0x0101_0101;
    [0x0001_0203, 0x0405_0607, 0x0809_0A0B, 0x0C0D_0E0F].map(|w| w + sh)
}
/// `lvsr`: permute control for a right shift by `ea & 15`.
#[inline]
pub fn lvsr(ea: u32) -> V128 {
    let sh = (ea & 15) * 0x0101_0101;
    [0x1011_1213, 0x1415_1617, 0x1819_1A1B, 0x1C1D_1E1F].map(|w| w - sh)
}

/// `lvx` and the element loads (`lvebx`, `lvehx`, `lvewx`), which load the
/// whole aligned quadword so the addressed element lands in its slot.
#[inline]
pub fn lvx(m: &GuestMemory, ea: u32) -> V128 {
    m.read_v128(ea & !15)
}
#[inline]
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

#[inline]
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

    /// Lane values at the edges: zeros, denormals, the smallest and largest
    /// normals, infinities, quiet and signalling NaNs of both signs.
    const EDGES: [u32; 18] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0001,
        0x00FF_FFFF,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0001,
        0x7F80_0001,
        0x3F80_0000,
        0xBF80_0000,
        0x4040_0000,
        0x1F80_0000,
    ];

    /// Deterministic vectors: each lane an edge value or random bits (whose
    /// spread of exponents gives under- and overflowing results).
    fn vectors() -> impl FnMut() -> V128 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            core::array::from_fn(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                match s & 3 {
                    0 | 2 => EDGES[(s >> 8) as usize % EDGES.len()],
                    // Random denormals, or infinities and NaNs with random payloads.
                    1 => (s >> 32) as u32 & 0x807F_FFFF | if s & 4 == 0 { 0 } else { 0x7F80_0000 },
                    _ => (s >> 32) as u32,
                }
            })
        }
    }

    #[test]
    fn lvsl_lvsr_match_bytewise() {
        for sh in 0..16u8 {
            let l = from_bytes(core::array::from_fn(|i| sh + i as u8));
            let r = from_bytes(core::array::from_fn(|i| 16 - sh + i as u8));
            // Low bits only, as an address.
            let ea = 0x8000_0010 | sh as u32;
            assert_eq!(lvsl(ea), l, "lvsl {sh}");
            assert_eq!(lvsr(ea), r, "lvsr {sh}");
        }
    }

    #[test]
    fn fast_paths_match_scalar() {
        let mut next = vectors();
        // Millions in a release build (run on every target with a fast path).
        let n = if cfg!(debug_assertions) {
            200_000
        } else {
            4_000_000
        };
        for _ in 0..n {
            let (a, b, c) = (next(), next(), next());
            assert_eq!(
                vaddfp(a, b),
                vaddfp_scalar(a, b),
                "vaddfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vsubfp(a, b),
                vsubfp_scalar(a, b),
                "vsubfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vmulfp(a, b),
                vmulfp_scalar(a, b),
                "vmulfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vmaxfp(a, b),
                vmaxfp_scalar(a, b),
                "vmaxfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vminfp(a, b),
                vminfp_scalar(a, b),
                "vminfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vrefp(b),
                vrefp_scalar(b),
                "vrefp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vrsqrtefp(b),
                vrsqrtefp_scalar(b),
                "vrsqrtefp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vmaddfp(a, c, b),
                vmaddfp_scalar(a, c, b),
                "vmaddfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vnmsubfp(a, c, b),
                vnmsubfp_scalar(a, c, b),
                "vnmsubfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vmsum3fp(a, b),
                dot(a, b, 3),
                "vmsum3fp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vmsum4fp(a, b),
                dot(a, b, 4),
                "vmsum4fp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vcmpeqfp(a, b),
                vcmpeqfp_scalar(a, b),
                "vcmpeqfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vcmpgefp(a, b),
                vcmpgefp_scalar(a, b),
                "vcmpgefp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vcmpgtfp(a, b),
                vcmpgtfp_scalar(a, b),
                "vcmpgtfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vcmpbfp(a, b),
                vcmpbfp_scalar(a, b),
                "vcmpbfp {a:08x?} {b:08x?} {c:08x?}"
            );
            assert_eq!(
                vperm(a, b, c),
                vperm_scalar(a, b, c),
                "vperm {a:08x?} {b:08x?} {c:08x?}"
            );
        }
    }

    #[test]
    fn edge_results() {
        let one = [0x3F80_0000; 4];
        // An exact zero from vnmsubfp is -0: the sign flips after rounding.
        assert_eq!(vnmsubfp(one, one, one), [0x8000_0000; 4]);
        assert_eq!(vmaddfp(one, one, [0xBF80_0000; 4]), [0; 4]);
        // Equal zeros: max takes +0, min takes -0.
        let (p, n) = ([0; 4], [0x8000_0000; 4]);
        assert_eq!(vmaxfp(n, p), p);
        assert_eq!(vminfp(p, n), n);
        // Invalid operations give the default NaN; NaN operands come out
        // quieted, the first one in operand order.
        let inf = [0x7F80_0000; 4];
        assert_eq!(vsubfp(inf, inf), [DEFAULT_NAN; 4]);
        assert_eq!(vrsqrtefp(one.map(|w| w | 0x8000_0000)), [DEFAULT_NAN; 4]);
        let snan = [0x7F80_0001, 0x3F80_0000, 0x3F80_0000, 0x3F80_0000];
        assert_eq!(vaddfp(one, snan)[0], 0x7FC0_0001);
        assert_eq!(vmsum4fp(one, snan), [0x7FC0_0001; 4]);
        // A NaN in element 3 does not reach vmsum3fp.
        assert_eq!(vmsum3fp(one, [0x3F80_0000, 0, 0, 0x7FC0_0000]), one);
        // Denormal products are flushed.
        assert_eq!(vmulfp([0x0080_0000; 4], [0x3F00_0000; 4]), [0; 4]);
    }

    /// Fast path against scalar code for the hot helpers:
    /// `cargo test -p xenon-rt --release vmx_bench -- --ignored --nocapture`
    /// (and with `RUSTFLAGS="-C target-cpu=native"` for the FMA and `pshufb`
    /// paths a game build uses).
    #[test]
    #[ignore = "micro-benchmark"]
    fn vmx_bench() {
        use std::hint::black_box;
        fn time(f: impl Fn(V128, V128, V128) -> V128) -> f64 {
            let (a, b, c) = (
                [0x3F80_0000, 0x4000_0000, 0xBF00_0000, 0x3E80_0000],
                [0x3F00_0000, 0xC040_0000, 0x3F80_0000, 0x4080_0000],
                [0x0302_0100, 0x1716_1514, 0x0B0A_0908, 0x1F1E_1D1C],
            );
            let n = 20_000_000u32;
            let t = std::time::Instant::now();
            for _ in 0..n {
                black_box(f(black_box(a), black_box(b), black_box(c)));
            }
            t.elapsed().as_secs_f64() * 1e9 / n as f64
        }
        type Op = fn(V128, V128, V128) -> V128;
        let ops: [(&str, Op, Op); 16] = [
            (
                "vaddfp",
                |a, b, _| vaddfp(a, b),
                |a, b, _| vaddfp_scalar(a, b),
            ),
            (
                "vsubfp",
                |a, b, _| vsubfp(a, b),
                |a, b, _| vsubfp_scalar(a, b),
            ),
            (
                "vminfp",
                |a, b, _| vminfp(a, b),
                |a, b, _| vminfp_scalar(a, b),
            ),
            ("vrefp", |a, _, _| vrefp(a), |a, _, _| vrefp_scalar(a)),
            (
                "vcmpeqfp",
                |a, b, _| vcmpeqfp(a, b),
                |a, b, _| vcmpeqfp_scalar(a, b),
            ),
            (
                "vcmpgefp",
                |a, b, _| vcmpgefp(a, b),
                |a, b, _| vcmpgefp_scalar(a, b),
            ),
            (
                "vmulfp",
                |a, b, _| vmulfp(a, b),
                |a, b, _| vmulfp_scalar(a, b),
            ),
            (
                "vmaddfp",
                |a, b, c| vmaddfp(a, b, c),
                |a, b, c| vmaddfp_scalar(a, b, c),
            ),
            (
                "vnmsubfp",
                |a, b, c| vnmsubfp(a, b, c),
                |a, b, c| vnmsubfp_scalar(a, b, c),
            ),
            (
                "vmaxfp",
                |a, b, _| vmaxfp(a, b),
                |a, b, _| vmaxfp_scalar(a, b),
            ),
            (
                "vrsqrtefp",
                |a, _, _| vrsqrtefp(a),
                |a, _, _| vrsqrtefp_scalar(a),
            ),
            ("vmsum3fp", |a, b, _| vmsum3fp(a, b), |a, b, _| dot(a, b, 3)),
            ("vmsum4fp", |a, b, _| vmsum4fp(a, b), |a, b, _| dot(a, b, 4)),
            (
                "vcmpgtfp",
                |a, b, _| vcmpgtfp(a, b),
                |a, b, _| vcmpgtfp_scalar(a, b),
            ),
            (
                "vcmpbfp",
                |a, b, _| vcmpbfp(a, b),
                |a, b, _| vcmpbfp_scalar(a, b),
            ),
            (
                "vperm",
                |a, b, c| vperm(a, b, c),
                |a, b, c| vperm_scalar(a, b, c),
            ),
        ];
        for (name, fast, scalar) in ops {
            let (f, s) = (time(fast), time(scalar));
            println!(
                "[vmx] {name:10} fast {f:5.2} ns  scalar {s:5.2} ns  x{:.1}",
                s / f
            );
        }
    }

    #[test]
    fn half_round_trip() {
        for h in [0x3800u16, 0xB800, 0x3C00, 0x0001, 0x7BFF] {
            assert_eq!(f32_to_half(half_to_f32(h)), h, "{h:#x}");
        }
    }
}
