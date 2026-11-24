//! Scalar floating point helpers called by translated code.
//!
//! The scalar unit is IEEE double precision and, unlike the vector unit,
//! keeps denormals. NaN results follow PowerPC rules rather than the host's:
//! the first NaN operand (in the order A, B, C) is returned quieted, and an
//! invalid operation gives the positive default NaN.
//!
//! Not modelled yet: the FPSCR exception and result-class bits (only the
//! compare result field is written), and non-default rounding modes for
//! arithmetic. The integer conversions do honour FPSCR[RN].

pub const DEFAULT_NAN: u64 = 0x7FF8_0000_0000_0000;
const QUIET: u64 = 0x0008_0000_0000_0000;

/// FPSCR bits 16..19, the floating point condition code.
pub const FPCC_MASK: u32 = 0x0000_F000;

#[inline]
fn quiet(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | QUIET)
}

#[inline]
fn nan2(a: f64, b: f64) -> Option<f64> {
    if a.is_nan() {
        Some(quiet(a))
    } else if b.is_nan() {
        Some(quiet(b))
    } else {
        None
    }
}

#[inline]
fn nan3(a: f64, b: f64, c: f64) -> Option<f64> {
    nan2(a, b).or_else(|| c.is_nan().then(|| quiet(c)))
}

#[inline]
fn fix(r: f64) -> f64 {
    if r.is_nan() {
        f64::from_bits(DEFAULT_NAN)
    } else {
        r
    }
}

/// Rounds a double result to single precision, keeping it in double format.
#[inline]
pub fn round_single(x: f64) -> f64 {
    if x.is_nan() {
        // Keep the payload's high bits, as the hardware does.
        let b = x.to_bits();
        f64::from_bits(b & 0xFFFF_FFFF_E000_0000)
    } else {
        x as f32 as f64
    }
}

#[inline]
pub fn add(a: f64, b: f64) -> f64 {
    nan2(a, b).unwrap_or_else(|| fix(a + b))
}
#[inline]
pub fn sub(a: f64, b: f64) -> f64 {
    nan2(a, b).unwrap_or_else(|| fix(a - b))
}
#[inline]
pub fn mul(a: f64, c: f64) -> f64 {
    nan2(a, c).unwrap_or_else(|| fix(a * c))
}
#[inline]
pub fn div(a: f64, b: f64) -> f64 {
    nan2(a, b).unwrap_or_else(|| fix(a / b))
}

/// `fmadd`: `a * c + b`, fused.
#[inline]
pub fn madd(a: f64, c: f64, b: f64) -> f64 {
    nan3(a, b, c).unwrap_or_else(|| fix(a.mul_add(c, b)))
}
/// `fmsub`: `a * c - b`, fused.
#[inline]
pub fn msub(a: f64, c: f64, b: f64) -> f64 {
    nan3(a, b, c).unwrap_or_else(|| fix(a.mul_add(c, -b)))
}
/// Negates unless the value is a NaN (the `fnm*` forms leave NaNs alone).
#[inline]
pub fn neg_num(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        -x
    }
}

#[inline]
pub fn sqrt(b: f64) -> f64 {
    if b.is_nan() {
        quiet(b)
    } else if b < 0.0 {
        // Checked explicitly: the optimiser assumes `sqrt` of a non-NaN is
        // never NaN, so testing the result is not reliable.
        f64::from_bits(DEFAULT_NAN)
    } else {
        b.sqrt()
    }
}

/// `fres`: reciprocal estimate, computed exactly and rounded to single.
#[inline]
pub fn fres(b: f64) -> f64 {
    if b.is_nan() {
        round_single(quiet(b))
    } else {
        (1.0 / b) as f32 as f64
    }
}

/// `frsqrte`: reciprocal square root estimate, computed exactly.
#[inline]
pub fn frsqrte(b: f64) -> f64 {
    if b.is_nan() {
        quiet(b)
    } else if b < 0.0 {
        f64::from_bits(DEFAULT_NAN)
    } else {
        1.0 / b.sqrt()
    }
}

/// `fsel`: `c` if `a >= 0` (including -0), else `b`; a NaN `a` picks `b`.
#[inline]
pub fn fsel(a: f64, c: f64, b: f64) -> f64 {
    if a >= 0.0 {
        c
    } else {
        b
    }
}

/// `frsp`.
#[inline]
pub fn frsp(b: f64) -> f64 {
    if b.is_nan() {
        round_single(quiet(b))
    } else {
        b as f32 as f64
    }
}

fn round_mode(x: f64, rn: u32) -> f64 {
    match rn & 3 {
        0 => x.round_ties_even(),
        1 => x.trunc(),
        2 => x.ceil(),
        _ => x.floor(),
    }
}

/// `fctiw` / `fctiwz` (pass `rn = 1`). The integer is in the low word; the
/// high word is the sign extension, or zero for a NaN or saturated result
/// that the emulator reports that way.
pub fn fctiw(b: f64, rn: u32) -> f64 {
    let v: u64 = if b.is_nan() {
        0x8000_0000
    } else {
        let r = round_mode(b, rn);
        if r > i32::MAX as f64 {
            0x7FFF_FFFF
        } else if r < i32::MIN as f64 {
            0xFFFF_FFFF_8000_0000
        } else {
            r as i32 as i64 as u64
        }
    };
    f64::from_bits(v)
}

/// `fctid` / `fctidz`.
pub fn fctid(b: f64, rn: u32) -> f64 {
    let v: u64 = if b.is_nan() {
        0x8000_0000_0000_0000
    } else {
        let r = round_mode(b, rn);
        if r >= 9_223_372_036_854_775_808.0 {
            0x7FFF_FFFF_FFFF_FFFF
        } else if r < -9_223_372_036_854_775_808.0 {
            0x8000_0000_0000_0000
        } else {
            r as i64 as u64
        }
    };
    f64::from_bits(v)
}

/// `fcfid`: the register's bits as a signed integer, converted to double.
#[inline]
pub fn fcfid(b: f64) -> f64 {
    b.to_bits() as i64 as f64
}

/// Compare result as a CR field / FPCC value: LT, GT, EQ, unordered.
#[inline]
pub fn cmp(a: f64, b: f64) -> u32 {
    if a.is_nan() || b.is_nan() {
        0b0001
    } else if a < b {
        0b1000
    } else if a > b {
        0b0100
    } else {
        0b0010
    }
}

/// `lfs`: single bits to double without changing a signalling NaN.
pub fn single_to_double(w: u32) -> f64 {
    let sign = ((w >> 31) as u64) << 63;
    let exp = (w >> 23) & 0xFF;
    let frac = (w & 0x7F_FFFF) as u64;
    let bits = if exp == 0xFF {
        sign | 0x7FF0_0000_0000_0000 | frac << 29
    } else if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Denormal single: normalise.
            let mut e: i32 = 1 - 127;
            let mut m = frac;
            while m & 0x80_0000 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((e + 1023) as u64) << 52) | (m & 0x7F_FFFF) << 29
        }
    } else {
        sign | ((exp as u64 + 1023 - 127) << 52) | frac << 29
    };
    f64::from_bits(bits)
}

/// `stfs`: double to single bits as the store instruction does it (no
/// rounding; NaNs keep their high payload bits).
pub fn double_to_single(x: f64) -> u32 {
    let b = x.to_bits();
    let exp = ((b >> 52) & 0x7FF) as i32;
    if exp > 896 || b & 0x7FFF_FFFF_FFFF_FFFF == 0 {
        // Normal single range, zero, infinity or NaN: take the bits directly.
        (((b >> 32) & 0xC000_0000) | ((b >> 29) & 0x3FFF_FFFF)) as u32
    } else if exp >= 874 {
        // Denormal single.
        let sign = ((b >> 63) as u32) << 31;
        let mut frac = (b & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
        frac >>= 896 - exp + 1;
        sign | (frac >> 29) as u32
    } else {
        ((b >> 32) as u32) & 0x8000_0000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_round_trip() {
        for w in [
            0u32,
            0x8000_0000,
            0x3F80_0000,
            0x7F80_0001,
            0x0000_0001,
            0x807F_FFFF,
            0xFFC0_0000,
        ] {
            assert_eq!(double_to_single(single_to_double(w)), w, "{w:#x}");
        }
    }

    #[test]
    fn invalid_gives_positive_default_nan() {
        assert_eq!(sub(f64::INFINITY, f64::INFINITY).to_bits(), DEFAULT_NAN);
        let m = std::hint::black_box(-2.5f64);
        assert_eq!(sqrt(m).to_bits(), DEFAULT_NAN);
        assert_eq!(frsqrte(m).to_bits(), DEFAULT_NAN);
    }
}
