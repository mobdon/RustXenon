//! Integer helpers called by translated code.
//!
//! Results are 64-bit and overflows come from the full 64-bit result, but the
//! carry (CA) comes from the low word, as titles see it on the console. Where the architecture leaves result bits undefined (the
//! high word of `mulhw` or `divw`, results of division by zero) these helpers
//! give the same value as the reference CPU emulator the tests were checked
//! against, so behaviour is at least repeatable; `mulhw` sign-extends its
//! high word instead, as the translators that run titles do.

/// `a + b + cin` over all 64 bits, with the carry out of the low word (into
/// bit 32 in PowerPC numbering): titles see CA, like CR0 of record forms,
/// from the low 32 bits. A sorted search comparing zero-extended keys with
/// `subfc`/`subfe` depends on it when the key has stray high bits.
#[inline]
pub fn add_ca(a: u64, b: u64, cin: bool) -> (u64, bool) {
    let s = a.wrapping_add(b).wrapping_add(cin as u64);
    let k = (a as u32 as u64 + b as u32 as u64 + cin as u64) >> 32 != 0;
    (s, k)
}

/// Signed overflow of `a + b (+ carry)` given its result `r`.
#[inline]
pub fn add_ov(a: u64, b: u64, r: u64) -> bool {
    ((a ^ r) & (b ^ r)) >> 63 != 0
}

/// Mask with PowerPC bits `mb..=me` set (bit 0 is the most significant of 64),
/// wrapping when `mb > me`.
#[inline]
pub const fn mask64(mb: u32, me: u32) -> u64 {
    let a = u64::MAX >> mb;
    let b = u64::MAX << (63 - me);
    if mb <= me {
        a & b
    } else {
        a | b
    }
}

/// Mask for the 32-bit rotate forms (`rlwinm` and friends), whose bit numbers
/// count within the low word.
#[inline]
pub const fn mask32(mb: u32, me: u32) -> u64 {
    mask64(mb + 32, me + 32)
}

/// The low word rotated left by `n` and copied into both halves, which is
/// what the 32-bit rotate instructions feed into their mask in 64-bit mode.
#[inline]
pub fn rotl32(x: u64, n: u32) -> u64 {
    let w = (x as u32).rotate_left(n & 31) as u64;
    w << 32 | w
}

#[inline]
pub fn slw(rs: u64, rb: u64) -> u64 {
    let n = rb & 0x3F;
    if n > 31 {
        0
    } else {
        ((rs as u32) << n) as u64
    }
}

#[inline]
pub fn srw(rs: u64, rb: u64) -> u64 {
    let n = rb & 0x3F;
    if n > 31 {
        0
    } else {
        ((rs as u32) >> n) as u64
    }
}

/// `sraw`/`srawi`: result sign-extended to 64 bits, and the carry set when a
/// negative value loses one bits.
#[inline]
pub fn sraw(rs: u64, n: u64) -> (u64, bool) {
    let s = rs as u32 as i32;
    let n = n & 0x3F;
    if n > 31 {
        ((s >> 31) as i64 as u64, s < 0)
    } else {
        let lost = (s as u32) & ((1u64 << n) - 1) as u32;
        ((s >> n) as i64 as u64, s < 0 && lost != 0)
    }
}

#[inline]
pub fn sld(rs: u64, rb: u64) -> u64 {
    let n = rb & 0x7F;
    if n > 63 {
        0
    } else {
        rs << n
    }
}

#[inline]
pub fn srd(rs: u64, rb: u64) -> u64 {
    let n = rb & 0x7F;
    if n > 63 {
        0
    } else {
        rs >> n
    }
}

/// `srad`/`sradi`.
#[inline]
pub fn srad(rs: u64, n: u64) -> (u64, bool) {
    let s = rs as i64;
    let n = n & 0x7F;
    if n > 63 {
        ((s >> 63) as u64, s < 0)
    } else {
        let lost = if n == 0 {
            0
        } else {
            rs & (u64::MAX >> (64 - n))
        };
        ((s >> n) as u64, s < 0 && lost != 0)
    }
}

/// `mullw`: the full 64-bit product of the signed low words; overflow when it
/// does not fit in 32 bits.
#[inline]
pub fn mullw(a: u64, b: u64) -> (u64, bool) {
    let p = (a as i32 as i64) * (b as i32 as i64);
    (p as u64, p != p as i32 as i64)
}

#[inline]
pub fn mulhw(a: u64, b: u64) -> u64 {
    let p = (a as i32 as i64) * (b as i32 as i64);
    (p >> 32) as u64
}

#[inline]
pub fn mulhwu(a: u64, b: u64) -> u64 {
    let p = (a as u32 as u64) * (b as u32 as u64);
    p >> 32
}

#[inline]
pub fn mulld(a: u64, b: u64) -> (u64, bool) {
    let (p, o) = (a as i64).overflowing_mul(b as i64);
    (p as u64, o)
}

#[inline]
pub fn mulhd(a: u64, b: u64) -> u64 {
    ((a as i64 as i128 * b as i64 as i128) >> 64) as u64
}

#[inline]
pub fn mulhdu(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// `divw`: signed low words. Returns (result, overflow). Division by zero and
/// `i32::MIN / -1` give 0 with overflow set.
#[inline]
pub fn divw(a: u64, b: u64) -> (u64, bool) {
    let (x, y) = (a as i32, b as i32);
    if y == 0 || (x == i32::MIN && y == -1) {
        (0, true)
    } else {
        ((x / y) as u32 as u64, false)
    }
}

#[inline]
pub fn divwu(a: u64, b: u64) -> (u64, bool) {
    match (a as u32).checked_div(b as u32) {
        Some(q) => (q as u64, false),
        None => (0, true),
    }
}

#[inline]
pub fn divd(a: u64, b: u64) -> (u64, bool) {
    let (x, y) = (a as i64, b as i64);
    if y == 0 || (x == i64::MIN && y == -1) {
        (0, true)
    } else {
        ((x / y) as u64, false)
    }
}

#[inline]
pub fn divdu(a: u64, b: u64) -> (u64, bool) {
    match a.checked_div(b) {
        Some(q) => (q, false),
        None => (0, true),
    }
}

/// Trap condition for `tw`/`twi` (32-bit operands) and `td`/`tdi`.
/// `to` bits: 16 = less than, 8 = greater than, 4 = equal, 2 = unsigned
/// less than, 1 = unsigned greater than.
#[inline]
pub fn trap_cond(to: u32, a: i64, b: i64) -> bool {
    (to & 16 != 0 && a < b)
        || (to & 8 != 0 && a > b)
        || (to & 4 != 0 && a == b)
        || (to & 2 != 0 && (a as u64) < (b as u64))
        || (to & 1 != 0 && (a as u64) > (b as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks() {
        assert_eq!(mask64(0, 63), u64::MAX);
        assert_eq!(mask64(32, 63), 0xFFFF_FFFF);
        assert_eq!(mask64(63, 0), 0x8000_0000_0000_0001);
        assert_eq!(mask32(0, 31), 0xFFFF_FFFF);
        assert_eq!(mask32(31, 0), 0xFFFF_FFFF_8000_0001);
    }

    #[test]
    fn shifts_with_carry() {
        assert_eq!(sraw(0xFFFF_FFF5, 1), (0xFFFF_FFFF_FFFF_FFFA, true));
        assert_eq!(sraw(0x8000_0000, 40), (u64::MAX, true));
        assert_eq!(
            srad(0x8000_0000_0000_0001, 1),
            (0xC000_0000_0000_0000, true)
        );
        assert_eq!(srad(5, 0), (5, false));
    }

    #[test]
    fn carry_comes_from_the_low_word() {
        // Plain carry out of bit 31; the sum keeps all 64 bits.
        assert_eq!(add_ca(0xFFFF_FFFF, 1, false), (0x1_0000_0000, true));
        assert_eq!(add_ca(u64::MAX, 1, false), (0, true));
        // A carry out of bit 63 alone is not the low word's carry.
        assert_eq!(
            add_ca(0x1_0000_0000, 0xFFFF_FFFF_0000_0000, false),
            (0, false)
        );
        // `subfc t, entry, key; subfe t, t, t` gives -1 when key < entry and
        // 0 when equal. The key has stray high bits (a hash built with 64-bit
        // adds), the entry was loaded with lwz.
        let (key, entry) = (0x0000_0005_1689_FF8C_u64, 0x7D27_B1E8_u64);
        let (t, ca) = add_ca(!entry, key, true);
        let (t, _) = add_ca(!t, t, ca);
        assert_eq!(t, u64::MAX);
        let (t, ca) = add_ca(!entry, 0x0000_0005_7D27_B1E8, true);
        assert_eq!(add_ca(!t, t, ca).0, 0);
    }
}
