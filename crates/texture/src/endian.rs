// Endian modes and swap rules ported from Xenia src/xenia/gpu/xenos.h
// (Endian, GpuSwap) and src/xenia/gpu/shaders/endian.xesli (BSD 3-Clause, see
// CREDITS.md).

//! The GPU's per-fetch byte swapping.
//!
//! Guest memory is big-endian; the fetch constant says how to swap it so that
//! each element can then be read little-endian, with component X in the low
//! bits. Swapping happens in guest address order, before detiling.

/// Xenos `Endian` (2 bits in the fetch constant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Endian {
    None = 0,
    /// Swap the bytes of every 16-bit half word.
    K8in16 = 1,
    /// Reverse the bytes of every 32-bit word.
    K8in32 = 2,
    /// Swap the 16-bit halves of every 32-bit word.
    K16in32 = 3,
}

impl Endian {
    pub fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Endian::None,
            1 => Endian::K8in16,
            2 => Endian::K8in32,
            _ => Endian::K16in32,
        }
    }
}

/// Applies `endian` to `buf` in place. A trailing partial unit (1 byte for
/// 8in16, up to 3 for the 32-bit modes) is left as is.
pub fn swap_in_place(buf: &mut [u8], endian: Endian) {
    match endian {
        Endian::None => {}
        Endian::K8in16 => buf.chunks_exact_mut(2).for_each(|c| c.swap(0, 1)),
        Endian::K8in32 => buf.chunks_exact_mut(4).for_each(|c| c.reverse()),
        Endian::K16in32 => buf.chunks_exact_mut(4).for_each(|c| {
            c.swap(0, 2);
            c.swap(1, 3);
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    fn swapped(endian: Endian) -> [u8; 9] {
        let mut b = SRC;
        swap_in_place(&mut b, endian);
        b
    }

    #[test]
    fn modes() {
        assert_eq!(swapped(Endian::None), SRC);
        assert_eq!(swapped(Endian::K8in16), [1, 0, 3, 2, 5, 4, 7, 6, 8]);
        assert_eq!(swapped(Endian::K8in32), [3, 2, 1, 0, 7, 6, 5, 4, 8]);
        assert_eq!(swapped(Endian::K16in32), [2, 3, 0, 1, 6, 7, 4, 5, 8]);
    }

    #[test]
    fn matches_gpu_swap_on_words() {
        // Xenia's GpuSwap works on a 32-bit value loaded little-endian from
        // memory; our byte swap must give the same value back.
        let w = 0x1122_3344u32;
        let cases = [
            (Endian::K8in16, 0x2211_4433u32),
            (Endian::K8in32, 0x4433_2211),
            (Endian::K16in32, 0x3344_1122),
        ];
        for (endian, want) in cases {
            let mut b = w.to_le_bytes();
            swap_in_place(&mut b, endian);
            assert_eq!(u32::from_le_bytes(b), want, "{endian:?}");
        }
        assert_eq!(Endian::from_bits(6), Endian::K8in32);
    }
}
