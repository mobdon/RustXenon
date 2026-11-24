//! Big-endian guest memory addressed by 32-bit guest pointers.
//!
//! The write-up calls for one large reserved mapping with a base pointer
//! (512 MB of guest RAM, 32-bit pointers) rather than a growable buffer with
//! bounds checks. This scaffold uses a plain boxed slice so it builds and
//! tests everywhere; the reserved mapping replaces it in a later step.

pub struct GuestMemory {
    bytes: Box<[u8]>,
}

impl GuestMemory {
    /// Allocates `size` bytes of zeroed guest memory.
    pub fn new(size: usize) -> Self {
        Self {
            bytes: vec![0u8; size].into_boxed_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn slice<const N: usize>(&self, addr: u32) -> [u8; N] {
        let a = addr as usize;
        self.bytes[a..a + N].try_into().unwrap()
    }

    fn slice_mut<const N: usize>(&mut self, addr: u32) -> &mut [u8] {
        let a = addr as usize;
        &mut self.bytes[a..a + N]
    }

    pub fn read_u8(&self, addr: u32) -> u8 {
        self.bytes[addr as usize]
    }
    pub fn read_u16(&self, addr: u32) -> u16 {
        u16::from_be_bytes(self.slice(addr))
    }
    pub fn read_u32(&self, addr: u32) -> u32 {
        u32::from_be_bytes(self.slice(addr))
    }
    pub fn read_u64(&self, addr: u32) -> u64 {
        u64::from_be_bytes(self.slice(addr))
    }

    pub fn write_u8(&mut self, addr: u32, v: u8) {
        self.bytes[addr as usize] = v;
    }
    pub fn write_u16(&mut self, addr: u32, v: u16) {
        self.slice_mut::<2>(addr).copy_from_slice(&v.to_be_bytes());
    }
    pub fn write_u32(&mut self, addr: u32, v: u32) {
        self.slice_mut::<4>(addr).copy_from_slice(&v.to_be_bytes());
    }
    pub fn write_u64(&mut self, addr: u32, v: u64) {
        self.slice_mut::<8>(addr).copy_from_slice(&v.to_be_bytes());
    }

    /// Reads 16 bytes at `addr` as a vector (element 0 at the lowest address).
    pub fn read_v128(&self, addr: u32) -> crate::V128 {
        let b: [u8; 16] = self.slice(addr);
        crate::vmx::from_bytes(b)
    }

    pub fn write_v128(&mut self, addr: u32, v: crate::V128) {
        self.slice_mut::<16>(addr)
            .copy_from_slice(&crate::vmx::bytes(v));
    }

    /// Zeroes the `size`-byte cache block holding `addr` (`dcbz`, `dcbz128`).
    pub fn zero_block(&mut self, addr: u32, size: u32) {
        let a = (addr & !(size - 1)) as usize;
        self.bytes[a..a + size as usize].fill(0);
    }

    /// Conditional store for `stwcx.`: writes `new` only if the word still
    /// holds `old`. Single-threaded today; step 3 makes it a host
    /// compare-and-swap.
    pub fn cas_u32(&mut self, addr: u32, old: u32, new: u32) -> bool {
        if self.read_u32(addr) == old {
            self.write_u32(addr, new);
            true
        } else {
            false
        }
    }

    /// Conditional store for `stdcx.`.
    pub fn cas_u64(&mut self, addr: u32, old: u64, new: u64) -> bool {
        if self.read_u64(addr) == old {
            self.write_u64(addr, new);
            true
        } else {
            false
        }
    }

    /// Raw bytes, for loaders and tests.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_endian_round_trip() {
        let mut m = GuestMemory::new(64);
        m.write_u32(8, 0x1234_5678);
        assert_eq!(m.read_u8(8), 0x12);
        assert_eq!(m.read_u16(10), 0x5678);
        assert_eq!(m.read_u32(8), 0x1234_5678);
        m.write_u64(16, 0x0102_0304_0506_0708);
        assert_eq!(m.read_u64(16), 0x0102_0304_0506_0708);
        assert_eq!(m.read_u8(23), 0x08);
    }
}
