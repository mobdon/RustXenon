// Address mask mirrors Xenia src/xenia/memory.h TranslatePhysical and
// src/xenia/gpu/xenos.h CpuToGpu (BSD 3-Clause, see CREDITS.md).

//! Read access to big-endian guest memory by guest physical address.

/// Mask the recorder applies to every guest address before reading or
/// recording it. Xenia's `Memory::TranslatePhysical` masks every GPU access
/// this way, so the ring base, indirect buffer pointers (`CpuToGpu`), index
/// buffer bases, IM_LOAD and LOAD_ALU_CONSTANT addresses all end up as
/// 29-bit physical addresses. LOAD_ALU_CONSTANT first masks with 0x3FFFFFFF
/// in Xenia, which this mask contains.
pub const PHYSICAL_MASK: u32 = 0x1FFF_FFFF;

/// Guest memory as the GPU sees it. Addresses are guest physical; the
/// recorder has already masked them with [`PHYSICAL_MASK`].
pub trait GuestRead {
    /// The big-endian dword at `addr`, or None if unmapped.
    fn read_u32(&self, addr: u32) -> Option<u32>;
    /// `len` raw bytes from `addr`, or None if any byte is unmapped.
    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>>;
}

/// Reads `count` big-endian dwords from `addr`.
pub fn read_words(mem: &dyn GuestRead, addr: u32, count: u32) -> Option<Vec<u32>> {
    let bytes = mem.read_bytes(addr, count as usize * 4)?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// A flat byte block mapped at `base`, for tests and tools.
#[derive(Clone, Debug, Default)]
pub struct SliceMemory {
    pub base: u32,
    pub data: Vec<u8>,
}

impl SliceMemory {
    /// Memory at `base` holding `words` as big-endian dwords.
    pub fn from_words(base: u32, words: &[u32]) -> Self {
        Self {
            base,
            data: crate::pm4::build::to_be_bytes(words),
        }
    }

    /// Overwrites `words` at `addr` (big-endian), growing the block if needed.
    pub fn write_words(&mut self, addr: u32, words: &[u32]) {
        assert!(addr >= self.base, "write_words below the block's base");
        let start = (addr - self.base) as usize;
        let end = start + words.len() * 4;
        if self.data.len() < end {
            self.data.resize(end, 0);
        }
        self.data[start..end].copy_from_slice(&crate::pm4::build::to_be_bytes(words));
    }
}

impl GuestRead for SliceMemory {
    fn read_u32(&self, addr: u32) -> Option<u32> {
        let b = self.read_bytes(addr, 4)?;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_bytes(&self, addr: u32, len: usize) -> Option<Vec<u8>> {
        let start = addr.checked_sub(self.base)? as usize;
        let end = start.checked_add(len)?;
        self.data.get(start..end).map(<[u8]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_memory_reads_big_endian_and_bounds() {
        let mem = SliceMemory {
            base: 0x1000,
            data: vec![0x12, 0x34, 0x56, 0x78, 0x9A],
        };
        assert_eq!(mem.read_u32(0x1000), Some(0x1234_5678));
        assert_eq!(mem.read_u32(0x1002), None);
        assert_eq!(mem.read_u32(0x0FFF), None);
        assert_eq!(mem.read_bytes(0x1004, 1), Some(vec![0x9A]));
        assert_eq!(read_words(&mem, 0x1000, 1), Some(vec![0x1234_5678]));
    }
}
