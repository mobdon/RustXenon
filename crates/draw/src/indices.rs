// Index byte swapping follows Xenia src/xenia/gpu/primitive_processor.cc
// (guest indices are read as little-endian words after the GpuSwap of the
// index endian, 32-bit swap modes on 16-bit indices normalised, 32-bit
// indices masked to 24 bits as in primitive_processor.h's To24 transforms),
// and the quad list / triangle fan conversions follow the
// conversions in primitive_processor.h (QuadListToTriangleList,
// TriangleFanToList) (BSD 3-Clause, see CREDITS.md).

//! Guest index buffers -> host u32 index lists and primitive topologies.

use xenon_gpu::{GuestRead, IndexFormat, IndexSource, PrimitiveType};

/// How a guest primitive type is drawn on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expand {
    /// Indices are used as they are.
    None,
    /// Every 4 indices (a b c d) become two triangles (a b c) (a c d).
    Quads,
    /// Indices (a b c d ...) become triangles (b c a) (c d a) ..., the
    /// Direct3D order Xenia uses.
    Fan,
    /// Rectangle lists: whole groups of 3 indices are kept as they are; the
    /// drawer's vertex shader wrapper makes 6 host vertices per rectangle
    /// from them (see `rect_list_wgsl` in lib.rs).
    Rects,
}

/// Host topology and index expansion for a guest primitive type, or the
/// reason it is not drawn.
pub fn topology(p: PrimitiveType) -> Result<(wgpu::PrimitiveTopology, Expand), String> {
    use wgpu::PrimitiveTopology as T;
    Ok(match p {
        PrimitiveType::PointList => (T::PointList, Expand::None),
        PrimitiveType::LineList => (T::LineList, Expand::None),
        PrimitiveType::LineStrip => (T::LineStrip, Expand::None),
        PrimitiveType::TriangleList => (T::TriangleList, Expand::None),
        PrimitiveType::TriangleStrip => (T::TriangleStrip, Expand::None),
        PrimitiveType::TriangleFan => (T::TriangleList, Expand::Fan),
        PrimitiveType::QuadList => (T::TriangleList, Expand::Quads),
        PrimitiveType::RectangleList => (T::TriangleList, Expand::Rects),
        other => return Err(format!("primitive type {other:?}")),
    })
}

/// The guest indices of a draw as host values (before expansion).
pub fn guest_indices(
    mem: &dyn GuestRead,
    source: &IndexSource,
    count: u32,
) -> Result<Vec<u32>, String> {
    // VGT_DRAW_INITIATOR::num_indices is 16 bits (Xenia registers.h); a
    // larger count can only come from a hand-made command and would make
    // the drawer allocate without bound.
    if count > 0xFFFF {
        return Err(format!("index count {count}"));
    }
    match source {
        IndexSource::Auto => Ok((0..count).collect()),
        IndexSource::Dma {
            address,
            format,
            endian,
            ..
        } => {
            use xenon_gpu::Endian as E;
            let size = format.bytes() as usize;
            let bytes = mem
                .read_bytes(*address, count as usize * size)
                .ok_or_else(|| format!("index buffer at 0x{address:08X} is not readable"))?;
            Ok(match format {
                // 32-bit swap modes on 16-bit indices: Xenia treats 8in32 as
                // 8in16 and 16in32 as none rather than swapping across
                // adjacent indices.
                IndexFormat::Int16 => bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let v = u16::from_le_bytes([c[0], c[1]]);
                        u32::from(match endian {
                            E::E8in16 | E::E8in32 => v.swap_bytes(),
                            E::None | E::E16in32 => v,
                        })
                    })
                    .collect(),
                // Vertex indices are 24-bit (Xenia's To24 index transforms),
                // which also keeps 0xFFFFFFFF from restarting host strips.
                IndexFormat::Int32 => bytes
                    .chunks_exact(4)
                    .map(|c| endian.from_guest_bytes([c[0], c[1], c[2], c[3]]) & 0xFF_FFFF)
                    .collect(),
            })
        }
        IndexSource::Immediate { .. } => Err("immediate index source".into()),
    }
}

/// Primitive reset (PA_SU_SC_MODE_CNTL::multi_prim_ib_ena with
/// VGT_MULTI_PRIM_IB_RESET_INDX, both 24-bit after the swap, as Xenia's
/// ReplaceResetIndex32To24): matching indices become the host strip restart
/// value `u32::MAX`. Returns whether any index matched.
pub fn mark_resets(indices: &mut [u32], reset_index: u32) -> bool {
    let reset_index = reset_index & 0xFF_FFFF;
    let mut any = false;
    for i in indices.iter_mut().filter(|i| **i == reset_index) {
        *i = u32::MAX;
        any = true;
    }
    any
}

/// Applies the topology conversion.
pub fn expand(indices: Vec<u32>, e: Expand) -> Vec<u32> {
    match e {
        Expand::None => indices,
        Expand::Rects => {
            let mut v = indices;
            v.truncate(v.len() / 3 * 3);
            v
        }
        Expand::Quads => indices
            .chunks_exact(4)
            .flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]])
            .collect(),
        Expand::Fan => {
            let mut out = Vec::new();
            for w in indices.windows(2).skip(1) {
                out.extend([w[0], w[1], indices[0]]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenon_gpu::SliceMemory;

    #[test]
    fn dma_16_bit_8in16_reads_big_endian_halves() {
        let mem = SliceMemory {
            base: 0x100,
            data: vec![0x00, 0x03, 0x01, 0x02, 0x00, 0x07],
        };
        let src = IndexSource::Dma {
            address: 0x100,
            buffer_indices: 3,
            format: IndexFormat::Int16,
            endian: xenon_gpu::Endian::E8in16,
        };
        assert_eq!(guest_indices(&mem, &src, 3), Ok(vec![3, 0x102, 7]));
    }

    #[test]
    fn dma_32_bit_8in32() {
        let mem = SliceMemory::from_words(0x100, &[5, 0x0102_0304]);
        let src = IndexSource::Dma {
            address: 0x100,
            buffer_indices: 2,
            format: IndexFormat::Int32,
            endian: xenon_gpu::Endian::E8in32,
        };
        assert_eq!(guest_indices(&mem, &src, 2), Ok(vec![5, 0x02_0304]));
        assert!(guest_indices(&mem, &src, 3).is_err());
    }

    #[test]
    fn dma_16_bit_with_32_bit_swap_modes() {
        // An odd count, and swaps that must not cross adjacent indices.
        let mem = SliceMemory {
            base: 0x100,
            data: vec![0x00, 0x03, 0x01, 0x02, 0x00, 0x07],
        };
        let src = |endian| IndexSource::Dma {
            address: 0x100,
            buffer_indices: 3,
            format: IndexFormat::Int16,
            endian,
        };
        let e8in32 = src(xenon_gpu::Endian::E8in32);
        assert_eq!(guest_indices(&mem, &e8in32, 3), Ok(vec![3, 0x102, 7]));
        let e16in32 = src(xenon_gpu::Endian::E16in32);
        assert_eq!(
            guest_indices(&mem, &e16in32, 3),
            Ok(vec![0x300, 0x201, 0x700])
        );
    }

    #[test]
    fn reset_indices_are_marked() {
        let mut v = vec![1, 0xFF_FFFF, 2];
        assert!(mark_resets(&mut v, 0xFFFF_FFFF));
        assert_eq!(v, vec![1, u32::MAX, 2]);
        assert!(!mark_resets(&mut v, 7));
    }

    #[test]
    fn quads_and_fans_expand() {
        assert_eq!(
            expand(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], Expand::Quads),
            vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        );
        assert_eq!(
            expand(vec![9, 1, 2, 3], Expand::Fan),
            vec![1, 2, 9, 2, 3, 9]
        );
        assert_eq!(
            topology(PrimitiveType::RectangleList),
            Ok((wgpu::PrimitiveTopology::TriangleList, Expand::Rects))
        );
        assert_eq!(expand(vec![1, 2, 3, 4, 5], Expand::Rects), vec![1, 2, 3]);
        assert!(topology(PrimitiveType::LineLoop).is_err());
        let mem = SliceMemory::default();
        assert!(guest_indices(&mem, &IndexSource::Auto, 0x1_0000).is_err());
    }
}
