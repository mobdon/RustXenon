// Index byte swapping follows Xenia src/xenia/gpu/primitive_processor.cc
// (guest indices are read as little-endian words after the GpuSwap of the
// index endian), and the quad list / triangle fan conversions follow the
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
        // A rectangle's fourth corner is derived from the other three after
        // the vertex shader, which index expansion cannot do.
        other => return Err(format!("primitive type {other:?}")),
    })
}

/// The guest indices of a draw as host values (before expansion).
pub fn guest_indices(
    mem: &dyn GuestRead,
    source: &IndexSource,
    count: u32,
) -> Result<Vec<u32>, String> {
    match source {
        IndexSource::Auto => Ok((0..count).collect()),
        IndexSource::Dma {
            address,
            format,
            endian,
            ..
        } => {
            let size = format.bytes() as usize;
            let mut bytes = mem
                .read_bytes(*address, count as usize * size)
                .ok_or_else(|| format!("index buffer at 0x{address:08X} is not readable"))?;
            let e = xenon_texture::Endian::from_bits(match endian {
                xenon_gpu::Endian::None => 0,
                xenon_gpu::Endian::E8in16 => 1,
                xenon_gpu::Endian::E8in32 => 2,
                xenon_gpu::Endian::E16in32 => 3,
            });
            // The swap works on whole 32-bit words, so a 16-bit buffer with
            // an odd count needs the word's other half present.
            if size == 2 && bytes.len() % 4 != 0 {
                if let Some(pad) = mem.read_bytes(*address + bytes.len() as u32, 2) {
                    bytes.extend(pad);
                } else {
                    bytes.extend([0, 0]);
                }
            }
            xenon_texture::endian::swap_in_place(&mut bytes, e);
            Ok(match format {
                IndexFormat::Int16 => bytes
                    .chunks_exact(2)
                    .take(count as usize)
                    .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                IndexFormat::Int32 => bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            })
        }
        IndexSource::Immediate { .. } => Err("immediate index source".into()),
    }
}

/// Applies the topology conversion.
pub fn expand(indices: Vec<u32>, e: Expand) -> Vec<u32> {
    match e {
        Expand::None => indices,
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
        assert_eq!(guest_indices(&mem, &src, 2), Ok(vec![5, 0x0102_0304]));
        assert!(guest_indices(&mem, &src, 3).is_err());
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
        assert!(topology(PrimitiveType::RectangleList).is_err());
    }
}
