// Enum values and the vertex fetch constant layout ported from Xenia
// src/xenia/gpu/xenos.h and register bitfields from src/xenia/gpu/registers.h
// (BSD 3-Clause, see CREDITS.md).

//! Xenos enums and small bitfield decoders a recorder or drawer needs.

/// `VGT_DRAW_INITIATOR::prim_type` (6 bits). Values 0x10..=0x12 are also
/// line/triangle/quad patches when tessellation is enabled in
/// `VGT_OUTPUT_PATH_CNTL`; the drawer has to look at that register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    None,
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleFan,
    TriangleStrip,
    TriangleWithWFlags,
    RectangleList,
    LineLoop,
    QuadList,
    QuadStrip,
    Polygon,
    CopyRectListV0,
    CopyRectListV1,
    CopyRectListV2,
    CopyRectListV3,
    FillRectList,
    LineStrip2D,
    TriStrip2D,
    /// A value the enum in xenos.h does not name.
    Other(u32),
}

impl PrimitiveType {
    pub fn from_raw(raw: u32) -> Self {
        use PrimitiveType::*;
        match raw {
            0x00 => None,
            0x01 => PointList,
            0x02 => LineList,
            0x03 => LineStrip,
            0x04 => TriangleList,
            0x05 => TriangleFan,
            0x06 => TriangleStrip,
            0x07 => TriangleWithWFlags,
            0x08 => RectangleList,
            0x0C => LineLoop,
            0x0D => QuadList,
            0x0E => QuadStrip,
            0x0F => Polygon,
            0x10 => CopyRectListV0,
            0x11 => CopyRectListV1,
            0x12 => CopyRectListV2,
            0x13 => CopyRectListV3,
            0x14 => FillRectList,
            0x15 => LineStrip2D,
            0x16 => TriStrip2D,
            other => Other(other),
        }
    }
}

/// `VGT_DRAW_INITIATOR::index_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    Int16,
    Int32,
}

impl IndexFormat {
    pub fn bytes(self) -> u32 {
        match self {
            IndexFormat::Int16 => 2,
            IndexFormat::Int32 => 4,
        }
    }
}

/// Two-bit GPU byte swap mode (`xenos::Endian`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    None,
    E8in16,
    E8in32,
    E16in32,
}

impl Endian {
    /// Decodes the low two bits of `raw`.
    pub fn from_bits(raw: u32) -> Self {
        match raw & 3 {
            0 => Endian::None,
            1 => Endian::E8in16,
            2 => Endian::E8in32,
            _ => Endian::E16in32,
        }
    }

    /// Applies the swap to a 32-bit value, as Xenia's `GpuSwap`.
    pub fn swap(self, v: u32) -> u32 {
        match self {
            Endian::None => v,
            Endian::E8in16 => ((v >> 8) & 0x00FF_00FF) | ((v << 8) & 0xFF00_FF00),
            Endian::E8in32 => v.swap_bytes(),
            Endian::E16in32 => v.rotate_left(16),
        }
    }
}

/// `RB_COPY_CONTROL::copy_command`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyCommand {
    Raw,
    Convert,
    ConstantOne,
    Null,
}

impl CopyCommand {
    pub fn from_bits(raw: u32) -> Self {
        match raw & 3 {
            0 => CopyCommand::Raw,
            1 => CopyCommand::Convert,
            2 => CopyCommand::ConstantOne,
            _ => CopyCommand::Null,
        }
    }
}

/// `xenos::FetchConstantType`, the low two bits of a fetch constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchConstantType {
    InvalidTexture,
    InvalidVertex,
    Texture,
    Vertex,
}

/// A decoded vertex fetch constant (`xe_gpu_vertex_fetch_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFetch {
    pub fetch_type: FetchConstantType,
    /// Guest physical byte address (the field is stored in dwords).
    pub address: u32,
    pub endian: Endian,
    /// Buffer size in dwords.
    pub size_dwords: u32,
}

impl VertexFetch {
    pub fn decode(words: [u32; 2]) -> Self {
        let fetch_type = match words[0] & 3 {
            0 => FetchConstantType::InvalidTexture,
            1 => FetchConstantType::InvalidVertex,
            2 => FetchConstantType::Texture,
            _ => FetchConstantType::Vertex,
        };
        Self {
            fetch_type,
            address: words[0] & !3,
            endian: Endian::from_bits(words[1]),
            size_dwords: (words[1] >> 2) & 0x00FF_FFFF,
        }
    }
}

/// Sign-extends the low `bits` bits of `v`.
pub(crate) fn sign_extend(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v << shift) as i32) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endian_swaps() {
        assert_eq!(Endian::None.swap(0x1122_3344), 0x1122_3344);
        assert_eq!(Endian::E8in16.swap(0x1122_3344), 0x2211_4433);
        assert_eq!(Endian::E8in32.swap(0x1122_3344), 0x4433_2211);
        assert_eq!(Endian::E16in32.swap(0x1122_3344), 0x3344_1122);
    }

    #[test]
    fn sign_extend_15_bits() {
        assert_eq!(sign_extend(0x7FFF, 15), -1);
        assert_eq!(sign_extend(0x0010, 15), 16);
    }

    #[test]
    fn primitive_type_round_trip() {
        assert_eq!(PrimitiveType::from_raw(4), PrimitiveType::TriangleList);
        assert_eq!(PrimitiveType::from_raw(0x0C), PrimitiveType::LineLoop);
        assert_eq!(PrimitiveType::from_raw(0x09), PrimitiveType::Other(9));
    }
}
