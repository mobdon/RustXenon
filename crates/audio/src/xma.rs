//! Facts about the console's compressed audio (XMA2) that the decoder path
//! needs. Layouts follow FFmpeg's `wmaprodec.c` and Xenia's XMA notes
//! (format facts only, no code; see CREDITS.md).
//!
//! An XMA2 stream is a run of 2048-byte packets. Each packet starts with a
//! 32-bit big-endian header, then bit-packed frames of 512 samples per
//! channel; a frame may run on into the next packet.

/// Bytes in one packet.
pub const PACKET_BYTES: usize = 2048;
/// Samples per channel in one decoded frame.
pub const FRAME_SAMPLES: usize = 512;
/// Bits in a frame's length prefix (log2 of the packet size + 4).
pub const FRAME_LEN_BITS: u32 = 15;

/// The 32-bit header at the start of every packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketHeader {
    /// Frames that start in this packet.
    pub frame_count: u8,
    /// Bits after the header that finish the previous packet's last frame;
    /// the first frame starting here begins at bit 32 + this.
    pub frame_offset_bits: u16,
    pub metadata: u8,
    /// Packets of other streams to skip before this stream's next packet.
    pub skip_count: u8,
}

impl PacketHeader {
    pub fn parse(packet: &[u8]) -> Option<Self> {
        let w = u32::from_be_bytes(packet.get(..4)?.try_into().ok()?);
        Some(Self {
            frame_count: (w >> 26) as u8,
            frame_offset_bits: ((w >> 11) & 0x7fff) as u16,
            metadata: ((w >> 8) & 7) as u8,
            skip_count: w as u8,
        })
    }

    pub fn to_bytes(self) -> [u8; 4] {
        ((self.frame_count as u32 & 0x3f) << 26
            | (self.frame_offset_bits as u32 & 0x7fff) << 11
            | (self.metadata as u32 & 7) << 8
            | self.skip_count as u32)
            .to_be_bytes()
    }
}

/// The 34-byte XMA2WAVEFORMATEX that FFmpeg's `xma2` decoder takes as
/// extradata for one stream of `channels` (1 or 2). FFmpeg reads only the
/// stream count and the speaker mask; the rest stays zero.
pub fn xma2_extradata(channels: u16) -> [u8; 34] {
    assert!(
        channels == 1 || channels == 2,
        "one XMA stream holds 1 or 2 channels"
    );
    let mut e = [0u8; 34];
    e[0..2].copy_from_slice(&1u16.to_le_bytes());
    let mask: u32 = if channels == 1 { 0x4 } else { 0x3 }; // centre / front left+right
    e[2..6].copy_from_slice(&mask.to_le_bytes());
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = PacketHeader {
            frame_count: 5,
            frame_offset_bits: 0x1234,
            metadata: 1,
            skip_count: 0xff,
        };
        let b = h.to_bytes();
        assert_eq!(PacketHeader::parse(&b), Some(h));
        // frame_count 1, 32 bits of the previous frame, rest 0
        assert_eq!(
            PacketHeader::parse(&[0x04, 0x01, 0x00, 0x00])
                .unwrap()
                .frame_offset_bits,
            32
        );
        assert_eq!(PacketHeader::parse(&[0; 3]), None);
    }
}
