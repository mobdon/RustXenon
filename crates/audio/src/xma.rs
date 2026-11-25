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
/// Bytes in one block of a decoder context's output ring.
pub const BLOCK_BYTES: usize = 256;
/// Decoded bytes a [`Stream`] holds at most while its output ring is full.
pub const STREAM_CAP: usize = 64 << 10;
/// Sample rates a context's 2-bit rate field selects.
pub const RATES: [u32; 4] = [24_000, 32_000, 44_100, 48_000];

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

/// The first ten words of a hardware decoder context, as the title writes
/// them (big-endian words; bit numbers from the low end). Word 0: input
/// buffer 0 packet count (0..12), buffer 0 valid (20) and 1 valid (21),
/// output block count (22..27), write offset (27..32); word 1: buffer 1
/// packet count (0..12), rate (27..29), stereo (29), output valid (31);
/// word 2: input read offset in bits (0..26); word 4: current input buffer
/// (31); words 5 and 6: input buffers, word 7: output ring (physical
/// addresses); word 9: output read offset (0..5). Layout as in Xenia's XMA
/// notes (format facts only).
#[derive(Clone, Copy, Debug)]
pub struct Context(pub [u32; 10]);

impl Context {
    pub fn input_valid(&self, buf: u32) -> bool {
        self.0[0] >> (20 + buf) & 1 != 0
    }
    /// 2048-byte packets in input buffer `buf` (0 or 1).
    pub fn packets(&self, buf: u32) -> u32 {
        self.0[buf as usize & 1] & 0xfff
    }
    pub fn input_ptr(&self, buf: u32) -> u32 {
        self.0[5 + (buf as usize & 1)]
    }
    pub fn current(&self) -> u32 {
        self.0[4] >> 31
    }
    pub fn read_offset(&self) -> u32 {
        self.0[2] & 0x03ff_ffff
    }
    pub fn output_valid(&self) -> bool {
        self.0[1] >> 31 != 0
    }
    pub fn output_ptr(&self) -> u32 {
        self.0[7]
    }
    pub fn blocks(&self) -> u32 {
        self.0[0] >> 22 & 31
    }
    pub fn write_offset(&self) -> u32 {
        self.0[0] >> 27 & 31
    }
    pub fn output_read(&self) -> u32 {
        self.0[9] & 31
    }
    pub fn channels(&self) -> u16 {
        1 + (self.0[1] >> 29 & 1) as u16
    }
    pub fn rate(&self) -> u32 {
        RATES[(self.0[1] >> 27 & 3) as usize]
    }
    /// Blocks the decoder may write before it meets the title's read
    /// offset; one block always stays free so a full ring differs from an
    /// empty one.
    pub fn out_space(&self) -> u32 {
        let b = self.blocks();
        if b == 0 {
            return 0;
        }
        (self.output_read() % b + 2 * b - 1 - self.write_offset() % b) % b
    }
}

/// Decoded output of one context waiting for room in its output ring: 16-bit
/// big-endian PCM, channels interleaved (what the console's decoder writes).
#[derive(Default)]
pub struct Stream {
    pcm: std::collections::VecDeque<u8>,
    skip: u32,
    /// Samples thrown away because [`STREAM_CAP`] was reached.
    pub dropped: u64,
}

impl Stream {
    /// Takes the next 2048-byte packet of the input. Packets of other
    /// streams (the previous header's skip count) are passed over and
    /// `false` returned; own packets go to `decode`.
    pub fn feed(&mut self, packet: &[u8], decode: impl FnOnce(&[u8], &mut Vec<f32>)) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        let Some(h) = PacketHeader::parse(packet) else {
            return false;
        };
        // shortcut: a skip count of 0xFF is read as "none"; check it against
        // a title whose streams interleave in one buffer.
        self.skip = if h.skip_count == 0xff {
            0
        } else {
            h.skip_count as u32
        };
        let mut f = Vec::new();
        decode(packet, &mut f);
        for s in f {
            if self.pcm.len() >= STREAM_CAP {
                self.dropped += 1;
                continue;
            }
            self.pcm
                .extend(((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_be_bytes());
        }
        true
    }

    /// The next full output block, if one is decoded.
    pub fn block(&mut self) -> Option<[u8; BLOCK_BYTES]> {
        if self.pcm.len() < BLOCK_BYTES {
            return None;
        }
        let mut b = [0; BLOCK_BYTES];
        b.iter_mut()
            .zip(self.pcm.drain(..BLOCK_BYTES))
            .for_each(|(d, s)| *d = s);
        Some(b)
    }

    /// Decoded bytes waiting.
    pub fn queued(&self) -> usize {
        self.pcm.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(skip: u8) -> Vec<u8> {
        let mut p = vec![0u8; PACKET_BYTES];
        p[..4].copy_from_slice(
            &PacketHeader {
                frame_count: 1,
                frame_offset_bits: 0,
                metadata: 0,
                skip_count: skip,
            }
            .to_bytes(),
        );
        p
    }

    #[test]
    fn stream_skips_other_streams_and_blocks_be16() {
        let mut s = Stream::default();
        let mut fed = 0;
        let mut dec = |_: &[u8], o: &mut Vec<f32>| {
            fed += 1;
            o.extend([0.5, -1.0].repeat(64)); // 128 samples = 256 bytes
        };
        assert!(s.feed(&packet(2), &mut dec));
        assert!(!s.feed(&packet(0), &mut dec));
        assert!(!s.feed(&packet(0), &mut dec));
        assert!(s.feed(&packet(0xff), &mut dec));
        assert!(s.feed(&packet(0), &mut dec));
        assert_eq!(fed, 3);
        assert_eq!(s.queued(), 3 * BLOCK_BYTES);
        let b = s.block().unwrap();
        assert_eq!(&b[..4], &[0x40, 0x00, 0x80, 0x01]); // 16384, -32767
        s.block();
        s.block();
        assert!(s.block().is_none());
    }

    #[test]
    fn stream_caps_its_backlog() {
        let mut s = Stream::default();
        for _ in 0..200 {
            s.feed(&packet(0), |_, o| o.extend([0.1; 512]));
        }
        assert_eq!(s.queued(), STREAM_CAP);
        assert_eq!(s.dropped, 200 * 512 - STREAM_CAP as u64 / 2);
    }

    #[test]
    fn context_fields_and_ring_space() {
        let mut w = [0u32; 10];
        // buffer 0: 7 packets, valid; 8 blocks, write offset 6
        w[0] = 7 | 1 << 20 | 8 << 22 | 6 << 27;
        // buffer 1: 3 packets; 44.1 kHz stereo; output valid
        w[1] = 3 | 2 << 27 | 1 << 29 | 1 << 31;
        w[4] = 1 << 31;
        w[9] = 2;
        let c = Context(w);
        assert!(c.input_valid(0) && !c.input_valid(1));
        assert_eq!((c.packets(0), c.packets(1), c.current()), (7, 3, 1));
        assert_eq!(
            (c.rate(), c.channels(), c.output_valid()),
            (44_100, 2, true)
        );
        // read at 2, write at 6: blocks 6, 7, 0 are free (1 stays empty)
        assert_eq!(c.out_space(), 3);
        w[9] = 7;
        assert_eq!(Context(w).out_space(), 0);
        w[0] = 8 << 22 | 6 << 27;
        w[9] = 6;
        assert_eq!(Context(w).out_space(), 7);
    }

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
