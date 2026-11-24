//! LZX decoder for executable images and title-update delta records.
//!
//! Written from the published description of the LZX format (Microsoft's
//! cabinet LZX and MS-PATCH LZX documentation). The input is one continuous
//! bitstream: 16-bit little-endian words read most significant bit first,
//! output in 32 KB frames with the bitstream realigned after each frame. A
//! delta record decodes with its old bytes preloaded at the end of the window
//! (`reference`), so matches can reach back into them.

use crate::xex::{err, Result};

const FRAME: usize = 32768;
const MIN_MATCH: usize = 2;
const PRETREE: usize = 20;
const LENGTH_TREE: usize = 249;

/// Extra offset bits per position slot.
fn extra_bits(slot: usize) -> u32 {
    if slot < 4 {
        0
    } else {
        ((slot as u32 - 2) / 2).min(17)
    }
}

/// Smallest offset (plus 2) per position slot.
fn position_base(slot: usize) -> u32 {
    (0..slot).map(|s| 1u32 << extra_bits(s)).sum()
}

fn position_slots(window_bits: u32) -> usize {
    [30, 32, 34, 36, 38, 42, 50][(window_bits - 15) as usize]
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    /// Pending bits, most significant first. Fewer than 16 between reads.
    buf: u32,
    n: u32,
}

impl<'a> Bits<'a> {
    fn word(&mut self) -> Result<u32> {
        let w = match self.data.get(self.pos..self.pos + 2) {
            Some(b) => u32::from(b[0]) | u32::from(b[1]) << 8,
            // The encoder may end mid-word; allow one word of zero padding.
            None if self.pos <= self.data.len() => 0,
            None => return err("LZX: input ended early"),
        };
        self.pos += 2;
        Ok(w)
    }

    fn bits(&mut self, k: u32) -> Result<u32> {
        if k == 0 {
            return Ok(0);
        }
        while self.n < k {
            self.buf |= self.word()? << (16 - self.n);
            self.n += 16;
        }
        let v = self.buf >> (32 - k);
        self.buf = if k == 32 { 0 } else { self.buf << k };
        self.n -= k;
        Ok(v)
    }

    /// Drops pending bits so the next read starts on a word.
    fn realign(&mut self) {
        self.buf = 0;
        self.n = 0;
    }

    /// Alignment before an uncompressed block: 1 to 16 bits are skipped.
    fn align_uncompressed(&mut self) -> Result<()> {
        if self.n == 0 {
            self.word()?;
        }
        self.realign();
        Ok(())
    }

    fn raw(&mut self, len: usize) -> Result<&'a [u8]> {
        let s = self
            .data
            .get(self.pos..self.pos + len)
            .ok_or_else(|| crate::xex::Error("LZX: uncompressed block past end".into()))?;
        self.pos += len;
        Ok(s)
    }
}

/// Canonical Huffman decoder (codes assigned by length, then symbol).
struct Tree {
    counts: [u16; 17],
    symbols: Vec<u16>,
}

impl Tree {
    fn new(lens: &[u8]) -> Result<Tree> {
        let mut counts = [0u16; 17];
        for &l in lens {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let kraft: u32 = (1..=16).map(|l| u32::from(counts[l]) << (16 - l)).sum();
        if kraft > 1 << 16 {
            return err("LZX: over-subscribed Huffman table");
        }
        let mut symbols = Vec::new();
        for l in 1..=16u8 {
            symbols.extend((0..lens.len() as u16).filter(|&s| lens[s as usize] == l));
        }
        Ok(Tree { counts, symbols })
    }

    fn decode(&self, b: &mut Bits) -> Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=16 {
            code |= b.bits(1)? as i32;
            let count = i32::from(self.counts[len]);
            if code - first < count {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        err("LZX: invalid Huffman code")
    }
}

/// Reads tree lengths `lens[range]` as deltas from their previous values.
fn read_lengths(b: &mut Bits, lens: &mut [u8]) -> Result<()> {
    let mut pre = [0u8; PRETREE];
    for l in &mut pre {
        *l = b.bits(4)? as u8;
    }
    let pre = Tree::new(&pre)?;
    let mut i = 0;
    while i < lens.len() {
        let (run, value) = match pre.decode(b)? {
            17 => (4 + b.bits(4)? as usize, 0),
            18 => (20 + b.bits(5)? as usize, 0),
            19 => {
                let run = 4 + b.bits(1)? as usize;
                let z = pre.decode(b)?;
                if z > 16 {
                    return err("LZX: bad pretree symbol");
                }
                (run, (lens[i] + 17 - z as u8) % 17)
            }
            s => (1, (lens[i] + 17 - s as u8) % 17),
        };
        let run = lens
            .get_mut(i..i + run)
            .ok_or_else(|| crate::xex::Error("LZX: length run past end of tree".into()))?;
        run.fill(value);
        i += run.len();
    }
    Ok(())
}

/// Decodes `out_len` bytes. `window_size` is a power of two from 32 KB to
/// 2 MB. `reference` (at most `window_size` bytes) is preloaded for deltas.
pub fn decompress(
    input: &[u8],
    out_len: usize,
    window_size: u32,
    reference: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let window_bits = window_size.trailing_zeros();
    if !window_size.is_power_of_two() || !(15..=21).contains(&window_bits) {
        return err(format!("LZX: unsupported window size {window_size:#x}"));
    }
    let window = window_size as usize;
    let main_count = 256 + position_slots(window_bits) * 8;

    // One linear buffer: zero padding, reference bytes, then output. Offsets
    // never exceed the window, so this matches a circular window.
    let mut buf = Vec::with_capacity(window + out_len);
    if let Some(r) = reference {
        if r.len() > window {
            return err("LZX: reference data larger than window");
        }
        buf.resize(window - r.len(), 0);
        buf.extend_from_slice(r);
    }
    let base = buf.len();
    let end = base + out_len;

    let mut b = Bits {
        data: input,
        pos: 0,
        buf: 0,
        n: 0,
    };
    if b.bits(1)? != 0 {
        // shortcut: x86 call translation is never used for PowerPC images; add it if a file sets it.
        return err("LZX: x86 call translation is not supported");
    }

    let mut main_lens = vec![0u8; main_count];
    let mut length_lens = [0u8; LENGTH_TREE];
    let mut main = Tree::new(&[])?;
    let mut length = Tree::new(&[])?;
    let mut aligned = Tree::new(&[])?;
    let mut r = [1u32; 3];
    let (mut kind, mut block_len, mut remaining) = (0u32, 0usize, 0usize);

    while buf.len() < end {
        let frame_end = end.min(base + ((buf.len() - base) / FRAME + 1) * FRAME);
        while buf.len() < frame_end {
            if remaining == 0 {
                if kind == 3 && block_len % 2 == 1 {
                    b.raw(1)?; // pad byte after an odd uncompressed block
                }
                kind = b.bits(3)?;
                block_len = ((b.bits(16)? << 8) | b.bits(8)?) as usize;
                remaining = block_len;
                if block_len == 0 {
                    return err("LZX: empty block");
                }
                match kind {
                    1 | 2 => {
                        if kind == 2 {
                            let mut lens = [0u8; 8];
                            for l in &mut lens {
                                *l = b.bits(3)? as u8;
                            }
                            aligned = Tree::new(&lens)?;
                        }
                        read_lengths(&mut b, &mut main_lens[..256])?;
                        read_lengths(&mut b, &mut main_lens[256..])?;
                        main = Tree::new(&main_lens)?;
                        read_lengths(&mut b, &mut length_lens)?;
                        length = Tree::new(&length_lens)?;
                    }
                    3 => {
                        b.align_uncompressed()?;
                        for v in &mut r {
                            *v = u32::from_le_bytes(b.raw(4)?.try_into().unwrap());
                        }
                    }
                    _ => return err(format!("LZX: bad block type {kind}")),
                }
            }

            let start = buf.len();
            let run = remaining.min(frame_end - start);
            if kind == 3 {
                buf.extend_from_slice(b.raw(run)?);
            } else {
                while buf.len() < start + run {
                    let sym = main.decode(&mut b)? as usize;
                    if sym < 256 {
                        buf.push(sym as u8);
                        continue;
                    }
                    let sym = sym - 256;
                    let mut len = sym & 7;
                    if len == 7 {
                        len += length.decode(&mut b)? as usize;
                    }
                    len += MIN_MATCH;
                    let slot = sym >> 3;
                    let offset = match slot {
                        0 => r[0],
                        1 => {
                            r.swap(0, 1);
                            r[0]
                        }
                        2 => {
                            r.swap(0, 2);
                            r[0]
                        }
                        _ => {
                            let extra = extra_bits(slot);
                            let mut o = position_base(slot) - 2;
                            if kind == 2 && extra >= 3 {
                                o += b.bits(extra - 3)? << 3;
                                o += u32::from(aligned.decode(&mut b)?);
                            } else {
                                o += b.bits(extra)?;
                            }
                            r = [o, r[0], r[1]];
                            o
                        }
                    } as usize;
                    if offset == 0 || offset > window || offset > buf.len() {
                        return err("LZX: match offset outside window");
                    }
                    if buf.len() + len > frame_end {
                        return err("LZX: match crosses a frame boundary");
                    }
                    for _ in 0..len {
                        buf.push(buf[buf.len() - offset]);
                    }
                }
            }
            let produced = buf.len() - start;
            remaining = remaining
                .checked_sub(produced)
                .ok_or_else(|| crate::xex::Error("LZX: match runs past end of block".into()))?;
        }
        b.realign();
    }
    Ok(buf.split_off(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{lzx_encode, LzxBlock, Tok};

    #[test]
    fn slot_tables() {
        assert_eq!(position_base(4), 4);
        assert_eq!(position_base(5), 6);
        assert_eq!(position_base(8), 16);
        assert_eq!(extra_bits(40), 17);
    }

    #[test]
    fn uncompressed_blocks_with_odd_length_and_frames() {
        let a: Vec<u8> = (0..40000u32).map(|i| (i * 7) as u8).collect();
        let c: Vec<u8> = (0..5u8).collect();
        let s = lzx_encode(
            0x8000,
            &[
                LzxBlock::Uncompressed(a.clone()),
                LzxBlock::Uncompressed(c.clone()),
            ],
        );
        let out = decompress(&s, a.len() + c.len(), 0x8000, None).unwrap();
        assert_eq!(out[..a.len()], a[..]);
        assert_eq!(out[a.len()..], c[..]);
    }

    #[test]
    fn verbatim_and_aligned_matches() {
        let toks = vec![
            Tok::Lit(b'a'),
            Tok::Lit(b'b'),
            Tok::Lit(b'c'),
            Tok::Match { len: 9, off: 3 }, // abcabcabc
            Tok::Match {
                len: 300 - 43,
                off: 1,
            }, // long run, length tree
            Tok::Lit(b'z'),
            Tok::Rep { len: 4, slot: 0 },    // repeats offset 1
            Tok::Match { len: 5, off: 100 }, // extra bits >= 3: aligned tree path
            Tok::Rep { len: 3, slot: 1 },    // swaps back to offset 1
        ];
        let mut expect = b"abc".to_vec();
        let copy = |e: &mut Vec<u8>, len: usize, off: usize| {
            for _ in 0..len {
                e.push(e[e.len() - off]);
            }
        };
        copy(&mut expect, 9, 3);
        copy(&mut expect, 257, 1);
        expect.push(b'z');
        copy(&mut expect, 4, 1);
        copy(&mut expect, 5, 100);
        copy(&mut expect, 3, 1);
        for aligned in [false, true] {
            let s = lzx_encode(
                0x10000,
                &[LzxBlock::Compressed {
                    toks: toks.clone(),
                    aligned,
                }],
            );
            assert_eq!(decompress(&s, expect.len(), 0x10000, None).unwrap(), expect);
        }
    }

    #[test]
    fn reference_window_and_errors() {
        let reference = b"0123456789".to_vec();
        let s = lzx_encode(
            0x8000,
            &[LzxBlock::Compressed {
                toks: vec![Tok::Match { len: 10, off: 10 }, Tok::Lit(b'!')],
                aligned: false,
            }],
        );
        let out = decompress(&s, 11, 0x8000, Some(&reference)).unwrap();
        assert_eq!(out, b"0123456789!");
        // Without the reference the same match points before the output.
        assert!(decompress(&s, 11, 0x8000, None).is_err());
        assert!(decompress(&s, 11, 0x1234, None).is_err());
        assert!(decompress(&[], 11, 0x8000, None).is_err());
    }
}
