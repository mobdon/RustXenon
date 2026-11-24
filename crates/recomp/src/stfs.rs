//! Reads files out of a console content package (STFS), such as a title
//! update, so the player can unpack their own update next to the game.
//!
//! Layout: magic `CON `, `LIVE` or `PIRS`; header size (big-endian) at
//! 0x340; the volume descriptor at 0x379 holds the flags (bit 0 set for a
//! read-only "female" package), the file table's block count (16-bit
//! little-endian) and first block (24-bit little-endian). Data blocks are
//! 4 KB, interleaved with hash tables every 0xAA blocks (and at the next
//! levels every 0x70E4 and 0x4AF768 blocks); a writable package keeps two
//! copies of each hash table. File table entries are 0x40 bytes: name
//! (0x28), flags (name length in bits 0..5, contiguous blocks in bit 6,
//! directory in bit 7), block counts and first block (24-bit little-endian
//! at 0x29, 0x2C, 0x2F), parent entry (16-bit big-endian at 0x32, 0xFFFF
//! for the root) and size (32-bit big-endian at 0x34).

use crate::xex::{err, Error, Result};

pub fn is_package(b: &[u8]) -> bool {
    matches!(b.get(..4), Some(b"CON " | b"LIVE" | b"PIRS"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub path: String,
    pub dir: bool,
    pub data: Vec<u8>,
}

fn le24(b: &[u8]) -> u32 {
    b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16
}

fn block_offset(header_size: u32, female: bool, b: u32) -> u64 {
    let step = |n: u32| if female { n } else { n << 1 };
    let mut block = b as u64 + step((b + 0xAA) / 0xAA) as u64;
    if b >= 0xAA {
        block += step((b + 0x70E4) / 0x70E4) as u64;
        if b >= 0x70E4 {
            block += step((b + 0x4A_F768) / 0x4A_F768) as u64;
        }
    }
    (((header_size + 0xFFF) & !0xFFF) as u64) + (block << 12)
}

/// Every file and directory in the package, parents before children.
pub fn entries(p: &[u8]) -> Result<Vec<Entry>> {
    if !is_package(p) || p.len() < 0x3A0 {
        return err("not a content package");
    }
    let header_size = u32::from_be_bytes(p[0x340..0x344].try_into().unwrap());
    let female = p[0x37B] & 1 != 0;
    let table_blocks = u16::from_le_bytes([p[0x37C], p[0x37D]]) as u32;
    let table_first = le24(&p[0x37E..]);
    let block = |b: u32| -> Result<&[u8]> {
        let o = block_offset(header_size, female, b) as usize;
        p.get(o..o + 0x1000)
            .ok_or_else(|| Error(format!("package block {b} outside the file")))
    };
    let mut raw = Vec::new();
    for i in 0..table_blocks {
        raw.extend_from_slice(block(table_first + i)?);
    }
    let mut names: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for e in raw.chunks_exact(0x40) {
        let flags = e[0x28];
        let n = (flags & 0x3F) as usize;
        if n == 0 {
            break;
        }
        let name = String::from_utf8_lossy(&e[..n.min(0x28)]).into_owned();
        if name.contains(['/', '\\']) || name == ".." || name == "." {
            return err(format!("unsafe name in package: {name}"));
        }
        let parent = u16::from_be_bytes([e[0x32], e[0x33]]);
        let path = match parent {
            0xFFFF => name,
            i => format!(
                "{}/{name}",
                names
                    .get(i as usize)
                    .ok_or_else(|| Error(format!("package entry {name} has a bad parent")))?
            ),
        };
        names.push(path.clone());
        let dir = flags & 0x80 != 0;
        let mut data = Vec::new();
        if !dir {
            if flags & 0x40 == 0 {
                return err(format!("{path}: scattered blocks are not supported"));
            }
            let size = u32::from_be_bytes(e[0x34..0x38].try_into().unwrap()) as usize;
            let first = le24(&e[0x2F..]);
            let mut b = first;
            while data.len() < size {
                data.extend_from_slice(block(b)?);
                b += 1;
            }
            data.truncate(size);
        }
        out.push(Entry { path, dir, data });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_offsets_skip_hash_tables() {
        // Female: one hash table before block 0, another after 0xAA blocks.
        assert_eq!(block_offset(0x971A, true, 0), 0xA000 + 0x1000);
        assert_eq!(block_offset(0x971A, true, 0xA9), 0xA000 + 0xAA * 0x1000);
        assert_eq!(
            block_offset(0x971A, true, 0xAA),
            0xA000 + (0xAA + 3) * 0x1000
        );
        // Male packages keep two copies of each table.
        assert_eq!(block_offset(0x971A, false, 0), 0xA000 + 0x2000);
    }

    #[test]
    fn reads_a_small_female_package() {
        let hs = 0x971Au32;
        let base = 0xA000usize;
        let blk = |b: u32| block_offset(hs, true, b) as usize;
        let mut p = vec![0u8; blk(3) + 0x1000];
        p[..4].copy_from_slice(b"LIVE");
        p[0x340..0x344].copy_from_slice(&hs.to_be_bytes());
        p[0x37B] = 1;
        p[0x37C] = 1; // one file table block
        p[0x37E] = 0; // at block 0
        let entry = |p: &mut Vec<u8>,
                     i: usize,
                     name: &str,
                     flags: u8,
                     first: u32,
                     parent: u16,
                     size: u32| {
            let e = blk(0) + 0x40 * i;
            p[e..e + name.len()].copy_from_slice(name.as_bytes());
            p[e + 0x28] = flags | name.len() as u8;
            p[e + 0x2F..e + 0x32].copy_from_slice(&first.to_le_bytes()[..3]);
            p[e + 0x32..e + 0x34].copy_from_slice(&parent.to_be_bytes());
            p[e + 0x34..e + 0x38].copy_from_slice(&size.to_be_bytes());
        };
        entry(&mut p, 0, "data", 0x80, 0, 0xFFFF, 0);
        entry(&mut p, 1, "a.txt", 0x40, 1, 0, 0x1003);
        p[blk(1)..blk(1) + 0x1000].fill(b'x');
        p[blk(2)..blk(2) + 3].copy_from_slice(b"end");
        assert!(base < blk(0));
        let e = entries(&p).unwrap();
        assert_eq!((e[0].path.as_str(), e[0].dir), ("data", true));
        assert_eq!(e[1].path, "data/a.txt");
        assert_eq!(e[1].data.len(), 0x1003);
        assert_eq!(&e[1].data[0x1000..], b"end");
        assert!(entries(b"nope").is_err());
    }
}
