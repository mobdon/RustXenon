//! Reads the console's disc file system (XDVDFS) from a disc image, so the
//! player can unpack their own disc into a game folder.
//!
//! Layout (little-endian, 2048-byte sectors): the volume descriptor sits 32
//! sectors into the game partition and starts with
//! `MICROSOFT*XBOX*MEDIA`, then the root directory's sector and size. A
//! directory is a binary tree of 4-byte aligned entries: left and right
//! child offsets (in 4-byte units, 0 for none), first sector, size,
//! attributes (0x10 = directory), name length and name. Entries never cross a
//! sector; unused space is 0xFF.

use crate::xex::{err, Error, Result};
use std::io::{Read, Seek, SeekFrom};

const SECTOR: u64 = 2048;
const MAGIC: &[u8; 20] = b"MICROSOFT*XBOX*MEDIA";
/// Where the game partition starts in a full disc image of each disc
/// generation; 0 for an image of the partition alone.
const PARTITIONS: [u64; 4] = [0, 0xFD9_0000, 0x208_0000, 0x1830_0000];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Path inside the disc, `/`-separated.
    pub path: String,
    pub dir: bool,
    /// Byte offset in the image.
    pub offset: u64,
    pub size: u64,
}

pub struct Disc<R> {
    r: R,
    base: u64,
}

impl<R: Read + Seek> Disc<R> {
    pub fn open(mut r: R) -> Result<Self> {
        for base in PARTITIONS {
            let mut m = [0u8; 20];
            if r.seek(SeekFrom::Start(base + 32 * SECTOR)).is_ok()
                && r.read_exact(&mut m).is_ok()
                && &m == MAGIC
            {
                return Ok(Disc { r, base });
            }
        }
        err("no console disc file system found in this image")
    }

    fn read(&mut self, at: u64, n: usize) -> Result<Vec<u8>> {
        let mut b = vec![0; n];
        self.r
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.r.read_exact(&mut b))
            .map_err(|e| Error(format!("disc read at {at:#x}: {e}")))?;
        Ok(b)
    }

    /// Every file and directory on the disc, parents before children.
    pub fn entries(&mut self) -> Result<Vec<Entry>> {
        let v = self.read(self.base + 32 * SECTOR + 20, 8)?;
        let root = u32::from_le_bytes(v[..4].try_into().unwrap()) as u64;
        let size = u32::from_le_bytes(v[4..].try_into().unwrap()) as usize;
        let mut out = Vec::new();
        self.dir(root, size, "", &mut out, 0)?;
        Ok(out)
    }

    fn dir(
        &mut self,
        sector: u64,
        size: usize,
        prefix: &str,
        out: &mut Vec<Entry>,
        depth: u32,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if depth > 64 {
            return err("disc directories nest too deeply");
        }
        let t = self.read(self.base + sector * SECTOR, size)?;
        let mut todo = vec![0usize];
        let mut seen = std::collections::HashSet::new();
        while let Some(off) = todo.pop() {
            if !seen.insert(off) || off + 14 > t.len() {
                continue;
            }
            let u16_at = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]) as usize;
            let u32_at = |i: usize| u32::from_le_bytes(t[i..i + 4].try_into().unwrap()) as u64;
            if u16_at(off) == 0xFFFF {
                continue;
            }
            let (left, right) = (u16_at(off), u16_at(off + 2));
            let (start, len, attr, nlen) = (
                u32_at(off + 4),
                u32_at(off + 8),
                t[off + 12],
                t[off + 13] as usize,
            );
            let name = t.get(off + 14..off + 14 + nlen).ok_or_else(|| {
                Error(format!(
                    "disc entry name outside directory at sector {sector}"
                ))
            })?;
            let name = String::from_utf8_lossy(name);
            if name.contains(['/', '\\']) || name == ".." || name == "." {
                return err(format!("unsafe name on disc: {name}"));
            }
            let path = format!("{prefix}{name}");
            let is_dir = attr & 0x10 != 0;
            out.push(Entry {
                path: path.clone(),
                dir: is_dir,
                offset: self.base + start * SECTOR,
                size: if is_dir { 0 } else { len },
            });
            if is_dir {
                self.dir(start, len as usize, &format!("{path}/"), out, depth + 1)?;
            }
            for c in [left, right] {
                if c != 0 {
                    todo.push(c * 4);
                }
            }
        }
        Ok(())
    }

    /// Copies a file's bytes into `w`.
    pub fn copy(&mut self, e: &Entry, w: &mut impl std::io::Write) -> Result<()> {
        self.r
            .seek(SeekFrom::Start(e.offset))
            .map_err(|x| Error(x.to_string()))?;
        let n =
            std::io::copy(&mut (&mut self.r).take(e.size), w).map_err(|x| Error(x.to_string()))?;
        if n != e.size {
            return err(format!("{}: image ends early", e.path));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A partition with `a.bin` (3 bytes) and `dir/b.txt` (5 bytes).
    fn image(base: u64) -> Vec<u8> {
        let mut img = vec![0u8; (base + 40 * SECTOR) as usize];
        let at = |s: u64| (base + s * SECTOR) as usize;
        let v = at(32);
        img[v..v + 20].copy_from_slice(MAGIC);
        img[v + 20..v + 24].copy_from_slice(&33u32.to_le_bytes());
        img[v + 24..v + 28].copy_from_slice(&2048u32.to_le_bytes());
        let entry = |img: &mut Vec<u8>,
                     at: usize,
                     right: u16,
                     sector: u32,
                     size: u32,
                     attr: u8,
                     name: &str| {
            img[at..at + 2].copy_from_slice(&0u16.to_le_bytes());
            img[at + 2..at + 4].copy_from_slice(&right.to_le_bytes());
            img[at + 4..at + 8].copy_from_slice(&sector.to_le_bytes());
            img[at + 8..at + 12].copy_from_slice(&size.to_le_bytes());
            img[at + 12] = attr;
            img[at + 13] = name.len() as u8;
            img[at + 14..at + 14 + name.len()].copy_from_slice(name.as_bytes());
        };
        let root = at(33);
        img[root..root + 2048].fill(0xFF);
        entry(&mut img, root, 5, 35, 3, 0x80, "a.bin"); // right child at 20 bytes
        entry(&mut img, root + 20, 0, 34, 2048, 0x10, "dir");
        let d = at(34);
        img[d..d + 2048].fill(0xFF);
        entry(&mut img, d, 0, 36, 5, 0x80, "b.txt");
        img[at(35)..at(35) + 3].copy_from_slice(b"abc");
        img[at(36)..at(36) + 5].copy_from_slice(b"hello");
        img
    }

    #[test]
    fn lists_and_copies_files_at_any_partition_offset() {
        for base in [0, 0x208_0000] {
            let mut d = Disc::open(std::io::Cursor::new(image(base))).unwrap();
            let e = d.entries().unwrap();
            let names: Vec<_> = e.iter().map(|x| (x.path.as_str(), x.dir, x.size)).collect();
            assert_eq!(
                names,
                [
                    ("a.bin", false, 3),
                    ("dir", true, 0),
                    ("dir/b.txt", false, 5)
                ]
            );
            let mut out = Vec::new();
            d.copy(&e[2], &mut out).unwrap();
            assert_eq!(out, b"hello");
        }
    }

    #[test]
    fn rejects_images_without_the_file_system() {
        assert!(Disc::open(std::io::Cursor::new(vec![0u8; 0x30000])).is_err());
    }
}
