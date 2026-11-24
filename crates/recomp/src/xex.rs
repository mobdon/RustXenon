//! Xbox 360 executable (XEX2) loader.
//!
//! Reads the container header and its optional headers, removes the
//! encryption layer, undoes basic or LZX compression, then reads the PE image
//! inside: sections by virtual address, the exception table and the import
//! table. The image is left exactly as decoded; import thunks are reported,
//! not rewritten.
//!
//! Field layout and the load steps follow XenonRecomp's `XenonUtils/xex.cpp`
//! (MIT, see CREDITS.md), rewritten in Rust with every read bounds-checked.
//!
//! Keys: an encrypted file's image key is itself encrypted with a console
//! key, which is not in this repository. The caller passes candidate keys
//! (the player supplies the retail key); the all-zero development key is
//! always tried last. The right key is the one that yields a valid image.

use sha1::{Digest, Sha1};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

pub const MAGIC: &[u8; 4] = b"XEX2";
/// The development console key. Retail keys come from the player.
pub const DEVKIT_KEY: [u8; 16] = [0; 16];

pub mod key {
    pub const FILE_FORMAT_INFO: u32 = 0x0000_03FF;
    pub const DELTA_PATCH_DESCRIPTOR: u32 = 0x0000_05FF;
    pub const ENTRY_POINT: u32 = 0x0001_0100;
    pub const IMAGE_BASE_ADDRESS: u32 = 0x0001_0201;
    pub const IMPORT_LIBRARIES: u32 = 0x0001_03FF;
    pub const EXECUTION_INFO: u32 = 0x0004_0006;
}

pub(crate) fn be32(b: &[u8], at: usize) -> Result<u32> {
    match b.get(at..at + 4) {
        Some(s) => Ok(u32::from_be_bytes(s.try_into().unwrap())),
        None => err(format!("read past end at {at:#x}")),
    }
}

pub(crate) fn be16(b: &[u8], at: usize) -> Result<u16> {
    match b.get(at..at + 2) {
        Some(s) => Ok(u16::from_be_bytes(s.try_into().unwrap())),
        None => err(format!("read past end at {at:#x}")),
    }
}

fn le16(b: &[u8], at: usize) -> Result<u16> {
    match b.get(at..at + 2) {
        Some(s) => Ok(u16::from_le_bytes(s.try_into().unwrap())),
        None => err(format!("PE read past end at {at:#x}")),
    }
}

fn le32(b: &[u8], at: usize) -> Result<u32> {
    match b.get(at..at + 4) {
        Some(s) => Ok(u32::from_le_bytes(s.try_into().unwrap())),
        None => err(format!("PE read past end at {at:#x}")),
    }
}

pub(crate) fn slice(b: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    b.get(at..at.saturating_add(len))
        .ok_or_else(|| Error(format!("range {at:#x}+{len:#x} past end")))
}

/// Header fields the loader and patcher need.
#[derive(Debug, Clone)]
pub struct Header {
    pub module_flags: u32,
    /// Size of the whole header area; the image data starts here.
    pub header_size: u32,
    pub security_offset: u32,
    /// Optional headers as (key, value or file offset).
    pub optional: Vec<(u32, u32)>,
}

impl Header {
    pub fn parse(b: &[u8]) -> Result<Header> {
        if b.get(..4) != Some(MAGIC) {
            return err("not an XEX2 file");
        }
        let count = be32(b, 0x14)? as usize;
        let optional = (0..count)
            .map(|i| Ok((be32(b, 0x18 + i * 8)?, be32(b, 0x1C + i * 8)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Header {
            module_flags: be32(b, 4)?,
            header_size: be32(b, 8)?,
            security_offset: be32(b, 0x10)?,
            optional,
        })
    }

    /// Inline value of an optional header whose key's low byte is 0 or 1.
    pub fn value(&self, key: u32) -> Option<u32> {
        debug_assert!(key & 0xFF <= 1);
        self.optional.iter().find(|o| o.0 == key).map(|o| o.1)
    }

    /// File offset and length of an optional header's data. Keys ending in
    /// 0xFF store their size in the data's first word.
    pub fn data(&self, b: &[u8], key: u32) -> Result<Option<(usize, usize)>> {
        let Some(&(_, off)) = self.optional.iter().find(|o| o.0 == key) else {
            return Ok(None);
        };
        let off = off as usize;
        let len = match key & 0xFF {
            0xFF => be32(b, off)? as usize,
            n => n as usize * 4,
        };
        slice(b, off, len)?;
        Ok(Some((off, len)))
    }
}

/// Security info fields (at `Header::security_offset`).
#[derive(Debug, Clone)]
pub struct Security {
    pub image_size: u32,
    pub image_flags: u32,
    pub load_address: u32,
    /// Image key, encrypted with the console key.
    pub aes_key: [u8; 16],
    /// Page descriptors: (pages, kind: 1 code, 2 data, 3 read-only).
    pub pages: Vec<(u32, u8)>,
}

impl Security {
    pub fn parse(b: &[u8], at: usize) -> Result<Security> {
        let count = be32(b, at + 0x180)? as usize;
        let pages = (0..count)
            .map(|i| be32(b, at + 0x184 + i * 24).map(|v| (v >> 4, (v & 0xF) as u8)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Security {
            image_size: be32(b, at + 4)?,
            image_flags: be32(b, at + 0x10C)?,
            load_address: be32(b, at + 0x110)?,
            aes_key: slice(b, at + 0x150, 16)?.try_into().unwrap(),
            pages,
        })
    }

    pub fn page_size(&self) -> u32 {
        if self.image_flags & 0x1000_0000 != 0 {
            0x1000
        } else {
            0x10000
        }
    }
}

/// File format info: encryption, compression and its parameters.
#[derive(Debug, Clone)]
pub struct Format {
    pub encrypted: bool,
    pub compression: Compression,
}

#[derive(Debug, Clone)]
pub enum Compression {
    None,
    /// (data bytes, zero bytes) runs.
    Basic(Vec<(u32, u32)>),
    /// LZX in a chain of hashed blocks.
    Normal {
        window: u32,
        first: Block,
    },
    /// Title-update delta records in a chain of hashed blocks.
    Delta {
        window: u32,
        first: Block,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub size: u32,
    pub hash: [u8; 20],
}

impl Format {
    pub fn parse(b: &[u8], h: &Header) -> Result<(Format, usize)> {
        let Some((at, len)) = h.data(b, key::FILE_FORMAT_INFO)? else {
            return err("missing file format header");
        };
        let encrypted = match be16(b, at + 4)? {
            0 => false,
            1 => true,
            n => return err(format!("unknown encryption type {n}")),
        };
        let chained = || -> Result<(u32, Block)> {
            Ok((
                be32(b, at + 8)?,
                Block {
                    size: be32(b, at + 12)?,
                    hash: slice(b, at + 16, 20)?.try_into().unwrap(),
                },
            ))
        };
        let compression = match be16(b, at + 6)? {
            0 => Compression::None,
            1 => Compression::Basic(
                (1..len / 8)
                    .map(|i| Ok((be32(b, at + i * 8)?, be32(b, at + i * 8 + 4)?)))
                    .collect::<Result<_>>()?,
            ),
            2 => {
                let (window, first) = chained()?;
                Compression::Normal { window, first }
            }
            3 => {
                let (window, first) = chained()?;
                Compression::Delta { window, first }
            }
            n => return err(format!("unknown compression type {n}")),
        };
        Ok((
            Format {
                encrypted,
                compression,
            },
            at,
        ))
    }
}

pub(crate) fn aes_ecb_decrypt(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    use aes::cipher::{BlockDecrypt, KeyInit};
    let mut b = aes::Block::clone_from_slice(block);
    aes::Aes128::new(key.into()).decrypt_block(&mut b);
    b.into()
}

/// AES-128-CBC with a zero IV, in place, whole 16-byte blocks only.
pub(crate) fn aes_cbc_decrypt(key: &[u8; 16], data: &mut [u8]) {
    use aes::cipher::{BlockDecrypt, KeyInit};
    let cipher = aes::Aes128::new(key.into());
    let mut prev = [0u8; 16];
    for chunk in data.chunks_exact_mut(16) {
        let ct: [u8; 16] = (*chunk).try_into().unwrap();
        let mut b = aes::Block::clone_from_slice(chunk);
        cipher.decrypt_block(&mut b);
        for (o, (p, x)) in chunk.iter_mut().zip(prev.iter().zip(b.iter())) {
            *o = p ^ x;
        }
        prev = ct;
    }
}

/// Walks a chain of SHA-1 hashed blocks. Each block starts with the next
/// block's size and hash; returns each block's payload after those 24 bytes.
pub(crate) fn block_chain(data: &[u8], first: Block) -> Result<Vec<&[u8]>> {
    let (mut at, mut cur, mut out) = (0usize, first, Vec::new());
    while cur.size != 0 {
        let block = slice(data, at, cur.size as usize)?;
        if block.len() < 24 {
            return err("compressed block too small");
        }
        if Sha1::digest(block)[..] != cur.hash {
            return err(format!("block hash mismatch at data offset {at:#x}"));
        }
        out.push(&block[24..]);
        at += block.len();
        cur = Block {
            size: be32(block, 0)?,
            hash: block[4..24].try_into().unwrap(),
        };
    }
    Ok(out)
}

/// Decrypts and decompresses the image data of `file` with the given
/// session (image) key. Returns `image_size` bytes.
pub(crate) fn decode_image(
    file: &[u8],
    h: &Header,
    sec: &Security,
    fmt: &Format,
    session_key: Option<&[u8; 16]>,
) -> Result<Vec<u8>> {
    let mut data = slice(
        file,
        h.header_size as usize,
        file.len().saturating_sub(h.header_size as usize),
    )?
    .to_vec();
    if fmt.encrypted {
        let Some(k) = session_key else {
            return err("encrypted image needs a key");
        };
        aes_cbc_decrypt(k, &mut data);
    }
    let size = sec.image_size as usize;
    let mut image = match &fmt.compression {
        Compression::None => {
            let mut v = data;
            v.resize(size, 0);
            v
        }
        Compression::Basic(runs) => {
            let mut v = Vec::with_capacity(size);
            let mut at = 0usize;
            for &(n, zeros) in runs {
                v.extend_from_slice(slice(&data, at, n as usize)?);
                at += n as usize;
                v.resize(v.len() + zeros as usize, 0);
            }
            v
        }
        Compression::Normal { window, first } => {
            let mut stream = Vec::new();
            for payload in block_chain(&data, *first)? {
                let mut p = 0;
                loop {
                    let n = be16(payload, p)? as usize;
                    p += 2;
                    if n == 0 {
                        break;
                    }
                    stream.extend_from_slice(slice(payload, p, n)?);
                    p += n;
                }
            }
            crate::lzx::decompress(&stream, size, *window, None)?
        }
        Compression::Delta { .. } => {
            return err("this is a title-update patch; apply it to its base executable")
        }
    };
    image.resize(size, 0);
    Ok(image)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    /// Guest address.
    pub start: u32,
    pub size: u32,
    pub code: bool,
}

impl Section {
    pub fn end(&self) -> u32 {
        self.start + self.size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub library: String,
    pub ordinal: u16,
    /// Guest address of the import record (a thunk for functions).
    pub address: u32,
    /// True for a function thunk, false for a variable.
    pub thunk: bool,
}

/// One exception-table (.pdata) entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeFunction {
    pub start: u32,
    /// Length in bytes.
    pub size: u32,
    pub prolog_words: u8,
}

/// A decoded executable, mapped at `base`.
#[derive(Debug, Clone)]
pub struct Image {
    pub base: u32,
    pub entry: u32,
    pub title_id: Option<u32>,
    pub version: Option<u32>,
    pub data: Vec<u8>,
    pub sections: Vec<Section>,
    pub imports: Vec<Import>,
    pub exception_table: Vec<RuntimeFunction>,
}

impl Image {
    /// Bytes at guest address `addr`, if inside the image.
    pub fn bytes(&self, addr: u32, len: usize) -> Option<&[u8]> {
        let off = addr.checked_sub(self.base)? as usize;
        self.data.get(off..off.checked_add(len)?)
    }

    pub fn u32(&self, addr: u32) -> Option<u32> {
        Some(u32::from_be_bytes(self.bytes(addr, 4)?.try_into().unwrap()))
    }

    pub fn section_at(&self, addr: u32) -> Option<&Section> {
        self.sections
            .iter()
            .find(|s| addr >= s.start && addr < s.end())
    }

    pub fn is_code(&self, addr: u32) -> bool {
        self.section_at(addr).is_some_and(|s| s.code)
    }
}

/// Loads an executable. `keys` are candidate console keys; see module docs.
pub fn load(file: &[u8], keys: &[[u8; 16]]) -> Result<Image> {
    let h = Header::parse(file)?;
    let sec = Security::parse(file, h.security_offset as usize)?;
    let (fmt, _) = Format::parse(file, &h)?;
    if !fmt.encrypted {
        return map(file, &h, &sec, decode_image(file, &h, &sec, &fmt, None)?);
    }
    let mut last = Error("no key".into());
    for k in keys.iter().chain([&DEVKIT_KEY]) {
        let session = aes_ecb_decrypt(k, &sec.aes_key);
        match decode_image(file, &h, &sec, &fmt, Some(&session)) {
            Ok(img) if img.starts_with(b"MZ") => return map(file, &h, &sec, img),
            Ok(_) => last = Error("decrypted image has no PE header (wrong key?)".into()),
            Err(e) => last = e,
        }
    }
    err(format!("could not decrypt image with any key: {last}"))
}

fn map(file: &[u8], h: &Header, sec: &Security, data: Vec<u8>) -> Result<Image> {
    let base = h.value(key::IMAGE_BASE_ADDRESS).unwrap_or(sec.load_address);
    let entry = h.value(key::ENTRY_POINT).unwrap_or(0);

    // PE headers are little-endian; section contents are big-endian.
    if !data.starts_with(b"MZ") {
        return err("image has no PE header");
    }
    let nt = le32(&data, 0x3C)? as usize;
    if data.get(nt..nt + 4) != Some(b"PE\0\0") {
        return err("bad PE signature");
    }
    let count = le16(&data, nt + 6)? as usize;
    let first = nt + 24 + le16(&data, nt + 20)? as usize;
    let mut sections = Vec::with_capacity(count);
    for i in 0..count {
        let s = first + i * 40;
        let raw = slice(&data, s, 8)?;
        let name = String::from_utf8_lossy(raw.split(|&c| c == 0).next().unwrap()).into_owned();
        let size = le32(&data, s + 8)?;
        let rva = le32(&data, s + 12)?;
        let flags = le32(&data, s + 36)?;
        sections.push(Section {
            name,
            start: base.wrapping_add(rva),
            size,
            code: flags & 0x20 != 0,
        });
    }

    let (title_id, version) = match h.data(file, key::EXECUTION_INFO)? {
        Some((at, _)) => (Some(be32(file, at + 12)?), Some(be32(file, at + 4)?)),
        None => (None, None),
    };

    let mut image = Image {
        base,
        entry,
        title_id,
        version,
        data,
        sections,
        imports: Vec::new(),
        exception_table: Vec::new(),
    };
    image.exception_table = exception_table(&image)?;
    image.imports = imports(file, h, &image)?;
    Ok(image)
}

fn exception_table(image: &Image) -> Result<Vec<RuntimeFunction>> {
    let Some(p) = image.sections.iter().find(|s| s.name == ".pdata") else {
        return Ok(Vec::new());
    };
    let Some(bytes) = image.bytes(p.start, p.size as usize) else {
        return err(".pdata outside image");
    };
    bytes
        .chunks_exact(8)
        .map(|e| {
            let v = be32(e, 4)?;
            Ok(RuntimeFunction {
                start: be32(e, 0)?,
                size: ((v >> 8) & 0x3F_FFFF) * 4,
                prolog_words: v as u8,
            })
        })
        .filter(|f| !matches!(f, Ok(RuntimeFunction { start: 0, .. })))
        .collect()
}

fn imports(file: &[u8], h: &Header, image: &Image) -> Result<Vec<Import>> {
    let Some((at, _)) = h.data(file, key::IMPORT_LIBRARIES)? else {
        return Ok(Vec::new());
    };
    let strings_size = be32(file, at + 4)? as usize;
    let count = be32(file, at + 8)? as usize;
    let table = slice(file, at + 12, strings_size)?;
    let names: Vec<String> = table
        .split(|&c| c == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();

    let mut out = Vec::new();
    let mut lib = at + 12 + strings_size;
    for _ in 0..count {
        let name_index = be16(file, lib + 36)? as usize;
        let records = be16(file, lib + 38)? as usize;
        let library = names
            .get(name_index)
            .cloned()
            .ok_or_else(|| Error(format!("import library name {name_index} missing")))?;
        for r in 0..records {
            let address = be32(file, lib + 40 + r * 4)?;
            let value = image
                .u32(address)
                .ok_or_else(|| Error(format!("import record {address:#x} outside image")))?;
            out.push(Import {
                library: library.clone(),
                ordinal: value as u16,
                address,
                thunk: value >> 24 != 0,
            });
        }
        lib += 40 + records * 4;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Packing, XexSpec};

    const KEY: [u8; 16] = *b"test console key";

    fn check(img: &Image, spec: &XexSpec) {
        assert_eq!(img.base, spec.base);
        assert_eq!(img.entry, spec.entry);
        assert_eq!(img.title_id, Some(0x1234_5678));
        let text = img.sections.iter().find(|s| s.name == ".text").unwrap();
        assert!(text.code);
        assert_eq!(
            img.bytes(text.start, spec.code.len() * 4).unwrap()[..4],
            spec.code[0].to_be_bytes()
        );
        assert_eq!(img.exception_table, spec.pdata);
        assert_eq!(
            img.imports,
            vec![
                Import {
                    library: "kernel.lib".into(),
                    ordinal: 0x10,
                    address: spec.import_addr,
                    thunk: false
                },
                Import {
                    library: "kernel.lib".into(),
                    ordinal: 0x10,
                    address: spec.import_addr + 4,
                    thunk: true
                },
            ]
        );
    }

    #[test]
    fn every_packing_and_encryption() {
        let spec = XexSpec::sample();
        let plain = spec.image();
        for packing in [Packing::None, Packing::Basic, Packing::Lzx] {
            for key in [None, Some(KEY), Some(DEVKIT_KEY)] {
                let file = spec.build(&plain, packing, key);
                let img =
                    load(&file, &[KEY]).unwrap_or_else(|e| panic!("{packing:?} {key:?}: {e}"));
                assert_eq!(img.data, plain, "{packing:?} {key:?}");
                check(&img, &spec);
            }
        }
    }

    #[test]
    fn wrong_key_and_corruption_are_errors() {
        let spec = XexSpec::sample();
        let plain = spec.image();
        let file = spec.build(&plain, Packing::Basic, Some(KEY));
        assert!(load(&file, &[]).is_err());

        let mut lzx = spec.build(&plain, Packing::Lzx, None);
        let n = lzx.len();
        lzx[n - 3] ^= 1; // breaks the block hash
        assert!(load(&lzx, &[]).unwrap_err().0.contains("hash"));

        assert!(load(b"XEX1", &[]).is_err());
        assert!(load(&file[..0x40], &[]).is_err());
    }
}
