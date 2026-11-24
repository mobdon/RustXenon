//! Test-only builders for synthetic inputs: an LZX encoder and a small XEX2
//! writer. Everything here is made from scratch in the tests; no game bytes.

use crate::xex::{key, RuntimeFunction, DEVKIT_KEY};
use sha1::{Digest, Sha1};

// ---- LZX ------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Tok {
    Lit(u8),
    /// Match with an explicit offset (len 2..=257).
    Match {
        len: usize,
        off: u32,
    },
    /// Match reusing repeat offset slot 0..=2.
    Rep {
        len: usize,
        slot: usize,
    },
}

pub enum LzxBlock {
    Uncompressed(Vec<u8>),
    Compressed { toks: Vec<Tok>, aligned: bool },
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, bits: u32, value: u32) {
        for i in (0..bits).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.n += 1;
            if self.n == 16 {
                self.out.extend((self.acc as u16).to_le_bytes());
                self.acc = 0;
                self.n = 0;
            }
        }
    }
    fn pad(&mut self) {
        if self.n > 0 {
            self.put(16 - self.n, 0);
        }
    }
}

fn codes(lens: &[u8]) -> Vec<u32> {
    let mut codes = vec![0; lens.len()];
    let mut code = 0u32;
    for l in 1..=16u8 {
        for (s, _) in lens.iter().enumerate().filter(|(_, &x)| x == l) {
            codes[s] = code;
            code += 1;
        }
        code <<= 1;
    }
    codes
}

fn extra_bits(slot: usize) -> u32 {
    if slot < 4 {
        0
    } else {
        ((slot as u32 - 2) / 2).min(17)
    }
}

fn position_base(slot: usize) -> u32 {
    (0..slot).map(|s| 1u32 << extra_bits(s)).sum()
}

/// Writes tree lengths as pretree deltas (every pretree code is 5 bits).
fn write_lengths(w: &mut BitWriter, prev: &mut [u8], new_len: u8) {
    for _ in 0..20 {
        w.put(4, 5);
    }
    for p in prev.iter_mut() {
        w.put(5, u32::from((*p + 17 - new_len) % 17));
        *p = new_len;
    }
}

fn bits_for(count: usize) -> u8 {
    (usize::BITS - (count - 1).leading_zeros()) as u8
}

/// Encodes `blocks` into one LZX stream. Tokens must not cross a 32 KB frame.
pub fn lzx_encode(window: u32, blocks: &[LzxBlock]) -> Vec<u8> {
    let slots = [30, 32, 34, 36, 38, 42, 50][(window.trailing_zeros() - 15) as usize];
    let main_count = 256 + slots * 8;
    let (main_len, length_len) = (bits_for(main_count), 8u8);
    let (mut prev_main, mut prev_length) = (vec![0u8; main_count], vec![0u8; 249]);
    let mut w = BitWriter::default();
    let mut produced = 0usize;
    let mut advance = |w: &mut BitWriter, n: usize| {
        produced += n;
        if produced % 32768 == 0 {
            w.pad();
        }
    };
    w.put(1, 0); // no x86 call translation
    for block in blocks {
        match block {
            LzxBlock::Uncompressed(data) => {
                w.put(3, 3);
                w.put(24, data.len() as u32);
                if w.n == 0 {
                    w.put(16, 0);
                } else {
                    w.pad();
                }
                for _ in 0..3 {
                    w.out.extend(1u32.to_le_bytes());
                }
                for &b in data {
                    w.out.push(b);
                    advance(&mut w, 1);
                }
                if data.len() % 2 == 1 {
                    w.out.push(0);
                }
            }
            LzxBlock::Compressed { toks, aligned } => {
                let size: usize = toks
                    .iter()
                    .map(|t| match t {
                        Tok::Lit(_) => 1,
                        Tok::Match { len, .. } | Tok::Rep { len, .. } => *len,
                    })
                    .sum();
                w.put(3, if *aligned { 2 } else { 1 });
                w.put(24, size as u32);
                if *aligned {
                    for _ in 0..8 {
                        w.put(3, 3);
                    }
                }
                let (pm, pl) = prev_main.split_at_mut(256);
                write_lengths(&mut w, pm, main_len);
                write_lengths(&mut w, pl, main_len);
                write_lengths(&mut w, &mut prev_length, length_len);
                let mc = codes(&vec![main_len; main_count]);
                let lc = codes(&[length_len; 249]);
                for t in toks {
                    let (len, slot, off) = match *t {
                        Tok::Lit(b) => {
                            w.put(u32::from(main_len), mc[b as usize]);
                            advance(&mut w, 1);
                            continue;
                        }
                        Tok::Rep { len, slot } => (len, slot, 0),
                        Tok::Match { len, off } => {
                            let slot = (3..slots)
                                .rev()
                                .find(|&s| position_base(s) - 2 <= off)
                                .unwrap();
                            (len, slot, off - (position_base(slot) - 2))
                        }
                    };
                    let header = (len - 2).min(7);
                    w.put(u32::from(main_len), mc[256 + slot * 8 + header]);
                    if header == 7 {
                        w.put(u32::from(length_len), lc[len - 9]);
                    }
                    let extra = extra_bits(slot);
                    if *aligned && extra >= 3 {
                        w.put(extra - 3, off >> 3);
                        w.put(3, off & 7); // aligned tree: all lengths 3, code = symbol
                    } else {
                        w.put(extra, off);
                    }
                    advance(&mut w, len);
                }
            }
        }
    }
    w.pad();
    w.out
}

/// Tokens that rebuild `new` from `old` when `old` is the reference window.
pub fn delta_tokens(old: &[u8], new: &[u8]) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut i = 0;
    while i < new.len() {
        let run = (i..new.len().min(i + 257))
            .take_while(|&j| old.get(j) == Some(&new[j]))
            .count();
        if run >= 2 {
            toks.push(Tok::Match {
                len: run,
                off: old.len() as u32,
            });
            i += run;
        } else {
            toks.push(Tok::Lit(new[i]));
            i += 1;
        }
    }
    toks
}

// ---- XEX2 -----------------------------------------------------------------

pub fn aes_ecb_encrypt(key: &[u8; 16], block: &[u8; 16]) -> [u8; 16] {
    use aes::cipher::{BlockEncrypt, KeyInit};
    let mut b = aes::Block::clone_from_slice(block);
    aes::Aes128::new(key.into()).encrypt_block(&mut b);
    b.into()
}

pub fn aes_cbc_encrypt(key: &[u8; 16], data: &mut [u8]) {
    let mut prev = [0u8; 16];
    for chunk in data.chunks_exact_mut(16) {
        for (c, p) in chunk.iter_mut().zip(prev) {
            *c ^= p;
        }
        prev = aes_ecb_encrypt(key, &(*chunk).try_into().unwrap());
        chunk.copy_from_slice(&prev);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Packing {
    None,
    Basic,
    Lzx,
}

#[derive(Clone, Debug)]
pub struct XexSpec {
    pub base: u32,
    pub entry: u32,
    pub code: Vec<u32>,
    pub pdata: Vec<RuntimeFunction>,
    pub import_addr: u32,
    pub image_key: [u8; 16],
}

const TEXT_RVA: u32 = 0x3000;

impl XexSpec {
    pub fn sample() -> XexSpec {
        let base = 0x8200_0000;
        XexSpec {
            base,
            entry: base + TEXT_RVA,
            code: vec![0x3860_0001, 0x4E80_0020, 0x3860_0002, 0x4E80_0020],
            pdata: vec![
                RuntimeFunction {
                    start: base + TEXT_RVA,
                    size: 8,
                    prolog_words: 0,
                },
                RuntimeFunction {
                    start: base + TEXT_RVA + 8,
                    size: 8,
                    prolog_words: 1,
                },
            ],
            import_addr: base + 0x1000,
            image_key: *b"image session ky",
        }
    }

    pub fn code_rva(&self) -> u32 {
        TEXT_RVA
    }

    /// The PE image: headers, .rdata (import records), .pdata, .text.
    pub fn image(&self) -> Vec<u8> {
        let text_len = self.code.len() as u32 * 4;
        let mut img = vec![0u8; (TEXT_RVA + text_len).next_multiple_of(0x1000) as usize];
        img[..2].copy_from_slice(b"MZ");
        img[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        img[0x80..0x84].copy_from_slice(b"PE\0\0");
        img[0x86..0x88].copy_from_slice(&3u16.to_le_bytes());
        img[0x94..0x96].copy_from_slice(&0xE0u16.to_le_bytes());
        let sections: [(&[u8], u32, u32, u32); 3] = [
            (b".rdata", 0x1000, 8, 0x4000_0040),
            (b".pdata", 0x2000, self.pdata.len() as u32 * 8, 0x4000_0040),
            (b".text", TEXT_RVA, text_len, 0x6000_0020),
        ];
        for (i, (name, rva, size, flags)) in sections.iter().enumerate() {
            let s = 0x178 + i * 40;
            img[s..s + name.len()].copy_from_slice(name);
            img[s + 8..s + 12].copy_from_slice(&size.to_le_bytes());
            img[s + 12..s + 16].copy_from_slice(&rva.to_le_bytes());
            img[s + 36..s + 40].copy_from_slice(&flags.to_le_bytes());
        }
        let r = (self.import_addr - self.base) as usize;
        img[r..r + 4].copy_from_slice(&0x0000_0010u32.to_be_bytes());
        img[r + 4..r + 8].copy_from_slice(&0x0100_0010u32.to_be_bytes());
        for (i, f) in self.pdata.iter().enumerate() {
            let p = 0x2000 + i * 8;
            img[p..p + 4].copy_from_slice(&f.start.to_be_bytes());
            let v = (f.size / 4) << 8 | u32::from(f.prolog_words);
            img[p + 4..p + 8].copy_from_slice(&v.to_be_bytes());
        }
        for (i, w) in self.code.iter().enumerate() {
            let p = TEXT_RVA as usize + i * 4;
            img[p..p + 4].copy_from_slice(&w.to_be_bytes());
        }
        img
    }

    /// An executable holding `image`. `console` is the console key the image
    /// key is wrapped with; `None` writes it unencrypted (devkit key wrap).
    pub fn build(&self, image: &[u8], packing: Packing, console: Option<[u8; 16]>) -> Vec<u8> {
        let (format, payload) = match packing {
            Packing::None => (fmt_info(0, &[]), image.to_vec()),
            Packing::Basic => {
                let (mut runs, mut data) = (Vec::new(), Vec::new());
                for page in image.chunks(0x1000) {
                    let used = page
                        .iter()
                        .rposition(|&b| b != 0)
                        .map_or(0, |p| (p + 16) & !15)
                        .min(page.len());
                    data.extend_from_slice(&page[..used]);
                    runs.extend((used as u32).to_be_bytes());
                    runs.extend(((page.len() - used) as u32).to_be_bytes());
                }
                (fmt_info(1, &runs), data)
            }
            Packing::Lzx => {
                let stream = lzx_encode(0x8000, &[LzxBlock::Uncompressed(image.to_vec())]);
                let blocks: Vec<Vec<u8>> = stream
                    .chunks(0x3000)
                    .collect::<Vec<_>>()
                    .chunks(2)
                    .map(|cs| {
                        let mut b = Vec::new();
                        for c in cs {
                            b.extend((c.len() as u16).to_be_bytes());
                            b.extend(*c);
                        }
                        b.extend([0, 0]);
                        b
                    })
                    .collect();
                let (first, data) = chain(&blocks);
                (
                    fmt_info(2, &[0x8000u32.to_be_bytes().as_slice(), &first].concat()),
                    data,
                )
            }
        };
        let mut sec = security(image.len() as u32, self.base, &self.wrapped_key(console));
        sec.extend([0u8; 24]); // one page descriptor
        sec[0x180..0x184].copy_from_slice(&1u32.to_be_bytes());
        sec[0x184..0x188].copy_from_slice(&((image.len() as u32 / 0x10000) << 4 | 1).to_be_bytes());
        let exec = [0u32, 1, 1, 0x1234_5678, 0, 0]
            .map(u32::to_be_bytes)
            .concat();
        let imports = imports(self);
        let mut file = header(
            0x1,
            &[
                (key::ENTRY_POINT, Opt::Inline(self.entry)),
                (key::IMAGE_BASE_ADDRESS, Opt::Inline(self.base)),
                (key::EXECUTION_INFO, Opt::Data(exec)),
                (
                    key::FILE_FORMAT_INFO,
                    Opt::Data(with_encryption(format, console.is_some())),
                ),
                (key::IMPORT_LIBRARIES, Opt::Data(imports)),
            ],
            &sec,
        );
        let mut payload = payload;
        if console.is_some() {
            aes_cbc_encrypt(&self.image_key, &mut payload);
        }
        file.extend(payload);
        file
    }

    fn wrapped_key(&self, console: Option<[u8; 16]>) -> [u8; 16] {
        aes_ecb_encrypt(&console.unwrap_or(DEVKIT_KEY), &self.image_key)
    }

    /// A title-update patch taking `base` (built from `old`) to `new`: the
    /// header gets version 2 and the new image size; the image is patched by
    /// LZX records that reuse the old bytes.
    pub fn build_patch(
        &self,
        base: &[u8],
        old: &[u8],
        new: &[u8],
        console: Option<[u8; 16]>,
    ) -> Vec<u8> {
        let hs = u32::from_be_bytes(base[8..12].try_into().unwrap()) as usize;
        let so = u32::from_be_bytes(base[0x10..0x14].try_into().unwrap()) as usize;
        let mut new_hdr = base[..hs].to_vec();
        new_hdr[so + 4..so + 8].copy_from_slice(&(new.len() as u32).to_be_bytes());
        let exec = find_opt(&new_hdr, key::EXECUTION_INFO) as usize;
        new_hdr[exec + 4..exec + 8].copy_from_slice(&2u32.to_be_bytes());

        let mut header_recs = record(
            0,
            0,
            hs,
            &lzx_encode(0x8000, &[LzxBlock::Uncompressed(new_hdr)]),
        );
        header_recs.extend([0u8; 12]);

        let mut image_recs = Vec::new();
        let mut padded_old = old.to_vec();
        padded_old.resize(new.len(), 0);
        for (i, (o, n)) in padded_old
            .chunks(0x1000)
            .zip(new.chunks(0x1000))
            .enumerate()
        {
            if o != n {
                let toks = delta_tokens(o, n);
                let data = lzx_encode(
                    0x8000,
                    &[LzxBlock::Compressed {
                        toks,
                        aligned: false,
                    }],
                );
                image_recs.extend(record(i * 0x1000, i * 0x1000, n.len(), &data));
            }
        }
        image_recs.extend([0u8; 12]);
        let (first, mut payload) = chain(&[image_recs]);

        // Keys: the base's image key, unchanged; the patch has its own.
        let patch_key = *b"patch image key!";
        if console.is_some() {
            aes_cbc_encrypt(&patch_key, &mut payload);
        }
        let mut desc = [0u32, 2, 1].map(u32::to_be_bytes).concat();
        desc.extend([0u8; 20]);
        desc.extend(aes_ecb_encrypt(&self.image_key, &self.image_key));
        desc.extend([hs as u32, 0, 0, 0, 0, 0, 0].map(u32::to_be_bytes).concat());
        desc.extend(header_recs);
        let n = desc.len() as u32;
        desc[..4].copy_from_slice(&n.to_be_bytes());

        let sec = security(0, 0, &aes_ecb_encrypt(&self.image_key, &patch_key));
        let format = fmt_info(3, &[0x8000u32.to_be_bytes().as_slice(), &first].concat());
        let mut file = header(
            0x40,
            &[
                (
                    key::FILE_FORMAT_INFO,
                    Opt::Data(with_encryption(format, console.is_some())),
                ),
                (key::DELTA_PATCH_DESCRIPTOR, Opt::Data(desc)),
            ],
            &sec,
        );
        file.extend(payload);
        file
    }
}

fn record(old: usize, new: usize, len: usize, data: &[u8]) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend((old as u32).to_be_bytes());
    r.extend((new as u32).to_be_bytes());
    r.extend((len as u16).to_be_bytes());
    r.extend((data.len() as u16).to_be_bytes());
    r.extend(data);
    r
}

/// Hashed block chain; returns the first block's (size, hash) and the data.
fn chain(payloads: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
    let (mut next, mut data) = (vec![0u8; 24], Vec::new());
    for p in payloads.iter().rev() {
        let mut block = next.clone();
        block.extend(p);
        next = (block.len() as u32).to_be_bytes().to_vec();
        next.extend(Sha1::digest(&block));
        data.splice(0..0, block);
    }
    (next, data)
}

fn fmt_info(compression: u16, rest: &[u8]) -> Vec<u8> {
    let mut v = ((8 + rest.len()) as u32).to_be_bytes().to_vec();
    v.extend(0u16.to_be_bytes());
    v.extend(compression.to_be_bytes());
    v.extend(rest);
    v
}

fn with_encryption(mut format: Vec<u8>, encrypted: bool) -> Vec<u8> {
    format[5] = u8::from(encrypted);
    format
}

fn security(image_size: u32, load: u32, wrapped_key: &[u8; 16]) -> Vec<u8> {
    let mut s = vec![0u8; 0x184];
    s[4..8].copy_from_slice(&image_size.to_be_bytes());
    s[0x110..0x114].copy_from_slice(&load.to_be_bytes());
    s[0x150..0x160].copy_from_slice(wrapped_key);
    s
}

fn imports(spec: &XexSpec) -> Vec<u8> {
    let strings = b"kernel.lib\0\0".to_vec();
    let mut lib = (48u32).to_be_bytes().to_vec();
    lib.extend([0u8; 20 + 12]);
    lib.extend(0u16.to_be_bytes());
    lib.extend(2u16.to_be_bytes());
    lib.extend(spec.import_addr.to_be_bytes());
    lib.extend((spec.import_addr + 4).to_be_bytes());
    let mut v = ((12 + strings.len() + lib.len()) as u32)
        .to_be_bytes()
        .to_vec();
    v.extend((strings.len() as u32).to_be_bytes());
    v.extend(1u32.to_be_bytes());
    v.extend(strings);
    v.extend(lib);
    v
}

enum Opt {
    Inline(u32),
    Data(Vec<u8>),
}

fn header(flags: u32, opts: &[(u32, Opt)], security: &[u8]) -> Vec<u8> {
    let mut head = Vec::new();
    let mut data = Vec::new();
    let data_start = 0x18 + opts.len() * 8;
    for (k, o) in opts {
        let v = match o {
            Opt::Inline(v) => *v,
            Opt::Data(d) => {
                let at = data_start + data.len();
                data.extend(d);
                data.resize(data.len().next_multiple_of(4), 0);
                at as u32
            }
        };
        head.extend(k.to_be_bytes());
        head.extend(v.to_be_bytes());
    }
    let security_offset = (data_start + data.len()).next_multiple_of(16);
    let header_size = (security_offset + security.len()).next_multiple_of(0x100);
    let mut f = b"XEX2".to_vec();
    for v in [
        flags,
        header_size as u32,
        0,
        security_offset as u32,
        opts.len() as u32,
    ] {
        f.extend(v.to_be_bytes());
    }
    f.extend(head);
    f.extend(data);
    f.resize(security_offset, 0);
    f.extend(security);
    f.resize(header_size, 0);
    f
}

fn find_opt(file: &[u8], k: u32) -> u32 {
    let count = u32::from_be_bytes(file[0x14..0x18].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| 0x18 + i * 8)
        .find(|&o| file[o..o + 4] == k.to_be_bytes())
        .map(|o| u32::from_be_bytes(file[o + 4..o + 8].try_into().unwrap()))
        .unwrap()
}
