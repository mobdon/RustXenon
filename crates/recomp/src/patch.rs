//! Title-update (delta patch) application, in memory.
//!
//! A patch is itself an XEX2 file whose image is a list of delta records:
//! zero a range, move a range, or LZX-decode new bytes with the old bytes as
//! the reference window. One record list patches the header, a chain of
//! hashed blocks patches the image. The result is a plain executable (no
//! encryption, no compression) that [`crate::xex::load`] reads.
//!
//! The steps follow XenonRecomp's `XenonUtils/xex_patcher.cpp` (MIT; that
//! file notes it was referenced from Xenia, BSD-3; see CREDITS.md). Records
//! are variable length and unaligned, so every field is read as bytes and
//! every range is bounds-checked.

use crate::xex::{
    aes_cbc_decrypt, aes_ecb_decrypt, be16, be32, block_chain, decode_image, err, key, slice,
    Compression, Error, Format, Header, Result, Security, DEVKIT_KEY,
};

const MODULE_PATCH: u32 = 0x10 | 0x20 | 0x40;
/// Offset of the first record inside the delta patch descriptor.
const DESCRIPTOR_RECORDS: usize = 0x4C;

/// Applies delta records to `dst`. Stops at an all-zero record or the end.
fn apply_records(recs: &[u8], window: u32, dst: &mut [u8]) -> Result<()> {
    let mut p = 0;
    while p + 12 <= recs.len() {
        let old = be32(recs, p)? as usize;
        let new = be32(recs, p + 4)? as usize;
        let len = be16(recs, p + 8)? as usize;
        let packed = be16(recs, p + 10)? as usize;
        if old == 0 && new == 0 && len == 0 && packed == 0 {
            break;
        }
        let bad = || Error(format!("delta record at {p:#x} writes outside its target"));
        let target = new..new.checked_add(len).ok_or_else(bad)?;
        if target.end > dst.len() {
            return Err(bad());
        }
        match packed {
            0 => dst[target].fill(0),
            1 => {
                if old + len > dst.len() {
                    return Err(bad());
                }
                dst.copy_within(old..old + len, new);
            }
            _ => {
                let data = slice(recs, p + 12, packed)?;
                let reference = slice(dst, old, len)?.to_vec();
                let out = crate::lzx::decompress(data, len, window, Some(&reference))?;
                dst[target].copy_from_slice(&out);
            }
        }
        p += 12 + if packed > 1 { packed } else { 0 };
    }
    Ok(())
}

/// Applies `patch` to `base`, both as read from the player's files.
pub fn apply(base: &[u8], patch: &[u8], keys: &[[u8; 16]]) -> Result<Vec<u8>> {
    let bh = Header::parse(base)?;
    let ph = Header::parse(patch).map_err(|e| Error(format!("patch: {e}")))?;
    if ph.module_flags & MODULE_PATCH == 0 {
        return err("patch: not a patch module");
    }
    let Some((d, _)) = ph.data(patch, key::DELTA_PATCH_DESCRIPTOR)? else {
        return err("patch: missing delta descriptor");
    };
    let (pfmt, _) = Format::parse(patch, &ph)?;
    let Compression::Delta { window, first } = pfmt.compression else {
        return err("patch: not delta compressed");
    };

    let field = |i: usize| be32(patch, d + 0x30 + i * 4);
    let desc_size = be32(patch, d)? as usize;
    let image_key_source: [u8; 16] = slice(patch, d + 0x20, 16)?.try_into().unwrap();
    let (target_headers, hdr_src_off, hdr_src_size, hdr_dst_off) = (
        field(0)? as usize,
        field(1)? as usize,
        field(2)? as usize,
        field(3)? as usize,
    );
    let (img_src_off, img_src_size, img_dst_off) =
        (field(4)? as usize, field(5)? as usize, field(6)? as usize);

    let base_hs = bh.header_size as usize;
    if hdr_src_off > base_hs
        || hdr_src_size > base_hs - hdr_src_off
        || hdr_dst_off > target_headers
        || hdr_src_size > target_headers - hdr_dst_off
    {
        return err("patch does not fit this executable's header");
    }
    let header_size = match target_headers {
        0 => hdr_dst_off + hdr_src_size,
        n => n,
    };

    // New header: base bytes, then the header delta records.
    let mut hdr = vec![0u8; header_size.max(base_hs)];
    let n = header_size.min(base.len());
    hdr[..n].copy_from_slice(&base[..n]);
    if hdr_src_off > 0 {
        hdr.copy_within(hdr_src_off..hdr_src_off + hdr_src_size, hdr_dst_off);
    }
    let recs = slice(
        patch,
        d + DESCRIPTOR_RECORDS,
        desc_size.saturating_sub(DESCRIPTOR_RECORDS),
    )?;
    apply_records(recs, window, &mut hdr)?;
    hdr.truncate(header_size);

    let nh = Header::parse(&hdr).map_err(|e| Error(format!("patched header: {e}")))?;
    let nsec = Security::parse(&hdr, nh.security_offset as usize)?;
    let bsec = Security::parse(base, bh.security_offset as usize)?;
    let psec = Security::parse(patch, ph.security_offset as usize)?;

    // The patch names the base's image key; that also tells us the console key.
    let mut matched = None;
    for k in keys.iter().chain([&DEVKIT_KEY]) {
        let base_key = aes_ecb_decrypt(k, &bsec.aes_key);
        let new_key = aes_ecb_decrypt(k, &nsec.aes_key);
        if aes_ecb_decrypt(&new_key, &image_key_source) == base_key {
            matched = Some((base_key, aes_ecb_decrypt(&new_key, &psec.aes_key)));
            break;
        }
    }
    let Some((base_key, patch_key)) = matched else {
        return err("patch is not for this executable (or the console key is missing)");
    };

    let (bfmt, _) = Format::parse(base, &bh)?;
    let mut image = decode_image(base, &bh, &bsec, &bfmt, Some(&base_key))?;
    image.resize(nsec.image_size as usize, 0);

    let mut payload = slice(
        patch,
        ph.header_size as usize,
        patch.len().saturating_sub(ph.header_size as usize),
    )?
    .to_vec();
    if pfmt.encrypted {
        aes_cbc_decrypt(&patch_key, &mut payload);
    }
    if img_src_off > 0 {
        if img_src_off + img_src_size > image.len() || img_dst_off + img_src_size > image.len() {
            return err("patch image move outside image");
        }
        image.copy_within(img_src_off..img_src_off + img_src_size, img_dst_off);
    }
    for block in block_chain(&payload, first)? {
        apply_records(block, window, &mut image)?;
    }

    // Mark the result as plain: no encryption, no compression.
    let (_, fat) = Format::parse(&hdr, &nh)?;
    hdr[fat + 4..fat + 8].fill(0);
    hdr.resize(nh.header_size as usize, 0);
    hdr.extend_from_slice(&image);
    Ok(hdr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{lzx_encode, LzxBlock, Packing, Tok, XexSpec};
    use crate::xex::load;

    const KEY: [u8; 16] = *b"test console key";

    fn record(old: u32, new: u32, len: u16, data: &[u8]) -> Vec<u8> {
        let mut r = Vec::new();
        r.extend(old.to_be_bytes());
        r.extend(new.to_be_bytes());
        r.extend(len.to_be_bytes());
        r.extend(
            (if data.is_empty() {
                0u16
            } else {
                data.len() as u16
            })
            .to_be_bytes(),
        );
        r.extend(data);
        r
    }

    #[test]
    fn records_zero_move_and_lzx() {
        let mut dst = b"0123456789abcdef".to_vec();
        let lzx = lzx_encode(
            0x8000,
            &[LzxBlock::Compressed {
                toks: vec![Tok::Match { len: 4, off: 4 }, Tok::Lit(b'!')],
                aligned: false,
            }],
        );
        let mut recs = record(0, 0, 2, &[]); // zero "01"
        let mut mv = record(4, 8, 2, &[]);
        mv[11] = 1; // move "45" over "89"
        recs.extend(mv);
        recs.extend(record(10, 11, 5, &lzx)); // "bcde" from the old bytes, then '!'
        recs.extend([0u8; 12]);
        recs.extend(record(0, 0, 4, &[])); // after terminator: ignored
        apply_records(&recs, 0x8000, &mut dst).unwrap();
        assert_eq!(&dst, b"\x00\x0023456745abcde!");

        let bad = record(0, 15, 4, &[]);
        assert!(apply_records(&bad, 0x8000, &mut dst).is_err());
    }

    #[test]
    fn title_update_round_trip() {
        let spec = XexSpec::sample();
        let old = spec.image();
        let mut new = old.clone();
        let code = (spec.code_rva() as usize) + 8;
        new[code..code + 4].copy_from_slice(&0x6000_0000u32.to_be_bytes()); // nop
        new.extend([0xEE; 0x100]); // image grows

        for packing in [Packing::Basic, Packing::Lzx] {
            for key in [Some(KEY), None] {
                let base = spec.build(&old, packing, key);
                let patch = spec.build_patch(&base, &old, &new, key);
                let out =
                    apply(&base, &patch, &[KEY]).unwrap_or_else(|e| panic!("{packing:?}: {e}"));
                let img = load(&out, &[]).unwrap();
                assert_eq!(img.data, new, "{packing:?} {key:?}");
                assert_eq!(img.version, Some(2), "header records applied");
            }
        }
    }

    #[test]
    fn patch_for_another_executable_is_refused() {
        let spec = XexSpec::sample();
        let old = spec.image();
        let base = spec.build(&old, Packing::Basic, Some(KEY));
        let patch = spec.build_patch(&base, &old, &old, Some(KEY));
        let mut other = spec.clone();
        other.image_key = *b"another image ke";
        let other_base = other.build(&old, Packing::Basic, Some(KEY));
        assert!(apply(&other_base, &patch, &[KEY]).is_err());
        assert!(apply(&base, &base, &[KEY]).is_err(), "base is not a patch");
    }
}
