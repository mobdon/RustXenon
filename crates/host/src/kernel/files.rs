//! File access on the player's own unpacked game folder (read only) and a
//! local data folder for everything a title writes (cache, saves).
//!
//! Guest paths look like `game:\dir\file`, `d:\dir\file` or
//! `\Device\Cdrom0\dir\file`. A link table maps device prefixes to host
//! folders; titles add links with `ObCreateSymbolicLink` and content
//! packages (`XamContentCreateEx`) add theirs.

use super::*;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct File {
    pub path: PathBuf,
    pub file: Option<fs::File>,
    pub pos: u64,
    /// Directory listing for `NtQueryDirectoryFile`, and the next entry.
    pub listing: Option<(Vec<(String, fs::Metadata)>, usize)>,
}

/// (prefix, host folder, writable). Prefixes are lower case.
pub fn default_links(
    game: &Path,
    data: &Path,
    update: Option<&Path>,
) -> Vec<(String, PathBuf, bool)> {
    let w = |s: &str| data.join(s);
    let mut v: Vec<(String, PathBuf, bool)> = update
        .map(|u| ("update:".into(), u.into(), false))
        .into_iter()
        .collect();
    v.extend([
        ("game:".into(), game.into(), false),
        ("d:".into(), game.into(), false),
        ("\\device\\cdrom0".into(), game.into(), false),
        ("cache:".into(), w("cache"), true),
        ("\\device\\harddisk0\\cache0".into(), w("cache"), true),
        ("\\device\\harddisk0\\cache1".into(), w("cache"), true),
        ("\\device\\harddisk0\\partition1".into(), w("hdd"), true),
        ("\\device\\harddisk0\\partition0".into(), w("hdd"), true),
        ("hdd:".into(), w("hdd"), true),
        ("\\device\\mu0".into(), w("mu"), true),
    ]);
    v
}

/// Maps a guest path to (host path, writable). Link targets are resolved
/// when a link is made, so one lookup is enough.
pub fn resolve(guest: &str) -> Option<(PathBuf, bool)> {
    let mut p = guest.replace('/', "\\");
    if let Some(r) = p.strip_prefix(r"\??\") {
        p = r.to_string();
    }
    let low = p.to_ascii_lowercase();
    let links = k().links.lock().unwrap();
    // Longest prefix wins.
    let (pre, host, w) = links
        .iter()
        .filter(|(pre, ..)| {
            low.starts_with(pre.as_str())
                && matches!(low.as_bytes().get(pre.len()), None | Some(b'\\'))
        })
        .max_by_key(|(pre, ..)| pre.len())?;
    let mut out = host.clone();
    for part in p[pre.len()..]
        .split('\\')
        .filter(|s| !s.is_empty() && *s != ".")
    {
        if part == ".." {
            return None;
        }
        out.push(part);
    }
    Some((out, *w))
}

/// Name from OBJECT_ATTRIBUTES (root directory handle, name string).
fn object_path(rt: &Runtime, attrs: u32) -> Option<(PathBuf, bool)> {
    let root = rt.mem.read_u32(attrs);
    let name = ansi(rt, rt.mem.read_u32(attrs + 4));
    if root != 0 && root != 0xFFFF_FFFD {
        if let Some(Handle::File(f)) = k().handle(root) {
            let base = f.lock().unwrap().path.clone();
            let w = base.starts_with(&k().data_dir);
            let mut out = base;
            for part in name.split(['\\', '/']).filter(|s| !s.is_empty()) {
                if part == ".." {
                    return None;
                }
                out.push(part);
            }
            return Some((out, w));
        }
    }
    resolve(&name)
}

fn iosb(rt: &Runtime, p: u32, status: u32, info: u32) {
    if p != 0 {
        rt.mem.write_u32(p, status);
        rt.mem.write_u32(p + 4, info);
    }
}

fn signal(rt: &Runtime, c: &Context, event: u32) {
    if event != 0 {
        if let Some(a) = super::objects::object_of(rt, c, event) {
            if let super::objects::KObj::Event(e) = &*super::objects::get(rt, a) {
                e.set();
            }
        }
    }
}

/// FILETIME (100 ns since 1601) of a host time.
fn filetime(t: std::io::Result<std::time::SystemTime>) -> u64 {
    let unix = t
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .unwrap_or_default();
    116_444_736_000_000_000 + unix.as_nanos() as u64 / 100
}

/// Attributes: 0x10 directory, 0x80 normal, 0x01 read only.
fn attributes(m: &fs::Metadata, writable: bool) -> u32 {
    if m.is_dir() {
        0x10
    } else if writable {
        0x80
    } else {
        0x01
    }
}

/// FILE_NETWORK_OPEN_INFORMATION: four times, allocation size, end of file,
/// attributes (56 bytes).
fn network_info(rt: &Runtime, out: u32, m: &fs::Metadata, w: bool) {
    let t = filetime(m.modified());
    for i in 0..4 {
        rt.mem.write_u64(out + 8 * i, t);
    }
    rt.mem.write_u64(out + 32, (m.len() + 0xFFF) & !0xFFF);
    rt.mem.write_u64(out + 40, m.len());
    rt.mem.write_u32(out + 48, attributes(m, w));
    rt.mem.write_u32(out + 52, 0);
}

fn open(
    rt: &Runtime,
    c: &mut Context,
    out: u32,
    attrs: u32,
    io: u32,
    disposition: u32,
    options: u32,
) {
    open_inner(rt, c, out, attrs, io, disposition, options);
    let status = c.r[3] as u32;
    if status != STATUS_SUCCESS || std::env::var_os("XENON_TRACE_FILES").is_some() {
        let name = ansi(rt, rt.mem.read_u32(attrs + 4));
        let h = if status == 0 { rt.mem.read_u32(out) } else { 0 };
        crate::log(&format!(
            "[files] open {name:?} (disposition {disposition}): {status:#010x}, handle {h:#x}"
        ));
    }
}

fn open_inner(
    rt: &Runtime,
    c: &mut Context,
    out: u32,
    attrs: u32,
    io: u32,
    disposition: u32,
    options: u32,
) {
    let Some((path, writable)) = object_path(rt, attrs) else {
        iosb(rt, io, STATUS_OBJECT_PATH_NOT_FOUND, 0);
        return ret(c, STATUS_OBJECT_PATH_NOT_FOUND);
    };
    let exists = path.exists();
    let want_dir = options & 1 != 0;
    // 0 supersede, 1 open, 2 create, 3 open if, 4 overwrite, 5 overwrite if
    let status = match (disposition, exists) {
        (1 | 4, false) => Some(STATUS_OBJECT_NAME_NOT_FOUND),
        (2, true) => Some(STATUS_OBJECT_NAME_COLLISION),
        (0 | 2..=5, _) if !writable && !(exists && disposition == 3) => Some(STATUS_ACCESS_DENIED),
        _ => None,
    };
    if let Some(s) = status {
        iosb(rt, io, s, 0);
        return ret(c, s);
    }
    let mut info = 1; // FILE_OPENED
    let file = if want_dir || (exists && path.is_dir()) {
        if !exists && fs::create_dir_all(&path).is_err() {
            iosb(rt, io, STATUS_ACCESS_DENIED, 0);
            return ret(c, STATUS_ACCESS_DENIED);
        }
        if !exists {
            info = 2;
        }
        None
    } else {
        let mut o = fs::OpenOptions::new();
        o.read(true);
        if writable {
            o.write(true);
            match disposition {
                0 | 4 | 5 => {
                    o.create(true).truncate(true);
                    info = if exists { 3 } else { 2 };
                }
                2 | 3 => {
                    o.create(true);
                    if !exists {
                        info = 2;
                    }
                }
                _ => {}
            }
            if let Some(p) = path.parent() {
                let _ = fs::create_dir_all(p);
            }
        }
        match o.open(&path) {
            Ok(f) => Some(f),
            Err(_) => {
                iosb(rt, io, STATUS_OBJECT_NAME_NOT_FOUND, 0);
                return ret(c, STATUS_OBJECT_NAME_NOT_FOUND);
            }
        }
    };
    let h = k().add_handle(Handle::File(Arc::new(Mutex::new(File {
        path,
        file,
        pos: 0,
        listing: None,
    }))));
    rt.mem.write_u32(out, h);
    iosb(rt, io, STATUS_SUCCESS, info);
    ret(c, STATUS_SUCCESS);
}

pub fn nt_create_file(rt: &Runtime, c: &mut Context) {
    let (out, attrs, io, disposition, options) = (
        arg(rt, c, 0),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 7),
        arg(rt, c, 8),
    );
    open(rt, c, out, attrs, io, disposition, options);
}

pub fn nt_open_file(rt: &Runtime, c: &mut Context) {
    let (out, attrs, io, options) = (arg(rt, c, 0), arg(rt, c, 2), arg(rt, c, 3), arg(rt, c, 4));
    open(rt, c, out, attrs, io, 1, options);
}

fn file(h: u32) -> Option<Arc<Mutex<File>>> {
    let f = match k().handle(h) {
        Some(Handle::File(f)) => Some(f),
        _ => None,
    };
    if f.is_none() {
        crate::log(&format!(
            "[files] {}: {h:#x} is not a file handle",
            super::current_import()
        ));
    }
    f
}

/// Byte offset argument: null or -2 means "current position".
fn offset(rt: &Runtime, p: u32, f: &File) -> u64 {
    if p == 0 {
        return f.pos;
    }
    match rt.mem.read_u64(p) {
        0xFFFF_FFFF_FFFF_FFFE => f.pos,
        v => v,
    }
}

fn read_into(rt: &Runtime, f: &mut File, at: u64, dst: &[(u32, u32)]) -> std::io::Result<u32> {
    let Some(h) = f.file.as_mut() else {
        return Ok(0);
    };
    h.seek(SeekFrom::Start(at))?;
    let mut total = 0;
    for &(a, n) in dst {
        let mut b = vec![0; n as usize];
        let mut got = 0;
        while got < b.len() {
            match h.read(&mut b[got..])? {
                0 => break,
                x => got += x,
            }
        }
        rt.mem.write_bytes(a, &b[..got]);
        total += got as u32;
        if got < b.len() {
            break;
        }
    }
    f.pos = at + total as u64;
    Ok(total)
}

fn finish_read(
    rt: &Runtime,
    c: &mut Context,
    r: std::io::Result<u32>,
    io: u32,
    event: u32,
    len: u32,
) {
    let (status, n) = match r {
        Ok(0) if len > 0 => (STATUS_END_OF_FILE, 0),
        Ok(n) => (STATUS_SUCCESS, n),
        Err(_) => (STATUS_UNSUCCESSFUL, 0),
    };
    if status != STATUS_SUCCESS || n < len {
        crate::log(&format!(
            "[files] read of {len} bytes gave {n}: {status:#010x}"
        ));
    }
    iosb(rt, io, status, n);
    signal(rt, c, event);
    ret(c, status);
}

pub fn nt_read_file(rt: &Runtime, c: &mut Context) {
    let (h, event, apc, io, buf, len, off) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
        arg(rt, c, 7),
    );
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    if apc != 0 {
        eprintln!("[kernel] read completion routine {apc:#x} not run");
    }
    let mut f = f.lock().unwrap();
    let at = offset(rt, off, &f);
    let r = read_into(rt, &mut f, at, &[(buf, len)]);
    drop(f);
    finish_read(rt, c, r, io, event, len);
}

/// Reads into a list of 4 KB pages.
pub fn nt_read_file_scatter(rt: &Runtime, c: &mut Context) {
    let (h, event, io, segs, len, off) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
        arg(rt, c, 7),
    );
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let pages: Vec<(u32, u32)> = (0..len.div_ceil(0x1000))
        .map(|i| {
            (
                rt.mem.read_u32(segs + 8 * i),
                (len - 0x1000 * i).min(0x1000),
            )
        })
        .collect();
    let mut f = f.lock().unwrap();
    let at = offset(rt, off, &f);
    let r = read_into(rt, &mut f, at, &pages);
    drop(f);
    finish_read(rt, c, r, io, event, len);
}

pub fn nt_write_file(rt: &Runtime, c: &mut Context) {
    let (h, event, io, buf, len, off) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
        arg(rt, c, 7),
    );
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let mut f = f.lock().unwrap();
    let at = offset(rt, off, &f);
    let mut b = vec![0; len as usize];
    rt.mem.read_bytes(buf, &mut b);
    let r = match f.file.as_mut() {
        Some(x) => x.seek(SeekFrom::Start(at)).and_then(|_| x.write_all(&b)),
        None => Err(std::io::ErrorKind::PermissionDenied.into()),
    };
    let status = if r.is_ok() {
        f.pos = at + len as u64;
        STATUS_SUCCESS
    } else {
        STATUS_ACCESS_DENIED
    };
    drop(f);
    iosb(rt, io, status, if status == 0 { len } else { 0 });
    signal(rt, c, event);
    ret(c, status);
}

fn sector(p: &Path) -> u32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    p.hash(&mut h);
    (h.finish() as u32) & 0x00FF_FFFF
}

fn meta(f: &File) -> Option<fs::Metadata> {
    fs::metadata(&f.path).ok()
}

pub fn nt_query_information_file(rt: &Runtime, c: &mut Context) {
    let (h, io, out, len, class) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
    );
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let f = f.lock().unwrap();
    let Some(m) = meta(&f) else {
        return ret(c, STATUS_NO_SUCH_FILE);
    };
    let w = f.path.starts_with(&k().data_dir);
    let n = match class {
        // FileBasicInformation: four times, attributes.
        4 => {
            let t = filetime(m.modified());
            for i in 0..4 {
                rt.mem.write_u64(out + 8 * i, t);
            }
            rt.mem.write_u32(out + 32, attributes(&m, w));
            40
        }
        // FileStandardInformation: allocation, end of file, links,
        // delete pending, directory.
        5 => {
            rt.mem.write_u64(out, (m.len() + 0xFFF) & !0xFFF);
            rt.mem.write_u64(out + 8, m.len());
            rt.mem.write_u32(out + 16, 1);
            rt.mem.write_u8(out + 20, 0);
            rt.mem.write_u8(out + 21, m.is_dir() as u8);
            24
        }
        // FilePositionInformation
        14 => {
            rt.mem.write_u64(out, f.pos);
            8
        }
        // FileInternalInformation: a file index.
        6 => {
            rt.mem.write_u64(out, sector(&f.path) as u64);
            8
        }
        // FileModeInformation, FileAlignmentInformation,
        // FileIoPriorityInformation
        16 | 17 | 32 => {
            rt.mem.write_u32(out, 0);
            4
        }
        // FileSectorInformation: where the file starts on the disc. Any
        // stable number keeps a title's read ordering sensible.
        26 => {
            rt.mem.write_u32(out, sector(&f.path));
            4
        }
        // FileEndOfFileInformation
        20 => {
            rt.mem.write_u64(out, m.len());
            8
        }
        // FileNetworkOpenInformation
        34 => {
            network_info(rt, out, &m, w);
            56
        }
        _ => {
            eprintln!("[kernel] file information class {class} not supported");
            return ret(c, STATUS_INVALID_PARAMETER);
        }
    };
    let _ = len;
    iosb(rt, io, STATUS_SUCCESS, n);
    ret(c, STATUS_SUCCESS);
}

pub fn nt_set_information_file(rt: &Runtime, c: &mut Context) {
    let (h, io, info, class) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2), arg(rt, c, 4));
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let mut f = f.lock().unwrap();
    let s = match class {
        14 => {
            f.pos = rt.mem.read_u64(info);
            STATUS_SUCCESS
        }
        19 | 20 => match f.file.as_mut().map(|x| x.set_len(rt.mem.read_u64(info))) {
            Some(Ok(())) => STATUS_SUCCESS,
            _ => STATUS_ACCESS_DENIED,
        },
        // FileDispositionInformation: delete on close.
        13 => {
            if rt.mem.read_u8(info) != 0 && f.path.starts_with(&k().data_dir) {
                f.file = None;
                let _ = fs::remove_file(&f.path);
            }
            STATUS_SUCCESS
        }
        _ => {
            eprintln!("[kernel] set file information class {class} not supported");
            STATUS_INVALID_PARAMETER
        }
    };
    iosb(rt, io, s, 0);
    ret(c, s);
}

pub fn nt_query_full_attributes_file(rt: &Runtime, c: &mut Context) {
    let (attrs, out) = (arg(rt, c, 0), arg(rt, c, 1));
    match object_path(rt, attrs).and_then(|(p, w)| Some((fs::metadata(p).ok()?, w))) {
        Some((m, w)) => {
            network_info(rt, out, &m, w);
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_OBJECT_NAME_NOT_FOUND),
    }
}

pub fn nt_query_volume_information_file(rt: &Runtime, c: &mut Context) {
    let (h, io, out, len, class) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
    );
    if file(h).is_none() {
        return ret(c, STATUS_INVALID_HANDLE);
    }
    let n = match class {
        // FileFsVolumeInformation: creation time, serial, label length,
        // supports objects, label.
        1 => {
            fill(rt, out, len.min(24), 0);
            rt.mem.write_u32(out + 8, 0x1234_5678);
            18
        }
        // FileFsSizeInformation: total units, free units, sectors per
        // unit, bytes per sector.
        3 => {
            rt.mem.write_u64(out, 0x10_0000);
            rt.mem.write_u64(out + 8, 0x8_0000);
            rt.mem.write_u32(out + 16, 8);
            rt.mem.write_u32(out + 20, 512);
            24
        }
        // FileFsAttributeInformation: attributes, max name length, name.
        5 => {
            rt.mem.write_u32(out, 0);
            rt.mem.write_u32(out + 4, 255);
            rt.mem.write_u32(out + 8, 4);
            rt.mem.write_bytes(out + 12, b"FATX");
            16
        }
        _ => {
            eprintln!("[kernel] volume information class {class} not supported");
            return ret(c, STATUS_INVALID_PARAMETER);
        }
    };
    iosb(rt, io, STATUS_SUCCESS, n);
    ret(c, STATUS_SUCCESS);
}

/// `*` and `?` match, case-insensitive.
fn matches(pat: &[u8], s: &[u8]) -> bool {
    match (pat.first(), s.first()) {
        (None, None) => true,
        (Some(b'*'), _) => matches(&pat[1..], s) || (!s.is_empty() && matches(pat, &s[1..])),
        (Some(b'?'), Some(_)) => matches(&pat[1..], &s[1..]),
        (Some(p), Some(x)) => p.eq_ignore_ascii_case(x) && matches(&pat[1..], &s[1..]),
        _ => false,
    }
}

/// One FILE_DIRECTORY_INFORMATION entry per call: next offset, index, four
/// times, end of file, allocation size, attributes, name length, name.
pub fn nt_query_directory_file(rt: &Runtime, c: &mut Context) {
    let (h, event, io, out, len, name, restart) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
        arg(rt, c, 7),
        arg(rt, c, 8),
    );
    let Some(f) = file(h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let mut f = f.lock().unwrap();
    if f.listing.is_none() || restart != 0 {
        let pat = if name != 0 {
            ansi(rt, name)
        } else {
            "*".into()
        };
        let mut v: Vec<_> = fs::read_dir(&f.path)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| Some((e.file_name().into_string().ok()?, e.metadata().ok()?)))
            .filter(|(n, _)| matches(pat.as_bytes(), n.as_bytes()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        let empty = v.is_empty();
        f.listing = Some((v, 0));
        if empty {
            iosb(rt, io, STATUS_NO_SUCH_FILE, 0);
            return ret(c, STATUS_NO_SUCH_FILE);
        }
    }
    let w = f.path.starts_with(&k().data_dir);
    let (list, i) = f.listing.as_mut().unwrap();
    let Some((n, m)) = list.get(*i) else {
        iosb(rt, io, STATUS_NO_MORE_FILES, 0);
        signal(rt, c, event);
        return ret(c, STATUS_NO_MORE_FILES);
    };
    let need = 0x40 + n.len() as u32;
    if need > len {
        return ret(c, 0x8000_0005); // STATUS_BUFFER_OVERFLOW
    }
    let t = filetime(m.modified());
    rt.mem.write_u32(out, 0);
    rt.mem.write_u32(out + 4, *i as u32);
    for j in 0..4 {
        rt.mem.write_u64(out + 8 + 8 * j, t);
    }
    rt.mem.write_u64(out + 40, m.len());
    rt.mem.write_u64(out + 48, (m.len() + 0xFFF) & !0xFFF);
    rt.mem.write_u32(out + 56, attributes(m, w));
    rt.mem.write_u32(out + 60, n.len() as u32);
    rt.mem.write_bytes(out + 64, n.as_bytes());
    *i += 1;
    drop(f);
    iosb(rt, io, STATUS_SUCCESS, need);
    signal(rt, c, event);
    ret(c, STATUS_SUCCESS);
}

pub fn nt_flush_buffers_file(rt: &Runtime, c: &mut Context) {
    iosb(rt, arg(rt, c, 1), STATUS_SUCCESS, 0);
    ret(c, STATUS_SUCCESS);
}

pub fn nt_device_io_control_file(rt: &Runtime, c: &mut Context) {
    eprintln!("[kernel] device control {:#x} refused", arg(rt, c, 5));
    ret(c, STATUS_NOT_IMPLEMENTED);
}

pub fn ob_create_symbolic_link(rt: &Runtime, c: &mut Context) {
    let (from, to) = (ansi(rt, arg(rt, c, 0)), ansi(rt, arg(rt, c, 1)));
    let from = from.trim_start_matches("\\??\\").to_ascii_lowercase();
    match resolve(&to) {
        Some((host, w)) => {
            let mut l = k().links.lock().unwrap();
            l.retain(|x| x.0 != from);
            l.push((from, host, w));
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_OBJECT_NAME_NOT_FOUND),
    }
}

pub fn ob_delete_symbolic_link(rt: &Runtime, c: &mut Context) {
    let from = ansi(rt, arg(rt, c, 0))
        .trim_start_matches("\\??\\")
        .to_ascii_lowercase();
    k().links.lock().unwrap().retain(|x| x.0 != from);
    ret(c, STATUS_SUCCESS);
}

/// Links a content root name (`save`) to a local folder.
pub fn link_root(root: &str, host: PathBuf) {
    let pre = format!("{}:", root.trim_end_matches(':').to_ascii_lowercase());
    let _ = fs::create_dir_all(&host);
    let mut l = k().links.lock().unwrap();
    l.retain(|x| x.0 != pre);
    l.push((pre, host, true));
}

pub fn unlink_root(root: &str) {
    let pre = format!("{}:", root.trim_end_matches(':').to_ascii_lowercase());
    k().links.lock().unwrap().retain(|x| x.0 != pre);
}

// Driver-side I/O manager calls: the title's own device objects. Nothing to
// do without real devices.
pub fn io_create_device(rt: &Runtime, c: &mut Context) {
    let out = arg(rt, c, 5);
    if out != 0 {
        let a = k().pool_alloc(rt, 0x100);
        rt.mem.write_u32(out, a);
    }
    ret(c, STATUS_SUCCESS);
}
pub fn io_delete_device(_: &Runtime, _: &mut Context) {}
pub fn io_check_share_access(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
pub fn io_set_share_access(_: &Runtime, _: &mut Context) {}
pub fn io_remove_share_access(_: &Runtime, _: &mut Context) {}
pub fn io_complete_request(_: &Runtime, _: &mut Context) {}
pub fn io_invalid_device_request(_: &Runtime, c: &mut Context) {
    ret(c, 0xC000_0010);
}
pub fn io_dismount_volume(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
pub fn io_dismount_volume_by_file_handle(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_SUCCESS);
}
pub fn stfs_create_device(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_UNSUCCESSFUL);
}
pub fn stfs_control_device(_: &Runtime, c: &mut Context) {
    ret(c, STATUS_UNSUCCESSFUL);
}

#[cfg(test)]
mod tests {
    #[test]
    fn wildcards() {
        assert!(super::matches(b"*.DAT", b"level.dat"));
        assert!(super::matches(b"a?c", b"abc"));
        assert!(!super::matches(b"a?c", b"ac"));
        assert!(super::matches(b"*", b""));
    }
}
