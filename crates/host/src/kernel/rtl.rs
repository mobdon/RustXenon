//! Run-time library calls: strings, formatting, time, hashing, the
//! executable header, and the calls that stop the title (bug checks,
//! exceptions, reboot).

use super::*;

/// System time: 100 ns units since 1601.
pub fn system_time() -> u64 {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    116_444_736_000_000_000 + (d.as_nanos() / 100) as u64
}

pub fn ke_query_system_time(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u64(arg(rt, c, 0), system_time());
}

/// Matches the runtime's time base (`mftb`).
pub fn ke_query_performance_frequency(_: &Runtime, c: &mut Context) {
    c.r[3] = 49_875_000;
}

// ---- formatting ------------------------------------------------------------

/// Where `%` arguments come from: integer registers then the stack (a
/// call with variable arguments), or an argument list in memory.
pub enum Args {
    Regs { next: usize, stack: u32 },
    List(u32),
}

impl Args {
    fn next(&mut self, rt: &Runtime, c: &Context) -> u64 {
        match self {
            Args::Regs { next, stack } => {
                let v = if *next <= 10 {
                    c.r[*next]
                } else {
                    let a = *stack + 8 * (*next as u32 - 11);
                    rt.mem.read_u64(a)
                };
                *next += 1;
                v
            }
            Args::List(p) => {
                let v = rt.mem.read_u64(*p);
                *p += 8;
                v
            }
        }
    }
}

/// C `printf` formatting of guest arguments (each in an 8-byte slot).
pub fn format(rt: &Runtime, c: &Context, fmt: &str, mut args: Args) -> String {
    let b = fmt.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'%' {
            let start = i;
            while i < b.len() && b[i] != b'%' {
                i += 1;
            }
            out.push_str(&fmt[start..i]);
            continue;
        }
        i += 1;
        let (mut left, mut zero, mut plus, mut space) = (false, false, false, false);
        while i < b.len() && b"-0+ #".contains(&b[i]) {
            match b[i] {
                b'-' => left = true,
                b'0' => zero = true,
                b'+' => plus = true,
                b' ' => space = true,
                _ => {}
            }
            i += 1;
        }
        let num = |i: &mut usize, args: &mut Args| -> Option<usize> {
            if b.get(*i) == Some(&b'*') {
                *i += 1;
                return Some(args.next(rt, c) as i32 as usize);
            }
            let s = *i;
            while *i < b.len() && b[*i].is_ascii_digit() {
                *i += 1;
            }
            fmt[s..*i].parse().ok()
        };
        let width = num(&mut i, &mut args).unwrap_or(0);
        let prec = if b.get(i) == Some(&b'.') {
            i += 1;
            Some(num(&mut i, &mut args).unwrap_or(0))
        } else {
            None
        };
        let mut long = 0;
        while i < b.len() && b"hlLqzjtI".contains(&b[i]) {
            if b[i] == b'l' || b[i] == b'q' || b[i] == b'L' {
                long += 1;
            }
            if b[i..].starts_with(b"I64") {
                long = 2;
                i += 2;
            }
            i += 1;
        }
        let Some(&conv) = b.get(i) else { break };
        i += 1;
        let mut s = match conv {
            b'%' => "%".to_string(),
            b'd' | b'i' => {
                let v = args.next(rt, c);
                let v = if long >= 2 { v as i64 } else { v as i32 as i64 };
                let mut t = v.unsigned_abs().to_string();
                if let Some(p) = prec {
                    t = format!("{t:0>p$}");
                }
                let sign = if v < 0 {
                    "-"
                } else if plus {
                    "+"
                } else if space {
                    " "
                } else {
                    ""
                };
                format!("{sign}{t}")
            }
            b'u' | b'x' | b'X' | b'o' => {
                let v = args.next(rt, c);
                let v = if long >= 2 { v } else { v as u32 as u64 };
                let mut t = match conv {
                    b'u' => v.to_string(),
                    b'x' => format!("{v:x}"),
                    b'X' => format!("{v:X}"),
                    _ => format!("{v:o}"),
                };
                if let Some(p) = prec {
                    t = format!("{t:0>p$}");
                }
                t
            }
            b'p' => format!("{:08X}", args.next(rt, c) as u32),
            b'c' => ((args.next(rt, c) as u8) as char).to_string(),
            b's' | b'S' => {
                let p = args.next(rt, c) as u32;
                let wide = conv == b'S' || long > 0;
                let mut t = if p == 0 {
                    "(null)".into()
                } else if wide {
                    wstr(rt, p)
                } else {
                    cstr(rt, p)
                };
                if let Some(p) = prec {
                    t = t.chars().take(p).collect();
                }
                t
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let v = f64::from_bits(args.next(rt, c));
                let p = prec.unwrap_or(6);
                match conv {
                    b'e' | b'E' => format!("{v:.p$e}"),
                    b'g' | b'G' => format!("{v}"),
                    _ => format!("{v:.p$}"),
                }
            }
            b'n' => {
                args.next(rt, c);
                String::new()
            }
            other => format!("%{}", other as char),
        };
        if s.len() < width {
            let pad = width - s.len();
            s = if left {
                format!("{s}{}", " ".repeat(pad))
            } else if zero && !matches!(conv, b's' | b'c') {
                let (sign, rest) = s.split_at(s.starts_with(['-', '+']) as usize);
                format!("{sign}{}{rest}", "0".repeat(pad))
            } else {
                format!("{}{s}", " ".repeat(pad))
            };
        }
        out.push_str(&s);
    }
    out
}

/// Reads a zero-terminated big-endian UTF-16 string.
pub fn wstr(rt: &Runtime, mut a: u32) -> String {
    let mut v = Vec::new();
    while v.len() < 4096 {
        let ch = rt.mem.read_u16(a);
        if ch == 0 {
            break;
        }
        v.push(ch);
        a += 2;
    }
    String::from_utf16_lossy(&v)
}

/// Writes at most `n` bytes including the terminator; returns the length
/// written, or -1 if it did not fit (the C library's `_vsnprintf`).
fn put_limited(rt: &Runtime, dst: u32, n: u32, s: &str) -> u32 {
    let b = s.as_bytes();
    if (b.len() as u32) < n {
        write_cstr(rt, dst, s);
        b.len() as u32
    } else {
        rt.mem.write_bytes(dst, &b[..n as usize]);
        u32::MAX
    }
}

pub fn sprintf(rt: &Runtime, c: &mut Context) {
    let (dst, fmt) = (arg(rt, c, 0), cstr(rt, arg(rt, c, 1)));
    let s = format(
        rt,
        c,
        &fmt,
        Args::Regs {
            next: 5,
            stack: c.r[1] as u32 + 0x50,
        },
    );
    write_cstr(rt, dst, &s);
    ret(c, s.len() as u32);
}

pub fn vsnprintf(rt: &Runtime, c: &mut Context) {
    let (dst, n, fmt, list) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        cstr(rt, arg(rt, c, 2)),
        arg(rt, c, 3),
    );
    let s = format(rt, c, &fmt, Args::List(list));
    let r = put_limited(rt, dst, n, &s);
    ret(c, r);
}

pub fn dbg_print(rt: &Runtime, c: &mut Context) {
    let fmt = cstr(rt, arg(rt, c, 0));
    let s = format(
        rt,
        c,
        &fmt,
        Args::Regs {
            next: 4,
            stack: c.r[1] as u32 + 0x50,
        },
    );
    crate::log(&format!("[guest] {}", s.trim_end()));
    ret(c, 0);
}

/// Data sections of the image: where a title's message strings live.
static DATA_SECTIONS: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();

/// `XENON_GAME_LOG=<addr>[,<addr>...]`: hex guest addresses of functions the
/// title calls with a printf-style message (often an empty function in a
/// release build). Each becomes a hook that logs `[game] <message>` and
/// returns. Release builds fold every empty function into one, so a call is
/// logged only when its first argument is a printable string in a data
/// section of the image; other calls return silently, as before.
pub fn install_game_log(rt: &mut Runtime, image: &xenon_recomp::xex::Image) {
    let Ok(v) = std::env::var("XENON_GAME_LOG") else {
        return;
    };
    let _ = DATA_SECTIONS.set(
        image
            .sections
            .iter()
            .filter(|s| !s.code)
            .map(|s| (s.start, s.end()))
            .collect(),
    );
    for a in v.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        let addr = u32::from_str_radix(a.trim_start_matches("0x"), 16)
            .unwrap_or_else(|_| panic!("XENON_GAME_LOG: bad address {a:?}"));
        rt.set_override(addr, game_log);
    }
}

fn game_log(rt: &Runtime, c: &mut Context) {
    let p = c.r[3] as u32;
    let in_data = DATA_SECTIONS
        .get()
        .is_some_and(|v| v.iter().any(|&(s, e)| p >= s && p < e));
    if !in_data {
        return;
    }
    let raw: Vec<u8> = (0..MESSAGE_MAX)
        .map(|i| rt.mem.read_u8(p.wrapping_add(i)))
        .collect();
    let Some(fmt) = message(&raw) else {
        return;
    };
    let s = format(
        rt,
        c,
        fmt,
        Args::Regs {
            next: 4,
            stack: c.r[1] as u32 + 0x50,
        },
    );
    // Some callers repeat one message every frame: log it once per run of repeats.
    static LAST: Mutex<String> = Mutex::new(String::new());
    let s = s.trim_end();
    let mut last = LAST.lock().unwrap();
    if *last != s {
        crate::log(&format!("[game] {s}"));
        *last = s.to_string();
    }
}

const MESSAGE_MAX: u32 = 512;

/// The text before the first NUL when it reads as a message: at least two
/// characters, all printable ASCII or tab/newline. `None` for anything else.
fn message(raw: &[u8]) -> Option<&str> {
    let n = raw.iter().position(|&b| b == 0)?;
    let t = &raw[..n];
    let ok = t.len() >= 2
        && t.iter()
            .all(|&b| (0x20..0x7f).contains(&b) || b == b'\n' || b == b'\t' || b == b'\r');
    ok.then(|| std::str::from_utf8(t).unwrap())
}

// ---- strings ---------------------------------------------------------------

pub fn rtl_init_ansi_string(rt: &Runtime, c: &mut Context) {
    let (d, s) = (arg(rt, c, 0), arg(rt, c, 1));
    let n = if s == 0 { 0 } else { cstr(rt, s).len() as u16 };
    rt.mem.write_u16(d, n);
    rt.mem.write_u16(d + 2, if s == 0 { 0 } else { n + 1 });
    rt.mem.write_u32(d + 4, s);
}

pub fn rtl_init_unicode_string(rt: &Runtime, c: &mut Context) {
    let (d, s) = (arg(rt, c, 0), arg(rt, c, 1));
    let n = if s == 0 {
        0
    } else {
        2 * wstr(rt, s).encode_utf16().count() as u16
    };
    rt.mem.write_u16(d, n);
    rt.mem.write_u16(d + 2, if s == 0 { 0 } else { n + 2 });
    rt.mem.write_u32(d + 4, s);
}

pub fn rtl_free_ansi_string(rt: &Runtime, c: &mut Context) {
    let p = rt.mem.read_u32(arg(rt, c, 0) + 4);
    k().pool.lock().unwrap().free(p);
}

pub fn rtl_unicode_string_to_ansi_string(rt: &Runtime, c: &mut Context) {
    let (d, s, alloc) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let n = rt.mem.read_u16(s) as u32 / 2;
    let src = rt.mem.read_u32(s + 4);
    let text: String = (0..n)
        .map(|i| {
            char::from_u32(rt.mem.read_u16(src + 2 * i) as u32)
                .filter(char::is_ascii)
                .unwrap_or('?')
        })
        .collect();
    let buf = if alloc != 0 {
        let b = k().pool_alloc(rt, n + 1);
        rt.mem.write_u32(d + 4, b);
        rt.mem.write_u16(d + 2, (n + 1) as u16);
        b
    } else {
        if (rt.mem.read_u16(d + 2) as u32) < n + 1 {
            return ret(c, 0x8000_0005);
        }
        rt.mem.read_u32(d + 4)
    };
    write_cstr(rt, buf, &text);
    rt.mem.write_u16(d, n as u16);
    ret(c, STATUS_SUCCESS);
}

pub fn rtl_multi_byte_to_unicode_n(rt: &Runtime, c: &mut Context) {
    let (d, dn, written, s, sn) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
    );
    let n = sn.min(dn / 2);
    for i in 0..n {
        rt.mem.write_u16(d + 2 * i, rt.mem.read_u8(s + i) as u16);
    }
    if written != 0 {
        rt.mem.write_u32(written, 2 * n);
    }
    ret(c, STATUS_SUCCESS);
}

pub fn rtl_unicode_to_multi_byte_n(rt: &Runtime, c: &mut Context) {
    let (d, dn, written, s, sn) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
    );
    let n = (sn / 2).min(dn);
    for i in 0..n {
        let ch = rt.mem.read_u16(s + 2 * i);
        rt.mem
            .write_u8(d + i, if ch < 0x100 { ch as u8 } else { b'?' });
    }
    if written != 0 {
        rt.mem.write_u32(written, n);
    }
    ret(c, STATUS_SUCCESS);
}

pub fn rtl_upcase_unicode_char(rt: &Runtime, c: &mut Context) {
    let ch = arg(rt, c, 0) as u16;
    let up = char::from_u32(ch as u32)
        .and_then(|x| x.to_uppercase().next())
        .map_or(ch, |x| x as u16);
    ret(c, up as u32);
}

pub fn rtl_compare_string_n(rt: &Runtime, c: &mut Context) {
    let (a, an, b, bn, ci) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
    );
    for i in 0..an.min(bn) {
        let (mut x, mut y) = (rt.mem.read_u8(a + i), rt.mem.read_u8(b + i));
        if ci != 0 {
            x = x.to_ascii_uppercase();
            y = y.to_ascii_uppercase();
        }
        if x != y {
            return ret(c, (x as i32 - y as i32) as u32);
        }
    }
    ret(c, (an as i32 - bn as i32) as u32);
}

/// Bytes from `src` matching the repeated 32-bit pattern.
pub fn rtl_compare_memory_ulong(rt: &Runtime, c: &mut Context) {
    let (s, n, pat) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let mut i = 0;
    while i + 4 <= n && rt.mem.read_u32(s + i) == pat {
        i += 4;
    }
    ret(c, i);
}

pub fn rtl_fill_memory_ulong(rt: &Runtime, c: &mut Context) {
    let (d, n, pat) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    for i in (0..n / 4).map(|i| 4 * i) {
        rt.mem.write_u32(d + i, pat);
    }
}

/// NT status to Win32 error, for the codes titles test.
pub fn rtl_nt_status_to_dos_error(rt: &Runtime, c: &mut Context) {
    let s = arg(rt, c, 0);
    let e = match s {
        0 => 0,
        STATUS_TIMEOUT => 1460,
        STATUS_PENDING => 997,
        STATUS_NO_MORE_FILES => 18,
        STATUS_INVALID_HANDLE => 6,
        STATUS_INVALID_PARAMETER => 87,
        STATUS_NO_SUCH_FILE | STATUS_OBJECT_NAME_NOT_FOUND => 2,
        STATUS_OBJECT_PATH_NOT_FOUND => 3,
        STATUS_END_OF_FILE => 38,
        STATUS_NO_MEMORY => 8,
        STATUS_ACCESS_DENIED => 5,
        STATUS_OBJECT_NAME_COLLISION => 183,
        STATUS_NOT_IMPLEMENTED => 1,
        _ if s & 0xFFFF_0000 == 0x8007_0000 => s & 0xFFFF,
        _ => 317,
    };
    ret(c, e);
}

// ---- time fields -------------------------------------------------------------

/// Days between 1601-01-01 and 1970-01-01.
const EPOCH_DAYS: i64 = 134_774;

/// Days since 1601-01-01 to (year, month, day) (Howard Hinnant's
/// civil-from-days algorithm, public domain).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days - EPOCH_DAYS + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - (m <= 2) as i64;
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468 + EPOCH_DAYS
}

/// TIME_FIELDS: year, month, day, hour, minute, second, milliseconds,
/// weekday (16 bits each).
pub fn rtl_time_to_time_fields(rt: &Runtime, c: &mut Context) {
    let (t, out) = (rt.mem.read_u64(arg(rt, c, 0)), arg(rt, c, 1));
    let ms = t / 10_000;
    let days = (ms / 86_400_000) as i64;
    let (y, m, d) = civil(days);
    let r = ms % 86_400_000;
    let f = [
        y as u16,
        m as u16,
        d as u16,
        (r / 3_600_000) as u16,
        (r / 60_000 % 60) as u16,
        (r / 1000 % 60) as u16,
        (r % 1000) as u16,
        ((days + 1) % 7) as u16,
    ];
    for (i, v) in f.into_iter().enumerate() {
        rt.mem.write_u16(out + 2 * i as u32, v);
    }
}

pub fn rtl_time_fields_to_time(rt: &Runtime, c: &mut Context) {
    let (p, out) = (arg(rt, c, 0), arg(rt, c, 1));
    let f = |i: u32| rt.mem.read_u16(p + 2 * i) as i64;
    let days = days_from_civil(f(0), f(1) as u32, f(2) as u32);
    let ms = ((days * 24 + f(3)) * 60 + f(4)) * 60_000 + f(5) * 1000 + f(6);
    rt.mem.write_u64(out, (ms * 10_000) as u64);
    ret(c, 1);
}

// ---- executable --------------------------------------------------------------

/// The value of an optional header (keys ending 0x00/0x01), else a pointer
/// to its data in the guest copy of the header.
pub fn header_field(rt: &Runtime, h: u32, key: u32) -> u32 {
    let n = rt.mem.read_u32(h + 0x14);
    for i in 0..n.min(256) {
        if rt.mem.read_u32(h + 0x18 + 8 * i) == key {
            let v = rt.mem.read_u32(h + 0x1C + 8 * i);
            return if key & 0xFF <= 1 { v } else { h + v };
        }
    }
    0
}

pub fn rtl_image_xex_header_field(rt: &Runtime, c: &mut Context) {
    let v = header_field(rt, arg(rt, c, 0), arg(rt, c, 1));
    ret(c, v);
}

/// Privilege bits are the system flags header (key 0x00030000).
pub fn xex_check_executable_privilege(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 0);
    let flags = header_field(rt, k().xex_header, 0x0003_0000);
    ret(c, (p < 32 && flags >> p & 1 != 0) as u32);
}

pub fn xex_get_module_handle(rt: &Runtime, c: &mut Context) {
    let (name, out) = (arg(rt, c, 0), arg(rt, c, 1));
    let n = if name == 0 {
        String::new()
    } else {
        cstr(rt, name).to_ascii_lowercase()
    };
    if n.is_empty() || n == "default.xex" {
        rt.mem.write_u32(out, k().module);
        ret(c, STATUS_SUCCESS)
    } else {
        rt.mem.write_u32(out, 0);
        ret(c, 0xC000_0225) // STATUS_NOT_FOUND
    }
}

pub fn xex_get_procedure_address(rt: &Runtime, c: &mut Context) {
    crate::log(&format!(
        "[kernel] procedure lookup of ordinal {} refused",
        arg(rt, c, 1)
    ));
    rt.mem.write_u32(arg(rt, c, 2), 0);
    ret(c, 0xC000_0225);
}

pub fn ex_get_x_config_setting(rt: &Runtime, c: &mut Context) {
    let (cat, set, buf, size, req) = (
        arg(rt, c, 0) as u16,
        arg(rt, c, 1) as u16,
        arg(rt, c, 2),
        arg(rt, c, 3) as u16,
        arg(rt, c, 4),
    );
    // (category, setting) -> big-endian bytes
    let v: Option<Vec<u8>> = match (cat, set) {
        // user settings: language English, video flags, audio flags,
        // retail flags, time zone, region
        (3, 9) => Some(1u32.to_be_bytes().to_vec()),
        (3, 10) => Some(0x0004_0000u32.to_be_bytes().to_vec()),
        (3, 11) => Some(0x0001_0001u32.to_be_bytes().to_vec()),
        (3, 1) => Some(0x40u32.to_be_bytes().to_vec()),
        (3, 2) => Some(0u32.to_be_bytes().to_vec()),
        (3, 14) => Some(0u32.to_be_bytes().to_vec()),
        // console: AV region (NTSC-M), game region
        (2, 2) => Some(0x0000_1000u32.to_be_bytes().to_vec()),
        (2, 3) => Some(0x00FFu16.to_be_bytes().to_vec()),
        _ => None,
    };
    let Some(v) = v else {
        crate::log(&format!("[kernel] config setting {cat}.{set} unknown"));
        return ret(c, STATUS_INVALID_PARAMETER);
    };
    if buf != 0 {
        if (size as usize) < v.len() {
            return ret(c, 0xC000_0023); // STATUS_BUFFER_TOO_SMALL
        }
        rt.mem.write_bytes(buf, &v);
    }
    if req != 0 {
        rt.mem.write_u16(req, v.len() as u16);
    }
    ret(c, STATUS_SUCCESS);
}

pub fn ex_register_title_terminate_notification(_: &Runtime, _: &mut Context) {}
pub fn ki_apc_normal_routine_nop(_: &Runtime, _: &mut Context) {}

// ---- hashing ---------------------------------------------------------------

/// SHA-1 state: byte count, five words, 64-byte buffer.
fn sha_update(rt: &Runtime, st: u32, mut data: u32, mut n: u32) {
    let mut count = rt.mem.read_u32(st);
    let mut h: [u32; 5] = std::array::from_fn(|i| rt.mem.read_u32(st + 4 + 4 * i as u32));
    while n > 0 {
        let fill_at = count % 64;
        let take = (64 - fill_at).min(n);
        let mut b = vec![0; take as usize];
        rt.mem.read_bytes(data, &mut b);
        rt.mem.write_bytes(st + 24 + fill_at, &b);
        count += take;
        data += take;
        n -= take;
        if count % 64 == 0 {
            let mut block = [0u8; 64];
            rt.mem.read_bytes(st + 24, &mut block);
            sha1::compress(&mut h, &[block.into()]);
        }
    }
    rt.mem.write_u32(st, count);
    for (i, w) in h.iter().enumerate() {
        rt.mem.write_u32(st + 4 + 4 * i as u32, *w);
    }
}

fn sha_init(rt: &Runtime, st: u32) {
    rt.mem.write_u32(st, 0);
    for (i, w) in [
        0x6745_2301u32,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ]
    .iter()
    .enumerate()
    {
        rt.mem.write_u32(st + 4 + 4 * i as u32, *w);
    }
}

fn sha_final(rt: &Runtime, st: u32, out: u32, out_n: u32) {
    let count = rt.mem.read_u32(st) as u64;
    let mut pad = vec![0x80u8];
    pad.resize(((119 - count % 64) % 64) as usize + 1, 0);
    pad.extend_from_slice(&(count * 8).to_be_bytes());
    let tmp = k().pool_alloc(rt, pad.len() as u32);
    rt.mem.write_bytes(tmp, &pad);
    sha_update(rt, st, tmp, pad.len() as u32);
    k().pool.lock().unwrap().free(tmp);
    for i in 0..out_n.min(20) {
        rt.mem.write_u8(out + i, rt.mem.read_u8(st + 4 + i));
    }
}

pub fn xe_crypt_sha_init(rt: &Runtime, c: &mut Context) {
    sha_init(rt, arg(rt, c, 0));
}
pub fn xe_crypt_sha_update(rt: &Runtime, c: &mut Context) {
    sha_update(rt, arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
}
pub fn xe_crypt_sha_final(rt: &Runtime, c: &mut Context) {
    sha_final(rt, arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
}
/// Hashes up to three buffers in one call.
pub fn xe_crypt_sha(rt: &Runtime, c: &mut Context) {
    let st = k().pool_alloc(rt, 0x58);
    sha_init(rt, st);
    for i in 0..3 {
        let (p, n) = (arg(rt, c, 2 * i), arg(rt, c, 2 * i + 1));
        if p != 0 && n != 0 {
            sha_update(rt, st, p, n);
        }
    }
    sha_final(rt, st, arg(rt, c, 6), arg(rt, c, 7));
    k().pool.lock().unwrap().free(st);
}

/// Console signing needs the console's own key; refuse.
pub fn xe_keys_console_private_key_sign(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
/// Treats data the title signed earlier as valid.
pub fn xe_keys_console_signature_verification(_: &Runtime, c: &mut Context) {
    ret(c, 1);
}

// ---- stops -----------------------------------------------------------------

pub fn ke_bug_check(rt: &Runtime, c: &mut Context) {
    panic!("guest bug check {:#x}", arg(rt, c, 0));
}
pub fn ke_bug_check_ex(rt: &Runtime, c: &mut Context) {
    panic!(
        "guest bug check {:#x} ({:#x}, {:#x}, {:#x}, {:#x})",
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4)
    );
}
pub fn hal_return_to_firmware(rt: &Runtime, c: &mut Context) {
    panic!("guest asked to return to the dashboard ({})", arg(rt, c, 0));
}
/// Exception record: code, flags, nested record, address, parameter count,
/// parameters. Code 0x406D1388 is the debugger's "name this thread"
/// convention, which titles wrap in a handler that just continues: do the
/// same. Anything else stops.
pub fn rtl_raise_exception(rt: &Runtime, c: &mut Context) {
    let r = arg(rt, c, 0);
    let code = rt.mem.read_u32(r);
    if code == 0x406D_1388 {
        // Parameters: type (0x1000), name, thread id, flags.
        let name = cstr(rt, rt.mem.read_u32(r + 0x18));
        crate::log(&format!(
            "[kernel] thread {:#x} named {name:?}",
            rt.mem.read_u32(r + 0x1C)
        ));
        return;
    }
    panic!(
        "guest raised exception {code:#010x} at {:#010x}",
        rt.mem.read_u32(r + 12)
    );
}
pub fn rtl_unwind(_: &Runtime, _: &mut Context) {
    panic!("structured exception unwinding is not supported yet");
}
pub fn c_specific_handler(_: &Runtime, _: &mut Context) {
    panic!("structured exception handlers are not supported yet");
}
/// Fills a CONTEXT record with the caller's registers (integer, LR, CTR, CR).
pub fn rtl_capture_context(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 0);
    for i in 0..32 {
        rt.mem.write_u64(p + 0x80 + 8 * i, c.r[i as usize]);
    }
    rt.mem.write_u64(p + 0x180, c.ctr);
    rt.mem.write_u64(p + 0x188, c.lr);
    rt.mem.write_u32(p + 0x190, c.lr as u32);
    rt.mem.write_u32(p + 0x194, c.cr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_round_trip() {
        // 1601-01-01 is day 0; 2000-02-29 and 1970-01-01 known.
        assert_eq!(civil(0), (1601, 1, 1));
        assert_eq!(days_from_civil(1601, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 1), 134_774);
        for d in [0, 134_774, 145_731, 160_000] {
            let (y, m, dd) = civil(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        assert_eq!(civil(days_from_civil(2000, 2, 29)), (2000, 2, 29));
    }

    #[test]
    fn message_accepts_only_printable_text() {
        assert_eq!(message(b"loaded %s\n\0junk"), Some("loaded %s\n"));
        assert_eq!(message(b"ok\0"), Some("ok"));
        // An object pointer: starts with a vtable address byte.
        assert_eq!(message(b"\x82\x00\x10\x00\0"), None);
        assert_eq!(message(b"x\0"), None);
        assert_eq!(message(b"no terminator"), None);
    }

    #[test]
    fn command_line_adds_switches_after_name() {
        assert_eq!(command_line(None), "\"default.xex\"");
        assert_eq!(command_line(Some("  ")), "\"default.xex\"");
        assert_eq!(command_line(Some(" -a -b=1 ")), "\"default.xex\" -a -b=1");
    }
}
