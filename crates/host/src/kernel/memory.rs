//! Virtual memory, physical memory and pool allocations.

use super::*;

const MEM_RELEASE: u32 = 0x8000;
const MEM_LARGE_PAGES: u32 = 0x2000_0000;

/// Frees of addresses no heap handed out (a double free or a stale
/// pointer), for the run report.
pub static UNKNOWN_FREES: AtomicU32 = AtomicU32::new(0);

/// `XENON_TRACE_ALLOC`: one line per allocate or free import with heap, size,
/// address and caller. Failed allocations and frees of unknown blocks are
/// always logged (the first 20), since both look like something else later.
fn note(c: &Context, what: &str, heap: &str, addr: u32, size: u32, ok: bool) {
    static ON: OnceLock<bool> = OnceLock::new();
    static ODD: AtomicU32 = AtomicU32::new(0);
    let on = *ON.get_or_init(|| std::env::var_os("XENON_TRACE_ALLOC").is_some());
    if !ok && what == "free" {
        UNKNOWN_FREES.fetch_add(1, Ordering::Relaxed);
    }
    if on || (!ok && ODD.fetch_add(1, Ordering::Relaxed) < 20) {
        crate::log(&format!(
            "[alloc] {} {what} {heap} {size:#x} at {addr:#010x} from {:#010x}{}",
            std::thread::current().name().unwrap_or("?"),
            c.lr as u32,
            match (ok, what) {
                (true, _) => "",
                (false, "free") => ": not a live block (double free or stale pointer)",
                _ => ": FAILED",
            }
        ));
    }
}

fn take(rt: &Runtime, h: &Mutex<crate::heap::Heap>, size: u32, align: u32) -> Option<u32> {
    let r = h.lock().unwrap().alloc(size, align);
    let (a, reused) = r?;
    if reused {
        fill(rt, a, size, 0);
    }
    Some(a)
}

pub fn nt_allocate_virtual_memory(rt: &Runtime, c: &mut Context) {
    let (base_p, size_p, ty) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let (want, size) = (rt.mem.read_u32(base_p), rt.mem.read_u32(size_p));
    let kx = k();
    let page = if ty & MEM_LARGE_PAGES != 0 {
        0x1_0000
    } else {
        0x1000
    };
    let size = (size + page - 1) & !(page - 1);
    let got = if want != 0 {
        let mut v = kx.virt.lock().unwrap();
        match v.find(want) {
            // Committing inside an earlier reservation.
            Some((s, n)) if want + size <= s + n => Some(want),
            Some(_) => None,
            None => v.alloc_at(want, size).map(|reused| {
                drop(v);
                if reused {
                    fill(rt, want & !0xFFFF, size, 0);
                }
                want & !0xFFFF
            }),
        }
    } else {
        take(rt, &kx.virt, size, page)
    };
    note(
        c,
        "allocate",
        "virtual",
        got.unwrap_or(want),
        size,
        got.is_some(),
    );
    match got {
        Some(a) => {
            rt.mem.write_u32(base_p, a);
            rt.mem.write_u32(size_p, size);
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_NO_MEMORY),
    }
}

pub fn nt_free_virtual_memory(rt: &Runtime, c: &mut Context) {
    let (base_p, size_p, ty) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let a = rt.mem.read_u32(base_p);
    // Decommitting keeps the range reserved: nothing to do.
    if ty & MEM_RELEASE == 0 {
        return ret(c, STATUS_SUCCESS);
    }
    let freed = k().virt.lock().unwrap().free(a);
    note(c, "free", "virtual", a, freed.unwrap_or(0), freed.is_some());
    match freed {
        Some(n) => {
            if size_p != 0 {
                rt.mem.write_u32(size_p, n);
            }
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_INVALID_PARAMETER),
    }
}

/// Fills MEMORY_BASIC_INFORMATION: base, allocation base, allocation
/// protection, region size, state, protection, type.
pub fn nt_query_virtual_memory(rt: &Runtime, c: &mut Context) {
    let (a, out) = (arg(rt, c, 0), arg(rt, c, 1));
    let found = [&k().virt, &k().phys, &k().pool]
        .iter()
        .find_map(|h| h.lock().unwrap().find(a));
    let (base, size, state, prot) = match found {
        Some((s, n)) => (s, n, 0x1000, 4),
        None => (a & !0xFFF, 0x1000, 0x1_0000, 1),
    };
    for (i, v) in [base, base, prot, size, state, prot, 0x2_0000]
        .into_iter()
        .enumerate()
    {
        rt.mem.write_u32(out + 4 * i as u32, v);
    }
    ret(c, STATUS_SUCCESS);
}

pub fn mm_allocate_physical_memory_ex(rt: &Runtime, c: &mut Context) {
    let (size, align) = (arg(rt, c, 1), arg(rt, c, 5));
    let a = take(rt, &k().phys, size, align.max(0x1000)).unwrap_or(0);
    note(c, "allocate", "physical", a, size, a != 0);
    ret(c, a);
}

pub fn mm_free_physical_memory(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 1);
    let n = k().phys.lock().unwrap().free(a);
    note(c, "free", "physical", a, n.unwrap_or(0), n.is_some());
}

pub fn mm_get_physical_address(rt: &Runtime, c: &mut Context) {
    ret(c, arg(rt, c, 0) & 0x1FFF_FFFF);
}

fn find_any(a: u32) -> Option<(u32, u32)> {
    [&k().virt, &k().phys, &k().pool]
        .iter()
        .find_map(|h| h.lock().unwrap().find(a))
}

pub fn mm_query_address_protect(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    let p = if find_any(a).is_some() || (0x8200_0000..0x9000_0000).contains(&a) {
        4
    } else {
        0
    };
    ret(c, p);
}

pub fn mm_query_allocation_size(rt: &Runtime, c: &mut Context) {
    ret(c, find_any(arg(rt, c, 0)).map_or(0, |x| x.1));
}

pub fn ex_allocate_pool_type_with_tag(rt: &Runtime, c: &mut Context) {
    let size = arg(rt, c, 0);
    let a = take(rt, &k().pool, size, 16).unwrap_or(0);
    note(c, "allocate", "pool", a, size, a != 0);
    ret(c, a);
}

pub fn ex_free_pool(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    let n = k().pool.lock().unwrap().free(a);
    note(c, "free", "pool", a, n.unwrap_or(0), n.is_some());
}

/// A physical allocation for the host's own use (zeroed); panics when
/// physical memory is exhausted.
pub fn physical(rt: &Runtime, size: u32, align: u32) -> u32 {
    take(rt, &k().phys, size, align).expect("physical memory exhausted")
}
