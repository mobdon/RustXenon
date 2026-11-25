//! Guest threads: creation (stack, processor control region, thread object,
//! thread-local storage), termination, priorities and dynamic TLS slots.
//!
//! r13 holds the thread's processor control region (PCR): +0 its
//! thread-local data, +0x30 itself, +0x70/+0x74 the stack bounds, +0x100 the
//! thread object, +0x10C the processor number. The thread object starts
//! with a dispatcher header of type 6 and holds the stack at +0x5C/+0x60,
//! TLS at +0x68, the thread id at +0x14C and the start address at +0x150.

use super::objects::{self, KObj};
use super::*;
use std::sync::Arc;
use xenon_rt::Event;

pub struct ThreadInfo {
    /// Runtime thread id (for suspend and resume).
    pub id: u32,
    /// Dynamic TLS slots: one word per slot.
    pub tls_dynamic: u32,
    /// Processor control region (r13).
    pub pcr: u32,
}

/// Hardware thread (0..5) for a processor mask: its highest bit, as the
/// console's kernel picks it; 0 for an empty mask.
pub fn cpu_for_mask(mask: u32) -> u8 {
    match mask & 0x3F {
        0 => 0,
        m => (31 - m.leading_zeros()) as u8,
    }
}

/// Records which hardware thread a thread runs on, in its thread object
/// (+0xBF) and control region (+0x10C), where titles read it.
pub fn set_cpu(rt: &Runtime, kthread: u32, pcr: u32, cpu: u8) {
    rt.mem.write_u8(kthread + 0xBF, cpu);
    rt.mem.write_u8(pcr + 0x10C, cpu);
}

/// Size of the guest stack frame padding left above the first frame;
/// some code writes a little past its caller's frame.
const STACK_PAD: u32 = 0x100;

/// Allocates a thread's stack, control region, thread object and TLS, and
/// returns a context ready to run on it.
pub fn new_context(rt: &Runtime, stack_size: u32) -> (Context, u32) {
    let kx = k();
    let size = (stack_size.max(64 << 10) + 0xFFFF) & !0xFFFF;
    let (stack, reused) = kx
        .virt
        .lock()
        .unwrap()
        .alloc(size, 0x1_0000)
        .expect("no memory for a guest stack");
    if reused {
        fill(rt, stack, size, 0);
    }
    let pcr = kx.pool_alloc(rt, 0x2D8);
    let kthread = kx.pool_alloc(rt, 0xAB0);
    let (raw, data_size, raw_size) = kx.tls_data;
    let tls = kx.pool_alloc(rt, data_size + kx.tls_slots * 4);
    if raw_size > 0 {
        let mut b = vec![0; raw_size as usize];
        rt.mem.read_bytes(raw, &mut b);
        rt.mem.write_bytes(tls, &b);
    }
    let top = stack + size;
    if std::env::var_os("XENON_TRACE_STACKS").is_some() {
        crate::log(&format!(
            "[thread] stack {stack:#x}..{top:#x} reused {reused}"
        ));
    }
    rt.mem.write_u32(pcr, tls);
    rt.mem.write_u32(pcr + 0x30, pcr);
    rt.mem.write_u32(pcr + 0x70, top);
    rt.mem.write_u32(pcr + 0x74, stack);
    rt.mem.write_u32(pcr + 0x100, kthread);
    rt.mem.write_u8(kthread, 6);
    rt.mem.write_u32(kthread + 0x5C, top);
    rt.mem.write_u32(kthread + 0x60, stack);
    rt.mem.write_u32(kthread + 0x68, tls);
    let mut c = Context::new();
    c.r[1] = (top - STACK_PAD) as u64;
    c.r[13] = pcr as u64;
    rt.object(kthread, || KObj::Thread(Event::new(true, false)));
    (c, kthread)
}

/// The thread object of the calling guest thread.
pub fn current(rt: &Runtime, c: &Context) -> u32 {
    rt.mem.read_u32(c.r[13] as u32 + 0x100)
}

/// Starts a guest thread at `entry`; returns (handle, thread object).
pub fn create(
    rt: &Runtime,
    stack: u32,
    entry: u32,
    r3: u32,
    r4: u32,
    suspended: bool,
    cpu: u8,
) -> (u32, u32) {
    let kx = k();
    let (mut c, kthread) = new_context(rt, if stack == 0 { kx.default_stack } else { stack });
    set_cpu(rt, kthread, c.r[13] as u32, cpu);
    c.r[3] = r3 as u64;
    c.r[4] = r4 as u64;
    rt.mem.write_u32(kthread + 0x150, entry);
    let tls = rt.mem.read_u32(kthread + 0x68);
    kx.threads.lock().unwrap().insert(
        kthread,
        ThreadInfo {
            id: 0,
            tls_dynamic: tls + kx.tls_data.1,
            pcr: c.r[13] as u32,
        },
    );
    // Started suspended so the id is recorded before the first instruction.
    let id = rt.spawn(entry, c, true);
    rt.mem.write_u32(kthread + 0x14C, id);
    kx.threads.lock().unwrap().get_mut(&kthread).unwrap().id = id;
    if !suspended {
        rt.resume(id);
    }
    let handle = kx.add_handle(Handle::Object(kthread));
    // The thread object is signalled when the thread ends.
    let rt2 = crate::runtime();
    std::thread::Builder::new()
        .name(format!("watch-{id}"))
        .spawn(move || {
            let code = rt2.join(id);
            finished(&rt2, kthread, code as u32);
        })
        .expect("cannot start a host thread");
    (handle, kthread)
}

/// Ended threads, for the memory line of the run log.
pub static ENDED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn finished(rt: &Runtime, kthread: u32, code: u32) {
    rt.mem.write_u32(kthread + 0x140, code);
    rt.mem.write_u32(kthread + 4, 1);
    // The stack and TLS block go back to their heaps: no guest code runs on
    // them again. The thread object and control region stay, since handles
    // and affinity calls can still reach them.
    let kx = k();
    kx.virt
        .lock()
        .unwrap()
        .free(rt.mem.read_u32(kthread + 0x60));
    kx.pool
        .lock()
        .unwrap()
        .free(rt.mem.read_u32(kthread + 0x68));
    ENDED.fetch_add(1, Ordering::Relaxed);
    if let KObj::Thread(e) = &*objects::get(rt, kthread) {
        e.set();
    }
}

fn thread_of(rt: &Runtime, c: &Context, handle: u32) -> Option<u32> {
    if handle == CURRENT_THREAD {
        return Some(current(rt, c));
    }
    match k().handle(handle)? {
        Handle::Object(a) => Some(a),
        _ => None,
    }
}

pub fn ex_create_thread(rt: &Runtime, c: &mut Context) {
    let (handle_out, stack, id_out, startup, start, ctx, flags) = (
        arg(rt, c, 0),
        arg(rt, c, 1),
        arg(rt, c, 2),
        arg(rt, c, 3),
        arg(rt, c, 4),
        arg(rt, c, 5),
        arg(rt, c, 6),
    );
    // With a startup routine the thread runs startup(start, context).
    let (entry, r3, r4) = if startup != 0 {
        (startup, start, ctx)
    } else {
        (start, ctx, 0)
    };
    // The top byte of the flags is the processor mask.
    let (h, kthread) = create(
        rt,
        stack,
        entry,
        r3,
        r4,
        flags & 1 != 0,
        cpu_for_mask(flags >> 24),
    );
    if handle_out != 0 {
        rt.mem.write_u32(handle_out, h);
    }
    if id_out != 0 {
        rt.mem.write_u32(id_out, rt.mem.read_u32(kthread + 0x14C));
    }
    ret(c, STATUS_SUCCESS);
}

pub fn ex_terminate_thread(rt: &Runtime, c: &mut Context) {
    // shortcut: the host thread parks instead of unwinding the guest stack;
    // upgrade when a title ends and recreates many threads.
    finished(rt, current(rt, c), arg(rt, c, 0));
    loop {
        std::thread::park();
    }
}

pub fn nt_resume_thread(rt: &Runtime, c: &mut Context) {
    let (h, out) = (arg(rt, c, 0), arg(rt, c, 1));
    let Some(t) = thread_of(rt, c, h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let prev = resume(rt, t);
    if out != 0 {
        rt.mem.write_u32(out, prev);
    }
    ret(c, STATUS_SUCCESS);
}

fn resume(rt: &Runtime, kthread: u32) -> u32 {
    let id = k().threads.lock().unwrap().get(&kthread).map(|t| t.id);
    id.map_or(0, |id| rt.resume(id))
}

pub fn ke_resume_thread(rt: &Runtime, c: &mut Context) {
    let prev = resume(rt, arg(rt, c, 0));
    ret(c, prev);
}

/// (thread object, processor mask, previous mask out): moves the thread to
/// the mask's processor and answers the previous mask.
pub fn ke_set_affinity_thread(rt: &Runtime, c: &mut Context) {
    let (t, mask, out) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let prev = 1u32 << rt.mem.read_u8(t + 0xBF).min(5);
    if let Some(pcr) = k().threads.lock().unwrap().get(&t).map(|i| i.pcr) {
        set_cpu(rt, t, pcr, cpu_for_mask(mask));
    }
    if out != 0 {
        rt.mem.write_u32(out, prev);
    }
    ret(c, prev);
}

pub fn ke_set_base_priority_thread(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

pub fn ke_set_disable_boost_thread(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}

/// 1 = title process.
pub fn ke_get_current_process_type(_: &Runtime, c: &mut Context) {
    ret(c, 1);
}

/// Converts a kernel timeout (100 ns units; negative is relative, positive
/// absolute system time; null pointer waits forever).
pub fn timeout(rt: &Runtime, p: u32) -> Option<std::time::Duration> {
    if p == 0 {
        return None;
    }
    let v = rt.mem.read_u64(p) as i64;
    let ticks = if v < 0 {
        v.unsigned_abs()
    } else {
        (v as u64).saturating_sub(rtl::system_time())
    };
    Some(std::time::Duration::from_nanos(ticks.saturating_mul(100)))
}

pub fn ke_delay_execution_thread(rt: &Runtime, c: &mut Context) {
    match timeout(rt, arg(rt, c, 2)) {
        Some(d) if !d.is_zero() => std::thread::sleep(d),
        _ => std::thread::yield_now(),
    }
    ret(c, STATUS_SUCCESS);
}

pub fn ke_tls_alloc(_: &Runtime, c: &mut Context) {
    let mut used = k().tls_used.lock().unwrap();
    let n = k().tls_slots as usize;
    used.resize(n, false);
    match used.iter().position(|u| !u) {
        Some(i) => {
            used[i] = true;
            ret(c, i as u32)
        }
        None => ret(c, u32::MAX),
    }
}

pub fn ke_tls_free(rt: &Runtime, c: &mut Context) {
    let i = arg(rt, c, 0) as usize;
    let mut used = k().tls_used.lock().unwrap();
    let ok = used.get(i).copied().unwrap_or(false);
    if ok {
        used[i] = false;
    }
    ret(c, ok as u32);
}

fn slot(rt: &Runtime, c: &Context, i: u32) -> Option<u32> {
    let t = current(rt, c);
    let base = k().threads.lock().unwrap().get(&t).map(|t| t.tls_dynamic)?;
    (i < k().tls_slots).then_some(base + 4 * i)
}

pub fn ke_tls_get_value(rt: &Runtime, c: &mut Context) {
    let v = slot(rt, c, arg(rt, c, 0)).map_or(0, |a| rt.mem.read_u32(a));
    ret(c, v);
}

pub fn ke_tls_set_value(rt: &Runtime, c: &mut Context) {
    let ok = slot(rt, c, arg(rt, c, 0))
        .map(|a| rt.mem.write_u32(a, arg(rt, c, 1)))
        .is_some();
    ret(c, ok as u32);
}

/// Shared by the main thread start in `lib.rs`.
pub fn register_main(rt: &Arc<Runtime>, entry: u32) -> u32 {
    create(rt, k().default_stack, entry, 0, 0, false, 0).1
}
