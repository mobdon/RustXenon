//! Kernel dispatcher objects (events, semaphores, mutants, timers, threads),
//! critical sections, spin locks and waits.
//!
//! Every object is keyed by the guest address of its dispatcher header
//! (type byte at +0, signal state at +4, semaphore limit at +0x10); the
//! host object behind it comes from `xenon_rt::sync`. Objects made with
//! `Nt*` calls get a header in the kernel pool and a handle. Critical
//! sections keep the owner fields a title may read: lock count +0x10,
//! recursion +0x14, owning thread +0x18.

use super::threads::{current, timeout};
use super::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use xenon_rt::{CriticalSection, Event, Semaphore};

pub enum KObj {
    Event(Event),
    Semaphore(Semaphore),
    Mutant(CriticalSection),
    Thread(Event),
    Timer(Arc<Event>, AtomicU32),
    Section(CriticalSection),
}

impl KObj {
    fn kind(&self) -> u8 {
        match self {
            KObj::Event(_) => 0,
            KObj::Semaphore(_) => 5,
            KObj::Mutant(_) => 2,
            KObj::Thread(_) => 6,
            KObj::Timer(..) => 8,
            KObj::Section(_) => 0xFF,
        }
    }

    /// Waits; true when acquired.
    pub fn wait(&self, t: Option<Duration>) -> bool {
        match self {
            KObj::Event(e) | KObj::Thread(e) => e.wait(t),
            KObj::Timer(e, _) => e.wait(t),
            KObj::Semaphore(s) => s.wait(t),
            KObj::Mutant(m) | KObj::Section(m) => m.enter(t),
        }
    }
}

fn from_header(rt: &Runtime, a: u32) -> KObj {
    let signal = rt.mem.read_u32(a + 4) as i32;
    match rt.mem.read_u8(a) {
        0 => KObj::Event(Event::new(true, signal != 0)),
        2 => KObj::Mutant(CriticalSection::new()),
        5 => KObj::Semaphore(Semaphore::new(
            signal.max(0) as u32,
            rt.mem.read_u32(a + 0x10).max(signal.max(1) as u32),
        )),
        6 => KObj::Thread(Event::new(true, false)),
        8 => KObj::Timer(Arc::new(Event::new(true, signal != 0)), AtomicU32::new(0)),
        9 => KObj::Timer(Arc::new(Event::new(false, signal != 0)), AtomicU32::new(0)),
        _ => KObj::Event(Event::new(false, signal != 0)),
    }
}

/// The host object for the guest dispatcher header at `a`. A header whose
/// type no longer matches (the memory was reused) gets a fresh object.
pub fn get(rt: &Runtime, a: u32) -> Arc<KObj> {
    let o = rt.object(a, || from_header(rt, a));
    if std::env::var("XENON_TRACE_OBJ")
        .ok()
        .and_then(|v| u32::from_str_radix(&v, 16).ok())
        == Some(a)
    {
        crate::log(&format!(
            "[obj] {} {a:#x}: host {:p} kind {} header {:02x} {:08x}",
            super::current_import(),
            Arc::as_ptr(&o),
            o.kind(),
            rt.mem.read_u8(a),
            rt.mem.read_u32(a + 4)
        ));
    }
    let ty = match rt.mem.read_u8(a) {
        1 => 0,
        9 => 8,
        t => t,
    };
    if o.kind() != ty && !matches!(*o, KObj::Section(_)) && matches!(ty, 0 | 2 | 5 | 6 | 8) {
        rt.drop_object(a);
        return rt.object(a, || from_header(rt, a));
    }
    o
}

/// A new object with a pool header and a handle.
fn create(rt: &Runtime, ty: u8, signal: u32, limit: u32, out: u32) -> u32 {
    let a = k().pool_alloc(rt, 0x20);
    rt.mem.write_u8(a, ty);
    rt.mem.write_u32(a + 4, signal);
    rt.mem.write_u32(a + 0x10, limit);
    get(rt, a);
    let h = k().add_handle(Handle::Object(a));
    if out != 0 {
        rt.mem.write_u32(out, h);
    }
    h
}

/// The dispatcher header behind a handle.
pub fn object_of(rt: &Runtime, c: &Context, h: u32) -> Option<u32> {
    if h == CURRENT_THREAD {
        return Some(current(rt, c));
    }
    match k().handle(h)? {
        Handle::Object(a) => Some(a),
        _ => None,
    }
}

pub fn nt_create_event(rt: &Runtime, c: &mut Context) {
    // Event type 0 is notification (manual reset), 1 synchronization.
    create(rt, arg(rt, c, 2) as u8, arg(rt, c, 3), 0, arg(rt, c, 0));
    ret(c, STATUS_SUCCESS);
}

pub fn nt_create_semaphore(rt: &Runtime, c: &mut Context) {
    create(rt, 5, arg(rt, c, 2), arg(rt, c, 3), arg(rt, c, 0));
    ret(c, STATUS_SUCCESS);
}

pub fn nt_create_mutant(rt: &Runtime, c: &mut Context) {
    let h = create(rt, 2, 1, 0, arg(rt, c, 0));
    if arg(rt, c, 2) != 0 {
        if let Some(a) = object_of(rt, c, h) {
            get(rt, a).wait(None);
        }
    }
    ret(c, STATUS_SUCCESS);
}

pub fn nt_create_timer(rt: &Runtime, c: &mut Context) {
    // Timer type 0 is notification, 1 synchronization.
    create(rt, 8 + arg(rt, c, 2).min(1) as u8, 0, 0, arg(rt, c, 0));
    ret(c, STATUS_SUCCESS);
}

fn event_op(rt: &Runtime, c: &mut Context, a: Option<u32>, set: bool) -> Option<bool> {
    let _ = c;
    let o = get(rt, a?);
    let r = match &*o {
        KObj::Event(e) => Some(if set { e.set() } else { e.reset() }),
        _ => None,
    };
    if r.is_none() {
        crate::log(&format!(
            "[kernel] {} on {:#x}, which is not an event (type {})",
            super::current_import(),
            a?,
            rt.mem.read_u8(a?)
        ));
    }
    r
}

pub fn ke_set_event(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    let was = event_op(rt, c, Some(a), true).unwrap_or(false);
    ret(c, was as u32);
}

pub fn ke_reset_event(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    let was = event_op(rt, c, Some(a), false).unwrap_or(false);
    ret(c, was as u32);
}

pub fn nt_set_event(rt: &Runtime, c: &mut Context) {
    let (h, prev) = (arg(rt, c, 0), arg(rt, c, 1));
    let a = object_of(rt, c, h);
    match event_op(rt, c, a, true) {
        Some(was) => {
            if prev != 0 {
                rt.mem.write_u32(prev, was as u32);
            }
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_INVALID_HANDLE),
    }
}

pub fn nt_clear_event(rt: &Runtime, c: &mut Context) {
    let a = object_of(rt, c, arg(rt, c, 0));
    let s = event_op(rt, c, a, false).map_or(STATUS_INVALID_HANDLE, |_| STATUS_SUCCESS);
    ret(c, s);
}

pub fn ke_initialize_semaphore(rt: &Runtime, c: &mut Context) {
    let (a, count, limit) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    rt.mem.write_u8(a, 5);
    rt.mem.write_u32(a + 4, count);
    rt.mem.write_u32(a + 0x10, limit);
    rt.drop_object(a);
    get(rt, a);
}

fn release(rt: &Runtime, a: u32, n: u32) -> Option<u32> {
    match &*get(rt, a) {
        KObj::Semaphore(s) => s.release(n),
        _ => None,
    }
}

pub fn ke_release_semaphore(rt: &Runtime, c: &mut Context) {
    let prev = release(rt, arg(rt, c, 0), arg(rt, c, 2));
    ret(c, prev.unwrap_or(0));
}

pub fn nt_release_semaphore(rt: &Runtime, c: &mut Context) {
    let (h, n, out) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let Some(a) = object_of(rt, c, h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    match release(rt, a, n) {
        Some(p) => {
            if out != 0 {
                rt.mem.write_u32(out, p);
            }
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_SEMAPHORE_LIMIT_EXCEEDED),
    }
}

pub fn nt_release_mutant(rt: &Runtime, c: &mut Context) {
    let Some(a) = object_of(rt, c, arg(rt, c, 0)) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    match &*get(rt, a) {
        // The console returns this status when the caller does not own it.
        KObj::Mutant(m) if m.try_leave() => ret(c, STATUS_SUCCESS),
        _ => ret(c, STATUS_MUTANT_NOT_OWNED),
    }
}

pub fn nt_set_timer_ex(rt: &Runtime, c: &mut Context) {
    let (h, due, period) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 6));
    let Some(a) = object_of(rt, c, h) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let o = get(rt, a);
    let KObj::Timer(e, gen) = &*o else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    if arg(rt, c, 2) != 0 {
        eprintln!("[kernel] timer completion routines are not run");
    }
    let mine = gen.fetch_add(1, Ordering::SeqCst) + 1;
    e.reset();
    let first = timeout(rt, due).unwrap_or_default();
    let (e, o2) = (e.clone(), o.clone());
    std::thread::spawn(move || {
        let mut next = Instant::now() + first;
        loop {
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
            let KObj::Timer(_, g) = &*o2 else { return };
            if g.load(Ordering::SeqCst) != mine {
                return;
            }
            e.set();
            if period == 0 {
                return;
            }
            next += Duration::from_millis(period as u64);
        }
    });
    ret(c, STATUS_SUCCESS);
}

pub fn nt_cancel_timer(rt: &Runtime, c: &mut Context) {
    let Some(a) = object_of(rt, c, arg(rt, c, 0)) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    if let KObj::Timer(e, g) = &*get(rt, a) {
        g.fetch_add(1, Ordering::SeqCst);
        e.reset();
    }
    ret(c, STATUS_SUCCESS);
}

/// What a stall report says about the guest object at `a`: its kind and
/// state, and for a lock the thread holding it.
pub fn describe(rt: &Runtime, a: u32) -> String {
    let held = |m: &CriticalSection| match m.owner() {
        Some(id) => format!("held by {}", crate::thread_name(id)),
        None => "free".into(),
    };
    let set = |e: &Event| if e.is_set() { "set" } else { "not set" };
    match rt.find_object::<KObj>(a).as_deref() {
        None => "no host object".into(),
        Some(KObj::Event(e)) => format!("event, {}", set(e)),
        Some(KObj::Thread(e)) => {
            format!("thread, {}", if e.is_set() { "ended" } else { "running" })
        }
        Some(KObj::Timer(e, _)) => format!("timer, {}", set(e)),
        Some(KObj::Semaphore(s)) => format!("semaphore, count {}", s.count()),
        Some(KObj::Mutant(m)) => format!("mutant, {}", held(m)),
        Some(KObj::Section(m)) => format!("critical section, {}", held(m)),
    }
}

/// Waits for one object: STATUS_SUCCESS or STATUS_TIMEOUT.
fn wait_one(rt: &Runtime, a: u32, t: Option<Duration>) -> u32 {
    let o = get(rt, a);
    let ok = {
        let _w = crate::waiting(a, 1);
        o.wait(t)
    };
    if std::env::var("XENON_TRACE_OBJ")
        .ok()
        .and_then(|v| u32::from_str_radix(&v, 16).ok())
        == Some(a)
    {
        crate::log(&format!("[obj] wait on {a:#x} returned {ok}"));
    }
    if ok {
        STATUS_SUCCESS
    } else {
        STATUS_TIMEOUT
    }
}

/// Waits for any (wait_type 1) or all (0) of `objs`.
fn wait_many(rt: &Runtime, objs: &[u32], all: bool, t: Option<Duration>) -> u32 {
    let deadline = t.map(|t| Instant::now() + t);
    if all {
        // shortcut: acquires one object after another, not atomically;
        // upgrade if a title waits for all of several contended objects.
        for &a in objs {
            let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            if wait_one(rt, a, left) != STATUS_SUCCESS {
                return STATUS_TIMEOUT;
            }
        }
        return STATUS_SUCCESS;
    }
    let os: Vec<_> = objs.iter().map(|&a| get(rt, a)).collect();
    let _w = crate::waiting(objs.first().copied().unwrap_or(0), objs.len() as u32);
    let mut nap = Duration::from_micros(50);
    loop {
        for (i, o) in os.iter().enumerate() {
            if o.wait(Some(Duration::ZERO)) {
                return i as u32;
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return STATUS_TIMEOUT;
        }
        std::thread::sleep(nap);
        nap = (nap * 2).min(Duration::from_millis(1));
    }
}

pub fn ke_wait_for_single_object(rt: &Runtime, c: &mut Context) {
    let s = wait_one(rt, arg(rt, c, 0), timeout(rt, arg(rt, c, 4)));
    ret(c, s);
}

pub fn nt_wait_for_single_object_ex(rt: &Runtime, c: &mut Context) {
    let Some(a) = object_of(rt, c, arg(rt, c, 0)) else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let s = wait_one(rt, a, timeout(rt, arg(rt, c, 3)));
    ret(c, s);
}

pub fn ke_wait_for_multiple_objects(rt: &Runtime, c: &mut Context) {
    let (n, p, ty) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let objs: Vec<u32> = (0..n).map(|i| rt.mem.read_u32(p + 4 * i)).collect();
    let s = wait_many(rt, &objs, ty == 0, timeout(rt, arg(rt, c, 6)));
    ret(c, s);
}

pub fn nt_wait_for_multiple_objects_ex(rt: &Runtime, c: &mut Context) {
    let (n, p, ty) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let mut objs = Vec::new();
    for i in 0..n {
        match object_of(rt, c, rt.mem.read_u32(p + 4 * i)) {
            Some(a) => objs.push(a),
            None => return ret(c, STATUS_INVALID_HANDLE),
        }
    }
    let s = wait_many(rt, &objs, ty == 0, timeout(rt, arg(rt, c, 5)));
    ret(c, s);
}

pub fn nt_close(rt: &Runtime, c: &mut Context) {
    let s = match k().close(arg(rt, c, 0)) {
        Some(Handle::Object(a)) => {
            // A timer's last handle closed: its periodic thread stops, as
            // the console deletes the timer.
            if !k().referenced(a) {
                if let KObj::Timer(_, g) = &*get(rt, a) {
                    g.fetch_add(1, Ordering::SeqCst);
                }
            }
            STATUS_SUCCESS
        }
        Some(_) => STATUS_SUCCESS,
        None => STATUS_INVALID_HANDLE,
    };
    ret(c, s);
}

pub fn nt_duplicate_object(rt: &Runtime, c: &mut Context) {
    let (h, out, options) = (arg(rt, c, 0), arg(rt, c, 1), arg(rt, c, 2));
    let target = if h == CURRENT_THREAD {
        Some(Handle::Object(current(rt, c)))
    } else {
        k().handle(h)
    };
    let Some(t) = target else {
        return ret(c, STATUS_INVALID_HANDLE);
    };
    let n = k().add_handle(t);
    if out != 0 {
        rt.mem.write_u32(out, n);
    }
    // DUPLICATE_CLOSE_SOURCE
    if options & 1 != 0 {
        k().close(h);
    }
    ret(c, STATUS_SUCCESS);
}

pub fn ob_reference_object_by_handle(rt: &Runtime, c: &mut Context) {
    let (h, out) = (arg(rt, c, 0), arg(rt, c, 2));
    let a = match k().handle(h) {
        _ if h == CURRENT_THREAD => Some(current(rt, c)),
        Some(Handle::Object(a)) => Some(a),
        Some(_) => Some(k().pool_alloc(rt, 0x20)),
        None => None,
    };
    match a {
        Some(a) => {
            if out != 0 {
                rt.mem.write_u32(out, a);
            }
            ret(c, STATUS_SUCCESS)
        }
        None => ret(c, STATUS_INVALID_HANDLE),
    }
}

pub fn ob_reference_object(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn ob_dereference_object(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn ob_is_title_object(_: &Runtime, c: &mut Context) {
    ret(c, 1);
}

// ---- critical sections -----------------------------------------------------

fn section(rt: &Runtime, a: u32) -> Arc<KObj> {
    let o = rt.object(a, || KObj::Section(CriticalSection::new()));
    if matches!(*o, KObj::Section(_)) {
        return o;
    }
    rt.drop_object(a);
    rt.object(a, || KObj::Section(CriticalSection::new()))
}

fn enter(rt: &Runtime, c: &Context, a: u32, t: Option<Duration>) -> bool {
    let KObj::Section(cs) = &*section(rt, a) else {
        unreachable!()
    };
    let ok = if t == Some(Duration::ZERO) {
        cs.enter(t)
    } else {
        let _w = crate::waiting(a, 1);
        cs.enter(t)
    };
    if !ok {
        return false;
    }
    rt.mem
        .write_u32(a + 0x10, rt.mem.read_u32(a + 0x10).wrapping_add(1));
    rt.mem
        .write_u32(a + 0x14, rt.mem.read_u32(a + 0x14).wrapping_add(1));
    rt.mem.write_u32(a + 0x18, current(rt, c));
    true
}

pub fn rtl_initialize_critical_section(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    rt.mem.write_u8(a, 1);
    rt.mem.write_u32(a + 0x10, u32::MAX);
    rt.mem.write_u32(a + 0x14, 0);
    rt.mem.write_u32(a + 0x18, 0);
    rt.drop_object(a);
    section(rt, a);
}

pub fn rtl_enter_critical_section(rt: &Runtime, c: &mut Context) {
    enter(rt, c, arg(rt, c, 0), None);
}

pub fn rtl_try_enter_critical_section(rt: &Runtime, c: &mut Context) {
    let ok = enter(rt, c, arg(rt, c, 0), Some(Duration::ZERO));
    ret(c, ok as u32);
}

pub fn rtl_leave_critical_section(rt: &Runtime, c: &mut Context) {
    let a = arg(rt, c, 0);
    let rec = rt.mem.read_u32(a + 0x14).wrapping_sub(1);
    rt.mem.write_u32(a + 0x14, rec);
    if rec == 0 {
        rt.mem.write_u32(a + 0x18, 0);
    }
    rt.mem
        .write_u32(a + 0x10, rt.mem.read_u32(a + 0x10).wrapping_sub(1));
    if let KObj::Section(cs) = &*section(rt, a) {
        cs.leave();
    }
}

// ---- spin locks and interrupt levels -----------------------------------------

/// Spin locks are a guest word holding the owner (its control region).
fn spin(rt: &Runtime, c: &Context, a: u32, once: bool) -> bool {
    let me = c.r[13] as u32;
    loop {
        if rt.mem.cas_u32(a, 0, me) {
            return true;
        }
        if once {
            return false;
        }
        std::hint::spin_loop();
        std::thread::yield_now();
    }
}

pub fn kf_acquire_spin_lock(rt: &Runtime, c: &mut Context) {
    spin(rt, c, arg(rt, c, 0), false);
    ret(c, 0);
}
pub fn ke_acquire_spin_lock_at_raised_irql(rt: &Runtime, c: &mut Context) {
    spin(rt, c, arg(rt, c, 0), false);
}
pub fn ke_try_to_acquire_spin_lock_at_raised_irql(rt: &Runtime, c: &mut Context) {
    let ok = spin(rt, c, arg(rt, c, 0), true);
    ret(c, ok as u32);
}
pub fn kf_release_spin_lock(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 0), 0);
}
pub fn ke_release_spin_lock_from_raised_irql(rt: &Runtime, c: &mut Context) {
    rt.mem.write_u32(arg(rt, c, 0), 0);
}
pub fn ke_raise_irql_to_dpc_level(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn kf_lower_irql(_: &Runtime, _: &mut Context) {}
pub fn ke_enter_critical_region(_: &Runtime, _: &mut Context) {}
pub fn ke_leave_critical_region(_: &Runtime, _: &mut Context) {}
pub fn ke_lock_l2(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn ke_unlock_l2(_: &Runtime, _: &mut Context) {}
