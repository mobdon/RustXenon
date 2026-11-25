//! System-call replacements, checked with synthetic guest state: no game
//! bytes. One kernel per process, shared by every test here (tests that
//! need their own guest scratch memory use separate addresses).

use std::sync::{Arc, OnceLock};
use xenon_host::kernel::{self, files, gpu::Gpu, memory, objects, rtl, threads, Setup};
use xenon_recomp::xex::Image;
use xenon_rt::{Context, GuestMemory, Runtime};

const STATUS_TIMEOUT: u64 = 0x102;

/// A minimal executable header: magic, sizes, one optional header (the
/// default stack size).
fn header() -> Vec<u8> {
    let mut h = vec![0u8; 0x200];
    h[..4].copy_from_slice(b"XEX2");
    h[8..12].copy_from_slice(&0x200u32.to_be_bytes());
    h[0x10..0x14].copy_from_slice(&0x100u32.to_be_bytes());
    h[0x14..0x18].copy_from_slice(&1u32.to_be_bytes());
    h[0x18..0x1C].copy_from_slice(&0x0002_0200u32.to_be_bytes());
    h[0x1C..0x20].copy_from_slice(&0x4_0000u32.to_be_bytes());
    h
}

/// Guest entry point used by the thread test: sets the event whose handle
/// is in r3, uses a TLS slot, and exits with 7.
const THREAD_ENTRY: u32 = 0x8200_0000;
fn thread_entry(rt: &Runtime, c: &mut Context) {
    let event = c.r[3];
    threads::ke_tls_alloc(rt, c);
    let slot = c.r[3];
    c.r[4] = 0x1234;
    threads::ke_tls_set_value(rt, c);
    c.r[3] = slot;
    threads::ke_tls_get_value(rt, c);
    rt.mem.write_u32(0x5000, c.r[3] as u32);
    c.r[3] = event;
    c.r[4] = 0;
    objects::nt_set_event(rt, c);
    c.r[3] = 7;
}

struct Env {
    rt: Arc<Runtime>,
    game: std::path::PathBuf,
}

fn env() -> &'static Env {
    static E: OnceLock<Env> = OnceLock::new();
    E.get_or_init(|| {
        let root = std::env::temp_dir().join(format!("xenon-kernel-{}", std::process::id()));
        let game = root.join("game");
        std::fs::create_dir_all(game.join("dir")).unwrap();
        std::fs::write(game.join("a.txt"), b"hello").unwrap();
        std::fs::write(game.join("dir").join("b.bin"), [0u8; 10]).unwrap();
        let file = header();
        let image = Image {
            base: 0x8200_0000,
            entry: THREAD_ENTRY,
            title_id: None,
            version: None,
            data: vec![0; 0x1000],
            sections: vec![],
            imports: vec![],
            exception_table: vec![],
        };
        let mut rt = Runtime::new(GuestMemory::new(xenon_host::GUEST_MEMORY));
        rt.set_override(THREAD_ENTRY, thread_entry);
        rt.set_override(QUICK_ENTRY, quick_entry);
        kernel::install(
            &mut rt,
            &Setup {
                file: &file,
                image: &image,
                game_dir: game.clone(),
                data_dir: root.join("data"),
                update_dir: None,
            },
        );
        let rt = rt.share();
        xenon_host::set_runtime(rt.clone());
        Env { rt, game }
    })
}

/// A context with a guest stack of its own at `stack`.
fn ctx(stack: u32) -> Context {
    let mut c = Context::new();
    c.r[1] = stack as u64;
    c
}

fn call(rt: &Runtime, c: &mut Context, f: fn(&Runtime, &mut Context), args: &[u64]) -> u64 {
    for (i, a) in args.iter().enumerate() {
        c.r[3 + i] = *a;
    }
    f(rt, c);
    c.r[3]
}

#[test]
fn failure_statuses_are_negative_in_the_full_register() {
    let e = env();
    let mut c = ctx(0x1_8000);
    // NtClose of a handle that does not exist: STATUS_INVALID_HANDLE.
    let r = call(&e.rt, &mut c, objects::nt_close, &[0xDEAD]);
    assert_eq!(r, 0xFFFF_FFFF_C000_0008);
    assert!((r as i64) < 0);
}

#[test]
fn virtual_and_physical_memory_with_headroom() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_9000);
    let (base, size) = (0x1_0000, 0x1_0004);
    rt.mem.write_u32(base, 0);
    rt.mem.write_u32(size, 0x2_0000);
    assert_eq!(
        call(
            rt,
            &mut c,
            memory::nt_allocate_virtual_memory,
            &[base as u64, size as u64, 0x3000, 4]
        ),
        0
    );
    let a = rt.mem.read_u32(base);
    assert!((0x4000_0000..0x7F00_0000).contains(&a));
    assert_eq!(
        call(rt, &mut c, memory::mm_query_allocation_size, &[a as u64]),
        0x2_0000
    );
    // Freed memory comes back zeroed.
    rt.mem.write_u32(a, 0xFFFF_FFFF);
    assert_eq!(
        call(
            rt,
            &mut c,
            memory::nt_free_virtual_memory,
            &[base as u64, size as u64, 0x8000]
        ),
        0
    );
    rt.mem.write_u32(base, 0);
    rt.mem.write_u32(size, 0x2_0000);
    call(
        rt,
        &mut c,
        memory::nt_allocate_virtual_memory,
        &[base as u64, size as u64, 0x3000, 4],
    );
    assert_eq!(rt.mem.read_u32(rt.mem.read_u32(base)), 0);
    // Headroom: asking for more than the region holds fails cleanly.
    rt.mem.write_u32(base, 0);
    rt.mem.write_u32(size, 0x7FFF_0000);
    let r = call(
        rt,
        &mut c,
        memory::nt_allocate_virtual_memory,
        &[base as u64, size as u64, 0x3000, 4],
    );
    assert_eq!(r as u32, 0xC000_0017);
    // Physical memory honours alignment and sits in the 0xA0000000 view.
    let p = call(
        rt,
        &mut c,
        memory::mm_allocate_physical_memory_ex,
        &[0, 0x3000, 4, 0, u32::MAX as u64, 0x1_0000],
    ) as u32;
    assert_eq!(p & 0xFFFF, 0);
    assert_eq!(p >> 29, 5);
    assert_eq!(
        call(rt, &mut c, memory::mm_get_physical_address, &[p as u64]) as u32,
        p & 0x1FFF_FFFF
    );
}

#[test]
fn threads_events_and_tls() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_A000);
    // Synchronization event, not signalled.
    call(rt, &mut c, objects::nt_create_event, &[0x6000, 0, 1, 0]);
    let ev = rt.mem.read_u32(0x6000) as u64;
    // A 1 ms relative timeout runs out.
    rt.mem.write_u64(0x6010, (-10_000i64) as u64);
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_single_object_ex,
            &[ev, 1, 0, 0x6010]
        ),
        STATUS_TIMEOUT
    );
    // ExCreateThread(handle out, stack, id out, startup, start, context, flags)
    let r = call(
        rt,
        &mut c,
        threads::ex_create_thread,
        &[0x6020, 0, 0x6024, 0, THREAD_ENTRY as u64, ev, 0],
    );
    assert_eq!(r, 0);
    let th = rt.mem.read_u32(0x6020) as u64;
    rt.mem.write_u64(0x6010, (-50_000_000i64) as u64); // 5 s
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_single_object_ex,
            &[ev, 1, 0, 0x6010]
        ),
        0
    );
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_single_object_ex,
            &[th, 1, 0, 0x6010]
        ),
        0
    );
    assert_eq!(
        rt.mem.read_u32(0x5000),
        0x1234,
        "TLS value read back in the thread"
    );
}

#[test]
fn semaphores_respect_their_limit_and_wait_any() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_B000);
    call(rt, &mut c, objects::nt_create_semaphore, &[0x7000, 0, 0, 1]);
    call(rt, &mut c, objects::nt_create_semaphore, &[0x7004, 0, 1, 1]);
    let (a, b) = (
        rt.mem.read_u32(0x7000) as u64,
        rt.mem.read_u32(0x7004) as u64,
    );
    assert_eq!(
        call(rt, &mut c, objects::nt_release_semaphore, &[b, 1, 0]) as u32,
        0xC000_0047
    );
    // Wait for any of (a, b): b is signalled, so index 1.
    rt.mem.write_u32(0x7010, a as u32);
    rt.mem.write_u32(0x7014, b as u32);
    rt.mem.write_u64(0x7020, (-10_000_000i64) as u64);
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_multiple_objects_ex,
            &[2, 0x7010, 1, 1, 0, 0x7020]
        ),
        1
    );
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_multiple_objects_ex,
            &[2, 0x7010, 1, 1, 0, 0x7020]
        ),
        STATUS_TIMEOUT
    );
}

#[test]
fn critical_sections_are_recursive_and_record_the_owner() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_C000);
    let (_, kthread) = threads::new_context(rt, 64 << 10);
    // The owner is the thread object the control region (r13) names.
    c.r[13] = 0x8000;
    rt.mem.write_u32(0x8000 + 0x100, kthread);
    let cs = 0x8400;
    call(rt, &mut c, objects::rtl_initialize_critical_section, &[cs]);
    call(rt, &mut c, objects::rtl_enter_critical_section, &[cs]);
    assert_eq!(
        call(rt, &mut c, objects::rtl_try_enter_critical_section, &[cs]),
        1
    );
    assert_eq!(rt.mem.read_u32(cs as u32 + 0x14), 2);
    assert_eq!(rt.mem.read_u32(cs as u32 + 0x18), kthread);
    call(rt, &mut c, objects::rtl_leave_critical_section, &[cs]);
    call(rt, &mut c, objects::rtl_leave_critical_section, &[cs]);
    assert_eq!(rt.mem.read_u32(cs as u32 + 0x18), 0);
}

/// Writes an ANSI_STRING and OBJECT_ATTRIBUTES for `name` at `at`.
fn attrs(rt: &Runtime, at: u32, name: &str) -> u64 {
    rt.mem.write_bytes(at + 0x40, name.as_bytes());
    rt.mem.write_u16(at, name.len() as u16);
    rt.mem.write_u16(at + 2, name.len() as u16);
    rt.mem.write_u32(at + 4, at + 0x40);
    rt.mem.write_u32(at + 0x10, 0);
    rt.mem.write_u32(at + 0x14, at);
    rt.mem.write_u32(at + 0x18, 0);
    (at + 0x10) as u64
}

#[test]
fn files_on_the_game_folder() {
    let e = env();
    let rt = &e.rt;
    assert!(e.game.join("a.txt").exists());
    let mut c = ctx(0x2_0000);
    rt.mem.write_u32(0x2_0054, 0); // create options (ninth argument, on the stack)
    let a = attrs(rt, 0x9000, "game:\\a.txt");
    // NtCreateFile(out, access, attrs, iosb, allocation, attributes, share, disposition = open)
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_create_file,
            &[0x9100, 0x8000_0000, a, 0x9110, 0, 0, 1, 1]
        ),
        0
    );
    let h = rt.mem.read_u32(0x9100) as u64;
    // NtReadFile(handle, event, apc, apc context, iosb, buffer, length, offset)
    rt.mem.write_u64(0x9120, 1);
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_read_file,
            &[h, 0, 0, 0, 0x9110, 0x9200, 16, 0x9120]
        ),
        0
    );
    assert_eq!(rt.mem.read_u32(0x9114), 4);
    let mut b = [0u8; 4];
    rt.mem.read_bytes(0x9200, &mut b);
    assert_eq!(&b, b"ello");
    // Past the end.
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_read_file,
            &[h, 0, 0, 0, 0x9110, 0x9200, 16, 0]
        ) as u32,
        0xC000_0011
    );
    // Standard information: end of file at +8.
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_query_information_file,
            &[h, 0x9110, 0x9300, 24, 5]
        ),
        0
    );
    assert_eq!(rt.mem.read_u64(0x9308), 5);
    // The game folder is read only; a missing file is "not found".
    let w = attrs(rt, 0x9400, "game:\\new.txt");
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_create_file,
            &[0x9100, 0x4000_0000, w, 0x9110, 0, 0, 0, 2]
        ) as u32,
        0xC000_0022
    );
    let m = attrs(rt, 0x9500, "d:\\missing.txt");
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_create_file,
            &[0x9100, 0x8000_0000, m, 0x9110, 0, 0, 1, 1]
        ) as u32,
        0xC000_0034
    );
    // Directory listing with a pattern.
    let d = attrs(rt, 0x9600, "game:\\dir");
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_open_file,
            &[0x9100, 0x8000_0000, d, 0x9110, 1]
        ),
        0
    );
    let dh = rt.mem.read_u32(0x9100) as u64;
    let pat = attrs(rt, 0x9700, "*.bin") - 0x10;
    rt.mem.write_u32(0x2_0054, 1); // restart scan (ninth argument)
    assert_eq!(
        call(
            rt,
            &mut c,
            files::nt_query_directory_file,
            &[dh, 0, 0, 0, 0x9110, 0x9800, 0x100, pat]
        ),
        0
    );
    assert_eq!(rt.mem.read_u32(0x9800 + 60), 5);
    assert_eq!(rt.mem.read_u64(0x9800 + 40), 10);
}

#[test]
fn printf_formatting() {
    let e = env();
    let rt = &e.rt;
    let c = ctx(0x2_1000);
    rt.mem.write_bytes(0xA000, b"str\0");
    for (i, v) in [42u64, (-3i64) as u64, 0xBEEF, 0xA000, 1.5f64.to_bits()]
        .into_iter()
        .enumerate()
    {
        rt.mem.write_u64(0xA100 + 8 * i as u32, v);
    }
    let s = rtl::format(rt, &c, "%d|%5d|%08X|%s|%.2f|%%", rtl::Args::List(0xA100));
    assert_eq!(s, "42|   -3|0000BEEF|str|1.50|%");
}

#[test]
fn command_processor_applies_writes_waits_and_interrupts() {
    let e = env();
    let rt = &e.rt;
    // A 64-word ring at physical 0x1000.
    let ring = 0x1000;
    let type3 = |op: u32, n: u32| 0xC000_0000 | (n - 1) << 16 | op << 8;
    let words = [
        // MEM_WRITE two words at 0x100, guest byte order (endian 2).
        type3(0x3D, 3),
        0x100 | 2,
        0x1111,
        0x2222,
        // EVENT_WRITE_SHD of 9 at 0x200.
        type3(0x58, 3),
        3,
        0x200 | 2,
        9,
        // WAIT_REG_MEM until the word at 0x300 equals 1.
        type3(0x3C, 5),
        0x13,
        0x300 | 2,
        1,
        0xFFFF_FFFF,
        0x100,
        // INTERRUPT for processor 2.
        type3(0x54, 1),
        4,
    ];
    for (i, w) in words.iter().enumerate() {
        rt.mem.write_u32(0xA000_0000 + ring + 4 * i as u32, *w);
    }
    let mut gpu = Gpu::default();
    // The wait releases once the test writes the word it polls.
    let mut waited = 0;
    let mut idle = |_: &mut Gpu| {
        waited += 1;
        if waited == 3 {
            rt.mem.write_u32(0xA000_0300, 1);
        }
    };
    let r = gpu.run_ring(rt, ring, 64, 0, words.len() as u32, &mut idle);
    assert_eq!(r, words.len() as u32);
    assert_eq!(
        (rt.mem.read_u32(0xA000_0100), rt.mem.read_u32(0xA000_0104)),
        (0x1111, 0x2222)
    );
    assert_eq!(rt.mem.read_u32(0xA000_0200), 9);
    assert!(waited >= 3, "the wait blocked until its word was written");
    assert_eq!(gpu.interrupts, vec![4]);
}

#[test]
fn signed_in_profile_and_settings() {
    use xenon_host::kernel::profile;
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x2_2000);
    assert_eq!(
        call(rt, &mut c, profile::xam_user_get_signin_state, &[0]),
        1
    );
    assert_eq!(
        call(rt, &mut c, profile::xam_user_get_signin_state, &[1]),
        0
    );
    assert_eq!(
        call(rt, &mut c, profile::xam_user_get_xuid, &[0, 7, 0xB000]),
        0
    );
    assert_eq!(rt.mem.read_u64(0xB000), profile::XUID);
    // Two settings: an integer with a non-zero default, and 1000 bytes of
    // title data that was never written.
    rt.mem.write_u32(0xB010, 0x1004_0003);
    rt.mem.write_u32(0xB014, 0x63E8_3FFF);
    rt.mem.write_u32(0xB020, 0);
    rt.mem.write_u32(0x2_2054, 0); // no request block (ninth argument)
    let args = [0, 0, 0, 0, 2, 0xB010, 0xB020, 0xB100];
    assert_eq!(
        call(rt, &mut c, profile::xam_user_read_profile_settings, &args),
        122
    );
    let need = rt.mem.read_u32(0xB020);
    assert_eq!(need, 8 + 2 * 40 + 0x3E8);
    assert_eq!(
        call(rt, &mut c, profile::xam_user_read_profile_settings, &args),
        0
    );
    assert_eq!(rt.mem.read_u32(0xB100), 2);
    let rec = rt.mem.read_u32(0xB104);
    assert_eq!(
        (
            rt.mem.read_u32(rec),
            rt.mem.read_u32(rec + 0x10),
            rt.mem.read_u32(rec + 0x20)
        ),
        (1, 0x1004_0003, 3)
    );
    assert_eq!(
        rt.mem.read_u32(rec + 40),
        0,
        "unwritten title data is unset"
    );
}

#[test]
fn startup_notifications_then_none() {
    use xenon_host::kernel::xam;
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x2_3000);
    let h = call(rt, &mut c, xam::xam_notify_create_listener, &[1, 1]);
    // Sign-in changed is waiting; asking for it by id takes it out of turn.
    assert_eq!(
        call(
            rt,
            &mut c,
            xam::x_notify_get_next,
            &[h, 0xA, 0xC000, 0xC004]
        ),
        1
    );
    assert_eq!((rt.mem.read_u32(0xC000), rt.mem.read_u32(0xC004)), (0xA, 1));
    let mut n = 0;
    while call(rt, &mut c, xam::x_notify_get_next, &[h, 0, 0xC000, 0xC004]) == 1 {
        n += 1;
    }
    assert_eq!(n, 7);
}

#[test]
fn affinity_moves_a_thread_to_its_processor() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x2_4000);
    call(rt, &mut c, objects::nt_create_event, &[0xD000, 0, 0, 0]);
    let ev = rt.mem.read_u32(0xD000) as u64;
    // Created suspended on processor 4 (mask 0x10 in the top byte).
    let flags = 1 | 0x10 << 24;
    call(
        rt,
        &mut c,
        threads::ex_create_thread,
        &[0xD010, 0, 0, 0, THREAD_ENTRY as u64, ev, flags],
    );
    let h = rt.mem.read_u32(0xD010) as u64;
    call(
        rt,
        &mut c,
        objects::ob_reference_object_by_handle,
        &[h, 0, 0xD020],
    );
    let t = rt.mem.read_u32(0xD020);
    assert_eq!(rt.mem.read_u8(t + 0xBF), 4);
    assert_eq!(
        call(
            rt,
            &mut c,
            threads::ke_set_affinity_thread,
            &[t as u64, 0x4, 0xD030]
        ),
        0x10
    );
    assert_eq!(rt.mem.read_u8(t + 0xBF), 2);
    call(rt, &mut c, threads::nt_resume_thread, &[h, 0]);
    rt.mem.write_u64(0xD040, (-50_000_000i64) as u64);
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_wait_for_single_object_ex,
            &[h, 1, 0, 0xD040]
        ),
        0
    );
}

#[test]
fn empty_enumerations_and_request_completion() {
    use xenon_host::kernel::xam;
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x2_5000);
    // Synchronous: no more files, zero items.
    assert_eq!(
        call(
            rt,
            &mut c,
            xam::xam_enumerate,
            &[0x1234, 0, 0xE100, 0x134, 0xE000, 0]
        ),
        18
    );
    assert_eq!(rt.mem.read_u32(0xE000), 0);
    // With a request block: pending, completed with success and no items.
    rt.mem.write_u32(0xE200, 997);
    assert_eq!(
        call(
            rt,
            &mut c,
            xam::xam_enumerate,
            &[0x1234, 0, 0xE100, 0x134, 0, 0xE200]
        ),
        997
    );
    assert_eq!(
        (
            rt.mem.read_u32(0xE200),
            rt.mem.read_u32(0xE204),
            rt.mem.read_u32(0xE218)
        ),
        (0, 0, 0)
    );
    // A failure's extended error is the system error code form.
    xam::complete(rt, &c, 0xE300, 18, 0);
    assert_eq!(rt.mem.read_u32(0xE318), 0x8007_0012);
}

/// Guest entry for the leak test: returns at once.
const QUICK_ENTRY: u32 = 0x8200_0100;
fn quick_entry(_: &Runtime, c: &mut Context) {
    c.r[3] = 0;
}

#[test]
fn ended_threads_give_back_their_stacks() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_B000);
    let used = || kernel::k().virt.lock().unwrap().usage().0;
    let before = used();
    // Each thread takes the header's 256 KB default stack; 16 of them would
    // hold 4 MB if their stacks were never freed.
    for i in 0..16u32 {
        call(
            rt,
            &mut c,
            threads::ex_create_thread,
            &[0x7000, 0, 0, 0, QUICK_ENTRY as u64, 0, 0],
        );
        let th = rt.mem.read_u32(0x7000) as u64;
        rt.mem.write_u64(0x7010, (-50_000_000i64) as u64);
        assert_eq!(
            call(
                rt,
                &mut c,
                objects::nt_wait_for_single_object_ex,
                &[th, 1, 0, 0x7010]
            ),
            0,
            "thread {i} ended"
        );
        // The watcher frees the stack just after signalling the object.
        std::thread::sleep(std::time::Duration::from_millis(20));
        call(rt, &mut c, objects::nt_close, &[th]);
    }
    let after = used();
    assert!(
        after < before + (1 << 20),
        "virtual memory grew from {before:#x} to {after:#x}"
    );
}

#[test]
fn closing_a_periodic_timer_stops_it() {
    let e = env();
    let rt = &e.rt;
    let mut c = ctx(0x1_C000);
    // Notification timer, so it stays signalled once it fires.
    call(rt, &mut c, objects::nt_create_timer, &[0x7100, 0, 0]);
    let h = rt.mem.read_u32(0x7100) as u64;
    call(
        rt,
        &mut c,
        objects::ob_reference_object_by_handle,
        &[h, 0, 0x7104],
    );
    let a = rt.mem.read_u32(0x7104);
    rt.mem.write_u64(0x7110, (-10_000i64) as u64); // 1 ms, then every 2 ms
    assert_eq!(
        call(
            rt,
            &mut c,
            objects::nt_set_timer_ex,
            &[h, 0x7110, 0, 0, 0, 0, 2]
        ),
        0
    );
    let objects::KObj::Timer(ev, _) = &*objects::get(rt, a) else {
        panic!("not a timer")
    };
    let ev = ev.clone();
    assert!(
        ev.wait(Some(std::time::Duration::from_secs(5))),
        "timer fired"
    );
    assert_eq!(call(rt, &mut c, objects::nt_close, &[h]), 0);
    std::thread::sleep(std::time::Duration::from_millis(10));
    ev.reset();
    assert!(
        !ev.wait(Some(std::time::Duration::from_millis(50))),
        "a closed timer kept firing"
    );
}
