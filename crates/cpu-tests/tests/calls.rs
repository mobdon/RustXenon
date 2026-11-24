//! Calls, returns, jump tables and traps: everything that needs the runtime's
//! call choke point rather than a single register diff.

include!(concat!(env!("OUT_DIR"), "/calls.rs"));

use xenon_rt::GuestMemory;

const CALLEE: u32 = 0x8210_0000;

fn rt() -> Runtime {
    Runtime::new(GuestMemory::new(0x1000))
}

/// Checks that all eight argument registers arrive at full 64-bit width and
/// returns a 64-bit result.
fn callee(_: &Runtime, c: &mut Context) {
    for (i, r) in (3..=10).enumerate() {
        assert_eq!(c.r[r], 0xF000_0000_0000_0000 | i as u64, "r{r} truncated");
    }
    c.r[3] = 0xFFFF_FFFF_0000_0000;
}

fn args() -> Context {
    let mut c = Context::new();
    for (i, r) in (3..=10).enumerate() {
        c.r[r] = 0xF000_0000_0000_0000 | i as u64;
    }
    c
}

#[test]
fn bl_passes_and_returns_64_bit_values() {
    let mut rt = rt();
    rt.register(CALLEE, callee);
    let mut c = args();
    call_bl(&rt, &mut c);
    assert_eq!(c.r[3], 0xFFFF_FFFF_0000_0001);
    assert_eq!(c.lr, 0x8200_0004);
}

#[test]
fn tail_call_goes_through_the_runtime() {
    let mut rt = rt();
    rt.register(CALLEE, callee);
    let mut c = args();
    call_tail(&rt, &mut c);
    assert_eq!(c.r[3], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn override_applies_to_translated_calls() {
    fn hook(_: &Runtime, c: &mut Context) {
        c.r[3] = 41;
    }
    let mut rt = rt();
    rt.register(CALLEE, callee);
    rt.set_override(CALLEE, hook);
    let mut c = args();
    call_bl(&rt, &mut c);
    assert_eq!(c.r[3], 42);
}

#[test]
fn bctrl_and_blrl_call_the_register_target() {
    fn set7(_: &Runtime, c: &mut Context) {
        c.r[3] = 7;
    }
    let mut rt = rt();
    rt.register(0x8230_0000, set7);
    let mut c = Context::new();
    c.r[4] = 0x8230_0000;
    call_bctrl(&rt, &mut c);
    assert_eq!((c.r[3], c.lr), (7, 0x8200_0008));
    let mut c = Context::new();
    c.r[4] = 0x8230_0000;
    call_blrl(&rt, &mut c);
    assert_eq!((c.r[3], c.lr), (7, 0x8200_0008));
}

#[test]
fn bctr_to_a_local_address_stays_in_the_function() {
    let rt = rt();
    let mut c = Context::new();
    c.r[4] = 0x8200_000C;
    jump_local(&rt, &mut c);
    assert_eq!(c.r[3], 2);
}

#[test]
fn bctr_elsewhere_is_a_tail_call() {
    let mut rt = rt();
    fn plain(_: &Runtime, c: &mut Context) {
        c.r[3] = 99;
    }
    rt.set_override(CALLEE, plain);
    let mut c = Context::new();
    c.r[4] = CALLEE as u64;
    jump_local(&rt, &mut c);
    assert_eq!(c.r[3], 99);
}

#[test]
#[should_panic(expected = "guest trap at 0x82000000")]
fn unconditional_trap_stops_with_its_address() {
    trap(&rt(), &mut Context::new());
}

#[test]
fn conditional_trap_not_taken() {
    let mut c = Context::new();
    c.r[3] = 1;
    c.r[4] = 2;
    trap_eq(&rt(), &mut c);
}

#[test]
#[should_panic(expected = "guest trap")]
fn conditional_trap_taken() {
    let mut c = Context::new();
    c.r[3] = 5;
    c.r[4] = 5;
    trap_eq(&rt(), &mut c);
}

#[test]
#[should_panic(expected = "untranslated instruction 0x00000000 at 0x82000000")]
fn unknown_instruction_reports_word_and_address() {
    unknown(&rt(), &mut Context::new());
}

#[test]
#[should_panic(expected = "system call instruction at 0x82000000")]
fn sc_reports_its_address() {
    syscall(&rt(), &mut Context::new());
}
