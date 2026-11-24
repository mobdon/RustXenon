//! The whole-executable layout from `emit::block_function`: one arm per basic
//! block, functions split into parts, falling off the end as a tail call.

include!(concat!(env!("OUT_DIR"), "/blocks.rs"));

use xenon_rt::GuestMemory;

const START: u32 = 0x8200_0000;

fn rt() -> Runtime {
    Runtime::new(GuestMemory::new(0x1000))
}

#[test]
fn loop_crosses_part_boundaries() {
    let mut c = Context::new();
    count_loop(&rt(), &mut c);
    assert_eq!(c.r[3], 10);
}

#[test]
fn falling_off_the_end_calls_the_next_function() {
    fn next(_: &Runtime, c: &mut Context) {
        c.r[3] += 1;
    }
    let mut rt = rt();
    rt.register(START + 4, next);
    let mut c = Context::new();
    fall_off(&rt, &mut c);
    assert_eq!(c.r[3], 6);
}

#[test]
fn jump_table_targets_are_blocks() {
    for (target, want) in [(START + 8, 1), (START + 16, 2)] {
        let mut c = Context::new();
        c.r[3] = target as u64;
        table(&rt(), &mut c);
        assert_eq!(c.r[4], want);
    }
}

#[test]
#[should_panic(expected = "no guest function at 0x8200000c")]
fn a_jump_into_the_middle_of_a_block_reports_it() {
    let mut c = Context::new();
    c.r[3] = (START + 12) as u64;
    table(&rt(), &mut c);
}
