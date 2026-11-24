//! Every guest call goes through one choke point so that hooks can override
//! any function by address, and system calls are looked up by import.

use std::collections::HashMap;

use crate::{Context, GuestMemory};

/// A recompiled (or hand-written override) guest function.
pub type GuestFn = fn(&mut Runtime, &mut Context);

pub struct Runtime {
    pub mem: GuestMemory,
    /// Game functions by guest address.
    functions: HashMap<u32, GuestFn>,
    /// Overrides installed by hooks; checked before `functions`.
    overrides: HashMap<u32, GuestFn>,
}

impl Runtime {
    pub fn new(mem: GuestMemory) -> Self {
        Self {
            mem,
            functions: HashMap::new(),
            overrides: HashMap::new(),
        }
    }

    /// Registers a recompiled function at its guest address.
    pub fn register(&mut self, addr: u32, f: GuestFn) {
        self.functions.insert(addr, f);
    }

    /// Replaces the function at `addr` with a hook.
    pub fn set_override(&mut self, addr: u32, f: GuestFn) {
        self.overrides.insert(addr, f);
    }

    /// Calls the guest function at `addr`. Panics with the address if nothing
    /// is registered there, so an unbuilt range reports where it was reached.
    pub fn call(&mut self, ctx: &mut Context, addr: u32) {
        let f = self
            .overrides
            .get(&addr)
            .or_else(|| self.functions.get(&addr))
            .copied()
            .unwrap_or_else(|| panic!("no guest function at {addr:#010x}"));
        f(self, ctx);
    }

    /// A trap instruction (`tw`, `td`, `twi`, `tdi`) whose condition held.
    /// Games use these for assertions, so stop with the address.
    #[cold]
    pub fn trap(&mut self, _ctx: &mut Context, pc: u32) -> ! {
        panic!("guest trap at {pc:#010x}");
    }

    /// `sc`. Titles reach the kernel through imports, not system calls.
    #[cold]
    pub fn syscall(&mut self, _ctx: &mut Context, pc: u32) -> ! {
        panic!("guest system call instruction at {pc:#010x}");
    }

    /// An instruction the translator does not know. Generated code still
    /// compiles; reaching it stops with the address and the word.
    #[cold]
    pub fn unknown(&mut self, _ctx: &mut Context, pc: u32, word: u32) -> ! {
        panic!("untranslated instruction {word:#010x} at {pc:#010x}");
    }

    /// Time base (`mftb`), ticking at the console's 49.875 MHz.
    pub fn timebase(&self) -> u64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        let t = START.get_or_init(Instant::now).elapsed();
        (t.as_nanos() * 49_875 / 1_000_000) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_one(_: &mut Runtime, c: &mut Context) {
        c.r[3] += 1;
    }
    fn add_ten(_: &mut Runtime, c: &mut Context) {
        c.r[3] += 10;
    }

    #[test]
    fn override_wins() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        let mut c = Context::new();
        rt.register(0x8200_0000, add_one);
        rt.call(&mut c, 0x8200_0000);
        assert_eq!(c.r[3], 1);
        rt.set_override(0x8200_0000, add_ten);
        rt.call(&mut c, 0x8200_0000);
        assert_eq!(c.r[3], 11);
    }

    #[test]
    #[should_panic(expected = "0x82001000")]
    fn missing_function_reports_address() {
        let mut rt = Runtime::new(GuestMemory::new(16));
        rt.call(&mut Context::new(), 0x8200_1000);
    }
}
