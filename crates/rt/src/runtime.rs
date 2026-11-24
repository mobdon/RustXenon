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
