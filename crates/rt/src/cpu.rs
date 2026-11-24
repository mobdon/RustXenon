//! Guest CPU state for one hardware thread of the Xbox 360 CPU (64-bit PowerPC).

/// Register file of one guest thread. Generated functions take `&mut Context`.
#[derive(Clone, Debug)]
pub struct Context {
    /// General purpose registers (64-bit).
    pub r: [u64; 32],
    /// Scalar floating point registers.
    pub f: [f64; 32],
    /// Vector registers (128-bit), stored as host-order u32 lanes.
    pub v: [[u32; 4]; 128],
    /// Condition register, eight 4-bit fields.
    pub cr: u32,
    /// Link register.
    pub lr: u64,
    /// Count register.
    pub ctr: u64,
    /// Fixed-point exception register (carry, overflow, summary overflow).
    pub xer: u64,
    /// Floating point status and control register.
    pub fpscr: u32,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            r: [0; 32],
            f: [0.0; 32],
            v: [[0; 4]; 128],
            cr: 0,
            lr: 0,
            ctr: 0,
            xer: 0,
            fpscr: 0,
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads condition register field `n` (0..8) as its 4-bit value (LT, GT, EQ, SO).
    pub fn cr_field(&self, n: usize) -> u32 {
        debug_assert!(n < 8);
        (self.cr >> (28 - 4 * n)) & 0xF
    }

    /// Writes condition register field `n` (0..8).
    pub fn set_cr_field(&mut self, n: usize, value: u32) {
        debug_assert!(n < 8);
        let shift = 28 - 4 * n;
        self.cr = (self.cr & !(0xF << shift)) | ((value & 0xF) << shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cr_fields_round_trip() {
        let mut c = Context::new();
        c.set_cr_field(0, 0b1000);
        c.set_cr_field(7, 0b0010);
        assert_eq!(c.cr_field(0), 0b1000);
        assert_eq!(c.cr_field(7), 0b0010);
        assert_eq!(c.cr, 0x8000_0002);
    }
}
