//! Guest CPU state for one hardware thread of the Xbox 360 CPU (64-bit PowerPC).
//!
//! The CPU runs in 64-bit mode: integer results, carries and record-form
//! comparisons are 64 bits wide, while guest addresses are 32 bits (the
//! effective address is truncated when memory is touched).

/// A vector register: four 32-bit words in big-endian element order.
/// `v[0]` is the leftmost element (lowest address when stored), and each word
/// holds its value as a host integer.
pub type V128 = [u32; 4];

/// A pending load reservation from `lwarx` / `ldarx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub addr: u32,
    /// The value that was loaded; the conditional store succeeds only if the
    /// memory still holds it (host compare-and-swap semantics).
    pub value: u64,
    /// 4 for `lwarx`, 8 for `ldarx`.
    pub size: u8,
}

/// Register file of one guest thread. Generated functions take `&mut Context`.
#[derive(Clone, Debug)]
pub struct Context {
    /// General purpose registers (64-bit).
    pub r: [u64; 32],
    /// Scalar floating point registers (IEEE double; moves keep exact bits).
    pub f: [f64; 32],
    /// Vector registers. The console's vector unit has 128 of them.
    pub v: [V128; 128],
    /// Condition register, eight 4-bit fields; field 0 is the top nibble.
    pub cr: u32,
    /// Link register.
    pub lr: u64,
    /// Count register.
    pub ctr: u64,
    /// XER carry.
    pub ca: bool,
    /// XER overflow.
    pub ov: bool,
    /// XER summary overflow (sticky).
    pub so: bool,
    /// XER byte count (low 7 bits), used only by string instructions.
    pub xer_bc: u8,
    /// Floating point status and control register.
    pub fpscr: u32,
    /// Vector status and control register (NJ = 0x1_0000, SAT = 0x1).
    pub vscr: u32,
    /// VRSAVE special register (not interpreted).
    pub vrsave: u32,
    /// Machine state register as last written by `mtmsr(d)`.
    pub msr: u64,
    /// Load reservation for the next conditional store.
    pub reserve: Option<Reservation>,
    /// Guest calls made through `Runtime::call` (checked against the call
    /// budget).
    pub calls: u64,
    /// This host thread's [`crate::thread`] state, set by `Runtime::spawn`
    /// so the per-call suspend check needs no thread-local lookup (each one
    /// is a libc call on Android). Null: look it up.
    pub(crate) thread: ThreadSlot,
    /// Opaque per-thread pointer for the host's call hook (its record of
    /// what this thread is doing), so the hook needs no thread-local lookup.
    /// Null until the host sets it; `Runtime::spawn` and `clone` reset it,
    /// so it never names another host thread's record.
    pub host: ThreadSlot,
}

/// A pointer that belongs to one host thread. Cloning gives null, so a
/// context copied for use elsewhere looks its thread up again.
#[derive(Debug)]
pub struct ThreadSlot(pub(crate) *const ());

impl ThreadSlot {
    pub const NULL: Self = Self(std::ptr::null());

    /// The address held (null when unset).
    pub fn get(&self) -> *const () {
        self.0
    }

    /// Sets the address.
    ///
    /// # Safety
    ///
    /// `p` must stay valid for as long as this context is used, and the
    /// context must stay on the current host thread: a move to another
    /// thread (not `clone`, which resets the slot) keeps the old address.
    pub unsafe fn set(&mut self, p: *const ()) {
        self.0 = p;
    }
}

impl Clone for ThreadSlot {
    fn clone(&self) -> Self {
        Self::NULL
    }
}

// SAFETY: only an address; the owners of the slots decide when it is used.
unsafe impl Send for ThreadSlot {}

impl Default for Context {
    fn default() -> Self {
        Self {
            r: [0; 32],
            f: [0.0; 32],
            v: [[0; 4]; 128],
            cr: 0,
            lr: 0,
            ctr: 0,
            ca: false,
            ov: false,
            so: false,
            xer_bc: 0,
            fpscr: 0,
            vscr: crate::vmx::VSCR_NJ,
            vrsave: 0,
            msr: 0x8000_0000_0000_0000,
            reserve: None,
            calls: 0,
            thread: ThreadSlot::NULL,
            host: ThreadSlot::NULL,
        }
    }
}

/// Condition field values.
pub const LT: u32 = 0b1000;
pub const GT: u32 = 0b0100;
pub const EQ: u32 = 0b0010;
pub const SO: u32 = 0b0001;

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads condition register field `n` (0..8) as its 4-bit value (LT, GT, EQ, SO).
    #[inline]
    pub fn cr_field(&self, n: usize) -> u32 {
        debug_assert!(n < 8);
        (self.cr >> (28 - 4 * n)) & 0xF
    }

    /// Writes condition register field `n` (0..8).
    #[inline]
    pub fn set_cr_field(&mut self, n: usize, value: u32) {
        debug_assert!(n < 8);
        let shift = 28 - 4 * n;
        self.cr = (self.cr & !(0xF << shift)) | ((value & 0xF) << shift);
    }

    /// Condition register bit `i` (0 = most significant, as PowerPC numbers them).
    #[inline]
    pub fn bit(&self, i: u32) -> bool {
        self.cr >> (31 - i) & 1 != 0
    }

    #[inline]
    pub fn set_bit(&mut self, i: u32, b: bool) {
        let m = 1u32 << (31 - i);
        self.cr = if b { self.cr | m } else { self.cr & !m };
    }

    /// Signed comparison into field `n`, with the summary-overflow copy.
    #[inline]
    pub fn cmp_s(&mut self, n: usize, a: i64, b: i64) {
        let v = if a < b {
            LT
        } else if a > b {
            GT
        } else {
            EQ
        };
        self.set_cr_field(n, v | self.so as u32);
    }

    /// Unsigned comparison into field `n`.
    #[inline]
    pub fn cmp_u(&mut self, n: usize, a: u64, b: u64) {
        let v = if a < b {
            LT
        } else if a > b {
            GT
        } else {
            EQ
        };
        self.set_cr_field(n, v | self.so as u32);
    }

    /// Record form: CR0 from the low 32 bits of the result compared with
    /// zero, as titles expect (a C runtime stream count stored as -1 and
    /// reloaded zero-extended must still test negative after `addic.`).
    #[inline]
    pub fn rec(&mut self, v: u64) {
        self.cmp_s(0, v as i32 as i64, 0);
    }

    /// XER as the 32-bit register `mfxer` reads.
    pub fn xer(&self) -> u64 {
        (self.so as u64) << 31
            | (self.ov as u64) << 30
            | (self.ca as u64) << 29
            | self.xer_bc as u64
    }

    pub fn set_xer(&mut self, v: u64) {
        self.so = v >> 31 & 1 != 0;
        self.ov = v >> 30 & 1 != 0;
        self.ca = v >> 29 & 1 != 0;
        self.xer_bc = (v & 0x7F) as u8;
    }

    /// Sets OV and the sticky SO for an `o` form instruction.
    #[inline]
    pub fn set_ov(&mut self, ov: bool) {
        self.ov = ov;
        self.so |= ov;
    }

    /// `mtcrf`: replaces the fields selected by `mask` (bit 7 = field 0).
    pub fn mtcrf(&mut self, mask: u32, v: u32) {
        let mut m = 0u32;
        for i in 0..8 {
            if mask >> (7 - i) & 1 != 0 {
                m |= 0xF << (28 - 4 * i);
            }
        }
        self.cr = (self.cr & !m) | (v & m);
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
        assert!(c.bit(0) && c.bit(30) && !c.bit(1));
    }

    #[test]
    fn record_compares_the_low_32_bits() {
        let mut c = Context::new();
        c.rec(0x8000_0000);
        assert_eq!(c.cr_field(0), LT);
        c.rec(0xFFFF_FFFF_0000_0000);
        assert_eq!(c.cr_field(0), EQ);
        // 0xFFFFFFFF loaded by lwz, minus one.
        c.rec(0x0000_0000_FFFF_FFFE);
        assert_eq!(c.cr_field(0), LT);
    }

    #[test]
    fn xer_round_trip() {
        let mut c = Context::new();
        c.set_xer(0xE000_0011);
        assert!(c.so && c.ov && c.ca);
        assert_eq!(c.xer(), 0xE000_0011);
    }
}
