//! `xenon-recomp` library: Xbox 360 machine code to Rust.
//!
//! - [`ppc`]: one instruction to Rust statements (64-bit PowerPC, VMX, VMX128).
//! - [`emit`]: a guest function as a Rust function.
//! - [`xex`]: the executable loader; [`patch`]: title updates; [`lzx`]: the
//!   decompressor both use.
//! - [`analysis`]: function and jump-table discovery, helped by the per-game
//!   [`hooks`] file.

pub mod analysis;
pub mod emit;
#[cfg(test)]
mod fixture;
pub mod hooks;
pub mod lzx;
pub mod patch;
pub mod ppc;
pub mod xex;
