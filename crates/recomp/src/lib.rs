//! `xenon-recomp` library: Xbox 360 machine code to Rust.
//!
//! - [`ppc`]: one instruction to Rust statements (64-bit PowerPC, VMX, VMX128).
//! - [`emit`]: a guest function as a Rust function.
//!
//! The executable loader and function analysis arrive in step 2.

pub mod emit;
pub mod ppc;
