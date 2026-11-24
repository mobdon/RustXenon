//! `xenon-rt`: the small runtime that recompiled Xbox 360 code links against.
//!
//! Kept deliberately tiny so that a change here does not force the (very large)
//! generated code to be rebuilt more than necessary. Three layers:
//!
//! 1. this runtime crate,
//! 2. generated code (never in git; written locally from the player's own files),
//! 3. hooks and the host (`xenon-host`).
//!
//! Step 0 scaffold: guest CPU state, big-endian guest memory and the call
//! choke point. Instruction helpers, atomics and threads arrive in later steps.

pub mod cpu;
pub mod mem;
pub mod runtime;

pub use cpu::Context;
pub use mem::GuestMemory;
pub use runtime::{GuestFn, Runtime};
