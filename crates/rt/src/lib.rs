//! `xenon-rt`: the small runtime that recompiled Xbox 360 code links against.
//!
//! Kept deliberately tiny so that a change here does not force the (very large)
//! generated code to be rebuilt more than necessary. Three layers:
//!
//! 1. this runtime crate,
//! 2. generated code (never in git; written locally from the player's own files),
//! 3. hooks and the host (`xenon-host`).
//!
//! Contents: guest CPU state, big-endian guest memory, the call choke point,
//! and the helpers that translated instructions call (`ops` for integer,
//! `fpu` for scalar floating point, `vmx` for the vector unit), plus
//! concurrent guest threads (`Runtime::spawn`) and the host objects behind
//! guest locks, events and semaphores (`sync`).

pub mod cpu;
pub mod fpu;
pub mod mem;
pub mod ops;
pub mod runtime;
pub mod sync;
mod thread;
pub mod vmx;

pub use cpu::{Context, Reservation, V128};
pub use mem::GuestMemory;
pub use runtime::{GuestFn, Runtime};
pub use sync::{CriticalSection, Event, Semaphore};
