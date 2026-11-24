//! `xenon-gpu`: the Xenos GPU side of the host, as pure logic.
//!
//! The guest writes PM4 command packets into a ring buffer in guest memory
//! and moves a write pointer; packets may call indirect buffers. The
//! [`Recorder`] parses that stream into a typed [`Command`] list and keeps the
//! register state, so a drawer can later replay the commands with no window
//! or GPU involved in the parsing.

pub mod command;
pub mod memory;
pub mod pm4;
pub mod recorder;
pub mod regs;
pub mod xenos;

pub use command::{
    Command, Draw, EventKind, EventWrite, IndexSource, Rect, Resolve, ShaderStage, WaitRegMem,
};
pub use memory::{GuestRead, SliceMemory, PHYSICAL_MASK};
pub use recorder::{Error, Recorder, Warning};
pub use regs::RegisterFile;
pub use xenos::{CopyCommand, Endian, FetchConstantType, IndexFormat, PrimitiveType, VertexFetch};
