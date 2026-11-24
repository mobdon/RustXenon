//! `xenon-host`: runs recompiled code on `xenon-rt`.
//!
//! Step 0 scaffold. The first milestone is headless: load the player's own
//! unpacked game folder, run until the first panic, and report it. Window,
//! input, audio and graphics come later. Tests never open a window or play
//! sound.

use xenon_rt::{Context, GuestMemory, Runtime};

/// Guest RAM size on the console.
const GUEST_RAM: usize = 512 * 1024 * 1024;

fn main() {
    let headless_probe = std::env::args().any(|a| a == "--probe");
    if !headless_probe {
        println!("xenon-host: scaffold only. Run with --probe to check the runtime links.");
        return;
    }
    let rt = Runtime::new(GuestMemory::new(GUEST_RAM));
    let mut ctx = Context::new();
    rt.mem.write_u32(0, 0xDEAD_BEEF);
    ctx.r[1] = rt.mem.read_u32(0) as u64;
    println!(
        "xenon-host: runtime ok, {} MiB guest RAM, r1={:#x}",
        rt.mem.len() >> 20,
        ctx.r[1]
    );
}
