# RustXenon

A static recompiler for Xbox 360 software, written in Rust.

`xenon-recomp` reads Xbox 360 machine code (64-bit big-endian PowerPC with the
console's extended vector unit) and writes Rust. That Rust links against a small
runtime crate, `xenon-rt`, and a host program, `xenon-host`, so a game the player
owns runs as a native program.

> **Status:** step 1, instruction translation. Every user-mode instruction
> group of the console CPU translates to Rust and is checked per instruction
> (about 4,000 cases). Step 2 has started: the executable loader, title-update
> patching and function/jump-table analysis work on synthetic test files
> (`xenon-recomp scan`). Code generation for a whole executable is next.
> Step 3 is in: guest threads run at the same time on host threads, reserved
> stores are host compare-and-swap, and the runtime has the locks, events and
> semaphores that guest kernel objects map onto (checked with synthetic
> multi-thread programs).

## You need your own copy

RustXenon contains no game code or data and never will. You supply the files
from a game you own. Generated code and built programs stay on your machine.
See [DATA-RULES.md](DATA-RULES.md).

## Layout

| Path | What it is |
|---|---|
| `crates/recomp` | `xenon-recomp`: executable loader, function analysis, PowerPC to Rust translation |
| `crates/rt` | `xenon-rt`: guest CPU state, big-endian guest memory, call dispatch and overrides. Kept tiny so generated code rebuilds rarely |
| `crates/cpu-tests` | Per-instruction tests: hand-assembled words, reference results as text, no game bytes |
| `tools/ppc-oracle` | Regenerates the reference results on a CPU emulator (needs LLVM and QEMU; CI does not) |
| `crates/host` | `xenon-host`: loads the player's files, runs recompiled code, replaces system calls; later graphics, input and audio |
| `games/<sha256>/hooks.toml` | Per-game hook addresses with role names. The only per-game file in git |
| `third_party/licenses` | Licence texts for credited sources |

## Build

```sh
cargo build
cargo test
cargo run -p xenon-host -- --probe
```

## Plan

| Step | Done when |
|---|---|
| 0. Repository, ignore rules, credits, data rules | First commit has no game references |
| 1. 64-bit PowerPC and vector instruction translation | Per-instruction tests pass |
| 2. Executable loader, package unpackers, function analysis | Generated Rust for a whole executable compiles within a set memory and time budget |
| 3. Concurrent guest threads and atomics | Synthetic multi-thread tests pass |
| 4. System call replacements | A target reaches its main loop headless and keeps running |
| 5. GPU command recorder, shader translation, wgpu renderer | First frames captured headless |
| 6. Window, controller, audio | Playable |
| 7. Saves, performance, high refresh rates | Smooth at high refresh rates |

### Instruction coverage (step 1)

| Group | Covered | Checked against |
|---|---|---|
| Integer, 64-bit mode | All arithmetic, logical, compare, shift and rotate forms with record (`.`) and overflow (`o`) variants, carries, `rld*`, `mulhd`, `divd`, `extsw`, `cntlzd` | Reference CPU emulator |
| Branches and CR | `b`, `bc`, `bclr`, `bcctr` with link and CTR forms, all CR logical ops, `mcrf`, `mfcr`/`mtcrf`/`mfocrf`/`mtocrf`, `mtspr`/`mfspr` for XER, LR, CTR, VRSAVE, time base | Emulator, plus call tests through `Runtime::call` |
| Loads and stores | Byte, half, word, doubleword, algebraic, update and indexed forms, byte-reversed, `lmw`/`stmw`, `lwarx`/`stwcx.`, `ldarx`/`stdcx.`, `dcbz`, `dcbz128`, cache hints and barriers | Emulator; hand values for `lmw`/`stmw` and the 32/128-byte `dcbz` |
| Scalar floating point | Arithmetic (single and double, fused multiply-add family), `fsqrt`, `fsel`, `fres`, `frsqrte`, conversions, moves, compares, FPSCR moves, all float loads and stores | Emulator |
| VMX (AltiVec) | All integer, saturating, multiply-sum, pack/unpack, permute, merge, splat, shift, compare (with CR6) and float instructions, VSCR moves, all vector loads and stores including the console's load/store left and right | Emulator; hand values for `vexptefp`, `vlogefp`, load/store left and right |
| VMX128 | 128 registers; all arithmetic, logic, compare, shift, conversion, rounding, permute, merge and memory forms; `vmsum3fp128`, `vmsum4fp128`, `vpermwi128`, `vrlimi128`, `vpkd3d128`, `vupkd3d128` | Same operation on the emulator with registers renamed, or hand values |

Every vector case also runs with the destination equal to each source
register. Not covered yet: string loads and stores (`lswi`, `lswx`, `stswi`,
`stswx`), the 10:10:10:2 and 20:20:20:4 formats of `vpkd3d128`/`vupkd3d128`,
FPSCR exception and result-class bits, non-default rounding modes for float
arithmetic, and privileged instructions. Reaching an untranslated instruction
stops with its address and word.

### Threads and atomics (step 3)

| Part | What it does |
|---|---|
| Guest memory | One allocation with a base pointer, shared by every thread. Every access is a host atomic (relaxed, of its own width when aligned, byte by byte when not), so concurrent guest threads are sound Rust. A range outside guest memory stops with its address |
| `lwarx`/`stwcx.`, `ldarx`/`stdcx.` | The load records the value; the conditional store is a host compare-and-swap against it. An unaligned reserved store stops with its address, where the console raises an alignment interrupt |
| Barriers | `sync` is a full fence, `lwsync` an acquire/release fence, `isync` an acquire fence, `eieio` a release fence |
| Threads | `Runtime::spawn` (optionally suspended), `join` (exit value in r3; a panic in the thread continues in the joiner), `suspend`/`resume`, `current_thread` |
| Kernel objects | `Event` (manual and auto reset), `Semaphore` (with maximum), `CriticalSection` (recursive), each wait with an optional timeout; `Runtime::object` finds the host object for a guest object's address |
| Call point | `Runtime::call` also counts calls against a per-thread `call_budget`, calls an optional `trace`, and is where a suspended thread stops |

Tests in `crates/cpu-tests/tests/threads.rs` run translated guest code (atomic
32- and 64-bit counters, a spin lock guarding plain stores) and hook-based
programs (a critical section, an event wake-up, a producer/consumer queue with
two semaphores, suspend and resume) on up to eight threads at once. Thread order
changes between runs, so tests check results that hold for every order.

Not covered yet: the kernel calls themselves (step 4 maps them onto these
objects), waiting on several objects at once, alertable waits and APCs, ending
a thread from deep inside guest code, thread priorities and processor
affinity, and the guest address layout (the image sits at `0x82000000`, so the
full 4 GB guest address range needs reserving rather than allocating). A
compare-and-swap succeeds where the console would fail a store whose word was
written back with the same value in between; locks and counters do not
depend on that.

### Design notes

- **One choke point for calls.** Every guest call goes through
  `Runtime::call`, so hooks can override any function by address and an
  unbuilt range panics with its address.
- **Real threads.** The console runs several guest threads at once. Each guest
  thread gets its own host thread, and reserve/conditional-store sequences
  become host compare-and-swap.
- **Vector sources are read first.** Each vector helper takes its sources by
  value and returns the result, so `vD == vA` cannot overwrite an input
  half-way. Tests check every vector instruction that way anyway.
- **64-bit mode.** Registers, carries and record-form compares are 64 bits
  wide; only the effective address is cut to 32 bits.
- **Explicit system boundary.** Kernel and system-module calls are imports by
  number, replaced in the host.
- **Built for huge executables.** Generated code is split by fixed address
  range, rewritten only when bytes change, oversized functions are split, and
  parallel compile jobs are capped.

## Credits

RustXenon is modelled on [XenonRecomp](https://github.com/hedge-dev/XenonRecomp)
(MIT). See [CREDITS.md](CREDITS.md) for all sources and licences.

## Licence

RustXenon is licensed under the [MIT licence](LICENSE). Third-party sources
keep their own licences; see [CREDITS.md](CREDITS.md) and `third_party/licenses`.
