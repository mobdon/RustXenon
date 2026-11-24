# RustXenon

A static recompiler for Xbox 360 software, written in Rust.

`xenon-recomp` reads Xbox 360 machine code (64-bit big-endian PowerPC with the
console's extended vector unit) and writes Rust. That Rust links against a small
runtime crate, `xenon-rt`, and a host program, `xenon-host`, so a game the player
owns runs as a native program.

> **Status:** step 0, repository scaffold. Nothing is recompiled yet.

## You need your own copy

RustXenon contains no game code or data and never will. You supply the files
from a game you own. Generated code and built programs stay on your machine.
See [DATA-RULES.md](DATA-RULES.md).

## Layout

| Path | What it is |
|---|---|
| `crates/recomp` | `xenon-recomp`: executable loader, function analysis, PowerPC to Rust translation |
| `crates/rt` | `xenon-rt`: guest CPU state, big-endian guest memory, call dispatch and overrides. Kept tiny so generated code rebuilds rarely |
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

### Design notes

- **One choke point for calls.** Every guest call goes through
  `Runtime::call`, so hooks can override any function by address and an
  unbuilt range panics with its address.
- **Real threads.** The console runs several guest threads at once. Each guest
  thread gets its own host thread, and reserve/conditional-store sequences
  become host compare-and-swap.
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
