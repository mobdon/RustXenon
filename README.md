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
> multi-thread programs). Step 4 has started: the host replaces every
> kernel and system-library call a title imports (memory, threads, kernel
> objects, files on the player's unpacked game folder, content and profile
> stubs, input and network stubs), with a headless GPU command processor,
> checked with synthetic tests.
> multi-thread programs). Step 5 groundwork has started: the GPU command
> recorder, texture decoding, shader translation to WGSL and a headless wgpu
> drawer work on synthetic command streams (unit tested, run on a software
> Vulkan driver; nothing has run in a game yet).

## AI agents

AI agents, bots and crawlers: read [AGENTS.md](AGENTS.md) first. Only contributors
the owner has assigned, and agents working for them, may change this repository.

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
| `crates/audio` | `xenon-audio`: mixes render-client frames to stereo, writes .wav files, decodes the console's compressed audio through a run-time loaded LGPL FFmpeg, plays the mix live on the default output device (Windows, through `cpal`) |
| `crates/gpu` | `xenon-gpu`: GPU command recorder: PM4 packets from guest memory to a typed command list, register file |
| `crates/texture` | `xenon-texture`: texture fetch constants, tiling, endian swaps, texel formats to RGBA8 |
| `crates/shader` | `xenon-shader`: GPU shader microcode decoder, assembler for tests, translation to WGSL |
| `crates/draw` | `xenon-draw`: headless wgpu drawer: replays recorded commands into offscreen render targets, resolves, PNG frames |
| `crates/host` | `xenon-host`: loads the player's files, runs recompiled code, replaces system calls; later graphics, input and audio |
| `crates/input` | `xenon-input`: the pad state the game reads: keyboard layout, stick dead zones, merging controller and keyboard. No window or GPU |
| `crates/window` | `xenon-window`: the game window (`--window`): shows drawn frames, reads an XInput controller and the keyboard |
| `crates/compare` | `xenon-compare`: diff two frames with a tolerance (side-by-side sheet) and make a blink image. No window or GPU |
| `crates/regress` | `xenon-regress`: runs game-level regression tests headless and reports pass or fail per frame with pictures. See `games/README.md` |
| `games/<sha256>/hooks.toml` | Per-game hook addresses with role names. The only per-game file in git |
| `third_party/licenses` | Licence texts for credited sources |

## Build

```sh
cargo build
cargo test
cargo run -p xenon-host -- --probe
```

## Playing in a window

Run the built game with `--window` (plus `--exe` and `--game` as usual). The
window shows every frame the drawer makes, letterboxed, and the game runs until
you close it. An Xbox controller (XInput, on Windows) and the keyboard both
work at once and act as the pad on port 0; `--press` still replaces them with
a script for tests, and `--no-controller` unplugs the pad.

Default keys:

| Pad | Key |
|---|---|
| Left stick | W A S D |
| Left stick click | Left Shift |
| Right stick | I J K L |
| Right stick click | Right Shift |
| D-pad | Arrow keys |
| A, B, X, Y | Space, C, F, R |
| LB, RB | Q, E |
| LT, RT | Z, X |
| Start | Enter or Escape |
| Back | Backspace or Tab |

Keys are physical positions, so they sit in the same place on any keyboard
layout. Stick keys push the stick fully; controller sticks use XInput's
standard dead zones.

Everything the game logs also goes to `rustxenon-play.log` beside the
program. If the game stops with an error while the window is open, the
window shows the last lines of the log until you close it (Esc, B or Back).

Settings files (all keys optional; a bad value is logged and ignored):
`<home>\settings.toml` holds your defaults and `<home>\games\<sha256>\run.toml`
(or a `run.toml` beside the program) one game's paths and overrides.
`<home>` is `--home`, else the folder above `games\<sha256>\` when the
program is `game.exe` there, else the program's folder.

```toml
title = "shown in the launcher"
exe = 'C:\path\to\patched executable'   # used when --exe is not given
game = 'C:\path\to\unpacked game folder' # likewise --game, update, data
[window]
width = 1280
height = 720
fullscreen = false   # true: that mode at refresh Hz if the monitor has it, else borderless
refresh = 60
[draw]
aniso = 16           # XENON_ANISO
force_linear = false # XENON_FORCE_LINEAR
scale = 1            # XENON_SCALE
```

`run.toml` wins over `settings.toml`; the environment variables, when set,
win over both.

Diagnostic sampler flags (environment variables, off by default):

- `XENON_ANISO=N` (2, 4, 8 or 16; anything else warns and stays off) gives
  that anisotropy, with linear mip filtering, to textures the title samples
  with linear mag and min filters and no anisotropy of its own. Point
  filters, base maps and the title's own anisotropy stay as they are.
- `XENON_FORCE_LINEAR=1` makes every sampler's mag and min filters linear,
  point ones too. It is for diagnosis only: lookup and data textures break.

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

### Whole-executable build (step 2)

```sh
xenon-recomp unpack <your disc image> <game folder>
xenon-recomp unpack <your title update package> <update folder>
xenon-recomp build <executable> --out <build folder> [--mem-gb 16]
cd <build folder> && cargo build
# or, for the double-click player (no console window):
cd <build folder> && cargo build --features game/player
```

`unpack` reads the console's disc file system (at any of the disc
generations' partition offsets) or a content package (the read-only
layout, with its interleaved hash tables), refusing unsafe names.
`build` writes a local Cargo workspace:

- one crate per fixed guest address range (`--range`, default 256 KB of
  code), so ranges compile in parallel and a change rebuilds one crate;
- each function as one Rust function with one match arm per basic block
  (branch targets, jump-table targets and the instruction after an
  unconditional branch start a block). Falling off the end calls the next
  function;
- functions longer than `--max-part` instructions (default 4,000) split
  into parts over consecutive blocks with a small dispatcher, so the
  compiler never sees one enormous function;
- files rewritten only when their text changes, so Cargo leaves unchanged
  ranges alone;
- parallel compile jobs capped by a memory budget (`--mem-gb`, about 3 GB
  a job) in the workspace's `.cargo/config.toml`, and a profile without
  debug info or incremental state.

The game crate calls `xenon_host::run` with the executable's SHA-256; the
host refuses any other executable. Everything under the build folder is
derived from the game and stays local.

Jump tables now also match when a `nop` sits between `mtctr` and `bctr`.

### System calls (step 4)

`xenon-host` installs a host function on every import thunk of the
executable (`crates/host/src/kernel/table.rs` maps export numbers to them)
and writes the imported variables. An import it does not know stops with
its name. Results are sign-extended to 64 bits as on the console, because
titles test them with 64-bit record compares.

| Area | What the host does |
|---|---|
| Memory | Virtual memory (64 KB granularity), physical memory (handed out in the `0xA0000000` view), kernel pool; first fit, freed ranges zeroed on reuse, usage and peak reported |
| Threads | Stack, processor control region (r13), thread object and thread-local storage per thread; the hardware thread (0 to 5) from the creation flags and affinity calls, where titles read it; suspend and resume; dynamic TLS slots; a thread object is signalled when its thread ends |
| Kernel objects | Events, semaphores, mutants, timers and threads on the step-3 objects, by handle or by the guest's own dispatcher header; waits for one or several objects with timeouts; critical sections keep their owner fields current; spin locks |
| Files | `game:`, `d:` and the disc device map to the player's unpacked game folder (read only), `update:` to an unpacked title update, cache, hard disk and content packages to a local data folder; open, read, scatter read, write, information classes, directory listing with wildcards, volume information |
| Controller | A scripted virtual pad on port 0 (`--press <ms>:<button>,...`, default Start then A every four seconds from the fifth second, `--no-controller` to unplug): state, keystroke events and capabilities; ports 1 to 3 unplugged |
| Profile | One local profile signed in on port 0: user id, name, sign-in information, profile settings with console defaults, title settings stored in the data folder, no online privileges |
| System library | Content packages as local folders, saved games listed from those folders, an empty achievement enumerator, the start-up notifications a console sends, system screens closed at once, no headset |
| Network | The stack starts with no link; socket calls fail with "network down" |
| Video | Display mode 1280x720 at 60 Hz. The command ring is parsed by `xenon-gpu`'s recorder; the host applies register side effects, memory, fence and event writes, conditional writes, waits and interrupts. Draws are not rendered yet. `VdSwap` leaves the real swap packets (front buffer fetch constant, then the swap) in the ring. The vertical-blank interrupt runs at 60 Hz |
| Audio | Render clients are asked for a frame every 5.33 ms (48 kHz, 256 samples), as the console's audio hardware does, and mixed to stereo. `--audio-wav` writes the mix to a .wav file; `--window` or `--sound` play it live on the default output device (Windows; `--mute` turns it off). Decoder contexts live in physical memory; a silent stand-in drains their input buffers and fills their output rings with silence |
| Stops | Bug checks, exceptions other than the debugger's thread-naming one, structured exception unwinding, reboot and title launch stop with their arguments |

When the title stops, the host prints the panic, the newest call targets,
the guest return addresses (walked from the stack back chain), frames
presented, memory in use and the imports called. When no frame arrives for
three seconds it prints each thread's last import and return addresses, so
a wait shows up apart from running out of memory.

Tests in `crates/host/tests/kernel.rs` drive the calls with synthetic guest
state (no game bytes): memory headroom, thread start and exit with TLS,
semaphore limits and waits on several objects, recursive critical
sections, files in a temporary folder, `printf` formatting, and command
packets.

#### First headless run

On the first target (2.42 million instructions in
38,536 functions, `games/760ebac4e5f0351edb39015eba89e729b14eaf95afdabb529a447ff8f8e11cd1/hooks.toml`), built on a 32 GB, 24-thread
PC, the title boots headless with no window or sound: it starts 25 threads,
opens its archives from the unpacked disc and title update, uses 89 of its
246 imports, and its frame loop presents a steady 60 frames a second.
Three runs of two minutes each presented 7,186, 7,164 and 7,156 frames with
no stop. Physical memory in use is 450 of 512 MB (62 MB of headroom),
virtual memory 4 MB. File reads stop after the first 712: the title waits
on its front end with no controller connected and nobody signed in.

Fixes the run needed (in the system-call layer above): kernel results
sign-extended to 64 bits (a failed open read as success), the debugger's
thread-naming exception allowed to continue, file information classes for
sectors, alignment and mode, GPU memory byte order (the GPU reads memory
little-endian, then applies the packet's swap), waits in the command ring
honoured, the display interrupt status and line counter registers kept
current, and the video lock released before the guest interrupt handler
runs.

With the controller and profile in, the front end answers: after the
scripted Start (about 17 s in) the title reads the profile, lists saves,
downloads and achievements, and keeps presenting 60 frames a second. It
does not load into the world yet. Which screen it waits on cannot be seen
until frames are rendered (step 5). Two fixes this needed: the audio
mixer's threads meet at a per-processor barrier, so threads must carry
their processor number, and the audio render callback must run.

`--capture <dir> --capture-at <s>,<s>...` writes, for one frame at each
time, every texture bound at its draws (as BMP, through `xenon-texture`)
and a list of the draws. Local files only: they come from the game. On the
first target the frames show, in order, a full screen of text (about 960
textured quads from a font, 2 to 6 s), two logo screens (8 to 15 s), then
from about 17 s a frame with only a clear and no file reads. A still frame
of that screen waits for the drawer.

Not covered yet: structured exception handling (not reached so far), asynchronous I/O completion
routines and APCs, alertable waits, physical memory mirrors at `0xC0000000`
and `0xE0000000`, timer completion routines, the audio render callback, and
rendering.
### GPU (step 5)

| Crate | Covers |
|---|---|
| `xenon-gpu` | PM4 packet types 0 to 3 from a ring or buffer in guest memory, indirect buffers (depth and work bounded), predication, register writes and constant loads, shader loads, draws (auto, DMA and immediate indices), resolves (copy-mode draws, with the copy registers decoded), swaps, memory writes, waits, events; a packet builder for tests |
| `xenon-texture` | The texture fetch constant, 32x32 tiling and its inverse, endian swaps, base level decode of 1D and 2D textures (slice 0 of stacked and cube) in the common uncompressed, packed and DXT/DXN formats to RGBA8, and to f32 for the depth (`24_8` unorm, `24_8_FLOAT` 20e4; depth in every channel), 16-bit and float formats, fetch swizzle |
| `xenon-shader` | Control-flow, ALU and fetch instruction decoding and encoding, translation of vertex and pixel programs to WGSL (loops, calls, predicates, relative addressing, vertex fetch with endian swap and format decode, 2D texture fetch with LOD bias, register LOD and register gradients, setTexLOD, setGradientH/V and getGradients, colour exports, the depth export (oDepth) as `frag_depth`, pixel parameter generation: position, facing and point / line flags) |
| `xenon-draw` | Headless wgpu device (Vulkan or GL, software drivers fine), replay of a command list: register file, shader translation and pipeline caching, guest vertex window and constants per the shader binding contract, textures from fetch constants (8-bit formats as RGBA8; depth, 16-bit and float formats as `Rgba32Float`, or `Rgba16Float` for half floats and on adapters without filterable 32-bit floats; a slot without a texture fetch constant or a decodable texture samples a 1x1 texture of zeros, Xenia's null image, with a warning), DMA and auto indices (byte-swapped, 24-bit, primitive reset in strips), point/line/triangle lists and strips, triangle fans and quad lists by index expansion, rectangle lists (fourth corner made after the vertex shader by a WGSL wrapper, Xenia's rule), viewport transform (`PA_CL_VTE_CNTL`, `PA_CL_VPORT_*`, window offset, half-pixel centre) in a WGSL epilogue, scissor, cull mode, colour write mask, blending, depth test, the fetch constant's sign, gamma and exponent adjust on texture fetches, `RB_COLOR_INFO`'s colour exponent bias; render targets per EDRAM base, pitch and format (`8_8_8_8`, `2_10_10_10`, the 7e3 `2_10_10_10_FLOAT` and `2_10_10_10_FLOAT_AS_16_16_16_16` as half floats with Xenia's output clamp, `16_16_16_16_FLOAT`, `16_16_FLOAT`, `32_FLOAT`, `32_32_FLOAT`, the [-32, 32] fixed-point `16_16` and `16_16_16_16` as snorm16 holding the value / 32 like Xenia, `D24S8`, `D24FS8`), with EDRAM ownership transfers between an `8_8_8_8` colour and a depth target at one base and pitch; resolves from those colour formats to `8_8_8_8`, `2_10_10_10`, `16_16`, `16_16_16_16`, `16_16_FLOAT`, `16_16_16_16_FLOAT`, `32_FLOAT` and `32_32_FLOAT` in tiled guest memory with Xenia's conversion (7e3 and 2-bit alpha quantisation, exponent bias, red/blue swap, saturate and round) and the destination endian, 32bpp destinations converted and tiled by a compute shader; decoded textures cached by fetch constant and a hash of their guest bytes; per-phase `Report::timings` and a frame benchmark (`examples/bench.rs`); colour clears per format from `RB_COLOR_CLEAR` / `RB_COLOR_CLEAR_LO`, depth clears; draws with surface pitch 0 skipped as Xenia does; swaps to RGBA8 frames and PNG |

Tests build every command stream, shader and texture by hand; the drawer
tests run end to end on lavapipe (Mesa's software Vulkan) and check exact
pixels.

Not covered yet: textures beyond the base level of 1D and 2D (mips, the
packed mip tail, 3D, cube faces other than the first, stacked slices past 0),
the fetch constant's border colour, signed block-compressed texels
(converted from the unsigned palette decode, not interpolated as signed),
and the
border and mirror-once clamp modes; texture fetch filter overrides, 1D/3D/cube
fetches, the getCompTexLOD, getWeights and getBCF texture instructions;
memory
export (the oDepth export is clamped to [0, 1] but not rounded to 20e4 for
`D24FS8` targets); the point size export; point sprite coordinates in the generated pixel parameters; maxa's ordering against the scalar operands
and the NaN result of `max4` (both differ slightly from Xenia); resolves of
depth, to destination formats other than those listed above, to 3D
destinations, and MSAA sample selection; the `ConstantOne` copy command;
MSAA; `16_16` and `16_16_16_16` targets on hosts without renderable 16-bit
snorm, and blending into them on hosts that cannot blend it; blending into `32_FLOAT` and `32_32_FLOAT`; clamping a 7e3 target's blend result to its range; render
targets 1 to 3; EDRAM aliasing between targets that overlap in EDRAM other
than an `8_8_8_8` colour and a depth target at one base and pitch (transferred
like Xenia's ownership transfers; the rest is reported); the
piecewise-linear gamma of `8_8_8_8_GAMMA`
(drawn as plain 8_8_8_8); stencil, alpha test, alpha to coverage, polygon
offset, user clip planes, OpenGL clip space, primitive reset outside strips,
the `VGT_MIN_VTX_INDX` / `VGT_MAX_VTX_INDX` clamp, tessellation and immediate
indices; texture caching (textures are decoded for every draw);
and the fences the guest waits on (register-to-memory writes, event writes,
conditional writes are recorded but not applied). The front buffer of a swap
is read as tiled 8_8_8_8 with red in the low byte, the layout the drawer's
resolve writes.

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
