# Credits and third-party notices

Licence texts are kept in [`third_party/licenses/`](third_party/licenses/).

## XenonRecomp

- Project: https://github.com/hedge-dev/XenonRecomp
- Licence: MIT, Copyright (c) 2025 hedge-dev and contributors.
  Full text: [`third_party/licenses/XenonRecomp-LICENSE.md`](third_party/licenses/XenonRecomp-LICENSE.md)
- Use: RustXenon is modelled on XenonRecomp's approach to static
  recompilation of Xbox 360 PowerPC code (one host function per guest
  function, a small runtime, function and jump-table analysis). Any code
  ported from it will be marked at the place it is used and keeps this
  notice.
- Ported code: the single-to-half conversion used by `vpkd3d128`
  (`f32_to_half` in `crates/rt/src/vmx.rs`), rewritten in Rust from
  XenonRecomp's recompiler. Its operand order for the VMX128 multiply-add
  forms was also our reference.
- Ported code (step 2, rewritten in Rust with bounds checks, marked at the
  top of each file):
  - `crates/recomp/src/xex.rs`: executable header layout and load steps,
    from `XenonUtils/xex.h` and `xex.cpp`.
  - `crates/recomp/src/patch.rs`: title-update application steps, from
    `XenonUtils/xex_patcher.cpp`. That file says it was referenced from
    Xenia's `xex_module.cc` (BSD 3-Clause, see the Xenia section).
  - `crates/recomp/src/analysis.rs`: the function extent walk
    (`XenonAnalyse/function.cpp`), the four jump-table shapes
    (`XenonAnalyse/main.cpp`) and the register save/restore helper sizes
    (`XenonRecomp/recompiler.cpp`).
  Studied at commit `ddd128bcca99fe8bfbb99bea583c972351fa6ace`.

## ReXGlue SDK

- Project: https://github.com/rexglue/rexglue-sdk
- Licence: BSD 3-Clause, Copyright (c) 2026 Tom Clay, with portions derived
  from the Xenia project, Copyright (c) 2022 Ben Vanik and Xenia project
  contributors.
  Full text: [`third_party/licenses/ReXGlue-LICENSE.txt`](third_party/licenses/ReXGlue-LICENSE.txt)
- Use: ideas only, no code. We studied its recompiler design: the order of
  its analysis phases (register, scan, discover, gap fill, merge, validate),
  the jump-table patterns it recognises, and lessons from its host/guest call
  bridge and title-update decoder.

## Xenia

- Project: https://github.com/xenia-project/xenia
- Licence: BSD 3-Clause, Copyright (c) Ben Vanik and Xenia project
  contributors. Its notice is part of
  [`third_party/licenses/ReXGlue-LICENSE.txt`](third_party/licenses/ReXGlue-LICENSE.txt).
- Use: ideas only, no code. Much of what ReXGlue does for guest memory
  layout, kernel objects, the GPU command processor and audio decoding comes
  from Xenia, so what we learned there traces back to Xenia too.
- Use in the loader (through XenonRecomp's port, see above): the order of
  steps for applying a title-update delta patch.
- Use in the instruction translator (facts about the hardware, no code):
  the VMX128 opcode values and register field layout, the D3D pack and
  unpack formats of `vpkd3d128`/`vupkd3d128` and where the packed words land,
  and the whole-quadword behaviour of the element loads. We checked our own
  decoder against the VMX128 encodings published in Xenia's instruction
  tests; none of its test data is copied into this repository.

- Use in the host (step 4, facts about the console, no code): the export
  numbers of the kernel and system-library functions and variables, the
  argument order of the calls we replace, the layout of the processor
  control region and thread object, the file information classes, the
  command-packet formats of the GPU's command processor, which GPU
  registers the hardware keeps current, and that the vertical-blank
  interrupt runs as processor 2. Studied at Xenia's `master` branch on
  2026-11-24 (`src/xenia/kernel`, `src/xenia/gpu`).

## Calendar arithmetic

- Howard Hinnant's `days_from_civil` / `civil_from_days` algorithms
  (public domain), rewritten in `crates/host/src/kernel/rtl.rs` for the
  kernel's time-field conversions.

## LZX format description (Microsoft)

- Source: Microsoft's published LZX descriptions (the cabinet LZX format and
  the LZX DELTA section of the MS-PATCH open specification).
- Use: `crates/recomp/src/lzx.rs` is written from these format descriptions.
  No decoder code from any other project (libmspack, Xenia, lzxd) is used.

## XenosRecomp

- Project: https://github.com/hedge-dev/XenosRecomp
- Licence: MIT, Copyright (c) 2025 hedge-dev and contributors.
  Full text: [`third_party/licenses/XenosRecomp-LICENSE.md`](third_party/licenses/XenosRecomp-LICENSE.md)
- Use: ideas only, no code. Reference for translating the console's GPU
  shader microcode ahead of time.

## LibertyRecomp, LibertyRecompAndroid and XHYN-IV-recomp

- Projects: https://github.com/OZORDI/LibertyRecomp,
  https://github.com/vaduur/LibertyRecompAndroid and
  https://github.com/XHYN-PH/XHYN-IV-recomp
- Licence: GNU General Public License v3.0.
- Use: ideas and published facts only. **No code, configuration or tables
  from these projects are in RustXenon**, so their licence text is not
  included. Their documentation of problems they found (vector instructions
  whose output register is also an input, 64-bit call arguments, unaligned
  title-update records, guest memory reservation, GPU driver workarounds,
  audio queue depth) shaped our test plan.

## QEMU and LLVM (tools only)

- Projects: https://www.qemu.org (GPL v2) and https://llvm.org (Apache 2.0
  with LLVM exceptions).
- Use: `tools/ppc-oracle` assembles test instructions with LLVM's `llvm-mc`
  and `ld.lld` and runs them on QEMU's 64-bit PowerPC user-mode emulator to
  record reference results. Neither is linked into or shipped with
  RustXenon, and no code from either is copied. The committed results in
  `crates/cpu-tests/data/oracle.txt` are measurements of emulated hardware
  behaviour. Where QEMU's behaviour differs from the console (its 128-byte
  `dcbz`, its inexact `vexptefp`/`vlogefp`, `fres` of zero) the case is a
  hand-written value instead.

## FFmpeg (planned, not used yet)

- Project: https://ffmpeg.org
- Licence: GNU Lesser General Public License (LGPL) v2.1 or later, for a
  build without the GPL-only parts.
- Planned use: decoding the console's compressed audio in the host.
- Conditions we will follow when it is added:
  1. Link FFmpeg as a separate dynamic library, never statically, so users
     can replace it.
  2. Use an FFmpeg build configured **without** `--enable-gpl` and
     **without** `--enable-nonfree`.
  3. Add the LGPL licence text to `third_party/licenses/` and a credit
     here, naming the FFmpeg version and its configure options.
  4. State where the matching FFmpeg source can be obtained.

## Our own earlier work

The instruction translator, runtime call choke point, headless host shape and
test method are carried over from the authors' earlier PowerPC recompiler
work.

## Adding a source

When code, tables or ideas come from somewhere new, add a section here and its
licence text to `third_party/licenses/` in the same commit.
