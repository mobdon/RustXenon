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
