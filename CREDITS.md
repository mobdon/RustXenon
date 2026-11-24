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
