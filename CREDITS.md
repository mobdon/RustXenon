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
- Use: ideas, and ported code in the GPU crates listed in the
  sections below. Much of what ReXGlue does for guest memory
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

## Xenia: ported code in `xenon-gpu`

- Licence: BSD 3-Clause, Copyright (c) 2015 Ben Vanik and Xenia project
  contributors. Full text:
  [`third_party/licenses/Xenia-LICENSE.txt`](third_party/licenses/Xenia-LICENSE.txt)
- Studied at commit `95a5c3ee250f80c3b9d139658649d9ffb6db3eec`. Logic and
  tables are rewritten in Rust; each file names its sources at the top.
- From `src/xenia/gpu/`:
  - `command_processor.cc`: PM4 packet decoding for types 0 to 3, the
    type-3 opcode handlers (field layouts, address and endian masks,
    `SET_CONSTANT` type bases, `REG_RMW`, `LOAD_ALU_CONSTANT`, `IM_LOAD`,
    draw initiator and index DMA, predication with bin mask and select, the
    indirect buffer length mask), ring buffer read semantics, and the guest
    byte order of GPU memory writes and polls.
  - `packet_disassembler.cc`: header fields and the `DRAW_INDX` layouts.
  - `xenos.h`: PM4 opcode values, primitive, index format, endian, copy
    command, fetch constant, EDRAM mode and event values, the vertex fetch
    constant layout, `CpuToGpu`.
  - `registers.h` and `register_table.inc`: register addresses and the
    bitfields of the draw, surface, colour, depth, copy and window registers.
  - `register_file.h`: the register file size.
  - `draw_util.cc` (`GetResolveInfo`) and `d3d12/d3d12_command_processor.cc`
    (`IssueDraw`): which registers a resolve reads, and that a draw in EDRAM
    copy mode is a resolve.
  - `graphics_system.cc`: the `CP_RB_WPTR` address.
- From `src/xenia/memory.h` (`TranslatePhysical`): the physical address mask.

## LZX format description (Microsoft)

- Source: Microsoft's published LZX descriptions (the cabinet LZX format and
  the LZX DELTA section of the MS-PATCH open specification).
- Use: `crates/recomp/src/lzx.rs` is written from these format descriptions.
  No decoder code from any other project (libmspack, Xenia, lzxd) is used.

## Xenia: ported code in `xenon-texture`

- Licence: BSD 3-Clause, Copyright (c) 2015 Ben Vanik and Xenia project
  contributors. Full text:
  [`third_party/licenses/Xenia-LICENSE.txt`](third_party/licenses/Xenia-LICENSE.txt)
- Studied at commit `95a5c3ee250f80c3b9d139658649d9ffb6db3eec`. Logic is
  rewritten in Rust; each file names its sources at the top.
- From `src/xenia/gpu/`:
  - `xenos.h`: the texture fetch constant layout, texture formats, endian
    modes, dimensions, `GpuSwap`, the swizzle encoding.
  - `texture_address.h`: the 2D and 3D tiled address formulas.
  - `texture_util.cc`: the tiled address upper bound, the base level part of
    the guest texture layout (32-block pitch and height alignment, linear and
    tiled extents, array slice stride), size selection from a fetch constant.
  - `texture_info_formats.cc`: block size and bits per pixel of each format.
  - `texture_cache.cc`: how swizzle values 6 and 7 are treated.
  - `shaders/endian.xesli`: endian swap rules.
  - `shaders/pixel_formats.xesli` and `shaders/texture_load_*.xesl`: channel
    layouts of packed formats, DXT endpoint expansion and interpolation, DXT1
    punch-through, DXT3, DXT5, DXN, DXT3A and DXT5A decoding.
  - `d3d12/d3d12_texture_cache.cc` (host format table): which formats share
    a data layout, component replication, `16_EXPAND` as half float.

## XenosRecomp

- Project: https://github.com/hedge-dev/XenosRecomp
- Licence: MIT, Copyright (c) 2025 hedge-dev and contributors.
  Full text: [`third_party/licenses/XenosRecomp-LICENSE.md`](third_party/licenses/XenosRecomp-LICENSE.md)
- Use: reference for translating the console's GPU shader microcode ahead
  of time. `crates/shader` follows its approach to non-structured control
  flow (a program-counter switch inside a loop) and was cross-checked
  against its instruction layouts (`XenosRecomp/shader_code.h`) and its ALU
  and export translation (`XenosRecomp/shader_recompiler.cpp`), studied at
  commit `70a084c2392378818a4da4086e8e3420b4c3ea8d`. No code is copied.

## Xenia: ported code in `xenon-shader`

- Licence: BSD 3-Clause, Copyright (c) 2015 Ben Vanik and Xenia project
  contributors. Full text:
  [`third_party/licenses/Xenia-LICENSE.txt`](third_party/licenses/Xenia-LICENSE.txt)
- Studied at commit `95a5c3ee250f80c3b9d139658649d9ffb6db3eec`. Logic and
  tables are rewritten in Rust; each file names its sources at the top.
- From `src/xenia/gpu/`:
  - `ucode.h`, `ucode.cc`: control-flow, ALU and fetch instruction layouts,
    opcode tables, operand rules (scalar operand slots, the constant-op temp
    register, const0/const1 relative addressing, export write masks).
  - `xenos.h`, `registers.h`: vertex formats and component counts, endian
    modes, loop and vertex fetch constant layouts, the texture fetch constant
    LOD bias field, the vertex/pixel float constant split.
  - `shader_translator.cc`: the control-flow area bound, exec walking, fetch
    and ALU operand parsing, mini vertex fetch reuse.
  - `spirv_shader_translator.cc`, `spirv_shader_translator_alu.cc`,
    `spirv_shader_translator_fetch.cc`: ALU semantics (Shader Model 3
    multiply by zero, max/min, cube, predicate set, kill, the clamped
    reciprocal, square root and log forms, `mul_prev2`), loop stack and
    `aL`, vertex fetch addressing, endian swap and format decoding, texture
    LOD bias.
  - `shader_interpreter.cc`, `shader_interpreter.h`: the call stack,
    conditional exec, jump and call checks, temp register wrap, `aL` clamp.
  - `d3d12/d3d12_command_processor.cc`: the float constant bases.
- From `src/xenia/base/math.h`: a cross-check of the console's half-float
  range.

## Xenia: ported code in `xenon-draw`

- Licence: BSD 3-Clause, Copyright (c) 2015 Ben Vanik and Xenia project
  contributors. Full text:
  [`third_party/licenses/Xenia-LICENSE.txt`](third_party/licenses/Xenia-LICENSE.txt)
- Studied at commit `95a5c3ee250f80c3b9d139658649d9ffb6db3eec`. Logic is
  rewritten in Rust; each file names its sources at the top.
- From `src/xenia/gpu/`:
  - `registers.h`: the bitfields of `PA_SU_SC_MODE_CNTL`, `PA_SU_VTX_CNTL`,
    `PA_CL_VTE_CNTL`, the window and screen scissors,
    `RB_COLOR_MASK`, `RB_BLENDCONTROL`, `RB_DEPTHCONTROL`, `RB_COLOR_INFO`,
    `RB_DEPTH_INFO` and `RB_SURFACE_INFO`.
  - `xenos.h`: blend factor, blend operation, compare function, colour and
    depth render target format and pixel centre values.
  - `xenos.cc` (`Float20e4To32`) and `xenos.h` (`UNorm24To32`): the 20e4
    and 24-bit unorm depth clear values.
  - `d3d12/pipeline_cache.cc`: depth writes only while the depth test is
    enabled; `draw_util.h` (`IsClearingColor`): depth resolves do not clear
    colour; `registers.h` (`VGT_DRAW_INITIATOR`, `VGT_INDX_OFFSET`,
    `VGT_MULTI_PRIM_IB_RESET_INDX`): 16-bit index counts, 24-bit indices and
    primitive reset.
  - `draw_util.cc`: the scissor (`GetScissor`), the window offset and
    half-pixel offset of the viewport (`GetHostViewportInfo`), and the
    resolve rectangle and destination addressing (`GetResolveInfo`: three
    vertices in vertex fetch constant 0, 16.8 rounding, scissor clamp,
    8-pixel alignment, pitch clamp, `Tiled2D` destination offsets).
  - `spirv_shader_translator.cc` and `vulkan/vulkan_command_processor.cc`:
    the vertex position epilogue (`PA_CL_VTE_CNTL` W, XY and Z formats, then
    NDC scale and offset).
  - `primitive_processor.cc` and `primitive_processor.h`: guest index byte
    order (including the 32-bit swap modes on 16-bit indices), the 24-bit
    index mask and reset compare, and the triangle fan and quad list to
    triangle list orders.
  - `shaders/pixel_formats.xesli`, `d3d12/d3d12_render_target_cache.cc`:
    the 8_8_8_8 EDRAM packing (red in the low byte) used for colour clears
    and resolves.

## Rust crates

Dependencies from crates.io keep their own licences and are not copied into
this repository, so no licence files are added for them. `xenon-draw` uses
`wgpu` and its `naga` shader compiler (MIT or Apache-2.0), `png` (MIT or
Apache-2.0) and `pollster` (MIT or Apache-2.0).

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
