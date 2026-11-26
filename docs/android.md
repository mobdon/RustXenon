# Running built games on an Android phone

Status: plan. Nothing here runs on a phone yet.

Done since (compiled and unit tested, not run on a phone): the Android path
in `xenon-window` (`android::run`, the NativeActivity event pump, CPU fill of
the native window), `xenon_host::android_main`, and a game crate whose
library builds as the Android `cdylib` (`cargo rustc -p game --lib
--crate-type cdylib --target aarch64-linux-android`). Its files live in
`<external files>/games/<sha256>/`: run.toml, an optional args.txt, the log
and stdout.log.

Goal: a game built by RustXenon runs on an Android phone, played with an Xbox
controller paired over Bluetooth. No touch controls. A game picker on the
phone is driven by the controller only.

## Verdict

Feasible. Most crates already build for `aarch64-linux-android` unchanged.
The real work is one new platform layer (window, surface, pad events, app
entry), packaging into an APK, an Android build of the LGPL FFmpeg, and
then speed on ARM.

## What was checked (2026-11-26)

- `cargo check --target aarch64-linux-android` (Rust target installed, no
  NDK): **passes** for `xenon-rt`, `xenon-input`, `xenon-gpu`,
  `xenon-texture`, `xenon-shader`, `xenon-compare`, `xenon-draw`,
  `xenon-audio`. **Fails** for `xenon-window` and so for `xenon-host`: winit
  on Android needs an `android-native-activity` or `android-game-activity`
  feature and an `AndroidApp`.
- Linked with NDK 30 (API 29 clang as the linker, `CC_`/`AR_` set for build
  scripts): `cargo test --no-run -p xenon-rt -p xenon-input -p xenon-draw
  -p xenon-audio --target aarch64-linux-android` builds every test binary,
  wgpu's included. Nothing was installed or run on a phone.
- The phone could not be inspected: `adb devices` lists nothing and Windows
  shows no phone on USB (unplugged, or a charge-only cable or mode). The
  owner says it is a foldable with a Snapdragon 8 Gen 3 class SoC (Adreno 750,
  Vulkan 1.3), 12 GB RAM, arm64-v8a, Android 14 or later. Those figures are
  the published specs for that model, not read from the device. Confirm them
  with step 0 below.

Small fixes made with this plan: Rust treats Android as `target_os =
"android"`, not `"linux"`, so guarded guest memory (`crates/rt/src/guard.rs`,
mmap and SIGSEGV) and the resident memory report (`host_rss` in
`crates/host/src/lib.rs`, `/proc/self/statm`) fell back to the "unsupported"
stubs. Their cfgs now include Android. `xenon-rt` now checks with no warnings
for Android.

## What the phone needs

- arm64-v8a (64-bit ARM). 32-bit phones are out of scope.
- Vulkan 1.1 or later (wgpu's Vulkan backend). The drawer already treats BC
  texture compression, `DEPTH32FLOAT_STENCIL8` and 16-bit norm formats as
  optional (`crates/draw/src/lib.rs`, adapter features), so Adreno and Mali
  drivers that lack some of them still work.
- RAM: at least 6 GB, 8 GB or more preferred. The guest needs 512 MiB of
  physical memory plus the host heap, the drawer's textures and the generated
  code. Android kills a foreground app well before physical RAM runs out.
  Measure `host_rss` on the phone. The AGENTS.md headroom test still applies.
- Android 10 (API 29) or later. AAudio, the current GameActivity and
  scoped storage all work from there.
- Bluetooth Xbox controller (Xbox One S or later, or Series). Android
  exposes it as a standard gamepad.

## Blockers, by crate

| Crate | On Android | Work |
|---|---|---|
| `crates/rt` | Builds. Guarded memory now uses the Linux mmap and signal path; plain memory is `alloc_zeroed` | None. Byte swaps cost the same on ARM |
| `crates/rt` `vmx.rs` | Builds. The SSSE3 and FMA fast paths are x86-only, so ARM takes the scalar fallbacks | Later: NEON paths for the hot vector ops, after profiling |
| `crates/host` kernel | No Win32 calls in `kernel/`. Only `host_rss` was platform-specific (fixed) | Where the player's files live: an app-private folder (see Files) |
| `crates/window` | **Blocker.** winit with `with_any_thread` on its own thread; XInput reader is Windows-only (others get a stub that returns an empty pad) | New Android path, below |
| `crates/audio` | Builds, but no live sound: `cpal` is a Windows-only dependency, and FFmpeg is looked up as `lib<name>.so.<major>` | Enable `cpal` on Android (AAudio). Load FFmpeg as `lib<name>.so`, the only name an APK can ship. Build the LGPL FFmpeg with the NDK |
| `crates/draw` | Builds (wgpu `vulkan` + `gles` features already on) | Check the adapter on the phone; see Speed |
| Generated game code | Plain Rust, no platform code | Cross-compile as a `cdylib` with the NDK linker and `--mem-gb`. ARM build size and time are unmeasured |

## Input: Xbox controller over Bluetooth

winit 0.30 has no gamepad events, and on Android it does not pass on joystick
`MotionEvent`s (sticks, triggers). gilrs has no Android backend. So the
Android path reads input from `android-activity` directly:
`AndroidApp::input_events_iter()` gives `KeyEvent`s and `MotionEvent`s, and
`MotionEvent` pointers carry `axis_value(Axis::…)`. That also means not using
winit on Android at all: `android-activity` gives the `NativeWindow` for the
surface, and both the window and the event loop are already
there. Check this in step 1 before writing much code.

The mapping onto `xenon_input::Pad` (XInput button bits) is pure logic. It
goes in `crates/input` as a function from Android key codes and axis values to
`Pad`, and is unit tested headless:

| Android | Pad |
|---|---|
| `KEYCODE_BUTTON_A/B/X/Y` | A, B, X, Y |
| `KEYCODE_BUTTON_L1/R1` | LB, RB |
| `KEYCODE_BUTTON_THUMBL/THUMBR` | LS, RS (stick clicks) |
| `KEYCODE_BUTTON_START`, `KEYCODE_BUTTON_SELECT` | Start, Back |
| `KEYCODE_DPAD_*` or `AXIS_HAT_X/Y` | D-pad (controllers send one or the other) |
| `AXIS_X`, `AXIS_Y` | left stick (`Y` points down on Android: negate) |
| `AXIS_Z`, `AXIS_RZ` | right stick (Xbox controllers over Bluetooth) |
| `AXIS_LTRIGGER`/`AXIS_BRAKE`, `AXIS_RTRIGGER`/`AXIS_GAS` | LT, RT (0.0–1.0 to 0–255; firmwares differ, take the larger) |
| `KEYCODE_BUTTON_MODE` | Guide. Android often keeps it for itself, so don't rely on it |

Mark `KeyEvent`s from the gamepad source as handled. Otherwise B can also
arrive as `KEYCODE_BACK` and close the activity. Dead zones and merging
already live in `xenon-input`. Fill the existing `Pad` and reuse them. Only
the first controller is read, the same as the Windows path.

## Screen

- Fullscreen, immersive (no system bars), screen kept on, landscape.
- Foldables: the inner screen is close to square and the cover screen is tall
  and thin. Letterbox the 16:9 frame to the largest fit. Folding or unfolding
  destroys and recreates the native window (`TerminateWindow` /
  `InitWindow`), so the surface must be rebuilt without stopping the game.
- Present through a wgpu surface on the drawer's device instead of
  softbuffer's CPU copy. The phone's memory bandwidth is shared with the GPU,
  so the per-frame RGBA copy that is cheap on a PC costs more here.

## Files

- The player's own game files are copied to the app's private folder. For
  development use `adb push` to `/sdcard/Android/data/<app id>/files/`.
  Later, an in-app import from a USB stick or Downloads through the system
  file picker (Storage Access Framework). The picker needs touch, so import
  stays a one-off setup step, not part of playing.
- Saves and profile go in the same private folder. Nothing goes into git, the
  same as on Windows.

## Game picker on the phone, controller only

- One APK holds the picker and every built game. Each game is a
  `lib<sha256-prefix>.so` under `lib/arm64-v8a/`, and the picker `dlopen`s
  the one chosen. Android 10+ blocks running code from app data folders, and
  the APK's own lib folder is the supported place. shortcut: adding a game
  means rebuilding and reinstalling the APK; move to one APK per game if
  that gets slow.
- The list shows each game's title from its files, not from the repo (no game
  names in git). D-pad or left stick moves, A starts, B goes back, and Start
  plus Back together (held) leaves a game back to the list. The picker runs
  fullscreen in the same surface, so no touch is ever needed after install.
- Share the list and selection logic (pure, headless, testable) with the
  Windows `rustxenon-play.exe` launcher being built in another session. Put
  only drawing and input glue in each platform.

## Order of work

0. **Phone facts** (owner: install Android platform-tools, enable USB
   debugging, accept the PC's key on the phone). Record
   `adb shell getprop ro.product.cpu.abi`, `ro.build.version.release`,
   `/proc/meminfo` MemTotal, `dumpsys SurfaceFlinger | grep -i vulkan` or
   `pm list features | grep vulkan`, and `dumpsys input` with the controller
   paired. Nothing is installed on the phone in this step. Size: minutes.
1. **Toolchain + hello surface** (needs owner OK to install the NDK and
   `cargo-ndk` or `cargo-apk`/`xbuild` on the PC, and to install a test APK
   on the phone): a `cdylib` with `android_main` that clears a wgpu surface
   and logs controller events through `logcat`. This settles the
   `android-activity` input route above. Size: small (1–2 days).
2. **Pad mapping** in `crates/input`, unit tested headless. Size: small.
3. **Android platform layer**: a new `xenon-window` path (cfg android):
   surface rebuild on fold or unfold, letterboxing, pad reader, the same
   `Link` the host uses today so `xenon-host` doesn't change. Size: medium.
4. **Audio**: `cpal` on Android, FFmpeg `.so` naming, an NDK build of
   `tools/ffmpeg-lgpl`. Size: small to medium.
5. **First game on the phone**: cross-compile one generated workspace for
   `aarch64-linux-android` with `--mem-gb`, push the player's files, run.
   Measure the build time and size of the `.so`, `host_rss`, and frame time. Size:
   medium, mostly unknowns.
6. **Picker** (shared with the Windows launcher) and APK packaging of
   several games. Size: medium.
7. **Speed on ARM**: profile with `simpleperf`, NEON for the hot `vmx` ops,
   then drawer work (the open drawer-speed item applies on the phone as well).
   Size: open-ended.

## Risks

- **Speed.** The phone's CPU cores are a fraction of a desktop's, and the
  vector unit runs on scalar fallbacks until NEON paths exist. Expect lower
  frame rates than on the PC at first, possibly unplayable for heavy games
  until step 7.
- **Heat.** Sustained load throttles phones within minutes. Measure after 10
  minutes of play, not at the start.
- **Build time and size.** The generated code is large. ARM cross-builds of a
  whole game have not been measured. Keep the 3 GB per job cap.
- **The Guide button** may never reach the app. The leave-game combo uses
  Start plus Back instead.
