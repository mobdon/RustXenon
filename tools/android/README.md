# Android packaging

Scripts that put a built game on an Android phone (arm64-v8a, Android 10 or
later) over USB debugging. They run in Git Bash on Windows and need Python 3.11+
on `PATH`. The plan these serve is [docs/android.md](../../docs/android.md).

Tool locations are environment variables. Their defaults are the paths on the
development PC:

| Variable | Default | Used for |
|---|---|---|
| `ANDROID_SDK` | `C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-sdk` | `aapt2`, `zipalign`, `apksigner`, `android.jar`, `adb` |
| `ANDROID_NDK` | `$ANDROID_SDK/ndk/30.0.16248370` | clang linker, `llvm-strip`, `llvm-readelf`, `libc++_shared.so` |
| `JAVA_HOME` | `C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-jdk/jdk-17.0.21.1+2` | `apksigner`, `keytool` |
| `BUILD_TOOLS` | `35.0.0` | build-tools version |
| `PLATFORM` | `android-35` | `android.jar` |
| `KEYSTORE` | `~/.android/debug.keystore` | signing key; made with the standard debug values if missing. Never put it in the repo |
| `ADB` | `$ANDROID_SDK/platform-tools/adb.exe` | `install.sh`, `push-game.sh` |

## build-apk.sh

```sh
tools/android/build-apk.sh [--release] [--no-strip] [--so libavcodec.so ...] \
    [--package org.rustxenon.player] [--label RustXenon] <workspace> <out.apk>
```

Builds package `game` in `<workspace>` as a `cdylib` for
`aarch64-linux-android` (dev profile unless `--release`, 4 cargo jobs unless
`--jobs`). The library must export `android_main` (the `android-activity`
crate, `native-activity` feature). The script then:

- strips debug info but keeps the symbol table, so tombstones still name
  functions (`--no-strip` keeps everything);
- reads the libraries' `NEEDED` entries, adds `libc++_shared.so` from the NDK
  when needed, and stops if something else is missing (pass it with `--so`);
- links `AndroidManifest.xml` with `aapt2` (no Gradle, no Java code: the
  activity is `android.app.NativeActivity` loading `libgame.so`);
- adds `lib/arm64-v8a/*.so` uncompressed, aligns for 16 KB pages
  (`zipalign -P 16`) and signs with the debug keystore.

The manifest asks for landscape, fullscreen, the game category, no touch
screen, and lists the config changes (fold, rotate, controller connect) the
game handles itself, so they never restart it.

## install.sh

```sh
tools/android/install.sh <app.apk> [package id]
```

`adb install -r`, then starts the activity. Set `ANDROID_SERIAL` when more
than one phone is attached.

## push-game.sh

```sh
tools/android/push-game.sh [--package org.rustxenon.player] [--sha256 <hex>] <run.toml>
```

Copies the player's own files named in a built game's `run.toml` to the app's
folder on the phone, `/storage/emulated/0/Android/data/<package>/files/games/<sha256>/`:
the executable under its own name, and the `game`, `update` and `data` folders
under those names. It writes a `run.toml` there whose paths point at those
copies. Paths are absolute, because the player opens them as given and the
app's working folder is not the game folder. The other keys in `run.toml` are
kept.

The sha256 is the name of the folder holding `run.toml` unless `--sha256` is
given. Relative paths in the PC's `run.toml` are taken from that folder. A
folder not yet on the phone goes over in one `adb push`. Otherwise only files
whose size differs are pushed, so the second run is quick.
`push-game.sh --self-test` checks the `run.toml` rewrite without a phone.

The files stay the player's own: they go to the phone only, never into git.
Uninstalling the app deletes that folder, saves included.
