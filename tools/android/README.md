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
| `JAVA_HOME` | `C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-jdk/jdk-17.0.21.1+2` | `javac`, `d8`, `apksigner`, `keytool` |
| `BUILD_TOOLS` | `35.0.0` | build-tools version (`d8` comes from here) |
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
- links `AndroidManifest.xml` with `aapt2` (no Gradle);
- compiles `java/org/rustxenon/GameActivity.java` with `javac --release 17`
  against `android.jar` and turns it into `classes.dex` with
  `d8 --min-api 29`. The class is `android.app.NativeActivity` (still loading
  `libgame.so`) plus what native code cannot do from its own thread: it hides
  the status bar, navigation bar and taskbar (`WindowInsetsController`; they
  come back for a moment on a swipe from the edge), draws under the display
  cutout, and keeps the screen on. It does this on create and every time the
  window gets focus back;
- adds `lib/arm64-v8a/*.so` uncompressed, aligns for 16 KB pages
  (`zipalign -P 16`) and signs with the debug keystore.

The manifest asks for landscape, fullscreen, the game category, no touch
screen, and lists the config changes (fold, rotate, controller connect) the
game handles itself, so they never restart it.

## install.sh

```sh
tools/android/install.sh <app.apk> [package id]
```

`adb install -r`, then starts the activity, `org.rustxenon.GameActivity`. Set `ANDROID_SERIAL` when more
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
whose size differs are pushed, so the second run is quick. The executable is
always pushed, since a re-patched one usually keeps its size. Files already in
the phone's `data` folder are never replaced: once the game has run there, that
folder holds the phone's saves, and only files missing on the phone are copied.

Files that `adb push` writes belong to the shell user, and the app cannot
open them (`Permission denied`). So the script pushes into a staging folder,
`/storage/emulated/0/Android/data/<package>/files/.push/<sha256>/`, has the
app's own user copy it into the game folder with
`adb shell run-as <package> cp -r`, and then deletes the staging folder. Only
the files that changed go through staging; sizes are compared against the
game folder itself. A game folder left by an older version of the script,
which pushed straight into it, is copied once as the app's user the same way,
so its files become the app's (saves included).

`run-as` works only for a debuggable APK, which is what `build-apk.sh` makes
by default. A `--release` build is not debuggable, and the script stops with
a message saying so; install a dev build to push files.

Install the APK before the first push. The script stops if the package is not
installed, because a folder that `adb` makes under `Android/data` before the
app exists can end up unreadable by the app.

`build-apk.sh --release` makes a non-debuggable APK; dev builds are debuggable
so that `adb shell run-as <package>` works. Both are signed with the debug key,
so they are for this phone only.
`push-game.sh --self-test` checks the `run.toml` rewrite without a phone.

The files stay the player's own: they go to the phone only, never into git.
Uninstalling the app deletes that folder, saves included.
