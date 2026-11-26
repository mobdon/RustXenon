#!/usr/bin/env bash
# Builds an APK from a cargo workspace whose package "game" has a lib target
# with an `android_main` (android-activity, native-activity).
#
#   tools/android/build-apk.sh [options] <workspace> <out.apk>
#
# Options:
#   --release         build the release profile, not debuggable (default: dev,
#                     as generated workspaces use, debuggable for run-as)
#   --no-strip        ship the library unstripped (default: debug info is
#                     stripped, the symbol table is kept for tombstones)
#   --so <file>       ship another .so beside libgame.so (repeatable, e.g.
#                     the FFmpeg libraries)
#   --package <id>    package id (default org.rustxenon.player)
#   --label <text>    launcher label (default RustXenon)
#   --jobs <n>        cargo jobs (default 4)
#
# Tool locations come from these variables, with this PC's defaults:
#   ANDROID_SDK, ANDROID_NDK, JAVA_HOME, BUILD_TOOLS (version), PLATFORM
#   (android-NN), KEYSTORE (debug keystore, kept outside the repo).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
: "${ANDROID_SDK:=C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-sdk}"
: "${ANDROID_NDK:=$ANDROID_SDK/ndk/30.0.16248370}"
: "${JAVA_HOME:=C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-jdk/jdk-17.0.21.1+2}"
: "${BUILD_TOOLS:=35.0.0}"
: "${PLATFORM:=android-35}"
: "${KEYSTORE:=$HOME/.android/debug.keystore}"

profile=dev
strip=1
sos=()
package=org.rustxenon.player
label=RustXenon
jobs=4
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --release) profile=release ;;
        --no-strip) strip=0 ;;
        --so) sos+=("$2"); shift ;;
        --package) package=$2; shift ;;
        --label) label=$2; shift ;;
        --jobs) jobs=$2; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        -*) echo "build-apk.sh: unknown option $1" >&2; exit 2 ;;
        *) args+=("$1") ;;
    esac
    shift
done
[ ${#args[@]} -eq 2 ] || { echo "usage: build-apk.sh [options] <workspace> <out.apk>" >&2; exit 2; }
workspace=${args[0]}
out=${args[1]}
case "$package" in
    *[!A-Za-z0-9_.]*|.*|*.|*..*) echo "build-apk.sh: bad package id $package" >&2; exit 2 ;;
esac

bin=$ANDROID_NDK/toolchains/llvm/prebuilt/windows-x86_64/bin
tools=$ANDROID_SDK/build-tools/$BUILD_TOOLS
jar=$ANDROID_SDK/platforms/$PLATFORM/android.jar
for f in "$bin/aarch64-linux-android29-clang.cmd" "$tools/aapt2.exe" "$tools/zipalign.exe" \
    "$tools/apksigner.bat" "$jar" "$JAVA_HOME/bin/keytool.exe"; do
    [ -e "$f" ] || { echo "build-apk.sh: missing $f (set ANDROID_SDK, ANDROID_NDK or JAVA_HOME)" >&2; exit 1; }
done
export JAVA_HOME
# PATH is ":"-separated, so a "C:/..." entry must be turned into "/c/..." first.
export PATH="$(cygpath -u "$JAVA_HOME")/bin:$PATH"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$bin/aarch64-linux-android29-clang.cmd"
export CC_aarch64_linux_android="$bin/aarch64-linux-android29-clang.cmd"
export AR_aarch64_linux_android="$bin/llvm-ar.exe"

echo "== cargo rustc ($profile)"
(cd "$workspace" && cargo rustc -j "$jobs" -p game --lib --crate-type cdylib \
    --target aarch64-linux-android --profile "$profile")
dir=debug
[ "$profile" = release ] && dir=release
target=$(cd "$workspace" && cargo metadata --format-version 1 --no-deps \
    | python -c "import json,sys; print(json.load(sys.stdin)['target_directory'])")
so=$target/aarch64-linux-android/$dir/libgame.so
[ -f "$so" ] || { echo "build-apk.sh: $so was not built" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/lib/arm64-v8a"
if [ $strip = 1 ]; then
    "$bin/llvm-strip.exe" --strip-debug -o "$stage/lib/arm64-v8a/libgame.so" "$so"
else
    cp "$so" "$stage/lib/arm64-v8a/libgame.so"
fi
for s in ${sos[@]+"${sos[@]}"}; do
    cp "$s" "$stage/lib/arm64-v8a/"
done

# Libraries the phone has itself; anything else must ship in the APK.
system="libc.so libm.so libdl.so liblog.so libandroid.so libEGL.so libGLESv2.so libGLESv3.so
    libvulkan.so libaaudio.so libOpenSLES.so libmediandk.so libnativewindow.so libjnigraphics.so
    libz.so libcamera2ndk.so libbinder_ndk.so libsync.so"
needed() { "$bin/llvm-readelf.exe" -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'; }
missing=0
for lib in "$stage"/lib/arm64-v8a/*.so; do
    for n in $(needed "$lib" | tr -d '\r'); do
        case " $(echo $system) " in *" $n "*) continue ;; esac
        [ -e "$stage/lib/arm64-v8a/$n" ] && continue
        if [ "$n" = libc++_shared.so ]; then
            cp "$bin/../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$stage/lib/arm64-v8a/"
            echo "shipping libc++_shared.so (needed by $(basename "$lib"))"
        else
            echo "build-apk.sh: $(basename "$lib") needs $n, which is not shipped (pass --so)" >&2
            missing=1
        fi
    done
done
[ $missing = 0 ] || exit 1

echo "== aapt2 link ($package)"
debuggable=true
[ "$profile" = release ] && debuggable=false
python - "$here/AndroidManifest.xml" "$stage/AndroidManifest.xml" "$package" "$label" "$debuggable" <<'EOF'
import sys
from xml.sax.saxutils import escape
src, dst, package, label, debuggable = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = text.replace("@PACKAGE@", package).replace("@LABEL@", escape(label, {'"': "&quot;"}))
text = text.replace("@DEBUGGABLE@", debuggable)
open(dst, "w", encoding="utf-8").write(text)
EOF
"$tools/aapt2.exe" link -o "$stage/base.apk" -I "$jar" --manifest "$stage/AndroidManifest.xml"

# Native libraries go in STORED, so the phone maps them straight from the APK
# (extractNativeLibs=false).
python - "$stage/base.apk" "$stage" <<'EOF'
import os, sys, zipfile
apk, stage = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(apk, "a") as z:
    libs = os.path.join(stage, "lib", "arm64-v8a")
    for name in sorted(os.listdir(libs)):
        z.write(os.path.join(libs, name), "lib/arm64-v8a/" + name, zipfile.ZIP_STORED)
EOF

echo "== zipalign, sign"
"$tools/zipalign.exe" -P 16 -f 4 "$stage/base.apk" "$stage/aligned.apk"
if [ ! -f "$KEYSTORE" ]; then
    mkdir -p "$(dirname "$KEYSTORE")"
    "$JAVA_HOME/bin/keytool.exe" -genkeypair -keystore "$KEYSTORE" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
fi
mkdir -p "$(dirname "$out")"
"$tools/apksigner.bat" sign --ks "$KEYSTORE" --ks-pass pass:android \
    --ks-key-alias androiddebugkey --key-pass pass:android --out "$out" "$stage/aligned.apk"
"$tools/zipalign.exe" -c -P 16 4 "$out"
ls -l "$out"
