#!/usr/bin/env bash
# Builds the small LGPL-only FFmpeg that xenon-audio loads at run time:
# shared libavcodec + libavutil with just the decoders we use. No
# --enable-gpl, no --enable-nonfree, so the result stays LGPL 2.1+.
# Usage: tools/ffmpeg-lgpl/build.sh <out-dir>   (needs git, a C compiler, make)
#        tools/ffmpeg-lgpl/build.sh <out-dir> --android <ndk-dir>
# The Android mode cross-builds for arm64-v8a, API 29, with the NDK's clang.
# It gives libavcodec.so and libavutil.so whose SONAME and NEEDED entries
# have no version suffix, the only names an APK can ship in lib/arm64-v8a/.
# The NDK's own make is used when none is on PATH, so it also runs in Git
# Bash. configure wants a C11 compiler for the build machine too; set HOST_CC
# when there is no cc on PATH (on Windows, for example HOST_CC="zig cc").
set -euo pipefail
VERSION=n7.1
OUT=$(realpath -m "${1:?usage: build.sh <out-dir> [--android <ndk-dir>]}")
CROSS=()
MAKE=make
if [ "${2:-}" = --android ]; then
  NDK=$(realpath "${3:?usage: build.sh <out-dir> --android <ndk-dir>}")
  TC=$(echo "$NDK"/toolchains/llvm/prebuilt/*/bin)
  [ -d "$TC" ] || { echo "no NDK toolchain in $NDK" >&2; exit 1; }
  # Plain clang with --target, as the NDK's per-API wrappers are .cmd files
  # on Windows, which configure cannot run.
  CROSS=(--enable-cross-compile --target-os=android --arch=aarch64
    --cc="$TC/clang --target=aarch64-linux-android29"
    --ar="$TC/llvm-ar" --nm="$TC/llvm-nm" --ranlib="$TC/llvm-ranlib"
    --strip="$TC/llvm-strip" ${HOST_CC:+--host-cc="$HOST_CC"})
  command -v make >/dev/null || MAKE=$(echo "$NDK"/prebuilt/*/bin/make)
fi
WORK=$(mktemp -d)
# configure's temporary files; a Windows TMPDIR with backslashes breaks it.
mkdir "$WORK/tmp" && export TMPDIR="$WORK/tmp"
git clone -q --depth 1 --branch "$VERSION" https://github.com/FFmpeg/FFmpeg.git "$WORK/src"
cd "$WORK/src"
./configure --prefix="$OUT" ${CROSS[@]+"${CROSS[@]}"} \
  --enable-shared --disable-static \
  --disable-everything --disable-programs --disable-doc \
  --disable-avdevice --disable-avformat --disable-swscale \
  --disable-swresample --disable-avfilter --disable-network \
  --disable-autodetect --disable-x86asm \
  --enable-decoder=xma1,xma2,pcm_s16be
grep -q 'define CONFIG_GPL 0' config.h  # refuse a GPL build before installing
"$MAKE" -j"$(nproc)" install
rm -rf "$WORK"
echo "FFmpeg $VERSION (LGPL) installed in $OUT/lib"
