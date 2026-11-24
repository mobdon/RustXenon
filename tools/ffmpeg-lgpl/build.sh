#!/usr/bin/env bash
# Builds the small LGPL-only FFmpeg that xenon-audio loads at run time:
# shared libavcodec + libavutil with just the decoders we use. No
# --enable-gpl, no --enable-nonfree, so the result stays LGPL 2.1+.
# Usage: tools/ffmpeg-lgpl/build.sh <out-dir>   (needs git, a C compiler, make)
set -euo pipefail
VERSION=n7.1
OUT=$(realpath -m "${1:?usage: build.sh <out-dir>}")
WORK=$(mktemp -d)
git clone -q --depth 1 --branch "$VERSION" https://github.com/FFmpeg/FFmpeg.git "$WORK/src"
cd "$WORK/src"
./configure --prefix="$OUT" \
  --enable-shared --disable-static \
  --disable-everything --disable-programs --disable-doc \
  --disable-avdevice --disable-avformat --disable-swscale \
  --disable-swresample --disable-avfilter --disable-network \
  --disable-autodetect --disable-x86asm \
  --enable-decoder=xma1,xma2,pcm_s16be
grep -q 'define CONFIG_GPL 0' config.h  # refuse a GPL build before installing
make -j"$(nproc)" install
rm -rf "$WORK"
echo "FFmpeg $VERSION (LGPL) installed in $OUT/lib"
