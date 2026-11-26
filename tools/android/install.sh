#!/usr/bin/env bash
# Installs an APK on the phone over adb and starts it.
#
#   tools/android/install.sh <app.apk> [package id]
#
# The package id defaults to org.rustxenon.player. ADB defaults to this PC's
# platform-tools; set ADB to use another, ANDROID_SERIAL to pick a phone.
set -euo pipefail

: "${ANDROID_SDK:=C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-sdk}"
: "${ADB:=$ANDROID_SDK/platform-tools/adb.exe}"
[ $# -ge 1 ] || { echo "usage: install.sh <app.apk> [package id]" >&2; exit 2; }
apk=$1
package=${2:-org.rustxenon.player}

"$ADB" install -r "$apk"
MSYS_NO_PATHCONV=1 "$ADB" shell am start -W -n "$package/android.app.NativeActivity"
