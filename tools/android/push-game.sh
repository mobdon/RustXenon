#!/usr/bin/env bash
# Copies a built game's run.toml and the player's files it names (exe, game,
# update, data) to the phone, under the app's own folder:
#
#   /storage/emulated/0/Android/data/<package>/files/games/<sha256>/
#     run.toml   rewritten so its paths point at the copies below
#     <exe name> the executable
#     game/      the unpacked game folder
#     update/    the title update, if run.toml names one
#     data/      the writable data folder, if run.toml names one
#
#   tools/android/push-game.sh [--package <id>] [--sha256 <hex>] <run.toml>
#
# The sha256 defaults to the name of the folder holding run.toml (built games
# sit in games/<sha256>/). Relative paths in run.toml are taken from that
# folder. Files already on the phone with the same size are skipped. ADB
# defaults to this PC's platform-tools; set ADB to use another.
# `push-game.sh --self-test` checks the run.toml rewrite without a phone.
set -euo pipefail

: "${ANDROID_SDK:=C:/Users/Admin/Documents/ChatGPT/mmo/.local/tools/android-sdk}"
: "${ADB:=$ANDROID_SDK/platform-tools/adb.exe}"
export ADB

package=org.rustxenon.player
sha=
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --package) package=$2; shift ;;
        --sha256) sha=$2; shift ;;
        --self-test) args=(--self-test) ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) echo "push-game.sh: unknown option $1" >&2; exit 2 ;;
        *) args+=("$1") ;;
    esac
    shift
done
[ ${#args[@]} -eq 1 ] || { echo "usage: push-game.sh [--package <id>] [--sha256 <hex>] <run.toml>" >&2; exit 2; }

python - "$package" "$sha" "${args[0]}" <<'EOF'
import os, re, subprocess, sys, tempfile, tomllib

KEYS = ("exe", "game", "update", "data")


def phone_name(key, local):
    """Where each key's file or folder goes inside the game's phone folder."""
    return os.path.basename(os.path.normpath(local)) if key == "exe" else key


def rewrite(text, phone):
    """run.toml with the top-level path keys replaced by `phone` (key -> path).

    Top-level keys come before the first [table], so only those lines change.
    """
    out, top = [], True
    for line in text.splitlines():
        if re.match(r"\s*\[", line):
            top = False
        if top and re.match(r"\s*(%s)\s*=" % "|".join(KEYS), line):
            continue
        out.append(line)
    head = ["%s = '%s'" % (k, phone[k]) for k in KEYS if k in phone]
    return "\n".join(head + out) + "\n"


def self_test():
    text = "title = \"t\"\nexe = 'C:\\x\\a.xex'\ngame = 'C:\\g'\n[window]\nexe = 1\n"
    got = rewrite(text, {"exe": "/p/a.xex", "game": "/p/game"})
    assert got == "exe = '/p/a.xex'\ngame = '/p/game'\ntitle = \"t\"\n[window]\nexe = 1\n", got
    parsed = tomllib.loads(got)
    assert parsed["exe"] == "/p/a.xex" and parsed["window"]["exe"] == 1
    assert phone_name("exe", "C:\\x\\a.xex") == "a.xex"
    assert phone_name("game", "C:\\x\\folder\\") == "game"
    print("push-game.sh self-test ok")


def adb(*args, **kw):
    return subprocess.run([os.environ["ADB"], *args], check=True, **kw)


def remote_sizes(path):
    """{relative path: size} of the files under `path` on the phone."""
    r = subprocess.run(
        [os.environ["ADB"], "shell", "find '%s' -type f -exec stat -c '%%s %%n' {} + 2>/dev/null" % path],
        capture_output=True, text=True,
    )
    sizes = {}
    for line in r.stdout.splitlines():
        size, _, name = line.strip().partition(" ")
        if size.isdigit() and name.startswith(path + "/"):
            sizes[name[len(path) + 1:]] = int(size)
    return sizes


def main(package, sha, run_toml):
    if run_toml == "--self-test":
        return self_test()
    if not re.fullmatch(r"[A-Za-z0-9_.]+", package):
        sys.exit("push-game.sh: bad package id " + package)
    folder = os.path.dirname(os.path.abspath(run_toml))
    sha = (sha or os.path.basename(folder)).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        sys.exit("push-game.sh: %s is not a sha256; pass --sha256" % sha)
    text = open(run_toml, encoding="utf-8").read()
    conf = tomllib.loads(text)
    local = {k: os.path.join(folder, conf[k]) for k in KEYS if isinstance(conf.get(k), str)}
    for k in ("exe", "game"):
        if k not in local:
            sys.exit("push-game.sh: run.toml has no %s" % k)
    for k, p in list(local.items()):
        if not os.path.exists(p):
            if k == "data":  # made by the game on first run
                continue
            sys.exit("push-game.sh: %s = %s does not exist" % (k, p))
    base = "/storage/emulated/0/Android/data/%s/files/games/%s" % (package, sha)
    phone = {k: base + "/" + phone_name(k, p) for k, p in local.items()}
    adb("shell", "mkdir -p '%s'" % base)
    have = remote_sizes(base)
    pushed = skipped = 0
    for k, src in local.items():
        if not os.path.exists(src):
            continue
        name = phone_name(k, src)
        if os.path.isfile(src):
            files = [(src, name)]
        else:
            files = []
            for root, _, names in os.walk(src):
                for n in names:
                    rel = os.path.relpath(os.path.join(root, n), src).replace(os.sep, "/")
                    files.append((os.path.join(root, n), name + "/" + rel))
        if not any(r.startswith(name + "/") or r == name for r in have) and os.path.isdir(src):
            print("pushing %s/ (%d files)" % (name, len(files)))
            adb("push", src + os.sep + ".", base + "/" + name)
            pushed += len(files)
            continue
        for path, rel in files:
            if have.get(rel) == os.path.getsize(path):
                skipped += 1
                continue
            adb("push", path, base + "/" + rel, stdout=subprocess.DEVNULL)
            pushed += 1
    with tempfile.TemporaryDirectory() as tmp:
        t = os.path.join(tmp, "run.toml")
        with open(t, "w", encoding="utf-8", newline="\n") as f:
            f.write(rewrite(text, phone))
        adb("push", t, base + "/run.toml", stdout=subprocess.DEVNULL)
    print("%s: %d files pushed, %d already there" % (base, pushed, skipped))


main(*sys.argv[1:])
EOF
