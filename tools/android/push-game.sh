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
# folder. The exe is always pushed (a re-patched exe keeps its size). Other
# files already on the phone with the same size are skipped, and a file
# already in data/ is never replaced: those are the phone's saves. The app
# must be installed first, so that Android owns its folder, and it must be a
# dev (debuggable) build: files are pushed to a staging folder and copied into
# place with `run-as`, so that the app owns them (it cannot read files adb
# pushed). ADB defaults to this PC's platform-tools; set ADB to use another.
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
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        -*) echo "push-game.sh: unknown option $1" >&2; exit 2 ;;
        *) args+=("$1") ;;
    esac
    shift
done
[ ${#args[@]} -eq 1 ] || { echo "usage: push-game.sh [--package <id>] [--sha256 <hex>] <run.toml>" >&2; exit 2; }

python -u - "$package" "$sha" "${args[0]}" <<'EOF'
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
    shell_test()
    print("push-game.sh self-test ok")


def shell_test():
    """Runs RECOVER and MIGRATE in a local sh on each interrupted state."""
    def sh(cmd):
        return subprocess.run(["sh", "-c", cmd]).returncode

    def tree(d):
        return sorted(os.listdir(d)) if os.path.isdir(d) else None

    def mk(d, *names):
        os.makedirs(d)
        for n in names:
            open(os.path.join(d, n), "w").close()

    with tempfile.TemporaryDirectory() as tmp:
        b = os.path.join(tmp, "g").replace(os.sep, "/")
        # Stopped after `mv b b.old`: b.own is the whole new copy.
        mk(b + ".old", "a", "s"); mk(b + ".own", "a", "s")
        assert sh(RECOVER.format(b)) == 0 and tree(b) == ["a", "s"]
        assert tree(b + ".old") is None and tree(b + ".own") is None
        # Stopped after `mv b.own b`, or inside `rm -rf b.old`.
        mk(b + ".old", "a")
        assert sh(RECOVER.format(b)) == 0 and tree(b) == ["a", "s"] and tree(b + ".old") is None
        # Stopped inside `cp`: b is whole, the partial b.own is thrown away.
        mk(b + ".own", "a")
        assert sh(RECOVER.format(b)) == 0 and sh(MIGRATE.format(b)) == 0
        assert tree(b) == ["a", "s"] and tree(b + ".own") is None and tree(b + ".old") is None
        # Both leftovers and b: RECOVER keeps everything and fails.
        mk(b + ".old", "a"); mk(b + ".own", "a")
        assert sh(RECOVER.format(b)) != 0 and tree(b + ".old") == ["a"]


def adb(*args, **kw):
    return subprocess.run([os.environ["ADB"], *args], check=True, **kw)


def run_as(package, cmd, **kw):
    """Runs `cmd` (no single quotes in it) on the phone as the app's own user."""
    return subprocess.run(
        [os.environ["ADB"], "shell", "run-as %s sh -c '%s'" % (package, cmd)], **kw
    )


def remote_sizes(package, path):
    """{relative path: size} of the files under `path` on the phone."""
    r = run_as(
        package,
        'find "%s" -type f -exec stat -c "%%s %%n" {} +' % path,
        capture_output=True,
        text=True,
    )
    # A partial list would make data/ look absent and its saves get overwritten.
    if r.returncode != 0:
        sys.exit("push-game.sh: could not list %s on the phone: %s" % (path, r.stderr.strip()))
    sizes = {}
    for line in r.stdout.splitlines():
        size, _, name = line.strip().partition(" ")
        if size.isdigit() and name.startswith(path + "/"):
            sizes[name[len(path) + 1:]] = int(size)
    return sizes


# An older push-game.sh pushed straight into the game folder, and the app
# cannot open those files. MIGRATE copies the folder once as the app's user.
# It swaps the folders before deleting anything, so an interrupted run never
# loses the only full copy; RECOVER finishes an interrupted swap first and
# fails if it cannot tell which copy is whole.
RECOVER = (
    'b="{0}"; if [ -d "$b.old" ]; then '
    'if [ ! -e "$b" ] && [ -d "$b.own" ]; then mv "$b.own" "$b"; fi; '
    'if [ -e "$b" ] && [ ! -e "$b.own" ]; then rm -rf "$b.old"; fi; fi; '
    '[ ! -e "$b.old" ]'
)
MIGRATE = (
    'b="{0}"; rm -rf "$b.own" && cp -r "$b" "$b.own" && mv "$b" "$b.old" '
    '&& mv "$b.own" "$b" && rm -rf "$b.old"'
)


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
    r = subprocess.run([os.environ["ADB"], "shell", "pm path " + package], capture_output=True, text=True)
    if "package:" not in r.stdout:
        sys.exit("push-game.sh: %s is not installed on the phone; install the APK first" % package)
    # Files adb pushes belong to the shell user, and the app cannot read them.
    # So they go to a staging folder first and the app's own user (run-as,
    # dev builds only) copies them into place.
    r = run_as(package, "true", capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(
            "push-game.sh: run-as %s failed (%s). It works only on a debuggable "
            "APK: install a dev build (build-apk.sh without --release)."
            % (package, (r.stdout + r.stderr).strip())
        )
    files_dir = "/storage/emulated/0/Android/data/%s/files" % package
    base = "%s/games/%s" % (files_dir, sha)
    stage = "%s/.push/%s" % (files_dir, sha)
    phone = {k: base + "/" + phone_name(k, p) for k, p in local.items()}
    if run_as(package, RECOVER.format(base)).returncode != 0:
        sys.exit(
            "push-game.sh: %s.old is left from an interrupted copy and %s.own is "
            "also there; check which one is whole on the phone by hand" % (base, base)
        )
    run_as(package, 'mkdir -p "%s"' % base, check=True)
    r = run_as(package, 'find "%s" ! -user $(id -u) | head -n 1' % base, capture_output=True, text=True)
    if r.stdout.strip():
        print("%s holds files the app cannot open; copying it as the app's user" % base)
        run_as(package, MIGRATE.format(base), check=True)
    have = remote_sizes(package, base)
    adb("shell", "rm -rf '%s' && mkdir -p '%s'" % (stage, stage))
    pushed = skipped = 0
    try:
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
                adb("push", src + os.sep + ".", stage + "/" + name)
                pushed += len(files)
                continue
            for path, rel in files:
                # data/ holds the phone's saves: never overwrite one with the PC copy.
                if (k == "data" and rel in have) or (k != "exe" and have.get(rel) == os.path.getsize(path)):
                    skipped += 1
                    continue
                adb("push", path, stage + "/" + rel, stdout=subprocess.DEVNULL)
                pushed += 1
        with tempfile.TemporaryDirectory() as tmp:
            t = os.path.join(tmp, "run.toml")
            with open(t, "w", encoding="utf-8", newline="\n") as f:
                f.write(rewrite(text, phone))
            adb("push", t, stage + "/run.toml", stdout=subprocess.DEVNULL)
        run_as(package, 'cp -r "%s/." "%s/"' % (stage, base), check=True)
    finally:
        subprocess.run([os.environ["ADB"], "shell", "rm -rf '%s'; rmdir '%s/.push' 2>/dev/null" % (stage, files_dir)])
    print("%s: %d files pushed, %d already there" % (base, pushed, skipped))


main(*sys.argv[1:])
EOF
