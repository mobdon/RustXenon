# Per-game hook files

This folder holds **only** short hook-address files. Everything else here is
ignored by git (see `DATA-RULES.md`).

## Layout

```
games/
  <sha256 of the executable, lowercase hex>/
    hooks.toml
```

The folder is named by the SHA-256 of the exact executable it applies to.
No game titles, publisher names or character names appear anywhere: not in
the folder name, not in the file, not in comments.

## `hooks.toml`

Only what the build needs and what we found ourselves: guest addresses with
neutral role names.

```toml
# Executable hash, repeated so a renamed folder is caught.
sha256 = "<64 hex chars>"

[hooks]
# role name          = guest address
main_loop_tick       = 0x82000000
file_open            = 0x82000100
start_in_world       = 0x82000200

[functions]
# Function starts that analysis missed.
extra = [0x82000300]
# Functions whose end analysis gets wrong: [start, end).
bounds = [{ start = 0x82000400, end = 0x82000440 }]

# A jump table analysis missed or read wrongly, keyed by its bctr.
[[jump_tables]]
bctr = 0x82000500
targets = [0x82000510, 0x82000520]
default = 0x82000530

# Data inside a code section: never treated as instructions.
[[not_code]]
start = 0x82000600
end = 0x82000700
```

Every hook address is also registered as a function start. Unknown keys are
an error, so a typo is caught. `xenon-recomp scan` prints the SHA-256 and
picks this file up from `games/<sha256>/hooks.toml` by itself.

The SHA-256 is of the executable **as loaded**: when a title update is
applied, that is the patched executable (plain header plus decoded image),
not the file on disc.

Role names describe what a function does (`file_open`, `frame_end`), never
what it is called in a particular game.

## Regression tests

A game-level test is a folder under the game's hash. Only `test.toml` is
tracked; everything else there is ignored by git.

```
games/<sha256>/
  tests/
    <test name>/
      test.toml          in git: input script, frames, tolerance
      expected/          local only: frame-<N>.png, game pixels
```

```toml
# Controller script, passed to the program as --press.
input = "5000:start,9000:a"
# Presented frames to capture and compare, counting from 1.
frames = [300, 600]
# Optional: extra program arguments, and a time limit (default 300 s).
args = []
timeout_secs = 300

# Required. Runs with several guest threads are not exactly repeatable.
[tolerance]
channel = 8               # a pixel matches if R, G and B each differ by at most this
max_bad_fraction = 0.001  # the frame passes if at most this fraction of pixels do not match
```

Unknown keys are an error. Run the tests headless with

```sh
xenon-regress --program <built game> games/<sha256>/tests -- <program args>
```

The first run with `--bless` saves the actual frames as the expected ones
(look at them first). Each run prints PASS or FAIL per frame and writes, under
`runs/regress/<test name>/`, the actual frames, `frame-<N>.sheet.png`
(expected, actual and difference side by side, bad pixels in red) and
`frame-<N>.blink.png` (an animated PNG that flips between expected and
actual). See `crates/regress` for the program contract.
