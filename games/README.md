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
