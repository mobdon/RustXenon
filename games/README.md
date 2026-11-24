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
```

Role names describe what a function does (`file_open`, `frame_end`), never
what it is called in a particular game.
