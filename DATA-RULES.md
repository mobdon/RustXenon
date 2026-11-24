# Data rules

RustXenon is a general Xbox 360 recompiler. It works only from the player's
own legally obtained copy of a game. These rules apply to every commit, branch,
issue, pull request and release.

## Never in git

1. **No game bytes.** No executables, discs, disc images, packages, title
   updates or downloadable content, in whole or in part.
2. **No generated Rust.** Code produced by `xenon-recomp` is derived from game
   code. It is written locally and stays local.
3. **No extracted assets.** No textures, models, audio, video, shaders,
   strings or any other file taken from a game.
4. **No built games.** No binaries that contain recompiled game code, and no
   captures, recordings, saves or traces from running one.
5. **No copied emulator code** and no tables derived from GPL code. Emulators
   are a reference for hardware behaviour only.

`.gitignore` blocks the usual locations and formats, but it is a safety net,
not permission: check `git status` before every commit.

## Public wording

- Nothing public names a game title, publisher or character: not code,
  comments, file or folder names, commit messages, branch names, issue titles,
  pull request text or release notes.
- Per-game files are named by the SHA-256 of the executable.
- The project is described as a general Xbox 360 recompiler.

## What per-game config may be in git

Only `games/<sha256>/hooks.toml`: a short file of guest addresses with neutral
role names, found by our own analysis. See `games/README.md`.

## Tests

Tests in git use no game bytes. CPU tests use hand-assembled instructions and
compare against a CPU emulator or hand-written reference values. Tests run
headless: no window, no focus change, no sound.

Game-level regression tests keep only `test.toml` (input script, frame
numbers, tolerance) in git. Their expected frames and every captured frame
are game output and stay local; the runner's own tests use a fake program
that draws synthetic frames.

## Credits

Every source we learn from or use is credited in `CREDITS.md`, with its licence
text in `third_party/licenses/`, from the commit that first uses it.
