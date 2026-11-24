# ppc-oracle

Regenerates `crates/cpu-tests/data/oracle.txt`: reference results for every
per-instruction test case, measured on an emulated 64-bit PowerPC CPU.

```sh
# Needs llvm-mc, ld.lld and qemu-ppc64 (Debian/Ubuntu: llvm, lld, qemu-user).
python3 tools/ppc-oracle/cases.py
cargo test -p xenon-cpu-tests
```

- `harness.py` builds one static program from all cases. For each case it
  loads the input registers and scratch memory, runs the instructions, and
  stores every register and the scratch memory back. QEMU runs it; the
  script diffs output against input.
- `cases.py` lists the cases: edge values and seeded random ones, each vector
  instruction repeated with the destination equal to each source. VMX128
  instructions are run as the equivalent standard VMX instruction with the
  registers renamed.
- Register r31 holds the harness's pointer, so cases never use it.
- Hand-written cases for what the emulator cannot do (VMX128-only
  instructions, the console's 32-byte `dcbz`, load and store left/right)
  live in `crates/cpu-tests/data/hand.txt`.

The output is plain text and is committed, so CI and contributors need none
of these tools. No game code is involved.
