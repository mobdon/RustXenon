"""Run instruction cases on a reference 64-bit PowerPC CPU and record results.

Needs: llvm-mc and ld.lld (assembler and linker for big-endian 64-bit
PowerPC) and qemu-ppc64 (Linux user-mode emulator). Each case is a short
instruction sequence with an input state. The whole batch becomes one static
program: for every case it loads the input state, runs the instructions and
stores the output state; at the end it writes all states to stdout.

The reference values this produces are committed as text, so the Rust tests
need none of these tools. No game code is involved anywhere.
"""

import os
import re
import struct
import subprocess
import tempfile

SCRATCH = 0x0010_0000  # guest scratch memory the cases may touch
SCRATCH_LEN = 512
DATA = 0x0100_0000  # state blocks
TEXT = 0x0200_0000

# State block layout (bytes).
GPR, FPR, VR = 0, 256, 512
CR, XER, CTR, LR, FPSCR, VSCR, MEM = 1024, 1032, 1040, 1048, 1056, 1072, 1088
BLOCK = MEM + SCRATCH_LEN  # 1600


def default_state():
    return {"vscr": 0x0001_0000}


def pack_state(st):
    """State dict -> bytes. Keys: r0..r31, f0..f31 (u64 bits), v0..v31 (u128),
    cr, xer, ctr, lr, fpscr, vscr, mem (dict addr -> bytes)."""
    b = bytearray(BLOCK)
    for k, v in st.items():
        if k == "mem":
            for addr, data in v.items():
                off = addr - SCRATCH
                assert 0 <= off and off + len(data) <= SCRATCH_LEN, hex(addr)
                b[MEM + off:MEM + off + len(data)] = data
        elif k[0] == "r":
            struct.pack_into(">Q", b, GPR + 8 * int(k[1:]), v)
        elif k[0] == "f" and k != "fpscr":
            struct.pack_into(">Q", b, FPR + 8 * int(k[1:]), v)
        elif k[0] == "v" and k != "vscr":
            b[VR + 16 * int(k[1:]):VR + 16 * int(k[1:]) + 16] = v.to_bytes(16, "big")
        elif k == "cr":
            struct.pack_into(">Q", b, CR, v)
        elif k == "xer":
            struct.pack_into(">Q", b, XER, v)
        elif k == "ctr":
            struct.pack_into(">Q", b, CTR, v)
        elif k == "lr":
            struct.pack_into(">Q", b, LR, v)
        elif k == "fpscr":
            struct.pack_into(">Q", b, FPSCR, v)
        elif k == "vscr":
            b[VSCR:VSCR + 16] = v.to_bytes(16, "big")
        else:
            raise KeyError(k)
    return bytes(b)


def unpack_state(b):
    st = {}
    for i in range(32):
        st[f"r{i}"] = struct.unpack_from(">Q", b, GPR + 8 * i)[0]
        st[f"f{i}"] = struct.unpack_from(">Q", b, FPR + 8 * i)[0]
        st[f"v{i}"] = int.from_bytes(b[VR + 16 * i:VR + 16 * i + 16], "big")
    st["cr"] = struct.unpack_from(">Q", b, CR)[0] & 0xFFFF_FFFF
    st["xer"] = struct.unpack_from(">Q", b, XER)[0] & 0xE000_007F
    st["ctr"] = struct.unpack_from(">Q", b, CTR)[0]
    st["lr"] = struct.unpack_from(">Q", b, LR)[0]
    st["fpscr"] = struct.unpack_from(">Q", b, FPSCR)[0] & 0xFFFF_FFFF
    st["vscr"] = int.from_bytes(b[VSCR:VSCR + 16], "big") & 0xFFFF_FFFF
    st["mem"] = bytes(b[MEM:MEM + SCRATCH_LEN])
    return st


def full_state(st):
    """Input state with every register present, for diffs."""
    full = unpack_state(pack_state(st))
    return full


def case_asm(idx, lines):
    inp = DATA + 2 * BLOCK * idx
    out = inp + BLOCK
    a = []
    e = a.append

    def li32(reg, val):
        e(f"lis {reg}, {(val >> 16) & 0xFFFF}")
        e(f"ori {reg}, {reg}, {val & 0xFFFF}")

    e(f"# case {idx}")
    li32(31, inp)
    # scratch memory from the input block
    li32(4, SCRATCH)
    e(f"addi 5, 31, {MEM}")
    e(f"li 3, {SCRATCH_LEN // 8}")
    e("mtctr 3")
    e(f"1: ld 6, 0(5)")
    e("std 6, 0(4)")
    e("addi 4, 4, 8")
    e("addi 5, 5, 8")
    e("bdnz 1b")
    e(f"li 3, {VSCR}")
    e("lvx 0, 31, 3")
    e("mtvscr 0")
    for i in range(32):
        e(f"li 3, {VR + 16 * i}")
        e(f"lvx {i}, 31, 3")
    e(f"lfd 0, {FPSCR}(31)")
    e("mtfsf 255, 0")
    for i in range(32):
        e(f"lfd {i}, {FPR + 8 * i}(31)")
    e(f"ld 3, {CR}(31)")
    e("mtcrf 255, 3")
    e(f"ld 3, {XER}(31)")
    e("mtxer 3")
    e(f"ld 3, {CTR}(31)")
    e("mtctr 3")
    e(f"ld 3, {LR}(31)")
    e("mtlr 3")
    for i in range(31):
        e(f"ld {i}, {GPR + 8 * i}(31)")
    e(f"# test {idx}")
    a.extend(lines)
    e(f"# save {idx}")
    e(f"addi 31, 31, {BLOCK}")
    for i in range(31):
        e(f"std {i}, {GPR + 8 * i}(31)")
    e("mfcr 3")
    e(f"std 3, {CR}(31)")
    e("mfxer 3")
    e(f"std 3, {XER}(31)")
    e("mfctr 3")
    e(f"std 3, {CTR}(31)")
    e("mflr 3")
    e(f"std 3, {LR}(31)")
    for i in range(32):
        e(f"stfd {i}, {FPR + 8 * i}(31)")
    e("mffs 0")
    e(f"stfd 0, {FPSCR}(31)")
    for i in range(32):
        e(f"li 3, {VR + 16 * i}")
        e(f"stvx {i}, 31, 3")
    e("mfvscr 0")
    e(f"li 3, {VSCR}")
    e("stvx 0, 31, 3")
    li32(4, SCRATCH)
    e(f"addi 5, 31, {MEM}")
    e(f"li 3, {SCRATCH_LEN // 8}")
    e("mtctr 3")
    e(f"1: ld 6, 0(4)")
    e("std 6, 0(5)")
    e("addi 4, 4, 8")
    e("addi 5, 5, 8")
    e("bdnz 1b")
    return a


def run(cases, workdir=None):
    """cases: list of (lines, input_state). Returns list of output states."""
    tmp = workdir or tempfile.mkdtemp(prefix="ppc-oracle-")
    os.makedirs(tmp, exist_ok=True)
    asm = [".abiversion 2", ".text", ".globl _start", "_start:"]
    for i, (lines, _) in enumerate(cases):
        asm.extend(case_asm(i, lines))
    total = 2 * BLOCK * len(cases)
    asm += [
        "li 0, 4",
        "li 3, 1",
        f"lis 4, {DATA >> 16}",
        f"lis 5, {(total >> 16) & 0xFFFF}",
        f"ori 5, 5, {total & 0xFFFF}",
        "sc",
        "li 0, 1",
        "li 3, 0",
        "sc",
        '.section .scratch,"aw",@nobits',
        f".zero {SCRATCH_LEN}",
        ".data",
    ]
    for _, st in cases:
        blk = pack_state(st)
        asm.append(".byte " + ",".join(str(x) for x in blk))
        asm.append(f".zero {BLOCK}")
    s = os.path.join(tmp, "cases.s")
    with open(s, "w") as fh:
        fh.write("\n".join(asm) + "\n")
    obj = os.path.join(tmp, "cases.o")
    exe = os.path.join(tmp, "cases.elf")
    ld = os.path.join(tmp, "link.ld")
    with open(ld, "w") as fh:
        fh.write(
            f"ENTRY(_start)\nSECTIONS {{ . = {SCRATCH:#x}; .scratch : {{ *(.scratch) }} . = {DATA:#x}; .data : {{ *(.data) }} . = {TEXT:#x}; .text : {{ *(.text) }} }}\n"
        )
    subprocess.run(
        ["llvm-mc", "-triple=powerpc64-unknown-linux-gnu", "-mcpu=pwr7", "-filetype=obj", s, "-o", obj],
        check=True,
    )
    subprocess.run(["ld.lld", "-static", "-T", ld, obj, "-o", exe], check=True)
    r = subprocess.run(["qemu-ppc64", "-cpu", "power7", exe], capture_output=True, check=True)
    data = r.stdout
    assert len(data) == total, (len(data), total)
    return [unpack_state(data[2 * BLOCK * i + BLOCK:2 * BLOCK * (i + 1)]) for i in range(len(cases))]


ENC = re.compile(r"encoding: \[([^\]]*)\]")


def encode(lines):
    """Assembles single instruction lines; returns their 32-bit words.
    A line `.long 0x...` is taken as is."""
    words = [None] * len(lines)
    todo = []
    for i, l in enumerate(lines):
        m = re.match(r"\s*\.long\s+(0x[0-9a-fA-F]+|\d+)\s*$", l)
        if m:
            words[i] = int(m.group(1), 0)
        else:
            todo.append(i)
    if todo:
        src = "\n".join(lines[i] for i in todo) + "\n"
        r = subprocess.run(
            ["llvm-mc", "-triple=powerpc64-unknown-linux-gnu", "-mcpu=pwr7", "-show-encoding"],
            input=src.encode(),
            capture_output=True,
            check=True,
        )
        encs = ENC.findall(r.stdout.decode())
        assert len(encs) == len(todo), (len(encs), len(todo), r.stderr.decode()[:2000])
        for i, enc in zip(todo, encs):
            bs = [int(x, 16) for x in enc.split(",")]
            words[i] = int.from_bytes(bytes(bs), "big")
    return words
