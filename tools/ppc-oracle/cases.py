"""Instruction cases for the reference CPU, and the generator for
`crates/cpu-tests/data/oracle.txt`.

Run from the repository root:

    python3 tools/ppc-oracle/cases.py

Every case is hand-picked edge values plus a few seeded random ones. Vector
cases are repeated with the destination equal to each source register, since
a translation that writes the destination before reading every source is a
real bug class on this CPU.

VMX128 instructions do not exist on the reference CPU. Where a VMX128
instruction is the same operation as a standard VMX one, the case runs the
standard instruction on the reference CPU with registers renamed, and the
test runs the VMX128 encoding with high register numbers. VMX128-only
operations have hand-written values in `crates/cpu-tests/data/hand.txt`.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

R = random.Random(0x5EED)
M64 = (1 << 64) - 1
SCR = harness.SCRATCH

I64 = [
    0, 1, 2, 7, M64, M64 - 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 0x1_0000_0000,
    0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, 0xFFFF_FFFF_8000_0000, 0x0000_0000_FFFF_0000,
    0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210, 0x0000_0001_0000_0001,
]
I64 += [R.getrandbits(64) for _ in range(4)] + [R.getrandbits(32) for _ in range(2)]

SHIFTS = [0, 1, 5, 31, 32, 33, 63, 64, 65, 127, 0xFFFF_FFFF_FFFF_FF83]

D = {
    "one": 0x3FF0_0000_0000_0000, "m2_5": 0xC004_0000_0000_0000, "zero": 0, "nzero": 1 << 63,
    "inf": 0x7FF0_0000_0000_0000, "ninf": 0xFFF0_0000_0000_0000, "qnan": 0x7FF8_0000_0000_0001,
    "snan": 0x7FF4_0000_0000_0000, "den": 1, "big": 0x7E37_E43C_8800_759C, "third": 0x3FD5_5555_5555_5555,
    "e31": 0x41E0_0000_0010_0000, "m31": 0xC1E0_0000_0010_0000, "e63": 0x43E0_0000_0000_0000,
    "half": 0x3FE0_0000_0000_0000, "m3_5": 0xC00C_0000_0000_0000, "pi": 0x4009_21FB_5444_2D18,
    "tiny": 0x3810_0000_0000_0000, "small_s": 0x36A0_0000_0000_0000,
}
DV = list(D.values())

F32 = [
    0x3F80_0000, 0xBFC0_0000, 0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0x7F80_0001,
    0x0000_0001, 0x8000_0010, 0x7F7F_FFFF, 0x0080_0000, 0x4049_0FDB, 0x3EAA_AAAB, 0x4B00_0001,
    0xCF00_0000, 0x4F80_0000, 0x3F00_0000, 0x3FC0_0000, 0xC020_0000, 0x4040_0000, 0x5F00_0000,
]


def rnd_f32():
    import struct
    return struct.unpack(">I", struct.pack(">f", R.uniform(-1000, 1000)))[0]


def vec(words):
    return (words[0] << 96) | (words[1] << 64) | (words[2] << 32) | words[3]


def fvec():
    pool = F32 + [rnd_f32() for _ in range(4)]
    return vec([R.choice(pool) for _ in range(4)])


def ivec():
    kind = R.randrange(4)
    if kind == 0:
        return R.getrandbits(128)
    if kind == 1:
        lanes = [0x00, 0x01, 0x7F, 0x80, 0xFF, 0xFE, 0x40]
        return int.from_bytes(bytes(R.choice(lanes) for _ in range(16)), "big")
    if kind == 2:
        lanes = [0x0000, 0x7FFF, 0x8000, 0xFFFF, 0x0001, 0x00FF, 0x0100]
        return int.from_bytes(b"".join(R.choice(lanes).to_bytes(2, "big") for _ in range(8)), "big")
    lanes = [0, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 1, 0xFFFF, 0x1_0000 - 1]
    return vec([R.choice(lanes) for _ in range(4)])


CASES = []  # (name, lines, input, flags, rust_words or None, rename or None, scrub)


def add(name, lines, inp, flags="", words=None, rename=None, scrub=()):
    st = harness.default_state()
    st.update(inp)
    CASES.append((name, lines, st, flags, words, rename, tuple(scrub)))


def nm(s):
    return s.replace(".", "_rc")


# ---------------------------------------------------------------- integer

def int_cases():
    xer_in = [0, 0x2000_0000]  # CA clear / set
    binops = [
        "add", "addc", "adde", "subf", "subfc", "subfe", "mullw", "mulld", "mulhw", "mulhwu", "mulhd", "mulhdu",
        "and", "andc", "or", "orc", "xor", "nor", "nand", "eqv",
    ]
    for op in binops:
        forms = [op, op + "."]
        if op in ("add", "addc", "adde", "subf", "subfc", "subfe", "mullw", "mulld"):
            forms += [op + "o", op + "o."]
        for f in forms:
            pairs = [(R.choice(I64), R.choice(I64)) for _ in range(6)] + [(0x7FFF_FFFF_FFFF_FFFF, 1), (0x8000_0000, 0x8000_0000)]
            for i, (x, y) in enumerate(pairs):
                logic = op in ("and", "andc", "or", "orc", "xor", "nor", "nand", "eqv")
                line = f"{f} 3, 4, 5" if not logic else f"{f} 3, 4, 5"
                add(f"{nm(f)}_{i}", [line], {"r4": x, "r5": y, "xer": R.choice(xer_in)})
            # destination is also a source
            add(f"{nm(f)}_alias", [f"{f} 4, 4, 5"], {"r4": pairs[0][0], "r5": pairs[0][1], "xer": 0x2000_0000})
    for op in ["addze", "addme", "subfze", "subfme", "neg"]:
        for f in [op, op + ".", op + "o", op + "o."]:
            for i, x in enumerate([0, 1, M64, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, R.getrandbits(64)]):
                add(f"{nm(f)}_{i}", [f"{f} 3, 4"], {"r4": x, "xer": [0, 0x2000_0000][i & 1]})
    for op in ["divw", "divwu", "divd", "divdu"]:
        for f in [op, op + ".", op + "o", op + "o."]:
            pairs = [(100, 7), (M64 - 99, 7), (0x8000_0000, 3), (7, M64), (0x1234_5678_9ABC_DEF0, 0x1234), (5, 0x1_0000_0003)]
            for i, (x, y) in enumerate(pairs):
                add(f"{nm(f)}_{i}", [f"{f} 3, 4, 5"], {"r4": x, "r5": y})
    imm = [0, 1, -1, 0x7FFF, -0x8000, 0x1234]
    for i, (x, s) in enumerate([(R.choice(I64), R.choice(imm)) for _ in range(8)] + [(M64, 1), (0, -1)]):
        add(f"addi_{i}", [f"addi 3, 4, {s}"], {"r4": x})
        add(f"addis_{i}", [f"addis 3, 4, {s}"], {"r4": x})
        add(f"addic_{i}", [f"addic 3, 4, {s}"], {"r4": x})
        add(f"addic_rc_{i}", [f"addic. 3, 4, {s}"], {"r4": x})
        add(f"subfic_{i}", [f"subfic 3, 4, {s}"], {"r4": x})
        add(f"mulli_{i}", [f"mulli 3, 4, {s}"], {"r4": x})
        u = s & 0xFFFF
        for op in ["andi.", "andis.", "ori", "oris", "xori", "xoris"]:
            add(f"{nm(op)}_{i}", [f"{op} 3, 4, {u}"], {"r4": x})
        add(f"cmpwi_{i}", [f"cmpwi 2, 4, {s}"], {"r4": x})
        add(f"cmpdi_{i}", [f"cmpdi 5, 4, {s}"], {"r4": x})
        add(f"cmplwi_{i}", [f"cmplwi 1, 4, {u}"], {"r4": x})
        add(f"cmpldi_{i}", [f"cmpldi 7, 4, {u}"], {"r4": x})
    add("li", ["li 3, -5"], {})
    add("lis", ["lis 3, -32768"], {})
    add("cmpwi_so", ["cmpwi 0, 4, 1"], {"r4": 1, "xer": 0x8000_0000})
    for i in range(10):
        x, y = R.choice(I64), R.choice(I64)
        if i == 9:
            y = x
        for op, crf in [("cmpw", 0), ("cmpd", 3), ("cmplw", 6), ("cmpld", 7)]:
            add(f"{op}_{i}", [f"{op} {crf}, 4, 5"], {"r4": x, "r5": y})
    for op in ["extsb", "extsh", "extsw", "cntlzw", "cntlzd"]:
        for f in [op, op + "."]:
            for i, x in enumerate([0, 0x80, 0x7F, 0x8000, 0x8000_0000, M64, 0x0000_0001_0000_0000, R.getrandbits(64)]):
                add(f"{nm(f)}_{i}", [f"{f} 3, 4"], {"r4": x})
    for op in ["slw", "srw", "sraw", "sld", "srd", "srad"]:
        for f in [op, op + "."]:
            for i, s in enumerate(SHIFTS):
                x = [0xFFFF_FFFF_8765_4321, 0x0000_0000_8000_0001, 0x8000_0000_0000_0001, R.getrandbits(64)][i % 4]
                add(f"{nm(f)}_{i}", [f"{f} 3, 4, 5"], {"r4": x, "r5": s})
    for f in ["srawi", "srawi.", "sradi", "sradi."]:
        lim = 32 if f.startswith("srawi") else 64
        for i, s in enumerate([0, 1, 4, 31, 32, 63]):
            if s >= lim:
                continue
            for j, x in enumerate([0xFFFF_FFFF_8765_4321, 0x8000_0000_0000_0001, 0x7FFF_FFFF_0000_0010, 0xFFFF_FFFF_FFFF_FFF0]):
                add(f"{nm(f)}_{i}_{j}", [f"{f} 3, 4, {s}"], {"r4": x})
    # 32-bit rotates in 64-bit mode
    for i in range(14):
        sh, mb, me = R.randrange(32), R.randrange(32), R.randrange(32)
        if i < 3:
            sh, mb, me = [(0, 0, 31), (4, 28, 3), (31, 31, 0)][i]
        x = R.choice(I64)
        for f in ["rlwinm", "rlwinm."]:
            add(f"{nm(f)}_{i}", [f"{f} 3, 4, {sh}, {mb}, {me}"], {"r4": x})
        add(f"rlwimi_{i}", [f"rlwimi 3, 4, {sh}, {mb}, {me}"], {"r4": x, "r3": R.choice(I64)})
        add(f"rlwnm_{i}", [f"rlwnm 3, 4, 5, {mb}, {me}"], {"r4": x, "r5": R.choice(I64)})
    for i in range(14):
        sh, mb = R.randrange(64), R.randrange(64)
        if i < 3:
            sh, mb = [(0, 0), (63, 1), (32, 32)][i]
        x = R.choice(I64)
        for f in ["rldicl", "rldicr", "rldic", "rldicl."]:
            add(f"{nm(f)}_{i}", [f"{f} 3, 4, {sh}, {mb}"], {"r4": x})
        add(f"rldimi_{i}", [f"rldimi 3, 4, {sh}, {mb}"], {"r4": x, "r3": R.choice(I64)})
        add(f"rldcl_{i}", [f"rldcl 3, 4, 5, {mb}"], {"r4": x, "r5": R.choice(SHIFTS)})
        add(f"rldcr_{i}", [f"rldcr 3, 4, 5, {mb}"], {"r4": x, "r5": R.choice(SHIFTS)})
    # condition register
    for op in ["crand", "crandc", "creqv", "crnand", "crnor", "cror", "crorc", "crxor"]:
        for i in range(4):
            cr = R.getrandbits(32)
            add(f"{op}_{i}", [f"{op} {R.randrange(32)}, {R.randrange(32)}, {R.randrange(32)}"], {"cr": cr})
    for i in range(4):
        add(f"mcrf_{i}", [f"mcrf {R.randrange(8)}, {R.randrange(8)}"], {"cr": R.getrandbits(32)})
        add(f"mfcr_{i}", ["mfcr 3"], {"cr": R.getrandbits(32), "r3": M64})
        add(f"mtcrf_{i}", [f"mtcrf {R.randrange(256)}, 4"], {"cr": R.getrandbits(32), "r4": R.getrandbits(64)})
        add(f"mfocrf_{i}", [f"mfocrf 3, {1 << R.randrange(8)}"], {"cr": R.getrandbits(32), "r3": M64})
        add(f"mtocrf_{i}", [f"mtocrf {1 << R.randrange(8)}, 4"], {"cr": R.getrandbits(32), "r4": R.getrandbits(64)})
    add("mtxer", ["mtxer 4"], {"r4": 0xFFFF_FFFF_E000_007F})
    add("mfxer", ["mfxer 3"], {"xer": 0xA000_0005})
    add("mtctr", ["mtctr 4"], {"r4": 0x1234_5678_9ABC_DEF0})
    add("mfctr", ["mfctr 3"], {"ctr": 0xFEDC_BA98_7654_3210})
    add("mtlr", ["mtlr 4"], {"r4": 0x1234_5678_9ABC_DEF0})
    add("mflr", ["mflr 3"], {"lr": 0x8200_1234})
    # traps that do not fire
    add("tw_not_taken", ["tw 4, 4, 5"], {"r4": 1, "r5": 2})
    add("twi_not_taken", ["twi 16, 4, 0"], {"r4": 5})
    add("td_not_taken", ["td 8, 4, 5"], {"r4": 0, "r5": 0x1_0000_0000})
    add("tdi_not_taken", ["tdi 2, 4, -1"], {"r4": M64})


# ---------------------------------------------------------------- branches

def branch_cases():
    # Branch over one instruction: taken leaves r5 alone, not taken sets it.
    skip = ["li 5, 1", "li 6, 2"]

    def bc(bo, bi, disp):
        return f".long {(16 << 26) | bo << 21 | bi << 16 | (disp & 0xFFFC):#x}"

    def b(disp):
        return f".long {(18 << 26) | (disp & 0x3FF_FFFC):#x}"

    conds = [(12, 2), (4, 2), (12, 0), (4, 1), (16, 0), (18, 0), (8, 2), (0, 2), (10, 3), (2, 1), (20, 0)]
    for i, (bo, bi) in enumerate(conds):
        for j, (cr, ctr) in enumerate([(0x2000_0000, 1), (0x8000_0000, 2), (0x4000_0000, 0), (0xF000_0000, 0x1_0000_0001)]):
            # The console tests only CTR's low word; QEMU's 64-bit mode tests
            # all of it. Those cases are in hand.txt (bc_ctr_low_word_*).
            if j == 3 and bo & 4 == 0:
                continue
            add(f"bc_{bo}_{bi}_{j}", [bc(bo, bi, 8)] + skip, {"cr": cr, "ctr": ctr})
    for i, ctr in enumerate([1, 2, 0, 0x1_0000_0000]):
        add(f"bdnz_{i}", [bc(16, 0, 8)] + skip, {"ctr": ctr})
        add(f"bdz_{i}", [bc(18, 0, 8)] + skip, {"ctr": ctr})
    add("b_forward", [b(8)] + skip, {})
    # A small counted loop: r3 += r4, ctr times.
    add("loop_bdnz", ["add 3, 3, 4", bc(16, 0, -4)], {"ctr": 5, "r4": 3})
    add("loop_cmp", ["addi 3, 3, 1", "cmpwi 3, 10", bc(12, 0, -8)], {"r3": 2})


# ---------------------------------------------------------------- memory

def mem_cases():
    pat = bytes(range(0x80, 0xC0))
    base = SCR + 0x40
    mem = {SCR + 0x20: pat}
    for i, d in enumerate([0, 2, -4, 0x10]):
        for op in ["lbz", "lhz", "lha", "lwz", "lbzu", "lhzu", "lhau", "lwzu"]:
            add(f"{op}_{i}", [f"{op} 3, {d}(4)"], {"r4": base - 0x10, "mem": mem})
        for op in ["ld", "ldu", "lwa"]:
            add(f"{op}_{i}", [f"{op} 3, {d * 2}(4)"], {"r4": base - 0x10, "mem": mem})
        for op in ["stb", "sth", "stw", "stbu", "sthu", "stwu"]:
            add(f"{op}_{i}", [f"{op} 5, {d}(4)"], {"r4": base, "r5": 0x1122_3344_5566_7788, "mem": mem})
        for op in ["std", "stdu"]:
            add(f"{op}_{i}", [f"{op} 5, {d * 2}(4)"], {"r4": base, "r5": 0x1122_3344_5566_7788, "mem": mem})
    for i, off in enumerate([0, 3, 8]):
        for op in ["lbzx", "lhzx", "lhax", "lwzx", "ldx", "lwax", "lbzux", "lhzux", "lhaux", "lwzux", "ldux", "lwaux",
                   "lhbrx", "lwbrx"]:
            add(f"{op}_{i}", [f"{op} 3, 4, 5"], {"r4": SCR + 0x20, "r5": off * 2, "mem": mem})
        for op in ["stbx", "sthx", "stwx", "stdx", "stbux", "sthux", "stwux", "stdux", "sthbrx", "stwbrx"]:
            add(f"{op}_{i}", [f"{op} 6, 4, 5"], {"r4": SCR + 0x20, "r5": off * 2, "r6": 0x8899_AABB_CCDD_EEFF, "mem": mem})
    add("lbzx_ra0", ["lbzx 3, 0, 5"], {"r5": SCR + 0x21, "mem": mem})
    for i, (a, c) in enumerate([(0, 0), (8, 8), (0, 4)]):
        add(f"lwarx_stwcx_{i}", ["lwarx 3, 4, 5", "addi 3, 3, 1", "stwcx. 3, 4, 6"],
            {"r4": SCR + 0x20, "r5": a, "r6": c, "mem": mem}, flags="no_reserve")
        add(f"ldarx_stdcx_{i}", ["ldarx 3, 4, 5", "addi 3, 3, 1", "stdcx. 3, 4, 6"],
            {"r4": SCR + 0x20, "r5": a, "r6": c, "mem": mem}, flags="no_reserve")
    # dcbz is a hand case: the reference CPU's cache block is 128 bytes, the console's is 32.
    for op in ["dcbt 4, 5", "dcbtst 4, 5", "dcbf 4, 5", "dcbst 4, 5", "icbi 4, 5", "sync", "eieio", "isync"]:
        add(f"nop_{op.split()[0]}", [op], {"r4": SCR, "r5": 0})


# ---------------------------------------------------------------- scalar floating point

def fp_cases():
    two = ["fadd", "fsub", "fmul", "fdiv", "fadds", "fsubs", "fmuls", "fdivs"]
    for op in two:
        pairs = [(R.choice(DV), R.choice(DV)) for _ in range(10)] + [(D["inf"], D["inf"]), (D["zero"], D["zero"])]
        for i, (x, y) in enumerate(pairs):
            add(f"{op}_{i}", [f"{op} 1, 2, 3"], {"f2": x, "f3": y}, flags="ignore_fpscr")
    for op in ["fmadd", "fmsub", "fnmadd", "fnmsub", "fmadds", "fmsubs", "fnmadds", "fnmsubs"]:
        for i in range(10):
            add(f"{op}_{i}", [f"{op} 1, 2, 3, 4"], {"f2": R.choice(DV), "f3": R.choice(DV), "f4": R.choice(DV)},
                flags="ignore_fpscr")
        add(f"{op}_alias", [f"{op} 2, 2, 3, 2"], {"f2": D["pi"], "f3": D["third"]}, flags="ignore_fpscr")
    for op in ["fsqrt", "fsqrts", "frsp", "fres", "frsqrte", "fctiw", "fctiwz", "fctid", "fctidz", "fneg", "fabs",
               "fnabs", "fmr"]:
        for i, x in enumerate(DV):
            if op in ("fres", "frsqrte") and x in (D["third"], D["pi"], D["big"], D["tiny"], D["small_s"]):
                continue  # estimates: the hardware value is not exact
            if op == "fres" and x in (D["zero"], D["nzero"]):
                continue  # the reference returns 0.5 here; a hand case checks infinity
            add(f"{op}_{i}", [f"{op} 1, 2"], {"f2": x}, flags="ignore_fpscr")
    for i, x in enumerate([0, 1, M64, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, 1 << 53 | 1, 12345]):
        add(f"fcfid_{i}", ["fcfid 1, 2"], {"f2": x}, flags="ignore_fpscr")
    for i, (x, y, z) in enumerate([(R.choice(DV), R.choice(DV), R.choice(DV)) for _ in range(8)]):
        add(f"fsel_{i}", ["fsel 1, 2, 3, 4"], {"f2": x, "f3": y, "f4": z}, flags="ignore_fpscr")
    for i, rn in enumerate([1, 2, 3]):
        for x in [D["m3_5"], D["half"], D["e31"], D["m31"]]:
            add(f"fctiw_rn{rn}_{x:x}", ["fctiw 1, 2"], {"f2": x, "fpscr": rn}, flags="ignore_fpscr")
            add(f"fctid_rn{rn}_{x:x}", ["fctid 1, 2"], {"f2": x, "fpscr": rn}, flags="ignore_fpscr")
    for i, (x, y) in enumerate([(R.choice(DV), R.choice(DV)) for _ in range(10)] + [(D["one"], D["one"]), (D["nzero"], 0)]):
        if x in (D["snan"],) or y in (D["snan"],):
            continue
        add(f"fcmpu_{i}", ["fcmpu 3, 2, 3"], {"f2": x, "f3": y})
        add(f"fcmpo_{i}", ["fcmpo 0, 2, 3"], {"f2": x, "f3": y}, flags="ignore_fpscr")
    add("mffs", ["mffs 1"], {"fpscr": 0x0000_8003})
    add("mtfsf", ["mtfsf 0x03, 2"], {"f2": 0x0000_0000_0000_00F2})
    add("mtfsfi", ["mtfsfi 7, 3"], {"fpscr": 0x0000_00F0})
    add("mtfsb1", ["mtfsb1 31"], {})
    add("mtfsb0", ["mtfsb0 30"], {"fpscr": 3})
    fmem = {SCR + 0x20: bytes.fromhex("3F800000 7F800001 00000001 C0490FDB 4009 21FB 5444 2D18 7FF4000000000000")}
    for op, d in [("lfs", 0), ("lfs", 4), ("lfs", 8), ("lfs", 12), ("lfd", 16), ("lfd", 24), ("lfsu", 4), ("lfdu", 16)]:
        add(f"{op}_{d}", [f"{op} 1, {d}(4)"], {"r4": SCR + 0x20, "mem": fmem})
    for op in ["lfsx", "lfdx", "lfsux", "lfdux"]:
        add(op, [f"{op} 1, 4, 5"], {"r4": SCR + 0x20, "r5": 16 if "d" in op[2:] else 4, "mem": fmem})
    for i, x in enumerate(DV):
        add(f"stfs_{i}", ["stfs 2, 0(4)"], {"r4": SCR + 0x40, "f2": x})
        add(f"stfd_{i}", ["stfd 2, 8(4)"], {"r4": SCR + 0x40, "f2": x})
    for op in ["stfsu", "stfdu"]:
        add(op, [f"{op} 2, 16(4)"], {"r4": SCR + 0x40, "f2": D["pi"]})
    for op in ["stfsx", "stfdx", "stfsux", "stfdux", "stfiwx"]:
        add(op, [f"{op} 2, 4, 5"], {"r4": SCR + 0x40, "r5": 8, "f2": D["pi"]})


# ---------------------------------------------------------------- vector (VMX)

VBIN_INT = [
    "vaddubm", "vadduhm", "vadduwm", "vsububm", "vsubuhm", "vsubuwm", "vaddcuw", "vsubcuw",
    "vaddubs", "vadduhs", "vadduws", "vaddsbs", "vaddshs", "vaddsws",
    "vsububs", "vsubuhs", "vsubuws", "vsubsbs", "vsubshs", "vsubsws",
    "vmaxub", "vmaxuh", "vmaxuw", "vmaxsb", "vmaxsh", "vmaxsw", "vminub", "vminuh", "vminuw", "vminsb", "vminsh",
    "vminsw", "vavgub", "vavguh", "vavguw", "vavgsb", "vavgsh", "vavgsw",
    "vrlb", "vrlh", "vrlw", "vslb", "vslh", "vslw", "vsrb", "vsrh", "vsrw", "vsrab", "vsrah", "vsraw",
    "vand", "vandc", "vor", "vnor", "vxor",
    "vmrghb", "vmrghh", "vmrghw", "vmrglb", "vmrglh", "vmrglw",
    "vpkuhum", "vpkuwum", "vpkuhus", "vpkuwus", "vpkshus", "vpkswus", "vpkshss", "vpkswss", "vpkpx",
    "vmuleub", "vmuloub", "vmulesb", "vmulosb", "vmuleuh", "vmulouh", "vmulesh", "vmulosh",
    "vsum4ubs", "vsum4sbs", "vsum4shs", "vsum2sws", "vsumsws",
]
VCMP_INT = ["vcmpequb", "vcmpequh", "vcmpequw", "vcmpgtub", "vcmpgtuh", "vcmpgtuw", "vcmpgtsb", "vcmpgtsh", "vcmpgtsw"]
VBIN_FP = ["vaddfp", "vsubfp", "vmaxfp", "vminfp"]
VCMP_FP = ["vcmpeqfp", "vcmpgefp", "vcmpgtfp", "vcmpbfp"]
VUN_INT = ["vupkhsb", "vupklsb", "vupkhsh", "vupklsh", "vupkhpx", "vupklpx"]
# vexptefp and vlogefp are hand cases: the reference's 2^x and log2 are not
# exact even where the true answer is.
VUN_FP = ["vrfin", "vrfiz", "vrfip", "vrfim", "vrefp", "vrsqrtefp"]
VTRI = ["vmhaddshs", "vmhraddshs", "vmladduhm", "vmsumubm", "vmsummbm", "vmsumuhm", "vmsumuhs", "vmsumshm",
        "vmsumshs", "vsel", "vperm"]


def exact_estimate_ok(w, op):
    """Inputs whose estimate the reference computes exactly (powers of two)."""
    import math
    import struct
    f = struct.unpack(">f", struct.pack(">I", w))[0]
    if math.isnan(f) or math.isinf(f) or f == 0:
        return True
    m, _ = math.frexp(abs(f))
    if op == "vrsqrtefp":
        return m == 0.5 and f > 0 and (math.frexp(f)[1] % 2 == 1)
    if op in ("vexptefp",):
        return f == int(f) and abs(f) < 100
    return m == 0.5


def vec_cases():
    def three(op, gen, n=6, rc=False, flags=""):
        forms = [op] + ([op + "."] if rc else [])
        for f in forms:
            for i in range(n):
                a, b = gen(), gen()
                add(f"{nm(f)}_{i}", [f"{f} 1, 2, 3"], {"v1": gen(), "v2": a, "v3": b}, flags)
            a, b = gen(), gen()
            add(f"{nm(f)}_d_eq_a", [f"{f} 2, 2, 3"], {"v2": a, "v3": b}, flags)
            add(f"{nm(f)}_d_eq_b", [f"{f} 3, 2, 3"], {"v2": a, "v3": b}, flags)
            add(f"{nm(f)}_all_same", [f"{f} 2, 2, 2"], {"v2": a}, flags)

    for op in VBIN_INT:
        three(op, ivec)
    for op in VCMP_INT:
        three(op, ivec, rc=True)
        add(f"{op}_rc_alltrue", [f"{op}. 1, 2, 2"], {"v2": ivec()})
    for op in VBIN_FP:
        three(op, fvec, n=8)
    for op in VCMP_FP:
        three(op, fvec, n=8, rc=True)
        add(f"{op}_rc_alltrue", [f"{op}. 1, 2, 3"], {"v2": vec([0x3F80_0000] * 4), "v3": vec([0x4000_0000] * 4)})
    for op in ["vsl", "vsr", "vslo", "vsro"]:
        for i in range(6):
            sh = R.randrange(128)
            add(f"{op}_{i}", [f"{op} 1, 2, 3"], {"v2": ivec(), "v3": int.from_bytes(bytes([sh] * 16), "big")})
        add(f"{op}_d_eq_a", [f"{op} 2, 2, 3"], {"v2": ivec(), "v3": int.from_bytes(bytes([0x2B] * 16), "big")})
    for op in VUN_INT:
        for i in range(4):
            add(f"{op}_{i}", [f"{op} 1, 2"], {"v2": ivec()})
        add(f"{op}_d_eq_b", [f"{op} 2, 2"], {"v2": ivec()})
    for op in VUN_FP:
        n = 0
        for i in range(40):
            v = fvec()
            words = [(v >> s) & 0xFFFF_FFFF for s in (96, 64, 32, 0)]
            if op in ("vrefp", "vrsqrtefp", "vexptefp", "vlogefp") and not all(exact_estimate_ok(w, op) for w in words):
                continue
            add(f"{op}_{n}", [f"{op} 1, 2"], {"v2": v})
            n += 1
            if n == 5:
                break
        add(f"{op}_d_eq_b", [f"{op} 2, 2"], {"v2": vec([0x4080_0000, 0x3F80_0000, 0x4100_0000, 0x7FC0_0000])})
    for op in VTRI:
        gen = ivec
        for i in range(5):
            add(f"{op}_{i}", [f"{op} 1, 2, 3, 4"], {"v2": gen(), "v3": gen(), "v4": gen() if op != "vperm" else R.getrandbits(128)})
        a, b, c = gen(), gen(), gen()
        add(f"{op}_d_eq_a", [f"{op} 2, 2, 3, 4"], {"v2": a, "v3": b, "v4": c})
        add(f"{op}_d_eq_b", [f"{op} 3, 2, 3, 4"], {"v2": a, "v3": b, "v4": c})
        add(f"{op}_d_eq_c", [f"{op} 4, 2, 3, 4"], {"v2": a, "v3": b, "v4": c})
    for op in ["vmaddfp", "vnmsubfp"]:
        for i in range(10):
            add(f"{op}_{i}", [f"{op} 1, 2, 3, 4"], {"v2": fvec(), "v3": fvec(), "v4": fvec()})
        a, b, c = fvec(), fvec(), fvec()
        for k, line in enumerate([f"{op} 2, 2, 3, 4", f"{op} 3, 2, 3, 4", f"{op} 4, 2, 3, 4"]):
            add(f"{op}_alias_{k}", [line], {"v2": a, "v3": b, "v4": c})
    for sh in [0, 1, 7, 15]:
        add(f"vsldoi_{sh}", [f"vsldoi 1, 2, 3, {sh}"], {"v2": ivec(), "v3": ivec()})
        add(f"vsldoi_{sh}_d_eq_b", [f"vsldoi 3, 2, 3, {sh}"], {"v2": ivec(), "v3": ivec()})
    for op, lim in [("vspltb", 16), ("vsplth", 8), ("vspltw", 4)]:
        for u in [0, 1, lim - 1]:
            add(f"{op}_{u}", [f"{op} 1, 2, {u}"], {"v2": ivec()})
        add(f"{op}_d_eq_b", [f"{op} 2, 2, 1"], {"v2": ivec()})
    for op in ["vspltisb", "vspltish", "vspltisw"]:
        for s in [-16, -1, 0, 5, 15]:
            add(f"{op}_{s & 31}", [f"{op} 1, {s}"], {"v1": ivec()})
    for op in ["vcfux", "vcfsx"]:
        for u in [0, 1, 31]:
            add(f"{op}_{u}", [f"{op} 1, 2, {u}"], {"v2": ivec()})
        add(f"{op}_d_eq_b", [f"{op} 2, 2, 3"], {"v2": ivec()})
    for op in ["vctuxs", "vctsxs"]:
        for u in [0, 1, 31]:
            for i in range(3):
                add(f"{op}_{u}_{i}", [f"{op} 1, 2, {u}"], {"v2": fvec()})
        add(f"{op}_d_eq_b", [f"{op} 2, 2, 3"], {"v2": fvec()})
    add("mfvscr", ["mfvscr 1"], {"vscr": 0x0001_0001, "v1": ivec()})
    add("mtvscr", ["mtvscr 2"], {"v2": vec([1, 2, 3, 0x0001_0001])})
    for i in range(16):
        add(f"lvsl_{i}", ["lvsl 1, 4, 5"], {"r4": SCR, "r5": i})
        add(f"lvsr_{i}", ["lvsr 1, 4, 5"], {"r4": SCR, "r5": i})
    vmem = {SCR + 0x20: bytes(range(0x40, 0x80))}
    def quad(off):
        """The aligned quadword holding SCR + 0x20 + off, as a register value."""
        o = (off & ~15)
        return int.from_bytes(vmem[SCR + 0x20][o:o + 16], "big")

    for off in [0, 4, 7, 0x10, 0x1F]:
        for op in ["lvx", "lvxl"]:
            add(f"{op}_{off}", [f"{op} 1, 4, 5"], {"r4": SCR + 0x20, "r5": off, "mem": vmem, "v1": ivec()})
        # The element loads leave the other elements undefined. We load the
        # whole quadword; start the register with that quadword so only the
        # addressed element is really checked here.
        for op in ["lvebx", "lvehx", "lvewx"]:
            add(f"{op}_{off}", [f"{op} 1, 4, 5"], {"r4": SCR + 0x20, "r5": off, "mem": vmem, "v1": quad(off)})
        for op in ["stvx", "stvxl", "stvebx", "stvehx", "stvewx"]:
            add(f"{op}_{off}", [f"{op} 1, 4, 5"], {"r4": SCR + 0x20, "r5": off, "mem": vmem, "v1": ivec()})
    add("lvx_ra0", ["lvx 1, 0, 5"], {"r5": SCR + 0x24, "mem": vmem})
    for i in range(4):
        add(f"dss_{i}", ["dss 0"], {})


# ---------------------------------------------------------------- VMX128 by equivalence

def enc_vx128(op_major, xo, d, a, b):
    return (op_major << 26) | (d & 31) << 21 | (a & 31) << 16 | (b & 31) << 11 | ((a >> 6) & 1) << 10 \
        | xo | ((a >> 5) & 1) << 5 | ((d >> 5) & 3) << 2 | ((b >> 5) & 3)


def enc_vx128_3(xo, d, b, imm):
    return (6 << 26) | (d & 31) << 21 | (imm & 31) << 16 | (b & 31) << 11 | xo | ((d >> 5) & 3) << 2 | ((b >> 5) & 3)


def enc_vx128_1(xo, d, ra, rb):
    return (4 << 26) | (d & 31) << 21 | (ra & 31) << 16 | (rb & 31) << 11 | xo | ((d >> 5) & 3) << 2


def enc_vx128_5(d, a, b, sh):
    return (4 << 26) | (d & 31) << 21 | (a & 31) << 16 | (b & 31) << 11 | ((a >> 6) & 1) << 10 | (sh & 15) << 6 \
        | 0x10 | ((a >> 5) & 1) << 5 | ((d >> 5) & 3) << 2 | ((b >> 5) & 3)


def enc_vperm128(d, a, b, c):
    return (5 << 26) | (d & 31) << 21 | (a & 31) << 16 | (b & 31) << 11 | ((a >> 6) & 1) << 10 | (c & 7) << 6 \
        | ((a >> 5) & 1) << 5 | ((d >> 5) & 3) << 2 | ((b >> 5) & 3)


VX128_EQ = [
    # (name, major, xo, standard form taking (d, a, b) -> asm, operand kind)
    ("vaddfp128", 5, 0x010, lambda d, a, b: f"vaddfp {d}, {a}, {b}", "f"),
    ("vsubfp128", 5, 0x050, lambda d, a, b: f"vsubfp {d}, {a}, {b}", "f"),
    ("vmaddfp128", 5, 0x0D0, lambda d, a, b: f"vmaddfp {d}, {a}, {b}, {d}", "f"),
    ("vmaddcfp128", 5, 0x110, lambda d, a, b: f"vmaddfp {d}, {a}, {d}, {b}", "f"),
    ("vnmsubfp128", 5, 0x150, lambda d, a, b: f"vnmsubfp {d}, {a}, {b}, {d}", "f"),
    ("vpkshss128", 5, 0x200, lambda d, a, b: f"vpkshss {d}, {a}, {b}", "i"),
    ("vand128", 5, 0x210, lambda d, a, b: f"vand {d}, {a}, {b}", "i"),
    ("vpkshus128", 5, 0x240, lambda d, a, b: f"vpkshus {d}, {a}, {b}", "i"),
    ("vandc128", 5, 0x250, lambda d, a, b: f"vandc {d}, {a}, {b}", "i"),
    ("vpkswss128", 5, 0x280, lambda d, a, b: f"vpkswss {d}, {a}, {b}", "i"),
    ("vnor128", 5, 0x290, lambda d, a, b: f"vnor {d}, {a}, {b}", "i"),
    ("vpkswus128", 5, 0x2C0, lambda d, a, b: f"vpkswus {d}, {a}, {b}", "i"),
    ("vor128", 5, 0x2D0, lambda d, a, b: f"vor {d}, {a}, {b}", "i"),
    ("vpkuhum128", 5, 0x300, lambda d, a, b: f"vpkuhum {d}, {a}, {b}", "i"),
    ("vxor128", 5, 0x310, lambda d, a, b: f"vxor {d}, {a}, {b}", "i"),
    ("vpkuhus128", 5, 0x340, lambda d, a, b: f"vpkuhus {d}, {a}, {b}", "i"),
    ("vsel128", 5, 0x350, lambda d, a, b: f"vsel {d}, {a}, {b}, {d}", "i"),
    ("vpkuwum128", 5, 0x380, lambda d, a, b: f"vpkuwum {d}, {a}, {b}", "i"),
    ("vslo128", 5, 0x390, lambda d, a, b: f"vslo {d}, {a}, {b}", "i"),
    ("vpkuwus128", 5, 0x3C0, lambda d, a, b: f"vpkuwus {d}, {a}, {b}", "i"),
    ("vsro128", 5, 0x3D0, lambda d, a, b: f"vsro {d}, {a}, {b}", "i"),
    ("vcmpeqfp128", 6, 0x000, lambda d, a, b: f"vcmpeqfp {d}, {a}, {b}", "f"),
    ("vcmpeqfp128.", 6, 0x040, lambda d, a, b: f"vcmpeqfp. {d}, {a}, {b}", "f"),
    ("vcmpgefp128", 6, 0x080, lambda d, a, b: f"vcmpgefp {d}, {a}, {b}", "f"),
    ("vcmpgefp128.", 6, 0x0C0, lambda d, a, b: f"vcmpgefp. {d}, {a}, {b}", "f"),
    ("vcmpgtfp128", 6, 0x100, lambda d, a, b: f"vcmpgtfp {d}, {a}, {b}", "f"),
    ("vcmpgtfp128.", 6, 0x140, lambda d, a, b: f"vcmpgtfp. {d}, {a}, {b}", "f"),
    ("vcmpbfp128", 6, 0x180, lambda d, a, b: f"vcmpbfp {d}, {a}, {b}", "f"),
    ("vcmpbfp128.", 6, 0x1C0, lambda d, a, b: f"vcmpbfp. {d}, {a}, {b}", "f"),
    ("vcmpequw128", 6, 0x200, lambda d, a, b: f"vcmpequw {d}, {a}, {b}", "i"),
    ("vcmpequw128.", 6, 0x240, lambda d, a, b: f"vcmpequw. {d}, {a}, {b}", "i"),
    ("vrlw128", 6, 0x050, lambda d, a, b: f"vrlw {d}, {a}, {b}", "i"),
    ("vslw128", 6, 0x0D0, lambda d, a, b: f"vslw {d}, {a}, {b}", "i"),
    ("vsraw128", 6, 0x150, lambda d, a, b: f"vsraw {d}, {a}, {b}", "i"),
    ("vsrw128", 6, 0x1D0, lambda d, a, b: f"vsrw {d}, {a}, {b}", "i"),
    ("vmaxfp128", 6, 0x280, lambda d, a, b: f"vmaxfp {d}, {a}, {b}", "f"),
    ("vminfp128", 6, 0x2C0, lambda d, a, b: f"vminfp {d}, {a}, {b}", "f"),
    ("vmrghw128", 6, 0x300, lambda d, a, b: f"vmrghw {d}, {a}, {b}", "i"),
    ("vmrglw128", 6, 0x340, lambda d, a, b: f"vmrglw {d}, {a}, {b}", "i"),
]

VX128_UN = [
    ("vupkhsb128", 6, 0x380, "vupkhsb", "i"),
    ("vupklsb128", 6, 0x3C0, "vupklsb", "i"),
]

VX128_3_EQ = [
    # name, xo, standard mnemonic, takes immediate, kind
    ("vcfpsxws128", 0x230, "vctsxs", True, "f"),
    ("vcfpuxws128", 0x270, "vctuxs", True, "f"),
    ("vcsxwfp128", 0x2B0, "vcfsx", True, "i"),
    ("vcuxwfp128", 0x2F0, "vcfux", True, "i"),
    ("vrfim128", 0x330, "vrfim", False, "f"),
    ("vrfin128", 0x370, "vrfin", False, "f"),
    ("vrfip128", 0x3B0, "vrfip", False, "f"),
    ("vrfiz128", 0x3F0, "vrfiz", False, "f"),
    ("vspltw128", 0x730, "vspltw", True, "i"),
]

# (Rust d, a, b) -> (oracle d, a, b): high registers and the aliasing shapes.
REG_SHAPES = [((70, 33, 127), (1, 2, 3)), ((5, 96, 64), (1, 2, 3)), ((100, 100, 9), (2, 2, 3)),
              ((45, 12, 45), (3, 2, 3)), ((127, 127, 127), (2, 2, 2))]


def vx128_cases():
    for name, major, xo, std, kind in VX128_EQ:
        gen = fvec if kind == "f" else ivec
        for k, ((d, a, b), (od, oa, ob)) in enumerate(REG_SHAPES):
            w = enc_vx128(major, xo, d, a, b)
            vals = {d: gen(), a: gen(), b: gen()}
            inp = {f"v{r}": v for r, v in vals.items()}
            ren = {d: od, a: oa, b: ob}
            add(f"{nm(name)}_{k}", [std(od, oa, ob)], inp, words=[w], rename=ren)
    for name, major, xo, std, kind in VX128_UN:
        for k, ((d, _, b), (od, _, ob)) in enumerate(REG_SHAPES):
            w = enc_vx128(major, xo, d, 0, b)
            vals = {d: ivec(), b: ivec()}
            add(f"{name}_{k}", [f"{std} {od}, {ob}"], {f"v{r}": v for r, v in vals.items()}, words=[w],
                rename={d: od, b: ob})
    for name, xo, std, has_imm, kind in VX128_3_EQ:
        gen = fvec if kind == "f" else ivec
        for k, ((d, _, b), (od, _, ob)) in enumerate(REG_SHAPES[:4]):
            imm = [0, 3, 1, 2][k] if has_imm else 0
            if std == "vspltw":
                imm &= 3
            w = enc_vx128_3(xo, d, b, imm)
            line = f"{std} {od}, {ob}, {imm}" if has_imm else f"{std} {od}, {ob}"
            vals = {d: gen(), b: gen()}
            add(f"{name}_{k}", [line], {f"v{r}": v for r, v in vals.items()}, words=[w], rename={d: od, b: ob})
    for k, s in enumerate([-16, -1, 7, 15]):
        d = [99, 31, 64, 127][k]
        add(f"vspltisw128_{k}", [f"vspltisw 1, {s}"], {f"v{d}": ivec()}, words=[enc_vx128_3(0x770, d, 0, s & 31)],
            rename={d: 1})
    for k, ((d, a, b), (od, oa, ob)) in enumerate(REG_SHAPES):
        sh = [0, 4, 9, 15, 1][k]
        vals = {d: ivec(), a: ivec(), b: ivec()}
        add(f"vsldoi128_{k}", [f"vsldoi {od}, {oa}, {ob}, {sh}"], {f"v{r}": v for r, v in vals.items()},
            words=[enc_vx128_5(d, a, b, sh)], rename={d: od, a: oa, b: ob})
    for k, ((d, a, b), (od, oa, ob)) in enumerate(REG_SHAPES):
        c = [4, 7, 0, 5, 6][k]
        vals = {d: ivec(), a: ivec(), b: ivec(), c: R.getrandbits(128)}
        ren = {d: od, a: oa, b: ob, c: c + 8}
        add(f"vperm128_{k}", [f"vperm {ren[d]}, {ren[a]}, {ren[b]}, {ren[c]}"], {f"v{r}": v for r, v in vals.items()},
            words=[enc_vperm128(d, a, b, c)], rename=ren)
    # vmulfp128: a * b + (-0.0), fused, is exactly the rounded product.
    for k, ((d, a, b), (od, oa, ob)) in enumerate(REG_SHAPES):
        vals = {d: fvec(), a: fvec(), b: fvec()}
        add(f"vmulfp128_{k}", ["vspltisw 31, -1", "vslw 31, 31, 31", f"vmaddfp {od}, {oa}, {ob}, 31"],
            {f"v{r}": v for r, v in vals.items()}, words=[enc_vx128(5, 0x090, d, a, b)], rename={d: od, a: oa, b: ob},
            scrub=["v31"])
    vmem = {SCR + 0x20: bytes(range(0x40, 0x80))}
    mem128 = [("lvx128", 0x0C3, "lvx"), ("lvxl128", 0x2C3, "lvxl"),
              ("stvx128", 0x1C3, "stvx"), ("stvxl128", 0x3C3, "stvxl"), ("stvewx128", 0x183, "stvewx"),
              ("lvsl128", 0x003, "lvsl"), ("lvsr128", 0x043, "lvsr")]
    for name, xo, std in mem128:
        for k, (vd, off) in enumerate([(65, 4), (127, 0x13), (2, 0)]):
            add(f"{name}_{k}", [f"{std} 1, 4, 5"], {"r4": SCR + 0x20, "r5": off, "mem": vmem, f"v{vd}": ivec()},
                words=[enc_vx128_1(xo, vd, 4, 5)], rename={vd: 1})


# Logical and shift ops (primary 31) that write rA rather than rD.
RA_XO = {28, 60, 444, 412, 316, 476, 124, 284, 24, 536, 792, 824, 27, 539, 794, 954, 922, 986, 26, 58}


def console_cr0(word, out):
    """The console's titles see CR0 of an integer record form from the low
    32 bits of the result (the reference CPU compares all 64), so a C
    runtime count stored as -1 and reloaded zero-extended still tests
    negative. Rewrites CR0 in `out` for such an instruction."""
    op = word >> 26
    rd, ra = (word >> 21) & 31, (word >> 16) & 31
    if op == 31 and word & 1:
        xo = (word >> 1) & 0x3FF
        if xo in (0, 32, 150, 214):  # compares and conditional stores
            return
        dest = ra if xo in RA_XO or (word >> 2) & 0x1FF == 413 else rd
    elif op == 13:
        dest = rd
    elif op in (28, 29) or (op in (20, 21, 23, 30) and word & 1):
        dest = ra
    else:
        return
    low = out.get(f"r{dest}", 0) & 0xFFFF_FFFF
    low -= (low >> 31) << 32
    field = 8 if low < 0 else 4 if low > 0 else 2
    field |= (out.get("xer", 0) >> 31) & 1
    out["cr"] = (out["cr"] & 0x0FFF_FFFF) | field << 28


# Carrying ops (primary 31, XO form): xo -> (complement rA, second operand, carry in).
CARRY_XO = {
    10: (False, "rb", 0), 138: (False, "rb", "ca"), 202: (False, 0, "ca"), 234: (False, "ones", "ca"),
    8: (True, "rb", 1), 136: (True, "rb", "ca"), 200: (True, 0, "ca"), 232: (True, "ones", "ca"),
}


def console_mulhw(word, out):
    """mulhw leaves the high word undefined; the reference CPU zero-extends,
    and both reference translators that run titles sign-extend. Rewrites rD
    in `out` to the sign-extended high product."""
    if word >> 26 == 31 and (word >> 1) & 0x1FF == 75:
        rd = (word >> 21) & 31
        v = out.get(f"r{rd}", 0) & 0xFFFF_FFFF
        out[f"r{rd}"] = v | (0xFFFF_FFFF_0000_0000 if v >> 31 else 0)


def console_ca(word, st, out):
    """The console's titles see CA as the carry out of the low 32 bits (the
    reference CPU carries out of all 64), as with CR0. Rewrites CA in `out`
    for addc, adde, addze, addme, subfc, subfe, subfze, subfme, addic(.) and
    subfic, from the input registers in `st`."""
    op = word >> 26
    ra, rb = (word >> 16) & 31, (word >> 11) & 31
    xer = st.get("xer", 0)
    cin = (xer >> 29) & 1
    simm = (word & 0xFFFF) - ((word & 0x8000) << 1)
    if op == 31 and (word >> 1) & 0x1FF in CARRY_XO:
        neg, b, c = CARRY_XO[(word >> 1) & 0x1FF]
        b = {"rb": st.get(f"r{rb}", 0), 0: 0, "ones": M64}[b]
        c = cin if c == "ca" else c
    elif op in (12, 13):
        neg, b, c = False, simm, 0
    elif op == 8:
        neg, b, c = True, simm, 1
    else:
        return
    a = st.get(f"r{ra}", 0) ^ (M64 if neg else 0)
    ca = ((a & 0xFFFF_FFFF) + (b & 0xFFFF_FFFF) + c) >> 32
    out["xer"] = (out.get("xer", xer) & ~0x2000_0000) | ca << 29


def main():
    int_cases()
    branch_cases()
    mem_cases()
    fp_cases()
    vec_cases()
    vx128_cases()
    names = set()
    for c in CASES:
        assert c[0] not in names, c[0]
        names.add(c[0])
    work = os.environ.get("ORACLE_WORK")
    runnable = [(lines, oracle_input(st, ren)) for (_, lines, st, _, _, ren, _) in CASES]
    outs = harness.run(runnable, work)
    std_lines = [l for (_, lines, _, _, words, _, _) in CASES if words is None for l in lines]
    std_words = iter(harness.encode(std_lines))
    rows = []
    for (name, lines, st, flags, words, ren, scrub), out in zip(CASES, outs):
        if words is None:
            words = [next(std_words) for _ in lines]
        if len(words) == 1:
            console_ca(words[0], oracle_input(st, ren), out)
            console_mulhw(words[0], out)
            console_cr0(words[0], out)
        oin = harness.full_state(oracle_input(st, ren))
        diff = {k: v for k, v in out.items() if oin[k] != v and k not in scrub}
        if ren:
            back = {f"v{o}": f"v{r}" for r, o in ren.items()}
            diff = {back.get(k, k): v for k, v in diff.items()}
        rows.append(fmt_row(name, words, st, diff, flags))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "crates", "cpu-tests", "data", "oracle.txt")
    with open(path, "w") as fh:
        fh.write("# Reference results from a 64-bit PowerPC CPU emulator, with CR0 of integer\n")
        fh.write("# record forms and CA from the low 32 bits as on the console; regenerate with\n")
        fh.write("# `python3 tools/ppc-oracle/cases.py`. Format: see crates/cpu-tests/src/lib.rs.\n")
        fh.write("\n".join(rows) + "\n")
    print(f"{len(rows)} cases -> {path}")


def oracle_input(st, ren):
    if not ren:
        return st
    o = {}
    for k, v in st.items():
        if k[0] == "v" and k != "vscr" and int(k[1:]) in ren:
            o[f"v{ren[int(k[1:])]}"] = v
        elif k[0] == "v" and k != "vscr":
            raise ValueError(f"unmapped vector register {k}")
        else:
            o[k] = v
    return o


def fmt_val(k, v):
    if k == "mem":
        raise AssertionError
    if k[0] == "v" and k != "vscr":
        return f"{v:032x}"
    return f"{v:x}"


def fmt_row(name, words, st, diff, flags):
    def fields(d, base_mem=None):
        out = []
        for k in sorted(d, key=sort_key):
            v = d[k]
            if k == "mem":
                continue
            out.append(f"{k}={fmt_val(k, v)}")
        return out

    ins = fields(st)
    for addr, data in sorted(st.get("mem", {}).items()):
        ins.append(f"m{addr:x}={data.hex()}")
    outs = fields(diff)
    if "mem" in diff:
        old = harness.full_state(st)["mem"]
        new = diff["mem"]
        i = 0
        while i < len(new):
            if new[i] != old[i]:
                j = i
                while j < len(new) and new[j] != old[j]:
                    j += 1
                outs.append(f"m{harness.SCRATCH + i:x}={new[i:j].hex()}")
                i = j
            else:
                i += 1
    return f"{name} | {' '.join(f'{w:08x}' for w in words)} | {' '.join(ins)} | {' '.join(outs)} | {flags}"


def sort_key(k):
    order = {"r": 0, "f": 1, "v": 2}
    if k[0] in order and k[1:].isdigit():
        return (order[k[0]], int(k[1:]))
    return (3, k)


if __name__ == "__main__":
    main()
