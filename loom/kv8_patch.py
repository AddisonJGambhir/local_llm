#!/usr/bin/env python3
"""int8 (q8_0-style) paged KV for the Inkling-2-Mini Loom package.

Per position and plane: 8 heads x 128 int8 (1024 B) + 8 heads x 4 blocks of 32 f16 scales (64 B) = 1088 B
(f16 was 2048 B). Only the target model's paged planes change; the sliding-window ring and the drafter stay f16.
Readers dequantize to the same f16 values the rest of each kernel already consumes.

usage: kv8_patch.py <f16 llm.loom> <out llm.loom> <f16 storage.loom> <out storage.loom>
"""
import re
import sys

STRIDE = 1088
SCALE_AT = 1024
TARGET = [
    "g_ink_kv_prep_store", "g_ink_kv_prep_store_lv", "g_ink_sconv1_dyn_kv_prep_store",
    "g_ink_attn_split", "g_ink_attn_direct", "g_ink_attn_split_span", "g16_ink_attn_split_listed",
    "g_ink_attention_flash2_ts2", "g_ink_attention_flash2_ts2_rdna4",
    "g_ink_attention_flash2_partial", "g_ink_attention_flash2_partial_rdna4",
]
EXPECT = {"view4": 18, "view16": 32, "load4": 12, "load16": 32, "store4": 6}

src, dst, ssrc, sdst = sys.argv[1:5]
L = open(src).read().split("\n")
defs = [i for i, l in enumerate(L) if re.match(r"\s*(kernel|func|command)\.def", l)]
n = {k: 0 for k in EXPECT}
uid = [0]


def u(base):
    uid[0] += 1
    return f"%kv8_{base}_{uid[0]}"


VIEW = re.compile(r"^(\s*)(%\w+) = buffer\.view %arena\[(%\w+)\] : buffer -> view<(8x32x4|8x128)xf16>$")
LOAD = re.compile(r"^(\s*)(%\w+) = vector\.load (%\w+)\[([^\]]*)\] : view<(8x32x4|8x128)xf16> -> vector<(4|16)xf16>$")
STORE = re.compile(r"^(\s*)vector\.store (%\w+), (%\w+)\[(%\w+), (%\w+), (%\w+)\] : vector<4xf16>, view<8x32x4xf16>$")


def patch_kernel(a, b):
    scales = {}  # view name -> scale view name
    out = []
    for line in L[a:b]:
        m = VIEW.match(line)
        if m:
            ind, name, off, shp = m.groups()
            c, so, sv = u("c"), u("so"), u("s")
            out += [f"{ind}{name} = buffer.view %arena[{off}] : buffer -> view<{shp}xi8>",
                    f"{ind}{c} = index.constant {SCALE_AT} : offset",
                    f"{ind}{so} = index.add {off}, {c} : offset",
                    f"{ind}{sv} = buffer.view %arena[{so}] : buffer -> view<8x4xf16>"]
            scales[name] = sv
            n["view4" if shp == "8x32x4" else "view16"] += 1
            continue
        m = LOAD.match(line)
        if m:
            ind, res, view, idx, shp, w = m.groups()
            assert view in scales, line
            parts = [p.strip() for p in idx.split(",")]
            head, inner = parts[0], parts[1]  # 8x32x4: [head, lane, 0] (block = lane/8); 8x128: [head, dim] (block = dim/32)
            div = 8 if shp == "8x32x4" else 32
            q, cd, bv, bb, s16, s, f, sp, d = (u(x) for x in "q cd bv bb s16 s f sp d".split())
            out += [f"{ind}{q} = vector.load {view}[{idx}] : view<{shp}xi8> -> vector<{w}xi8>",
                    f"{ind}{cd} = index.constant {div} : index",
                    f"{ind}{bv} = index.div {inner}, {cd} : index",
                    f"{ind}{bb} = index.assume {bv} [range({bv}, 0, 3)] : index",
                    f"{ind}{s16} = view.load {scales[view]}[{head}, {bb}] : view<8x4xf16> -> f16",
                    f"{ind}{s} = scalar.extf {s16} : f16 to f32",
                    f"{ind}{f} = vector.sitofp {q} : vector<{w}xi8> to vector<{w}xf32>",
                    f"{ind}{sp} = vector.splat {s} : vector<{w}xf32>",
                    f"{ind}{d} = vector.mulf {f}, {sp} : vector<{w}xf32>",
                    f"{ind}{res} = vector.fptrunc {d} : vector<{w}xf32> to vector<{w}xf16>"]
            n["load4" if w == "4" else "load16"] += 1
            continue
        m = STORE.match(line)
        if m:
            ind, val, view, head, lane, c0 = m.groups()
            assert view in scales, line
            r = {k: u(k) for k in "f a z m0 w d1 d2 d4 r1 x1 m1 r2 x2 m2 r4 x4 m3 k127 d dh dq one inv nz invs sv sc rd q c8 b bb rem c0i lead".split()}
            out += [
                f"{ind}{r['f']} = vector.extf {val} : vector<4xf16> to vector<4xf32>",
                f"{ind}{r['a']} = vector.absf {r['f']} : vector<4xf32>",
                f"{ind}{r['z']} = scalar.constant 0.0 : f32",
                f"{ind}{r['m0']} = vector.reduce<maxnumf> {r['a']}, {r['z']} : vector<4xf32>, f32",
                f"{ind}{r['w']} = scalar.constant 32 : i32",
                f"{ind}{r['d1']} = scalar.constant 1 : i32",
                f"{ind}{r['d2']} = scalar.constant 2 : i32",
                f"{ind}{r['d4']} = scalar.constant 4 : i32",
                f"{ind}{r['r1']}, {r['x1']} = kernel.subgroup.shuffle<xor> {r['m0']}, {r['d1']}, {r['w']} : f32, i32, i32",
                f"{ind}{r['m1']} = scalar.maxnumf {r['m0']}, {r['r1']} : f32",
                f"{ind}{r['r2']}, {r['x2']} = kernel.subgroup.shuffle<xor> {r['m1']}, {r['d2']}, {r['w']} : f32, i32, i32",
                f"{ind}{r['m2']} = scalar.maxnumf {r['m1']}, {r['r2']} : f32",
                f"{ind}{r['r4']}, {r['x4']} = kernel.subgroup.shuffle<xor> {r['m2']}, {r['d4']}, {r['w']} : f32, i32, i32",
                f"{ind}{r['m3']} = scalar.maxnumf {r['m2']}, {r['r4']} : f32",
                f"{ind}{r['k127']} = scalar.constant 127.0 : f32",
                f"{ind}{r['d']} = scalar.divf {r['m3']}, {r['k127']} : f32",
                f"{ind}{r['dh']} = scalar.fptrunc {r['d']} : f32 to f16",
                f"{ind}{r['dq']} = scalar.extf {r['dh']} : f16 to f32",
                f"{ind}{r['one']} = scalar.constant 1.0 : f32",
                f"{ind}{r['inv']} = scalar.divf {r['one']}, {r['dq']} : f32",
                f"{ind}{r['nz']} = scalar.cmpf one, {r['dq']}, {r['z']} : f32",
                f"{ind}{r['invs']} = scf.select {r['nz']}, {r['inv']}, {r['z']} : f32",
                f"{ind}{r['sv']} = vector.splat {r['invs']} : vector<4xf32>",
                f"{ind}{r['sc']} = vector.mulf {r['f']}, {r['sv']} : vector<4xf32>",
                f"{ind}{r['rd']} = vector.roundf {r['sc']} : vector<4xf32>",
                f"{ind}{r['q']} = vector.fptosi {r['rd']} : vector<4xf32> to vector<4xi8>",
                f"{ind}vector.store {r['q']}, {view}[{head}, {lane}, {c0}] : vector<4xi8>, view<8x32x4xi8>",
                f"{ind}{r['c8']} = index.constant 8 : index",
                f"{ind}{r['b']} = index.div {lane}, {r['c8']} : index",
                f"{ind}{r['bb']} = index.assume {r['b']} [range({r['b']}, 0, 3)] : index",
                f"{ind}{r['rem']} = index.rem {lane}, {r['c8']} : index",
                f"{ind}{r['c0i']} = index.constant 0 : index",
                f"{ind}{r['lead']} = index.cmp eq, {r['rem']}, {r['c0i']} : index",
                f"{ind}scf.if {r['lead']} {{",
                f"{ind}  view.store {r['dh']}, {scales[view]}[{head}, {r['bb']}] : f16, view<8x4xf16>",
                f"{ind}}}",
            ]
            n["store4"] += 1
            continue
        if "xf16>" in line and "%arena" in line:
            raise SystemExit(f"unhandled arena f16 line: {line.strip()}")
        line = line.replace("%pos_bytes = index.constant 2048 : offset", f"%pos_bytes = index.constant {STRIDE} : offset")
        line = line.replace("func.call @g_kv_offset(", "func.call @g_kv_offset8(")
        out.append(line)
    return out


# Every use of a patched view must be one of the three handled shapes.
spans = []
for k in TARGET:
    a = [i for i in defs if f'export("{k}")' in L[i]]
    assert len(a) == 1, k
    a = a[0]
    b = min(i for i in defs if i > a)
    spans.append((a, b))
spans.sort()
res, prev = [], 0
for a, b in spans:
    res += L[prev:a]
    body = patch_kernel(a, b)
    assert not any("2048 : offset" in x for x in body), "leftover 2048 stride"
    for x in body:
        if "%arena" in x:
            assert "xf16>" not in x or "8x4xf16" in x, x
    res += body
    prev = b
res += L[prev:]
assert n == EXPECT, n

# helper with the int8 stride, used only by the target kernels (the drafter keeps @g_kv_offset)
txt = "\n".join(res)
h0 = txt.index("func.def inline @g_kv_offset(")
h1 = txt.index("\n}\n", h0) + 3
helper = txt[h0:h1].replace("@g_kv_offset(", "@g_kv_offset8(").replace(
    "%position_bytes = index.constant 2048 : offset", f"%position_bytes = index.constant {STRIDE} : offset")
assert f"constant {STRIDE}" in helper
txt = txt[:h1] + "\n// int8 paged KV: 1024 B of int8 values + 64 B of per-32 f16 scales per position and plane.\n" + helper + txt[h1:]
open(dst, "w").write(txt)

s = open(ssrc).read()
for old, new in [("%kv_position_bytes = scalar.constant 2048 : i64", f"%kv_position_bytes = scalar.constant {STRIDE} : i64"),
                 ("%block_bytes = scalar.constant 32768 : i64", f"%block_bytes = scalar.constant {16 * STRIDE} : i64")]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
s = s.replace("of pool x 2048 bytes (F16 KV heads x head_dim)", f"of pool x {STRIDE} bytes (int8 KV heads x head_dim + f16 per-32 scales)")
open(sdst, "w").write(s)
print("ok", n, "uids", uid[0])
