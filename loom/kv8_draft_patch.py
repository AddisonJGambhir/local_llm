#!/usr/bin/env python3
"""int8 KV for the DSpark drafter's paged K/V (run after kv8_patch.py).

Same q8_0-style layout as the target planes: per position and plane 8 heads x 128 int8 (1024 B) + 8 x 4 f16 scales (64 B)
= 1088 B instead of 2048 B. Two kernels touch the drafter cache:
  g_dsp_kv_store_paged   lane l writes dims l, l+32, l+64, l+96 of head kh as scalar f16 stores
                         -> block b (dims 32b..32b+31) spans all 32 lanes: scale = subgroup max / 127, lane 0 stores it
  g_dsp_attn_split_full  vector<4xf16> loads from view<8x32x4xf16> -> int8 load + per-32 scale (same as kv8_patch.py)
storage.loom: draft_pos_bytes 2048 -> 1088, draft_block_bytes 32768 -> 17408.

usage: kv8_draft_patch.py <kv8 llm.loom> <out llm.loom> <kv8 storage.loom> <out storage.loom>
"""
import re
import sys

src, dst, ssrc, sdst = sys.argv[1:5]
L = open(src).read().split("\n")
assert any("@g_kv_offset8(" in x for x in L), "run kv8_patch.py first (no @g_kv_offset8)"
defs = [i for i, x in enumerate(L) if re.match(r"\s*(kernel|func|command)\.def", x)]
uid = [0]


def u(base):
    uid[0] += 1
    return f"%kd8_{base}_{uid[0]}"


def span(name):
    a = [i for i in defs if f'export("{name}")' in L[i]]
    assert len(a) == 1, name
    return a[0], min(i for i in defs if i > a[0])


VIEW128 = re.compile(r"^(\s*)(%\w+) = buffer\.view %arena\[(%\w+)\] : buffer -> view<8x128xf16>$")
SSTORE = re.compile(r"^(\s*)view\.store (%\w+), (%\w+)\[(%\w+), (%\w+)\] : f16, view<8x128xf16>$")
VIEW4 = re.compile(r"^(\s*)(%\w+) = buffer\.view %arena\[(%\w+)\] : buffer -> view<8x32x4xf16>$")
LOAD4 = re.compile(r"^(\s*)(%\w+) = vector\.load (%\w+)\[([^\]]*)\] : view<8x32x4xf16> -> vector<4xf16>$")
n = {"view128": 0, "sstore": 0, "view4": 0, "load4": 0}


def common(line):
    line = line.replace("%pos_bytes = index.constant 2048 : offset", "%pos_bytes = index.constant 1088 : offset")
    return line.replace("func.call @g_kv_offset(", "func.call @g_kv_offset8(")


def patch_store(a, b):
    out, scales = [], {}
    for line in L[a:b]:
        m = VIEW128.match(line)
        if m:
            ind, name, off = m.groups()
            c, so, sv = u("c"), u("so"), u("s")
            out += [f"{ind}{name} = buffer.view %arena[{off}] : buffer -> view<8x128xi8>",
                    f"{ind}{c} = index.constant 1024 : offset",
                    f"{ind}{so} = index.add {off}, {c} : offset",
                    f"{ind}{sv} = buffer.view %arena[{so}] : buffer -> view<8x4xf16>"]
            scales[name] = sv; n["view128"] += 1
            continue
        m = SSTORE.match(line)
        if m:
            ind, val, view, head, dim = m.groups()
            assert view in scales, line
            r = {k: u(k) for k in "f a m k127 d dh dq one z inv nz invs v1 sc rd q qs c32 b bb lane c0 lead".split()}
            out += [
                f"{ind}{r['f']} = scalar.extf {val} : f16 to f32",
                f"{ind}{r['a']} = scalar.absf {r['f']} : f32",
                f"{ind}{r['m']} = kernel.subgroup.reduce<maxnumf> {r['a']} : f32",   # block = this dim index across all lanes
                f"{ind}{r['k127']} = scalar.constant 127.0 : f32",
                f"{ind}{r['d']} = scalar.divf {r['m']}, {r['k127']} : f32",
                f"{ind}{r['dh']} = scalar.fptrunc {r['d']} : f32 to f16",
                f"{ind}{r['dq']} = scalar.extf {r['dh']} : f16 to f32",
                f"{ind}{r['one']} = scalar.constant 1.0 : f32",
                f"{ind}{r['z']} = scalar.constant 0.0 : f32",
                f"{ind}{r['inv']} = scalar.divf {r['one']}, {r['dq']} : f32",
                f"{ind}{r['nz']} = scalar.cmpf one, {r['dq']}, {r['z']} : f32",
                f"{ind}{r['invs']} = scf.select {r['nz']}, {r['inv']}, {r['z']} : f32",
                f"{ind}{r['sc']} = scalar.mulf {r['f']}, {r['invs']} : f32",
                f"{ind}{r['v1']} = vector.from_elements {r['sc']} : vector<1xf32>",
                f"{ind}{r['rd']} = vector.roundf {r['v1']} : vector<1xf32>",
                f"{ind}{r['q']} = vector.fptosi {r['rd']} : vector<1xf32> to vector<1xi8>",
                f"{ind}{r['qs']} = vector.extract {r['q']}[0] : vector<1xi8> -> i8",
                f"{ind}view.store {r['qs']}, {view}[{head}, {dim}] : i8, view<8x128xi8>",
                f"{ind}{r['c32']} = index.constant 32 : index",
                f"{ind}{r['b']} = index.div {dim}, {r['c32']} : index",
                f"{ind}{r['bb']} = index.assume {r['b']} [range({r['b']}, 0, 3)] : index",
                f"{ind}{r['lane']} = kernel.subgroup.lane.id : index",
                f"{ind}{r['c0']} = index.constant 0 : index",
                f"{ind}{r['lead']} = index.cmp eq, {r['lane']}, {r['c0']} : index",
                f"{ind}scf.if {r['lead']} {{",
                f"{ind}  view.store {r['dh']}, {scales[view]}[{head}, {r['bb']}] : f16, view<8x4xf16>",
                f"{ind}}}",
            ]
            n["sstore"] += 1
            continue
        if "%arena" in line and "xf16>" in line:
            raise SystemExit("unhandled: " + line.strip())
        out.append(common(line))
    return out


def patch_reader(a, b):
    out, scales = [], {}
    for line in L[a:b]:
        m = VIEW4.match(line)
        if m:
            ind, name, off = m.groups()
            c, so, sv = u("c"), u("so"), u("s")
            out += [f"{ind}{name} = buffer.view %arena[{off}] : buffer -> view<8x32x4xi8>",
                    f"{ind}{c} = index.constant 1024 : offset",
                    f"{ind}{so} = index.add {off}, {c} : offset",
                    f"{ind}{sv} = buffer.view %arena[{so}] : buffer -> view<8x4xf16>"]
            scales[name] = sv; n["view4"] += 1
            continue
        m = LOAD4.match(line)
        if m:
            ind, res, view, idx = m.groups()
            assert view in scales, line
            head, lane = [p.strip() for p in idx.split(",")][:2]
            q, cd, bv, bb, s16, s, f, sp, d = (u(x) for x in "q cd bv bb s16 s f sp d".split())
            out += [f"{ind}{q} = vector.load {view}[{idx}] : view<8x32x4xi8> -> vector<4xi8>",
                    f"{ind}{cd} = index.constant 8 : index",
                    f"{ind}{bv} = index.div {lane}, {cd} : index",
                    f"{ind}{bb} = index.assume {bv} [range({bv}, 0, 3)] : index",
                    f"{ind}{s16} = view.load {scales[view]}[{head}, {bb}] : view<8x4xf16> -> f16",
                    f"{ind}{s} = scalar.extf {s16} : f16 to f32",
                    f"{ind}{f} = vector.sitofp {q} : vector<4xi8> to vector<4xf32>",
                    f"{ind}{sp} = vector.splat {s} : vector<4xf32>",
                    f"{ind}{d} = vector.mulf {f}, {sp} : vector<4xf32>",
                    f"{ind}{res} = vector.fptrunc {d} : vector<4xf32> to vector<4xf16>"]
            n["load4"] += 1
            continue
        if "%arena" in line and "xf16>" in line:
            raise SystemExit("unhandled: " + line.strip())
        out.append(common(line))
    return out


(sa, sb), (ra, rb) = span("g_dsp_kv_store_paged"), span("g_dsp_attn_split_full")
jobs = sorted([(sa, sb, patch_store), (ra, rb, patch_reader)])
res, prev = [], 0
for a, b, fn in jobs:
    res += L[prev:a]
    body = fn(a, b)
    assert not any("2048 : offset" in x for x in body), "leftover 2048 stride"
    res += body
    prev = b
res += L[prev:]
assert n == {"view128": 2, "sstore": 8, "view4": 2, "load4": 2}, n
open(dst, "w").write("\n".join(res))

s = open(ssrc).read()
for old, new in [("%draft_pos_bytes = scalar.constant 2048 : i64", "%draft_pos_bytes = scalar.constant 1088 : i64"),
                 ("%draft_block_bytes = scalar.constant 32768 : i64", "%draft_block_bytes = scalar.constant 17408 : i64")]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(sdst, "w").write(s)
print("ok", n)
