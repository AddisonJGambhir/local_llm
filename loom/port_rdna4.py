"""Add RDNA4 (gfx12) twins of the Inkling flash2 attention kernels to a Loom llm.loom and wire per-target dispatch.

Re-implements the documented gfx11->gfx12 flash2 rewrite (loom-skills flash2_rdna4.py, read and reviewed) and the
existing package's template-selection pattern (template.decl @selN_K + target(@rdna3)/(@rdna4) providers).
usage: python3 -I port_rdna4.py SRC_llm.loom DST_llm.loom
"""
import re
import sys

KERNELS = ["g_ink_attention_flash2_ts2", "g_ink_attention_flash2_partial",
           "g_ink_attention_flash2_swa", "g_ink_attention_flash2_partial_swa"]


def to_rdna4(text, name):
    n = {}

    def sub(key, pat, rep, flags=re.M):
        nonlocal text
        text, c = re.subn(pat, rep, text, flags=flags)
        n[key] = c

    sub("lane_half", r"^(  %lane_half = index\.div %lane, %c16 : index\n)",
        r"\1  %r4_c8 = index.constant 8 : index\n  %lane_h8_v = index.mul %lane_half, %r4_c8 : index\n"
        r"  %lane_h8 = index.assume %lane_h8_v [range(%lane_h8_v, 0, 8)] : index\n"
        r"  %lane_h8_i32 = index.cast %lane_h8 : index to i32\n")
    sub("q_view", r"(%q_view\[%q_token, %head, %qc\d+, )%c0\]", r"\1%lane_h8]")
    text = re.sub(r"^( *)(%qraw(\d+) = vector\.load %q_view\[%q_token, %head, )(%qd\d+)\]", lambda m: (
        f"{m.group(1)}%qdh{m.group(3)}_v = index.add {m.group(4)}, %lane_h8 : index\n"
        f"{m.group(1)}%qdh{m.group(3)} = index.assume %qdh{m.group(3)}_v [range(%qdh{m.group(3)}_v, 0, 8191)] : index\n"
        f"{m.group(1)}{m.group(2)}%qdh{m.group(3)}]"), text, flags=re.M)
    sub("row_off", r"^  %row_off = vector\.iota %half32 step %two_i32 : vector<8xi32>\n",
        "  %one_row_i32 = scalar.constant 1 : i32\n"
        "  %row_off = vector.iota %lane_h8_i32 step %one_row_i32 : vector<8xi32>\n")
    sub("interleave", r"^ *%pi(\d+)_0((?:_[a-z]+)?) = vector\.extract .*?^ *%pt\1_f\2 = vector\.interleave.*?\n", "", re.M | re.S)
    sub("pt_h", r"(%pt(\d+)_h((?:_[a-z]+)?) = vector\.fptrunc )%pt\2_f\3 : vector<16xf32>", r"\1%p\2\3 : vector<8xf32>")
    sub("iota16", r"%iota16 = vector\.iota %c0_i32 step %c1_i32m : vector<16xi32>",
        "%iota16 = vector.iota %lane_h8_i32 step %c1_i32m : vector<8xi32>")
    operand = re.compile(r"vector\.fragment|vector\.mma|%zero16h|%iota16|%thr_v|%lo_v|%mhi|%mlo|%m\d+_\d+\w* = vector\.andi|"
                         r"%vs\d|%qraw|%qh\d+\w* = vector\.fptrunc|%pt\d+_h\w* = ")
    text = "\n".join(re.sub(r"vector<16x(f16|f32|i32|i1)>", r"vector<8x\1>", ln) if operand.search(ln) else ln
                     for ln in text.split("\n"))
    # Inkling relative-position bias: element i of a lane's S result is key row i + 8*half on gfx12 (2i + half on gfx11)
    sub("rb_base", r"(%d0_\d+(?:_[a-z]+)? = scalar\.subi %dbase\d+(?:_[a-z]+)?, )%half32 : i32", r"\1%lane_h8_i32 : i32")
    def _step(m):
        e, c = int(m.group(2)), int(m.group(3))
        assert c == 2 * e, (name, m.group(0))
        return f"{m.group(1)}%r4_k{e}_i32 : i32"
    text, c = re.subn(r"(%dist\d+_(\d+)(?:_[a-z]+)? = scalar\.subi %d0_\d+(?:_[a-z]+)?, )%c(\d+)_i32 : i32", _step, text)
    n["rb_step"] = c
    if c:
        text = text.replace("  %lane_h8_i32 = index.cast %lane_h8 : index to i32\n",
                            "  %lane_h8_i32 = index.cast %lane_h8 : index to i32\n" +
                            "".join(f"  %r4_k{k}_i32 = scalar.constant {k} : i32\n" for k in range(1, 8)), 1)
    assert "%half32 : i32" not in "\n".join(l for l in text.split("\n") if "scalar.subi %dbase" in l), (name, "rb base left")
    sub("col_base", r"(%col_base = index\.add %\w+, )%lane_half : index", r"\1%lane_h8 : index")
    text = re.sub(r"(%dc(\d+)_(\d+) = index\.constant )(\d+)( : index)",
                  lambda m: f"{m.group(1)}{int(m.group(2)) * 16 + int(m.group(3))}{m.group(5)}", text)
    print(f"  {name}: rewrites {n}", file=sys.stderr)
    assert n["lane_half"] == 1, (name, n)
    assert "%pi0_0" not in text and "interleave" not in text, (name, "interleave left")
    return text


def main(src, dst):
    text = open(src).read()
    assert text.startswith("target.decl @device\namdgpu.target<gfx11-generic> @rdna3\namdgpu.target<gfx12-generic> @rdna4")
    sel_max = max(int(x) for x in re.findall(r"^template\.decl @sel(\d+)_", text, re.M))
    # 1. twins next to each kernel; base gets target(@rdna3)
    for k in KERNELS:
        m = re.search(r'^kernel\.def export\("%s"\) @%s\(.*?^\}\n' % (k, k), text, re.M | re.S)
        assert m, k
        base = m.group(0)
        twin = to_rdna4(base, k)
        twin = twin.replace(f'kernel.def export("{k}") @{k}(', f'kernel.def target(@rdna4) export("{k}_rdna4") @{k}_rdna4(', 1)
        base3 = base.replace(f'kernel.def export("{k}") @{k}(', f'kernel.def target(@rdna3) export("{k}") @{k}(', 1)
        text = text[:m.start()] + base3 + "\n" + twin + text[m.end():]
    # 2. launches -> template.apply, one decl per (kernel, signature)
    decls = {}
    launch_re = re.compile(r"^( *)kernel\.launch @(%s)\[([^\]]*)\]\(([^)]*)\) : \[([^\]]*)\]\((.*)\)$" % "|".join(KERNELS), re.M)

    def split(s):
        out, depth, cur = [], 0, ""
        for ch in s:
            if ch in "<([": depth += 1
            if ch in ">)]": depth -= 1
            if ch == "," and depth == 0:
                out.append(cur.strip()); cur = ""
            else:
                cur += ch
        if cur.strip(): out.append(cur.strip())
        return out

    def rep(m):
        nonlocal sel_max
        ind, k, grid, args, gty, aty = m.groups()
        grid, args, gty, aty = split(grid), split(args), split(gty), split(aty)
        assert len(grid) == len(gty) and len(args) == len(aty), m.group(0)
        sig = (k, tuple(gty), tuple(aty))
        if sig not in decls:
            sel_max += 1
            decls[sig] = f"sel{sel_max}_{k}"
        sel = decls[sig]
        types = ", ".join(gty + aty)
        return f"{ind}template.apply<@{sel}>({', '.join(grid + args)}) : ({types})"

    text, nl = launch_re.subn(rep, text)
    assert nl > 0
    blocks = []
    for (k, gty, aty), sel in decls.items():
        params = [f"%g{i}: {t}" for i, t in enumerate(gty)] + [f"%a{i}: {t}" for i, t in enumerate(aty)]
        p = ", ".join(params)
        g = ", ".join(f"%g{i}" for i in range(len(gty)))
        a = ", ".join(f"%a{i}" for i in range(len(aty)))
        launch_ty = f"[{', '.join(gty)}]({', '.join(aty)})"
        blocks.append(f"template.decl @{sel}({p})\n")
        for tgt, kk in (("rdna3", k), ("rdna4", k + "_rdna4")):
            blocks.append(f"template.def<@{sel}> target(@{tgt}) @{sel}_{tgt}({p}) {{\n"
                          f"  kernel.launch @{kk}[{g}]({a}) : {launch_ty}\n  template.return\n}}\n")
    # insert after the last existing template.def block
    last = list(re.finditer(r"^template\.def<.*?^\}\n", text, re.M | re.S))[-1]
    text = text[:last.end()] + "".join(blocks) + text[last.end():]
    left = re.findall(r"kernel\.launch @(%s)\[" % "|".join(KERNELS), text)
    print(f"launch sites rewritten: {nl}; new selectors: {len(decls)}; plain launches left: {len(left)}", file=sys.stderr)
    open(dst, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
