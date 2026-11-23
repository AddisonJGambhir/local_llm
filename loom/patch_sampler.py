"""Speed up Loom's sampled top-k in g_head_select (Inkling-2-Mini package).

The stock kernel finds the top-k logits with k full passes over the 200,064-entry vocabulary on one 256-lane
workgroup per sampled position. The head matvec also writes, per 128-row chunk, the chunk's max logit
(partials[span][chunk] = (bits, id)). Every top-k token lies in a chunk whose max is among the top-k chunk
maxima, so: (1) select the top-k chunks from the 1,563 chunk maxima, (2) run the same k-round selection only
over those k*128 logits. Same ordering/tie rules (value desc, index asc) as the stock code.

usage: python3 -I patch_sampler.py llm.loom   (rewrites in place; refuses to patch twice)
"""
import re
import sys

path = sys.argv[1]
text = open(path).read()
m = re.search(r'^kernel\.def export\("g_head_select"\) @g_head_select\(.*?^\}\n', text, re.M | re.S)
assert m, "g_head_select not found"
k = m.group(0)
assert "%selc_v" not in k, "already patched"

# 1. workgroup array for the selected chunk ids, next to the candidate buffer
anchor = "  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>\n"
assert k.count(anchor) == 1
k = k.replace(anchor, anchor +
              "  %selc = buffer.alloca<workgroup> align(16) %c256o : buffer\n"
              "  %selc_v = buffer.view %selc[%zero] : buffer -> view<64xi32>\n"
              "  %c128i = index.constant 128 : index\n"
              "  %c1562_i32 = scalar.constant 1562 : i32\n", 1)

# 2. chunk pre-selection, inserted right before the k-round token loop
round_hdr = "        // k rounds: the largest logit strictly after the previous pick in (value desc, index asc) order.\n"
assert k.count(round_hdr) == 1
chunk_sel = """        // Patched: top-k chunks by chunk max (same order rules) -> the k-round loop scans only those chunks.
        %pchunk_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x1563x2xi32>
        %cl_v, %cl_i = scf.for %rc = [%c0 to %k step %c1](%pvc = %pos_inf : f32, %pic = %c0_i32 : i32) -> (f32, i32) {
          %minus_one_c = scalar.constant -1 : i32
          %prev_ic = scalar.addi %pic, %c0_i32 : i32
          %first_round_c = index.cmp eq, %rc, %c0 : index
          %floor_ic = scf.select %first_round_c, %minus_one_c, %prev_ic : i32
          %lvc, %lic = scf.for %cc = [%lane to %chunks step %c256](%bvc = %neg_inf : f32, %bic = %big : i32) -> (f32, i32) {
            %bounded_cc = index.assume %cc [range(%cc, 0, 1562)] : index
            %xbc = view.load %pchunk_view[%span, %bounded_cc, %c0] : view<[%spans]x1563x2xi32> -> i32
            %xc = scalar.bitcast %xbc : i32 to f32
            %c32c = index.cast %bounded_cc : index to i32
            %belowc = scalar.cmpf olt, %xc, %pvc : f32
            %tiec = scalar.cmpf oeq, %xc, %pvc : f32
            %afterc = scalar.cmpi sgt, %c32c, %floor_ic : i32
            %tie_afterc = scalar.andi %tiec, %afterc : i1
            %eligiblec = scalar.ori %belowc, %tie_afterc : i1
            %greaterc = scalar.cmpf ogt, %xc, %bvc : f32
            %takec = scalar.andi %eligiblec, %greaterc : i1
            %nvc = scf.select %takec, %xc, %bvc : f32
            %nic = scf.select %takec, %c32c, %bic : i32
            scf.yield %nvc, %nic : f32, i32
          }
          %bestc = kernel.workgroup.reduce<maxnumf> %lvc : f32
          %is_bestc = scalar.cmpf oeq, %lvc, %bestc : f32
          %cand_idc = scf.select %is_bestc, %lic, %big : i32
          %winnerc = kernel.workgroup.reduce<minsi> %cand_idc : i32
          scf.if %first {
            %rrc = index.assume %rc [range(%rc, 0, 63)] : index
            view.store %winnerc, %selc_v[%rrc] : i32, view<64xi32>
          }
          scf.yield %bestc, %winnerc : f32, i32
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        %kc_n = index.mul %k, %c128i : index
"""
k = k.replace(round_hdr, chunk_sel + round_hdr, 1)

# 3. the token loop: iterate the k*128 logits of the selected chunks instead of the whole vocabulary
old_loop = """          %lv, %li = scf.for %v = [%lane to %vocab step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %bounded_v = index.assume %v [range(%v, 0, 200057)] : index
"""
new_loop = """          %lv, %li = scf.for %j = [%lane to %kc_n step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %bj = index.assume %j [range(%j, 0, 8191)] : index
            %jc = index.div %bj, %c128i : index
            %jo = index.rem %bj, %c128i : index
            %jcb = index.assume %jc [range(%jc, 0, 63)] : index
            %chunk_raw = view.load %selc_v[%jcb] : view<64xi32> -> i32
            %chunk_lo = scalar.maxsi %chunk_raw, %c0_i32 : i32
            %chunk_cl = scalar.minsi %chunk_lo, %c1562_i32 : i32
            %chunk_b = scalar.assume %chunk_cl [range(%chunk_cl, 0, 1562)] : i32
            %chunk_ix = index.cast %chunk_b : i32 to index
            %row0 = index.mul %chunk_ix, %c128i : index
            %row = index.add %row0, %jo : index
            %bounded_v = index.assume %row [range(%row, 0, 200063)] : index
"""
assert k.count(old_loop) == 1, "token loop pattern not found"
k = k.replace(old_loop, new_loop, 1)

text = text[:m.start()] + k + text[m.end():]
open(path, "w").write(text)
print("patched g_head_select: chunk pre-selection + k*128 candidate scan")
