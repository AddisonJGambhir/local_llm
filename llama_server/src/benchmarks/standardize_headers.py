#!/usr/bin/env python3
"""Prepend a uniform FACTS block to every preset header.

The block is generated from `llamactl dry-run`, i.e. from the command the server
actually receives -- not from the filename and not from prose. That is deliberate:
labels drifted from reality before (a preset described as "layer/balanced" was
really L 38,27; one described as text-only carried an mmproj). Regenerate after
changing a profile:  python3 src/benchmarks/standardize_headers.py
"""
import re, subprocess, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROF = os.path.join(ROOT, "config", "profiles")
CEIL = 24200
BEGIN, END = "# ===== FACTS (generated: do not hand-edit) =====", "# ===== END FACTS ====="

def dry(p):
    r = subprocess.run([os.path.join(ROOT,"bin","llamactl"),"dry-run","--profile",p],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""

def mtmd_device(profile_path):
    """Resolve MTMD_BACKEND_DEVICE the way llamactl does: source the profile.

    Grepping the profile text alone is wrong -- most tensor-decode presets inherit
    the export from a parent via `source`, so a text grep reports "NOT routed" for
    a profile that is in fact routed. Verified 2026-08-23: all four *-tensor-decode
    presets inherit MTMD_BACKEND_DEVICE=ROCm1 from their parent.
    """
    env = dict(os.environ, LLAMA_SERVER_ROOT=ROOT, LLM_ROOT=os.path.dirname(ROOT))
    env.pop("MTMD_BACKEND_DEVICE", None)
    r = subprocess.run(["bash","-c",
                        f'source "{profile_path}" >/dev/null 2>&1; '
                        f'printf "%s" "${{MTMD_BACKEND_DEVICE:-}}"'],
                       capture_output=True, text=True, env=env)
    return r.stdout.strip() or None

def one(v, pat, default="-"):
    m = re.search(pat, v)
    return m.group(1).replace("\\","") if m else default

changed = 0
for p in sorted(glob.glob(os.path.join(PROF, "preset-*.sh"))):
    cmd = dry(p)
    if not cmd:
        print(f"  SKIP (dry-run failed): {os.path.basename(p)}"); continue
    src  = open(p).read()

    binp   = one(cmd, r"^(\S*llama-server)")
    build  = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(binp)))) if "llama.cpp" in binp else "-"
    model  = os.path.basename(one(cmd, r"-m (\S+)"))
    ctx    = one(cmd, r"-c (\d+)")
    kv     = one(cmd, r"--cache-type-k (\S+)")
    sm     = one(cmd, r"--split-mode (\S+)", "layer")
    ts     = one(cmd, r"--tensor-split ([0-9,\\]+)", "auto")
    mg     = one(cmd, r"--main-gpu (\d+)", "-")
    spec   = one(cmd, r"--spec-type (\S+)", "none")
    nmax   = one(cmd, r"--spec-draft-n-max (\d+)")
    nmin   = one(cmd, r"--spec-draft-n-min-adaptive (\d+)")
    ub     = one(cmd, r"--ubatch-size (\d+)")
    vis    = "yes" if "--mmproj" in cmd else "no"
    if vis == "yes":
        dev = mtmd_device(p)
        vis += f" (tower on {dev})" if dev else " (tower NOT routed - defaults to first device, the XTX)"
    spec_s = spec if spec == "none" else f"{spec}  n_max {nmax}" + (f"  n_min_adaptive {nmin}" if nmin != "-" else "")

    facts = "\n".join([
        BEGIN,
        f"# MODEL    {model}",
        f"# BUILD    {build}   backend {one(cmd, r'--device (\S+)', '-')}",
        f"# SPLIT    {sm} {ts}   (XTX,R9700)   main-gpu {mg}   ubatch {ub}",
        f"# CONTEXT  {ctx}   KV {kv}",
        f"# SPEC     {spec_s}",
        f"# VISION   {vis}",
        f"# VRAM     XTX ceiling {CEIL} MiB - see MEASURED below for this profile's peak",
        f"# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)",
        END, ""])

    body = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", "", src, flags=re.S)
    open(p, "w").write(facts + body)
    changed += 1

print(f"standardized {changed} preset headers")
