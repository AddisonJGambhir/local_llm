#!/usr/bin/env bash
# Wave 2: M4 (adaptive n_max kernel-shape cells) + M5 (three untested VK env vars).
# Only runs after wave 1 is complete. Kills only pids it starts.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
source "$SP/lib.sh"
CSV="$SP/depth_ab.csv"        # M4 appends here (same schema, comparable)
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT

COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
DRAFTKV="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
DEPTHS="0,120000"    # d0 for the kernel-shape effect, 120k for the regime that matters
REPS=15

phase(){ local label="$1"; shift
  log "=== WAVE2 PHASE $label"
  echo "===== $label =====" >> "$SP/server.log"
  start_server "$COMMON $*" || { log "PHASE $label FAILED TO START"; mykill; return 1; }
  log "PHASE $label up (peak $(cat $SP/.peak) MiB)"
  python3 "$SP/blockA.py" "$label" "$DEPTHS" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "PHASE $label done peak $(cat $SP/.peak) MiB"; mykill
}

log "WAVE 2 START"
# --- M4: is the per-n cost curve non-monotone? (Vulkan switches vec->matmul above 4 rows)
# n3 won every regime tested tonight, so sweep FIXED draft depth around it
# rather than spending the slot on adaptive ceilings of a losing config.
# f16 draft KV is +2.0% (p=0.0054, n=22) for adaptive. Never tested with n3 --
# if it carries over, n3+f16 is the fastest realistic-content config on the box.
phase mtp_n3_f16draft "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16"
phase mtp_n4_f16draft "--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-type-k f16 --spec-draft-type-v f16"
phase mtp_n2_q8draft  "--spec-type draft-mtp --spec-draft-n-max 2 $DRAFTKV"
phase mtp_n4_q8draft  "--spec-type draft-mtp --spec-draft-n-max 4 $DRAFTKV"
phase mtp_n5_q8draft  "--spec-type draft-mtp --spec-draft-n-max 5 $DRAFTKV"
# one adaptive ceiling cell kept: n8 is the kernel-shape boundary (Vulkan
# switches vec->matmul above 4 verify rows, and #25666 cuts MMVQ at 8)
phase adaptive_n8_q8draft  "--spec-type draft-mtp-adaptive --spec-draft-n-max 8  --spec-draft-n-min-adaptive 3 $DRAFTKV"

# --- M5: three GGML_VK_* envs the null table never covered. llama-bench (raw, no spec).
log "=== WAVE2 M5: untested Vulkan env vars (llama-bench, raw)"
for V in BASELINE GGML_VK_DISABLE_FUSION GGML_VK_DISABLE_COOPMAT GGML_VK_DISABLE_MULTI_ADD; do
  [ "$V" = BASELINE ] || export "$V=1"
  ( "$BENCH" -m "$MODEL" -ngl 99 -b 2048 -ub 1024 -fa 1 -ctk f16 -ctv f16 \
      -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split 49,51 \
      -p 4096 -n 256 -r 3 -o csv > "$SP/m5_$V.csv" 2>>"$SP/run.log" ) &
  BP=$!; MY_PID=$BP; peakwatch $BP & GUARD_PID=$!
  wait $BP; log "M5 $V done peak $(cat $SP/.peak) MiB"
  MY_PID=""; [ -n "$GUARD_PID" ] && kill $GUARD_PID 2>/dev/null; GUARD_PID=""
  [ "$V" = BASELINE ] || unset "$V"
done
log "WAVE 2 COMPLETE"
bash "$SP/wave_pp5.sh"
