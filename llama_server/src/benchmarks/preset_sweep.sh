#!/usr/bin/env bash
# Uniform pp/tg for every preset in llama-launch, so menu labels are comparable.
#
# ONE method for all: ~4.9k-token prompt (cold, cache_prompt=false) + 1,024-token
# generation, approved sampling (temp 1.0, top_p 0.95, top_k 20; never greedy) + ignore_eos, whatever speculation the preset
# declares. pp is prompt_per_second, tg is predicted_per_second.
#
# AGGREGATION: best (max) of 2 reps, not median -- 2 reps has no median. Fine for
# ranking menu labels; NOT valid for comparing speculators (that needs >=7 reps
# and a permutation test, see docs/HOW_TO_OPTIMIZE.md).
#
# PROMPT REGIME: the prompt is one sentence repeated. Per F24 that is the
# repetitive-content regime, which inflates MTP tg ~2.5x over novel content.
# Every tg here is a repetitive-regime UPPER BOUND. The csv `note` column says so.
#
# Presets differ in context/quant/split by design -- that is what is being
# compared. What must NOT differ is the prompt and the generation length.
#
# Needs the GPUs. REFUSES TO RUN if a llama-server is already up (it will not
# kill a server it did not start). Kills only the pid it spawned.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/output/benchmarks/preset-sweep-$(date +%Y-%m-%d_%H-%M-%S).csv}"
LOGDIR="${OUT%.csv}-logs"
XTX=/sys/class/drm/card1/device/mem_info_vram_used
CEIL=24000            # sustained MiB on the XTX that triggers a kill of OUR server
mkdir -p "$(dirname "$OUT")" "$LOGDIR"
echo "preset,backend,ctx,spec,pp,tg,acceptance,mean_acc_len,peak_xtx_mib,status,note" > "$OUT"

# --- refuse to touch a server we did not start -------------------------------
if pgrep -x llama-server >/dev/null; then
  echo "ABORT: a llama-server is already running (pids: $(pgrep -x llama-server | tr '\n' ' '))." >&2
  echo "       This sweep only kills servers it starts. Stop yours first." >&2
  exit 1
fi

MY_PID=""; WATCH_PID=""; PEAKF="$LOGDIR/.peak"
mykill(){
  [ -z "$MY_PID" ] && return 0
  kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null
  wait "$MY_PID" 2>/dev/null
  MY_PID=""
  [ -n "$WATCH_PID" ] && { kill "$WATCH_PID" 2>/dev/null; WATCH_PID=""; }
  sleep 4
}
trap 'mykill' EXIT INT TERM

# peak tracker + ceiling guard: 4 consecutive samples over CEIL kills OUR pid only
peakwatch(){
  local pid=$1 mx=0 over=0 u
  echo 0 > "$PEAKF"
  while kill -0 "$pid" 2>/dev/null; do
    u=$(( $(cat $XTX)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$PEAKF"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    if (( over >= 4 )); then
      echo "CEILING ${u} MiB sustained -> killing our server $pid" >&2
      kill -9 "$pid" 2>/dev/null; break
    fi
    sleep 1
  done
  echo "$mx" > "$PEAKF"
}

PROMPT=$(python3 -c "print(('The system processes requests through a pipeline. '*420)[:24000])")
PAYLOAD=$(python3 -c "
import json,sys; print(json.dumps({'prompt':sys.argv[1],'n_predict':1024,'temperature':1.0,'top_p':0.95,
'top_k':20,'ignore_eos':True,'cache_prompt':False}))" "$PROMPT")
NOTE="max-of-2;repetitive-prompt"

for f in $(grep -o 'preset-[a-z0-9.-]*\.sh' "$ROOT/bin/llama-launch" | sort -u); do
  P="$ROOT/config/profiles/$f"
  [[ -f "$P" ]] || { echo "$f,,,,,,,,,MISSING,$NOTE" >> "$OUT"; continue; }
  base=$(( $(cat $XTX)/1048576 ))
  if (( base > 5000 )); then echo "$f,,,,,,,,${base},ABORT_GPU_BUSY,$NOTE" >> "$OUT"; continue; fi

  CMD=$("$ROOT/bin/llamactl" dry-run --profile "$P" 2>/dev/null) || { echo "$f,,,,,,,,,DRYRUN_FAIL,$NOTE" >> "$OUT"; continue; }
  bk=$(grep -oE '^LLAMA_BACKEND="?[a-z]+' "$P" | head -1 | grep -oE '[a-z]+$')
  cx=$(printf '%s' "$CMD" | grep -oE '\-c [0-9]+' | grep -oE '[0-9]+')
  sp=$(printf '%s' "$CMD" | grep -oE '\-\-spec-type [a-z-]+' | awk '{print $2}')
  SLOG="$LOGDIR/${f%.sh}.log"

  setsid bash -c "$CMD" > "$SLOG" 2>&1 &
  MY_PID=$!
  peakwatch "$MY_PID" & WATCH_PID=$!

  ok=0
  for i in $(seq 1 300); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break
    sleep 1
  done
  if (( ok==0 )); then
    echo "$f,$bk,${cx:-},${sp:-none},,,,,$(cat $PEAKF),LOAD_FAIL,$NOTE" >> "$OUT"; mykill; continue
  fi

  best_pp=0; best_tg=0; best_acc=""; best_len=""
  for rep in 1 2; do
    r=$(curl -s -m 600 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' -d "$PAYLOAD" \
        | python3 -c "
import json,sys
try:
    t=json.load(sys.stdin)['timings']
    dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
    acc=(da/dn) if dn else 0.0
    print('%.1f %.2f %.4f %d'%(t.get('prompt_per_second',0),t['predicted_per_second'],acc,dn))
except Exception: print('0 0 0 0')")
    read -r pp tg acc dn <<< "$r"
    awk "BEGIN{exit !($tg > $best_tg)}" && { best_pp=$pp; best_tg=$tg; best_acc=$acc; best_len=$dn; }
  done
  echo "$f,$bk,${cx:-},${sp:-none},$best_pp,$best_tg,$best_acc,$best_len,$(cat $PEAKF),OK,$NOTE" >> "$OUT"
  mykill
done
echo "PRESET SWEEP DONE -> $OUT"
echo "  per-preset server logs (draft acceptance lines): $LOGDIR"
