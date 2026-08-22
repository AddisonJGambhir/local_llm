#!/usr/bin/env bash
# Uniform pp/tg for every preset in llama-launch, so menu labels are comparable.
#
# ONE method for all: 4,096-token prompt (cold, cache_prompt=false) + 1,024-token
# generation, temp 0 + top_k 1 + ignore_eos, whatever speculation the preset
# declares. pp is prompt_per_second, tg is predicted_per_second. 2 reps, median.
#
# Presets differ in context/quant/split by design -- that is what is being
# compared. What must NOT differ is the prompt and the generation length.
#
# Needs the GPUs. Aborts if either card is busy. Kills only llama-server.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/output/benchmarks/preset-sweep-$(date +%Y-%m-%d_%H-%M-%S).csv}"
XTX=/sys/class/drm/card1/device/mem_info_vram_used
mkdir -p "$(dirname "$OUT")"
echo "preset,backend,ctx,spec,pp,tg,peak_xtx_mib,status" > "$OUT"

kills(){ for p in $(pgrep -x llama-server); do kill "$p" 2>/dev/null; done
         for i in $(seq 1 25); do pgrep -x llama-server >/dev/null||break; sleep 1; done
         for p in $(pgrep -x llama-server); do kill -9 "$p" 2>/dev/null; done; sleep 5; }

PROMPT=$(python3 -c "print(('The system processes requests through a pipeline. '*420)[:24000])")
PAYLOAD=$(python3 -c "
import json,sys; print(json.dumps({'prompt':sys.argv[1],'n_predict':1024,'temperature':0,
'top_k':1,'ignore_eos':True,'cache_prompt':False}))" "$PROMPT")

for f in $(grep -o 'preset-[a-z0-9.-]*\.sh' "$ROOT/bin/llama-launch" | sort -u); do
  P="$ROOT/config/profiles/$f"
  [[ -f "$P" ]] || { echo "$f,,,,,,,MISSING" >> "$OUT"; continue; }
  kills
  base=$(( $(cat $XTX)/1048576 ))
  if (( base > 5000 )); then echo "$f,,,,,,${base},ABORT_GPU_BUSY" >> "$OUT"; continue; fi

  CMD=$("$ROOT/bin/llamactl" dry-run --profile "$P" 2>/dev/null) || { echo "$f,,,,,,,DRYRUN_FAIL" >> "$OUT"; continue; }
  bk=$(grep -oE '^LLAMA_BACKEND="?[a-z]+' "$P" | head -1 | grep -oE '[a-z]+$')
  cx=$(printf '%s' "$CMD" | grep -oE '\-c [0-9]+' | grep -oE '[0-9]+')
  sp=$(printf '%s' "$CMD" | grep -oE '\-\-spec-type [a-z-]+' | awk '{print $2}')
  eval "$CMD" >/dev/null 2>&1 &
  ok=0; peak=$base
  for i in $(seq 1 300); do
    u=$(( $(cat $XTX)/1048576 )); (( u>peak )) && peak=$u
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    pgrep -x llama-server >/dev/null || break
    sleep 1
  done
  if (( ok==0 )); then echo "$f,$bk,${cx:-},${sp:-none},,,${peak},LOAD_FAIL" >> "$OUT"; kills; continue; fi

  best_pp=0; best_tg=0
  for rep in 1 2; do
    r=$(curl -s -m 600 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' -d "$PAYLOAD" \
        | python3 -c "
import json,sys
try:
    t=json.load(sys.stdin)['timings']; print('%.1f %.2f'%(t.get('prompt_per_second',0),t['predicted_per_second']))
except Exception: print('0 0')")
    pp=${r% *}; tg=${r#* }
    awk "BEGIN{exit !($tg > $best_tg)}" && { best_pp=$pp; best_tg=$tg; }
    u=$(( $(cat $XTX)/1048576 )); (( u>peak )) && peak=$u
  done
  echo "$f,$bk,${cx:-},${sp:-none},$best_pp,$best_tg,$peak,OK" >> "$OUT"
  kills
done
echo "PRESET SWEEP DONE -> $OUT"
