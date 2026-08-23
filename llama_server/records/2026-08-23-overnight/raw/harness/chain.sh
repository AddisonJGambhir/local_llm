#!/usr/bin/env bash
# Wait until NO process of the named script remains (handles forked subshells,
# which is what broke this three times). Never uses pgrep -f.
waitfor(){
  local pat="$1" me=$$ n
  while :; do
    n=0
    for d in /proc/[0-9]*; do
      local p=${d#/proc/}
      [ "$p" = "$me" ] && continue
      [ -r "$d/cmdline" ] || continue
      case "$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)" in *"$pat"*) n=$((n+1));; esac
    done
    [ "$n" -eq 0 ] && return 0
    sleep 20
  done
}
settle(){ for i in $(seq 1 60); do pgrep -x llama-server >/dev/null || break; sleep 5; done; sleep 15; }

waitfor wave_rec.sh
echo "[$(date +%H:%M:%S)] wave_rec finished" >> /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/run.log
settle
for w in wave_rec2 wave_max wave_pp wave_bench wave_irl wave1b wave2; do
  echo "[$(date +%H:%M:%S)] CHAIN -> $w" >> /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/run.log
  bash /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/$w.sh
  settle
done
echo "[$(date +%H:%M:%S)] ALL WAVES COMPLETE" >> /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/run.log
