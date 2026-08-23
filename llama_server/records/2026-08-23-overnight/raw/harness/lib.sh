# shared helpers for the overnight run. never uses pkill -f / pgrep -f.
LLM=/home/addison-gambhir/Desktop/local_llm
LS=$LLM/llama_server
TREE=$LLM/llama.cpp-tp-adaptive
BIN=${BIN_OVERRIDE:-$TREE/build/bin/llama-server}
BENCH=$TREE/build/bin/llama-bench
MODEL=$LS/models/Qwen3.8-27B-Q8_0.gguf
XTXF=/sys/class/drm/card1/device/mem_info_vram_used
R97F=/sys/class/drm/card0/device/mem_info_vram_used
CEIL=24000
PORT=1234
export LD_LIBRARY_PATH=$(dirname "${BIN_OVERRIDE:-$TREE/build/bin/llama-server}")

MY_PID=""; GUARD_PID=""; PEAK=0
xtx(){ echo $(( $(cat $XTXF)/1048576 )); }

# kills ONLY the pid this harness started
mykill(){
  [ -z "${MY_PID:-}" ] && return 0
  kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null
  wait "$MY_PID" 2>/dev/null
  MY_PID=""
  [ -n "${GUARD_PID:-}" ] && { kill "$GUARD_PID" 2>/dev/null; GUARD_PID=""; }
  sleep 4
}

# ceiling guard: 4 consecutive samples over CEIL -> kill only our pid
guard(){
  local pid=$1 over=0
  while kill -0 "$pid" 2>/dev/null; do
    local u=$(xtx)
    (( u > PEAK )) && PEAK=$u
    echo "$u" > "$SP/.peak_sample"
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    if (( over >= 2 )); then
      echo "!! CEILING $u MiB sustained -> killing $pid" >> "$SP/run.log"
      kill -9 "$pid" 2>/dev/null; return 1
    fi
    sleep 1
  done
}

# peak tracker (no kill) writes max to $SP/.peak
peakwatch(){
  local pid=$1 mx=0
  while kill -0 "$pid" 2>/dev/null; do
    local u=$(xtx); (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then
      local o=1
      for i in 1 2 3; do sleep 0.5; local v=$(xtx); (( v > CEIL )) && o=$((o+1)); done
      if (( o >= 2 )); then echo "!! CEILING ${u} MiB sustained -> kill $pid" >> "$SP/run.log"; kill -9 "$pid" 2>/dev/null; return 1; fi
    fi
    sleep 1
  done
  echo "$mx" > "$SP/.peak"
}

start_server(){   # start_server "<extra args>"
  # HARD GUARD: never benchmark against a server this call did not start.
  if ss -lntH "sport = :$PORT" 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: port $PORT already in use -- another run is live" >> "$SP/run.log"
    return 1
  fi
  if pgrep -x llama-server >/dev/null; then
    echo "REFUSING TO START: a llama-server is already running" >> "$SP/run.log"
    return 1
  fi
  echo 0 > "$SP/.peak"
  setsid "$BIN" -m "$MODEL" --alias local --device Vulkan0,Vulkan1 -ngl 99 -fit off \
     -t 16 -tb 32 --parallel 1 --flash-attn on --no-mmap --no-context-shift \
     --split-mode tensor --main-gpu 1 --tensor-split 49,51 \
     --host 127.0.0.1 --port $PORT $1 >> "$SP/server.log" 2>&1 &
  MY_PID=$!
  peakwatch "$MY_PID" & GUARD_PID=$!
  for i in $(seq 1 400); do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 "$MY_PID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
