#!/usr/bin/env bash
# Guarded single-configuration benchmark for Qwen3.8-Flash-Next.
# Runs one warmup plus three measured generations and reports their median.
set -Eeuo pipefail

PROJECT_ROOT=/home/addison-gambhir/Desktop/local_llm
BIN="${FN_BIN:-$PROJECT_ROOT/llama.cpp-qwen4exp/build-vulkan/bin/llama-server}"
MODEL="${FN_MODEL:-$PROJECT_ROOT/llama_server/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf}"
GUARD="$PROJECT_ROOT/llama_server/bin/vram_guard.sh"

# 65K is deliberately the default: the recorded 128K config exceeds the new
# unattended R9700 ceiling. Raise context only with an explicit FN_CTX.
CTX="${FN_CTX:-65536}"
UB="${FN_UB:-1024}"
BATCH="${FN_B:-2048}"
TS="${FN_TS:-385,615}"
CTK="${FN_CTK:-q8_0}"
CTV="${FN_CTV:-q8_0}"
OT="${FN_OT:-per_layer_token_embd\\.weight=CPU}"
DEV="${FN_DEV:-Vulkan0,Vulkan1}"
LOADMODE="${FN_LOADMODE:-mmap}"
LAZY="${FN_LAZY:-on}"
NGL="${FN_NGL:-99}"
THREADS="${FN_THREADS:-}"
EXTRA="${FN_EXTRA:-}"
TAG="${FN_TAG:-untagged}"
PORT="${FN_PORT:-18253}"
NPRED="${FN_NPRED:-256}"
MEASURED_PASSES="${FN_PASSES:-3}"
MEM_NEED="${FN_MEM_NEED:-30}"
R9700_MAX="${R9700_MAX:-31600}"
XTX_MAX="${XTX_MAX:-24200}"
MEM_MIN="${MEM_MIN:-6}"

RUN="${FN_RUN:-/tmp/fn-sweep/$TAG}"
LOG="$RUN/server.log"
PID=
GUARD_PID=
mkdir -p "$RUN"

cleanup() {
    if [[ -n "${GUARD_PID:-}" ]] && kill -0 "$GUARD_PID" 2>/dev/null; then
        kill "$GUARD_PID" 2>/dev/null || true
        wait "$GUARD_PID" 2>/dev/null || true
    fi
    if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if pgrep -x llama-server >/dev/null; then
    echo "ABORT: an existing llama-server is running"
    pgrep -a -x llama-server
    exit 20
fi

echo "PRECHECK listeners"
ss -ltnp | grep -E ":(1234|8080|${PORT})[[:space:]]" || true
if ss -ltn | grep -qE ":(1234|${PORT})[[:space:]]"; then
    echo "ABORT: user or test port is occupied"
    exit 21
fi

available_gib=$(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
if (( available_gib < MEM_NEED )); then
    echo "ABORT: MemAvailable ${available_gib} GiB < ${MEM_NEED} GiB"
    exit 22
fi

card0_pre=$(( $(< /sys/class/drm/card0/device/mem_info_vram_used) / 1048576 ))
card1_pre=$(( $(< /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 ))
echo "PRECHECK card0=${card0_pre}MiB card1=${card1_pre}MiB MemAvailable=${available_gib}GiB"
if (( card0_pre > R9700_MAX || card1_pre > XTX_MAX )); then
    echo "ABORT: a GPU is already above the guard ceiling"
    exit 23
fi

args=(
    -m "$MODEL" -c "$CTX" --device "$DEV"
    --split-mode layer --tensor-split "$TS"
    -ub "$UB" -b "$BATCH" --flash-attn on
    --cache-type-k "$CTK" --cache-type-v "$CTV"
    --load-mode "$LOADMODE" -ot "$OT"
    --host 127.0.0.1 --port "$PORT" --parallel 1
    --fit off --log-verbosity 2
)
[[ "$LAZY" != "skip" ]] && args+=( --tensor-read-lazy "$LAZY" )
[[ "$NGL" != "skip" ]] && args+=( -ngl "$NGL" )
[[ -n "$THREADS" ]] && args+=( -t "$THREADS" )
# FN_EXTRA is intentionally a trusted, locally supplied argument string.
# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=( $EXTRA )

echo "CONFIG tag=$TAG ctx=$CTX ub=$UB ts=$TS kv=$CTK/$CTV load=$LOADMODE threads=${THREADS:-default}"
echo "CONFIG bin=$BIN run=$RUN"
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 "$BIN" "${args[@]}" >"$LOG" 2>&1 &
PID=$!
GUARD_LOG="$RUN/guard.log" R9700_MAX="$R9700_MAX" XTX_MAX="$XTX_MAX" MEM_MIN="$MEM_MIN" \
    "$GUARD" "$PID" &
GUARD_PID=$!
echo "LAUNCHED pid=$PID guard_pid=$GUARD_PID"

ready=0
for ((i = 0; i < 420; ++i)); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "RESULT tag=$TAG status=died_during_load"
        tail -80 "$LOG"
        tail -20 "$RUN/guard.log" 2>/dev/null || true
        exit 24
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
if (( ready == 0 )); then
    echo "RESULT tag=$TAG status=load_timeout"
    exit 25
fi

request="$RUN/request.json"
jq -n --arg prompt 'Write three sentences about why memory bandwidth matters for language model inference.' \
    --argjson n_predict "$NPRED" \
    '{prompt:$prompt,n_predict:$n_predict,temperature:0,top_k:1,seed:42,stream:false,cache_prompt:false}' \
    >"$request"

declare -a measured=()
total_passes=$(( MEASURED_PASSES + 1 ))
for ((pass = 1; pass <= total_passes; ++pass)); do
    output="$RUN/out${pass}.json"
    if ! curl -sf --max-time 600 "http://127.0.0.1:$PORT/completion" \
            -H 'Content-Type: application/json' --data-binary @"$request" >"$output"; then
        echo "RESULT tag=$TAG status=completion_failed pass=$pass"
        tail -20 "$RUN/guard.log" 2>/dev/null || true
        exit 26
    fi
    tg=$(jq -r '.timings.predicted_per_second' "$output")
    if (( pass == 1 )); then
        echo "PASS $pass warmup_tg=$tg"
    else
        echo "PASS $pass measured_tg=$tg"
        measured+=( "$tg" )
    fi
done

mapfile -t sorted < <(printf '%s\n' "${measured[@]}" | sort -n)
middle=$(( ${#sorted[@]} / 2 ))
if (( ${#sorted[@]} % 2 == 1 )); then
    median="${sorted[$middle]}"
else
    median=$(awk -v a="${sorted[$((middle - 1))]}" -v b="${sorted[$middle]}" 'BEGIN { printf "%.6f", (a+b)/2 }')
fi
minimum="${sorted[0]}"
maximum="${sorted[$(( ${#sorted[@]} - 1 ))]}"
spread=$(awk -v lo="$minimum" -v hi="$maximum" 'BEGIN { printf "%.6f", hi-lo }')

card0=$(( $(< /sys/class/drm/card0/device/mem_info_vram_used) / 1048576 ))
card1=$(( $(< /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 ))
rss_kib=$(awk '/VmRSS:/ {print $2}' "/proc/$PID/status")
echo "RESULT tag=$TAG status=ok tg_median=$median tg_min=$minimum tg_max=$maximum tg_spread=$spread card0=${card0}MiB card1=${card1}MiB rss_kib=$rss_kib"
