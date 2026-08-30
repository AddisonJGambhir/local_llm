#!/usr/bin/env bash
# Guard a test llama-server PID. Kills ONLY on sustained breach (3 consecutive
# samples), never on a transient load spike. Crash-prevention only:
#   card0 (R9700) sustained > R9700_MAX MiB
#   card1 (XTX, drives the desktop) sustained > XTX_MAX MiB
#   MemAvailable sustained < MEM_MIN GiB
set -Eeuo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 LLAMA_SERVER_PID" >&2
    exit 2
fi

PID="$1"
XTX_MAX="${XTX_MAX:-24200}"
R9700_MAX="${R9700_MAX:-32500}"
MEM_MIN="${MEM_MIN:-6}"
LOG="${GUARD_LOG:-/tmp/vram_guard.log}"
# Load-time allocation spikes high and settles. Enforce the configured
# ceilings only once the server is ready; until then enforce only the
# physical limits, which are the real crash boundary.
READY_FILE="${READY_FILE:-}"
XTX_PHYS=$(( $(< /sys/class/drm/card1/device/mem_info_vram_total) / 1048576 - 96 ))
R9700_PHYS=$(( $(< /sys/class/drm/card0/device/mem_info_vram_total) / 1048576 - 96 ))
strikes=0
echo "$(date +%T) GUARD start pid=$PID r9700_max=${R9700_MAX}MiB xtx_max=${XTX_MAX}MiB mem_min=${MEM_MIN}GiB" >>"$LOG"
while kill -0 "$PID" 2>/dev/null; do
    xtx=$(( $(< /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 ))
    r97=$(( $(< /sys/class/drm/card0/device/mem_info_vram_used) / 1048576 ))
    memav=$(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
    if [[ -n "$READY_FILE" && ! -e "$READY_FILE" ]]; then
        xtx_lim=$XTX_PHYS; r97_lim=$R9700_PHYS
    else
        xtx_lim=$XTX_MAX; r97_lim=$R9700_MAX
    fi
    breach=0
    (( xtx > xtx_lim )) && breach=1
    (( r97 > r97_lim )) && breach=1
    (( memav < MEM_MIN )) && breach=1
    if (( breach )); then
        strikes=$(( strikes + 1 ))
        echo "$(date +%T) BREACH $strikes/3 xtx=${xtx}/${xtx_lim} r9700=${r97}/${r97_lim} memavail=${memav}G" >>"$LOG"
        if (( strikes >= 3 )); then
            echo "$(date +%T) KILLING pid=$PID sustained breach r9700=${r97}MiB xtx=${xtx}MiB memavail=${memav}G" >>"$LOG"
            kill "$PID" 2>/dev/null
            sleep 10
            kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
            exit 1
        fi
    else
        strikes=0
    fi
    sleep 5
done
echo "$(date +%T) GUARD exit, pid $PID gone" >>"$LOG"
