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
R9700_MIGRATION_MAX="${R9700_MIGRATION_MAX:-512}"
XTX_MIGRATION_MAX="${XTX_MIGRATION_MAX:-512}"
LOG="${GUARD_LOG:-/tmp/vram_guard.log}"
# Load-time allocation spikes high and settles. Enforce the configured
# ceilings only once the server is ready; until then enforce only the
# physical limits, which are the real crash boundary.
READY_FILE="${READY_FILE:-}"
XTX_PHYS=$(( $(< /sys/class/drm/card1/device/mem_info_vram_total) / 1048576 - 96 ))
R9700_PHYS=$(( $(< /sys/class/drm/card0/device/mem_info_vram_total) / 1048576 - 96 ))
strikes=0
tracking_ready=0
r97_vram_hi=0
xtx_vram_hi=0
r97_gtt_lo=0
xtx_gtt_lo=0
echo "$(date +%T) GUARD start pid=$PID r9700_max=${R9700_MAX}MiB xtx_max=${XTX_MAX}MiB mem_min=${MEM_MIN}GiB migration=${R9700_MIGRATION_MAX}/${XTX_MIGRATION_MAX}MiB" >>"$LOG"
while kill -0 "$PID" 2>/dev/null; do
    xtx=$(( $(< /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 ))
    r97=$(( $(< /sys/class/drm/card0/device/mem_info_vram_used) / 1048576 ))
    xtx_gtt=$(( $(< /sys/class/drm/card1/device/mem_info_gtt_used) / 1048576 ))
    r97_gtt=$(( $(< /sys/class/drm/card0/device/mem_info_gtt_used) / 1048576 ))
    memav=$(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) / 1048576 ))
    if [[ -n "$READY_FILE" && ! -e "$READY_FILE" ]]; then
        xtx_lim=$XTX_PHYS; r97_lim=$R9700_PHYS
        migration=0
    else
        xtx_lim=$XTX_MAX; r97_lim=$R9700_MAX
        if (( tracking_ready == 0 )); then
            tracking_ready=1
            r97_vram_hi=$r97
            xtx_vram_hi=$xtx
            r97_gtt_lo=$r97_gtt
            xtx_gtt_lo=$xtx_gtt
            echo "$(date +%T) READY_BASELINE r9700=${r97}MiB/${r97_gtt}MiB_gtt xtx=${xtx}MiB/${xtx_gtt}MiB_gtt" >>"$LOG"
        fi
        (( r97 > r97_vram_hi )) && r97_vram_hi=$r97
        (( xtx > xtx_vram_hi )) && xtx_vram_hi=$xtx
        (( r97_gtt < r97_gtt_lo )) && r97_gtt_lo=$r97_gtt
        (( xtx_gtt < xtx_gtt_lo )) && xtx_gtt_lo=$xtx_gtt
        migration=0
        if (( r97_vram_hi - r97 >= R9700_MIGRATION_MAX && r97_gtt - r97_gtt_lo >= R9700_MIGRATION_MAX )); then
            migration=1
        fi
        if (( xtx_vram_hi - xtx >= XTX_MIGRATION_MAX && xtx_gtt - xtx_gtt_lo >= XTX_MIGRATION_MAX )); then
            migration=1
        fi
    fi
    breach=0
    (( xtx > xtx_lim )) && breach=1
    (( r97 > r97_lim )) && breach=1
    (( memav < MEM_MIN )) && breach=1
    (( migration )) && breach=1
    if (( breach )); then
        strikes=$(( strikes + 1 ))
        echo "$(date +%T) BREACH $strikes/3 xtx=${xtx}/${xtx_lim} r9700=${r97}/${r97_lim} memavail=${memav}G migration=${migration} r9700_gtt=${r97_gtt}MiB xtx_gtt=${xtx_gtt}MiB" >>"$LOG"
        if (( strikes >= 3 )); then
            echo "$(date +%T) KILLING pid=$PID sustained breach r9700=${r97}MiB xtx=${xtx}MiB memavail=${memav}G migration=${migration}" >>"$LOG"
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
