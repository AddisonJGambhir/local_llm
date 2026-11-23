#!/bin/bash
# lvtest.sh PKG CARD NCTX MODELID : Loom vision test on the R9700 (amdgpu://1), backend :11438, proxy :1239, then smoke test.
#   KEEP=1 leaves it running. Units loom-vtest-backend / loom-vtest-proxy (never touches loom-backend/loom-proxy on :1234).
L=~/Desktop/local_llm/loom; PKG=$1; CARD=${2:-model.json}; NCTX=$3; MID=${4:-local}; BP=11438; PP=1239
systemctl --user stop loom-vtest-proxy loom-vtest-backend 2>/dev/null; systemctl --user reset-failed loom-vtest-proxy loom-vtest-backend 2>/dev/null
systemd-run --user --quiet --unit=loom-vtest-backend --working-directory="$L" \
  -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS \
  "${LOOM_BIN:-$L/.venv/bin/loom-serve}" --model="$L/$PKG/$CARD" "--device=amdgpu://1?completion_spin_ns=4294967295" \
  --state_slab_bytes=4mib --weight_slab_bytes=64mib ${LOOM_EXTRA:-} --listen=127.0.0.1:$BP
t0=$SECONDS; while :; do s=$(curl -s -m 2 http://127.0.0.1:$BP/v1/status 2>/dev/null)
  grep -q '"phase":"idle"' <<<"$s" && break
  if grep -qE '"phase":"failed"|"registration":"failed"' <<<"$s" || ! systemctl --user is-active -q loom-vtest-backend; then
    echo "LOAD FAILED: $(echo "$s" | head -c 800)"; journalctl --user -u loom-vtest-backend -n 12 -o cat --no-pager | cut -c1-250; systemctl --user stop loom-vtest-backend; exit 1; fi
  (( SECONDS-t0 > 900 )) && { echo TIMEOUT; systemctl --user stop loom-vtest-backend; exit 1; }; sleep 2; done
echo "loaded in $((SECONDS-t0))s; image component: $(echo "$s" | grep -oE '"image":\{[^}]*\}' | head -1 | cut -c1-200)"
systemd-run --user --quiet --unit=loom-vtest-proxy -E LOOM_NCTX=$NCTX -E LOOM_VISION=1 /usr/bin/python3 -I $L/proxy.py $PP $BP "$MID"; sleep 1.5
echo "VRAM R9700 after load: $(rocm-smi --showmeminfo vram | awk '/GPU\[1\].*Used/{printf "%.1f GB",$NF/1e9}')"
[[ -n "${NOSMOKE:-}" ]] && exit 0; python3 -I ~/Desktop/local_llm/bench-scripts/loom-vision-smoke.py http://127.0.0.1:$PP; rc=$?
echo "VRAM R9700 after images: $(rocm-smi --showmeminfo vram | awk '/GPU\[1\].*Used/{printf "%.1f GB",$NF/1e9}')  smoke rc=$rc"
[[ -z "${KEEP:-}" ]] && systemctl --user stop loom-vtest-proxy loom-vtest-backend
exit $rc
