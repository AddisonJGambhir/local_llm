#!/bin/bash
# loom-watchdog.sh BPORT TAG : if Loom marks the model failed (e.g. GPU memory exhausted), restart loom-backend
# in place and re-pin the model, so :1234 recovers by itself instead of returning 503 "model is unavailable" forever.
BPORT=$1; TAG=$2; UNIT=${3:-loom-backend}
pin() { curl -s -m 30 -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$TAG\"}" "http://127.0.0.1:$BPORT/v1/models/pin" >/dev/null; }
while sleep 10; do
  systemctl --user is-active -q $UNIT || continue
  s=$(curl -s -m 3 "http://127.0.0.1:$BPORT/v1/status" 2>/dev/null) || continue
  if grep -q '"registration":"failed"' <<<"$s"; then
    echo "$(date '+%F %T') model failed: $(grep -oE '"error":\{[^}]*\}' <<<"$s" | head -1) - restarting $UNIT"
    systemctl --user restart $UNIT
    for _ in $(seq 300); do curl -s -m 2 "http://127.0.0.1:$BPORT/v1/status" 2>/dev/null | grep -q '"phase":"idle"' && break; sleep 2; done
    pin; echo "$(date '+%F %T') recovered and re-pinned"
  fi
done
