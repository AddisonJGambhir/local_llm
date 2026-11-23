#!/bin/bash
cd ~/Desktop/local_llm/bench-scripts || exit 1
while pgrep -f '[r]9700-queue.sh' >/dev/null; do sleep 5; done
echo "== 1b hipfire fp8 393K yarn1.5 $(date +%T)"
SMALL=1 NEEDLE=370000 POS=0.5 ./hbench-r9700.sh beta r-fp8-393k-yarn "--spec dflash --max-seq 393216" 1 HIPFIRE_YARN_FACTOR=1.5
journalctl --user -u hipfire-r9700-bench -o cat --no-pager | grep -E 'load summary|refus|exceeds' | tail -2 | cut -c1-300
echo "== done $(date +%T)"
