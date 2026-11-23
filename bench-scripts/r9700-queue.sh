#!/bin/bash
cd ~/Desktop/local_llm/bench-scripts || exit 1
echo "== 1 hipfire fp8 393K yarn1.5 $(date +%T)"
SMALL=1 NEEDLE=370000 POS=0.5 ./hbench-r9700.sh beta r-fp8-393k-yarn "--spec dflash --ctx 393216" 1 HIPFIRE_YARN_FACTOR=1.5
echo "== 2 hipfire MTP fp8 $(date +%T)"
./hbench-r9700.sh beta r-mtp-fp8 "--spec mtp" 3 HIPFIRE_MTP_K=3 HIPFIRE_PREFILL_MAX_BATCH=384
echo "== 3 loom r9700-vision needle 240K $(date +%T)"
KEEP=1 ./lvtest.sh Inkling-2-Mini-r9700-c262144-d3-vision model.json 262144 Inkling-2-Mini | tail -3
echo "needle: $(POS=0.5 python3 -I needle.py 1239 240000)"
systemctl --user stop loom-vtest-proxy loom-vtest-backend
echo "== done $(date +%T)"
