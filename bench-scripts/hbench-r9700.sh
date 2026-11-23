#!/bin/bash
# hbench-r9700.sh (R9700 copy of hbench.sh) VER NAME "FLAGS" PASSES [ENV=VAL ...] : hipfire serve on the XTX at :11446, BetterBench, one summary line.
#   VER = beta (~/.hipfire-beta) or stable (~/.hipfire 0.4.7). SMALL=1 / PFONLY=1 as usual; NEEDLE=<words> [POS=...] after.
VER=$1; NAME=$2; FLAGS=$3; PASSES=${4:-3}; shift 4
HH=$HOME/.hipfire-r9700   # beta binaries, separate home (R9700 JIT cache)
TAG="qwen3.8:27b-mq4-xts"; P=11447; U=hipfire-r9700-bench
OUTD=~/Desktop/local_llm/hipfire-r9700-bb; mkdir -p $OUTD
systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null
envs=(-E HIP_VISIBLE_DEVICES=1 -E HIPFIRE_HOME="$HH"); for e in "$@"; do envs+=(-E "$e"); done
systemd-run --user --quiet --unit=$U "${envs[@]}" $HH/bin/hipfire serve "$TAG" 127.0.0.1:$P $FLAGS --idle-timeout 0
t0=$SECONDS; until curl -sf -m 2 localhost:$P/health 2>/dev/null | grep -q "$TAG"; do
  systemctl --user is-active -q $U || { echo "$NAME: SERVER EXITED"; journalctl --user -u $U -n 12 -o cat --no-pager | cut -c1-200; exit 1; }
  (( SECONDS-t0 > 600 )) && { echo "$NAME: timeout"; systemctl --user stop $U; exit 1; }; sleep 2; done
ctx=$(journalctl --user -u $U -o cat --no-pager | awk '/^Started/{b=""} {b=b"\n"$0} END{print b}' | grep -oE '^max_seq: [0-9]+' | tail -1)
B=~/Desktop/local_llm/tools/BetterBench; OUT=$OUTD/$NAME.json
(cd $B && BETTERBENCH_NO_UPDATE_CHECK=1 timeout 4000 .venv/bin/betterbench run --endpoint http://127.0.0.1:$P/v1 --model "$TAG" --config config/qwen-thinking-xtx.json --passes $PASSES --warmup 1 --no-concurrency ${SMALL:+--no-prefill --categories prose chat code reasoning json} ${PFONLY:+--prefill} --max-model-len 32768 --note arm=$NAME --out $OUT >${OUT%.json}.log 2>&1)
[[ -n "${NEEDLE:-}" ]] && echo "$NAME needle: $(MODEL=$TAG POS=${POS:-0.5} python3 -I ~/Desktop/local_llm/bench-scripts/needle.py $P $NEEDLE)"
systemctl --user stop $U
$B/.venv/bin/betterbench report $OUT 2>/dev/null | python3 -c "
import sys,re
t=sys.stdin.read(); comb=re.search(r'decode t/s median ≈ \*\*([\d.]+)',t); cats={}
for l in t.splitlines():
    c=[x.strip() for x in l.split('|')]
    if len(c)>10 and c[1] in ('prose','reasoning','code','json','chat') and re.match(r'^[\d.]+$',c[2] or ''): cats[c[1]]=f'{c[8]}tg'
pp=[c.split('|')[5].strip() for c in t.splitlines() if re.match(r'^\| (2000|8000|16000) ',c)]
print('$NAME passes=$PASSES: combined',comb.group(1) if comb else '?','|',' '.join(f'{k}={v}' for k,v in cats.items()),'| pp',' / '.join(pp),'| $ctx')"
