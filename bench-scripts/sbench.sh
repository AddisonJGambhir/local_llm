#!/bin/bash
# sbench.sh VER NAME PASSES 'python edits to dict c (setarg/delarg)'
#   VER = 0213 (Desktop/strata-0213) or 0219 (Desktop/strata-0219). Both GPUs, :8103, BetterBench, one summary line.
#   SMALL=1 -> decode-only 5 categories; PFONLY=1 -> prefill only.
VER=$1; NAME=$2; PASSES=$3; EDIT=${4:-pass}
SD=~/Desktop/strata; OUTD=$SD/logs/bb; mkdir -p $OUTD; C=$OUTD/$NAME.json.cfg
if [[ $VER == 0219* ]]; then
  EXE=$HOME/Desktop/strata-0219/engine/build/hip-compile/strata; [[ $VER == 0219r ]] && EXE=$HOME/Desktop/strata-0219/engine/build/hip-rocm10/strata; WD=$HOME/Desktop/strata-0219/terminal; PY=$WD/.venv/bin/python
else
  EXE=$HOME/Desktop/strata-0213/build-hip/strata; WD=$HOME/Desktop/strata-0213; PY=$WD/.venv/bin/python
fi
python3 - "$C" <<PY
import json,sys
c=json.load(open('$SD/configs/iq4xs.json')); c['port']=8103; c['log']='$OUTD/$NAME.engine.log'; c['exe']='$EXE'
a=c['args']
def setarg(k,v):
    if k in a: a[a.index(k)+1]=v
    else: a.extend([k,v])
def delarg(k):
    if k in a: i=a.index(k); del a[i:i+2]
$EDIT
json.dump(c,open(sys.argv[1],'w'),indent=1)
PY
U=strata-bench; systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null
systemd-run --user --quiet --unit=$U --working-directory=$WD $PY -m serve.server --engine strata --config $C --host 127.0.0.1 --port 8103
t0=$SECONDS; until curl -s -m 2 localhost:8103/health | grep -q '"loaded": true'; do
  systemctl --user is-active -q $U || { echo "$NAME: SERVER EXITED"; journalctl --user -u $U -n 15 -o cat --no-pager | cut -c1-200; exit 1; }
  (( SECONDS-t0 > 900 )) && { echo "$NAME: timeout"; systemctl --user stop $U; exit 1; }; sleep 3; done
v=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[0\].*Used/{x=$NF} /GPU\[1\].*Used/{r=$NF} END{printf "XTX %.1f / R9700 %.1f GB",x/1e9,r/1e9}')
B=~/Desktop/local_llm/tools/BetterBench; OUT=$OUTD/$NAME.json
(cd $B && BETTERBENCH_NO_UPDATE_CHECK=1 timeout 5000 .venv/bin/betterbench run --endpoint http://127.0.0.1:8103/v1 --model local --config config/qwen-thinking-xtx.json --passes $PASSES --warmup 1 --no-concurrency ${SMALL:+--no-prefill --categories prose chat code reasoning json} ${PFONLY:+--prefill} --max-model-len 32768 --note arm=$NAME --out $OUT >${OUT%.json}.log 2>&1)
systemctl --user stop $U; sleep 5; for e in $(pgrep -f "^$EXE --serve"); do kill -KILL $e; done
L=$OUTD/$NAME.engine.log
$B/.venv/bin/betterbench report $OUT 2>/dev/null | python3 -c "
import sys,re
t=sys.stdin.read(); comb=re.search(r'decode t/s median ≈ \*\*([\d.]+)',t); cats={}
for l in t.splitlines():
    c=[x.strip() for x in l.split('|')]
    if len(c)>10 and c[1] in ('prose','reasoning','code','json','chat') and re.match(r'^[\d.]+$',c[2] or ''): cats[c[1]]=f'{c[8]}tg'
pp=[c.split('|')[5].strip() for c in t.splitlines() if re.match(r'^\| (2000|8000|16000) ',c)]
print('$NAME passes=$PASSES: combined',comb.group(1) if comb else '?','|',' '.join(f'{k}={v}' for k,v in cats.items()),'| pp',' / '.join(pp),'| $v | engine starts $(grep -c 'engine started' $L 2>/dev/null) errs $(grep -c 'native ERR' $L 2>/dev/null)')"
