#!/bin/bash
# rbench.sh NAME BASE_PROFILE PASSES ["key: value" ...] : radiance on the R9700 (:8101), BetterBench, one summary line.
#   SMALL=1 decode-only; PFONLY=1 prefill-only; KEEP=1 leaves the server up; NEEDLE=<words> [POS=0.1,0.5,0.9] runs needle tests after.
NAME=$1; BASE=$2; PASSES=${3:-1}; shift 3
R=~/Desktop/local_llm/radiance; C=$R/configs/tune/$NAME.yaml; mkdir -p $R/configs/tune $R/bb
sed 's/^  port: 1234/  port: 8101/' $R/configs/$BASE.yaml > $C
for kv in "$@"; do k=${kv%%:*}; grep -q "^    $k:" $C && sed -i "s|^    $k:.*|    $kv|" $C || echo "    $kv" >> $C; done
U=radiance-tune; systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null
systemd-run --user --quiet --unit=$U --working-directory=$R -E ROCR_VISIBLE_DEVICES=1 -E RADIANCE_HOME=$R/src/build/radiance_home $R/src/build/bin/radiance --config $C
t0=$SECONDS; until curl -sf -m 2 localhost:8101/health >/dev/null; do
  systemctl --user is-active -q $U || { echo "$NAME: SERVER EXITED"; journalctl --user -u $U -n 15 -o cat --no-pager | cut -c1-200; exit 1; }
  (( SECONDS-t0 > 900 )) && { echo "$NAME: load timeout"; systemctl --user stop $U; exit 1; }; sleep 2; done
kvline=$(journalctl --user -u $U -o cat --no-pager | awk '/^Started radiance-tune/{b=""} {b=b"\n"$0} END{print b}' | grep -m1 -oE 'kv cache +[0-9.]+ GiB a card: [0-9]+ tokens')
echo "$NAME: loaded in $((SECONDS-t0))s, VRAM $(rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\].*Used/{printf "%.1f GB",$NF/1e9}'), $kvline"
B=~/Desktop/local_llm/tools/BetterBench; OUT=$R/bb/$NAME.json
(cd $B && BETTERBENCH_NO_UPDATE_CHECK=1 timeout 3000 .venv/bin/betterbench run --endpoint http://127.0.0.1:8101/v1 --model local --config config/qwen-thinking-xtx.json --passes $PASSES --warmup 1 --no-concurrency ${SMALL:+--no-prefill --categories prose chat code reasoning json} ${PFONLY:+--prefill} --max-model-len 32768 --note arm=radiance-$NAME --out $OUT >$R/bb/$NAME.log 2>&1)
$B/.venv/bin/betterbench report $OUT 2>/dev/null | python3 -c "
import sys,re
t=sys.stdin.read(); comb=re.search(r'decode t/s median ≈ \*\*([\d.]+)',t); cats={}
for l in t.splitlines():
    c=[x.strip() for x in l.split('|')]
    if len(c)>10 and c[1] in ('prose','reasoning','code','json','chat') and re.match(r'^[\d.]+$',c[2] or ''): cats[c[1]]=f'{c[8]}tg'
pp=[c.split('|')[5].strip() for c in t.splitlines() if re.match(r'^\| (2000|8000|16000) ',c)]
print('$NAME passes=$PASSES: combined',comb.group(1) if comb else '?','|',' '.join(f'{k}={v}' for k,v in cats.items()),'| pp',' / '.join(pp))"
if [[ -n "${NEEDLE:-}" ]]; then echo "$NAME needle $NEEDLE: $(POS=${POS:-0.5} python3 -I ~/Desktop/local_llm/bench-scripts/needle.py 8101 $NEEDLE)"; fi
[[ -z "${KEEP:-}" ]] && systemctl --user stop $U
