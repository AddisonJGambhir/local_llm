#!/bin/bash
# lmax.sh PKG DEV NWORDS : Loom context probe. Loads PKG (dir under ~/Desktop/local_llm/loom) on amdgpu://DEV with the
# launcher's elastic small slabs, records VRAM after load + peak, runs one needle (OTTER-7712 at 60% depth) at NWORDS words.
PKG=$1; DEV=$2; N=$3
cd ~/Desktop/local_llm/loom || exit 1
U=loom-maxx; PORT=8099; systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null
systemd-run --user --quiet --unit=$U --working-directory=$PWD -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS \
  $PWD/.venv/bin/loom-serve --model=$PWD/$PKG "--device=amdgpu://$DEV?completion_spin_ns=4294967295" \
  --state_slab_bytes=4mib --weight_slab_bytes=64mib --listen=127.0.0.1:$PORT
for _ in $(seq 200); do s=$(curl -s localhost:$PORT/v1/status 2>/dev/null); echo "$s" | grep -qE '"phase":"(idle|failed)"|"registration":"failed"' && break; systemctl --user is-active -q $U || break; sleep 3; done
st=$(echo "$s" | python3 -c "import json,sys
try:
  d=json.load(sys.stdin)['data'][0]['loom'];print(d['registration'],d['status']['phase'],(d.get('error') or {}).get('message','')[:160])
except Exception: print('no-status')")
gpu() { rocm-smi --showmeminfo vram | grep "GPU\[$DEV\].*Used" | awk '{print $NF}'; }
load=$(gpu); res="not-run"; PK=$(mktemp)
if echo "$st" | grep -q idle; then
  ( peak=0; while systemctl --user is-active -q $U; do u=$(gpu); [ "$u" -gt "$peak" ] && peak=$u && echo $peak > $PK; sleep 1; done ) &
  res=$(python3 -I - "$N" "$PORT" <<'EOF'
import json,urllib.request,random,sys,time
n=int(sys.argv[1]); port=sys.argv[2]; random.seed(5)
w="the river city market morning train window garden letter winter music engine paper bridge forest".split()
f=lambda k:" ".join(random.choice(w) for _ in range(k))
doc=f(int(n*0.6))+" The secret code for the vault is OTTER-7712. "+f(n-int(n*0.6))
msg=[{"role":"user","content":doc+"\n\nWhat is the secret code for the vault? Reply with only the code."}]
t=time.time()
try:
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps({"model":"Inkling-2-Mini","messages":msg,"max_tokens":4000,"temperature":1.0,"top_p":0.95,"top_k":20}).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=1800)); m=d["choices"][0]["message"]; txt=(m.get("content") or "")+" "+(m.get("reasoning_content") or "")
    print(f"prompt={d['usage']['prompt_tokens']} found={'OTTER-7712' in txt} answer={(m.get('content') or '')[:24]!r} {time.time()-t:.0f}s")
except Exception as e:
    print("REQUEST FAILED", str(e)[:120])
EOF
)
fi
alive=$(systemctl --user is-active $U)
echo "$PKG dev$DEV | $st | VRAM after load $(awk "BEGIN{printf \"%.2f\",$load/1e9}") GB, peak $(awk '{printf "%.2f",$1/1e9}' $PK 2>/dev/null) GB | $res | server $alive"
rm -f $PK; systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null
