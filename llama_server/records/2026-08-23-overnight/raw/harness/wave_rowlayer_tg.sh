#!/usr/bin/env bash
# Decode WITH MTP on row and layer split, same rig as the tensor numbers
# (120k depth, novel source content, temp 0, 2048-tok generations) so they are
# directly comparable to: tensor n5 46.70 / n4+f16 45.79 / n3 42.11.
# Raw (unspeculated) decode favours tensor by 13-18%; the open question is
# whether that survives once MTP multiplies both, given layer prefills 2x faster.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"; source "$SP/lib.sh"
CSV="$SP/rowlayer_tg.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
BASE="-c 131072 --cache-type-k f16 --cache-type-v f16 --ubatch-size 512 --cache-ram 24576"
DKV="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase(){ local label="$1"; shift
  log "=== ROWLAYER $label"; echo "===== $label =====" >> "$SP/server.log"
  start_server "$*" || { log "$label FAILED TO START"; mykill; return 1; }
  log "$label up (peak $(cat $SP/.peak) MiB)"
  python3 "$SP/blockA.py" "$label" "120000" 10 "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "$label done peak $(cat $SP/.peak) MiB"; mykill; }
log "ROW/LAYER DECODE WITH MTP START"
# n3 and n5 on each split; batch 8192 since that is the recommended prefill setting
phase row_n3    "$BASE --batch-size 8192 --split-mode row   --spec-type draft-mtp --spec-draft-n-max 3 $DKV"
phase row_n5    "$BASE --batch-size 8192 --split-mode row   --spec-type draft-mtp --spec-draft-n-max 5 $DKV"
phase layer_n3  "$BASE --batch-size 8192 --split-mode layer --tensor-split 28,37 --spec-type draft-mtp --spec-draft-n-max 3 $DKV"
phase layer_n5  "$BASE --batch-size 8192 --split-mode layer --tensor-split 28,37 --spec-type draft-mtp --spec-draft-n-max 5 $DKV"
log "ROW/LAYER DECODE COMPLETE -> $CSV"
