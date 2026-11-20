#!/usr/bin/env bash
# One training seed of the vision experiment on one GPU, with the reported settings:
#   joint 100k updates (code dimension 8; SIGReg 0.05 on the states; CNN lr 1e-4, rest 1e-3)
#   -> frozen-encoder cache -> 16 code/predictor acquisitions of 100k updates
#   (acquisition seeds 1000 + 100 j + seed) -> selection by mean TRAIN prediction loss over
#   the last 5,000 updates -> projection 8 -> 4 -> adaptation 50k -> tau calibration
#   -> baseline and full sparse arms at the constant GECO tolerance 2 tau for 300k updates,
#   continued exactly (model, optimizer, sampler, GECO and RNG states) to 500k by `train fork`.
# Training never reads physical labels. Stages are resumable: rerunning skips completed stages.
#
# usage: run_seed.sh SEED DATA_DIR OUTPUT_ROOT [PARALLEL_ACQUISITIONS]
#   DATA_DIR from `python -m diagnostics.vision_dimer.data --data-dir DATA_DIR`;
#   outputs go to OUTPUT_ROOT/seed<SEED>/. Environment: PYTHON, CUDA_VISIBLE_DEVICES,
#   WANDB_MODE (default disabled), WANDB_GROUP (default vision-dimer), WANDB_ENTITY.
set -euo pipefail
SEED=$1 DATA=$2 ROOT=$3 WAVE=${4:-8}
CODE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-$CODE_DIR/.venv/bin/python}
export PYTHONPATH="$CODE_DIR:$CODE_DIR/src"
B=$ROOT/seed$SEED
TRAIN=("$PYTHON" -m diagnostics.vision_dimer.train)
WANDB=(--wandb-mode "${WANDB_MODE:-disabled}" --wandb-group "${WANDB_GROUP:-vision-dimer}-s$SEED")
log() { printf '%s seed %s: %s\n' "$(date -u +%FT%TZ)" "$SEED" "$*"; }
cd "$CODE_DIR"
mkdir -p "$B"

stage() {  # stage OUTPUT_DIR ARGS...: one training stage with its own log, resumed if interrupted
  local out=$1
  shift
  [ -f "$out/final.json" ] && return 0
  mkdir -p "$out"
  local resume=()
  [ -f "$out/last.pt" ] && resume=(--resume)
  "${TRAIN[@]}" train --output-dir "$out" "$@" "${WANDB[@]}" ${resume[@]+"${resume[@]}"} \
    >> "$out/log.txt" 2>&1
}
require() {  # require DIR...: every listed stage completed
  local dir
  for dir in "$@"; do
    [ -f "$dir/final.json" ] || { log "stage $dir did not complete; see $dir/log.txt"; exit 1; }
  done
}

log "joint stage"
stage "$B/joint" --stage joint --data-dir "$DATA" --seed "$SEED" --steps 100000 --code-dim 8 \
  --learning-rate 1e-3 --encoder-learning-rate 1e-4 --sigreg-weight 0.05 \
  --eval-every 10000 --log-every 500 --checkpoint-every 5000 --device cuda
"${TRAIN[@]}" cache --parent "$B/joint/final.pt" --data-dir "$DATA" --device cuda

runs=()
for first in $(seq 0 "$WAVE" 15); do
  for j in $(seq "$first" $((first + WAVE - 1))); do
    [ "$j" -le 15 ] || continue
    stage "$B/acquire/a$j" --stage acquire --data-dir "$DATA" --parent "$B/joint/final.pt" \
      --seed "$SEED" --acquisition-seed $((1000 + 100 * j + SEED)) --steps 100000 --code-dim 8 \
      --learning-rate 1e-3 --eval-every 100000 --log-every 500 --checkpoint-every 20000 \
      --device cuda &
    runs+=("$B/acquire/a$j")
  done
  wait
  log "acquisitions from a$first done"
done
require "${runs[@]}"
"${TRAIN[@]}" select --runs "${runs[@]}" --output "$B/selection.json"
SELECTED=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' \
  "$B/selection.json")
[ -f "$B/projected/final.pt" ] ||
  "${TRAIN[@]}" project --source "$SELECTED" --data-dir "$DATA" --output "$B/projected" --device cuda
stage "$B/adapt" --stage adapt --data-dir "$DATA" --parent "$B/projected/final.pt" \
  --seed "$SEED" --steps 50000 --code-dim 4 --learning-rate 1e-3 \
  --eval-every 100000 --log-every 500 --checkpoint-every 10000 --device cuda
require "$B/adapt"
"${TRAIN[@]}" calibrate --source "$B/adapt" --data-dir "$DATA" --output "$B/tau.json" --device cuda
TAU=$("$PYTHON" -c 'import json,sys; print(repr(2.0 * json.load(open(sys.argv[1]))["tau"]))' \
  "$B/tau.json")
log "dense initializer ready; sparse tolerance 2 tau = $TAU"

SPARSE=(--data-dir "$DATA" --parent "$B/adapt/final.pt" --seed "$SEED" --tau "$TAU" --code-dim 4
  --learning-rate 1e-3 --encoder-learning-rate 1e-4 --gate-learning-rate 1e-3 --initial-logit 3
  --sigreg-weight 0.05 --eval-every 25000 --log-every 500 --checkpoint-every 10000 --device cuda)
for arm in full baseline; do
  stage "$B/sparse-300k/$arm" --stage "$arm" --steps 300000 "${SPARSE[@]}" &
done
wait
require "$B/sparse-300k/full" "$B/sparse-300k/baseline"
for arm in full baseline; do
  out=$B/sparse-500k/$arm
  [ -f "$out/final.json" ] && continue
  mkdir -p "$out"
  "${TRAIN[@]}" fork --source "$B/sparse-300k/$arm/last.pt" --output-dir "$out" --steps 500000 \
    "${WANDB[@]}" >> "$out/log.txt" 2>&1 &
done
wait
require "$B/sparse-500k/full" "$B/sparse-500k/baseline"
log "complete"
