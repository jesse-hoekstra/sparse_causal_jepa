#!/usr/bin/env bash
# Five fixed seeds, one worker per host; immutable source and fresh outputs only.
set -euo pipefail
CODE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-$CODE_DIR/.venv/bin/python}
export PYTHONPATH="$CODE_DIR/src:$CODE_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE_DIR"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
run_job() {
  local name=$1 status=0
  shift
  mkdir "$BATCH/$name"
  printf '%q ' "$PYTHON" -u -m "$@" > "$BATCH/$name/command.sh"
  printf '\n' >> "$BATCH/$name/command.sh"
  log "starting $name on $(hostname -s)"
  "$PYTHON" -u -m "$@" > "$BATCH/$name/stdout-stderr.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$BATCH/$name/exit-code.txt"
  log "finished $name with exit $status"
  [ "$status" -eq 0 ] || exit "$status"
}

if [ "${1:-}" = worker ]; then
  PHASE=$2 BATCH=$3 DATA=$4 SEED=$5
  trap 'status=$?; printf "%s\n" "$status" > "$BATCH/$PHASE-s$SEED.exit"' EXIT
  LOGGING=(--wandb-mode online --wandb-project sparse_causal_jepa
    --wandb-entity jesse-hoekstra-university-of-oxford --wandb-group "$(basename "$BATCH")")
  if [ "$PHASE" = warmup ]; then
    run_job "states-s$SEED" diagnostics.masked_mlp_learned.run train \
      --data-dir "$DATA" --output-dir "$BATCH/states-s$SEED/results" \
      --condition dense --seed "$SEED" --steps 100000 --code-dim 12 \
      --initializer shared_feature --device cuda "${LOGGING[@]}"
    CANDIDATES=()
    for ATTEMPT in {0..15}; do
      NAME=acquisition-$ATTEMPT-s$SEED
      run_job "$NAME" diagnostics.masked_mlp_learned.run train \
        --data-dir "$DATA" --output-dir "$BATCH/$NAME/results" \
        --condition dense --seed "$SEED" --steps 100000 --code-dim 12 \
        --initializer shared_feature --states-from "$BATCH/states-s$SEED/results" \
        --acquisition-seed "$((1000 + 100 * ATTEMPT + SEED))" --device cuda "${LOGGING[@]}"
      CANDIDATES+=("$BATCH/$NAME/results")
    done
    "$PYTHON" -m diagnostics.masked_mlp_learned.selection --runs "${CANDIDATES[@]}" \
      --output "$BATCH/selection-s$SEED.json" --window 5000 --maximum-loss 0.025 \
      > "$BATCH/selection-s$SEED.log" 2>&1
    SELECTED=$("$PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["directory"])' \
      "$BATCH/selection-s$SEED.json")
    run_job "projected-s$SEED" diagnostics.masked_mlp_learned.run project \
      --data-dir "$DATA" --dense-dir "$SELECTED" \
      --output-dir "$BATCH/projected-s$SEED/results" --device cuda
    run_job "dense-s$SEED" diagnostics.masked_mlp_learned.run train \
      --data-dir "$DATA" --output-dir "$BATCH/dense-s$SEED/results" \
      --condition dense --seed "$SEED" --steps 50000 \
      --adapt-from "$BATCH/projected-s$SEED/results" --device cuda "${LOGGING[@]}"
  elif [ "$PHASE" = sparse ]; then
    for CONDITION in baseline full; do
      run_job "$CONDITION-s$SEED" diagnostics.masked_mlp_learned.budget_control \
        --data-dir "$DATA" --output-dir "$BATCH/$CONDITION-s$SEED/results" \
        --condition "$CONDITION" --seed "$SEED" --init-dir "$BATCH/dense-s$SEED/results" \
        --calibration-dir "$BATCH/calibration" --device cuda "${LOGGING[@]}"
    done
  else
    log "unknown worker phase: $PHASE"
    exit 2
  fi
  exit 0
fi

if [ "$#" -ne 9 ]; then
  echo "usage: $0 BATCH DATA SIGMA TAU_FACTOR HOST0 HOST1 HOST2 HOST3 HOST4" >&2
  exit 2
fi
BATCH=$1 DATA=$2 SIGMA=$3 TAU_FACTOR=$4
shift 4
HOSTS=("$@")
[[ "$BATCH" = /* && "$DATA" = /* ]] || { echo "paths must be absolute" >&2; exit 2; }
[ ! -e "$BATCH" ] && [ ! -e "$DATA" ] && [ ! -e "$DATA-physical" ] || {
  echo "require fresh batch and data paths" >&2; exit 2;
}
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "require a clean pinned checkout" >&2; exit 2;
}
mkdir -p "$BATCH"
cp "$0" "$BATCH/launch-source.sh"
printf '%q ' bash "$CODE_DIR/diagnostics/masked_mlp_learned/experiment_batch.sh" \
  "$BATCH" "$DATA" "$SIGMA" "$TAU_FACTOR" "${HOSTS[@]}" > "$BATCH/launch-command.sh"
printf '\n' >> "$BATCH/launch-command.sh"
git rev-parse HEAD > "$BATCH/source-commit.txt"
cat > "$BATCH/protocol.txt" <<EOF
Hypothesis: the five-seed state/graph recovery result persists with sigma=$SIGMA
and tau=$TAU_FACTOR times the adapted seed-zero calibration MSE.
Only these two scientific settings change. Data seed=0; observation seed=1729.
Seeds=0..4; hosts=${HOSTS[*]}; code=$CODE_DIR; data=$DATA.
Warm-up=joint12-code100k;16 frozen-state restarts100k;TRAIN-only selection;
last5000 mean <=.025;12-to-6 shared-covariance projection;adapt50k.
Acquisition seeds=1000+100*attempt+state_seed, attempt=0..15.
Matched baseline/full=300k each. No parameter/state labels in training or selection.
Validation only. Test evaluation is not authorized.
Assess all fixed endpoints: full-baseline MCC, baseline-dense MCC, block purity,
exact state/code-aligned parameter graphs, all-six retention R2>=.80, forecast quality.
Success: export and compare with the retained result. Failure: retain and report
all endpoints; no automatic retuning, extra restarts, or budget extensions.
Failed TRAIN admission or software failure stops this batch's workers.
EOF

on_host() {
  local host=$1
  shift
  if [ "$(hostname -s)" = "omi-rapid-$host" ]; then
    bash -c "$*"
  else
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "omi-rapid-$host" "$@"
  fi
}
stop_workers() {
  local host session
  [ -f "$BATCH/sessions.tsv" ] || return 0
  while read -r host session; do
    on_host "$host" "tmux kill-session -t '$session'" 2>/dev/null || true
  done < "$BATCH/sessions.tsv"
}
finish() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then stop_workers; fi
  printf '%s\n' "$status" > "$BATCH/coordinator.exit"
  log "coordinator exit $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM

# Other users may share these hosts. Require at least 6 GiB free before launching.
for host in "${HOSTS[@]}"; do
  free=$(on_host "$host" \
    'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits' | head -1)
  [[ "$free" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && [ "$free" -ge 6144 ] || {
    log "insufficient GPU memory on $host: $free MiB"; exit 2;
  }
  on_host "$host" "test -x '$PYTHON' && test -r '$CODE_DIR/diagnostics/shared/prepare.py'"
done

log "preparing sigma=$SIGMA data"
"$PYTHON" -m diagnostics.shared.prepare --output-dir "$DATA" --rotation-scale "$SIGMA" \
  --wandb-mode online --wandb-project sparse_causal_jepa \
  --wandb-entity jesse-hoekstra-university-of-oxford --wandb-group "$(basename "$BATCH")" \
  > "$BATCH/data-preparation.log" 2>&1

for PHASE in warmup sparse; do
  if [ "$PHASE" = sparse ]; then
    "$PYTHON" -m diagnostics.masked_mlp_learned.run calibrate \
      --dense-dir "$BATCH/dense-s0/results" --output-dir "$BATCH/calibration-original" \
      > "$BATCH/calibration-original.log" 2>&1
    "$PYTHON" -m diagnostics.masked_mlp_learned.bound_control \
      --calibration-dir "$BATCH/calibration-original" --output-dir "$BATCH/calibration" \
      --tau-factor "$TAU_FACTOR" \
      --hypothesis "Recovery persists with sigma=$SIGMA and tau=$TAU_FACTOR times calibration MSE; all other settings fixed." \
      > "$BATCH/calibration.log" 2>&1
  fi
  for seed in {0..4}; do
    host=${HOSTS[$seed]}
    session=scj-$(basename "$BATCH")-$PHASE-s$seed
    worker=$BATCH/$PHASE-s$seed.sh
    {
      printf '#!/usr/bin/env bash\nexec '
      printf '%q ' bash "$CODE_DIR/diagnostics/masked_mlp_learned/experiment_batch.sh" \
        worker "$PHASE" "$BATCH" "$DATA" "$seed"
      printf '> %q 2>&1\n' "$BATCH/$PHASE-s$seed.log"
    } > "$worker"
    printf '%s\t%s\n' "$host" "$session" >> "$BATCH/sessions.tsv"
    on_host "$host" "tmux new-session -d -s '$session' 'bash $worker'"
    log "started $PHASE seed $seed on $host"
  done
  while true; do
    finished=1
    for seed in {0..4}; do
      if [ ! -f "$BATCH/$PHASE-s$seed.exit" ]; then
        finished=0
      elif [ "$(cat "$BATCH/$PHASE-s$seed.exit")" != 0 ]; then
        log "failed $PHASE seed $seed; preserving outputs and stopping this batch"
        exit 1
      fi
    done
    [ "$finished" -eq 1 ] && break
    sleep 30
  done
  log "all five $PHASE workers completed"
done

"$PYTHON" -m diagnostics.masked_mlp_oracle.paper_analysis \
  --batch-dir "$BATCH" --output-dir "$BATCH/paper" > "$BATCH/paper.log" 2>&1
log "fixed endpoints and paper outputs complete: $BATCH/paper"
