#!/usr/bin/env bash
# Fixed pool of eight acquisition initializations for each of five frozen-state seeds.
# Run in detached tmux on the server. No sparse phase or test evaluation is launched.
set -euo pipefail
if [ "$#" -ne 7 ]; then
  echo "usage: $0 TAG STATE_BATCH HOST0 HOST1 HOST2 HOST3 HOST4" >&2
  exit 2
fi
TAG=$1
STATE_BATCH=$2
shift 2
HOSTS=("$@")
REPO_ROOT=/nfs/home/jesseh/sparse_causal_jepa
CODE_DIR=${CODE_DIR:-$REPO_ROOT}
PYTHON=$REPO_ROOT/.venv/bin/python
BATCH=$REPO_ROOT/outputs/diagnostics/masked-mlp-restarts-$TAG
DATA=$REPO_ROOT/outputs/diagnostics/balanced-action-prepare-balanced-20261028T202921Z-b577779/results
mkdir -p "$BATCH"
if [ -e "$BATCH/protocol.txt" ]; then
  echo "refusing to overwrite an existing batch: $BATCH" >&2
  exit 2
fi
printf '%s\n' "code=$CODE_DIR" "state_batch=$STATE_BATCH" "hosts=${HOSTS[*]}" \
  "attempts=8; updates=100000; initialization_seed=1000+100*attempt+state_seed" \
  "selection=mean logged TRAIN prediction, steps 95100..100000; maximum=0.025" \
  "no sparse launch; no test evaluation" > "$BATCH/protocol.txt"
git -C "$CODE_DIR" rev-parse HEAD >> "$BATCH/protocol.txt"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$BATCH/coordinator.log"; }
on_host() {
  local host=$1
  shift
  if [ "$(hostname -s)" = "omi-rapid-$host" ]; then
    bash -c "$*"
  else
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "omi-rapid-$host" "$@"
  fi
}
idle() {
  local memory utilization
  read -r memory utilization < <(on_host "$1" \
    "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" | tr ',' ' ')
  [ "${memory:-999999}" -lt 1000 ] && [ "${utilization:-100}" -lt 5 ]
}

log "waiting for all five GPUs to be idle before the first wave"
for attempt in {0..7}; do
  while true; do
    ready=1
    for host in "${HOSTS[@]}"; do idle "$host" || ready=0; done
    [ "$ready" -eq 1 ] && break
    sleep 30
  done
  log "starting acquisition wave $attempt on all five hosts"
  for seed in {0..4}; do
    host=${HOSTS[$seed]}
    job=$BATCH/attempt-$attempt-s$seed
    mkdir "$job"
    session=scj-acquire-$TAG-a$attempt-s$seed
    printf '%s\n' "$host" > "$job/host.txt"
    printf '%s\n' "$session" > "$job/session.txt"
    {
      printf '#!/usr/bin/env bash\nset -uo pipefail\ncd %q\n' "$CODE_DIR"
      printf 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
      printf '%q ' "$PYTHON" -u -m diagnostics.masked_mlp_learned.run train \
        --condition dense --seed "$seed" --acquisition-seed "$((1000 + 100 * attempt + seed))" \
        --steps 100000 --states-from "$STATE_BATCH/states-s$seed/results" \
        --data-dir "$DATA" --output-dir "$job/results" --device cuda \
        --wandb-mode online --wandb-project sparse_causal_jepa \
        --wandb-entity jesse-hoekstra-university-of-oxford --wandb-group "masked-mlp-restarts-$TAG"
      printf '> %q 2>&1\ncode=$?\nprintf "%%s\\n" "$code" > %q\nexit "$code"\n' \
        "$job/stdout-stderr.log" "$job/exit-code.txt"
    } > "$job/run.sh"
    on_host "$host" "tmux new-session -d -s '$session' 'bash $job/run.sh'"
  done
  while true; do
    finished=1
    for seed in {0..4}; do
      job=$BATCH/attempt-$attempt-s$seed
      if [ ! -f "$job/exit-code.txt" ]; then
        finished=0
      elif [ "$(cat "$job/exit-code.txt")" != 0 ]; then
        log "failure in $job; stopping this batch and its remaining jobs"
        for other in {0..4}; do
          other_job=$BATCH/attempt-$attempt-s$other
          on_host "${HOSTS[$other]}" "tmux kill-session -t '$(cat "$other_job/session.txt")'" || true
        done
        exit 1
      fi
    done
    [ "$finished" -eq 1 ] && break
    sleep 30
  done
  log "wave $attempt completed"
done

cd "$CODE_DIR"
admitted=1
for seed in {0..4}; do
  candidates=()
  for attempt in {0..7}; do candidates+=("$BATCH/attempt-$attempt-s$seed/results"); done
  "$PYTHON" -m diagnostics.masked_mlp_learned.selection --runs "${candidates[@]}" \
    --output "$BATCH/selection-s$seed.json" --window 5000 --maximum-loss 0.025 \
    > "$BATCH/selection-s$seed.log" 2>&1 || admitted=0
done
if [ "$admitted" -ne 1 ]; then
  log "acquisition screen failed; all endpoints retained; no sparse phase launched"
  exit 2
fi
log "all five TRAIN screens pass; selected endpoints are ready for projection and retention audit"
