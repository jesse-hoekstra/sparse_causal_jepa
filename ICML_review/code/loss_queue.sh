#!/usr/bin/env bash
# Run a method's pilot and any qualified expansion from a shared-checkout gateway.
set -euo pipefail
revision=${1:?commit}
suite=${2:?vision-loss, synthetic-loss, or vision-direct}
worker=${3:?primary worker}
second=${4:-"$worker"}
root=$(git rev-parse --show-toplevel)
cd "$root"
check_head() { [[ "$(git rev-parse HEAD)" == "$revision" ]]; }
check_head
case "$suite" in
  synthetic-loss) pilot=3; others=(0 1 2 4); module=synthetic_loss ;;
  vision-loss) pilot=0; others=(1 2 3 4); module=vision_loss ;;
  vision-direct) pilot=0; others=(1 2 3 4); module=vision_acquisition ;;
  *) exit 2 ;;
esac
printf 'Starting %s pilot %s at %s on commit %s\n' "$suite" "$pilot" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)"

run_on() {
  local node=$1 seed=$2
  shift 2
  local cmd q
  check_head
  printf -v cmd 'cd %q && test "$(git rev-parse HEAD)" = %q && ' "$root" "$revision"
  if [[ ${1:-} != assess ]]; then
    cmd+='while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || [[ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 8192 ]]; do sleep 45; done && '
  fi
  printf -v q '%q ' .venv/bin/python -m "ICML_review.code.$module" "$revision" "$seed" "$@"
  cmd+='export PYTHONPATH="$PWD:$PWD/src" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 && '
  printf '%s seed %s, phase %s, worker %s at %s\n' "$suite" "$seed" "${*:-train}" "$node" "$(date -u +%FT%TZ)"
  ssh -o BatchMode=yes "$node" "$cmd$q"
  check_head
}

run_seed() {
  local seed=$1 receipt="ICML_review/outputs/$suite-s$1/metrics.json" status=0
  if [[ "$suite" == vision-direct ]]; then
    run_on "$worker" "$seed" direct
  elif [[ "$suite" == vision-loss ]]; then
    run_on "$worker" "$seed"
  else
    run_on "$worker" "$seed" acquire
    if [[ ! -f "$receipt" ]]; then
      run_on "$worker" "$seed" prepare
      if [[ "$worker" == "$second" ]]; then
        run_on "$worker" "$seed" baseline
        run_on "$worker" "$seed" full
      else
        run_on "$worker" "$seed" full & local a=$!
        run_on "$second" "$seed" baseline & local b=$!
        wait "$a" || status=$?
        wait "$b" || status=$?
        [[ $status == 0 ]] || return "$status"
      fi
      run_on "$worker" "$seed" assess
    fi
  fi
  [[ -s "$receipt" ]]
}
run_seed "$pilot"
if .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["passed"] and r["commit"]==sys.argv[2] else 1)' "ICML_review/outputs/$suite-s$pilot/metrics.json" "$revision"; then
  for seed in "${others[@]}"; do run_seed "$seed"; done
else
  echo 'Pilot ended without qualifying for expansion.'
fi
check_head
