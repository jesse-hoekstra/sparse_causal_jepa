#!/usr/bin/env bash
# Coordinate from a gateway with shared storage and SSH access to the named workers.
set -euo pipefail
policy=single-acquisition
selection_flag=
if [[ ${1:-} == --first-admitted ]]; then
  policy=single-admitted
  selection_flag=--first-admitted
  shift
fi
revision=${1:?expected commit}
shift
[[ $# == 4 ]]
nodes=("$@")
[[ "$(printf '%s\n' "${nodes[@]}" | sort -u | wc -l)" -eq 4 ]]
root=$(git rev-parse --show-toplevel)
cd "$root"
check_head() { [[ "$(git rev-parse HEAD)" == "$revision" ]]; }
check_head
run_phase() {
  local node=$1 seed=$2 phase=$3 remote
  check_head
  printf -v remote 'cd %q && test "$(git rev-parse HEAD)" = %q && ' "$root" "$revision"
  if [[ "$phase" != assess ]]; then
    remote+='while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || [[ "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 6144 ]]; do sleep 45; done && '
  fi
  printf -v remote '%sexport PYTHONPATH="$PWD:$PWD/src" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 && .venv/bin/python -m ICML_review.code.single_acquisition %q %q %q' "$remote" "$revision" "$seed" "$phase"
  if [[ -n "$selection_flag" ]]; then
    remote+=" $selection_flag"
  fi
  printf 'Starting %s seed %s, worker %s at %s\n' "$phase" "$seed" "$node" "$(date -u +%FT%TZ)"
  ssh -o BatchMode=yes "$node" "$remote"
  check_head
}
train_seed() {
  local seed=$1 first=$2 second=$3 a b status=0
  run_phase "$first" "$seed" prepare
  if [[ -f "ICML_review/outputs/$policy-s$seed/metrics.json" ]]; then
    return
  fi
  run_phase "$first" "$seed" full & a=$!
  run_phase "$second" "$seed" baseline & b=$!
  wait "$a" || status=$?
  wait "$b" || status=$?
  [[ "$status" -eq 0 ]] || return "$status"
  run_phase "$first" "$seed" assess
}
train_seed 0 "${nodes[0]}" "${nodes[1]}"
if .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[2])); sys.exit(0 if r["commit"]==sys.argv[1] and r["passed"] else 1)' "$revision" "ICML_review/outputs/$policy-s0/metrics.json"; then
  for first in 1 3; do
    train_seed "$first" "${nodes[0]}" "${nodes[1]}" & a=$!
    train_seed "$((first+1))" "${nodes[2]}" "${nodes[3]}" & b=$!
    status=0
    wait "$a" || status=$?
    wait "$b" || status=$?
    [[ "$status" -eq 0 ]] || exit "$status"
  done
else
  echo 'Pilot did not qualify for expansion.'
fi
check_head
