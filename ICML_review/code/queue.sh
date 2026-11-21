#!/usr/bin/env bash
# Run a list of fixed-weight stages on a checked idle device, or replay logs on CPU.
set -euo pipefail
rev=${1:?commit}
mode=${2:?models or replay}
shift 2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
run_stage() {
  local dir="ICML_review/outputs/$1-$3-s$2"
  [[ "$(git rev-parse HEAD)" == "$rev" ]]
  if [[ -f "$dir/exit_code" && "$(cat "$dir/exit_code")" == 0 ]] && grep -Fq "$rev" "$dir/NOTE.md"; then
    return
  fi
  if [[ "$mode" == models ]]; then
    local apps memory
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    memory=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ -z "$apps" && "$memory" -ge 8000 ]] || { echo "Device unavailable"; return 1; }
  fi
  bash ICML_review/code/launch.sh "$rev" "$@"
}
if [[ "$mode" == models ]]; then
  for seed in "$@"; do
    run_stage donors "$seed" full
    run_stage donors "$seed" baseline
    run_stage chart "$seed" paired
  done
elif [[ "$mode" == replay ]]; then
  for dataset in synthetic vision; do
    run_stage halving 0 "$dataset"
    pilot="ICML_review/outputs/halving-$dataset-s0/metrics.json"
    if .venv/bin/python -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["matched_winner"] else 1)' "$pilot"; then
      for seed in 1 2 3 4; do run_stage halving "$seed" "$dataset"; done
    fi
  done
else
  exit 2
fi
