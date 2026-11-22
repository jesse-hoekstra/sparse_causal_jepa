#!/usr/bin/env bash
# Evaluate the bounded TRAIN-only fallback on authenticated endpoints from each seed.
set -euo pipefail
rev=${1:?Expected revision}
q=ICML_review/outputs/combined-queue
mkdir -p "$q"
exec 7>"$q/.warm-fallback.lock"
flock -n 7
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD:$PWD/src"
check_head() { [[ $(git rev-parse HEAD) == "$rev" ]] || { echo 'Checkout changed' >&2; return 1; }; }
for seed in 0 1 2 3 4; do
  check_head
  command=(.venv/bin/python -m ICML_review.code.warm_fallback "$rev" "$seed")
  {
    printf '%s HEAD=%s PYTHONPATH=$PWD:$PWD/src ' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >> "$q/warm-fallback.commands"
  "${command[@]}"
  check_head
  if ((seed == 0)); then
    verdict=$(.venv/bin/python - "$rev" <<'PY'
import json, sys
with open("ICML_review/outputs/warm-fallback-s0/metrics.json") as f:
    row = json.load(f)
if row["commit"] != sys.argv[1] or row["seed"] != 0 or type(row["passed"]) != bool:
    raise ValueError("Invalid bounded fallback pilot")
print(row["passed"])
PY
)
    [[ $verdict == True ]] || break
  fi
done
