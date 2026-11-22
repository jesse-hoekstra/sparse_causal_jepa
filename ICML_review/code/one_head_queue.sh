#!/usr/bin/env bash
# Run the one-head trial only after a scientifically unsuccessful zero-head route.
set -euo pipefail
if (( $# < 2 || $# > 5 )); then
  echo "Usage: bash one_head_queue.sh REV WORKER [WORKER ...] (up to four)" >&2
  exit 2
fi
revision=$1
shift
nodes=("$@")
for node in "${nodes[@]}"; do
  [[ $node =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || { echo "Invalid worker name" >&2; exit 2; }
done
if [[ $(printf '%s\n' "${nodes[@]}" | sort -u | wc -l) -ne ${#nodes[@]} ]]; then
  echo "Workers must be distinct." >&2; exit 2
fi
root=$(git rev-parse --show-toplevel)
cd "$root"
output=ICML_review/outputs
queue=$output/one-head-queue
module=ICML_review.code.vision_one_head
check_head() { [[ $(git rev-parse HEAD) == "$revision" ]] || { echo "Checkout changed" >&2; return 1; }; }
check_head
mkdir -p "$queue"
exec 8>"$queue/.lock"
flock -n 8 || { echo "Another one-head coordinator is active." >&2; exit 75; }
record() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$queue/commands.log"; }
record "HEAD $(git rev-parse HEAD); workers ${nodes[*]}"
for prior in synthetic-loss vision-loss vision-direct; do
  [[ $(cat "$output/loss-queues/$prior.exit_code") == 0 ]] || { echo "Prior queue $prior is incomplete or failed." >&2; exit 1; }
done
# Process success alone does not distinguish a failed scientific pilot from a successful method.
eligibility=$(.venv/bin/python - "$output" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
def direct(seed):
    p = out / f"vision-direct-s{seed}" / "metrics.json"
    row = json.loads(p.read_text())
    note = p.with_name("NOTE.md").read_text()
    if row["seed"] != seed or type(row["passed"]) is not bool or row["commit"] not in note:
        raise ValueError(f"Invalid direct-projection receipt {p}")
    adapt = row["adaptation_checks"]
    endpoint = row.get("checks", {})
    if not adapt or not all(type(v) is bool for v in [*adapt.values(), *endpoint.values()]):
        raise ValueError(f"Invalid scientific checks in {p}")
    if all(adapt.values()):
        if not endpoint or row["passed"] != all(endpoint.values()):
            raise ValueError(f"Missing or inconsistent downstream checks in {p}")
    elif row["passed"] or row.get("reason") != "adaptation":
        raise ValueError(f"Inconsistent adaptation decision in {p}")
    return row
pilot = direct(0)
rows = [pilot] if not pilot["passed"] else [pilot, *(direct(s) for s in range(1, 5))]
if any(r["commit"] != pilot["commit"] for r in rows):
    raise ValueError("Direct-projection trials have different revisions")
print("run" if any(not r["passed"] for r in rows) else "skip")
PY
)
if [[ $eligibility == skip ]]; then
  record "Direct projection passed on every seed; the conditional trial is unneeded."
  exit 0
fi
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
check_head

phase() {
  local node=$1 seed=$2 stage=$3 command shell_command rc
  check_head
  printf -v command '%q ' .venv/bin/python -m "$module" "$revision" "$seed" "$stage"
  printf -v shell_command 'bash -s -- %q %q %q %q %q %q' "$root" "$revision" "$node" "$seed" "$stage" "$module"
  record "worker=$node seed=$seed phase=$stage: PYTHONPATH=\$PWD:\$PWD/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $command"
  if ssh -o BatchMode=yes "$node" "$shell_command" <<'WORKER'
set -euo pipefail
cd "$1"
rev=$2 node=$3 seed=$4 stage=$5 module=$6
check_head() { [[ $(git rev-parse HEAD) == "$rev" ]] || { echo "Worker checkout changed" >&2; return 1; }; }
check_head
physical=$(hostname -s)
[[ $physical =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]
if [[ $stage != assess ]]; then
  exec 7>"ICML_review/outputs/one-head-queue/.gpu-${physical}.lock"
  flock 7
  while true; do
    check_head
    apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
    free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ $free =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo "Invalid GPU memory reading" >&2; exit 1; }
    if [[ -z $apps ]] && (( free >= 8192 )); then break; fi
    sleep 45
  done
fi
check_head
printf '%s worker=%s host=%s HEAD=%s seed=%s stage=%s starting\n' "$(date -u +%FT%TZ)" "$node" "$physical" "$(git rev-parse HEAD)" "$seed" "$stage" >> ICML_review/outputs/one-head-queue/commands.log
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/src"
.venv/bin/python -m "$module" "$rev" "$seed" "$stage"
check_head
WORKER
  then rc=0; else rc=$?; fi
  record "worker=$node seed=$seed phase=$stage exit_code=$rc"
  if (( rc != 0 )); then return "$rc"; fi
  check_head
}

decision() {
  .venv/bin/python - "$revision" "$1" "$output/vision-one-head-s$1/metrics.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[3]))
if row["commit"] != sys.argv[1] or row["seed"] != int(sys.argv[2]) or type(row["passed"]) is not bool:
    raise ValueError("Wrong one-head seed, revision, or decision")
if row["passed"] and (set(row.get("validation", {})) != {"full", "baseline"}
                      or not row.get("checks") or not all(v is True for v in row["checks"].values())):
    raise ValueError("A passing one-head decision requires both sparse endpoint assessments")
print("pass" if row["passed"] else "fail")
PY
}

wait_pair() {
  local first=$1 second=$2 first_rc=0 second_rc=0
  wait "$first" || first_rc=$?
  wait "$second" || second_rc=$?
  if (( first_rc != 0 )); then return "$first_rc"; fi
  return "$second_rc"
}

run_seed() {
  local seed=$1 first=$2 second=$3 full_pid base_pid verdict
  local receipt="$output/vision-one-head-s$seed/metrics.json"
  check_head
  if [[ ! -f $receipt ]]; then phase "$first" "$seed" prepare; fi
  if [[ ! -f $receipt ]]; then
    if [[ $first == "$second" ]]; then
      phase "$first" "$seed" full
      phase "$second" "$seed" baseline
    else
      phase "$first" "$seed" full & full_pid=$!
      phase "$second" "$seed" baseline & base_pid=$!
      wait_pair "$full_pid" "$base_pid"
    fi
    phase "$first" "$seed" assess
  fi
  verdict=$(decision "$seed")
  record "seed $seed scientific decision $verdict"
  check_head
}
first=${nodes[0]}
second=${nodes[1]:-$first}
run_seed 0 "$first" "$second"
pilot=$(decision 0)
if [[ $pilot == fail ]]; then
  record "Seed zero did not meet the declared criteria; expansion is closed."
  exit 0
fi
if (( ${#nodes[@]} < 3 )); then
  for seed in 1 2 3 4; do run_seed "$seed" "$first" "$second"; done
else
  third=${nodes[2]}
  fourth=${nodes[3]:-$third}
  for seed in 1 3; do
    run_seed "$seed" "$first" "$second" & p=$!
    run_seed "$((seed+1))" "$third" "$fourth" & q=$!
    wait_pair "$p" "$q"
  done
fi
check_head
