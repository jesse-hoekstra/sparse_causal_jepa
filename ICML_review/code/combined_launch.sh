#!/usr/bin/env bash
# Run one conditional interaction study from a detached gateway session.
set -euo pipefail
rev=${1:?Expected revision} dataset=${2:?Expected dataset}; shift 2
case "$dataset" in synthetic|vision) ;; *) exit 2;; esac
(($#)) || { echo 'Supply worker names' >&2; exit 2; }
workers=("$@")
for worker in "${workers[@]}"; do [[ $worker =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; done
q=ICML_review/outputs/combined-queue
base=$PWD
mkdir -p "$q"
exec 9>"$q/.$dataset-launch.lock"
flock -n 9
export PYTHONPATH="$PWD:$PWD/src" OMP_NUM_THREADS=1
head_check() { [[ $(git rev-parse HEAD) == "$rev" ]] || { echo 'Checkout changed' >&2; return 1; }; }
head_check
printf '%s HEAD=%s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" >> "$q/$dataset.launches"
bash ICML_review/code/combined_run.sh "$rev" "$dataset" 0 cohort
read_field() {
  .venv/bin/python - "$1" "$2" "$rev" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    record = json.load(f)
if record["commit"] != sys.argv[3]:
    raise ValueError("Mismatched decision revision")
print(record[sys.argv[2]])
PY
}
cohort=ICML_review/outputs/$dataset-combined-cohort/metrics.json
[[ -s $cohort ]] || { echo 'Constituent decisions still pending' >&2; exit 1; }
[[ $(read_field "$cohort" status) == eligible ]] || exit 0

run_seed() {
  local seed=$1 receipt needed worker name index marker cmd code
  receipt=ICML_review/outputs/$dataset-combined-s$seed/metrics.json
  needed=$(.venv/bin/python -m ICML_review.code.combined_device "$dataset" "$seed" run)
  if [[ $needed == cpu ]]; then
    bash ICML_review/code/combined_run.sh "$rev" "$dataset" "$seed" run
    read_field "$receipt" passed >/dev/null
    return
  fi
  name=icml-combined-$dataset-s$seed
  while true; do
    head_check
    for worker in "${workers[@]}"; do
      # A previous gateway invocation may already have detached this seed on a worker.
      marker="$q/$dataset-s$seed.$worker.active"
      [[ -s $marker ]] || continue
      index=$(cat "$marker")
      if [[ -s $index.rc ]]; then
        code=$(cat "$index.rc")
        mv "$marker" "$index.finished"
        if ((code == 75)); then continue; fi
        if ((code)); then echo "Worker failed; see $index.log" >&2; return "$code"; fi
        read_field "$receipt" passed >/dev/null
        return
      fi
      if ssh -n -o BatchMode=yes "$worker" "tmux has-session -t '=$name' 2>/dev/null"; then
        sleep 35; continue 2
      fi
      # NFS visibility can lag tmux exit. Keep the launch marker and inspect again.
      [[ -s $index.absent-since ]] || date +%s > "$index.absent-since"
      if [[ $(($(date +%s) - $(cat "$index.absent-since"))) -le 120 ]]; then sleep 35; continue 2; fi
      [[ -s $index.rc ]] && continue 2
      echo "Detached worker disappeared without an exit file at $index" >&2; return 1
    done
    if [[ -s $receipt ]]; then read_field "$receipt" passed >/dev/null; return; fi
    for worker in "${workers[@]}"; do
      index="$q/$dataset-s$seed.$worker-$(date +%s)"
      marker="$q/$dataset-s$seed.$worker.active"
      printf -v cmd 'bash -s -- %q %q %q %q %q %q %q %q' "$base" "$rev" "$dataset" "$seed" "$needed" "$name" "$index" "$marker"
      if ssh -o BatchMode=yes "$worker" "$cmd" <<'WORKER'
set -euo pipefail
cd "$1"
rev=$2 dataset=$3 seed=$4 needed=$5 name=$6 index=$7 marker=$8
[[ $(git rev-parse HEAD) == "$rev" ]]
model=$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)
apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ $model == "$needed" && -z $apps ]] && ((free >= 8192)) || exit 75
printf -v run '%q ' bash ICML_review/code/combined_run.sh "$rev" "$dataset" "$seed" run
job="$run >> $index.log 2>&1; rc=\$?; echo \$rc > $index.rc.tmp; mv $index.rc.tmp $index.rc; exit \$rc"
tmux new-session -d -s "$name" -c "$PWD" "$job"
echo "$index" > "$marker.tmp"
mv "$marker.tmp" "$marker"
WORKER
      then break; else code=$?; ((code == 75)) || return "$code"; fi
    done
    sleep 35
  done
}
run_seed 0
[[ $(read_field "ICML_review/outputs/$dataset-combined-s0/metrics.json" passed) == True ]] || exit 0
pids=()
for seed in 1 2 3 4; do run_seed "$seed" & pids+=("$!"); done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
head_check
exit "$rc"
