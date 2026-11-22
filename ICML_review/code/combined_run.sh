#!/usr/bin/env bash
# Run a registered combined study from its pinned checkout on the selected worker.
set -euo pipefail
revision=${1:?Expected checkout revision}
dataset=${2:?Expected dataset}
seed=${3:?Expected seed}
phase=${4:?Expected cohort or run}
case "$dataset" in synthetic|vision) ;; *) exit 2;; esac
[[ $seed =~ ^[0-4]$ ]]
case "$phase" in cohort) [[ $seed == 0 ]];; run) ;; *) exit 2;; esac
export PYTHONPATH="$PWD:$PWD/src" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
queue=ICML_review/outputs/combined-queue
mkdir -p "$queue"
label="$dataset-$phase-s$seed"
check_head() { [[ $(git rev-parse HEAD) == "$revision" ]] || { echo 'Checkout changed' >&2; return 1; }; }
check_head
exec 7>"$queue/.$label.lock"
flock -n 7 || exit 75
needed=$(.venv/bin/python -m ICML_review.code.combined_device "$dataset" "$seed" "$phase")
if [[ $needed != cpu ]]; then
  exec 8>"$queue/.worker-$(hostname -s).lock"
  flock -n 8 || exit 75
  model=$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)
  [[ $model == "$needed" ]] || exit 75
  apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
  free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ -z $apps && $free =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && ((free >= 8192)) || exit 75
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=''
fi
command=(.venv/bin/python -m "ICML_review.code.${dataset}_combined" "$revision" "$seed" "$phase")
{
  printf '%s host=%s HEAD=%s PYTHONPATH=$PWD:$PWD/src CUDA_VISIBLE_DEVICES=%q ' "$(date -u +%FT%TZ)" "$(hostname -s)" "$(git rev-parse HEAD)" "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${command[@]}"
  printf '\n'
} >> "$queue/$label.commands"
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
check_head
rc=0
"${command[@]}" >> "$queue/$label.log" 2>&1 || rc=$?
check_head
printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$revision" "$rc" >> "$queue/$label.exits"
exit "$rc"
