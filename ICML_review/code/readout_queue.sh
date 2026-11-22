#!/usr/bin/env bash
# Run an independent post-training readout study on one otherwise idle GPU.
set -euo pipefail
rev=${1:?Expected checkout revision}
method=${2:?Expected readout method}
case "$method" in
  linear-consensus-vision-window|stable-readout-synthetic|stable-readout-vision|support-pair-synthetic|support-pair-vision) ;;
  *) echo "Unknown readout method $method" >&2; exit 2 ;;
esac
q=ICML_review/outputs/readout-queue
mkdir -p "$q"
export PYTHONPATH="$PWD:$PWD/src" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
head_check() { test "$(git rev-parse HEAD)" = "$rev" || { echo "Checkout changed" >&2; return 1; }; }
head_check
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
host=$(hostname -s)
exec 6>"$q/.worker-$host.lock"
flock 6
exec 7>"$q/.$method.lock"
flock -n 7
execute() {
  local name=$1
  shift
  head_check
  while true; do
    head_check
    apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
    free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ $free =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]
    if [[ -z $apps ]] && ((free >= 8192)); then break; fi
    sleep 40
  done
  {
    printf '%s host=%q HEAD=%s PYTHONPATH=$PWD:$PWD/src ' "$(date -u +%FT%TZ)" "$host" "$(git rev-parse HEAD)"
    printf '%q ' "$@"
    printf '\n'
  } >>"$q/$name.log"
  local result=0
  "$@" >>"$q/$name.log" 2>&1 || result=$?
  echo "$result" >>"$q/$name.exits"
  if ((result)); then return "$result"; fi
  head_check
}
decision() {
  .venv/bin/python - "$@" <<'PY'
import json, sys
x=json.load(open(sys.argv[3]))
if x['seed']!=int(sys.argv[2]) or x['commit']!=sys.argv[1] or type(x.get('passed')) is not bool:
    raise ValueError('Inconsistent terminal receipt')
print('pass' if x['passed'] else 'fail')
PY
}
for seed in 0 1 2 3 4; do
  head_check
  case "$method" in
    linear-consensus-vision-window)
      command=(linear_consensus "$rev" vision-window "$seed") ;;
    stable-readout-*)
      command=(linear_consensus "$rev" "${method##*-}" "$seed" --stable) ;;
    support-pair-*)
      command=(support_pair "$rev" "${method##*-}" "$seed") ;;
  esac
  module=${command[0]}
  execute "$method-s$seed" .venv/bin/python -m "ICML_review.code.$module" "${command[@]:1}"
  result=$(decision "$rev" "$seed" "ICML_review/outputs/$method-s$seed/metrics.json")
  case "$method" in
    linear-consensus-vision-window|stable-readout-vision)
      execute "queries-$method-s$seed" .venv/bin/python -m ICML_review.code.consensus_queries "$rev" "$method" "$seed" ;;
    support-pair-vision)
      execute "queries-$method-s$seed" .venv/bin/python -m ICML_review.code.support_queries "$rev" "$seed" ;;
  esac
  if ((seed == 0)) && [[ $result == fail ]]; then break; fi
done
head_check
