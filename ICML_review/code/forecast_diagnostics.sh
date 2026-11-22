#!/usr/bin/env bash
# Run a reported-checkpoint diagnostic on one worker, starting with its technical pilot.
set -euo pipefail
revision=${1:?commit}
suite=${2:?response-geometry or mask-windows}
worker=${3:?worker}
root=$(git rev-parse --show-toplevel)
cd "$root"
case "$suite" in
  response-geometry) module=response_geometry ;;
  mask-windows) module=mask_windows ;;
  *) exit 2 ;;
esac
[[ "$(git rev-parse HEAD)" == "$revision" ]]
printf 'Starting %s on %s at %s, commit %s\n' "$suite" "$worker" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)"
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
status=0
for seed in 0 1 2 3 4; do
  [[ "$(git rev-parse HEAD)" == "$revision" ]]
  printf -v invocation '%q ' .venv/bin/python -m "ICML_review.code.$module" "$revision" "$seed"
  printf -v command 'set -eu; cd %q; test "$(git rev-parse HEAD)" = %q; ' "$root" "$revision"
  command+='while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || "$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 8192 ]]; do sleep 45; done; '
  command+='export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONPATH="$PWD:$PWD/src"; '
  printf 'Seed %s at %s\n' "$seed" "$(date -u +%FT%TZ)"
  ssh -o BatchMode=yes "$worker" "$command$invocation"
  [[ "$(git rev-parse HEAD)" == "$revision" ]]
  receipt="ICML_review/outputs/$suite-s$seed/metrics.json"
  [[ -s "$receipt" ]]
  if ! .venv/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["verified"] and r["commit"]==sys.argv[2] else 1)' "$receipt" "$revision"; then
    printf 'Technical verification failed for seed %s.\n' "$seed"
    [[ "$seed" != 0 ]] || exit 1
    status=1
  fi
done
exit "$status"
