#!/usr/bin/env bash
# Launch finite-support and frozen-readout comparisons from a gateway after the checkout is pinned.
set -euo pipefail
rev=${1:?Expected checkout revision}; shift
(($#)) || { echo "Supply worker names" >&2; exit 2; }
base=$PWD
jobs=ICML_review/outputs/readout-queue
four=ICML_review/outputs/four-queue
mkdir -p "$jobs" "$four"
exec 9>"$jobs/.launch.lock"
flock -n 9
check_head() { [[ $(git rev-parse HEAD) == "$rev" ]] || { echo "Checkout changed" >&2; return 1; }; }
check_head
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
free_nodes=()
while (("${#free_nodes[@]}" == 0)); do
  check_head
  for worker in "$@"; do
    [[ $worker =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]
    if ssh -n -o BatchMode=yes "$worker" '
      set -e
      a=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
      f=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
      [[ -z $a ]] && ((f >= 8192))
    '; then free_nodes+=("$worker"); fi
  done
  if (("${#free_nodes[@]}" == 0)); then sleep 40; fi
done
methods=(linear-consensus-vision-window stable-readout-vision support-pair-vision stable-readout-synthetic support-pair-synthetic)
for i in "${!methods[@]}"; do
  check_head
  method=${methods[$i]}
  worker=${free_nodes[$((i % ${#free_nodes[@]}))]}
  printf -v command 'bash -s -- %q %q %q' "$base" "$rev" "$method"
  ssh -o BatchMode=yes "$worker" "$command" <<'WORKER'
set -euo pipefail
cd "$1"
rev=$2 method=$3 name=icml-$3
[[ $(git rev-parse HEAD) == "$rev" ]]
prefix=ICML_review/outputs/readout-queue/$method
if tmux has-session -t "=$name" 2>/dev/null; then exit 0; fi
if [[ -s $prefix.exit_code && $(cat "$prefix.exit_code") == 0 ]]; then exit 0; fi
if [[ -e $prefix.exit_code ]]; then mv "$prefix.exit_code" "$prefix.prior-$(date +%s).exit_code"; fi
printf -v invocation '%q ' bash ICML_review/code/readout_queue.sh "$rev" "$method"
printf '%s HEAD=%s %s\n' "$(date -u +%FT%TZ)" "$rev" "$invocation" >>"$prefix.commands"
job="$invocation >>$prefix.log 2>&1; rc=\$?; echo \$rc > $prefix.exit_code; exit \$rc"
tmux new-session -d -s "$name" -c "$PWD" "$job"
WORKER
done
check_head
bash ICML_review/code/four_after_readouts.sh "$rev" "$@"
