#!/usr/bin/env bash
# Launch the independent four-acquisition comparison after the readout workers release their GPUs.
set -euo pipefail
revision=${1:?Expected checkout revision}; shift
(($#)) || { echo "Supply one or more worker names" >&2; exit 2; }
base=$PWD
q=ICML_review/outputs/four-queue
mkdir -p "$q"
exec 8>"$q/.launch.lock"
flock -n 8
check_head() { [[ $(git rev-parse HEAD) == "$revision" ]] || { echo "Checkout changed" >&2; return 1; }; }
methods=(linear-consensus-vision-window stable-readout-synthetic stable-readout-vision support-pair-synthetic support-pair-vision)
for method in "${methods[@]}"; do
  while [[ ! -s ICML_review/outputs/readout-queue/$method.exit_code ]]; do check_head; sleep 35; done
done
for node in "$@"; do
  [[ $node =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]
  check_head
  printf -v remote_command 'bash -s -- %q %q' "$base" "$revision"
  ssh -o BatchMode=yes "$node" "$remote_command" <<'WORKER'
set -euo pipefail
cd "$1"
revision=$2
[[ $(git rev-parse HEAD) == "$revision" ]]
name=$(hostname -s)
q=ICML_review/outputs/four-queue
if tmux has-session -t '=icml-four' 2>/dev/null; then exit 0; fi
if [[ -s $q/$name.exit_code ]] && [[ $(cat "$q/$name.exit_code") == 0 ]]; then exit 0; fi
# An occupied node can monitor the queue. The worker checks physical vacancy again before any GPU work.
if [[ -s $q/$name.exit_code ]]; then mv "$q/$name.exit_code" "$q/$name.previous-$(date +%s).exit_code"; fi
printf -v command '%q ' .venv/bin/python -m ICML_review.code.consensus_queue "$revision" --four
launch="PYTHONPATH=\$PWD:\$PWD/src $command >> $q/$name.log 2>&1; rc=\$?; echo \$rc > $q/$name.exit_code; exit \$rc"
tmux new-session -d -s icml-four -c "$PWD" "$launch"
WORKER
done
check_head
