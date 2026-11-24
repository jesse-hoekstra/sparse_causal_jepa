#!/usr/bin/env bash
# Run inside a detached gateway session: COMMIT {synthetic,rgb} {entire,sparse} HOST [HOST ...].
set -euo pipefail

worker=false
if [[ ${1:-} == --worker ]]; then worker=true; shift; fi
if (($# < 3)); then
    echo "Expected a full commit, dataset and schedule" >&2
    exit 2
fi
commit=$1 dataset=$2 schedule=$3
shift 3
[[ $commit =~ ^[[:xdigit:]]{40}$ ]] || exit 2
case $dataset in
    synthetic) module=ICML_review.code.timing_synthetic; seed=4 ;;
    rgb) module=ICML_review.code.timing_vision; seed=3 ;;
    *) exit 2 ;;
esac
case $schedule in entire|sparse) ;; *) exit 2 ;; esac
if "$worker"; then
    (($# == 0)) || exit 2
else
    (($# > 0)) || { echo "Supply integration worker names" >&2; exit 2; }
    for host in "$@"; do
        [[ $host =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || exit 2
    done
fi

root=$(git rev-parse --show-toplevel)
cd "$root"
arm=ICML_review/outputs/timing-$dataset-$schedule-s$seed
check_head() {
    [[ $(git rev-parse HEAD) == "$commit" ]] || { echo "Checkout changed" >&2; exit 1; }
}
check_note() {
    [[ -s $arm/NOTE.md ]] && grep -Fq -- "$commit" "$arm/NOTE.md" || {
        echo "Missing arm declaration for this commit: $arm/NOTE.md" >&2
        exit 1
    }
}
receipt() {
    .venv/bin/python - "$arm/metrics.json" "$commit" "$schedule" "$seed" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    row = json.load(f)
if (row["commit"], row["schedule"], row["seed"]) != (sys.argv[2], sys.argv[3], int(sys.argv[4])):
    raise ValueError("Timing receipt differs from the requested arm")
PY
}
stamp() { date -u +%FT%TZ; }
check_head
check_note
if [[ -e $arm/metrics.json ]]; then receipt; exit 0; fi

if ! "$worker"; then
    exec 9>>"$arm/dispatch.log"
    flock -n -E 73 9 || { echo "Cannot acquire the arm dispatcher lock" >&2; exit 73; }
    printf '%s head=%s hosts=%s\n' "$(stamp)" "$(git rev-parse HEAD)" "$*" >&9
    printf -v command 'cd %q && exec bash ICML_review/code/timing_dispatch.sh --worker %q %q %q' \
        "$root" "$commit" "$dataset" "$schedule"
    while true; do
        check_head
        if [[ -e $arm/metrics.json ]]; then receipt; exit 0; fi
        for host in "$@"; do
            rc=0
            ssh -n -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 "$host" "$command" \
                >&9 2>&1 9>&- || rc=$?
            printf '%s host=%s worker_exit=%d\n' "$(stamp)" "$host" "$rc" >&9
            case $rc in
                0) check_head; receipt; exit 0 ;;
                75) continue ;;
                *) exit "$rc" ;;
            esac
        done
        sleep 45
    done
fi

# Shared NFS receipts are append-only and double as lock files. Use write-capable descriptors
# for exclusive flock on NFS. Acquire the arm independently of the gateway for lost SSH sessions.
exec 7>>"$arm/worker.log"
flock -n -E 73 7 || { echo "Cannot acquire the arm worker lock" >&2; exit 73; }
if [[ -e $arm/metrics.json ]]; then receipt; exit 0; fi
physical=$(hostname -s)
[[ $physical =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || exit 2
gpu_dir=ICML_review/outputs/timing-dispatch
mkdir -p "$gpu_dir"
exec 8>>"$gpu_dir/$physical.gpu0.log"
flock -n -E 75 8 || exit "$?"

check_gpu() {
    local apps free util
    apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)
    free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    [[ $free =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && $util =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || {
        echo "Invalid GPU telemetry" >&2; exit 1
    }
    if [[ -n $apps ]] || (( free < 8192 || util != 0 )); then exit 75; fi
}
check_gpu
mkdir -p "$arm/attempts"
attempt=$(mktemp -d "$arm/attempts/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
on_exit() {
    local rc=$?
    printf '%d\n' "$rc" >"$attempt/exit_code.tmp"
    mv "$attempt/exit_code.tmp" "$attempt/exit_code"
    printf '%s exit=%d arm=%s attempt=%s\n' "$(stamp)" "$rc" "$arm" "$attempt" >&8
    printf '%s exit=%d host=%s attempt=%s\n' "$(stamp)" "$rc" "$physical" "$attempt" >&7
}
trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s head=%s host=%s attempt=%s\n' "$(stamp)" "$(git rev-parse HEAD)" "$physical" "$attempt" >&7
printf '%s head=%s arm=%s attempt=%s\n' "$(stamp)" "$(git rev-parse HEAD)" "$arm" "$attempt" >&8
exec >"$attempt/log.txt" 2>&1
check_head
git diff --quiet HEAD -- || { echo "Tracked files differ from the recorded commit"; exit 1; }
sha256sum "$arm/NOTE.md" >"$attempt/declaration.sha256"
command=(.venv/bin/python -m "$module" "$commit" "$schedule")
printf '%s host=%s head=%s command=' "$(stamp)" "$physical" "$(git rev-parse HEAD)"
printf '%q ' "${command[@]}"
printf '\n'
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
check_head
sha256sum -c --quiet "$attempt/declaration.sha256"
check_gpu

export PYTHONPATH="$PWD:$PWD/src" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_DIR="$PWD/$attempt" WANDB_CACHE_DIR="$PWD/$arm/wandb-cache"
export WANDB_DATA_DIR="$PWD/$arm/wandb-data" WANDB_ARTIFACT_DIR="$PWD/$arm/wandb-artifacts"
export WANDB_CONFIG_DIR="$PWD/$arm/wandb-config"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"
python_exit=0
"${command[@]}" 7>&- 8>&- || python_exit=$?
printf '%d\n' "$python_exit" >"$attempt/trainer_exit_code"
if ((python_exit == 75)); then exit 1; fi   # Only a busy GPU returns 75 to the dispatcher.
if ((python_exit)); then exit "$python_exit"; fi
check_head
sha256sum -c --quiet "$attempt/declaration.sha256"
receipt
