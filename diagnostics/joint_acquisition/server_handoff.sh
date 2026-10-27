#!/usr/bin/env bash
# Reviewed handoff only: authoring this file does not authorize executing a run.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash diagnostics/joint_acquisition/server_handoff.sh preflight|launch \
    --training-source-root /absolute/server/archive [--all-six-checkpoint /relocated/last.pt]
  bash diagnostics/joint_acquisition/server_handoff.sh status|attach --run-name NAME

Optional source overrides: --source-dir PATH --comparison-bundle PATH
                           --current-checkpoint PATH
Every invocation pushes committed local work, then pulls remotely before doing work.
Only explicit `launch` starts the fixed 1,000-update diagnostic in detached tmux.
USAGE
}

mode=${1:-help}
if [[ "$mode" == help || "$mode" == --help || "$mode" == -h ]]; then
    usage
    exit 0
fi
shift
case "$mode" in preflight|launch|status|attach) ;; *) usage >&2; exit 2 ;; esac
repo=/nfs/home/jesseh/sparse_causal_jepa
source_dir=$repo/outputs/diagnostics/predictive_supervision_1_4_16_frozen_mlp
comparison_bundle=$repo/outputs/diagnostics/frozen_predictor_control_20261027/source_bundle
current_checkpoint=$repo/outputs/diagnostics/fresh_encoder_frozen_all6_predictor_100k/seed-202610261301/last.pt
training_source_root=
all_six_checkpoint=
run_name=
while (($#)); do
    if (($# < 2)); then usage >&2; exit 2; fi
    case "$1" in
        --source-dir) source_dir=$2 ;;
        --comparison-bundle) comparison_bundle=$2 ;;
        --current-checkpoint) current_checkpoint=$2 ;;
        --training-source-root) training_source_root=$2 ;;
        --all-six-checkpoint) all_six_checkpoint=$2 ;;
        --run-name) run_name=$2 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift 2
done
if [[ "$mode" == preflight || "$mode" == launch ]]; then
    if [[ "$training_source_root" != /* ]]; then
        echo '--training-source-root must name the actual absolute server training-source archive.' >&2
        exit 2
    fi
    if [[ -n "$run_name" ]]; then
        echo 'Fresh launch names are generated uniquely; --run-name is only for status/attach.' >&2
        exit 2
    fi
else
    if [[ ! "$run_name" =~ ^joint-acquisition-[A-Za-z0-9_-]+$ ]]; then
        echo 'status/attach requires the exact reported --run-name.' >&2
        exit 2
    fi
fi
for source_path in "$source_dir" "$comparison_bundle" "$current_checkpoint"; do
    [[ "$source_path" == /* ]] || { echo 'Source paths must be absolute server paths.' >&2; exit 2; }
done
if [[ -n "$all_six_checkpoint" && "$all_six_checkpoint" != /* ]]; then
    echo '--all-six-checkpoint must be an absolute server path.' >&2
    exit 2
fi
local_repo=$(git rev-parse --show-toplevel)
cd "$local_repo"
git diff --quiet && git diff --cached --quiet || {
    echo 'Commit local tracked changes before synchronizing the diagnostic.' >&2
    exit 1
}
branch=$(git symbolic-ref --quiet --short HEAD)
revision=$(git rev-parse HEAD)
if [[ "$mode" == launch ]]; then
    run_name="joint-acquisition-$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:8}-$RANDOM"
fi
# Required before EVERY SSH invocation, including read-only status and attachment.
git push origin "$branch"
if [[ "$mode" == attach ]]; then
    printf -v remote_attach 'git -C %q pull --ff-only && cd %q && test "$(git rev-parse HEAD)" = %q && test "$(git symbolic-ref --quiet --short HEAD)" = %q && git diff --quiet && git diff --cached --quiet && exec tmux attach-session -t %q' \
        "$repo" "$repo" "$revision" "$branch" "$run_name"
    exec ssh -S /Users/hoekstra/.ssh/codex-omi-rapid-22 -o BatchMode=yes -tt \
        omi-rapid-22 "$remote_attach"
fi
printf -v remote_command '%q ' bash -s -- "$mode" "$repo" "$branch" "$revision" \
    "$source_dir" "$training_source_root" "$comparison_bundle" "$all_six_checkpoint" \
    "$current_checkpoint" "$run_name"
ssh_options=(-S /Users/hoekstra/.ssh/codex-omi-rapid-22 -o BatchMode=yes)
ssh "${ssh_options[@]}" omi-rapid-22 "$remote_command" <<'REMOTE'
set -euo pipefail
mode=$1 repo=$2 branch=$3 revision=$4 source_dir=$5 training_source_root=$6
comparison_bundle=$7 all_six_checkpoint=$8 current_checkpoint=$9 run_name=${10}
# Synchronize before any remote inspection or execution; reject a different branch.
git -C "$repo" pull --ff-only
cd "$repo"
[[ "$(git symbolic-ref --quiet --short HEAD)" == "$branch" ]] || {
    echo 'Remote checkout is on a different branch; no diagnostic was launched.' >&2; exit 1;
}
[[ "$(git rev-parse HEAD)" == "$revision" ]] || {
    echo 'Remote HEAD differs from the pushed local commit.' >&2; exit 1;
}
git diff --quiet && git diff --cached --quiet || {
    echo 'Remote tracked changes would invalidate the committed diagnostic identity.' >&2; exit 1;
}
run_dir=$repo/outputs/diagnostics/$run_name
output_dir=$run_dir/results
log_path=$run_dir/stdout-stderr.log
if [[ "$mode" == status ]]; then
    test -d "$run_dir"
    printf 'Session: %s\nOutput: %s\nLog: %s\n' "$run_name" "$output_dir" "$log_path"
    tmux list-sessions || true
    if [[ -f "$run_dir/exit_code" ]]; then printf 'Exit code: '; cat "$run_dir/exit_code"; fi
    .venv/bin/python - "$output_dir/tracking.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if path.exists():
    record = json.loads(path.read_text())
    print(f"W&B URL: https://wandb.ai/{record['entity']}/{record['project']}/runs/{record['id']}")
else:
    print("W&B URL pending: source validation/logger initialization has not written tracking.json.")
PY
    if [[ -f "$log_path" ]]; then tail -n 30 "$log_path"; fi
    exit 0
fi
command -v tmux >/dev/null
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
# Serialize launches through this handoff, and inspect unrelated GPU users too.
exec 8> /tmp/scjepa-joint-acquisition-launch.lock
flock -n 8 || { echo 'Another handoff is checking or launching a job.' >&2; exit 1; }
echo 'Existing tmux sessions:'
tmux list-sessions || true
panes=$(tmux list-panes -a -F '#S #{pane_dead}' 2>/dev/null || true)
if awk '$1 ~ /^joint-acquisition-/ && $2 == 0 {found=1} END {exit !found}' <<<"$panes"; then
    echo 'A live joint-acquisition session already exists; inspect it before launching.' >&2
    exit 1
fi
echo 'Existing GPU compute processes:'
gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
printf '%s\n' "$gpu_processes"
[[ -z "$gpu_processes" ]] || { echo 'GPU is occupied; only one run is permitted.' >&2; exit 1; }
# Resolve the real source encoder from provenance; never replace it with code statistics.
all_six_checkpoint=$(.venv/bin/python - "$comparison_bundle" "$all_six_checkpoint" <<'PY'
import hashlib
import json
import pathlib
import sys
bundle, override = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((bundle / "manifest.json").read_text())
path = pathlib.Path(override or manifest["provenance"]["original_paths"]["checkpoint"])
if not path.is_file():
    raise SystemExit(f"Actual all-six checkpoint unavailable at {path}; pass its relocated path explicitly.")
with path.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
if digest != manifest["provenance"]["sha256"]["checkpoint"]:
    raise SystemExit("Actual all-six checkpoint hash differs from portable-bundle provenance.")
print(path.resolve())
PY
)
.venv/bin/python - "$source_dir" "$training_source_root" "$comparison_bundle" "$current_checkpoint" <<'PY'
import hashlib
import json
import pathlib
import sys
from diagnostics.joint_acquisition.provenance import historical_training_identity, validate_training_semantics
from diagnostics.joint_acquisition.source import JOINT_CHECKPOINT_SHA256
source, archive, bundle, current = map(pathlib.Path, sys.argv[1:])
protocol = json.loads((source / "protocol.json").read_text())
if historical_training_identity(archive) != protocol["code_identity"]:
    raise SystemExit("Historical training-source archive does not match the recorded source identity.")
validate_training_semantics(archive, pathlib.Path.cwd())
checkpoint = source / "seed-202610261301/prediction_1/last.pt"
with checkpoint.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != JOINT_CHECKPOINT_SHA256:
        raise SystemExit("Primary checkpoint is not the recorded original 30k endpoint.")
for path in [source / "train_inputs.pt", source / "evaluation_data.pt", current,
             current.parent / "completed.json", current.parent.parent / "protocol.json",
             *(checkpoint.parent / name for name in ("completed.json", "results.json", "gradient_history.json")),
             *(bundle / name for name in ("manifest.json", "frozen_modules.pt", "train_inputs.pt", "evaluation_data.pt"))]:
    if not path.is_file():
        raise SystemExit(f"Required source artifact unavailable: {path}")
print("Preflight verified primary/source-encoder hashes and historical training-code identity; required files exist.")
print("The runner additionally verifies full cache, optimizer and compatible-module identities before fitting.")
PY
printf 'Resolved all-six checkpoint: %s\nHistorical source tree: %s\n' "$all_six_checkpoint" "$training_source_root"
if [[ "$mode" == preflight ]]; then exit 0; fi
mkdir "$run_dir"  # Atomic refusal to overwrite any previous handoff/results.
command=(.venv/bin/python -u -m diagnostics.joint_acquisition.run
    --source-dir "$source_dir" --output-dir "$output_dir"
    --training-source-root "$training_source_root" --comparison-bundle "$comparison_bundle"
    --all-six-checkpoint "$all_six_checkpoint" --current-checkpoint "$current_checkpoint"
    --seed 202610261301 --groups 64 --steps 1000 --bootstrap-samples 1000
    --displacement 0.01 --rcond 0.000001 --learning-rate 0.001 --threads 1
    --device cuda --wandb-mode online --wandb-project sparse_causal_jepa)
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$repo"
    printf 'exec >%q 2>&1\n' "$log_path"
    printf 'trap '\''code=$?; printf "%%s\\n" "$code" >%q'\'' EXIT\n' "$run_dir/exit_code"
    cat <<'JOB'
exec 9> /tmp/scjepa-joint-acquisition-gpu.lock
flock -n 9 || { echo 'Another diagnostic owns the GPU lock.'; exit 1; }
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_processes" ]] || { echo 'GPU became occupied before execution.'; exit 1; }
export WANDB_MODE=online
JOB
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$run_dir/run.sh"
printf -v tmux_command 'bash %q' "$run_dir/run.sh"
tmux new-session -d -s "$run_name" "$tmux_command" 8>&-
printf 'Session: %s\nLog: %s\nOutput: %s\nCommand script: %s\n' \
    "$run_name" "$log_path" "$output_dir" "$run_dir/run.sh"
printf 'W&B URL: pending; inspect %s/tracking.json with the status command below.\n' "$output_dir"
printf 'Status: bash diagnostics/joint_acquisition/server_handoff.sh status --run-name %q\n' "$run_name"
printf 'Reattach: bash diagnostics/joint_acquisition/server_handoff.sh attach --run-name %q\n' "$run_name"
REMOTE
