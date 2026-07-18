#!/usr/bin/env bash
# Detached server launch for the D21 bounce_baumgartner pipeline (nohup).
# Run FROM THE REPO ROOT ON THE SERVER, at the pushed commit (git_sha is
# recorded in each run's config — CLAUDE.md rule).
#
#   bash scripts/launch_bounce_server.sh [run_tag] [main_steps]
#
# Defaults: run_tag=d21_seed0, main_steps=300000. Logs + PID land in logs/.
# The pipeline runs BOTH stages sequentially (D12): dense-reference tau
# calibration at the same length, then the main sparse run — expect ~2x the
# single-run wall time. --tau-max=0.19 aborts before the main run if the
# calibrated tau reaches the measured mass-blind one-step floor (0.199
# normalized, D21) — a tau there is satisfiable by the empty graph.
set -euo pipefail

RUN_TAG=${1:-d21_seed0}
MAIN_STEPS=${2:-300000}
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
export PYTHON

nohup bash scripts/run_bounce_example.sh \
  "--run-tag=${RUN_TAG}" \
  "--main-steps=${MAIN_STEPS}" \
  --tau-max=0.19 \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  > "${LOG_DIR}/${RUN_TAG}.log" 2>&1 &

echo $! > "${LOG_DIR}/${RUN_TAG}.pid"
echo "launched pid $(cat "${LOG_DIR}/${RUN_TAG}.pid") -> ${LOG_DIR}/${RUN_TAG}.log"
echo "follow with: tail -f ${LOG_DIR}/${RUN_TAG}.log"
