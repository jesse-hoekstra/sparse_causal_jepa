#!/usr/bin/env bash
# Full bounce_states identifiability example in one command:
#   calibrate tau (fully connected) -> [optional identity-floor check]
#   -> train with sparsity -> evaluate.
#
# Any hydra overrides are passed through to every training run, which enforces the
# identical-config calibration rule (decisions.md D12) by construction:
#   bash scripts/run_bounce_example.sh data.num_balls=3 train.steps=25000
#
# Script flags (preferred — a typo errors loudly instead of being ignored):
#   --tau-factor=1.0   tau = factor x dense-reference held-out constraint_loss
#                      (default 1.0: SPARTAN App. A.2 sets tau to the fully
#                      connected model's loss; any slack is an explicit ablation)
#   --calib-steps=N    calibration run length. Default: SAME as the main run —
#                      tau from an undertrained reference is meaningless (the v2
#                      failure: 6k-step reference at 0.085 vs 0.045 achievable).
#                      Short values are for smoke tests only.
#   --main-steps=300000  MAIN run length only (default: config value)
#   --eval-episodes=5000 final identifiability sample size (default: 512;
#                        use 5000 for the paper-comparable Baumgartner run)
#   --final-seed-offset=29 held-out TEST split for the final report. Periodic
#                          curves and tau calibration use validation offset 17.
#   --eval-device=cpu    device used by evaluation model forwards
#   --identity-check   train a matched A≡0 mass-blind reference and abort unless
#                      calibrated tau is strictly below its held-out loss
#   --tau-max=VALUE    optional additional numeric guard: abort before the main
#                      run if calibrated tau exceeds VALUE
#                      (VALUE must use the configured constraint units)
#   --run-tag=rung1_seed0  output dir suffix; REQUIRED for parallel launches
#                          (the default timestamp collides across simultaneous starts)
# The calibration run uses model.spartan_dense=true (A≡1, deterministic dense
# attention): SPARTAN's "fully connected model" — NOT the gated model with
# sparsity off, whose Bernoulli gates keep injecting masking noise (audit F-8).
# All other arguments are hydra overrides, passed to every run (the D12
# identical-config rule; the dense/sparsity toggles are the reference's DEFINITION).
# The equivalent env vars (TAU_FACTOR, CALIB_STEPS, MAIN_STEPS, EVAL_EPISODES,
# FINAL_SEED_OFFSET, EVAL_DEVICE, IDENTITY_CHECK, RUN_TAG, PYTHON) still work as
# a fallback; flags win.
set -euo pipefail

PY=${PYTHON:-python}
TAU_FACTOR=${TAU_FACTOR:-1.0}
CALIB_STEPS=${CALIB_STEPS:-}
TAU_MAX=${TAU_MAX:-}
EVAL_EPISODES=${EVAL_EPISODES:-512}
FINAL_SEED_OFFSET=${FINAL_SEED_OFFSET:-29}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}
IDENTITY_CHECK=${IDENTITY_CHECK:-false}

HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tau-factor=*)  TAU_FACTOR="${arg#*=}" ;;
    --tau-max=*)     TAU_MAX="${arg#*=}" ;;
    --calib-steps=*) CALIB_STEPS="${arg#*=}" ;;
    --main-steps=*)  MAIN_STEPS="${arg#*=}" ;;
    --eval-episodes=*) EVAL_EPISODES="${arg#*=}" ;;
    --final-seed-offset=*) FINAL_SEED_OFFSET="${arg#*=}" ;;
    --eval-device=*) EVAL_DEVICE="${arg#*=}" ;;
    --identity-check) IDENTITY_CHECK=true ;;
    --run-tag=*)     RUN_TAG="${arg#*=}" ;;
    --*)             echo "unknown flag: $arg" >&2; exit 2 ;;
    *)               HYDRA_ARGS+=("$arg") ;;
  esac
done

BASE=outputs/bounce_example_${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}

# Calibration length: an undertrained reference inflates tau (v2 failure), so
# default to the main run's own length; --calib-steps only for smoke tests.
CALIB_STEPS_OVERRIDE=""
if [ -n "${CALIB_STEPS}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${CALIB_STEPS}"
  echo "WARNING: --calib-steps=${CALIB_STEPS} — tau from a non-converged reference is only for smoke tests" >&2
elif [ -n "${MAIN_STEPS:-}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"
fi

if [ "${IDENTITY_CHECK}" = true ]; then TOTAL_PHASES=4; else TOTAL_PHASES=3; fi

echo "== step 1/${TOTAL_PHASES}: tau calibration (dense A≡1 reference${CALIB_STEPS:+, ${CALIB_STEPS} steps}) =="
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/calibration" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  model.spartan_dense=true train.sparsity_enabled=false train.eval_every=null \
  ${CALIB_STEPS_OVERRIDE}
# Tau must be calibrated on the SAME configured quantity the Lagrangian dual
# compares against it: raw or variance-normalized prediction error plus
# lambda_logit * logit_penalty. The eval harness reports it as constraint_loss.
FC_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/calibration" --episodes 256 \
  --seed-offset 17 --device "${EVAL_DEVICE}" \
  | awk '/constraint_loss/ {print $2}')
TAU=$("$PY" -c "print(float('${FC_LOSS}') * float('${TAU_FACTOR}'))")
echo "dense-reference held-out constraint_loss=${FC_LOSS} -> tau=${TAU} (x${TAU_FACTOR})"
if [ -n "${TAU_MAX}" ] && [ "$("$PY" -c "print(1 if float('${TAU}') > float('${TAU_MAX}') else 0)")" = "1" ]; then
  echo "ABORT: tau=${TAU} > --tau-max=${TAU_MAX} (D17 go/no-go guard). A tau at or above" >&2
  echo "the mass-blind identity floor is satisfiable by the empty graph and cannot force" >&2
  echo "param edges — inspect the calibration run before spending the main-run compute." >&2
  exit 3
fi

MAIN_PHASE=2
if [ "${IDENTITY_CHECK}" = true ]; then
  echo "== step 2/4: feasibility check (mass-blind A≡0 reference) =="
  "$PY" scripts/train.py experiment=bounce_states \
    "hydra.run.dir=${BASE}/identity" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
    model.spartan_dense=false model.spartan_identity=true \
    train.sparsity_enabled=false train.eval_every=null ${CALIB_STEPS_OVERRIDE}
  IDENTITY_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/identity" --episodes 256 \
    --seed-offset 17 --device "${EVAL_DEVICE}" \
    | awk '/constraint_loss/ {print $2}')
  echo "mass-blind held-out constraint_loss=${IDENTITY_LOSS}; calibrated tau=${TAU}"
  if [ "$("$PY" -c "print(1 if float('${TAU}') >= float('${IDENTITY_LOSS}') else 0)")" = "1" ]; then
    echo "ABORT: tau=${TAU} >= mass-blind floor=${IDENTITY_LOSS}. The empty graph can" >&2
    echo "satisfy this constraint, so parameter identification is not forced." >&2
    exit 4
  fi
  MAIN_PHASE=3
fi

echo "== step ${MAIN_PHASE}/${TOTAL_PHASES}: main run (sparsity on, tau=${TAU}) =="
MAIN_STEPS_OVERRIDE=""
if [ -n "${MAIN_STEPS:-}" ]; then MAIN_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"; fi
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/main" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  "train.sparsity_tau=${TAU}" ${MAIN_STEPS_OVERRIDE}

FINAL_PHASE=$((MAIN_PHASE + 1))
echo "== step ${FINAL_PHASE}/${TOTAL_PHASES}: identifiability evaluation =="
"$PY" scripts/eval_identifiability.py "${BASE}/main" \
  --episodes "${EVAL_EPISODES}" \
  --seed-offset "${FINAL_SEED_OFFSET}" \
  --device "${EVAL_DEVICE}"
echo "artifacts in ${BASE}/main (checkpoint, resolved config, recovery_grid.png)"
