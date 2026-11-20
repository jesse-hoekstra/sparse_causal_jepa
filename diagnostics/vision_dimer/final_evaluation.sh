#!/usr/bin/env bash
# Final test evaluation of the vision experiment, after all five seeds are trained (run_seed.sh):
#   1. simulator identity check: the stored validation split is regenerated bit for bit;
#   2. the test split, generated once from the cohort seed (hash-verified if it already exists);
#   3. the fifteen fixed-budget endpoints (dense initializer adapt/final.pt; baseline and full
#      sparse-500k/final.pt; seeds 0-4) scored on test, and their figure data;
#   4. summaries (test; validation at 500k for reference), chart diagnostic, code invariance and
#      validation learning curves;
#   5. the downstream causal queries on the test contexts;
#   6. the paper's figures, tables and number macros (needs the `analysis` extra).
# Every probe and alignment is fitted on calibration; the test split is only scored.
#
# usage: final_evaluation.sh DATA_DIR OUTPUT_ROOT EVALUATION_DIR
#   OUTPUT_ROOT holds seed<s>/ from run_seed.sh. Environment: PYTHON, CUDA_VISIBLE_DEVICES.
set -euo pipefail
DATA=$1 ROOT=$2 OUT=$3
CODE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-$CODE_DIR/.venv/bin/python}
export PYTHONPATH="$CODE_DIR:$CODE_DIR/src"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
cd "$CODE_DIR"
mkdir -p "$OUT"
PKG=diagnostics.vision_dimer
log "start on $(hostname)"

log "simulator identity check (validation regenerated in memory)"
"$PYTHON" -m $PKG.data --data-dir "$DATA" --verify validation | tee "$OUT/regen-check.json"
log "test split (generated once; verified by hash if present)"
"$PYTHON" -m $PKG.data --data-dir "$DATA" --include-test | tee "$OUT/test-split-manifest.json"
cp "$DATA/manifest.json" "$OUT/data-manifest-with-test.json"

for s in 0 1 2 3 4; do
  for arm in dense baseline full; do
    if [ "$arm" = dense ]; then ck=$ROOT/seed$s/adapt/final.pt; else ck=$ROOT/seed$s/sparse-500k/$arm/final.pt; fi
    "$PYTHON" -m $PKG.train evaluate --checkpoint "$ck" --data-dir "$DATA" \
      --output "$OUT/test-seed$s-$arm.json" --split test > /dev/null
    "$PYTHON" -m $PKG.analysis figure-data --checkpoint "$ck" --data-dir "$DATA" \
      --stored "$OUT/test-seed$s-$arm.json" --output "$OUT/figdata-seed$s-$arm.json" > /dev/null
    log "scored seed $s $arm"
  done
done

test_runs=() validation_runs=()
for s in 0 1 2 3 4; do
  for arm in baseline full; do
    test_runs+=(--run "$s:$arm:$OUT/test-seed$s-$arm.json")
    validation_runs+=(--run "$s:$arm:$ROOT/seed$s/sparse-500k/$arm/final.json")
  done
done
"$PYTHON" -m $PKG.summarize "${test_runs[@]}" --output "$OUT/summary-test.json" \
  --reference "$DATA" --split test > /dev/null
"$PYTHON" -m $PKG.summarize "${validation_runs[@]}" --output "$OUT/summary-validation.json" \
  --reference "$DATA" --split validation > /dev/null
"$PYTHON" -m $PKG.analysis charts --data-dir "$DATA" --root "$ROOT" \
  --output "$OUT/chart-check.json" > "$OUT/chart-check.log"
"$PYTHON" -m $PKG.analysis code-invariance --data-dir "$DATA" --root "$ROOT" \
  --output "$OUT/code-invariance.json" > /dev/null
"$PYTHON" -m $PKG.analysis curves --root "$ROOT" --output "$OUT/validation_curves.json" > /dev/null
log "summaries and analyses written"

log "downstream queries on the test contexts"
"$PYTHON" -m $PKG.downstream --data-dir "$DATA" --root "$ROOT" --sub sparse-500k --split test \
  --stored-dir "$OUT" --output "$OUT/downstream" > "$OUT/downstream.log" 2>&1

"$PYTHON" -m $PKG.paper_figures --evaluation "$OUT" --curves "$OUT/validation_curves.json" \
  --output-dir "$OUT/paper" > "$OUT/paper.log"
"$PYTHON" -m $PKG.paper_downstream --downstream "$OUT/downstream" --output-dir "$OUT/paper" \
  >> "$OUT/paper.log"
(cd "$OUT" && sha256sum ./*.json downstream/*.json > SHA256SUMS)
log "final evaluation complete"
