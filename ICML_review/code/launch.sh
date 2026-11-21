#!/usr/bin/env bash
# Run from the checkout; the caller supplies a pinned commit and an unused run name.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
expected=${1:?expected Git commit}
kind=${2:?diagnostic}
seed=${3:?seed}
detail=${4:?arm or experiment}
[[ "$seed" =~ ^[0-4]$ ]]
case "$kind/$detail" in
  donors/full|donors/baseline|chart/paired|halving/synthetic|halving/vision) ;;
  *) exit 2 ;;
esac
check_head() { [[ "$(git rev-parse HEAD)" == "$expected" ]]; }
check_head
export PYTHONPATH="$PWD:$PWD/src"
out="ICML_review/outputs/$kind-$detail-s$seed"
mkdir "$out"
trap 'echo "$?" > "$out/exit_code"' EXIT
cmd=(.venv/bin/python -m ICML_review.code.run "$kind" "$seed" "$detail" "$out")
case "$kind" in
  donors)
    hypothesis="A fixed five-episode support set can supply one code to all disjoint recipient episodes with little change in two-step forecast error."
    change="Average donor codes before prediction. Reference averages losses of the five individual donors on exactly the same recipients, using both fixed half-panel assignments."
    verdict="Predeclared practical criterion: pooled MSE at most 1.02 times single-donor MSE in every full validation seed, with the alien pooled-code MSE at least twice the within-system MSE. Failure limits the deployment claim. Equal recipient codes conditional on fixed support do not imply equality across independent supports. Report within-support forecast dispersion and code variance." ;;
  chart)
    hypothesis="Some of the full-model advantage in the partner-bead hold is explained by the scalar intervention interface."
    change="Replace scalar assignment and isotonic change of units by an invertible multivariate affine chart fitted on the same calibration states. Fix the models, interventions, horizons and validation contexts."
    verdict="Use paired absolute partner-slope errors. If the mean full advantage shrinks by at least half and the affine baseline improves, narrow the causal claim to the scalar interface. Otherwise report the remaining error and whether the affine chart itself fits poorly. This control uses richer joint supervision." ;;
  halving)
    hypothesis="At half budget the upper half by training performance contains the original acquisition winner."
    change="Stop the eight worst training-loss candidates at 50k; complete the other eight with their original histories, seeds and per-survivor budget."
    verdict="Proceed beyond seed zero within a dataset only when the same admitted winner survives. A changed winner leaves its post-projection outcome unknown. If all winners survive, identical historical initializers and downstream branches can be reused in this replay. No claim about future seeds or wall time." ;;
esac
{
  printf '# %s / %s / seed %s\n\n' "$kind" "$detail" "$seed"
  printf '%s\n\n%s\n\n%s\n\n' "$hypothesis" "$change" "$verdict"
  printf 'Checkout %s\n\n' "$(git rev-parse HEAD)"
  printf 'Launch: bash ICML_review/code/launch.sh %q %q %q %q\n\n' "$expected" "$kind" "$seed" "$detail"
  printf 'Command: PYTHONPATH=$PWD:$PWD/src '; printf '%q ' "${cmd[@]}"; printf '\n\n'
  if [[ "$detail" == synthetic ]]; then
    grep 'data/masked-mlp-gelu-20261101/' outputs/reported-checkpoints.sha256 | grep -v test
  else
    sha256sum outputs/data/vision-dimer-rgb-v1/manifest.json
    grep 'data/vision-dimer-rgb-v1/' outputs/reported-checkpoints.sha256 | grep -v test
  fi
} > "$out/NOTE.md"
check_head
"${cmd[@]}" > "$out/log.txt" 2>&1
check_head
