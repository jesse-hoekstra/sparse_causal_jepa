# Published Experiment 1 architecture

The [experiments section](../publication/experiment1.tex) owns scientific methods;
[current experiment](current_experiment.md) owns the selected results and local
artifact record. [Reproduction](../publication/REPRODUCING.md) provides commands.

The two active entry points are `diagnostics.masked_mlp_oracle` and
`diagnostics.masked_mlp_learned`; `diagnostics.shared` contains their common
simulation, cache, authentication and evaluation utilities. Core modules in
`src/scjepa` provide framewise encoders, the transition parameter encoder, JEPA
rollout/EMA, losses, logging and optimizer state.

## Interfaces

`SCJEPA` receives observations/actions, with an explicit oracle-parameter input
only for the oracle control. The learned runner constructs a model-input mapping
containing only observations/actions. Ground-truth labels remain separately
accessible to evaluation, never to the ordinary learned objective.

Grouped tensors have shape `(B,R,T,D)`. Parameter codes are per context, not an
average of the ten episode codes. A non-self within-group donor permutation
routes one code to each recipient. Both forecast steps use the same attached
code, and the first predicted state is the second input without detaching.
Frozen predictor weights still permit gradients to parameter codes.

`MaskedTrunkPredictor` shares its two hidden layers across output coordinates,
with output-specific static gates and linear readouts. It exposes the established
`SpartanOutput` tensor contract to avoid changing the rollout implementation.
The old predictor class is retained only as an initialization/interface dependency;
it is replaced before either published model is trained. Removing its construction
would change the original RNG stream and parameter-count matching.

The state target encoder is an EMA copy. Dense frozen-state acquisition/adaptation
freezes both online and target state encoders and does not update the EMA.
Sparse baseline/full training unfreezes online modules and resumes EMA updates.

## Reproducibility contracts

Caches isolate physical reference states from ordinary input tensors and address
groups deterministically by split. Training, calibration, validation and reserved
test group identities are disjoint. The ordinary loader leaves test unopened.

Completion records bind artifact hashes, source, data and scientific settings.
Checkpoints retain model, optimizer, GECO, sampler and RNG state. Source changes
have a new identity; historical resume requires its frozen runtime. The publication
refactor does not rewrite source hashes on old results.

Evaluation matches state coordinates on calibration and holds those assignments
fixed. Whole-code probes fit/select on separate calibration halves. Parameter
mask scoring additionally aligns learned code columns to physical parameter pairs;
state-row-only parameter SHD is not the reported learned-code graph metric.

The tests cover label exclusion, gradients across both forecast steps, exact
resume, projection, numerical DGP contracts and deterministic predictor symmetry.
Software checks do not substitute for recovery evidence.
