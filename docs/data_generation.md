# Reproducible scientific data

The graph structure and spring process follow
`sources/Parameter_and_State_Identifiability.pdf`, Sections 10.1 and 10.4. The vector
process below is a user-approved modification of Section 10.2, with stronger
couplings and paired four-frame episodes. This layer supplies data, reference
calculations and structural graphs. Training/evaluation protocols remain separate choices.

## Selecting data

`DGPConfig` and `build_dgp_dataset` select a lazy, deterministic process:

```python
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig

vector = build_dgp_dataset(DGPConfig(
    family="controlled_vector", parameter_graph="complementary",
    root_seed=17, observation_seed=1729, split="train",
    num_episodes=8, trajectory_length=4, episodes_per_parameter=2,
))
training_pair = vector[0]         # observations, actions, episode_ids only
references = vector.ground_truth(0)  # explicit evaluation-only access
```

| Family | Observation shape per frame | True state / parameter / action dimensions | History |
|---|---|---|---|
| `controlled_vector` | `(10,)` | `10 / 10 / 10` | 1 |
| `independent_gaussian` | `(10,)` | `10 / 10 / 10` | 1 |
| `coupled_springs` | `(3,320,64)` | `10 / 10 / 5` | 2 |

Single-episode dataset items have observations `(T,*observation_shape)`, actions
`(T-1,d_a)`, and a scalar int64 `episode_ids`. With either vector family and
`episodes_per_parameter=2`, one item is a parameter group: observations `(2,T,10)`,
actions `(2,T-1,10)`, and one scalar group ID in `episode_ids`. `num_episodes`
counts groups in this mode, so 8 items contain 16 trajectories. The generic
`DGPConfig` defaults to one episode; the controlled experiment selects two.
Latents, parameters, masses, force histories, coefficient matrices and
`valid_transitions` live only in `ground_truth(index)`.
Graphs and observation-map assets are separate dataset metadata. None of these
labels or true graph locations are passed to the model or loss.

Hydra presets `data=controlled_vector` and `data=coupled_springs` compose data
settings with compatible model interface defaults. The spring preset uses the
paper's eight-frame parameter context. The generic vector preset uses
`context_len=2` and trajectory length 3; its dataset size 8 and seeds remain small
interface defaults. The controlled experiment uses four-frame paired episodes.
`model.context_len` remains independently configurable. Data
settings never select sparsity weights or turn true graphs into predictor masks.

## Controlled vector process

The dimensions `d_z=d_theta=d_a=10`, `L=1`, and graphs match the paper.
The following coefficients replace its original `epsilon=1/40` construction.
Each parameter group draws `theta ~ N(0,I10)` once, then independently draws
`z0 ~ N(0,I10)` and `Uniform(-1,1)` action coordinates for episodes A and B.
The same theta is used throughout both episodes:

```
F[i,j](a) = (1 + a[i]**2) * (1 + a[j]**3)
H = diag(J5, J5)
A(a) = 0.5 I + 0.1 * (H * F(a))
B_G(a) = exp(sum(a) / sqrt(10)) I + 0.25 * (G * F(a))
z[t+1] = A(a[t]) @ z[t] + B_G(a[t]) @ theta
```

Here `*` between matrices denotes elementwise multiplication. Rows index outputs
and columns index inputs; `F` is not transposed.

The controlled experiment uses four frames `z0 -> z1 -> z2 -> z3`: the first
transition provides the two-frame parameter context, and the next two are
prediction targets. Training uses A's context-derived parameter code for B's
two-step rollout, and B's code for A's rollout. With one state lag, the predictor
starts from the target episode's encoded `z1`, then uses its own prediction of
`z2` to predict `z3`; each step uses the target episode's action and the same
partner parameter code. Loss averages both steps and both directions. This
supplies known same-parameter grouping without passing
true parameter values or graphs to training. Independent initial states/actions
make shared parameter information useful across episodes; no new consistency
loss is added. The second prediction does not receive an observed later state
that could reveal more about theta. These choices do not guarantee learned parameter separation or a
loss margin for every edge under a learned representation.

| Parameter variant | G | State / action / parameter union edge counts |
|---|---|---|
| `dense` | `J10` | `50 / 100 / 100` |
| `complementary` | `[[I5,I5],[I5,I5]]` | `50 / 100 / 20` |
| `diagonal` | `I10` | `50 / 100 / 10` |

Both coefficient matrices are invertible for every allowed action: each masked
block is a positive scalar identity plus a positive rank-one update. One transition
therefore recovers theta smoothly, and invertible state propagation preserves full
conditional state support. Distinct monomials `a[i]**2 * a[j]**3` span every allowed
matrix entry; the exponential term cannot cancel a polynomial on an open set.
Since `DF(0)=0`, each action derivative at zero is `theta / sqrt(10)`. These facts
retain the paper's fixed-output sufficient-influence arguments. They establish DGP
properties, not injectivity or identifiability of the learned encoders.

Observations use `x=psi(W @ z)`, `psi(u)=u+0.1*u**3`. A dedicated observation seed
realizes one dense orthogonal W through float64 QR with a fixed sign convention.
It is independent of episode/split/root sampling, and shared across graph variants.
Both its actual entries and fingerprint are persisted. To reuse a stored matrix
when generating additional splits, pass
`build_dgp_dataset(config, observation_matrix=bundle.observation_matrix)`.
Seeded QR alone does not promise bitwise agreement across numerical libraries;
the stored realization is authoritative when loading a bundle.

`controlled_vector.simulate` accepts explicit initial states, constant parameters
and actions. `VectorObservationMap.inverse` uses a stable analytic cubic inverse
and the orthogonal inverse of W. `recover_parameters` implements Eq. (94) with a solve:
`solve(B_G(a), z_next - A(a) @ z)`. These reference utilities are evaluation-only.
Choose another graph with `dataclasses.replace(config, parameter_graph=...)` to
reuse exactly the same exogenous draws and W; only the parameter mask changes.

## Independent Gaussian vector process

`family="independent_gaussian"` selects the separate two-reflection and radial-rotation
construction. It supports the complementary graph only, with the same 50 state,
100 action and 20 parameter edges. Initial states, parameters and actions are
independent standard Gaussians; paired episodes share only their parameter draw.
Its separate generator version and resolved family enter dataset fingerprints.
The old controlled-vector sampler and coefficients remain unchanged.

The process uses raw action coordinates to construct two orthogonal block mixers.
Each mixed state is paired with a densely mixed Gaussian action and rotated by an
angle depending on its radius and the two permitted parameters. This preserves
`z_t | theta ~ N(0,I10)` exactly in the mathematical model. See the
[complete equations and run protocol](independent_gaussian_experiment.md).

The generic preset is `data=independent_gaussian`. The experiment preset uses five
frames and two episodes per parameter: the first three frames and two actions
form the recovery context, followed by two cross-episode autoregressive targets.
The declared minimum recovery context is three; compatibility checks reject a
two-frame model. Training items contain only observations, actions and group IDs.
Ground-truth states and parameters stay in the evaluation-only sidecar.
Observations reuse `VectorObservationMap`, including the persisted realization of W.

This family does not provide the older process's `state_matrices` and
`parameter_matrices`: its transition is nonlinear. Use its derivative utilities
for mechanism checks rather than treating it as a linear coefficient model.
Generate ordinary samples without the controlled-vector-only `--diagnostics` flag:

```bash
python scripts/generate_data.py --config configs/data/independent_gaussian.yaml \
  num_episodes=1
```

## Coupled springs and action indexing

The paper fixes five pairs, displacements only, `theta=1/m ~ Uniform(0.5,1.5)`,
`dt=0.1`, `gamma=0.2`, and `K=diag(K0,...,K0)`, `K0=[[2,-1],[-1,2]]`.
Five independent uniform actions act equally on the two members of each pair:

```
z[1] = z[0] + dt * v[0]
force[t] = -K @ z[t] - gamma * (z[t]-z[t-1])/dt + repeat_pairs(a[t])
z[t+1] = 2*z[t] - z[t-1] + dt**2 * theta * force[t]    # t >= 1
```

Gaussian positions and velocities are independently sampled. The paper leaves
their means/scales unspecified: defaults are mean zero and standard deviation one,
configurable via `position_mean`, `position_std`, `velocity_mean`, `velocity_std`.
These overrides apply only to springs; the vector process retains its prescribed
standard Gaussian distribution. No clipping, noise, rejection sampling, or
resampling is added.

**Slot `a[0]` is sampled and retained, but unused.** It does not generate `z[1]`.
`valid_transitions[0]=False` and all later entries are true; the force sidecar's
index zero is an explicitly invalid zero placeholder. Actions are never shifted.
For a context of frames `0..7`, reference parameter recovery uses transitions
`t=1..6` and actions `a[1:7]`. Prediction from `[z6,z7]` consumes `[a6,a7]`;
the true mechanism depends on current action `a7` only.

`coupled_springs.recover_parameters(states, actions)` implements Eq. (105):
`theta_i = sum_t(force_ti * (z[t+1,i]-2*z[t,i]+z[t-1,i]))`
`/ (dt**2 * sum_t(force_ti**2))`, summing valid sources `1..K-2`.
Zero/nonfinite or small force energies explicitly raise an error listing affected
coordinates. The default numerical threshold is
`1e-12 * max(1, max_i sum_t force_ti**2)`; this is an implementation conditioning
choice, not a sampling rule. At least three frames are needed for this reference.

## Fixed smooth observation renderer

`SpringRenderer` maps only the current ten positions to RGB frames. Five native
64-by-64 panels stack vertically. Each pair uses red/green channels, with blue
zero, and horizontal centers `32+20*tanh(z_i)`. Gaussian widths, amplitudes and
vertical tracks are fixed across episodes, masses, actions and time.

Explicit appearance choices are `sigma=3` pixels, amplitude `1`, and local track
y positions `(20,44)` on a `0..63` pixel grid. Intensities evaluate the smooth
Gaussian at that grid; centers are never rounded. There is no resizing,
thresholding, hard rasterization, normalization, clipping or quantization of
training frames. Simulation and rendering calculations use float64; observations
and actions are stored in the configured `float32` (default) or `float64` dtype.
Float32 conversion loses precision. Even float64 `tanh` saturates at large
positions, and Gaussian tails may underflow. Floating-point rendering is not
claimed to be globally injective. Preview conversion, if later added, must remain
separate from the training observation map.

## Graph orientation and numerical checks

`GraphMasks.state/action` have `(output,lag,input)` axes; lag zero is oldest.
`parameter` has `(output,parameter)` axes. `aggregate_lags()` computes logical
union, not a sum of edges across lags. For springs:

```
state lags:  [I10, diag(J2,J2,J2,J2,J2)]
action lags: [zeros(10,5), I5 kron [[1],[1]]]
parameter:   I10
```

Union edge counts are `20/10/10`; the state lag masks contain 30 entries if summed.
Masks express structural dependencies even when a derivative vanishes at a sample.
Both families expose analytic Jacobians and tests compare them with autodiff and
independent float64 transition references.

Controlled-vector diagnostics use the fixed-output constructions of Eqs. (96),
(98), with the modified zero-action derivative `theta / sqrt(10)`. They check matrix
spans and fixed outputs with explicit SVD tolerances `atol=1e-10, rtol=1e-9`. Tests attain state
rank 50, parameter ranks 100/20/10, and action-column rank 10. Spring checks
combine the two state-lag spans (10 each) to rank 20, check action-child rank 2,
and parameter rank 10, with fixed-output constructions and rank tolerance `1e-12`.
These are small
implementation diagnostics alongside the analytical arguments above, not
numerical proofs of identifiability or learned representation quality.

## Reproducibility, windows and bundles

Initialization, parameters and actions use independently addressed CPU generators.
A versioned SHA256 derivation uses root seed, split, episode/group index and stream
label; no global RNG, worker number, batch size, access order or graph variant enters.
The four splits are `train`, `validation`, `evaluation_calibration`, `test`.
IDs pack a 31-bit nonnegative root seed, two-bit split, and 30-bit episode/group
index injectively into int64. For paired vectors, independent initial states and
actions are drawn successively from their group streams; the parameter stream
supplies the one shared vector. Split addresses are disjoint and parameter draws
are independent; training, calibration, validation and test never share groups. Graph variants
intentionally share original group IDs and exogenous draws for comparisons.

`EpisodeWindows(dataset, window_length, stride)` retains the original ID and
slices action `t` with source frame `t`, preserving both views for paired vectors.
`window_bounds` exposes original frame indices and `original_ground_truth` returns full original references. Parameter
regularization still rejects multiple windows from one episode/group in a microbatch.

Retrieval renders one episode or parameter group on demand and caches no
dataset-sized tensor. `save_dataset_bundle` uses the same generator and writes
one item at a time:

```
manifest.json          version, resolved config/specification, graph masks,
                       dtype, seeds/indexing, provenance fingerprint, checksums
observation_map.pt     actual W, for the vector process
training/00000000.pt   observations, actions, episode_ids only
ground_truth/00000000.pt  separate scientific tensors
```

Renderer settings are in the manifest. Saves refuse existing destinations and
publish only after successful staging. Loading uses tensor-only deserialization,
validates provenance, paths, IDs and checksums, and checks each shard again on read.
Memory scales with one item, not all video episodes; checksum verification scans
files with bounded memory. Checksums ensure integrity, not cryptographic authorship.

Generated `.fingerprint` identifies the resolved process, masks and realized map.
A materialized bundle keeps that provenance fingerprint and adds a
`.content_fingerprint` binding actual artifact checksums. Trainer checkpoints bind
these identities, and reject changed data before restoring state. Accordingly,
exact resume stays within the same generated definition or stored artifact source;
changing from lazy generation to a bundle is conservatively rejected. Plain saved
tensor files remain supported through `TensorEpisodes.load`, with content identity.
Dataset definitions and contents must remain immutable after trainer setup.

Compatibility is checked before training: observation shape/dtype, action count
and alignment, sequence length, and declared minimum memory/recovery context.
Learned state and parameter dimensions remain model choices; true coordinate
counts do not silently set them. Float64 datasets require a float64 model when
used directly. Default CLI models and presets use float32.

## Commands and extension

Inspect one small vector pair or spring episode without writing data:

```bash
python scripts/generate_data.py --config configs/data/controlled_vector.yaml \
  --diagnostics parameter_graph=complementary num_episodes=1 trajectory_length=3 \
  episodes_per_parameter=2
python scripts/generate_data.py --config configs/data/coupled_springs.yaml \
  num_episodes=1 trajectory_length=9
```

Append `--output /new/bundle/path` to materialize; inspect an existing bundle with
`python scripts/generate_data.py --bundle /existing/bundle/path`.
Configuration-only training-entrypoint checks (no training):

```bash
python scripts/train.py data=coupled_springs --cfg job
python scripts/train.py data=controlled_vector data.parameter_graph=diagonal --cfg job
```

For future authorized runs, use the same selections without `--cfg job`, or keep
`data` unset and provide `data_path=/path/to/tensors.pt` or a bundle directory.
No scientific training schedule or evaluation protocol is selected by these presets.

To add a DGP, implement the small `DGP` protocol with separate sampling, deterministic
simulation, observation mapping and graphs; optionally subclass `DGPConfig`, then
call `register_dgp(name, constructor, config_type)`. Add its config preset and
contract/reference tests. The model and trainer need no family-specific branch.
