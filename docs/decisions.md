# Architecture & engineering decisions

Living record of decisions for the codebase implementing **"Causal Identification within JEPA Using
a SPARTAN"** (`sources/my_paper.pdf`). Each entry states the decision, the rationale, and what would
make us revisit it.

## D1 — Framework: PyTorch (decided 2026-07-09)

**Decision.** The codebase is PyTorch, not JAX.

**Rationale.** Plug-and-play reuse to minimize engineering errors is the stated priority. The
reusable reference codebases are PyTorch: [le-wm](https://github.com/lucas-maes/le-wm) (MIT; JEPA
training loop + Gaussian-embedding/SIGReg-style regularizer) and
[visreg](https://github.com/HaiyuWu/visreg) (official VISReg, CC BY-NC 4.0). The comparison/eval
ecosystem (SlotFormer, ALOE, DINO-WM, Push-T) is PyTorch as well. SPARTAN has **no public code**
(checked 2026-07-09; [arXiv:2411.06890](https://arxiv.org/abs/2411.06890)) and must be implemented
from the paper in either framework. The official SAVi repo
([google-research/slot-attention-video](https://github.com/google-research/slot-attention-video),
Apache 2.0, JAX, archived) is kept as the **correctness reference** for our PyTorch SAVi.

**Revisit if.** SPARTAN or C-JEPA code is released in JAX, or reuse of the PyTorch repos turns out
to be shallower than expected.

## D2 — Encoder: SAVi, not SAVi++ (decided 2026-07-09; re-confirmed 2026-07-09 against the full SAVi++ paper + official repo)

**Decision.** Use SAVi (`sources/SAVi++.pdf` describes the alternative).

**Rationale.** Benchmarks are CLEVRER and Push-T — synthetic/tabletop scenes. C-JEPA (Nam et al.
2026) and SlotFormer both build on SAVi for CLEVRER, so comparisons stay apples-to-apples. SAVi++'s
additions (depth-prediction targets, augmentation recipes) target real-world driving video (Waymo)
and need signals our datasets don't provide. Implementation source: a proven PyTorch SAVi (e.g.
SlotFormer's; verify license when vendoring), validated against the official JAX implementation.

**Re-confirmation evidence (sources/SAVi++.pdf + google-research/slot-attention-video, 2026-07-09).**
- The official repo ships both variants (`savi/configs/movi/savi_conditional_*.py` and
  `savi++_conditional.py`), JAX, Apache 2.0, **archived Apr 2024** — reference-only either way.
- The SAVi++ config hard-requires ground-truth **flow + depth** dataset fields as prediction
  targets (`targets: {"flow": 3, "depth": 1}`, `transform_depth(transform='log_plus')`,
  eval keys include `"depth"`). CLEVRER and Push-T ship neither. More fundamentally, in our JEPA
  the encoders are trained by the predictive loss + VISReg — there is no decoder-target loss at
  all, so SAVi++'s central contribution (depth as training target) has no place to plug in.
- SAVi++'s own Table 3: on the *simple* synthetic MOVi-A/B, SAVi++ **underperforms** SAVi
  (mIoU 76.1 vs 82.3 on MOVi-A; 25.8 vs 44.5 on MOVi-B), which the authors attribute to their
  scaling strategy overfitting on simple domains. CLEVRER's visual complexity is at or below
  MOVi-A (plain floor, simple geometric objects), so the SAVi++ recipe would likely hurt.
- Remaining SAVi++ ingredients (ResNet34+transformer encoder backbone, Inception-style crop
  augmentation) are orthogonal capacity/augmentation knobs, not architectural changes — they can
  be adopted individually via config later without revisiting this decision.
- Reuse: proven PyTorch SAVi implementations exist (SlotFormer); no established PyTorch SAVi++.

**Revisit if.** We move to real-world video or scenes where SAVi's segmentation visibly fails —
and then adopt SAVi++'s encoder-scaling knobs first, before its depth targets.

## D3 — Anti-collapse regularizer: VISReg (resolved 2026-07-09 after code inspection)

**Decision.** Use **VISReg** (Variance-Invariance-Sketching Regularization, official repo). The
SIGReg fallback rule does **not** trigger: inspection of both codebases shows the swap is trivial.

**Evidence (repos inspected 2026-07-09).**
- le-wm assembles its loss in one place (`train.py`):
  `loss = pred_loss + lambd * self.sigreg(emb.transpose(0, 1))` — the regularizer is a
  self-contained `SIGReg` module (from `module.py`) taking an embedding tensor and returning a
  scalar. The integration surface is literally one module attribute and one call site.
- The official visreg repo ships `visreg/losses/visreg.py`: `class VISReg(nn.Module)`
  (constructor: `num_projections`), forward takes a `(*, B, D)` embedding tensor, returns a scalar;
  torch-only, no extra dependencies. It matches Algorithm 1 of the paper (center + scale + SWD
  shape loss). The same directory also contains `sigreg.py`, `vicreg.py`, `swd.py`, `barlow.py` —
  the losses are designed to be interchangeable.
- So the swap = vendor `visreg/losses/`, replace `self.sigreg` with a config-selected regularizer
  module. No training-loop rewrite.

**Additional argument for VISReg (from `sources/VISReg.pdf`, Fig. 2):** SIGReg's gradient vanishes
as embeddings collapse — exactly when correction is needed. Since this project's collapse
prevention is *solely* the regularizer (joint training, no EMA/stop-gradient asymmetry), robust
gradients under collapse matter more here than in LeJEPA's setting. VISReg (and its variance term)
keeps a strong corrective gradient in the collapse regime.

**Implementation note.** The regularizer stays config-selectable (`visreg` | `sigreg`), vendored
from HaiyuWu/visreg (both files, CC BY-NC 4.0), so the ±regularizer comparison is a config flag —
useful as an ablation and as a safety hatch. Applied to slot embeddings: flatten `(B, N, d)` (and
any leading dims) to `(B·N, d)` samples — exact placement per Fig. 1 of the paper (both branches).

**Consequence.** The paper's Appendix B argument (bounded embedding variance ⇒ MSE ≡ maximizing
cross-covariance ⇒ spectral sufficiency) must hold under VISReg. Its variance (scale) term bounds
per-dimension variance directly, so the argument goes through; the manuscript text (which currently
says SIGReg) should be updated to VISReg.

## D4 — Attention pooling for causal parameters: per-slot temporal pooling (decided 2026-07-09)

**Decision.** The causal parameters are extracted by **per-slot temporal attention pooling** (a
PMA-style pooling-by-attention block, weights shared across slots), mapping the SAVi slot history
to `Ŝ^ph ∈ R^{N×d}`:

```
Input:  slot history  S̃ ∈ R^{B × Th × N × d}   (context SAVi encoder, all timesteps)
For each slot i (batched; parameters shared across slots):
    K = V = [ s̃ᵢ¹ + p₁, …, s̃ᵢ^Th + p_Th ]      # that slot's Th embeddings + learned temporal PE
    q ∈ R^d                                      # single learned query, shared across slots
    ŝ^ph_i = MHA(q, K, V)  (+ LayerNorm/MLP residual, standard PMA block)
Output: Ŝ^ph ∈ R^{B × N × d}
```

**Rationale.**
- Preserves SAVi's object binding: `ŝ^ph_i` is the parameter vector *of slot i* — no cross-slot
  mixing — which keeps Hungarian matching and the disentanglement diagnostics (SHD/MCC against
  per-object ground-truth parameters) structurally clean.
- Aggregates evidence **over the whole horizon**: time-invariant parameters (mass, friction,
  charge) are only observable from multi-frame behavior; temporal positional encodings let the
  pooling exploit order (finite-difference-like features).
- Matches the paper's Koopman story: pooling over time approximates projection onto the λ=1
  (time-invariant) eigenspace, while the kinematic channel `S_t = Linear(s̃^Th)` (last step, which
  has seen all frames) carries the λ≠1 evolving observables.
- Consequence to be aware of: *relational* parameters (e.g. pairwise coupling) cannot enter `ŝ^ph_i`
  through the pooling; they must be mediated by SPARTAN's cross-slot attention. This is consistent
  with the paper's causal-graph-over-slots formulation.

**Alternatives rejected.** (a) N learned queries jointly attending over all Th·N tokens — more
expressive but breaks slot↔parameter correspondence; (b) pooling only over the last timestep's slots
— relies entirely on SAVi's recurrence for temporal aggregation.

## D5 — Reuse-first policy

Vendor adapted third-party code under `src/scjepa/third_party/<name>/` with upstream license files
and a `PROVENANCE.md` (upstream URL + commit SHA + what was changed). Licenses recorded: SlotFormer
MIT (+ nerv MIT excerpts), le-wm MIT, visreg **CC BY-NC 4.0** (non-commercial — fine for this
academic project, flag before any commercial use), official SAVi Apache 2.0. Prefer adapting
vendored code over rewriting; prefer rewriting over depending on unmaintained packages at import
time.

**Amendment (2026-07-09, module 2).** Vendored code lives *inside* the package
(`src/scjepa/third_party/`, importable as `scjepa.third_party.<name>`) rather than at the repo
root: a root-level `third_party/` is not importable from the installed package and would need
sys.path hacks or a colliding top-level package name. Lint/type/pre-commit exemptions unchanged
(path patterns updated).

## D6 — Predictive loss: single-step, Hungarian-matched over N slots (decided 2026-07-09, Jesse)

**Decision.** Training prediction is **single-step**: SPARTAN maps (S_t, Ŝ^ph, optional U_t) →
Ŝ_{t+1}, compared against the target branch's S_{t+1}. Both are `(B, N, d)` — the time axis exists
only at the *input* to the Ŝ^ph pooling (slot history); S_t is the last-timestep SAVi output through
the linear head. The Hungarian match is therefore one assignment per sample over the N slots
(scipy `linear_sum_assignment` on a detached `(N, N)` squared-distance cost; gradients flow through
the matched pairs only). No multi-step matching-consistency machinery is needed at training time.

**Consequence.** Multi-step rollouts (CLEVRER 128→160) are an *eval-time* autoregressive procedure,
not a training objective; Tp is an eval knob. Loss invariants to test: permutation of target slots
leaves the loss unchanged; loss is 0 when prediction equals target.

## D7 — SAVi encoders train from scratch (decided 2026-07-09, Jesse)

**Decision.** No pretrained-checkpoint initialization; both SAVi encoders train from scratch,
jointly, driven only by the predictive loss + VISReg (+ SPARTAN sparsity). No config hatch for
pretrained init — that emergence-without-reconstruction is part of the claim being tested.

**Revisit if.** Slot binding demonstrably fails to emerge (collapse metrics healthy but slots don't
segment objects) — then surface to Jesse before adding any pretrained-init or reconstruction crutch.

## D8 — Packaging & tooling (decided 2026-07-09, module 1 scaffolding)

**Decision.** Python **3.12 only** (`>=3.12,<3.13`) — newest CPython line with universal wheel
coverage for the stack; one minor version keeps environments reproducible. `src/` layout, setuptools
backend, package name `scjepa`. **All tool config lives in `pyproject.toml`** (ruff, pyright,
pytest) — one file to read, one file to edit. Dependencies pinned to a stable line with an upper
bound guarding major-version jumps (torch `>=2.9,<3.0`, hydra `1.3.x`, einops `0.8.x`, scipy
`>=1.14,<2`, jaxtyping `0.3.x`); rationale comments inline in `pyproject.toml`.

**Quality gates.** ruff (broad rule set incl. docstrings, annotations, bugbear; line length 100),
pyright **strict**, pytest with `--strict-markers`. `third_party/` is excluded from ruff, pyright,
and pre-commit (D5) — its wrappers in `src/` carry the full bar. Pre-commit runs hygiene hooks +
ruff (rev pinned in sync with the pyproject pin) + pyright as a `language: system` hook so it sees
the project venv where torch is importable.

**Revisit if.** A dependency needs Python 3.13+, or a pinned line blocks a needed feature — bump
deliberately and record it here.

## D9 — Target branch encodes the single future frame (decided 2026-07-09, Jesse)

**Decision.** The target SAVi encoder consumes **only frame t+1** (le-wm-style):
`(B, 1, C, H, W) → (B, 1, N, d)`, and these raw slots (time axis squeezed) ARE the prediction
target S_{t+1} ∈ (B, N, d) — **no linear head on the target branch** (Jesse, 2026-07-09: "the
encoded target slots are sent to the loss function directly"). The channel-split heads
(AttnPooling, KinematicHead) exist on the context branch only.

**Dimensional consequence.** The prediction target lives in slot space, so SPARTAN's output — and
hence the context branch's S_t — must have dimension d: `KinematicHead.state_size` stays at its
default (= slot_size).

**Weight sharing (default, revisitable).** Context and target encoders are two SEPARATE SAVi
instances with independent weights (Fig. 1 read literally; sharing is possible since SAVi's
weights are clip-length-independent, but is not what we build unless Jesse decides otherwise).

**Consequences (accepted deliberately).**
- With one frame, SAVi reduces to per-image Slot Attention: no recurrence, appearance-only
  binding; slot order arbitrary relative to the context branch — the Hungarian match (D6) carries
  the correspondence.
- The target representation is structurally position-like (a single frame cannot encode velocity),
  so the predictive loss supervises no velocity content in S_{t+1}.
- The internal slot predictor (transformer+LSTM) is never invoked at T=1 (verified: 0/18 parameter
  grads), so the target encoder is built with `single_frame=True`, which skips constructing the
  predictor entirely rather than carrying untrained dead weight.

**Revisit if.** Learned dynamics look degenerate (e.g. SPARTAN forced to infer velocity it cannot
be supervised on) — then widen the target window (frames 1..t+1 or a short window) via config.

## D10 — SPARTAN implementation choices (2026-07-09, module 5; from sources/SPARTAN.pdf, no public code)

Everything traceable to the paper follows it: Eq. 3 (hard adjacency `A_ij ~ Bern(σ(q_i·k_j))`,
binary Gumbel-softmax straight-through), Eq. 4 (masked attention, `ŝ_i = MLP(h_i + s_i)`), Eq. 5
(path matrix `Ā = (A_L + I)···(A_1 + I)`, parent iff `Ā_ij ≥ 1`), Eq. 6 (penalty `|Ā|`, exposed as
`SpartanOutput.sparsity`; the App. A.2 Lagrangian-relaxation schedule `max_λ min_θ (MSE − τ) +
|Ā|/λ` belongs to the training loop, module 6). Choices the paper leaves open — flag to Jesse if any
looks wrong:

1. **Mask BEFORE softmax normalization** (renormalize over unmasked entries). The printed Eq. 4 is
   ambiguous (garbled denominator index); masking after normalization would leak masked tokens'
   information through the softmax denominator, contradicting the paper's own "adjacency …
   disallows information flows" claim and breaking the Ā-based causal readout. Verified by test:
   `Ā_ij = 0 ⇒ ∂pred_i/∂token_j = 0`.
2. **Token layout (from my_paper.pdf):** [N state S_t | N params Ŝ^ph | M aux U_t]; predictions
   read from state positions; aux appended as extra tokens (my_paper §4.1) and projected d_u → d.
3. **Learned role embeddings** (state/param/aux) added to tokens — roles distinguishable, slot
   symmetry kept (paper silent; slot-permutation equivariance pinned by test).
4. **Single-head attention** per layer (Eqs. 3-4 define one adjacency per layer).
5. **Eval mode deterministic:** `A_ij = 1 iff logit > 0` (no sampling); Gumbel temperature is a
   config knob (default 1.0; paper does not state its value).
6. `|Ā|` includes the constant pure-residual diagonal contribution (zero gradient, matches the
   paper's `|Ā|` definition literally).
7. All-masked rows: `h_i = 0` (+ ε in the denominator), i.e. pure self-dynamics via the residual.
8. **App. A.1 hyperparameters aligned (added after checking the full appendix, per Jesse):** the
   paper separates "token dimension" (32/64) from a larger "embedding dimension" (512) — so the
   model projects tokens `d → embed_dim` on entry and `embed_dim → d` at the prediction head
   (`embed_dim` config, default 512, `None` = work at d); each layer's MLP has 3 Linear layers
   (`mlp_num_layers`, default 3), hidden 512/1024; L = 3 layers. Training defaults for module 6:
   Adam, lr 5e-5. The remaining appendix (B datasets, C baselines, D pretrained representations)
   contains no further mechanics; the Gumbel temperature and Eq. 4's normalization order are
   genuinely unspecified in the paper — choices 1 and 5 above stand.

## D11 — Synthetic ground-truth systems: Baumgartner's environments rendered to pixels (2026-07-09, Jesse)

**Decision.** The identifiability experiments use the four environments of
`sources/dynamical_system.pdf` (dual particle, springs, local particle, bounce), **rendered to
64x64 video** — the primary observation is pixels through the full SAVi pipeline, since the paper's
theorem lives in the representation space learned from vision (Jesse: "I use vision data, and they
don't"). **Bounce is the guiding example** and is implemented first (`scjepa/data/bounce.py`).

**Every episode ships full ground truth** (the reason synthetic systems exist — Jesse): rendered
`frames`, kinematic `states`, causal `params` (masses, sampled per episode from a non-degenerate
range — sufficient variability), and the time-indexed local graph `contacts (T-1, N, N)` (True iff
the pair collided during that transition). Graph derivations for eval: state edge j→i iff contact
(+ self-edges); parameter edge mass_j → state_i iff contact involving i and j (own mass matters
only during collisions — free flight is mass-independent, pinned by test).

**Mass is NOT rendered** — equal radii, identity by color. A single frame reveals positions but
never S^ph, preserving the D4 premise that parameters are observable only from multi-frame
behavior. (If mass were geometric, the target encoder's single frame would leak S^ph.)

**GT-embedding diagnostic regime** (encoders bypassed) kept alongside, per Jesse's experiment list
item 1 and SPARTAN App. D: isolates channel-split+SPARTAN identifiability from encoder quality,
calibrates the Lagrangian target τ. The vision run's SHD uses Hungarian slot↔object alignment.

**Physics:** elastic impulse exchange (restitution 1), mass-independent wall reflections,
symplectic Euler with substeps; torch-only, generated on the fly, deterministic per (seed, index).
Invariants pinned by tests: momentum/energy conservation through collisions, mass-independent free
flight, contacts symmetric/zero-diagonal, containment in the box.

**Next up on the same skeleton:** dual particle (the Fig. 4/5 marginal-recovery reproduction),
local particle, springs.

## D12 — Lagrangian sparsity × joint training: the regularizer must anchor the embedding scale (observed 2026-07-10)

**Finding (bounce_states, empirical).** SPARTAN's A.2 constraint `pred_loss <= τ` is
**scale-dependent**, and in our architecture the embeddings are trainable (D7) — so with a weak
regularizer weight (λ_reg = 0.1) the model satisfied the constraint by **shrinking the target
embeddings** (partial collapse: target per-dim std 0.31 → 0.04) instead of learning dynamics:
pred_loss → 5e-5 at any graph, λ_s fell freely (1000 → 12), the |Ā|/λ term grew to dominate the
objective, and SHD *degraded* while the "constraint" stayed satisfied. SPARTAN never encounters
this failure mode because its object embeddings are frozen (VAE/ground-truth, App. D).

**Decision.** Wherever the Lagrangian sparsity schedule is enabled, λ_reg stays at full strength
(≥ 1.0): VISReg's scale term `(std − 1)²` is what anchors the loss scale and makes τ meaningful.
This is a methodological point for the manuscript (interaction of A.2 with joint training), not
just tuning.

**Watch items.** `health/target_slot_std_*` are the early-warning metrics; a falling std alongside
a falling pred_loss during the pruning phase is this failure mode. Note also: in the GT-embedding
regime a linear 4→d embed has rank 4, so VISReg's shape term has an unavoidable floor —
scale anchoring still works, isotropy is structurally unreachable there.

## D13 — Exact comparison to Baumgartner et al. (dynamical_system.pdf), bounce (2026-07-10, Jesse)

**Goal.** Match their App. E bounce setting exactly enough that our MCC is comparable to their
Fig. 3 (SPARTAN ≈ 0.9+), establishing that identification works in the simple setting before
probing the method's edge. Config: `experiment/bounce_baumgartner.yaml`.

**What was adopted from their paper (with source):**
1. **Radius ∝ mass** (`radius_from_mass`) and **masses ~ non-zero-mean normal** (`mass_normal`,
   exact parameters unspecified upstream — ours: N(1.5, 0.5) clamped to [0.5, 3]). Their App. E.
   NOT cheating in their framework: the VAE encoder is *supposed* to extract θ (no information
   asymmetry to protect); the disentanglement pressure lives in the sparse decoder. It also
   strengthens sufficient variability (Assumption 6): mass acts through contact geometry, not just
   impulse ratios. Deliberately departs from D11 invisibility FOR THIS COMPARISON; the
   invisible-mass variant stays as the harder ablation (the "edge of the method" ladder:
   their setting → invisible mass → pixels).
   **No-bypass clarification (Jesse's question, 2026-07-10):** in the STATES regime the tokens are
   [x, y, vx, vy] regardless of `radius_from_mass` — mass/radius appears in no input or target, so
   the "parameter migrates into the state channel" bypass CANNOT occur there; the flag only changes
   the physics. The bypass exists only for "pixels + rendered radius∝mass", which no config uses
   except as a future negative control. **Scope condition for the manuscript:** a JEPA with learned
   targets partitions information by observability timescale — Ŝ^ph can only capture factors
   observable exclusively through dynamics; single-frame-observable parameters are absorbed into
   the state channel by construction (D9). State this explicitly (it lives implicitly in
   Assumption 3's "kinematic" encoding), and use pixels+visible-mass as the control experiment
   demonstrating its necessity.
   **Deeper form (Jesse, 2026-07-10) — the vision-regime recurrence leak:** in the STATES regime
   the split is enforced by construction (per-frame linear embeds; KinematicHead reads the last
   frame only ⇒ S_t provably mass-free; only the pooling sees multiple frames). In the VISION
   regime SAVi's recurrence makes last-step slots history-aware, so S_t *could* carry
   dynamics-inferred mass even when mass is invisible — and the sparsity penalty may PREFER that
   route (mass inside state tokens reuses the contact edges needed for kinematics anyway; zero
   parameter edges ⇒ sparser graph than the intended solution). Assumption 3 is exactly what rules
   this out and is therefore the paper's most load-bearing untested assumption. The ladder measures
   it: rungs 1–2 = identification with the split guaranteed; rung 2→3 delta = what the recurrent
   encoder does to the split. Mitigation levers if rung 3 fails this way: restrict KinematicHead
   capacity (state_size < d), or an information penalty on S_t.
2. **Attention-logit loss** (their Eq. 11; `train.lambda_logit`; their λ_logit unspecified —
   "small"). Their F.4: without it the path loss PLATEAUS and the model never sparsifies — the
   symptom our CPU runs showed. Exposed as `SpartanOutput.logit_penalty` (per-layer
   mean of exp(q·k)+exp(−q·k), logits clamped ±30 for finiteness); included in the Lagrangian
   CONSTRAINT (their Eq. 9: non-causal losses ≤ L*), so τ must be calibrated with it enabled.
   VISReg stays OUTSIDE the constraint (D12 — the collapse anchor must not trade against sparsity).
3. **MCC = their F.1 metric**: per (true param, learned dim) pair, a 1-hidden-layer MLP (width
   32, ≤5000 samples, 90/10 held-out split) gives nonlinear R²; MCC = mean-max. Implemented as
   `nonlinear_mcc` (R² clamped at 0 = no explanatory power), reported as `mcc` by the harness;
   Pearson `mcc_linear` kept as a fast proxy. Verified: planted tanh diffeomorphism scores 0.955
   (nonlinear) vs 0.886 (Pearson) — the metric is diffeomorphism-invariant as required.

**Known intentional differences (the experiment itself):** their model is a conditional VAE
(trajectory encoder + KL, β=1e-6) with SPARTAN as decoder; ours is the JEPA (D6–D9). Their scale:
~300k steps × 8 seeds (their Fig. 17) — GPU/overnight territory, not laptop CPU.

**Architecture correspondence (settled with Jesse, 2026-07-10).** Their "encoder" = the
trajectory→θ̂ map p_φ(θ̂|τ) (MLP over the flat trajectory; conv over time for bounce "for longer
sequences"). Its analog in ours is the embed→AttnPooling chain — NOT the per-frame embeddings.
Every rung's parameter channel sees the full Th history; rungs 1–2 make only the STATE/TARGET
channel memoryless (that is what guarantees the split). Two comparability caveats:
(a) their encoder mixes ACROSS objects; our pooling is per-slot (D4) ⇒ rung 1 is the same
environment but a strictly harder inference problem (m_i from own-history statistics only). If
rung-1 MCC < their ≈0.9, run the D4-rejected cross-slot pooling as a one-off ablation to attribute.
(b) their bounce trajectories are long (hence the conv encoder); our clip_len=10 gives each ball
0–3 own-collisions — use clip_len 30–50 for comparison runs (max_history=64 accommodates).
Unspecified upstream (candidate email to Anson Lei — worth sending BEFORE spending cluster time):
trajectory horizon, mass distribution parameters, λ_logit, and — most load-bearing — the BOUNCE
per-object token content. Their state provably includes velocities for dual/local particle
(App. E: "four-dimensional vector containing position and velocity") and springs (Fig. 15 nodes
ẋ, ẏ), but bounce token content is never specified, and "radius ∝ mass for ease of observation"
could mean radius is in the observation. Internal-consistency argument: radius cannot be in their
DECODER's state input (it would bypass θ̂ and contradict their Fig. 12 disentanglement), so our
[x, y, vx, vy] tokens are the defensible default — but if their bounce tokens DO carry radius,
their setting corresponds to our leak-exposed control, not our clean rung 1.

**Rung 2.5 (planned):** states regime + recurrent per-object state encoder (small GRU replacing
the linear embed). Isolates the recurrence leak (S_t becomes mass-capable) without the pixel/slot
-binding confound, so rung 2→2.5 measures the leak and rung 2.5→3 measures binding — one factor
per rung.

## D14 — Pooling upgraded to cross-slot attention (decided 2026-07-10, Jesse: "we will need cross slot attention — otherwise we lose information which we could have obtained")

**Decision.** The default Ŝ^ph pooling is now **CrossSlotAttnPooling**: one query per slot,
projected from that slot's LAST-step embedding (the identity anchor), attending over ALL Th·N
tokens of the history with learned temporal PE and deliberately NO slot-identity PE. D4's
per-slot pooling is retained as `model.pooling_type: per_slot` — the per-slot-vs-cross-slot
comparison is itself a paper experiment.

**Why.** Physical parameters of relational mechanisms are not inferable from own-history alone:
the bounce collision equation Δv_i = 2 (v_rel·n) m_j/(m_i+m_j) n needs the PARTNER's velocity,
which D4's per-slot pooling structurally discards (D13 caveat (a)); Baumgartner's trajectory
encoder mixes across objects. Cross-slot pooling restores that information while keeping:
- **slot↔parameter correspondence** — ŝ^ph_i stays "object i's parameters" via its anchored query;
- **slot-permutation equivariance** — queries permute with slots, the key/value set is
  slot-order-free (pinned by test);
- shapes/contract identical: (B, Th, N, d) → (B, N, d).

**What is given up.** Strict slot-locality of Ŝ^ph (the old locality test applies only to the
per_slot variant). D4's "relational effects are SPARTAN's job" division softens: relational
EVIDENCE may now enter ŝ^ph_i, while relational STRUCTURE (the graph) remains SPARTAN's readout.
**Manuscript impact:** D4's pooling spec and its Koopman rationale need updating; the anchored
cross-slot form still projects onto time-invariant observables, but the per-slot claim is gone.

## D15 — Sliding-window training: one Ŝ^ph, many single-step predictions (decided 2026-07-10, Jesse)

**Why.** The identification mechanism (Baumgartner Lemma 1 / the entanglement-map-independence
premise in our theory) is operationalized by ONE parameter estimate serving MANY transitions
through different states of the same system. Our previous training (one window → one Ŝ^ph → one
transition per episode) exerted one transition of identification pressure per parameter draw and
left Ŝ^ph free to be window/state-dependent — the exact state-dependence Lemma 1 exists to rule
out. Jesse: "we need to keep [Ŝ^ph] equal across our training samples… take in a history context,
predict the next state, then let the window pass one step" — the data as available in practice.

**Decision.** Model forwards gain ``context_len``: Ŝ^ph is pooled ONCE from the first context_len
steps and held fixed while the window slides — K = L − context_len single-step predictions
(S_t, Ŝ^ph) → S_{t+1} for t = context_len−1 … L−2, loss = mean over all. D6 is intact per
prediction (no rollouts, no autoregression, no gradient through time chains). Vision: the context
encoder runs ONCE over frames[:, :−1] (recurrence makes slots at t a function of frames ≤ t, so
slicing is exact); the D9 single-frame target encoder runs per predicted frame, folded into the
batch. ``context_len=None`` = legacy K=1. Flattened outputs (B·K, N, d); causal_params stays
(B, N, d) — one estimate per episode; eval harness builds one gt graph per transition.

**Config:** ``train.context_len``; bounce_states now L=40, context_len=10 (K=30, batch 16).
**Deferred:** per-window Ŝ^ph constancy diagnostic (within- vs across-episode variance) — direct
empirical check of the Lemma-1 premise, worth a paper plot; explicit constancy loss only if the
diagnostic shows drift. **Caveat noted:** nonlinear MCC needs ≥ a few hundred samples (tiny
validation splits make max-over-dims R² spuriously high; standard evals use ~2.5k samples).
## D16 — Autoregressive rollout is THE training objective; dense τ reference (decided 2026-07-12, Jesse — implemented by Claude per instruction)

**Why.** The v2 run (W&B qqye6ug1) failed with the graph pruned to identity and MCC at the eval
noise floor despite a healthy optimizer. Root cause (docs/audits/2026-07-12-*.md): D15's
teacher-forced single-step objective forgives a mass/interaction error at the very next step, so
edges were worth only ~7% of MSE (forced-FC 0.0596 vs forced-identity 0.0639 at equal budget) —
the empty graph satisfied ANY realistic τ, and the papers' scheme prunes until the constraint
binds. The theory this repo exists to test is premised on rollouts: my_paper p7/p16 defines
S_Tp = [S_t, f(S_t,Ŝ^ph), f∘f(S_t,Ŝ^ph), …] and the invariance proof assumes autoregression;
Baumgartner's decoder reconstructs whole trajectories from (x₀, θ̂) (§3.1, App. B.4). Jesse
2026-07-12: "This is VERY much needed and indeed what the entire identification is built on …
Make this standard, no flag."

**Decision.** Both model forwards now ALWAYS roll out: chains anchored at true encoded states
feed their own predictions back, reusing one Ŝ^ph (kinematic-anchor vs target-space mismatch is
tied together by the predictive loss; documented in ``rollout_predictions``). New
``train.rollout_horizon`` sets chain length Tp; must divide K; None = one chain over all K
(paper-literal); **Tp=1 reproduces the old D15 behavior and is the ablation path.** Default for
bounce: Tp=15 (2 chains of 15 for L=40/ctx=10) — Baumgartner leave trajectory length unspecified
(D13 email item); at Tp=15, ~90% of chains contain an own-ball collision (1−(1−0.1425)^15) and
BPTT depth stays manageable (~0.08 s/step CPU, same order as teacher-forced).

**τ protocol (with F-8/F-9 fixes).** ``model.spartan_dense=true`` gives A≡1 — SPARTAN's actual
"fully connected model" (p.16); the gated model with sparsity off is NOT that reference (its
gates keep sampling; audit F-8). run_bounce_example.sh: calibration runs dense, calibration
length defaults to the MAIN run's length (an undertrained reference inflated τ in v2:
6k-step 0.085 vs 0.045 achievable), TAU_FACTOR default 1.1 (slack is spent on gate-closure
depth via the logit term inside the constraint — audit F-9). Dual step for 300k runs: 1e-3
(bounce_baumgartner; SPARTAN Fig. 5 λ timescale — audit F-10).

**Amended 2026-07-13 (Jesse): default rollout_horizon = null (one chain over all K=30).**
Matched-budget reference runs under the Tp=15 objective (identity A≡0, W&B 9q18lw7r, vs dense
A≡1, cakn11ye) measured the mass-blind margin at only ~0.010 constraint (identity floor 0.170
vs dense 0.160 at matched steps 40k–105k) — smaller than the gated model's ~0.022 logit share
inside the constraint, so no reliably placeable τ exists at Tp=15. Longer chains compound
mass-blind rollout error; τ and both reference floors must be re-measured at the new horizon
(H=15 numbers do not transfer, D12). Tp=15 is retained as an ablation value only.

**Environment fixes (audit G1–G3).** radii = radius·m/mass_ref with an episode-INDEPENDENT
reference (mass_normal mean, else mass_range midpoint) — episode-mean normalization made
absolute mass unidentifiable by any model (MCC ceiling 0.775). Wall bounces are recorded on the
contacts diagonal iff radius∝mass (bounce point depends on r_i ⇒ GT param self-edge); the D11
equal-radius regime keeps a False diagonal. Initial placement uses per-ball radii. NOTE:
episode RNG consumption changed ⇒ same (seed, index) yields different episodes than pre-D16;
shd_param baselines shift (the old zero-param-edge constant 1.4595 no longer applies).

**Verified (smokes, 2026-07-12).** 78 tests green incl. new rollout/dense/env tests; training
stable (grad_norm < 0.4 through 15-step BPTT). Edge economics under rollout at 1500 steps:
identity floor 0.0973 vs FC 0.0812 (20% gap, was 7%) — state-edge pruning is no longer free.
**Open, and the v3 go/no-go check:** a mass-blind-but-interaction-aware arm still matched FC at
1500 steps (param-edge value emerges only once the model has learned to exploit masses — the
converged dense calibration run must beat the mass-blind floor, else no τ can force param edges;
watch eval/shd_param and the pred/logit constraint split early in the main run).
