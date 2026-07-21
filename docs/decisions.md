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
3. **MCC = their F.1 metric**: one sample per episode (`E x 5` learned versus `E x 5` true
   in the scalar bounce baseline); per (true param, learned dim) pair, a 1-hidden-layer MLP
   (width 32, ≤5000 samples, 90/10 held-out split) gives nonlinear R²; MCC = mean-max.
   Implemented as `nonlinear_mcc` (R² clamped at 0 = no explanatory power), reported as
   `mcc`; the former pooled episode-object probe is `mcc_pooled`, and Pearson `mcc_linear`
   is the fast proxy. Verified: planted tanh diffeomorphism scores 0.955
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

**Amended 2026-07-13 (Jesse): context_len 10 → 30, clip_len 40 → 60 (K stays 30).**
At 0.1425 contacts/ball/step (measured), a 10-step context leaves ~22% of ball-slots with no
ball-ball collision — no mass evidence in Ŝ^ph, capping MCC (~0.8-ish vs Baumgartner's ~0.9)
and dragging the dense reference floor toward the mass-blind floor (narrowing the τ window).
30 steps ⇒ ~99% collision coverage. Verified upstream difference: Baumgartner's parameter
encoder p_φ(θ̂|τ) conditions on the FULL trajectory (dynamical_system.pdf p.6, p.22); our
context-window Ŝ^ph is the JEPA-side deviation, so the context must at least carry the
evidence. All reference floors and τ re-measured at (Tp=30, K=30, ctx=30); W&B run names now
carry -Tp{}-K{}- for disambiguation.

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

## D17 — The dual's constraint is scale-free: pred / Var(target) (decided 2026-07-14, Jesse — implemented by Claude per instruction)

**Why.** Raw pred loss is an MSE in a TRAINABLE target space; its scale is a solution-dependent
equilibrium between the pred loss (pushes std down) and VISReg's unit-variance pull (D3). The
Tp=30/ctx=30 reference pair proved the confound: identity (A≡0, W&B e2vwbrlo) equilibrates at
std ≈ 0.46 while dense (A≡1, j2h9xc2m) climbs to ≈ 0.54 and is still rising at 50k — so the RAW
floors differ by only +0.008 (0.191 vs 0.183, within eval noise; every τ is then either
unsatisfiable or empty-graph-satisfiable, the v2/v3 collapses), while the SCALE-FREE floors
separate decisively: identity 0.90 ± 0.01 (flat from 10k; ≈ predicting the batch mean, as
mass-blind chaos should) vs dense 0.63–0.66 and still improving — mass/interaction information
is worth ~30% of relative prediction error. Both upstream papers had a fixed ruler for free
(Baumgartner: observation space, Eq. 9 p.7 verified; SPARTAN: frozen embeddings); the trainable
JEPA target space broke an assumption their constraint never had to state.

**Decision.** The quantity the dual compares to τ — and the eval harness's `constraint_loss` —
is `pred / Var(target batch, detached, floored 1e-6) + λ_logit·logit_penalty` (variance = mean
per-dim variance of the batch's target slots; same formula in trainer and harness, mean of
per-batch ratios). τ is a relative-error target: 1.0 ≈ predicting the batch mean. The GRADIENT
objective is unchanged (raw pred + λ_logit·logit + λ_reg·reg + (1/λ)·|Ā|): the normalization
only drives the λ update, so pred/reg balance and scale equilibria are untouched, and the model
cannot game the constraint by inflating variance through a gradient path. L_logit STAYS inside
the constraint per Eq. 9's letter — its ~0.022 gated-model share was fatal in a 0.008-wide raw
window but is affordable in the ~0.26-wide normalized one. This is the project's one deliberate
deviation from Eq. 9 as printed and is flagged here and in loop.py/harness.py comments.

**Consequences.** eval/constraint_loss and sparsity/constraint are in normalized units from
D17 on — NOT comparable to any earlier run. τ protocol: 1.1× the dense reference's converged
normalized constraint (dense leg of run_bounce_example.sh, same length as main, D16), with the
sanity check τ < mass-blind floor (identity reference; 0.90 at Tp=30/ctx=30 — this transfers,
identity converged flat from 10k). Raw pred_loss and target_var stay logged for diagnostics.

## D18 — Grad-spike skip guard, non-finite-grad failure, rolling checkpoints (decided 2026-07-17, Jesse — implemented by Claude per instruction)

**Why (post-mortem of run 7wupt6pw).** At step ~67.1k a rare batch kicked the predictor
(pre-clip grad norms 3.8e5 → 3.3e8) into a >1 per-step gain regime; the Tp=30 autoregressive
chain amplified it exponentially (gain ~3 ⇒ outputs ~1e15, MSE ~1e30 — FINITE, so the
`isfinite(total)` guard passed); BPTT through the chain overflowed to grad_norm = inf, and
`clip_grad_norm_`'s coefficient max_norm/inf = 0 silently multiplied every gradient by zero.
The run finished its remaining 230k steps as a frozen zombie (47 byte-identical evals), and
because `last.pt` is overwritten every 2k steps, no pre-explosion state survived. The same
spike family appeared ~5 times in the dense calibration (peaks ~7e4) and recovered by luck.
It died ~1k steps before crossing τ: the D17 objective and τ=0.584 are vindicated, not
implicated — this is optimizer robustness, not objective design.

**Decision.** In `Trainer._train_step`: if the PRE-clip grad norm (clip_grad_norm_'s return)
is non-finite OR above `train.grad_skip_threshold` (default 1e3; healthy grads here are <1
with rare transients <25), the batch's update is rejected entirely — no optimizer step, no
dual/EMA update (a pathological batch must not jolt the λ controller). After
`grad_skip_max_consecutive` (default 2000; amended 2026-07-17, was 50) consecutive skips the
trainer RAISES: the model is no longer trainable and must die loudly, not finish. The limit is
a stuck-run test, not a broken-weights heuristic: weights are FROZEN during skips (patience is
free), every retry is a fresh draw (epoch reshuffles + per-forward gate sampling), and the
counter resets on ANY calm batch — so reaching N consecutive means the calm-batch rate is
below ~1/N. At 2000 (~8 bounce epochs, ~10 wall-clock minutes) that is a dead run, not an
unlucky one, while a true zombie still dies within minutes instead of finishing 230k frozen
steps. Evidence for the amendment: run 0ta5ymcw's first episode (step ~4000) recovered after
149 skips interleaved with calm batches (~50% pass rate), but the original limit of 50
executed its second episode (step 6594) mid-burst; deterministic data order means a resume
replays the identical death, so the limit had to change, not the seed. Cumulative skips are logged as
`health/skipped_steps` and ride along in checkpoints (exact resume). Additionally
`train.checkpoint_keep_every` (bounce: 25000) keeps step-tagged `step_<N>.pt` fallbacks so a
late failure is a resume, not a rerun. Applies identically to calibration and main runs
(D12-safe); no objective term changes; τ=0.584 remains valid.

**Watch.** `health/skipped_steps` should stay 0 or near-0 (isolated skips = the guard doing
its job on known spike episodes); a climbing counter or the consecutive-skip RuntimeError
means the instability got through anyway — diagnose the episode (suspect: exp logit-penalty
wall × rare batch × 30-step BPTT) rather than raising the threshold.

## D19 — Per-chain gate-noise coupling in rollouts (decided 2026-07-17, Jesse — implemented by Claude per instruction)

**The problem.** Three consecutive main runs (7wupt6pw, 0ta5ymcw, u94wqvcb) died in the same
regime: once train path_density enters ~0.55–0.7, straight-through gradients explode
(1e4–1e6) on essentially every batch (u94wqvcb: 100/100 batches skipped for 1600 straight
steps, weights frozen by the D18 guard). Reproduced locally in 3000 CPU steps: healthy grads
(~0.1) until density ~0.6, then the spike regime. Mechanism: a D16 chain calls the predictor
Tp=30 times, and each of the 2 layers redrew fresh Bernoulli noise per call — 60 independent
hard-mask resamplings inside ONE backward graph. At mid density gates flicker i.i.d. at
unchanged state, the chain's step-Jacobians are randomly rewired 30x, and their product is
heavy-tailed. Forward stays healthy; only backward detonates.

**What the sources say (all three PDFs read 2026-07-17).** Nothing — the situation cannot
arise upstream. SPARTAN's objective is single-transition (Eq. 6, p.5); rollouts are eval-only
(Table 1, Fig. 2); no public code exists (D1); its Gumbel-softmax reference (Jang et al.)
defines the trick for a single sampling site. Baumgartner's decoder "performs one-step
prediction" (Fig. 1 caption, p.2) and the paper nowhere states free-running training; App. E/F
contain no rollout-training mechanics. my_paper defines the S_Tp chain but says nothing about
gate sampling (its adjacencies are deterministic theory objects). No upstream gradient ever
crosses more than one sampling round, so the cross-step coupling of the noise is genuinely
unprescribed design space.

**Decision.** `Spartan.sample_gate_noise` draws each layer's logistic thresholds ONCE per
rollout chain; `rollout_predictions` passes them to every step of that chain. Precisely:
per-step Bernoulli marginals are preserved (P(open) = σ(logit) at every step — Eq. 3 exact,
so E|Ā| and the τ protocol are untouched); within-chain independence is deliberately replaced
by common-threshold coupling (a gate flips mid-chain only when its state-dependent logit
crosses the chain's fixed threshold — collision-driven local-graph switching fully preserved);
across-chain/batch independence is preserved (exploration volume unchanged). Honest note: the
chain loss is nonlinear, so changing the coupling changes the expected objective — this is a
choice between two unprescribed objectives, not a pure variance trick. The correlated one
matches the theory's semantics (one fixed f_θ generates the whole trajectory: Baumgartner
Eq. 27, my_paper S_Tp) and is optimizable; the i.i.d. one is demonstrably not. Each chain is
also a cleaner experiment: one coherent graph hypothesis held for the horizon, so the loss
verdict a logit learns from measures the 30-step value of an edge instead of a smear over
2^30 flicker patterns. Unchanged by construction: Tp=1 (bit-identical), dense/identity
references (sample nothing), eval (deterministic thresholding), τ = 0.584.

**Validation (repro: τ=0.55, dual step 5e-3 — pushes density 5x faster than production).**
Pre-D19: every-batch spike regime on first contact with density ~0.6. Post-D19: trains through
density 0.50 → 0.78 with grads 0.1–0.4 and λ responsive; residual rare spiky batches remain
(86/3000 ≈ 3%, isolated singles/pairs, zero consecutive runs) and are absorbed by the D18 skip
guard — D19 removes the systemic explosion, D18 handles the tail. 86 tests green, incl. new:
frozen-noise forward is deterministic; dense/identity draw nothing; a Tp=4 chain consumes
exactly the RNG of a Tp=1 chain.

**Watch.** During the mid-density transit, `health/skipped_steps` climbing SLOWLY (a few per
1k steps) is expected post-D19; the fatal signatures are consecutive-run growth or every-batch
skipping — if seen, D19 was insufficient and the next lever is the Gumbel temperature
(softer ST gradients), not the skip limit.

## D20 — gt_states: rollout in the raw GT state space (decided 2026-07-18, Jesse — implemented by Claude per instruction)

**The problem (2026-07-18 full-source audit, after run pxibnvjr).** The D17 constraint
`pred / Var(target)` turned out to be a variance thermostat, not a prediction-quality
measure: raw pred MSE is pinned at ~0.176–0.19 in EVERY model (dense kn8g9xgu from 10k to
175k steps; gated pxibnvjr; identity — the "raw floors differ by only 0.008" observation) by
the VISReg-vs-MSE scale equilibrium, so ALL quality information lives in the trainable
target variance (dense: 0.26 → 0.342 over 150k steps, still drifting at kill time — the
reference constraint never converges). τ from a dense reference therefore demanded the gated
model reach ~92% of the dense model's own variance ceiling; the dual railed λ for 15k steps
and the run died in the (independently fatal, see below) high-density Tp=30 BPTT detonation
— which the DENSE calibration run also suffered (grads 1.3e9 at 100k, terminal explosion at
180k), proving the horizon, not the gates, is the root instability. Root cause of the
measurement problem: SPARTAN's constrained optimization presumes a FIXED-scale loss space
(frozen embeddings; Baumgartner: observation space) and a jointly-trained JEPA target space
has none — D17 normalized by a ruler the model itself controls.

**Decision.** `model.gt_states=true` (StateJepa only): the rollout state space is the raw
ground-truth states — exactly Baumgartner's observation-space decoder (their Eq. 21,
h(x0, θ) autoregressive in x-space). Anchors and targets are the untouched `states` tensor;
`target_embed` and `kinematic_head` are dropped (nn.Identity / None); the predictor's state
tokens are k=4-dim (predictions are literal next states) while Ŝ^ph stays in slot space via
Spartan's new `param_size` (separate input projection; `param_size=None` keeps the shared
projection and a pre-D20-identical state_dict). Only the parameter path (context embed +
pooling) and the predictor train. Consequences: the constraint's MSE has a fixed ruler
(D17's denominator becomes the data constant Var(GT batch) ≈ 0.15 — kept for continuity of
the τ protocol, now harmless); τ from the dense reference transfers; scale collapse
(failure #3) and the two-space rollout (anchor space ≠ target space) are impossible by
construction; VISReg on target_slots contributes a constant with zero gradient (loss/reg
carries a fixed GT-shape offset in this mode — cosmetic only).

**Not addressed here.** The Tp=30 full-horizon BPTT instability is orthogonal and remains
(the dense run's explosions prove it needs no gates); if gt_states runs still detonate
during the density transit, the lever is `train.rollout_horizon` (Tp=5–10 chains keep D16
mass-blindness pressure with 3–6x fewer chained Jacobians), not the skip guard.

## D21 — Teacher-forced one-step objective on the GT ruler for bounce_baumgartner (decided 2026-07-19, Jesse — implemented by Claude per instruction)

**Supersedes D16's default for this experiment** (the rollout machinery stays available via
`train.rollout_horizon=null` as the ablation; D16's reasoning is preserved below).

**The change.** `bounce_baumgartner` now trains with `model.gt_states=true` (D20) +
`train.rollout_horizon=1`: every one of the K=30 transitions is predicted from the TRUE raw
state at its own timestep (teacher forcing), losses (aligned MSE, |Ā|, logit penalty) are
computed per transition and averaged, Ŝ^ph is pooled once per episode and shared across all
transitions, and the predictor weights are one shared module across time. Gate noise is
drawn fresh per transition (a Tp=1 chain = one draw), which is exactly SPARTAN Eq. 3's
i.i.d. Bernoulli — D19 is untouched and vacuous here. Hyperparameters restored to SPARTAN
App. A.1 Table 3 (Interventional Pong): 3 layers, embed 512, MLP hidden 512, lr 5e-5.

**Why teacher forcing is now safe (and D16's rollout was compensating for a bug we fixed).**
D16 existed because teacher-forced single-step was mass-blind: forced-FC 0.0596 vs
forced-identity 0.0639 (a 0.008 raw window) — measured PRE-D17 in the TRAINABLE embedding
space, where the encoder could smooth collisions into near-linear latent paths. On the fixed
GT ruler the picture inverts (scratch measurement 2026-07-19, 128 episodes, exact replication
of simulate_bounce verified at error 0.000000): the STRONGEST mass-blind one-step predictor
(true physics, nominal mass/radius) has normalized MSE 0.199 vs 0.0 for the oracle —
0.332 on ball-ball-contact transitions (32% of transitions), 0.135 even on contact-free ones
(nominal-radius wall-bounce timing errors; radius ∝ mass). A ~0.2 τ window at Tp=1 dwarfs
the pre-D17 0.008 one. Mass-blindness was a representation-shortcut artifact, not a property
of one-step prediction.

**Source fidelity.** This is the MOST paper-faithful configuration yet: SPARTAN's objective
is literally single-transition (Eq. 7, App. A.2); Baumgartner's decoder likelihood
factorizes over one-step Markov transitions conditioned on true states (their Fig. 1
caption: the decoder "performs one-step prediction"; free-running rollouts appear only at
generation/eval, their Fig. 4); my_paper p16's autoregressive-rollout argument constrains
the GENERATIVE model h(S_t, Ŝ^ph), which a Markov model trained by exact MLE defines
identically — the theory does not require the training gradient to flow through 30 chained
predictions. Bonus: with Tp=1 there is NO BPTT chain, so the Tp=30 detonation (which hit
even the dense reference — see D20 audit) is structurally impossible, and the 30
transitions are computed in ONE batched predictor call instead of 30 sequential ones.

**Context/prediction windows (empirically re-validated 2026-07-19).** Ball-ball contact
rate 0.1438/ball/transition (config comment's 0.1425 confirmed). Ball-slots with zero
ball-ball contact in context: Th=10 → 19.7%, Th=20 → 5.6%, **Th=30 → 1.41%**, Th=40 → 0.62%.
Th=30 stays: near-full mass-evidence coverage without cutting K below 30. K=30 at Tp=1 is
just 30 supervised transitions/episode (~4.3 ball-ball collision transitions per ball) —
there is no longer a stability reason to shrink it.

**Watch.** The empty-graph signature to watch is unchanged (eval/path_density = 1/T,
shd_param frozen at its constant) — but with the 0.199 window, if the graph STILL prunes
param edges under a sane τ, that would now be evidence against the method, not the plumbing.

## D22 — Fit the two-phase dual schedule into 300k steps (decided 2026-07-19, Jesse+Claude after run maj7im56)

**The evidence.** The first full D20/D21 pipeline (calibration fybk7ukv, main maj7im56, Isambard
GH200) was healthy end to end — zero skips, train/eval gap ~5%, τ=0.0923 passed the guard —
but the main run spent ALL 300k steps in GECO phase 1: the gated constraint converged ONTO τ
from above (0.0947 at 300k, still declining), λ rose monotonically to 3.8e5, sparsity weight
1/λ never exceeded ~3e-6, train density drifted to 0.977 unpruned. Yet the identifiability
signal had begun anyway: eval MCC bottomed at 0.026 (init-artifact washout) then climbed to
0.109, mcc_linear 0.15→0.291 monotone over the last 150k, real param edges appearing in the
thresholded graph. The movie was cut at the start of act two.

**The two mis-calibrations it exposed.**
(a) τ = 1.1 × (dense reference trained the SAME length) is only reachable at ~the end of the
main run BY CONSTRUCTION — the reference crossed the equivalent level at ~130k of its own
300k, and the gated model tracks the same curve ~10-15% higher (gate noise). Phase 2 is
scheduled at t ≈ end. (b) sparsity_step_size=1e-3 (F-10) was tuned to SPARTAN Fig. 5's λ
trajectories — which are Pong/CREATE at 4e6 steps. Baumgartner's bounce (Fig. 17 x-axis,
verified) completes BOTH phases in 300k. NOTE: the paper specifies NONE of these numbers
(no L* protocol, no dual step, no clamps, no lr/batch — only the GECO citation, the
qualitative two-phase description, and Fig. 17's 300k/8-seed scale); every value is our own
bridge, so the schedule SHAPE is the only fidelity target available.

**Decision.** For bounce_baumgartner: `sparsity_step_size: 3e-3` (two-phase arc fits 300k;
still 7x below F-10's disaster 0.02), `sparsity_lambda_max: 3e4` (newly wired through
TrainConfig — GECO's ascent is unbounded but its descent is rate-limited by the model's
bounded undershoot of τ; capping the overshoot makes the reversal immediate at crossing),
launch with `--tau-factor=1.2` (τ ≈ 0.10: crossable ~step 180k → ~120k of pruning budget;
kept below the observed learned state-only level ≤0.125 to limit F-8/F-9-style slack).

**Open risk (unchanged).** The learned no-param-edge floor (velocity→mass leak) is not
precisely measured; if it lies below τ, pruning can delete param edges and satisfy the
constraint (the outcome that would indict the setup, not the plumbing). A dense-but-
params-masked reference would measure it directly if the D22 run prunes param edges away.

**Watch.** After the constraint crosses τ (~180k): λ must reverse within ~20-30k steps,
train density must stop rising then FALL, shd_param must descend from its spurious-edge
peak toward the GT graphs, MCC must continue the ramp maj7im56 started. λ pinned at the 3e4
clamp for >50k steps post-crossing = the clamp is too low or τ still too tight.

## D23 — Strip the schedule back to paper-faithful; α is the only free knob (decided 2026-07-20, Jesse+Claude after run m5bje3yt)

**The evidence.** D22's run m5bje3yt (τ=0.1007=1.2×, `sparsity_step_size` 3e-3, `sparsity_lambda_max` 3e4) hit the exact failure D22's own "Watch" note named: λ railed at the 3e4 clamp from step ~28k to ~210k, and after the constraint finally crossed τ (~190k) λ fell only 30k→26k while `eval/path_density` sat flat at 0.128 for 200k steps — pruning never engaged. Re-reading the papers (SPARTAN App. A.2 / Eq. 7–8; Baumgartner §3.3 / Eq. 9–10, Fig. 17 x-axis = 300k): both prescribe *plain* GECO (Rezende & Viola 2018) with τ = the fully-connected model's loss, and specify NO dual step size, NO clamp, NO τ factor. So D22's clamp and 1.2× factor were both un-paper hacks. The bounce dataset was re-audited against Baumgartner App. E (5 balls, square box, elastic ball-ball + wall, mass ~ non-zero-mean normal, radius ∝ mass) and matches — `mass_normal=[1.5,0.5]`, `radius_from_mass=true` are set; the data is not the discrepancy.

**Mechanism the clamp broke.** 3e4 floored 1/λ at 3.3e-5, i.e. ~0.03 of standing sparsity pressure on every step (~37% of pred loss ~0.08). That pressure held pred loss ~0.0975, *above* the dense floor 0.0839 — so no tight τ was ever reachable, clamp or not. Reverting to the base 1e6 ceiling cuts standing pressure ~33× (1/λ→~1e-6), letting the model reach ~dense loss, cross τ, and only THEN prune. Offline dual sim on the run's own constraint magnitudes confirmed: at 3e-3, 1/λ is still ~1e-5 at 300k (no prune); at 2e-2 the descent completes with budget to spare, while a *truly* unclamped λ climbs to ~1e19 in phase 1 and no feasible α recovers it in 300k — so 1e6 stays as a recovery guard, not a functional cap.

**Decision.** For bounce_baumgartner: `sparsity_step_size: 2e-2` (the value bounce_states already validated; F-10's "0.02 too fast" verdict was against SPARTAN's 4e6-step horizon, not Baumgartner's 300k), `sparsity_lambda_max: 1e6` (base default; D22's 3e4 functional clamp removed), τ = **1.0 ×** dense reference (isambard_main_only.sbatch default TAU 0.1007→0.0839). Momentum stays 0.99 (it sets the equilibrium's lag/noise, not its location — not a pruning lever). Every schedule number is now either the paper's or, where the paper is silent (α only), ours.

**Open risk (the one that now decides the run).** D22 measured the *gated* model tracking the dense reference ~10–15% higher (stochastic gate noise). τ=1.0× therefore sits near/below the gated constraint floor: it is only crossable if, at 1/λ→0, gates open enough to shed that noise and reach ~dense loss. If `eval/constraint_loss` plateaus above 0.0839 through the first ~60k steps, τ=1.0× is unsatisfiable — and the culprit is gate commitment (Gumbel temperature, `lambda_logit`), NOT the schedule. Fallbacks in order: (1) re-calibrate the dense reference in-pipeline under the exact config (fybk7ukv is from a previous pipeline; at factor 1.0 any mismatch breaks the constraint directly), (2) only then consider the minimal gate-noise slack (~1.05×).

**Watch.** First 2–3 eval points (per the "check the first two eval points" rule): `eval/constraint_loss` must fall toward 0.0839 and cross it by ~100–150k. Once crossed: λ descends off 1e6, `eval/path_density` falls off 0.128, `shd_param` descends from its spurious-edge peak, MCC ramps. Failure signatures: constraint_loss stuck >0.0839 (τ unsatisfiable → gate-noise problem); or λ crashes to `lambda_min` with density→1/T and MCC at the noise floor (over-prune / empty-graph collapse #2 → α too high or τ too loose).

## D24 — Correct the true-state mass-identification contract (decided 2026-07-20, Codex audit; accepted by Jesse)

**Scope correction.** D20/D21 correctly fix the prediction ruler with `gt_states=true`; the
pipeline-calibrated τ, rather than the inherited smoke placeholder, remains the operative
constraint. The remaining failure was not evidence against the raw-state rung: its parameter
encoder and decoder did not represent five clean, paired masses.

**Decision.** `bounce_baumgartner` now uses `pooling_type=track_aware` and `param_dim=1`.
Each simulator track is pooled temporally before permutation-equivariant cross-object mixing,
then a shared scalar head emits exactly one latent per ball. SPARTAN receives a relative
parameter-i → state-i relation (`spartan_paired_object_attention=true`), which preserves joint
object-permutation equivariance but makes an independent reassignment of mass tokens visible.
Absolute slot labels remain absent. Prediction loss is aligned MSE in the literal GT-state
regime; learned slots retain set matching only as a temporary fallback pending a persistent
tracking/trajectory-level assignment.

**Sparsity phase.** The path objective now sums only paths ending at decoded state outputs.
The Stage-1 run starts λ at its 1e6 ceiling and uses a 10k-step non-sparsity warm-up: gates
still learn through prediction and logit gradients (with VISReg active), while the path term
and dual update are disabled.
This implements the papers' intended dynamics-first phase instead of allowing the raw path-count
gradient to close gates during the first few updates.

The primary `path_density` now uses those same decoded state rows; the former whole-token
density remains available as `path_density_full`. Parameter-token output rows are not decoded
and therefore are neither optimized nor allowed to make the main pruning curve look stuck.

**Constraint and feasibility.** D17 remains the default for trainable target embeddings, but
literal GT states already provide the papers' fixed ruler. This rung therefore uses raw aligned
MSE (plus the logit term) in both the primal and dual, superseding D17's normalization for this
regime. Tau is exactly `1.0 x` the converged dense-reference loss, as prescribed in SPARTAN
App. A.2; non-unit factors are permitted only as explicitly labelled slack ablations. All pre-D24
numeric tau and identity floors are invalid. The reporting launchers now run
a matched `spartan_identity=true` reference and abort unless the freshly calibrated tau is below
its held-out raw constraint loss.

**Simulator correction.** The old overlap detector changed event timing only in discrete
substep jumps: on a fixed collision branch, recorded positions were locally independent of
radius, leaving common mass scale non-identifiable despite `radius_from_mass=true`. Wall
overshoot is now reflected about the radius-dependent contact surface, and pair penetration is
projected to the exact radius-dependent surface before the elastic impulse. Pre-generated data
is versioned (`simulator_version=2`); old preloads are rejected and must be regenerated.

**Evaluation and reproducibility.** `mcc` is now Baumgartner App. F.1 shaped: one row per
episode (`E x 5` learned versus `E x 5` true for the scalar baseline). The previous pooled
episode-object diagnostic is retained as `mcc_pooled`. Model construction is seeded, and the
complete evaluation harness preserves training RNG state.

**Next rung.** When visual encoders are enabled, ordered loss is valid only if state and
parameter tokens descend from the same recurrent slot track and future targets retain that
track. Otherwise use one assignment over a complete trajectory, never independent per-frame
Hungarian assignments.

## D25 — Replace paired mass slots with persistent global coordinates (decided 2026-07-21, Jesse+Codex)

**Supersedes part of D24.** D24's raw-state ruler, aligned state prediction, simulator fix,
decoded-row path objective, and fresh τ calibration remain. Its `track_aware` per-object mass
head, learned same-index attention bias, 10k sparsity warm-up, and diagonal recovery diagnostic
do not. Those choices made the desired state–mass binding available before sparsity and therefore
could not test whether the graph discovered it. The same-index bias option and its learned
parameter are removed from the implementation rather than retained as an ablation.

**Architecture.** The parameter encoder now has five learned latent-coordinate queries. Each
query attends to the complete context tensor (all times and all tracked GT objects), with temporal
and source-track position embeddings, and emits one scalar. Thus the output is a persistent global
vector \(\hat\theta\in\mathbb R^5\), not five values defined as belonging to input tracks 1–5.
SPARTAN adds independent learned address tables to the five state nodes and five parameter nodes.
The tables are neither shared nor initialized alike, and no \(i=j\) state–parameter term is added.
They break within-episode coordinate exchangeability so a graph can learn a stable relation, while
leaving all 25 parameter-to-state edges available to sparsity. This fixed-address construction is
appropriate for ordered GT tracks; pixels still require recurrent slot continuation/tracking.

**Evaluation.** Mass recovery permits one global coordinate permutation but not a different
permutation per episode. Scalar MLP probes are fit on a probe-training fold for every
\((\hat\theta_j,m_i)\) pair; nonlinear R² on a disjoint alignment fold chooses one Hungarian
bijection; that frozen bijection is scored on a third fold and also aligns parameter columns before
graph SHD. `mass_mcc` is the only periodic MCC curve. Linear MCC, target variance, full-token
density, and the full pairwise matrices are final-only diagnostics. The recovery grid shows all
25 relations and highlights the chosen global assignment instead of rewarding its diagonal.

**Protocol.** The papers specify no explicit sparsity warm-up, so GECO/path pressure is active from
step zero; the large initial \(\lambda=10^6\) already makes its initial weight small. Baumgartner
calls `lambda_logit` only “small,” so `10^-3` is no longer treated as sourced: the experiment config
contains a neutral zero placeholder and reporting launchers require a value from a controlled dense
sweep. The sweep includes the zero control, uses a separate validation seed offset, and selects
without mass labels: within 5% of zero-control prediction, choose the smallest Pareto coefficient
that achieves 90% of the best admissible reduction of \(L_{logit}-2\). Final pipeline evaluation
uses a different test offset. This is only a dense feasibility screen; the decisive evidence remains
whether the gated run reaches τ and prunes while retaining prediction and `mass_mcc`.
