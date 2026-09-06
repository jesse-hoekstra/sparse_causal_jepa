# Project subagents

Specialized Claude Code subagents for building the codebase of **"Causal Identification within JEPA
Using a SPARTAN"** (`sources/SCJEPA.pdf`). Each agent owns one area and hands off to the next.
Claude routes automatically on the `description` field, or call one explicitly.

**Source of truth for settled decisions: [`docs/decisions.md`](../../docs/decisions.md).**
Framework is **PyTorch** (D1); reuse-first via vendored `third_party/` code (D5).

## Stack & reference codebases
- PyTorch · Hydra · W&B · einops · scipy (Hungarian) · ruff · pyright strict · pytest · jaxtyping
- [le-wm](https://github.com/lucas-maes/le-wm) (MIT) — JEPA training loop + SIGReg-style regularizer
- [visreg](https://github.com/HaiyuWu/visreg) (CC BY-NC 4.0) — vendored regularizers (not active in either current experiment)
- PyTorch SAVi (e.g. SlotFormer's), validated against
  [official JAX SAVi](https://github.com/google-research/slot-attention-video) (Apache 2.0) — D2: SAVi, not SAVi++
- SPARTAN: **no public code** — implemented from the paper (`sources/SPARTAN.pdf`, arXiv:2411.06890)

## Roster & rough order of use

| # | Agent | Owns | Use it when |
|---|-------|------|-------------|
| 1 | `research-repo-architect` | packaging, layout, tooling, third_party/ vendoring convention | first — scaffold + conventions |
| 2 | `paper-to-code-translator` | faithful method→code (SPARTAN, losses, channel split), verifying vendored code against its paper | anything traceable to an equation |
| 3 | `model-architecture-engineer` | `src/scjepa/models/` — SAVi, AttnPooling + KinematicHead split, SPARTAN, auxiliary conditioning | building the modules |
| 4 | `data-pipeline-engineer` | `src/scjepa/data/` — CLEVRER, Push-T, synthetic ground-truth systems, loaders | reproducible batches |
| 5 | `experiment-infra-engineer` | `training/`, `eval/`, `configs/`, `scripts/` — le-wm-based loop, loss assembly, SHD/MCC + CLEVRER + Push-T evals | wiring runnable experiments |
| 6 | `test-and-ci-engineer` | `tests/`, `.github/workflows/` | fast CPU tests + CI |
| 7 | `run-forensics` | W&B/run diagnosis, failure-mode naming | a run failed or looks wrong — BEFORE editing code |

## Architecture facts every agent must respect

- Two experiments: true-state `state_to_state`, and pixel-based `visual_to_visual` with a
  stop-gradient EMA target. The supervised visual-to-state bridge is retired.
- Both train teacher forcing plus eight sampled T=2 endpoints with one shared `theta_hat`;
  K=30 is evaluation-only. See D37/D39 for sampling and gradient contracts. D42 uses the same
  raw TF+weighted-T=2+logit constraint in both experiments, with the path penalty outside and
  target variance confined to diagnostics. Visual runs require `raw_tf_t2_v1` provenance.
- Visual parameter encoding applies relational attention across tracks before per-track temporal
  pooling. A shared row-wise head produces the dynamic state; SPARTAN predicts the next state.
- Experiment 2 matches online/target context-prefix slot trajectories once per episode and holds
  the permutation for all targets. Physical-object matching is an evaluation-only operation.
- Target encoder/head update only by EMA after accepted steps. No reconstruction or anti-collapse
  regularizer is active; track non-collapse and state-recovery empirically.
- Report MCC, SHD, and density together. Visual slot diagnostics and frozen state probes determine
  whether representation learning supports interpretation of those identification metrics.

## Conventions shared by all agents
- Reuse first: adapt vendored code over rewriting; record provenance (URL + SHA + changes).
- Strict typing (pyright strict, jaxtyping shape annotations), ruff, documented tensor shapes;
  `third_party/` exempt from lint/type gates but wrappers fully tested.
- Reproducibility first: seed everything, log git SHA + full config per run.
- Smoke-test on CPU before any expensive run; never claim tests pass without running them.
- Never commit data/checkpoints; new real decisions go in `docs/decisions.md`.