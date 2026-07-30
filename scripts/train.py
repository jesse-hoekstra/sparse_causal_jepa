"""Training entrypoint: ``python scripts/train.py [experiment=bounce_baumgartner] [key=value ...]``.

Hydra manages the config and the run directory; the resolved config is saved
next to the run outputs. W&B is opt-in (``wandb.enabled=true``) so offline
development and CI never block on it.
"""

import subprocess
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from scjepa.training import MetricLogger, NoopLogger, TrainConfig, Trainer, seed_everything
from scjepa.training.factory import build_dataset, build_model
from scjepa.training.visual_to_state import VisualToStateTrainer
from scjepa.training.visual_to_visual import VisualToVisualTrainer


class WandbLogger:
    """Thin W&B adapter satisfying ``MetricLogger`` (imported lazily)."""

    def __init__(self, project: str, mode: str, config: dict[str, Any], name: str) -> None:
        """Start a W&B run tagged with the resolved config and git SHA."""
        import wandb

        if mode not in ("online", "offline", "disabled"):
            raise ValueError(f"unknown wandb mode {mode!r}")
        self._run = wandb.init(project=project, mode=mode, config=config, name=name)

    @property
    def run_id(self) -> str:
        """W&B run id, so post-hoc evaluation can resume this same run."""
        return str(self._run.id)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Forward metrics to W&B."""
        self._run.log(metrics, step=step)


def _source_of(key: str, overrides: list[str], preset: DictConfig | None) -> str:
    """Where a config key's value came from — CLI, experiment preset, or base.

    Answers "why is it this value?" without making anyone diff three yaml files.
    """
    if any(o.lstrip("+~").startswith(f"{key}=") for o in overrides):
        return "CLI"
    section, _, leaf = key.partition(".")
    if preset is not None and leaf in (preset.get(section) or {}):
        return "preset"
    return "config.yaml"


def _print_run_banner(cfg: DictConfig, experiment: str, phase: str, git_sha: str) -> None:
    """Print the decision-critical resolved values and where each came from.

    Every slurm log then OPENS with exactly what the run trained, so a wrong
    config is visible in seconds instead of being reconstructed afterwards from
    resolved_config.yaml. Keys listed here are the ones whose wrong value is
    silent rather than loud.
    """
    overrides: list[str] = list(HydraConfig.get().overrides.task)
    preset: DictConfig | None = None
    preset_path = (
        Path(__file__).resolve().parent.parent / "configs" / "experiment" / f"{experiment}.yaml"
    )
    if preset_path.exists():
        loaded = OmegaConf.load(preset_path)
        preset = loaded if isinstance(loaded, DictConfig) else None

    sparsity_on = bool(cfg.train.sparsity_enabled)
    rows: list[tuple[str, str, str]] = [
        ("experiment", experiment, "preset"),
        ("phase", phase, "-"),
        # tau is MISSING-by-design on reference stages, which never read it.
        (
            "train.sparsity_tau",
            f"{float(cfg.train.sparsity_tau):.6g}" if sparsity_on else "(unused: sparsity off)",
            _source_of("train.sparsity_tau", overrides, preset) if sparsity_on else "-",
        ),
        (
            "train.lambda_logit",
            f"{float(cfg.train.lambda_logit):g}",
            _source_of("train.lambda_logit", overrides, preset),
        ),
        (
            "train.lambda_roll",
            f"{float(cfg.train.get('lambda_roll', 0.0)):g}",
            _source_of("train.lambda_roll", overrides, preset),
        ),
        (
            "train.lambda_roll_warmup_steps",
            str(cfg.train.get("lambda_roll_warmup_steps", 0)),
            _source_of("train.lambda_roll_warmup_steps", overrides, preset),
        ),
        (
            "train.rollout_len",
            str(cfg.train.get("rollout_len", None)),
            _source_of("train.rollout_len", overrides, preset),
        ),
        (
            "train.sparsity_lambda_init",
            f"{float(cfg.train.sparsity_lambda_init):g}",
            _source_of("train.sparsity_lambda_init", overrides, preset),
        ),
        (
            "train.sparsity_step_size",
            f"{float(cfg.train.sparsity_step_size):g}",
            _source_of("train.sparsity_step_size", overrides, preset),
        ),
        ("train.steps", str(cfg.train.steps), _source_of("train.steps", overrides, preset)),
        (
            "train.context_len",
            str(cfg.train.get("context_len", None)),
            _source_of("train.context_len", overrides, preset),
        ),
        ("data.clip_len", str(cfg.data.clip_len), _source_of("data.clip_len", overrides, preset)),
        (
            "data.num_clips",
            str(cfg.data.num_clips),
            _source_of("data.num_clips", overrides, preset),
        ),
        ("data.preload", str(cfg.data.preload), _source_of("data.preload", overrides, preset)),
        ("git_sha", git_sha, "-"),
    ]
    width = max(len(name) for name, _, _ in rows)
    print("─" * 78)
    print("RUN CONFIGURATION")
    for name, value, source in rows:
        print(f"  {name:<{width}}  {value:<22} [{source}]")
    print("─" * 78, flush=True)


def _git_sha() -> str:
    """Current commit SHA + dirty flag (reproducibility record)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build model + data + trainer from the config and run."""
    # Trainer.__init__ seeds the training stream, but that is too late for
    # reproducible dataset/model construction. Seed before either is built.
    seed_everything(int(cfg.train.seed))
    experiment = str(HydraConfig.get().runtime.choices.get("experiment") or cfg.data.name)
    if cfg.model.get("spartan_identity", False):
        phase = "token-local"
    elif cfg.model.get("spartan_dense", False):
        phase = "dense"
    elif cfg.train.sparsity_enabled:
        phase = "sparse"
    else:
        phase = "gated-no-sparsity"  # ablation only; NOT the dense reference
    git_sha = _git_sha()
    # BEFORE anything is written or any W&B run is opened: the banner touches
    # every mandatory key, so a missing one aborts here rather than after a run
    # directory, a checkpoint and a W&B entry already exist. NOTE: never read
    # these with DictConfig.get(key, default) — it returns the default for a
    # MISSING value and would silently reinstate the footgun this replaces.
    _print_run_banner(cfg, experiment, phase, git_sha)

    out_dir = Path(str(HydraConfig.get().runtime.output_dir))
    resolved: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # pyright: ignore[reportAssignmentType]
    resolved["git_sha"] = git_sha
    OmegaConf.save(config=OmegaConf.create(resolved), f=out_dir / "resolved_config.yaml")

    dataset = build_dataset(cfg.data)
    eval_dataset = None
    if cfg.train.get("eval_every") is not None:
        eval_cfg = OmegaConf.merge(
            cfg.data, {"num_clips": int(cfg.train.get("eval_episodes", 128))}
        )
        eval_dataset = build_dataset(eval_cfg, seed_offset=17)  # pyright: ignore[reportArgumentType]
    model = build_model(cfg.model)
    train_config = TrainConfig(
        steps=cfg.train.steps,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        grad_clip=cfg.train.grad_clip,
        sparsity_enabled=cfg.train.sparsity_enabled,
        # MANDATORY (`???`) but read ONLY when the dual is active: accessing a
        # MISSING value raises, which is exactly the guard we want for sparse
        # runs and exactly what must NOT fire on dense/token-local references.
        sparsity_tau=float(cfg.train.sparsity_tau) if cfg.train.sparsity_enabled else 0.0,
        sparsity_step_size=cfg.train.sparsity_step_size,
        sparsity_lambda_init=cfg.train.sparsity_lambda_init,
        sparsity_momentum=cfg.train.sparsity_momentum,
        lambda_logit=float(cfg.train.lambda_logit),  # mandatory: .get() would mask MISSING
        rollout_len=cfg.train.get("rollout_len", None),
        lambda_roll=float(cfg.train.get("lambda_roll", 0.0)),
        lambda_roll_warmup_steps=int(cfg.train.get("lambda_roll_warmup_steps", 0)),
        seed=cfg.train.seed,
        device=cfg.train.device,
        context_len=cfg.train.get("context_len", None),
        eval_every=cfg.train.get("eval_every", None),
        grad_skip_threshold=cfg.train.get("grad_skip_threshold", 1e3),
        grad_skip_max_consecutive=cfg.train.get("grad_skip_max_consecutive", 2000),
        num_workers=int(cfg.train.get("num_workers", 0)),
        prefetch_factor=int(cfg.train.get("prefetch_factor", 4)),
        log_every=cfg.train.log_every,
        checkpoint_every=cfg.train.checkpoint_every,
        checkpoint_keep_every=cfg.train.get("checkpoint_keep_every", None),
        out_dir=str(out_dir),
    )
    lambda_logit = float(cfg.train.lambda_logit)
    run_name = f"{experiment}-{phase}-ll{lambda_logit:g}-seed{cfg.train.seed}"
    if cfg.wandb.get("run_tag") is not None:
        run_name = f"{run_name}-{cfg.wandb.run_tag}"
    logger: MetricLogger = (
        WandbLogger(project=cfg.wandb.project, mode=cfg.wandb.mode, config=resolved, name=run_name)
        if cfg.wandb.enabled
        else NoopLogger()
    )
    # Record the run id so scripts/eval_identifiability.py can attach the final
    # 5000-episode metrics and recovery_grid.png to THIS run instead of opening
    # a detached second entry.
    if isinstance(logger, WandbLogger):
        (out_dir / "wandb_run_id.txt").write_text(logger.run_id)
    # Each regime keeps every guard, checkpoint and resume path of the shared
    # Trainer and overrides only what its observation contract implies. The
    # state-to-state path is untouched by this dispatch.
    trainers = {
        "state_to_state": Trainer,
        "visual_to_state": VisualToStateTrainer,
        "visual_to_visual": VisualToVisualTrainer,
    }
    trainer_class = trainers[str(cfg.model.get("regime", "state_to_state"))]
    final = trainer_class(model, dataset, train_config, logger, eval_dataset=eval_dataset).train()
    print(
        f"done at step {train_config.steps}: " + ", ".join(f"{k}={v:.4g}" for k, v in final.items())
    )


if __name__ == "__main__":
    main()
