"""Training entrypoint: ``python scripts/train.py [experiment=smoke] [key=value ...]``.

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

from scjepa.models.jepa import SCJepa
from scjepa.models.state_jepa import StateJepa
from scjepa.training import MetricLogger, NoopLogger, TrainConfig, Trainer
from scjepa.training.factory import build_dataset, build_model


class WandbLogger:
    """Thin W&B adapter satisfying ``MetricLogger`` (imported lazily)."""

    def __init__(self, project: str, mode: str, config: dict[str, Any], name: str) -> None:
        """Start a W&B run tagged with the resolved config and git SHA."""
        import wandb

        if mode not in ("online", "offline", "disabled"):
            raise ValueError(f"unknown wandb mode {mode!r}")
        self._run = wandb.init(project=project, mode=mode, config=config, name=name)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Forward metrics to W&B."""
        self._run.log(metrics, step=step)


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
    out_dir = Path(str(HydraConfig.get().runtime.output_dir))
    resolved: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # pyright: ignore[reportAssignmentType]
    resolved["git_sha"] = _git_sha()
    OmegaConf.save(config=cfg, f=out_dir / "resolved_config.yaml")

    dataset = build_dataset(cfg.data)
    eval_dataset = None
    if cfg.train.get("eval_every") is not None:
        eval_cfg = OmegaConf.merge(
            cfg.data, {"num_clips": int(cfg.train.get("eval_episodes", 128))}
        )
        eval_dataset = build_dataset(eval_cfg, seed_offset=17)  # pyright: ignore[reportArgumentType]
    model = build_model(cfg.model)
    assert isinstance(model, SCJepa | StateJepa)
    train_config = TrainConfig(
        steps=cfg.train.steps,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        grad_clip=cfg.train.grad_clip,
        lambda_reg=cfg.train.lambda_reg,
        sparsity_enabled=cfg.train.sparsity_enabled,
        sparsity_tau=cfg.train.sparsity_tau,
        sparsity_step_size=cfg.train.sparsity_step_size,
        sparsity_lambda_init=cfg.train.sparsity_lambda_init,
        sparsity_momentum=cfg.train.sparsity_momentum,
        lambda_logit=cfg.train.get("lambda_logit", 0.0),
        regularizer=cfg.train.regularizer,
        num_projections=cfg.train.num_projections,
        seed=cfg.train.seed,
        device=cfg.train.device,
        input_key=cfg.train.input_key,
        context_len=cfg.train.get("context_len", None),
        rollout_horizon=cfg.train.get("rollout_horizon", None),
        eval_every=cfg.train.get("eval_every", None),
        log_every=cfg.train.log_every,
        checkpoint_every=cfg.train.checkpoint_every,
        out_dir=str(out_dir),
    )
    experiment = HydraConfig.get().runtime.choices.get("experiment") or cfg.data.name
    phase = "sparse" if cfg.train.sparsity_enabled else "fc-calibration"
    run_name = f"{experiment}-{phase}-seed{cfg.train.seed}"
    logger: MetricLogger = (
        WandbLogger(project=cfg.wandb.project, mode=cfg.wandb.mode, config=resolved, name=run_name)
        if cfg.wandb.enabled
        else NoopLogger()
    )
    final = Trainer(model, dataset, train_config, logger, eval_dataset=eval_dataset).train()
    print(
        f"done at step {train_config.steps}: " + ", ".join(f"{k}={v:.4g}" for k, v in final.items())
    )


if __name__ == "__main__":
    main()
