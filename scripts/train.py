"""Generic training entrypoint for configured data adapters; no scientific protocol is imposed."""

import json
from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scjepa.data import TensorEpisodes
from scjepa.data.bundle import load_dataset_bundle
from scjepa.data.factory import build_dgp_dataset
from scjepa.training import TrainConfig, Trainer, seed_everything
from scjepa.training.distributed import (
    any_rank,
    destroy_distributed,
    initialize_distributed,
    is_main_process,
)
from scjepa.training.factory import build_model
from scjepa.training.logging import WandbConfig, WandbLogger


def run_training(cfg: DictConfig) -> None:
    """Validate/resume training before starting one rank-zero tracking run."""
    if (cfg.data_path is None) == (cfg.data is None):
        raise ValueError("select exactly one data preset or data_path; see README.md")
    train_cfg = cast(
        TrainConfig,
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg.train)),
    )
    wandb_cfg = cast(
        WandbConfig,
        OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(WandbConfig), cfg.wandb)),
    )
    train_cfg.device = str(initialize_distributed(train_cfg.device))
    logger: WandbLogger | None = None
    exit_code = 1
    try:
        seed_everything(train_cfg.seed)
        model = build_model(cfg.model)
        if cfg.data is not None:
            data = build_dgp_dataset(cfg.data)
            cfg.data = OmegaConf.structured(data.config)
        else:
            path = Path(str(cfg.data_path))
            data = (
                load_dataset_bundle(path)
                if path.is_dir() or path.suffix == ".json"
                else TensorEpisodes.load(path)
            )
        model_config = cast(dict[str, object], OmegaConf.to_container(cfg.model, resolve=True))
        trainer = Trainer(model, train_cfg, dataset=data, model_config=model_config)
        if cfg.resume is not None:
            trainer.load_checkpoint(Path(str(cfg.resume)))
        setup_error: Exception | None = None
        if is_main_process():
            try:
                cfg.train = OmegaConf.structured(trainer.config)
                cfg.wandb = OmegaConf.structured(wandb_cfg)
                run_dir = Path(trainer.config.out_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                saved_tracking = trainer.run_metadata.get("wandb")
                if saved_tracking is not None and not isinstance(saved_tracking, dict):
                    raise ValueError("checkpoint W&B identity must be a mapping")
                logger = WandbLogger(
                    wandb_cfg,
                    resolved_config=cast(
                        dict[str, object], OmegaConf.to_container(cfg, resolve=True)
                    ),
                    dataset_identity=trainer.dataset_identity,
                    out_dir=run_dir,
                    resume_state=cast(dict[str, object] | None, saved_tracking),
                )
                trainer.logger = logger
                if logger.metadata:
                    trainer.run_metadata["wandb"] = logger.metadata
                OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
                metadata_path = run_dir / "run_metadata.json"
                temporary = metadata_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(trainer.run_metadata, indent=2) + "\n")
                temporary.replace(metadata_path)
            except Exception as error:
                setup_error = error
        if any_rank(setup_error is not None, trainer.device):
            raise RuntimeError("run logging initialization failed on rank zero") from setup_error
        trainer.train()
        exit_code = 0
    finally:
        try:
            if logger is not None:
                logger.finish(exit_code=exit_code)
        finally:
            destroy_distributed()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run configured training with W&B and terminal metrics."""
    run_training(cfg)


if __name__ == "__main__":
    main()
