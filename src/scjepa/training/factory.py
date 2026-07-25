"""Config → model/dataset builders shared by scripts/train.py and the eval script.

One place to translate Hydra configs into typed constructor calls, so training
and evaluation can never drift apart on how a model is rebuilt from a run's
``resolved_config.yaml``.
"""

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data import BounceDataset
from scjepa.models.experiment1 import Experiment1Model, build_experiment1


def build_model(model_cfg: DictConfig) -> Experiment1Model:
    """Build the Experiment-1 model (dense/identity references via the flags)."""
    return build_experiment1(
        state_dim=model_cfg.state_dim,
        num_slots=model_cfg.num_slots,
        param_encoder_dim=model_cfg.param_encoder_dim,
        param_encoder_heads=model_cfg.param_encoder_heads,
        max_history=model_cfg.max_history,
        spartan_layers=model_cfg.spartan_layers,
        spartan_embed_dim=model_cfg.spartan_embed_dim,
        spartan_mlp_hidden=model_cfg.spartan_mlp_hidden,
        spartan_mlp_layers=model_cfg.spartan_mlp_layers,
        spartan_temperature=model_cfg.spartan_temperature,
        spartan_dense=bool(model_cfg.get("spartan_dense", False)),
        spartan_identity=bool(model_cfg.get("spartan_identity", False)),
    )


def build_dataset(data_cfg: DictConfig, seed_offset: int = 0) -> Dataset[dict[str, Tensor]]:
    """Build the bounce dataset; ``seed_offset`` selects held-out eval splits."""
    if data_cfg.name != "bounce":
        raise ValueError(f"unknown dataset {data_cfg.name!r} (bounce)")
    # preload applies ONLY to the train split (seed_offset 0): eval splits
    # use a shifted seed, and serving them from the train file would leak
    # training episodes into every eval metric.
    preload = data_cfg.get("preload") if seed_offset == 0 else None
    return BounceDataset(
        preload=preload,
        num_episodes=data_cfg.num_clips,
        clip_len=data_cfg.clip_len,
        num_balls=data_cfg.num_balls,
        resolution=data_cfg.resolution,
        radius=data_cfg.radius,
        mass_range=tuple(data_cfg.mass_range),
        mass_normal=(
            tuple(data_cfg.mass_normal) if data_cfg.get("mass_normal") is not None else None
        ),
        radius_from_mass=bool(data_cfg.get("radius_from_mass", False)),
        speed=data_cfg.speed,
        seed=data_cfg.seed + seed_offset,
        render=bool(data_cfg.get("render", False)),
        cache=bool(data_cfg.get("cache", False)),
    )


__all__ = ["build_dataset", "build_model"]
