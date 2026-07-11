"""Config → model/dataset builders shared by scripts/train.py and the eval script.

One place to translate Hydra configs into typed constructor calls, so training
and evaluation can never drift apart on how a model is rebuilt from a run's
``resolved_config.yaml``.
"""

from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import Dataset

from scjepa.data import BounceDataset, RandomClipDataset
from scjepa.models.jepa import build_scjepa
from scjepa.models.state_jepa import build_state_jepa


def build_model(model_cfg: DictConfig) -> nn.Module:
    """Build SCJepa (``type: vision``) or StateJepa (``type: states``)."""
    if model_cfg.type == "vision":
        return build_scjepa(
            resolution=model_cfg.resolution,
            num_slots=model_cfg.num_slots,
            slot_size=model_cfg.slot_size,
            slot_mlp_size=model_cfg.slot_mlp_size,
            num_iterations=model_cfg.num_iterations,
            enc_channels=tuple(model_cfg.enc_channels),
            enc_out_channels=model_cfg.enc_out_channels,
            pooling_heads=model_cfg.pooling_heads,
            pooling_type=str(model_cfg.get("pooling_type", "cross_slot")),
            max_history=model_cfg.max_history,
            spartan_layers=model_cfg.spartan_layers,
            spartan_embed_dim=model_cfg.spartan_embed_dim,
            spartan_mlp_hidden=model_cfg.spartan_mlp_hidden,
            spartan_mlp_layers=model_cfg.spartan_mlp_layers,
            spartan_temperature=model_cfg.spartan_temperature,
            aux_dim=model_cfg.aux_dim,
        )
    if model_cfg.type == "states":
        return build_state_jepa(
            state_dim=model_cfg.state_dim,
            slot_size=model_cfg.slot_size,
            pooling_heads=model_cfg.pooling_heads,
            pooling_type=str(model_cfg.get("pooling_type", "cross_slot")),
            max_history=model_cfg.max_history,
            spartan_layers=model_cfg.spartan_layers,
            spartan_embed_dim=model_cfg.spartan_embed_dim,
            spartan_mlp_hidden=model_cfg.spartan_mlp_hidden,
            spartan_mlp_layers=model_cfg.spartan_mlp_layers,
            spartan_temperature=model_cfg.spartan_temperature,
            aux_dim=model_cfg.aux_dim,
        )
    raise ValueError(f"unknown model.type {model_cfg.type!r} (vision | states)")


def build_dataset(data_cfg: DictConfig, seed_offset: int = 0) -> Dataset[dict[str, Tensor]]:
    """Build the dataset named in the config; ``seed_offset`` gives eval splits."""
    if data_cfg.name == "bounce":
        return BounceDataset(
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
            render=bool(data_cfg.get("render", True)),
            cache=bool(data_cfg.get("cache", False)),
        )
    if data_cfg.name == "synthetic_smoke":
        return RandomClipDataset(
            num_clips=data_cfg.num_clips,
            clip_len=data_cfg.clip_len,
            seed=data_cfg.seed + seed_offset,
        )
    raise ValueError(f"dataset {data_cfg.name!r} not implemented yet (CLEVRER/Push-T pending)")


__all__ = ["build_dataset", "build_model"]
