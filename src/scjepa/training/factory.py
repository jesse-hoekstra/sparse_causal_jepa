"""Typed model configuration shared by training and future experiment integrations."""

from dataclasses import dataclass
from typing import cast

from omegaconf import DictConfig, OmegaConf

from scjepa.models import SCJEPA, ImageEncoder, ParameterEncoder, Spartan, VectorEncoder


@dataclass
class ModelConfig:
    """Architecture choices independent of datasets, losses, and evaluation protocols."""

    observation_kind: str = "vector"
    observation_dim: int = 8
    in_channels: int = 3
    encoder_hidden_dims: tuple[int, ...] = (64, 64)
    image_channels: tuple[int, ...] = (16, 32, 64)
    image_pool_size: int = 2
    state_dim: int = 6
    param_dim: int = 3
    action_dim: int = 0
    context_len: int = 4
    history_len: int = 2
    parameter_embed_dim: int = 64
    parameter_heads: int = 4
    max_context_len: int = 64
    spartan_layers: int = 2
    spartan_embed_dim: int = 64
    spartan_mlp_hidden: int = 64
    spartan_mlp_layers: int = 3
    spartan_temperature: float = 1.0
    spartan_dense: bool = False
    spartan_identity: bool = False
    ema_decay: float = 0.996


def build_model(config: ModelConfig | DictConfig) -> SCJEPA:
    """Construct the single model interface; unknown config keys are rejected."""
    cfg = (
        cast(
            ModelConfig,
            OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ModelConfig), config)),
        )
        if isinstance(config, DictConfig)
        else config
    )
    if cfg.observation_kind == "vector":
        encoder = VectorEncoder(cfg.observation_dim, cfg.state_dim, cfg.encoder_hidden_dims)
    elif cfg.observation_kind == "image":
        encoder = ImageEncoder(
            cfg.in_channels, cfg.state_dim, cfg.image_channels, cfg.image_pool_size
        )
    else:
        raise ValueError("observation_kind must be vector or image")
    parameter_encoder = ParameterEncoder(
        state_dim=cfg.state_dim,
        param_dim=cfg.param_dim,
        action_dim=cfg.action_dim,
        dim=cfg.parameter_embed_dim,
        num_heads=cfg.parameter_heads,
        max_history=cfg.max_context_len,
    )
    predictor = Spartan(
        state_dim=cfg.state_dim,
        param_dim=cfg.param_dim,
        action_dim=cfg.action_dim,
        history_len=cfg.history_len,
        num_layers=cfg.spartan_layers,
        embed_dim=cfg.spartan_embed_dim,
        mlp_hidden_size=cfg.spartan_mlp_hidden,
        mlp_num_layers=cfg.spartan_mlp_layers,
        temperature=cfg.spartan_temperature,
        dense=cfg.spartan_dense,
        identity=cfg.spartan_identity,
    )
    return SCJEPA(encoder, parameter_encoder, predictor, cfg.context_len, cfg.ema_decay)


__all__ = ["ModelConfig", "build_model"]
