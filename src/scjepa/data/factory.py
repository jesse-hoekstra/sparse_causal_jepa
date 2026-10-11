"""Small registry for scientific processes, independent of model and trainer code."""

from collections.abc import Callable
from typing import cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor

from scjepa.data.datasets import GeneratedEpisodes
from scjepa.data.types import DGP, DGPConfig

_REGISTRY: dict[str, tuple[Callable[[DGPConfig], DGP], type[DGPConfig]]] = {}


def register_dgp(
    name: str,
    constructor: Callable[[DGPConfig], DGP],
    config_type: type[DGPConfig] = DGPConfig,
) -> None:
    """Register a constructor and optional config subclass; reject replacement."""
    if not name or name in _REGISTRY:
        raise ValueError(f"DGP name is empty or already registered: {name!r}")
    _REGISTRY[name] = (constructor, config_type)


def _unlock_config(config: DictConfig | ListConfig) -> None:
    OmegaConf.set_readonly(config, False)
    children = config.values() if isinstance(config, DictConfig) else config
    for child in children:
        if isinstance(child, DictConfig | ListConfig):
            _unlock_config(child)


def resolve_dgp_config(config: DGPConfig | DictConfig) -> DGPConfig:
    """Resolve structured configuration, rejecting unrecognized settings."""
    if isinstance(config, DGPConfig):
        return config
    family = str(config.get("family", "controlled_vector"))
    config_type = _REGISTRY[family][1] if family in _REGISTRY else DGPConfig
    schema = OmegaConf.structured(config_type)
    _unlock_config(schema)
    return cast(
        DGPConfig,
        OmegaConf.to_object(OmegaConf.merge(schema, config)),
    )


def build_dgp_dataset(
    config: DGPConfig | DictConfig, *, observation_matrix: Tensor | None = None
) -> GeneratedEpisodes:
    """Construct lazy episodes, optionally restoring their authoritative realized map."""
    from scjepa.data.processes import register_builtin_processes

    register_builtin_processes()
    cfg = resolve_dgp_config(config)
    if cfg.family not in _REGISTRY:
        raise ValueError(f"Unknown DGP family {cfg.family!r}; available: {sorted(_REGISTRY)}")
    process = _REGISTRY[cfg.family][0](cfg)
    if observation_matrix is not None:
        process.restore_observation_matrix(observation_matrix)
    return GeneratedEpisodes(process)
