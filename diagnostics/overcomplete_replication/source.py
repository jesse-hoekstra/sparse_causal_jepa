"""Authenticate the original source while generating genuinely fresh second-seed modules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import OmegaConf

from diagnostics.encoder_gain_acquisition.model import ORIGINAL_ENCODER_SHA256, apply_encoder_gains
from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import SourceBundle, load_primary_source
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.overcomplete_acquisition.model import initialize_overcomplete
from scjepa.models.model import SCJEPA
from scjepa.training.factory import build_model
from scjepa.training.rng import preserve_rng

Arm = Literal["gain6", "gain12"]

SOURCE_SEED = 202610261301
TRAINING_SEED = 202610262301
RAW_REPLICATION_ENCODER_SHA256 = "0c1d4d7dd25b250afb2efb13808d9d6f597fe0ee301583bfbebf62313a378fcc"


@dataclass
class Inputs:
    """Authenticated unchanged data/state maps and fresh twelve-code learned modules."""

    bundle: SourceBundle
    model: SCJEPA
    source_hashes: dict[str, str]
    source_initialization: dict[str, Any]
    replication_initialization: dict[str, Any]


def seeded_original_model(model_config: dict[str, Any], seed: int) -> SCJEPA:
    """Replay full factory construction, including pre-E state-encoder random draws."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.default_generator.manual_seed(seed)
        return build_model(OmegaConf.create(model_config))


def load_inputs(
    source: Path,
    source_seed: int,
    training_seed: int,
    *,
    arm: Arm = "gain12",
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Validate archived seed policy before replacing E/P with independent fresh modules."""
    if arm not in ("gain6", "gain12"):
        raise ValueError("Choose declared gain6 or gain12 reference arm")
    if source_seed == training_seed:
        raise ValueError("Replication requires a distinct training seed")
    if not verification and (source_seed, training_seed) != (SOURCE_SEED, TRAINING_SEED):
        raise ValueError("Only the declared source and second training seeds are authorized")
    bundle = load_primary_source(
        source,
        source_seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    unused_table, initialization, hashes = initialize_control(model, bundle, "fresh_zero")
    del unused_table
    original_config = cast(dict[str, Any], bundle.protocol["model_config"])
    replay = seeded_original_model(original_config, source_seed)
    replay_hashes = {
        name: module_fingerprint(getattr(replay, name))
        for name in ("parameter_encoder", "predictor")
    }
    if any(
        value != module_fingerprint(getattr(model, name)) for name, value in replay_hashes.items()
    ):
        raise ValueError("Full-config source-seed initialization does not reproduce archived E/P")
    if not verification and replay_hashes["parameter_encoder"] != ORIGINAL_ENCODER_SHA256:
        raise ValueError("Original archived encoder differs from the audited initializer")
    state_hashes = {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }
    flags = model.get_extra_state()
    fresh = seeded_original_model(original_config, training_seed)
    raw_encoder_hash = module_fingerprint(fresh.parameter_encoder)
    if raw_encoder_hash == replay_hashes["parameter_encoder"]:
        raise ValueError("Replication accidentally reuses the original encoder initialization")
    if not verification and raw_encoder_hash != RAW_REPLICATION_ENCODER_SHA256:
        raise ValueError("Second-seed raw encoder differs from the predeclared CPU audit")
    model.parameter_encoder.load_state_dict(fresh.parameter_encoder.state_dict(), strict=True)
    source_predictor_count = sum(p.numel() for p in model.predictor.parameters())
    with preserve_rng():
        mlp = initialize_mlp(model, training_seed)
    gains = apply_encoder_gains(model)
    expanded = initialize_overcomplete(model, training_seed) if arm == "gain12" else None
    if flags != model.get_extra_state() or any(
        module_fingerprint(getattr(model, name)) != value for name, value in state_hashes.items()
    ):
        raise RuntimeError("Fresh replication changed source state maps or rollout settings")
    receipt: dict[str, Any] = {
        "arm": arm,
        "code_dimension": 12 if arm == "gain12" else 6,
        "source_seed": source_seed,
        "training_seed": training_seed,
        "source_seed_factory_replay_hashes": replay_hashes,
        "raw_encoder_sha256": raw_encoder_hash,
        "source_predictor_parameter_count": source_predictor_count,
        "mlp_initialization": mlp,
        "encoder_gain_initialization": gains,
        "overcomplete_initialization": expanded,
        "frozen_state_hashes": state_hashes,
        "model_config": original_config,
        "policy": (
            "Validate full original-config factory replay at source_seed against archived fresh "
            "E/P, then repeat full CPUfloat32 factory construction at training_seed. Copy only "
            "fresh E into authenticated frozen-state model; seed fresh MLP P independently at "
            "training_seed, apply unchanged gains, plus twelve-code expansion at seed+1/+2 "
            "only for gain12. "
            "No trained E/P or source Adam moments inherited. Caller CPU and initialized "
            "current CUDA RNG preserved."
        ),
    }
    return Inputs(bundle, model, hashes, initialization, receipt)
