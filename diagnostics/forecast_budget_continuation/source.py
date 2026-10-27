"""Restore authenticated fixed30k E/P, Adam history and RNG for paired budget extensions."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.encoder_gain_acquisition.run import code_identity as gain_code_identity
from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_forecast_only_acquisition.run import code_identity as baseline_code_identity
from diagnostics.mlp_forecast_only_acquisition.run import fixed_hashes
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA

SourceArm = Literal["baseline", "gain"]
PINNED = {
    "baseline": {
        "last.pt": "2ec3d70479c6c9d00751a716c907320cc3f9f296ced7f2b2522e3bff96fa1115",
        "protocol.json": "2cd7f3f8030ba8d675725c311c21b847dd724067bbc5d9ac43454959b1a17cc0",
        "completed.json": "be8ab6d13bc98144f0131f2bb64f7f1790f6a7b678034824e739a3f9c8be8346",
    },
    "gain": {
        "last.pt": "f33d07ee8db735177290b151aab4973bb5de92ed94f64057567fbf33f6ecf1c0",
        "protocol.json": "bd168e8997bb7bea0c02f80986eefd742d0609b3c56243759e6f969244eb93aa",
        "completed.json": "741fdb36956b3d22b4e01534016eb0b0958b8a266cdd2a48c44faca8a070d1f9",
    },
}


@dataclass(frozen=True)
class Inputs:
    """Exact learned endpoint and optimizer; caches still exclude training labels."""

    bundle: SourceBundle
    model: SCJEPA
    optimizer: torch.optim.Adam
    rng: dict[str, Any]
    hashes: dict[str, str]
    provenance: dict[str, Any]


def load_inputs(
    source: Path,
    source_arm: SourceArm,
    *,
    device: torch.device | None = None,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Validate ancestry and restore moments/steps in the original parameter order."""
    if source_arm not in PINNED:
        raise ValueError("Choose the baseline or gain source arm")
    root = source.resolve()
    if not verification and any(
        sha(root / name) != value for name, value in PINNED[source_arm].items()
    ):
        raise ValueError("Source differs from declared completed30k arm")
    completion = authenticated_directory(
        root, {"protocol.json", "results.json", "last.pt", "initial_modules.pt"}
    )
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    config = protocol["config"]
    evidence_keys = [
        "settings",
        "source_hashes",
        "source_initialization",
        "initial_encoder_sha256",
        "mlp_initialization",
        "source_predictor_parameter_count",
        "frozen_hashes",
        "code_identity",
        "torch",
    ]
    if source_arm == "gain":
        if "encoder_gain_initialization" not in protocol:
            raise ValueError("Gain arm requires gain initialization ancestry")
        evidence_keys.append("encoder_gain_initialization")
    elif "encoder_gain_initialization" in protocol:
        raise ValueError("Baseline arm cannot load gain initialization ancestry")
    settings = dict(config)
    for key in ("source_dir", "output_dir", "training_source_root", "wandb_mode", "wandb_entity"):
        settings.pop(key)
    if (
        canonical_fingerprint({key: protocol[key] for key in evidence_keys}) != protocol["identity"]
        or settings != protocol["settings"]
        or config["verification"] != verification
        or protocol["code_identity"]
        != (gain_code_identity() if source_arm == "gain" else baseline_code_identity())
        or protocol["training_parameter_labels_absent"] is not True
        or (verification and not 1 <= config["steps"] <= 3)
        or (
            not verification
            and (config["steps"], config["seed"], config["groups"], config["bootstrap_samples"])
            != (30000, 202610261301, 64, 1000)
        )
    ):
        raise ValueError("Source protocol, objective implementation or fixed budget differs")
    primary = Path(config["source_dir"])
    archive = training_source_root or (
        Path(config["training_source_root"]) if config["training_source_root"] else None
    )
    bundle = load_primary_source(
        primary,
        config["seed"],
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=archive,
    )
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    unused_table, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del unused_table
    source_predictor_count = sum(parameter.numel() for parameter in model.predictor.parameters())
    if (
        source_hashes != protocol["source_hashes"]
        or initialization != protocol["source_initialization"]
        or source_predictor_count != protocol["source_predictor_parameter_count"]
    ):
        raise ValueError("Original source archive or initialization binding differs")
    mlp_initialization = initialize_mlp(model, config["seed"])
    if mlp_initialization != protocol["mlp_initialization"]:
        raise ValueError("Original fresh MLP policy differs")
    if (
        source_arm == "gain"
        and apply_encoder_gains(model) != protocol["encoder_gain_initialization"]
    ):
        raise ValueError("Fixed gain initialization differs")
    initial = cast(
        dict[str, Any],
        torch.load(root / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    if (
        initial["identity"] != protocol["identity"]
        or initial["initialization"] != mlp_initialization
    ):
        raise ValueError("Saved initialization identity differs")
    if (
        source_arm == "gain"
        and initial["encoder_gain_initialization"] != protocol["encoder_gain_initialization"]
    ):
        raise ValueError("Saved gain receipt differs")
    for name, expected in (
        ("parameter_encoder", protocol["initial_encoder_sha256"]),
        ("predictor", mlp_initialization["module_sha256"]),
    ):
        original = getattr(model, name)
        candidate = copy.deepcopy(original)
        candidate.load_state_dict(initial[name], strict=True)
        if module_fingerprint(original) != expected or module_fingerprint(candidate) != expected:
            raise ValueError("Saved source fresh module tensors differ")
    frozen, flags = fixed_hashes(model), model.get_extra_state()
    if frozen != protocol["frozen_hashes"]:
        raise ValueError("Source frozen state chart differs")
    checkpoint = cast(
        dict[str, Any], torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    )
    result = cast(dict[str, Any], read_json(root / "results.json"))
    step = int(config["steps"])
    if (
        checkpoint["identity"] != protocol["identity"]
        or checkpoint["step"] != step
        or result["step"] != step
    ):
        raise ValueError("Source is not its completed fixed endpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
        raise ValueError("Saved state maps or rollout flags changed")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Source model contains nonfinite parameters")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    model.to(device or torch.device("cpu"))
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named]
    state = cast(dict[str, Any], checkpoint["optimizer"])
    groups = cast(list[dict[str, Any]], state["param_groups"])
    if len(groups) != 1 or groups[0]["params"] != list(range(len(parameters))):
        raise ValueError("Source Adam parameter order differs from E/P traversal")
    group = groups[0]
    for key, expected in {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
    }.items():
        if group[key] != expected:
            raise ValueError(f"Source Adam {key} differs")
    if set(state["state"]) != set(group["params"]):
        raise ValueError("Source Adam lacks a parameter history")
    for index, parameter in enumerate(parameters):
        saved = state["state"][index]
        if set(saved) != {"step", "exp_avg", "exp_avg_sq"} or float(saved["step"]) != step:
            raise ValueError("Source Adam step or state fields differ")
        for key in ("exp_avg", "exp_avg_sq"):
            if saved[key].shape != parameter.shape or not bool(torch.isfinite(saved[key]).all()):
                raise ValueError("Source Adam moment shape/value differs")
        if bool((saved["exp_avg_sq"] < 0).any()):
            raise ValueError("Source Adam second moment is negative")
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    optimizer.load_state_dict(state)
    hashes = {
        **{str((primary / name).resolve()): value for name, value in source_hashes.items()},
        **{str(root / name): value for name, value in completion["hashes"].items()},
        str(root / "completed.json"): sha(root / "completed.json"),
    }
    provenance = {
        "source_arm": source_arm,
        "source_seed": config["seed"],
        "source_directory": str(root),
        "source_identity": protocol["identity"],
        "source_step": step,
        "source_checkpoint_sha256": sha(root / "last.pt"),
        "source_encoder_sha256": module_fingerprint(model.parameter_encoder),
        "source_predictor_sha256": module_fingerprint(model.predictor),
        "frozen_hashes": frozen,
        "rollout_flags": flags,
        "optimizer": "Exact source Adam moments, per-parameter steps and groups restored; no reset",
        "optimizer_parameter_names": [name for name, _ in named],
        "optimizer_step": step,
        "source_code_identity": protocol["code_identity"],
        "source_rng_has_cuda": checkpoint["rng"]["cuda"] is not None,
    }
    inputs = Inputs(bundle, model, optimizer, checkpoint["rng"], hashes, provenance)
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Keep original source files, frozen state tensors and rollout flags immutable."""
    if any(sha(Path(name)) != expected for name, expected in inputs.hashes.items()):
        raise ValueError("Immutable continuation source changed")
    if (
        fixed_hashes(inputs.model) != inputs.provenance["frozen_hashes"]
        or inputs.model.get_extra_state() != inputs.provenance["rollout_flags"]
    ):
        raise ValueError("Frozen states or rollout flags changed")
