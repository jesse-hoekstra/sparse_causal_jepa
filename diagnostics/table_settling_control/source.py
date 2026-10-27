"""Authenticate completed curriculum, cache binding and evaluation-only physical labels."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_table_audit import STATE_TOLERANCE
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import frozen_hashes, sha
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA


@dataclass(frozen=True)
class Inputs:
    """Immutable inputs; physical labels are exposed only to episode evaluation."""

    bundle: SourceBundle
    model: SCJEPA
    table: Tensor
    latent_optimizer: dict[str, Any]
    group_indices: Tensor
    theta_evaluation_only: Tensor
    curriculum: dict[str, Any]
    file_hashes: dict[str, str]


def authenticated_directory(root: Path, required: set[str]) -> dict[str, Any]:
    """Check all completion hashes and confine authenticated artifacts to their root."""
    completion = cast(dict[str, Any], read_json(root / "completed.json"))
    if not required.issubset(completion["hashes"]):
        raise ValueError("Completion omits required source artifacts")
    for name, expected in completion["hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    for name in ("protocol.json", "results.json"):
        if read_json(root / name)["identity"] != completion["identity"]:
            raise ValueError("Source completion identity mismatch")
    return completion


def model_fingerprint(model: SCJEPA) -> str:
    """Hash all four module states together with non-tensor rollout flags."""
    return canonical_fingerprint(
        {
            "modules": {
                name: module_fingerprint(getattr(model, name))
                for name in ("online_encoder", "target_encoder", "parameter_encoder", "predictor")
            },
            "flags": model.get_extra_state(),
        }
    )


def load_inputs(
    curriculum_root: Path,
    audit_root: Path,
    *,
    verification: bool = False,
    groups: int = 256,
    selection_seed: int = 2026102703,
) -> Inputs:
    """Accept a teacher only after its full curriculum and exact cache audit finished."""
    completed = authenticated_directory(
        curriculum_root, {"protocol.json", "results.json", "teacher_final.pt", "washout_final.pt"}
    )
    audited = authenticated_directory(
        audit_root, {"protocol.json", "results.json", "audit_inputs.pt"}
    )
    curriculum = cast(dict[str, Any], read_json(curriculum_root / "protocol.json"))
    config = curriculum["config"]
    expected_budgets = (2, 2, 2) if verification else (30000, 10000, 10000)
    if (
        bool(config["verification"]) != verification
        or tuple(config[p + "_steps"] for p in ("teacher", "amortization", "washout"))
        != expected_budgets
        or read_json(curriculum_root / "results.json")["global_step"] != sum(expected_budgets)
    ):
        raise ValueError("Require the completed fixed-budget curriculum")
    source = Path(config["source_dir"])
    bundle = load_primary_source(
        source,
        config["seed"],
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=Path(config["training_source_root"])
        if config["training_source_root"]
        else None,
    )
    if any(curriculum["source_hashes"].get(name) != value for name, value in bundle.hashes.items()):
        raise ValueError("Curriculum and primary source hashes differ")
    if not 4 <= groups <= bundle.train_groups:
        raise ValueError("Invalid training-group subset size")
    # Select IDs before opening any evaluation-only theta payload.
    group_indices = torch.randperm(
        bundle.train_groups, generator=torch.Generator().manual_seed(selection_seed)
    )[:groups]
    audit_protocol = cast(dict[str, Any], read_json(audit_root / "protocol.json"))
    if (
        canonical_fingerprint({k: v for k, v in audit_protocol.items() if k != "identity"})
        != audited["identity"]
        or any(
            audit_protocol["source_hashes"].get(name) != value
            for name, value in bundle.hashes.items()
        )
        or audit_protocol["training_dataset_fingerprint"]
        != bundle.evaluation["metadata"]["training_dataset_fingerprint"]
    ):
        raise ValueError("Code audit does not authenticate this exact training cache")
    for name, expected in audit_protocol["source_hashes"].items():
        if name not in bundle.hashes:
            initialization = bundle.protocol["initializations"][str(bundle.seed)]
            if name != initialization.get("path") or expected != initialization["sha256"]:
                raise ValueError("Audit has an unrecognized extra source artifact")
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != expected:
            raise ValueError("Audit source artifact changed")
    audit_result = cast(dict[str, Any], read_json(audit_root / "results.json"))
    receipt = audit_result["reconstruction"]
    if (
        receipt["groups_verified"] != bundle.train_groups
        or receipt["dataset_fingerprint"] != audit_protocol["training_dataset_fingerprint"]
        or not receipt["all_actions_equal"]
        or not receipt["all_parameter_payloads_reproduce_observations"]
        or receipt["state_tolerance"] != STATE_TOLERANCE
        or any(value > STATE_TOLERANCE for value in receipt["state_max_absolute_error"].values())
    ):
        raise ValueError("Code audit did not fully verify parameter/cache binding")
    labels = cast(
        dict[str, Any],
        torch.load(audit_root / "audit_inputs.pt", map_location="cpu", weights_only=False),
    )
    theta = cast(Tensor, labels["theta_evaluation_only"])
    if (
        labels["identity"] != audited["identity"]
        or theta.shape != (bundle.train_groups, 6)
        or not bool(torch.isfinite(theta).all())
        or tensor_fingerprint(theta) != receipt["theta_tensor_fingerprint"]
    ):
        raise ValueError("Authenticated evaluation labels changed")
    model, discarded = build_source_model(bundle, torch.device("cpu"))
    del discarded
    states, flags = frozen_hashes(model), model.get_extra_state()
    checkpoint = cast(
        dict[str, Any],
        torch.load(curriculum_root / "teacher_final.pt", map_location="cpu", weights_only=False),
    )
    if (
        checkpoint["identity"] != completed["identity"]
        or checkpoint["phase"] != "teacher"
        or checkpoint["phase_step"] != expected_budgets[0]
        or checkpoint["global_step"] != expected_budgets[0]
    ):
        raise ValueError("Teacher checkpoint identity or counters differ")
    model.load_state_dict(checkpoint["model"], strict=True)
    if frozen_hashes(model) != states or model.get_extra_state() != flags:
        raise ValueError("Teacher changed the frozen state maps or rollout")
    expected_encoder = curriculum["initial_encoder_hash"]
    if module_fingerprint(model.parameter_encoder) != expected_encoder:
        raise ValueError("Teacher changed its frozen parameter encoder")
    table = cast(Tensor, checkpoint["table"]["weight"])
    if table.shape != theta.shape or not bool(torch.isfinite(table).all()):
        raise ValueError("Invalid teacher table")
    latent_optimizer = copy.deepcopy(checkpoint["latent_optimizer"])
    groups_state = latent_optimizer["param_groups"]
    if (
        len(groups_state) != 1
        or groups_state[0]["params"] != [0]
        or groups_state[0]["lr"] != 0.01
        or tuple(groups_state[0]["betas"]) != (0.9, 0.999)
        or groups_state[0]["eps"] != 1e-8
        or groups_state[0].get("maximize", False)
        or set(latent_optimizer["state"]) != {0}
    ):
        raise ValueError("Teacher SparseAdam semantics changed")
    moments = latent_optimizer["state"][0]
    if moments["step"] != expected_budgets[0]:
        raise ValueError("Teacher SparseAdam step differs from teacher budget")
    for name in ("exp_avg", "exp_avg_sq"):
        if moments[name].shape != table.shape or not bool(torch.isfinite(moments[name]).all()):
            raise ValueError("Teacher SparseAdam moments are invalid")
        moments[name] = moments[name][group_indices].clone()
    hashes: dict[str, str] = {}
    for root, records in (
        (source, curriculum["source_hashes"]),
        (source, audit_protocol["source_hashes"]),
        (curriculum_root, completed["hashes"]),
        (audit_root, audited["hashes"]),
    ):
        for name, expected in records.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != expected:
                raise ValueError("Authenticated source changed")
            hashes[str(path)] = expected
    for root in (curriculum_root, audit_root):
        hashes[str((root / "completed.json").resolve())] = sha(root / "completed.json")
    return Inputs(
        bundle,
        model.eval().requires_grad_(False),
        table,
        latent_optimizer,
        group_indices,
        theta,
        curriculum,
        hashes,
    )
