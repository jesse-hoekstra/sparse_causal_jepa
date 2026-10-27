"""Bind both diagnostic continuation arms to one immutable learned70k endpoint."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.compressed_teacher_curriculum.run import code_identity as curriculum_code_identity
from diagnostics.compressed_teacher_curriculum.source import Inputs as CurriculumInputs
from diagnostics.compressed_teacher_curriculum.source import load_inputs as load_curriculum_inputs
from diagnostics.compressed_teacher_curriculum.source import (
    verify_sources as verify_curriculum_sources,
)
from diagnostics.encoder_gradient_control.source import SourceBundle
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.multiteacher_curriculum.forward import Heads
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA

SELECTED_ENDPOINT_HASHES = {
    "last.pt": "bb5e57a23a7a4a809a14fe093f4acbf8d6b8235b303a1669a94225f20bf487b0",
    "protocol.json": "42de7e5439b262c49d78a8aeac6df35d7550bd611ee45a3c7198d84c58493e31",
    "completed.json": "495fd5d164f78be8ebcd3b59c7f19d07ba79921deb0e414506960fdbcae5cbb6",
}


@dataclass(frozen=True)
class Inputs:
    """Only original label-free caches and endpoint E/P enter continuation training."""

    bundle: SourceBundle
    model: SCJEPA
    heads: Heads
    hashes: dict[str, str]
    provenance: dict[str, Any]
    reference: CurriculumInputs


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Keep learned state coordinates fixed while allowing both E and P to update."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }


def load_inputs(source_dir: Path, *, verification: bool = False) -> Inputs:
    """Authenticate completed70k curriculum, copy its exact E/P, discard optimizer state."""
    root = source_dir.resolve()
    if not verification and any(
        sha(root / name) != expected for name, expected in SELECTED_ENDPOINT_HASHES.items()
    ):
        raise ValueError("Continuation source differs from selected fixed70k endpoint")
    completion = authenticated_directory(
        root,
        {
            "protocol.json",
            "results.json",
            "last.pt",
            "washout_final.pt",
            "initial_modules.pt",
            "target_manifest.json",
            "teacher_targets.pt",
            "final_test.json",
        },
    )
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    config = protocol["config"]
    evidence_keys = (
        "settings",
        "source_hashes",
        "compression_provenance",
        "source_initialization",
        "initial_encoder_sha256",
        "mlp_initialization",
        "state_hashes",
        "code_identity",
        "torch",
    )
    settings = dict(config)
    for name in ("compression_source_dir", "output_dir", "wandb_mode", "wandb_entity"):
        settings.pop(name)
    budgets = tuple(config[name] for name in ("distill_steps", "predictor_steps", "washout_steps"))
    if (
        canonical_fingerprint({key: protocol[key] for key in evidence_keys}) != protocol["identity"]
        or settings != protocol["settings"]
        or protocol["code_identity"] != curriculum_code_identity()
        or config["verification"] != verification
        or protocol["training_parameter_labels_absent"] is not True
        or (verification and any(not 1 <= budget <= 3 for budget in budgets))
        or (
            not verification
            and (budgets != (30000, 30000, 10000) or config["seed"] != 202610261301)
        )
    ):
        raise ValueError("Curriculum protocol, implementation or fixed budgets differ")
    reference = load_curriculum_inputs(
        Path(config["compression_source_dir"]), verification=verification
    )
    if (
        reference.hashes != protocol["source_hashes"]
        or reference.provenance != protocol["compression_provenance"]
        or reference.initialization != protocol["source_initialization"]
    ):
        raise ValueError("Curriculum ancestor binding differs")
    model = copy.deepcopy(reference.model)
    if initialize_mlp(model, config["seed"]) != protocol["mlp_initialization"]:
        raise ValueError("Original predictor initialization policy differs")
    initial = cast(
        dict[str, Any],
        torch.load(root / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    if initial["identity"] != protocol["identity"]:
        raise ValueError("Curriculum initial snapshot identity differs")
    for name, key, expected in (
        ("parameter_encoder", "encoder", protocol["initial_encoder_sha256"]),
        ("predictor", "predictor", protocol["mlp_initialization"]["module_sha256"]),
    ):
        module = getattr(model, name)
        if module_fingerprint(module) != expected:
            raise ValueError("Curriculum fresh initialization differs")
        module.load_state_dict(initial[key], strict=True)
        if module_fingerprint(module) != expected:
            raise ValueError("Curriculum saved initialization differs")
    flags = model.get_extra_state()
    checkpoint = cast(
        dict[str, Any], torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    )
    result = cast(dict[str, Any], read_json(root / "results.json"))
    manifest = cast(dict[str, Any], read_json(root / "target_manifest.json"))
    source_step = sum(budgets)
    if (
        checkpoint["identity"] != protocol["identity"]
        or checkpoint["step"] != source_step
        or result["step"] != source_step
        or checkpoint["stage"] != "washout"
        or checkpoint["heads"] is not None
        or checkpoint["target_manifest_sha256"] != sha(root / "target_manifest.json")
        or manifest["identity"] != protocol["identity"]
        or manifest["sha256"] != sha(root / "teacher_targets.pt")
    ):
        raise ValueError("Continuation requires the exact completed washout endpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    if fixed_hashes(model) != protocol["state_hashes"] or model.get_extra_state() != flags:
        raise ValueError("Endpoint frozen state maps or rollout flags changed")
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise ValueError("Endpoint model contains nonfinite values")
    heads = copy.deepcopy(reference.heads).requires_grad_(False).eval()
    provenance: dict[str, Any] = {
        "source_directory": str(root),
        "source_identity": protocol["identity"],
        "source_step": source_step,
        "source_checkpoint_sha256": sha(root / "last.pt"),
        "source_encoder_sha256": module_fingerprint(model.parameter_encoder),
        "source_predictor_sha256": module_fingerprint(model.predictor),
        "state_hashes": fixed_hashes(model),
        "rollout_flags": flags,
        "heads_sha256": module_fingerprint(heads),
        "compression_provenance": reference.provenance,
        "source_optimizer_reused": False,
        "training_boundary": (
            "Only online/target/actions caches; no teacher targets or heads in loss"
        ),
        "selection_scope": (
            "Adaptive continuation from selected70k label-free learned endpoint after its test was "
            "observed. Neither new endpoint is a blind independent replication. Original directq "
            "and expanded fixed-head readout results stay separate; predictive use is separate."
        ),
    }
    hashes = {
        **reference.hashes,
        **{str(root / name): value for name, value in completion["hashes"].items()},
        str(root / "completed.json"): sha(root / "completed.json"),
    }
    inputs = Inputs(reference.bundle, model, heads, hashes, provenance, reference)
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Protect every ancestor and fixed evaluation/state map, while E/P may learn."""
    verify_curriculum_sources(inputs.reference)
    if any(sha(Path(name)) != value for name, value in inputs.hashes.items()):
        raise ValueError("Immutable continuation source changed")
    if (
        fixed_hashes(inputs.model) != inputs.provenance["state_hashes"]
        or inputs.model.get_extra_state() != inputs.provenance["rollout_flags"]
        or module_fingerprint(inputs.heads) != inputs.provenance["heads_sha256"]
        or any(parameter.requires_grad for parameter in inputs.heads.parameters())
    ):
        raise ValueError("Frozen continuation state maps, flags or readout heads changed")
