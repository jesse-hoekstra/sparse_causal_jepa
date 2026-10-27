"""Authenticate complementary teacher encoders and construct train-only detached targets."""

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA

# The completed artifacts selected in the recorded adaptive compression audit.
# Paths are supplied explicitly; their immutable contents, not local/NFS spelling, bind them.
TEACHER_NAMES = ("primary_1_4", "fresh_2_5", "mlp_3_6")
TEACHER_HASHES = (
    {
        "teacher_final.pt": "5a70e08200d2573b4243a66c657f7731e5b8d6929df7e085b1006232b8b28274",
        "protocol.json": "293153377e1eae0b12233290a51bc42fcbd3e7fae9ffb6d325384791a67e1967",
        "completed.json": "9f9dcc58c79788e78519d4bed98608f907a9302df3e2742aef764df5ded275a4",
    },
    {
        "last.pt": "297395f92d8b71f64fd8fb16ebff10dccf5b8631130482322f8083170c03fa14",
        "protocol.json": "f4ea273d738d4b453c521ebe994d7679af7ceb73a579b0154cf52ecf154abd96",
        "completed.json": "63d1ddd627cdc81a8ab5b18ea32cbb59221fa1f420c5cfe8f3303ac210acc8f6",
    },
    {
        "last.pt": "2354f6aaebac38494bc65d10345dd09d3120756070691eb828a7715329f97eec",
        "protocol.json": "f5cc285a1fdf3100fcc7f01d9c6d08ee2c2c85bec7f0cdab7a4c1d9e6529d8cf",
        "completed.json": "0fd38f73159a8b72097f634dd50644c3048193ec3f046ed0355aa81b8fa03f19",
    },
)
SELECTION_SCOPE = (
    "Teacher selection was adaptive and informed by physical-parameter retention, including "
    "previous held-out/test reports. This is exploratory artifact repair, not blinded "
    "teacher selection or independent confirmation. All teacher optimization inputs and "
    "ancestry are label-free learned-state inputs; no assisted or physical-state teacher is used."
)


@dataclass(frozen=True)
class Inputs:
    """Original source and fresh student plus frozen, separately authenticated teachers."""

    bundle: SourceBundle
    model: SCJEPA
    teachers: tuple[nn.Module, ...]
    initialization: dict[str, Any]
    hashes: dict[str, str]
    teacher_provenance: dict[str, Any]


@dataclass(frozen=True)
class TeacherTargets:
    """CPU teacher codes, train-fitted transforms and detached nonself-context targets."""

    raw: Tensor
    mean: Tensor
    scale: Tensor
    targets: Tensor
    receipt: dict[str, Any]


def _historical_hashes(module: nn.Module) -> set[str]:
    """Accept both historical insertion order and current sorted tensor-name hashing."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {module_fingerprint(module), digest.hexdigest()}


def _module_state(payload: dict[str, Any], name: str) -> dict[str, Tensor]:
    prefix = name + "."
    return {key[len(prefix) :]: value for key, value in payload.items() if key.startswith(prefix)}


def _check_state_maps(payload: dict[str, Any], model: SCJEPA, protocol: dict[str, Any]) -> None:
    for name in ("online_encoder", "target_encoder"):
        expected = cast(nn.Module, getattr(model, name))
        actual = _module_state(payload, name)
        reference = expected.state_dict()
        if set(actual) != set(reference) or any(
            actual[key].dtype != value.dtype or not torch.equal(actual[key], value.cpu())
            for key, value in reference.items()
        ):
            raise ValueError("Teacher frozen learned-state tensors differ from original source")
        if protocol["frozen_hashes"][name] not in _historical_hashes(expected):
            raise ValueError("Teacher frozen-state metadata differ from original source")
    if payload.get("_extra_state") != model.get_extra_state():
        raise ValueError("Teacher context or attached-rollout settings differ from original source")


def _check_initialization(
    record: dict[str, Any], initialization: dict[str, Any], model: SCJEPA
) -> None:
    if (
        record["mode"] != "fresh_zero"
        or record["source_initialization"] != initialization["source_initialization"]
        or record["parameter_encoder_hash"] not in _historical_hashes(model.parameter_encoder)
        or record["predictor_hash"] not in _historical_hashes(model.predictor)
    ):
        raise ValueError("Teacher fresh initialization ancestry differs from archived original")


def _load_teacher(
    root: Path,
    index: int,
    bundle: SourceBundle,
    model: SCJEPA,
    primary_encoder: nn.Module,
    initialization: dict[str, Any],
    source_hashes: dict[str, str],
    *,
    verification: bool,
) -> tuple[nn.Module, dict[str, Any], dict[str, str]]:
    root = root.resolve()
    filename = "teacher_final.pt" if index == 0 else "last.pt"
    if not verification and any(
        sha(root / name) != expected for name, expected in TEACHER_HASHES[index].items()
    ):
        raise ValueError("Teacher differs from the preselected scientific artifact hashes")
    completion = authenticated_directory(root, {"protocol.json", "results.json", filename})
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    config = protocol["config"]
    payload = cast(
        dict[str, Any], torch.load(root / filename, map_location="cpu", weights_only=False)
    )
    expected_hashes = bundle.hashes if index == 0 else source_hashes
    if (
        payload["identity"] != completion["identity"]
        or config["seed"] != bundle.seed
        or config["verification"] != verification
        or protocol["source_hashes"] != expected_hashes
    ):
        raise ValueError("Teacher identity, seed or original source/cache ancestry differs")
    if protocol.get("diagnostic_physical_labels_in_training", False):
        raise ValueError("Supervised oracle teachers are prohibited")
    state = cast(dict[str, Any], payload["model"])
    _check_state_maps(state, model, protocol)
    encoder = copy.deepcopy(model.parameter_encoder)
    encoder.load_state_dict(_module_state(state, "parameter_encoder"), strict=True)
    if index < 2:
        budgets = tuple(
            int(config[name]) for name in ("teacher_steps", "amortization_steps", "washout_steps")
        )
        if (verification and not all(1 <= count <= 3 for count in budgets)) or (
            not verification and budgets != (30000, 10000, 10000)
        ):
            raise ValueError("Teacher curriculum budget differs from selected endpoint")
        expected_phase = "teacher" if index == 0 else "washout"
        expected_step = budgets[0] if index == 0 else budgets[2]
        global_step = budgets[0] if index == 0 else sum(budgets)
        if (
            protocol["training_labels_absent"] is not True
            or protocol["source_optimizer_retained"] is not False
            or protocol["source_step"] != bundle.source_step
            or config.get("teacher_objective_arm", "original") != "original"
            or payload["phase"] != expected_phase
            or payload["phase_step"] != expected_step
            or payload["global_step"] != global_step
        ):
            raise ValueError("Teacher endpoint is not the selected label-free original curriculum")
        objective = protocol["objective"]
        if any(
            objective[key] != expected
            for key, expected in {
                "parameter_regularizer": "sigreg",
                "parameter_regularizer_weight": 0.4,
                "parameter_consistency_weight": 1.0,
                "logit_weight": 0.0001,
                "state_sparsity_weight": 0.0,
                "action_sparsity_weight": 0.0,
                "parameter_sparsity_weight": 0.0,
            }.items()
        ):
            raise ValueError("Teacher original label-free objective differs")
        if index == 0:
            if config.get("initialization", "warm") != "warm" or (
                module_fingerprint(encoder) != module_fingerprint(primary_encoder)
                or protocol["initial_encoder_hash"] not in _historical_hashes(encoder)
            ):
                raise ValueError(
                    "Warm teacher encoder is not the unchanged original primary encoder"
                )
        else:
            if config["initialization"] != "fresh_zero":
                raise ValueError("Fresh group teacher must start from archived fresh modules")
            _check_initialization(protocol["initialization"], initialization, model)
    else:
        if (
            protocol["training_parameter_labels_absent"] is not True
            or payload["step"] != config["steps"]
            or (verification and not 1 <= config["steps"] <= 3)
            or (not verification and config["steps"] != 30000)
            or protocol["initial_encoder_sha256"] not in _historical_hashes(model.parameter_encoder)
            or protocol["objective"]
            != (
                "Mean two-step MSE + .4 per-view parameter SIGReg + within-group consistency1. "
                "Constant frozen-state SIGReg omitted, preserving its projection draw. "
                "No attention logits exist: absent logit regularizer is an explicit confound."
            )
        ):
            raise ValueError(
                "MLP teacher is not the selected fresh label-free acquisition endpoint"
            )
        _check_initialization(protocol["source_initialization"], initialization, model)
    provenance = {
        "role": TEACHER_NAMES[index],
        "identity": completion["identity"],
        "checkpoint": str(root / filename),
        "checkpoint_sha256": sha(root / filename),
        "protocol_sha256": sha(root / "protocol.json"),
        "completed_sha256": sha(root / "completed.json"),
        "encoder_sha256": module_fingerprint(encoder),
        "frozen_state_tensors_exactly_original": True,
        "source_hashes": protocol["source_hashes"],
        "checkpoint_code_identity": protocol["code_identity"],
        "used_modules": "parameter_encoder only; no teacher predictor or latent table",
        "label_free_ancestry_verified": True,
    }
    hashes = {str(root / name): value for name, value in completion["hashes"].items()}
    hashes[str(root / "completed.json")] = sha(root / "completed.json")
    return encoder.requires_grad_(False).eval(), provenance, hashes


def load_inputs(
    source: Path,
    primary_teacher: Path,
    fresh_teacher: Path,
    mlp_teacher: Path,
    seed: int,
    *,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Expose only original train inputs and teacher encoders after ancestry authentication."""
    bundle = load_primary_source(
        source,
        seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    primary_encoder = copy.deepcopy(model.parameter_encoder)
    discarded_table, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del discarded_table
    hashes = {str((source / name).resolve()): value for name, value in source_hashes.items()}
    teachers: list[nn.Module] = []
    provenance: list[dict[str, Any]] = []
    for index, root in enumerate((primary_teacher, fresh_teacher, mlp_teacher)):
        encoder, record, teacher_hashes = _load_teacher(
            root,
            index,
            bundle,
            model,
            primary_encoder,
            initialization,
            source_hashes,
            verification=verification,
        )
        teachers.append(encoder)
        provenance.append(record)
        hashes.update(teacher_hashes)
    inputs = Inputs(
        bundle,
        model.requires_grad_(False).eval(),
        tuple(teachers),
        initialization,
        hashes,
        {"selection_scope": SELECTION_SCOPE, "teachers": provenance},
    )
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Recheck immutable file bytes and teacher weights at run boundaries."""
    if any(sha(Path(name)) != expected for name, expected in inputs.hashes.items()):
        raise ValueError("Immutable teacher or source artifact changed")
    for teacher, record in zip(inputs.teachers, inputs.teacher_provenance["teachers"], strict=True):
        if module_fingerprint(teacher) != record["encoder_sha256"] or any(
            parameter.requires_grad for parameter in teacher.parameters()
        ):
            raise ValueError("Frozen teacher encoder changed")


def leave_one_out_targets(raw: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    """Apply fixed TRAIN moments to nonself teacher means, excluding each own context."""
    if (
        raw.ndim != 4
        or raw.shape[1:] != (10, 3, 6)
        or mean.shape != (3, 6)
        or scale.shape != (3, 6)
        or not bool(torch.isfinite(raw).all())
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(scale).all())
        or bool((scale <= 1e-8).any())
    ):
        raise ValueError("Invalid teacher codes or fixed TRAIN normalization")
    precise = raw.detach().cpu().double()
    # Explicit exclusion avoids own-view numerical cancellation in sum-minus-self.
    others = torch.stack(
        [
            torch.cat((precise[:, :view], precise[:, view + 1 :]), dim=1).mean(dim=1)
            for view in range(10)
        ],
        dim=1,
    )
    return ((others - mean.detach().cpu().double()) / scale.detach().cpu().double()).to(raw)


def normalize_targets(raw: Tensor) -> TeacherTargets:
    """Fit TRAIN population moments and average other nine normalized teacher contexts."""
    if (
        raw.ndim != 4
        or raw.shape[1:] != (10, 3, 6)
        or len(raw) < 2
        or not raw.is_floating_point()
        or not bool(torch.isfinite(raw).all())
    ):
        raise ValueError("Expected finite train teacher codes [groups,10,3,6]")
    codes = raw.detach().cpu().clone()
    precise = codes.double()
    mean = precise.mean(dim=(0, 1))
    scale = precise.std(dim=(0, 1), correction=0)
    if bool((scale <= 1e-8).any()):
        raise ValueError("Teacher coordinate has degenerate train variance")
    mean, scale = mean.to(codes), scale.to(codes)
    targets = leave_one_out_targets(codes, mean, scale)
    receipt: dict[str, Any] = {
        "fit_split": "train",
        "groups": len(codes),
        "contexts_per_group": 10,
        "teacher_order": list(TEACHER_NAMES),
        "teacher_coordinates": 6,
        "normalization": "Per-teacher/per-coordinate population mean/std over TRAIN groups/views",
        "target": "Detached normalized mean of other nine contexts; student context excluded",
        "arithmetic": (
            "Float64 moments cast to raw dtype; fixed stored moments and means applied in float64"
        ),
        "raw_sha256": tensor_fingerprint(codes),
        "mean_sha256": tensor_fingerprint(mean),
        "scale_sha256": tensor_fingerprint(scale),
        "targets_sha256": tensor_fingerprint(targets),
    }
    receipt["identity"] = canonical_fingerprint(receipt)
    return TeacherTargets(codes, mean, scale, targets, receipt)


def prepare_targets(
    inputs: Inputs,
    *,
    device: torch.device,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> TeacherTargets:
    """Encode TRAIN individual contexts only, never evaluating labels or held-out groups."""
    if batch_size < 1:
        raise ValueError("Teacher cache batch size must be positive")
    verify_sources(inputs)
    train = inputs.bundle.train
    groups = inputs.bundle.train_groups
    if set(train) != {"online", "target", "actions"} or (
        train["online"].shape != (groups, 10, 12, 6)
        or train["actions"].shape != (groups, 10, 11, 16)
        or any(value.requires_grad for value in train.values())
    ):
        raise ValueError("Teacher cache requires original label-free TRAIN tensors")
    teachers = [copy.deepcopy(teacher).to(device).eval() for teacher in inputs.teachers]
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, groups, batch_size):
            end = min(start + batch_size, groups)
            states = train["online"][start:end, :, :10].flatten(0, 1).to(device)
            actions = train["actions"][start:end, :, :9].flatten(0, 1).to(device)
            values = [teacher(states, actions).reshape(end - start, 10, 6) for teacher in teachers]
            chunks.append(torch.stack(values, dim=2).cpu())
            if progress is not None:
                progress(end, groups)
    targets = normalize_targets(torch.cat(chunks, dim=0))
    targets.receipt["train_inputs_sha256"] = inputs.bundle.hashes["train_inputs.pt"]
    targets.receipt["teacher_encoders"] = [module_fingerprint(teacher) for teacher in teachers]
    targets.receipt["identity"] = canonical_fingerprint(
        {key: value for key, value in targets.receipt.items() if key != "identity"}
    )
    verify_sources(inputs)
    return targets
