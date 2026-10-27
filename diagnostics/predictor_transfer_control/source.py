"""Authenticate the canonical learned endpoint and its TRAIN-only six-code chart."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import SourceBundle, load_primary_source
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.evaluation import encode_codes
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from diagnostics.predictor_transfer_control.model import (
    audit_model_fingerprint,
    compose_pca,
    fit_pca,
)
from diagnostics.shared_feature_acquisition.run import code_identity as parent_code_identity
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.training.rng import preserve_rng

Arm = Literal["spartan", "mlp"]
SEED = 202610261301
ENDPOINT_HASHES = {
    "last.pt": "5daecb051d2243fa5c747704bea7fa3b40836e3255180b13966124cb5a6b2301",
    "protocol.json": "c62290bd8c7516e91a777f04b1251cb1e0adb1a5c0c1aa89cffe6d3d8a201b4d",
    "completed.json": "3cc638091541c7e8cbff0000d00bd01011d041c3fadd064a1c3d4ace78c53f5d",
}
AUDIT_HASHES = {
    "results.json": "22767825f85ee99c01467a861c1d5935fbd90e39ebfd88b66349a2cbbe4f33fd",
    "measurements.pt": "f7e61b23f34be454465417d02b0cb6a3558a28f48ccf1b6969c6cbeacaa6ef63",
}


@dataclass(frozen=True)
class Inputs:
    """Identical frozen E6/state maps, fresh arm-specific P and untouched composed reference."""

    bundle: SourceBundle
    model: SCJEPA
    reference_model: SCJEPA
    hashes: dict[str, str]
    provenance: dict[str, Any]
    projection: dict[str, Tensor]


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Only predictor weights train in this control."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder", "parameter_encoder")
    }


@torch.no_grad()
def materialization_replay(
    reference: SCJEPA, candidate: SCJEPA, bundle: SourceBundle
) -> dict[str, float]:
    """Verify float32 deployment against float64 composition on first1024 TRAIN contexts."""
    states = bundle.train["online"].flatten(0, 1)[:1024]
    actions = bundle.train["actions"].flatten(0, 1)[:1024]
    codes: list[Tensor] = []
    predictions: list[Tensor] = []
    for model in (reference, candidate):
        parameter = next(model.parameters())
        z, a = states.to(parameter), actions.to(parameter)
        q = model.parameter_encoder(z[:, :10], a[:, :9])
        history = z[:, 9:10]
        outputs: list[Tensor] = []
        for step in (9, 10):
            predicted = model.predictor(history, q, a[:, step : step + 1]).prediction
            outputs.append(predicted)
            history = predicted[:, None]
        codes.append(q.double())
        predictions.append(torch.stack(outputs, 1).double())
    report: dict[str, float] = {}
    for name, values in (("code", codes), ("forecast", predictions)):
        rms = float(values[0].square().mean().sqrt())
        difference = float((values[1] - values[0]).square().mean().sqrt())
        tolerance = 1e-4 * rms + 1e-6
        if not difference <= tolerance:
            raise ValueError("Float32 PCA chart materialization exceeds numerical replay tolerance")
        report[name + "_rms_difference"] = difference
        report[name + "_rms_tolerance"] = tolerance
    return report


def load_inputs(
    source: Path,
    endpoint: Path,
    pca_audit: Path,
    seed: int,
    arm: Arm,
    *,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Bind source, fixed validation gate and TRAIN-fitted chart; never inherit trained P."""
    if arm not in ("spartan", "mlp") or (not verification and seed != SEED):
        raise ValueError("Require one declared predictor arm and canonical seed")
    if not verification and (
        any(sha(endpoint / name) != value for name, value in ENDPOINT_HASHES.items())
        or any(sha(pca_audit / name) != value for name, value in AUDIT_HASHES.items())
    ):
        raise ValueError("Canonical endpoint or fixed PCA audit changed")
    complete = authenticated_directory(endpoint, {"protocol.json", "last.pt", "results.json"})
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    if protocol["code_identity"] != parent_code_identity() or protocol["config"]["seed"] != seed:
        raise ValueError("Canonical shared-source code/seed differs")
    bundle = load_primary_source(
        source,
        seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    model, discarded = build_source_model(bundle, torch.device("cpu"))
    del discarded
    unused, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del unused
    fresh_spartan = copy.deepcopy(model.predictor)
    if not all(layer.dense and not layer.identity for layer in fresh_spartan.layers):
        raise ValueError("Require original dense SPARTAN with all gates held open")
    for name, value in source_hashes.items():
        recorded = str(Path(protocol["config"]["source_dir"]) / name)
        if protocol["source_hashes"].get(recorded) != value:
            raise ValueError("Endpoint and transfer use different original inputs")
    with preserve_rng():
        model.parameter_encoder = TransitionParameterEncoder(
            6, 12, 16, model.parameter_encoder.max_history
        )
        model.predictor = cast(Spartan, OvercompletePredictor())
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        payload["identity"] != protocol["identity"]
        or payload["step"] != protocol["config"]["steps"]
    ):
        raise ValueError("Endpoint is not its declared fixed final checkpoint")
    if not verification and payload["step"] != 100000:
        raise ValueError("Require fixed100k canonical endpoint")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval()
    for name, value in protocol["frozen_hashes"].items():
        if module_fingerprint(getattr(model, name)) != value:
            raise ValueError("Canonical source frozen states differ")
    audit = cast(dict[str, Any], read_json(pca_audit / "results.json"))
    if audit["measurement_sha256"] != sha(pca_audit / "measurements.pt") or audit[
        "source_model_sha256"
    ] != audit_model_fingerprint(model):
        raise ValueError("PCA audit is not bound to this learned endpoint")
    if (
        audit["predeclared_transfer_gate"]["passed"] is not True
        or not all(
            value >= 0.8 for value in audit["full_validation"]["pca6"]["selected_validation_r2"]
        )
        or len(audit["full_validation"]["pca6"]["selected_validation_r2"]) != 6
    ):
        raise ValueError("Predeclared all-six PCA validation gate failed")
    for path, value in {
        **{
            str((source / name).resolve()): v
            for name, v in source_hashes.items()
            if name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt")
        },
        **{str((endpoint / name).resolve()): v for name, v in complete["hashes"].items()},
    }.items():
        if value not in audit["source_hashes"].values():
            raise ValueError(f"PCA audit missing bound source artifact: {path}")
    record = cast(
        dict[str, Any],
        torch.load(pca_audit / "measurements.pt", map_location="cpu", weights_only=False),
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        codes = encode_codes(
            model, {"states_online": bundle.train["online"], "actions": bundle.train["actions"]}
        )
        mean, basis, eigenvalues = fit_pca(codes)
    finally:
        torch.set_num_threads(old_threads)
    # Sign alignment is numerical chart authentication, not a fitted rotation or whitening.
    for index in range(6):
        if float(basis[:, index] @ record["basis"][:, index]) < 0:
            basis[:, index].neg_()
    torch.testing.assert_close(mean, record["mean"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(basis, record["basis"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(eigenvalues, record["eigenvalues"], atol=1e-5, rtol=1e-4)
    with preserve_rng():
        composed = compose_pca(model.double(), record["mean"], record["basis"])
    for name, value in composed.state_dict().items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(value, record["composed_model"][name], atol=1e-8, rtol=1e-8)
        elif value != record["composed_model"][name]:
            raise ValueError("Composed PCA model flags differ")
    composed.load_state_dict(record["composed_model"], strict=True)
    if audit_model_fingerprint(composed) != audit["composed_model_sha256"]:
        raise ValueError("Saved composed reference hash differs")
    reference = copy.deepcopy(composed).float().requires_grad_(False).eval()
    replay = materialization_replay(composed, reference, bundle)
    model = copy.deepcopy(reference)
    if arm == "spartan":
        model.predictor = fresh_spartan
        predictor_receipt = {
            "policy": "Exact original archived fresh dense SPARTAN",
            "source_initialization": initialization,
        }
    else:
        with preserve_rng():
            predictor_receipt = initialize_mlp(model, seed)
    model.requires_grad_(False).eval()
    count = sum(p.numel() for p in model.predictor.parameters())
    if not verification and count != (52289 if arm == "spartan" else 52123):
        raise ValueError("Fresh predictor architecture count changed")
    frozen = fixed_hashes(model)
    if frozen != fixed_hashes(reference):
        raise RuntimeError("Fresh predictor replacement changed frozen E or state maps")
    hashes = {
        **{str((source / name).resolve()): value for name, value in source_hashes.items()},
        **{str((endpoint / name).resolve()): value for name, value in complete["hashes"].items()},
        str((endpoint / "completed.json").resolve()): sha(endpoint / "completed.json"),
        **{
            str((pca_audit / name).resolve()): sha(pca_audit / name)
            for name in ("results.json", "measurements.pt")
        },
    }
    provenance = {
        "arm": arm,
        "seed": seed,
        "source_identity": protocol["identity"],
        "source_endpoint_sha256": sha(endpoint / "last.pt"),
        "pca_validation_gate": audit["predeclared_transfer_gate"],
        "pca_validation_r2": audit["full_validation"]["pca6"]["selected_validation_r2"],
        "fresh_predictor": predictor_receipt,
        "fresh_predictor_sha256": module_fingerprint(model.predictor),
        "predictor_parameter_count": count,
        "frozen_hashes": frozen,
        "rollout_flags": model.get_extra_state(),
        "reference_model_sha256": audit_model_fingerprint(reference),
        "materialization_replay": replay,
        "pca_chart": "Authenticated native centered top6 TRAIN PCA; no extra scaling or whitening",
        "initialization": (
            "Only fresh P trained; E/state frozen; no source optimizer state or distillation"
        ),
    }
    result = Inputs(
        bundle,
        model,
        reference,
        hashes,
        provenance,
        {"mean": record["mean"], "basis": record["basis"], "eigenvalues": record["eigenvalues"]},
    )
    verify_sources(result)
    return result


def verify_sources(inputs: Inputs) -> None:
    """Keep source files, E/state maps, chart and composed reference unchanged."""
    if any(sha(Path(name)) != value for name, value in inputs.hashes.items()):
        raise ValueError("Immutable predictor-transfer source changed")
    if (
        fixed_hashes(inputs.model) != inputs.provenance["frozen_hashes"]
        or inputs.model.get_extra_state() != inputs.provenance["rollout_flags"]
    ):
        raise ValueError("Frozen transfer encoder/state or rollout settings changed")
    if (
        audit_model_fingerprint(inputs.reference_model)
        != inputs.provenance["reference_model_sha256"]
    ):
        raise ValueError("Composed frozen MLP6 reference changed")
