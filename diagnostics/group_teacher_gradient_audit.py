"""Read-only, label-free component and optimizer audit of a completed warm teacher.

Disposable one-step counterfactuals share batches, RNG and optimizer moments.
They diagnose local predictive interference, not physical-factor acquisition.
"""

import argparse
import copy
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.forward import clip_teacher_gradients, group_teacher_forward
from diagnostics.group_latent_control.run import phase_optimizers
from diagnostics.group_latent_table_audit import verified_curriculum
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import source_fingerprint, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

AUDIT_SEED = 2026102702
COMPONENTS = ("prediction", "sigreg", "consistency", "logit")
FRACTIONS = (1.0, 0.1, 0.01)
Gradients = dict[str, tuple[Tensor, ...]]


def _norm(values: tuple[Tensor, ...]) -> float:
    return sum(float(value.double().square().sum()) for value in values) ** 0.5


def _dot(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> float:
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(first, second, strict=True))


def _geometry(first: tuple[Tensor, ...], second: tuple[Tensor, ...]) -> dict[str, Any]:
    dot = _dot(first, second)
    denominator = _norm(first) * _norm(second)
    return {"dot": dot, "cosine": dot / denominator if denominator else None}


def component_gradients(
    model: SCJEPA,
    batch: TensorSplit,
    codes: Tensor,
    config: ObjectiveConfig,
    seed: int,
) -> tuple[Gradients, dict[str, float]]:
    """Differentiate one shared forward with respect to actual B6 code rows and P.

    Row derivatives include SIGReg's cross-group coupling. They are not an
    additive group decomposition of the regularizer or physical-factor scores.
    """
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Gradient audit accepts label-free training inputs only")
    reference = next(model.parameters())
    with preserve_rng(reference.device):
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        q = codes.detach().clone().requires_grad_(True)
        output = group_teacher_forward(model, batch, q)
        total, terms = PredictiveSupervisionObjective(config, 1.0).to(reference)(output)
        losses = {
            "prediction": terms["prediction"],
            "sigreg": config.parameter_regularizer_weight * terms["parameter_regularizer"],
            "consistency": config.parameter_consistency_weight * terms["parameter_consistency"],
            "logit": config.logit_weight * terms["logit"],
            "total": total,
        }
        if not torch.allclose(torch.stack([losses[name] for name in COMPONENTS]).sum(), total):
            raise ValueError("Recorded objective contains an unaudited component")
        inputs = (q, *model.predictor.parameters())
        gradients: Gradients = {}
        for name, loss in losses.items():
            raw = (
                torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
                if loss.requires_grad
                else (None,) * len(inputs)
            )
            gradients[name] = tuple(
                torch.zeros_like(tensor) if value is None else value.detach().clone()
                for tensor, value in zip(inputs, raw, strict=True)
            )
        combined = tuple(
            torch.stack(values).sum(0)
            for values in zip(*(gradients[n] for n in COMPONENTS), strict=True)
        )
        for actual, expected in zip(combined, gradients["total"], strict=True):
            if not torch.allclose(actual, expected, rtol=2e-5, atol=1e-7):
                raise ValueError("Weighted component gradients do not sum to the full gradient")
        if not all(bool(torch.isfinite(g).all()) for row in gradients.values() for g in row):
            raise FloatingPointError("Nonfinite component gradient")
        return gradients, {name: float(value.detach()) for name, value in losses.items()}


def optimizer_direction(
    model: SCJEPA,
    weights: Tensor,
    indices: Tensor,
    gradients: tuple[Tensor, ...],
    optimizer_state: Mapping[str, Any] | None,
    latent_optimizer_state: Mapping[str, Any] | None,
) -> tuple[tuple[Tensor, ...], dict[str, float]]:
    """Clone teacher Adam/SparseAdam moments and apply one globally clipped update.

    Return sampled code-row and predictor displacements. All source parameters,
    optimizer moments and unsampled rows remain untouched.
    """
    if len(set(int(value) for value in indices)) != len(indices):
        raise ValueError("Audit batches require distinct training-group indices")
    if (optimizer_state is None) != (latent_optimizer_state is None):
        raise ValueError("Both optimizer states must be supplied together")
    candidate = copy.deepcopy(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        weights.detach().clone(), freeze=False, sparse=True
    )
    optimizer, latent_optimizer = phase_optimizers(candidate, "teacher", table)
    assert latent_optimizer is not None
    if optimizer_state is not None and latent_optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(dict(optimizer_state)))
        latent_optimizer.load_state_dict(copy.deepcopy(dict(latent_optimizer_state)))
    before = (
        table.weight[indices].detach().clone(),
        *(parameter.detach().clone() for parameter in candidate.predictor.parameters()),
    )
    table.weight.grad = torch.sparse_coo_tensor(  # pyright: ignore[reportUnknownMemberType]
        indices[None],
        gradients[0].clone(),
        size=weights.shape,
        device=weights.device,
        check_invariants=True,
    ).coalesce()
    for parameter, gradient in zip(candidate.predictor.parameters(), gradients[1:], strict=True):
        parameter.grad = gradient.clone()
    norm = clip_teacher_gradients(candidate.predictor.parameters(), table, 1.0)
    latent_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    after = (table.weight[indices].detach(), *candidate.predictor.parameters())
    direction = tuple(new.detach() - old for old, new in zip(before, after, strict=True))
    return direction, {
        "preclip_global_norm": norm,
        "clip_coefficient": min(1.0, 1.0 / (norm + 1e-6)),
        "table_update_norm": _norm(direction[:1]),
        "predictor_update_norm": _norm(direction[1:]),
    }


def forecast_changes(
    model: SCJEPA,
    batch: TensorSplit,
    codes: Tensor,
    directions: Mapping[str, tuple[Tensor, ...]],
    prediction_gradient: tuple[Tensor, ...],
    seed: int,
) -> dict[str, Any]:
    """Measure actual forecast changes along identical fitted update directions.

    Table-only and predictor-only movements isolate each block's contribution.
    These are training-batch forecast tests, not held-out or acquisition metrics.
    """
    candidate = copy.deepcopy(model)
    parameters = tuple(candidate.predictor.parameters())
    original = tuple(parameter.detach().clone() for parameter in parameters)

    def measure(q: Tensor) -> float:
        with preserve_rng(q.device), torch.no_grad():
            torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
            output = group_teacher_forward(candidate, batch, q)
            return float((output.predictions - output.targets).square().mean())

    baseline = measure(codes)
    result: dict[str, Any] = {"baseline_prediction": baseline, "directions": {}}
    with torch.no_grad():
        for branch, direction in directions.items():
            blocks: dict[str, Any] = {}
            for block in ("joint", "table", "predictor"):
                active = (
                    direction
                    if block == "joint"
                    else tuple(
                        value if (index == 0) == (block == "table") else torch.zeros_like(value)
                        for index, value in enumerate(direction)
                    )
                )
                slope = _dot(prediction_gradient, active)
                finite: dict[str, Any] = {}
                for fraction in FRACTIONS:
                    for parameter, old, delta in zip(parameters, original, active[1:], strict=True):
                        parameter.copy_(old + fraction * delta)
                    changed = measure(codes + fraction * active[0]) - baseline
                    expected = fraction * slope
                    finite[str(fraction)] = {
                        "actual_prediction_change": changed,
                        "linear_prediction_change": expected,
                        "linearization_valid": abs(changed - expected)
                        <= 1e-7 + 0.05 * abs(expected),
                    }
                blocks[block] = {"prediction_slope": slope, "finite": finite}
            result["directions"][branch] = blocks
    return result


def analyze_batch(
    model: SCJEPA,
    weights: Tensor,
    batch: TensorSplit,
    indices: Tensor,
    config: ObjectiveConfig,
    seed: int,
    optimizer_state: Mapping[str, Any] | None = None,
    latent_optimizer_state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare factual and no-SIGReg updates without changing saved state or RNG."""
    with preserve_rng(next(model.parameters()).device):
        gradients, losses = component_gradients(model, batch, weights[indices], config, seed)
        report: dict[str, Any] = {"losses": losses, "blocks": {}, "optimizer": {}}
        for block, sliced in (("table", slice(0, 1)), ("predictor", slice(1, None))):
            rows = {name: value[sliced] for name, value in gradients.items()}
            report["blocks"][block] = {
                "gradient_norm": {name: _norm(value) for name, value in rows.items()},
                "pairs": {
                    first + "__" + second: _geometry(rows[first], rows[second])
                    for first, second in combinations(rows, 2)
                },
            }
        # Sum retained terms directly: subtracting a dominant SIGReg derivative
        # from the float32 total could erase the weak predictive derivative.
        omitted = tuple(
            torch.stack(parts).sum(0)
            for parts in zip(
                *(gradients[name] for name in ("prediction", "consistency", "logit")),
                strict=True,
            )
        )
        directions: dict[str, tuple[Tensor, ...]] = {}
        for branch, gradient in (("factual", gradients["total"]), ("without_sigreg", omitted)):
            direction, metadata = optimizer_direction(
                model, weights, indices, gradient, optimizer_state, latent_optimizer_state
            )
            directions[branch] = direction
            report["optimizer"][branch] = {**metadata, "blocks": {}}
            for block, sliced in (("table", slice(0, 1)), ("predictor", slice(1, None))):
                report["optimizer"][branch]["blocks"][block] = {
                    name: _geometry(gradient[sliced], direction[sliced])
                    for name, gradient in gradients.items()
                }
        report["forecast"] = forecast_changes(
            model, batch, weights[indices], directions, gradients["prediction"], seed
        )
        raw = {
            "indices": indices.cpu(),
            "seed": seed,
            "table_gradients": {name: value[0].cpu() for name, value in gradients.items()},
            "table_updates": {name: value[0].cpu() for name, value in directions.items()},
        }
        return report, raw


def summarize(records: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Bootstrap independent batches after averaging paired projection replicates."""
    generator = torch.Generator().manual_seed(AUDIT_SEED + 1)
    samples = torch.randint(len(records), (2000, len(records)), generator=generator)
    summary: dict[str, Any] = {}
    for block in ("joint", "table", "predictor"):
        for fraction in FRACTIONS:
            values: list[Tensor] = []
            for batch in records:
                pairs: list[list[float]] = []
                for row in batch:
                    branches = row["forecast"]["directions"]
                    pairs.append(
                        [
                            branches[name][block]["finite"][str(fraction)][
                                "actual_prediction_change"
                            ]
                            for name in ("factual", "without_sigreg")
                        ]
                    )
                values.append(torch.tensor(pairs, dtype=torch.float64).mean(0))
            data = torch.stack(values)
            data = torch.cat((data, (data[:, 1] - data[:, 0])[:, None]), dim=1)
            interval = torch.quantile(
                data[samples].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0
            )
            summary[block + "/" + str(fraction)] = {
                name: {
                    "mean": float(data[:, column].mean()),
                    "ci95": interval[:, column].tolist(),  # pyright: ignore[reportUnknownMemberType]
                }
                for column, name in enumerate(("factual", "without_sigreg", "paired_difference"))
            }
    return summary


def audit(curriculum_output: Path, output: Path, *, wandb_mode: str = "disabled") -> dict[str, Any]:
    """Audit authenticated warm initialization and teacher endpoint on fixed CPU batches."""
    curriculum_output, output = curriculum_output.resolve(), output.resolve()
    curriculum, completion = verified_curriculum(curriculum_output)
    config = curriculum["config"]
    if config.get("initialization", "warm") != "warm":
        raise ValueError("This gradient audit requires a completed warm curriculum")
    source = Path(config["source_dir"]).resolve()
    if any(output.is_relative_to(path) for path in (source, curriculum_output)):
        raise ValueError("Audit output must be outside source and curriculum directories")
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with preserve_rng(torch.device("cpu")):
            bundle = load_primary_source(
                source,
                config["seed"],
                training_source_root=Path(config["training_source_root"])
                if config["training_source_root"]
                else None,
            )
            if bundle.hashes != curriculum["source_hashes"] or bundle.batch_size != 16:
                raise ValueError("Curriculum source or batch-size mismatch")
            primary, _ = build_source_model(bundle, torch.device("cpu"))
            primary.requires_grad_(False)
            primary.predictor.requires_grad_(True)
            primary.train()
            initial = cast(
                dict[str, Any],
                torch.load(
                    curriculum_output / "initial_group_codes.pt",
                    map_location="cpu",
                    weights_only=False,
                ),
            )
            checkpoint = cast(
                dict[str, Any],
                torch.load(
                    curriculum_output / "teacher_final.pt", map_location="cpu", weights_only=False
                ),
            )
            if (
                initial["identity"] != curriculum["identity"]
                or checkpoint["identity"] != curriculum["identity"]
                or checkpoint["phase"] != "teacher"
                or checkpoint["phase_step"] != 30000
                or checkpoint["global_step"] != 30000
                or checkpoint["optimizer"] is None
                or checkpoint["latent_optimizer"] is None
            ):
                raise ValueError("Teacher checkpoint identity or counters differ")
            teacher = copy.deepcopy(primary)
            teacher.load_state_dict(checkpoint["model"], strict=True)
            for name in ("online_encoder", "target_encoder", "parameter_encoder"):
                if module_fingerprint(getattr(primary, name)) != module_fingerprint(
                    getattr(teacher, name)
                ):
                    raise ValueError("Teacher changed an inactive model block")
            if teacher.get_extra_state() != primary.get_extra_state():
                raise ValueError("Teacher changed rollout configuration")
            generator = torch.Generator().manual_seed(AUDIT_SEED)
            selected = torch.randperm(bundle.train_groups, generator=generator)[:256].reshape(
                16, 16
            )
            seeds = [AUDIT_SEED + 100 + index for index in range(3)]
            evidence: dict[str, Any] = {
                "curriculum_identity": curriculum["identity"],
                "curriculum_artifact_hashes": completion["hashes"],
                "source_hashes": bundle.hashes,
                "audit_code_sha256": sha(Path(__file__)),
                "executed_source_fingerprint": source_fingerprint(),
                "executed_helper_code_sha256": {
                    str(path): sha(Path(__file__).resolve().parents[1] / path)
                    for path in (
                        "diagnostics/group_latent_table_audit.py",
                        "diagnostics/joint_acquisition/source.py",
                        "diagnostics/encoder_gradient_control/source.py",
                        "diagnostics/group_latent_control/forward.py",
                        "diagnostics/group_latent_control/run.py",
                        "diagnostics/predictive_supervision_control/objective.py",
                        "diagnostics/multiquery_control/objective.py",
                        "src/scjepa/training/loop.py",
                        "src/scjepa/losses/regularization.py",
                        "src/scjepa/losses/predictive.py",
                        "src/scjepa/models/model.py",
                        "src/scjepa/models/spartan.py",
                    )
                },
                "seed": AUDIT_SEED,
                "group_indices": selected.tolist(),  # pyright: ignore[reportUnknownMemberType]
                "projection_seeds": seeds,
                "objective": bundle.protocol["objective_config"],
                "fractions": FRACTIONS,
                "scope": (
                    "Label-free, training-batch local interference; "
                    "not factor recovery or held-out performance"
                ),
                "comparison": (
                    "Same gradients and moments; omit weighted SIGReg, "
                    "then repeat actual global clipping and Adam/SparseAdam"
                ),
                "initial_moments": "Fresh phase optimizers, exactly as at teacher initialization",
                "endpoint_moments": (
                    "Saved teacher-final optimizer moments; counterfactual next steps, "
                    "not a replay of the preceding update"
                ),
                "bootstrap": (
                    "2000 paired batch resamples; average three projection "
                    "replicates within each batch first"
                ),
                "interpretation": (
                    "Negative paired difference favors omitted SIGReg on this forecast batch; "
                    "norms alone do not establish interference or missing-factor causation"
                ),
                "limits": (
                    "Saved endpoint moments contain historical SIGReg contributions. "
                    "A null omission result cannot exclude past regularizer effects. "
                    "A favorable omission result establishes local forecast interference, "
                    "not acquisition of the four missing physical factors. "
                    "Finite changes include second-order effects; use the reported slope "
                    "and linearization checks when interpreting directionality."
                ),
            }
            identity = canonical_fingerprint(evidence)
            output.mkdir(parents=True)
            write_json(output / "protocol.json", {"identity": identity, **evidence})
            logger = WandbLogger(
                WandbConfig(
                    mode=wandb_mode,
                    project=config["wandb_project"],
                    entity=config["wandb_entity"],
                    name=output.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=evidence,
                dataset_identity=bundle.protocol["identity"],
                out_dir=output,
            )
            write_json(output / "tracking.json", logger.metadata)
            try:
                reports: dict[str, Any] = {}
                raw: dict[str, Any] = {}
                progress = 0
                for stage, model, weights, adam, sparse in (
                    ("initial", primary, initial["weight"], None, None),
                    (
                        "teacher_final",
                        teacher,
                        checkpoint["table"]["weight"],
                        checkpoint["optimizer"],
                        checkpoint["latent_optimizer"],
                    ),
                ):
                    if weights.shape != (bundle.train_groups, 6) or not bool(
                        torch.isfinite(weights).all()
                    ):
                        raise ValueError("Invalid authenticated teacher code table")
                    records: list[list[dict[str, Any]]] = []
                    raw[stage] = []
                    for batch_number, indices in enumerate(selected):
                        batch = {key: value[indices] for key, value in bundle.train.items()}
                        replicas: list[dict[str, Any]] = []
                        for seed in seeds:
                            report, tensors = analyze_batch(
                                model,
                                weights,
                                batch,
                                indices,
                                ObjectiveConfig(**bundle.protocol["objective_config"]),
                                seed,
                                adam,
                                sparse,
                            )
                            replicas.append(report)
                            raw[stage].append(tensors)
                        records.append(replicas)
                        metrics = {
                            f"{stage}/{block}/{name}_gradient_norm": sum(
                                row["blocks"][block]["gradient_norm"][name] for row in replicas
                            )
                            / len(replicas)
                            for block in ("table", "predictor")
                            for name in (*COMPONENTS, "total")
                        }
                        for block in ("table", "predictor"):
                            cosines = [
                                row["blocks"][block]["pairs"]["prediction__sigreg"]["cosine"]
                                for row in replicas
                            ]
                            if all(value is not None for value in cosines):
                                metrics[f"{stage}/{block}/prediction_sigreg_cosine"] = sum(
                                    cosines
                                ) / len(cosines)
                        metrics[f"audit/{stage}/completed_batches"] = float(batch_number + 1)
                        logger.log(progress, metrics)
                        progress += 1
                    summary = summarize(records)
                    reports[stage] = {"records": records, "summary": summary}
                    logger.log(
                        progress,
                        {
                            f"{stage}/{key}/{branch}": float(item["mean"])
                            for key, branches in summary.items()
                            for branch, item in branches.items()
                        },
                    )
                    progress += 1
                result = {
                    "identity": identity,
                    "reports": reports,
                    "tracking": logger.metadata,
                    "scope": evidence["scope"],
                }
                write_json(output / "results.json", result)
                atomic_save(output / "row_gradients.pt", {"identity": identity, "stages": raw})
                verified_curriculum(curriculum_output)
                if any(sha(source / name) != expected for name, expected in bundle.hashes.items()):
                    raise RuntimeError("Source changed during read-only audit")
                logger.log_artifact(
                    output / "results.json", name=f"teacher-gradient-{identity[:12]}-results"
                )
                logger.finish()
                write_json(
                    output / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(output / name)
                            for name in ("protocol.json", "results.json", "row_gradients.pt")
                        },
                    },
                )
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Run a fixed-protocol CPU audit into a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    args = parser.parse_args()
    print(audit(args.curriculum_output, args.output_dir, wandb_mode=args.wandb_mode)["identity"])


if __name__ == "__main__":
    main()
