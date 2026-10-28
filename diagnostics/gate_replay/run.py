"""Replay factual effective gates for three fixed checkpoints on original256 validation groups."""

import argparse
import copy
import hashlib
import math
import shlex
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.gate_replay.core import replay_matched
from diagnostics.global_code_calibration.source import verify_files
from diagnostics.joint_acquisition.baseline import (
    MatchedBatch,
    _interval,  # pyright: ignore[reportPrivateUsage]
    response_report,
)
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_use_precision.evaluation import direct_decomposition, join_batches
from diagnostics.sparse_comparison.artifacts import code_identity as source_code_identity
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng

CONDITIONS = ("dense", "baseline", "full")
PINS = {
    "dense": "b06d65a99041118e25c53abe0d9891185615c60334f3f9d34df09062728d75f6",
    "baseline": "2a964ddc90091c554145151b578d888658d68f4b455eb3bcd1646f28e24f35c3",
    "full": "ba2c0217013f073cdf23fc74f33956c7434e9a7b6ee886bc2cd028eafc3f3e01",
}
SOURCE_FILES = {
    "protocol.json",
    "run_config.json",
    "last.pt",
    "last.receipt.json",
    "validation.json",
    "final.receipt.json",
    "results.json",
    "status.json",
    "tracking.json",
    *{
        "evaluation/" + name
        for name in (
            "metrics.json",
            "protocol.json",
            "mcc_samples.pt",
            "evaluation_references.pt",
            "evaluated_checkpoint.json",
            "diagnostic_final.json",
            "diagnostic_measurements.pt",
        )
    },
}


@dataclass(frozen=True)
class Config:
    """Fixed source pins, validation cohort and paired uncertainty."""

    dense_dir: str
    baseline_dir: str
    full_dir: str
    output_dir: str
    device: str = "cpu"
    chunk_size: int = 16
    threads: int = 1
    groups: int = 256
    bootstrap_samples: int = 1000
    seed: int = 202610261301
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Preserve the declared sample and reject source overlap before creating output."""
        if self.verification:
            if not 2 <= self.groups <= 4:
                raise ValueError("Verification requires2..4 validation groups")
        elif (self.groups, self.bootstrap_samples, self.seed) != (256, 1000, 202610261301):
            raise ValueError("Scientific replay fixes256validation/1000bootstrap/canonical seed")
        if min(self.chunk_size, self.threads) < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Require positive execution settings")
        output = Path(self.output_dir).resolve()
        for name in (self.dense_dir, self.baseline_dir, self.full_dir):
            source = Path(name).resolve()
            if source == output or source in output.parents or output in source.parents:
                raise ValueError("Output must be disjoint from immutable sources")
        if (
            len(
                {
                    Path(name).resolve()
                    for name in (self.dense_dir, self.baseline_dir, self.full_dir)
                }
            )
            != 3
        ):
            raise ValueError("Require three distinct declared source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the small replay adapter and all unchanged source/response dependencies."""
    digest = hashlib.sha256(source_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_source(
    path: Path, condition: str, *, device: torch.device, verification: bool = False
) -> tuple[SCJEPA, EvaluationData, dict[str, Any], dict[str, str]]:
    """Verify completed native source and its ancestry without creating a Trainer or optimizer."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown source condition")
    complete = authenticated_directory(path, SOURCE_FILES)
    if set(complete["hashes"]) != SOURCE_FILES:
        raise ValueError("Require the fixed16-artifact sparse completion")
    protocol = cast(dict[str, Any], read_json(path / "protocol.json"))
    record = cast(dict[str, Any], read_json(path / "run_config.json"))
    status = read_json(path / "status.json")
    checkpoint_sha = sha(path / "last.pt")
    if (
        protocol["code_identity"] != source_code_identity()
        or protocol["config"]["condition"] != condition
        or protocol["config"]["verification"] != verification
    ):
        raise ValueError("Source comparison code/condition/verification differs")
    if not verification and checkpoint_sha != PINS[condition]:
        raise ValueError("Require the pinned fixed30k source checkpoint")
    if (
        record["run_identity"] != complete["identity"]
        or record["condition"] != condition
        or status["status"] != "completed"
        or status["run_identity"] != complete["identity"]
    ):
        raise ValueError("Source native run identity/status differs")
    verify_files({**protocol["source_hashes"], **protocol["shared_hashes"]})
    receipt = read_json(path / "last.receipt.json")
    payload = cast(
        dict[str, Any], torch.load(path / "last.pt", map_location="cpu", weights_only=False)
    )
    budget = int(protocol["config"]["steps"])
    if (
        payload["format_version"] != 1
        or payload["step"] != budget
        or payload["optimizer_steps"] != budget
        or payload["run_metadata"]["experiment_run_identity"] != complete["identity"]
        or payload["model_config"] != record["model"]
        or payload["dataset_identity"] != record["dataset_identity"]
        or receipt["identity"] != complete["identity"]
        or receipt["sha256"] != checkpoint_sha
    ):
        raise ValueError("Source final checkpoint counters/configuration/receipt differ")
    if not verification and (budget, protocol["source_step"], record["seed"]) != (
        30000,
        200000,
        202610261301,
    ):
        raise ValueError("Require original global230k matched endpoint")
    model_config = ModelConfig(**record["model"])
    if (
        model_config.spartan_dense,
        model_config.spartan_parameter_gates_open,
        model_config.spartan_action_gates_open,
        model_config.spartan_identity,
    ) != (condition == "dense", condition == "baseline", False, False):
        raise ValueError("Source native gate policy differs")
    with preserve_rng(device):
        model = build_model(model_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval()
    if (
        model.context_len,
        model.autoregressive,
        model.ema_decay,
        model.parameter_encoder.param_dim,
        model.predictor.param_dim,
        model.parameter_source,
    ) != (10, True, 0.996, 6, 6, "learned") or any(
        not bool(torch.isfinite(p).all()) for p in model.parameters()
    ):
        raise ValueError("Source model interface or finite weights differ")
    shared_path = Path(protocol["config"]["shared_dir"])
    shared = cast(dict[str, Any], read_json(shared_path / "protocol.json"))
    raw_protocol_path = Path(shared["config"]["endpoint_dir"]) / "protocol.json"
    if protocol["source_hashes"].get(str(raw_protocol_path.resolve())) != sha(raw_protocol_path):
        raise ValueError("Raw initializer protocol is not authenticated")
    raw_protocol = cast(dict[str, Any], read_json(raw_protocol_path))
    cache = Path(raw_protocol["config"]["source_dir"]) / "evaluation_data.pt"
    if protocol["source_hashes"].get(str(cache.resolve())) != sha(cache):
        raise ValueError("Original diagnostic validation cache is not authenticated")
    original = cast(EvaluationData, torch.load(cache, map_location="cpu", weights_only=False))
    data: EvaluationData = {"metadata": original["metadata"], "validation": original["validation"]}
    if not verification and len(data["validation"]["theta"]) != 256:
        raise ValueError("Require all original256 diagnostic validation groups")
    hashes = {
        **protocol["source_hashes"],
        **protocol["shared_hashes"],
        **{str((path / name).resolve()): value for name, value in complete["hashes"].items()},
        str((path / "completed.json").resolve()): sha(path / "completed.json"),
    }
    return (
        model,
        data,
        {
            "run_identity": complete["identity"],
            "comparison_identity": record["experiment_identity"],
            "dataset_identity": record["dataset_identity"],
            "checkpoint_sha256": checkpoint_sha,
            "model_sha256": model_fingerprint(model),
            "model_config": record["model"],
            "condition": condition,
            "local_step": budget,
            "global_step": protocol["source_step"] + budget,
            "validation_cache": str(cache),
            "validation_cache_sha256": sha(cache),
            "model_loading_scope": (
                "Native checkpoint includes optimizer/history tensors; only model tensors "
                "are restored, no Trainer or optimizer is constructed."
            ),
        },
        hashes,
    )


def _summarize(
    base: Tensor,
    changed: Tensor,
    batch: MatchedBatch,
    samples: Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    report, raw = response_report(
        base, changed, batch, bootstrap_samples=bootstrap_samples, seed=seed
    )
    decomposition = direct_decomposition(base, changed, batch.targets, batch.changed_targets)
    raw.update(decomposition)
    report["decomposition"] = {
        name: [
            _interval(values[:, factor], torch.ones(len(base), dtype=torch.float64), samples)
            for factor in range(6)
        ]
        for name, values in decomposition.items()
    }
    torch.testing.assert_close(
        raw["predictive_benefit"], raw["benefit_from_direct_vectors"], atol=2e-15, rtol=1e-9
    )
    return report, raw


@torch.no_grad()
def evaluate_arm(
    model: SCJEPA,
    data: EvaluationData,
    *,
    chunk_size: int,
    bootstrap_samples: int,
    seed: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare natural and replayed gates on identical validation trajectories and bootstrap."""
    if (
        model.training
        or next(model.parameters()).dtype != torch.float32
        or set(data) != {"metadata", "validation"}
        or chunk_size < 1
    ):
        raise ValueError("Require eval float32 model and validation-only physical inputs")
    groups = len(data["validation"]["theta"])
    before = model_fingerprint(model)
    device = next(model.parameters()).device
    with preserve_rng(device):
        refreshed = refresh_evaluation(model, data, ("validation",))
        candidate = copy.deepcopy(model).double().requires_grad_(False).eval()
        fixed = model_fingerprint(candidate)
        batches: list[MatchedBatch] = []
        factual: list[Tensor] = []
        natural: list[Tensor] = []
        replayed: list[Tensor] = []
        flips: list[Tensor] = []
        factual_open: list[Tensor] = []
        natural_open: list[Tensor] = []
        for start in range(0, groups, chunk_size):
            indices = tuple(range(start, min(start + chunk_size, groups)))
            batch, _, _ = matched_batch(candidate, refreshed, "validation", indices)
            result = replay_matched(candidate, batch)
            torch.testing.assert_close(
                result.factual_replay, result.natural.predictions, rtol=0, atol=0
            )
            batches.append(batch.to(torch.zeros((), dtype=torch.float64)))
            factual.append(result.natural.predictions.detach().cpu())
            natural.append(result.natural.changed_predictions.detach().cpu())
            replayed.append(result.replayed_predictions.detach().cpu())
            flips.append(result.flip_counts.detach().cpu())
            factual_open.append(result.factual_masks.sum((-1, -2)).cpu())
            natural_open.append(result.natural_changed_masks.sum((-1, -2)).cpu())
            if progress:
                progress(min(start + chunk_size, groups))
        batch = join_batches(batches)
        base, changed, replay = torch.cat(factual), torch.cat(natural), torch.cat(replayed)
        samples = torch.randint(
            groups, (bootstrap_samples, groups), generator=torch.Generator().manual_seed(seed)
        )
        natural_report, natural_raw = _summarize(
            base, changed, batch, samples, bootstrap_samples=bootstrap_samples, seed=seed
        )
        replay_report, replay_raw = _summarize(
            base, replay, batch, samples, bootstrap_samples=bootstrap_samples, seed=seed
        )
        gate_change = changed - replay
        exact_response = batch.changed_targets - batch.targets[:, :, None]
        replay_error = replay - base[:, :, None] - exact_response
        gate_energy = gate_change.square().sum((1, 3, 4, 5))
        gate_cross = (2 * replay_error * gate_change).sum((1, 3, 4, 5))
        absolute_error_change = (
            natural_raw["response_error_energy"] - replay_raw["response_error_energy"]
        )
        torch.testing.assert_close(
            absolute_error_change, gate_energy + gate_cross, atol=2e-12, rtol=1e-9
        )
        gate_terms = {
            "gate_response_energy": gate_energy,
            "gate_error_cross_term": gate_cross,
            "natural_minus_replayed_error_energy": absolute_error_change,
        }
        gate_summary = {
            name: [
                _interval(value[:, factor], torch.ones(groups, dtype=torch.float64), samples)
                for factor in range(6)
            ]
            for name, value in gate_terms.items()
        }
        reference = natural_raw["exact_response_energy"]
        differences = {
            "normalized_response_error": (
                replay_raw["response_error_energy"] - natural_raw["response_error_energy"],
                reference,
            ),
            "response_energy_ratio": (
                replay_raw["learned_response_energy"] - natural_raw["learned_response_energy"],
                reference,
            ),
        }
        for name in (
            "predictive_benefit",
            "finite_response_gain",
            "twice_residual_dot_learned_response",
            "learned_response_energy",
            "response_error_energy",
        ):
            value = replay_raw[name] - natural_raw[name]
            differences[name] = (value, torch.ones_like(value))
        paired = {
            name: [
                _interval(numerator[:, factor], denominator[:, factor], samples)
                for factor in range(6)
            ]
            for name, (numerator, denominator) in differences.items()
        }
        counts = torch.cat(flips)
        totals = counts.sum((1, 3, 4, 5)).double()
        opportunities = (
            math.prod(counts.shape[1:2] + counts.shape[3:]) * candidate.predictor.num_tokens**2
        )
        flip_report = [
            _interval(
                totals[:, factor],
                torch.full((groups,), float(opportunities), dtype=torch.float64),
                samples,
            )
            for factor in range(6)
        ]
        dense = all(layer.dense for layer in candidate.predictor.layers)
        if dense and (not torch.equal(changed, replay) or bool(counts.any())):
            raise RuntimeError("Dense replay control differs or reports effective gate flips")
        if (
            model_fingerprint(model) != before
            or model_fingerprint(candidate) != fixed
            or any(parameter.grad is not None for parameter in candidate.parameters())
        ):
            raise RuntimeError("Read-only replay modified model weights or accumulated gradients")
    return {
        "natural": natural_report,
        "replayed_factual_gates": replay_report,
        "paired_replayed_minus_natural": paired,
        "effective_flip_fraction": flip_report,
        "gate_error_decomposition": gate_summary,
        "gate_error_identity_max_absolute_error": float(
            (absolute_error_change - gate_energy - gate_cross).abs().max()
        ),
        "dense_identity_control": dense,
        "groups": groups,
        "split": "original_diagnostic_validation",
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "paired": True},
        "scope": (
            "Frozen within-arm state chart and weights; changed code/attention remain "
            "natural while gate masks are replayed from the factual rollout. Diagnostic "
            "intervention only; no acceptance change, training, probe fitting or "
            "fresh-test claim."
        ),
    }, {
        "natural": natural_raw,
        "replayed_factual_gates": replay_raw,
        "vectors": {
            "factual_predictions": base,
            "natural_counterfactual_predictions": changed,
            "replayed_counterfactual_predictions": replay,
            "factual_targets": batch.targets.cpu(),
            "counterfactual_targets": batch.changed_targets.cpu(),
            "gate_contribution": gate_change,
        },
        "gate_error_decomposition": gate_terms,
        "effective_flip_counts": counts,
        "factual_open_gate_counts": torch.cat(factual_open),
        "natural_counterfactual_open_gate_counts": torch.cat(natural_open),
        "bootstrap_group_indices": samples,
        "paired_group_numerators": {name: values[0] for name, values in differences.items()},
        "gate_axes": (
            "flip:G,D,F,Q,H,L; open factual:G,D,Q,H,L; natural:G,D,F,Q,H,L; counts sum "
            "both token axes; baseline forced-open routes included as effective masks"
        ),
    }


def run(config: Config) -> dict[str, Any]:
    """Authenticate all sources, process every arm once, and preserve raw paired evidence."""
    out, device = Path(config.output_dir), torch.device(config.device)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            if any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError(
                    "Use a unique fresh output directory; replay does not overwrite"
                )
            models: dict[str, SCJEPA] = {}
            source_records: dict[str, Any] = {}
            hashes: dict[str, str] = {}
            data: EvaluationData | None = None
            for condition in CONDITIONS:
                model, current, record, files = load_source(
                    Path(getattr(config, condition + "_dir")),
                    condition,
                    device=device,
                    verification=config.verification,
                )
                if len(current["validation"]["theta"]) != config.groups:
                    raise ValueError(
                        "Requested sample differs from full original validation cohort"
                    )
                if source_records and (
                    record["comparison_identity"],
                    record["dataset_identity"],
                    record["validation_cache_sha256"],
                ) != tuple(
                    source_records["dense"][name]
                    for name in (
                        "comparison_identity",
                        "dataset_identity",
                        "validation_cache_sha256",
                    )
                ):
                    raise ValueError(
                        "Conditions do not share comparison/data/validation identities"
                    )
                models[condition], source_records[condition] = model, record
                hashes.update(files)
                data = current
            assert data is not None
            for name in hashes:
                source = Path(name).resolve().parent
                if (
                    out.resolve() == source
                    or source in out.resolve().parents
                    or out.resolve() in source.parents
                ):
                    raise ValueError("Output overlaps an immutable ancestor source")
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "sources": source_records,
                "source_hashes": hashes,
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "Sparse physical-response errors may be amplified by threshold gate "
                    "changes under small parameter perturbations."
                ),
                "controlled_change": (
                    "Within each fixed checkpoint and current state chart, compare "
                    "natural counterfactual gates with factual effective gate masks at "
                    "the same query/horizon/layer; attention logits and autoregressive "
                    "predictions still recompute. Dense is an exact no-op control."
                ),
                "evaluation": (
                    "All original256 diagnostic validation groups, donor0/query5..9, "
                    "positive.01 support-clipped physical interventions, two "
                    "autoregressive steps; same paired1000-bootstrap seed202610261301. No "
                    "labels for fitting, optimizer, training, test inference, probe "
                    "selection or acceptance changes. Checkpoint files contain historical "
                    "optimizer tensors, which are discarded."
                ),
            }
            identity = canonical_fingerprint(
                {key: value for key, value in protocol.items() if key != "command"}
            )
            protocol["identity"] = identity
            write_json(out / "protocol.json", protocol)
            write_json(
                out / "sources.json",
                {"identity": identity, "sources": source_records, "hashes": hashes},
            )
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=out.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=source_records["dense"]["dataset_identity"],
                out_dir=out,
            )
            write_json(out / "tracking.json", logger.metadata)
            event = 0

            def log(metrics: dict[str, float]) -> None:
                nonlocal event
                event += 1
                logger.log(event, metrics)
                print({"event": event, **metrics}, flush=True)

            try:
                reports: dict[str, Any] = {}
                for condition in CONDITIONS:
                    log({condition + "/started": 1.0})
                    report, raw = evaluate_arm(
                        models[condition],
                        data,
                        chunk_size=config.chunk_size,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        progress=lambda count, condition=condition: log(
                            {condition + "/validation_groups": float(count)}
                        ),
                    )
                    reports[condition] = report
                    atomic_save(
                        out / (condition + ".pt"),
                        {"identity": identity, "source": source_records[condition], **raw},
                    )
                    write_json(out / (condition + ".json"), {"identity": identity, **report})
                    metrics: dict[str, float] = {}
                    for mode in ("natural", "replayed_factual_gates"):
                        for row in report[mode]["per_parameter"]:
                            for metric in (
                                "normalized_response_error",
                                "predictive_benefit",
                                "response_energy_ratio",
                            ):
                                values = row[metric]
                                if values["defined"]:
                                    prefix = f"{condition}/{mode}/theta_{row['parameter']}/{metric}"
                                    metrics[prefix] = float(values["estimate"])
                                    metrics[prefix + "/lower95"], metrics[prefix + "/upper95"] = (
                                        values["interval_95"]
                                    )
                    for factor, value in enumerate(report["effective_flip_fraction"], 1):
                        metrics[f"{condition}/theta_{factor}/effective_flip_fraction"] = value[
                            "estimate"
                        ]
                    for metric, rows in report["paired_replayed_minus_natural"].items():
                        for factor, value in enumerate(rows, 1):
                            if value["defined"]:
                                prefix = f"{condition}/paired_delta/theta_{factor}/{metric}"
                                metrics[prefix] = float(value["estimate"])
                                metrics[prefix + "/lower95"], metrics[prefix + "/upper95"] = value[
                                    "interval_95"
                                ]
                    log(metrics)
                verify_files(hashes)
                result = {
                    "identity": identity,
                    "reports": reports,
                    "tracking": logger.metadata,
                    "scope": protocol["evaluation"],
                }
                write_json(out / "results.json", result)
                log({"diagnostic/complete": 1.0})
                for name in ("protocol.json", "results.json"):
                    logger.log_artifact(
                        out / name, name=f"gate-replay-{identity[:12]}-{Path(name).stem}"
                    )
                names = [
                    "protocol.json",
                    "sources.json",
                    "tracking.json",
                    "results.json",
                    *[
                        condition + suffix
                        for condition in CONDITIONS
                        for suffix in (".json", ".pt")
                    ],
                ]
                write_json(
                    out / "completed.json",
                    {"identity": identity, "hashes": {name: sha(out / name) for name in names}},
                )
                logger.finish()
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run the fixed three-arm read-only assay with online W&B progress."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dense-dir", "baseline-dir", "full-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
