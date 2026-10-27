"""Read-only optimizer-history and code-use audit after completed SIGReg omission."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.table_settling_control.evaluation import forecast_errors, paired_change
from diagnostics.table_settling_control.run import TableSettlingObjective, train_update
from diagnostics.table_settling_control.source import (
    authenticated_directory,
    load_inputs,
    model_fingerprint,
)
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, source_fingerprint, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

FRACTIONS = (1.0, 0.1, 0.01)
SEED = 2026102704


def embedding(values: Tensor) -> nn.Embedding:
    """Create a disposable sparse table without consuming RNG."""
    return nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        values.detach().clone(), freeze=False, sparse=True
    )  # pyright: ignore[reportUnknownMemberType]


def inherited_moments(source: Tensor, final: Tensor, updates: int, beta2: float) -> dict[str, Any]:
    """Decompose the final second moment into inherited and newly accumulated mass."""
    if source.shape != final.shape or updates < 0 or not 0 < beta2 < 1:
        raise ValueError("Invalid second-moment comparison")
    if any(not bool(torch.isfinite(v).all()) or bool((v < 0).any()) for v in (source, final)):
        raise ValueError("Second moments must be finite and nonnegative")
    inherited = source.double() * beta2**updates
    defined = final > 0
    fraction = torch.where(defined, inherited / final.double().clamp_min(1e-300), 0)
    if bool((inherited > final.double() * (1 + 2e-5) + 1e-30).any()):
        raise ValueError("Final moments cannot contain the claimed inherited contribution")
    values = fraction[defined]
    return {
        "fraction": fraction,
        "defined": defined,
        "summary": {
            "updates_per_row": updates,
            "inherited_multiplier": beta2**updates,
            "defined_coordinates": int(defined.sum()),
            "undefined_zero_coordinates": int((~defined).sum()),
            "mean": float(values.mean()) if values.numel() else None,
            "quantiles_0_25_50_75_100": torch.quantile(
                values, torch.tensor([0, 0.25, 0.5, 0.75, 1], dtype=torch.float64)
            ).tolist()  # pyright: ignore[reportUnknownMemberType]
            if values.numel()
            else [],
        },
    }


def optimizer_slice(state: dict[str, Any], indices: Tensor) -> dict[str, Any]:
    """Keep exact global step and options while selecting the matching moment rows."""
    result = copy.deepcopy(state)
    for name in ("exp_avg", "exp_avg_sq"):
        result["state"][0][name] = result["state"][0][name][indices].clone()
    return result


def next_direction(
    model: SCJEPA,
    values: Tensor,
    state: dict[str, Any],
    batch: TensorSplit,
    config: ObjectiveConfig,
    seed: int,
    *,
    reset: bool,
) -> tuple[Tensor, dict[str, float]]:
    """Run one actual omitted-SIGReg update on a private table with matched RNG."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        with preserve_rng(torch.device("cpu")):
            torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
            table = embedding(values)
            optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
            restored = copy.deepcopy(state)
            if reset:
                restored["state"] = {}
            optimizer.load_state_dict(restored)
            terms = train_update(
                model,
                table,
                optimizer,
                batch,
                torch.arange(len(values)),
                TableSettlingObjective(config, "without_parameter_sigreg"),
            )
            return table.weight.detach() - values, terms
    finally:
        for module, training in modes:
            module.training = training


def endpoint_measurements(
    model: SCJEPA,
    values: Tensor,
    state: dict[str, Any],
    train: TensorSplit,
    fresh: TensorSplit,
    config: ObjectiveConfig,
    *,
    batch_size: int = 16,
    seed: int = SEED,
) -> dict[str, Any]:
    """Keep fresh episodes strictly in no-grad forecast evaluation, never update fitting."""
    if len(values) % batch_size or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Need complete batches and a fully frozen model")
    if any(v.requires_grad for data in (train, fresh) for v in data.values()):
        raise ValueError("Cached and fresh evaluation tensors must be detached")
    frozen = model_fingerprint(model)
    modes = [(module, module.training) for module in model.modules()]
    raw: dict[str, Any] = {"baseline": {}, "permuted": {}, "updates": {}, "directions": []}
    try:
        with preserve_rng(torch.device("cpu")):
            data_sets = {"cached": train, "new_episodes": fresh}
            for name, data in data_sets.items():
                raw["baseline"][name] = forecast_errors(model, embedding(values), data, batch_size)
            order = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))
            permutations: list[Tensor] = []
            for shift in range(1, 5):
                permutation = torch.empty_like(order)
                permutation[order] = order.roll(1 + (shift - 1) % (len(values) - 1))
                permutations.append(permutation)
            raw["permutations"] = torch.stack(permutations)
            for name, data in data_sets.items():
                raw["permuted"][name] = torch.stack(
                    [
                        forecast_errors(model, embedding(values[p]), data, batch_size)
                        for p in permutations
                    ]
                )
            for arm in ("retained", "reset"):
                raw["updates"][arm] = {str(f): {name: [] for name in data_sets} for f in FRACTIONS}
            for start in range(0, len(values), batch_size):
                indices = torch.arange(start, start + batch_size)
                batch = {key: value[indices] for key, value in train.items()}
                for arm in ("retained", "reset"):
                    direction, terms = next_direction(
                        model,
                        values[indices],
                        optimizer_slice(state, indices),
                        batch,
                        config,
                        seed + start,
                        reset=arm == "reset",
                    )
                    raw["directions"].append(
                        {"start": start, "arm": arm, "delta": direction, "terms": terms}
                    )
                    for fraction in FRACTIONS:
                        proposal = embedding(values[indices] + fraction * direction)
                        for name, data in data_sets.items():
                            # Fresh episodes only reach this @no_grad evaluation helper.
                            errors = forecast_errors(
                                model,
                                proposal,
                                {key: value[indices] for key, value in data.items()},
                                batch_size,
                            )
                            raw["updates"][arm][str(fraction)][name].append(errors)
            for arm in raw["updates"].values():
                for fraction in arm.values():
                    for name, parts in fraction.items():
                        fraction[name] = torch.cat(parts)
    finally:
        for module, training in modes:
            module.training = training
    if model_fingerprint(model) != frozen:
        raise RuntimeError("Audit changed frozen model")
    return raw


def summarize(raw: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Pair systems; cluster next-step intervals by B16 because clipping couples rows."""
    result: dict[str, Any] = {
        "baseline_mse": {name: values.mean(0).tolist() for name, values in raw["baseline"].items()},
        "permutation": {},
        "updates": {},
        "reset_minus_retained": {},
        "direction_rms": {},
    }
    for arm in ("retained", "reset"):
        deltas = torch.cat([row["delta"] for row in raw["directions"] if row["arm"] == arm])
        result["direction_rms"][arm] = float(deltas.double().square().mean().sqrt())
    for name, baseline in raw["baseline"].items():
        result["permutation"][name] = paired_change(
            baseline, raw["permuted"][name].mean(0), samples=2000, seed=SEED
        )
    for arm in ("retained", "reset"):
        result["updates"][arm] = {}
        for fraction in FRACTIONS:
            result["updates"][arm][str(fraction)] = {}
            result["reset_minus_retained"][str(fraction)] = {}
            for name, baseline in raw["baseline"].items():
                before = baseline.reshape(-1, batch_size, 2).mean(1)
                after = raw["updates"][arm][str(fraction)][name].reshape(-1, batch_size, 2).mean(1)
                result["updates"][arm][str(fraction)][name] = paired_change(
                    before, after, samples=2000, seed=SEED
                )
                result["reset_minus_retained"][str(fraction)][name] = paired_change(
                    raw["updates"]["retained"][str(fraction)][name]
                    .reshape(-1, batch_size, 2)
                    .mean(1),
                    raw["updates"]["reset"][str(fraction)][name].reshape(-1, batch_size, 2).mean(1),
                    samples=2000,
                    seed=SEED,
                )
    return result


def scalar_metrics(value: object, prefix: str = "audit") -> dict[str, float]:
    """Expose all finite numeric summaries, including CI endpoints, in W&B history."""
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, item in cast(dict[str, Any], value).items():
            result.update(scalar_metrics(item, f"{prefix}/{key}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(cast(list[Any], value)):
            result.update(scalar_metrics(item, f"{prefix}/{index}"))
        return result
    return {prefix: float(value)} if isinstance(value, float | int) else {}


def run(
    settling_dir: Path,
    output_dir: Path,
    *,
    wandb_mode: str = "disabled",
    verification: bool = False,
) -> dict[str, Any]:
    """Authenticate a completed omission endpoint, write independent audit artifacts."""
    completion = authenticated_directory(
        settling_dir,
        {
            "protocol.json",
            "results.json",
            "last.pt",
            "evaluation_only.pt",
            "evaluation_manifest.json",
        },
    )
    protocol = cast(dict[str, Any], read_json(settling_dir / "protocol.json"))
    settings = protocol["config"]
    if settings.get("train_predictor", False):
        raise ValueError("Optimizer-history audit requires a fixed predictor endpoint")
    if (
        settings["objective_arm"] != "without_parameter_sigreg"
        or bool(settings["verification"]) != verification
        or (not verification and (settings["steps"], settings["groups"]) != (4000, 256))
    ):
        raise ValueError("Require completed fixed-budget SIGReg-omission settling")
    inputs = load_inputs(
        Path(settings["curriculum_dir"]),
        Path(settings["audit_dir"]),
        verification=verification,
        groups=settings["groups"],
        selection_seed=settings["seed"],
    )
    if inputs.curriculum["config"].get("initialization") != "fresh_zero":
        raise ValueError("History audit requires the fresh teacher")
    if protocol["source_hashes"] != inputs.file_hashes:
        raise ValueError("Settling source hashes differ from authenticated teacher")
    paths = dict(inputs.file_hashes)
    paths.update(
        {
            str((settling_dir / name).resolve()): value
            for name, value in completion["hashes"].items()
        }
    )
    paths[str((settling_dir / "completed.json").resolve())] = sha(settling_dir / "completed.json")
    for root in (
        settling_dir,
        inputs.bundle.source_dir,
        Path(settings["curriculum_dir"]),
        Path(settings["audit_dir"]),
    ):
        if (
            output_dir.resolve() == root.resolve()
            or output_dir.resolve() in root.resolve().parents
            or root.resolve() in output_dir.resolve().parents
        ):
            raise ValueError("Output overlaps immutable inputs")
    checkpoint = cast(
        dict[str, Any], torch.load(settling_dir / "last.pt", map_location="cpu", weights_only=False)
    )
    evaluation = cast(
        dict[str, Any],
        torch.load(settling_dir / "evaluation_only.pt", map_location="cpu", weights_only=False),
    )
    manifest = read_json(settling_dir / "evaluation_manifest.json")
    if (
        checkpoint["identity"] != completion["identity"]
        or checkpoint["step"] != settings["steps"]
        or evaluation["identity"] != completion["identity"]
        or manifest["identity"] != completion["identity"]
        or manifest["sha256"] != sha(settling_dir / "evaluation_only.pt")
        or checkpoint["evaluation_sha256"] != manifest["sha256"]
        or not torch.equal(evaluation["group_indices"], inputs.group_indices)
        or protocol["selected_group_ids"] != inputs.group_indices.tolist()  # pyright: ignore[reportUnknownMemberType]
        or protocol["frozen_model_hash"] != model_fingerprint(inputs.model)
    ):
        raise ValueError("Settling checkpoint, evaluation or frozen model identity differs")
    state = checkpoint["optimizer"]
    source_state = inputs.latent_optimizer
    values = checkpoint["table"]["weight"]
    if values.shape != (settings["groups"], 6) or not bool(torch.isfinite(values).all()):
        raise ValueError("Endpoint table is invalid")
    for name in ("exp_avg", "exp_avg_sq"):
        value = state["state"][0][name]
        if value.shape != values.shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Endpoint optimizer moments are invalid")
    if state["param_groups"] != source_state["param_groups"] or set(state["state"]) != {0}:
        raise ValueError("Settling changed SparseAdam semantics")
    if state["state"][0]["step"] != source_state["state"][0]["step"] + settings["steps"]:
        raise ValueError("Settling optimizer step differs")
    batch_size = inputs.bundle.batch_size
    batches = settings["groups"] // batch_size
    if settings["groups"] % batch_size or settings["steps"] % batches:
        raise ValueError("Moment formula requires equal complete epochs")
    moments = inherited_moments(
        source_state["state"][0]["exp_avg_sq"],
        state["state"][0]["exp_avg_sq"],
        settings["steps"] // batches,
        state["param_groups"][0]["betas"][1],
    )
    helpers = {str(Path(__file__).resolve()): sha(Path(__file__))}
    for directory in (
        "table_settling_control",
        "group_latent_control",
        "joint_acquisition",
        "encoder_gradient_control",
        "multiquery_control",
        "predictive_supervision_control",
        "parameter_sigreg_control",
    ):
        helpers.update(
            {
                str(path.resolve()): sha(path)
                for path in sorted(Path(__file__).parent.joinpath(directory).glob("*.py"))
            }
        )
    for name in ("group_latent_table_audit.py", "evaluate_context_variance.py"):
        path = Path(__file__).parent / name
        helpers[str(path.resolve())] = sha(path)
    evidence = {
        "source_hashes": paths,
        "source_fingerprint": source_fingerprint(),
        "helper_hashes": helpers,
        "seed": SEED,
        "fractions": FRACTIONS,
        "batch_size": batch_size,
        "verification": verification,
        "torch": str(torch.__version__),
    }
    identity = canonical_fingerprint(evidence)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with file_lock(output_dir / ".lock"), preserve_rng(torch.device("cpu")):
            if any(path.name != ".lock" for path in output_dir.iterdir()):
                raise FileExistsError("Audit requires a fresh output directory")
            record: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "scope": (
                    "Read-only TRAIN-system code use and local omission updates; no factor "
                    "acquisition claim. Reset clears both moments and global step; it does not "
                    "isolate second moments alone."
                ),
                "uncertainty": (
                    "Paired system bootstrap for permutation; paired B16 batch-cluster bootstrap "
                    "for next steps."
                ),
            }
            write_json(output_dir / "protocol.json", record)
            logger = WandbLogger(
                WandbConfig(
                    mode=wandb_mode,
                    project="sparse_causal_jepa",
                    name=f"teacher-history-{output_dir.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=record,
                dataset_identity=inputs.bundle.protocol["identity"],
                out_dir=output_dir,
            )
            write_json(output_dir / "tracking.json", logger.metadata)
            try:
                logger.log(0, scalar_metrics(moments["summary"], "history"))
                train = {
                    key: value[inputs.group_indices] for key, value in inputs.bundle.train.items()
                }
                raw = endpoint_measurements(
                    inputs.model,
                    checkpoint["table"]["weight"],
                    state,
                    train,
                    evaluation["new_episodes"],
                    ObjectiveConfig(**inputs.bundle.protocol["objective_config"]),
                    batch_size=batch_size,
                )
                result = {
                    "identity": identity,
                    "history": moments["summary"],
                    "measurements": summarize(raw, batch_size),
                    "tracking": logger.metadata,
                    "scope": record["scope"],
                }
                if any(sha(Path(name)) != expected for name, expected in paths.items()):
                    raise RuntimeError("Immutable source changed during audit")
                if (
                    any(sha(Path(name)) != expected for name, expected in helpers.items())
                    or source_fingerprint() != evidence["source_fingerprint"]
                ):
                    raise RuntimeError("Executed source code changed during audit")
                atomic_save(
                    output_dir / "raw.pt",
                    {"identity": identity, "moments": moments, "measurements": raw},
                )
                write_json(output_dir / "results.json", result)
                logger.log(1, scalar_metrics(result))
                for name in ("protocol.json", "results.json"):
                    logger.log_artifact(
                        output_dir / name, name=f"teacher-history-{identity[:12]}-{Path(name).stem}"
                    )
                logger.finish()
                write_json(
                    output_dir / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(output_dir / name)
                            for name in ("protocol.json", "results.json", "raw.pt")
                        },
                    },
                )
                print(
                    json.dumps(
                        {
                            "event": "complete",
                            "output_dir": str(output_dir),
                            "tracking": logger.metadata,
                        }
                    ),
                    flush=True,
                )
                return result
            except BaseException as error:
                write_json(output_dir / "failure.json", {"identity": identity, "error": str(error)})
                logger.log(1, {"diagnostic/failed": 1})
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run the fixed CPU audit with optional online W&B logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settling-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--verification", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
