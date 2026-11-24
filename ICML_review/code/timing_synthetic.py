"""Train one prescribed cold synthetic head under either sibling-guidance schedule."""

import argparse
import json
import shutil
import subprocess
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.sampling import GroupSampler
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective

from . import consensus_gradient, timing_loss
from .single_acquisition import DATA, PARENT, PRIVATE
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import _idle_gpu as check_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import declared

SEED = 4
STAGES = ("joint", "acquire", "project", "adapt", "sparse", "endpoint")


@torch.no_grad()
def adapted_teacher(directory: Path, device: torch.device) -> tuple[Tensor, Tensor]:
    """Use the adapted dense encoders on all TRAIN episodes in natural system order."""
    cfg = read_json(directory / "config.json")["config"]
    if cfg["condition"] != "dense" or cfg["adapt_from"] is None:
        raise ValueError("fixed teacher must be the dense adaptation immediately before sparse")
    data = runner.load_data(Path(cfg["data_dir"]))[0]["train"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = runner.build_control("dense", cfg["seed"], 6, predictor_activation="gelu").to(device)
    saved = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    outputs: list[Tensor] = []
    length = model.context_len
    for start in range(0, len(data), cfg["batch_size"]):
        stop = min(start + cfg["batch_size"], len(data))
        x, a = (
            data.tensors[key][start:stop].to(device).flatten(0, 1)
            for key in ("observations", "actions")
        )
        z = model.online_encoder.encode_sequence(x)
        theta = model.parameter_encoder(z[:, :length], a[:, : length - 1])
        outputs.append(theta.reshape(stop - start, -1, theta.shape[-1]))
    bank = torch.cat(outputs)
    return bank, consensus_gradient.whitening(bank)


@contextmanager
def guidance(teacher: Path | None, device: str) -> Generator[None, None, None]:
    """Attach online pre-sparse or fixed sparse guidance to the native gradient step."""
    fixed = None if teacher is None else adapted_teacher(teacher, torch.device(device))
    native_forward = runner.forward_loss
    native_spec = runner._scientific_config  # pyright: ignore[reportPrivateUsage]
    indices: Tensor | None = None
    recipe = {
        "teacher": "online_minibatch" if teacher is None else "frozen_adapted",
        "target": "leave-one-episode-out TRAIN mean",
        "metric": "regularized detached minibatch covariance"
        if teacher is None
        else "fixed pooled TRAIN covariance",
        "checkpoint_sha256": None if teacher is None else sha(teacher / "last.pt"),
        "source_sha256": {
            p.name: sha(p)
            for p in (Path(__file__), Path(timing_loss.__file__), Path(consensus_gradient.__file__))
        },
    }

    class Sampled(GroupSampler):
        def next(self) -> Tensor:
            nonlocal indices
            indices = super().next()
            return indices

    def scientific(config: runner.TrainConfig) -> dict[str, Any]:
        return native_spec(config) | {
            "parameter_training": "forecast plus capped sibling gradient on parameter encoder",
            "sibling_guidance": recipe,
        }

    def forward(
        model: SCJEPA,
        batch: Mapping[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        loss, terms, output = native_forward(model, batch, objective, forward_seed=forward_seed)
        if model.training and torch.is_grad_enabled():
            params = tuple(model.parameter_encoder.parameters())
            if fixed is None:
                stats = timing_loss.add_sibling_gradient(
                    terms["prediction"], output.parameters, params
                )
            else:
                if indices is None or len(indices) != output.parameters.shape[0]:
                    raise RuntimeError("teacher and student must use the same TRAIN groups")
                bank, matrix = fixed
                aux = consensus_gradient.anchor(output.parameters, bank[indices], matrix, "other")
                stats = consensus_gradient.add_capped_gradient(terms["prediction"], aux, params)
            terms.update(stats)
        return loss, terms, output

    with (
        patch.object(runner, "forward_loss", forward),
        patch.object(runner, "GroupSampler", Sampled),
        patch.object(runner, "_scientific_config", scientific),
    ):
        yield


@contextmanager
def original_start(directory: Path) -> Generator[None, None, None]:
    """Keep the actual joint step-zero tensors including the data-dependent shared head."""
    record = runner.verify_completion(directory, kind="train", require_current_source=True)
    if record["state_source"] is not None or record["adaptation_source"] is not None:
        raise ValueError("expected an original joint checkpoint")
    saved = cast(
        dict[str, Any], torch.load(directory / "initial.pt", map_location="cpu", weights_only=False)
    )

    def restore(model: SCJEPA, _data: object, _device: torch.device) -> dict[str, Any]:
        model.load_state_dict(saved["model"], strict=True)
        return cast(dict[str, Any], saved["initialization"])

    with patch.object(runner, "initialize_shared_output", restore):
        yield


def _config(source: Path, dest: Path) -> dict[str, Any]:
    cfg = read_json(source / "config.json")["config"]
    return {f.name: cfg[f.name] for f in fields(runner.TrainConfig) if f.init} | {
        "output_dir": str(dest),
        "data_dir": str(DATA),
        "wandb_group": "icml-review-cold-sibling-timing",
    }


def _original(seed: int) -> dict[str, Path]:
    cold = PRIVATE / f"single-acquisition-s{seed}"
    return {
        "joint": PARENT / f"states-s{seed}/results",
        "acquire": PARENT / f"acquisition-0-s{seed}/results",
        "project": cold / "project/results",
        "adapt": cold / "adapt/results",
    }


def _verify(directory: Path, data_id: str, stage: str) -> dict[str, Any]:
    record = runner.verify_completion(
        directory, kind="projection" if stage == "project" else "train", require_current_source=True
    )
    if (
        record["seed"] != SEED
        or record["data_identity"] != data_id
        or record["predictor_activation"] != "gelu"
    ):
        raise ValueError(f"unmatched {stage} checkpoint {directory}")
    budget = {
        "joint": 100000,
        "acquire": 100000,
        "project": 100000,
        "adapt": 50000,
        "sparse": 300000,
        "endpoint": 350000,
    }[stage]
    if record["steps"] != budget or (stage != "project" and record["successful_updates"] != budget):
        raise ValueError(f"unmatched update budget at {directory}")
    return record


def reconcile_history(directory: Path) -> None:
    """Archive interrupted log tails before replaying updates since the last checkpoint."""
    history = directory / "history.jsonl"
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    step = int(checkpoint["step"])
    rows = history.read_bytes().splitlines(keepends=True)
    keep: list[bytes] = []
    for index, row in enumerate(rows):
        try:
            logged = json.loads(row)
        except ValueError:
            if index != len(rows) - 1 or row.endswith(b"\n"):
                raise
            break
        if int(logged["step"]) <= step:
            keep.append(row)
    if len(keep) != len(rows):
        shutil.copy2(history, directory.parent / f"history-before-resume-{time.time_ns()}.jsonl")
        temp = directory / "history.resume"
        temp.write_bytes(b"".join(keep))
        temp.replace(history)


def run(out: Path, schedule: str, commit: str) -> None:
    """Execute the mandatory first-head path at fixed HEAD, with stage receipts and resume."""
    if schedule not in {"entire", "sparse"}:
        raise ValueError("schedule must be entire or sparse")
    if not out.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("outputs must live in the review folder")
    note = (out / "NOTE.md").read_text()
    check_head(commit)
    if commit not in note:
        raise ValueError("declaration must identify the running checkout")
    data_id = read_json(DATA / "completion.json")["identity"]
    original = _original(SEED)
    cold = PRIVATE / f"single-acquisition-s{SEED}"
    controls = read_json(cold / "metrics.json")
    history = read_json(PARENT / f"selection-s{SEED}.json")
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}
    for stage in STAGES:
        check_head(commit)
        folder = out / stage
        declared(
            folder / "NOTE.md",
            note + f"\nSynthetic stage {stage}, schedule {schedule}, seed {SEED}.\n",
        )
        reuse = schedule == "sparse" and stage in original
        dest = original[stage] if reuse else folder / "results"
        if not reuse and (
            not (dest / "completion.json").is_file()
            or (stage == "endpoint" and not (dest / "continuation-completion.json").is_file())
        ):
            resume = (dest / "last.pt").is_file()
            if dest.exists() and (
                not resume
                or stage == "project"
                or (stage == "endpoint" and not (dest / "continuation.json").is_file())
            ):
                dest.rename(folder / f"incomplete-{time.time_ns()}")
                resume = False
            if resume and not (dest / "completion.json").is_file():
                reconcile_history(dest)
            if stage == "project":
                runner.project(
                    runner.ProjectConfig(str(dest), str(paths["acquire"]), str(DATA), device="cuda")
                )
            elif stage in ("joint", "acquire", "adapt"):
                template = (
                    original[stage] if stage != "adapt" else PARENT / f"dense-s{SEED}/results"
                )
                args = _config(template, dest)
                if stage == "acquire":
                    args["states_from"] = str(paths["joint"])
                    if args["acquisition_seed"] != 1000 + SEED:
                        raise ValueError("candidate zero initialization mismatch")
                if stage == "adapt":
                    args["adapt_from"] = str(paths["project"])
                with (
                    original_start(original["joint"]) if stage == "joint" else nullcontext(),
                    guidance(None, "cuda"),
                ):
                    runner.train(runner.TrainConfig(**args), resume=resume)
            else:
                with guidance(paths["adapt"], "cuda"):
                    if stage == "sparse":
                        args = _config(PARENT / f"full-s{SEED}/results", dest)
                        args.update(
                            init_dir=str(paths["adapt"]),
                            calibration_dir=str(PARENT / "calibration"),
                        )
                        runner.train(BudgetConfig(**args), resume=resume)
                    else:
                        continue_training(
                            paths["sparse"],
                            dest,
                            additional_steps=50000,
                            device="cuda",
                            threads=1,
                            resume=resume,
                        )
        record = _verify(dest, data_id, stage)
        if (
            reuse
            and stage in {"project", "adapt"}
            and record["hashes"]["last.pt"] != controls["checkpoint_sha256"][stage]
        ):
            raise ValueError("cold prefix differs from its recorded comparator")
        if (
            stage == "joint"
            and record["initial_model_identity"]
            != read_json(original["joint"] / "config.json")["initial_model_identity"]
        ):
            raise ValueError("joint initial tensors differ from the reported run")
        if stage == "acquire":
            archived = next(
                c for c in history["candidates"] if c["acquisition_seed"] == 1000 + SEED
            )
            if reuse and sha(dest / "completion.json") != archived["completion_sha256"]:
                raise ValueError("cold acquisition differs from historical first candidate")
            if record["state_source"]["checkpoint_sha256"] != records["joint"]["hashes"]["last.pt"]:
                raise ValueError(
                    "frozen acquisition states do not come from this run's joint stage"
                )
            admission = select(
                [dest], window=history["window"], maximum_loss=history["maximum_loss"]
            )
            if admission["selected"]["acquisition_seed"] != 1000 + SEED:
                raise ValueError("incorrect cold acquisition seed")
            write_json(out / "training-score.json", admission | {"controls_execution": False})
        if stage in {"project", "adapt", "sparse"}:
            key, parent = {
                "project": ("source", "acquire"),
                "adapt": ("adaptation_source", "project"),
                "sparse": ("dense_endpoint", "adapt"),
            }[stage]
            if record[key]["checkpoint_sha256"] != records[parent]["hashes"]["last.pt"]:
                raise ValueError(f"{stage} does not descend from {parent}")
        if stage == "endpoint":
            receipt = read_json(dest / "continuation-completion.json")
            if (
                receipt["parent_checkpoint_sha256"] != records["sparse"]["hashes"]["last.pt"]
                or receipt["checkpoint_sha256"] != record["hashes"]["last.pt"]
            ):
                raise ValueError("different continuation parent or endpoint")
        check_head(commit)
        paths[stage], records[stage] = dest, record
        manifest[stage] = {
            "directory": str(dest),
            "reused": reuse,
            "checkpoint_sha256": record["hashes"]["last.pt"],
            "completion_sha256": sha(dest / "completion.json"),
            "steps": record["steps"],
        }
        write_json(
            out / "stages.json",
            {
                "commit": commit,
                "seed": SEED,
                "schedule": schedule,
                "data_identity": data_id,
                "stages": manifest,
            },
        )
    assess(out, schedule, commit)


def assess(out: Path, schedule: str, commit: str) -> None:
    """Measure paired held-out systems against both first-head and reported full endpoints."""
    from . import synthetic_consensus as previous

    check_head(commit)
    declared(
        out / "assess/NOTE.md",
        (out / "NOTE.md").read_text()
        + "\nFixed synthetic calibration and validation assessment.\n",
    )
    stages = read_json(out / "stages.json")
    if stages["commit"] != commit or stages["schedule"] != schedule:
        raise ValueError("endpoints must belong to the declared run")
    source_parent, reported, late = previous.roots(SEED)
    reported_inputs = previous.control_inputs(source_parent, reported)
    cold = PRIVATE / f"single-acquisition-s{SEED}"
    paths = {
        "treatment": out / "endpoint/results",
        "cold_native": cold / "full-350k/results",
        "reported": reported,
        "late_sibling": late / "other/results",
    }
    expected = {
        "treatment": stages["stages"]["endpoint"]["checkpoint_sha256"],
        "cold_native": read_json(cold / "metrics.json")["checkpoint_sha256"]["full-350k"],
        "reported": reported_inputs["native_sha256"],
        "late_sibling": read_json(late / "metrics.json")["checkpoint_sha256"]["other"],
    }
    records: dict[str, dict[str, Any]] = {}
    codes: dict[str, dict[str, Tensor]] = {}
    errors: dict[str, dict[str, Tensor]] = {}
    for key, path in paths.items():
        cp = _verify(path, stages["data_identity"], "endpoint")
        if cp["hashes"]["last.pt"] != expected[key] or cp["steps"] != 350000:
            raise ValueError("unmatched endpoint for assessment")
        records[key], codes[key], errors[key] = previous.endpoint(path, SEED)
        validation = read_json(path / "validation.json")["metrics"]
        records[key]["validation"].update(
            {
                k: validation[k]
                for k in (
                    "normalized_prediction_mse",
                    "block_purity",
                    "code_mcc",
                    "code_pair_purity",
                    "forecast/code_use_ratio",
                )
            }
        )
    checks: dict[str, bool] = {}
    intervals: dict[str, Any] = {}
    for split, seed in previous.SPLITS.items():
        student = records["treatment"][split]
        checks[f"{split}_full_rank"] = (
            student["total_numerical_rank"] == 6 and student["sixth_share"] > 0
        )
        for name in ("reported", "cold_native"):
            control = records[name][split]
            checks[f"{split}_shared_vs_{name}"] = (
                student["sixth_share"] >= 0.8 * control["sixth_share"]
            )
            checks[f"{split}_worst_vs_{name}"] = (
                student["worst_within_fraction"] is not None
                and control["worst_within_fraction"] is not None
                and student["worst_within_fraction"] <= 0.8 * control["worst_within_fraction"]
            )
            if split == "validation":
                checks.update(
                    {
                        f"{metric}_vs_{name}": bool(verdict)
                        for metric, verdict in {
                            "forecast": student["prediction"] <= 1.1 * control["prediction"],
                            "retention": student["code_r2_min"]
                            >= max(0.8, control["code_r2_min"] - 0.02),
                            "state_mcc": student["state_mcc"] >= control["state_mcc"] - 0.01,
                            "total_shd": student["graph_shd"] <= control["graph_shd"],
                        }.items()
                    }
                )
        intervals[split] = {}
        for comparator in ("cold_native", "late_sibling"):
            aliases = {"native": "reported", "same": comparator, "other": "treatment"}
            try:
                values = consensus_gradient.context_intervals(
                    {k: codes[v][split] for k, v in aliases.items()},
                    {k: errors[v][split] for k, v in aliases.items()},
                    seed,
                )
                for key in ("native", "same"):
                    intervals[split][f"treatment_minus_{aliases[key]}"] = values[
                        f"other_minus_{key}"
                    ]
            except ValueError as exc:
                intervals[split][f"treatment_minus_{comparator}"] = {"undefined": str(exc)}
                if comparator == "cold_native":
                    checks[f"{split}_bootstrap_defined"] = False
    check_head(commit)
    write_json(
        out / "metrics.json",
        {
            "commit": commit,
            "seed": SEED,
            "schedule": schedule,
            "data_identity": stages["data_identity"],
            "records": records,
            "checks": checks,
            "passed": all(checks.values()),
            "paired_intervals": intervals,
            "bootstrap": {"draws": 2000, "seeds": previous.SPLITS, "unit": "whole systems"},
            "checkpoint_sha256": expected,
            "directories": {name: str(path) for name, path in paths.items()},
            "stage_receipt_sha256": sha(out / "stages.json"),
        },
    )


def main() -> None:
    """Launch a predeclared single-head run from the server checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("schedule", choices=("entire", "sparse"))
    args = parser.parse_args()
    out = PRIVATE / f"timing-synthetic-{args.schedule}-s{SEED}"
    check_head(args.commit)
    if not (out / "NOTE.md").is_file():
        raise ValueError("missing prospective experiment declaration")
    with file_lock(out / ".run.lock"):
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
        )
        check_gpu()
        run(out, args.schedule, args.commit)


if __name__ == "__main__":
    main()
