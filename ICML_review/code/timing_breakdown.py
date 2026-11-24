"""Diagnose the first frozen-state adaptation batch and its preceding synthetic projection."""

import argparse
import json
import math
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as native
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.oracle import EvaluationResult, objective_config
from diagnostics.shared.sampling import GroupSampler
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models import model as model_module
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.loop import TrainingObjective
from scjepa.training.rng import preserve_rng

from .consensus_gradient import add_capped_gradient
from .single_acquisition import DATA, PARENT, PRIVATE
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .synthetic_loss_math import code_quality
from .timing_loss import sibling_loss

OUT = PRIVATE / "timing-breakdown-synthetic-s4"
FAILED = PRIVATE / "timing-synthetic-entire-s4"
COLD = PRIVATE / "single-acquisition-s4"
SPLITS = ("calibration", "validation")
SEED = 4


def scalar(value: Tensor | float) -> float | str:
    """Preserve overflow, NaN and finite values distinctly in standards-compliant JSON."""
    number = float(value)
    return number if math.isfinite(number) else str(number)


def describe(values: Sequence[Tensor]) -> dict[str, Any]:
    """Distinguish nonfinite gradient elements, FP32 cap overflow and global-clip overflow."""
    vectors = [x.detach() for x in values]
    norms = {
        "sum_squares_fp32": torch.stack([x.float().square().sum() for x in vectors]).sum().sqrt(),
        "sum_squares_fp64": torch.stack([x.double().square().sum() for x in vectors]).sum().sqrt(),
        "native_clip_norm": torch.nn.utils.get_total_norm(vectors),
    }
    return {k: scalar(v) for k, v in norms.items()} | {
        "nonfinite_elements": sum(int((~x.isfinite()).sum()) for x in vectors),
        "largest_absolute_element": scalar(torch.stack([x.abs().max() for x in vectors]).max()),
    }


def chart(codes: Tensor) -> dict[str, Any]:
    """Measure code support with a translation that preserves exactly constant coordinates."""
    rows = codes.detach().double().flatten(0, 1)
    centered = rows - rows[:1]
    centered = centered - centered.mean(0)
    diff = (rows.amax(0) - rows.amin(0)).abs()
    # A translated copy avoids nonzero variance from summing a repeated floating-point value.
    values = code_quality((codes.detach().double() - rows[0]).contiguous())
    return values | {
        "exact_constant_coordinates": int((diff == 0).sum()),
        "all_codes_exactly_equal": bool((diff == 0).all()),
        "exactly_identical_sibling_groups": int((codes == codes[:, :1]).flatten(1).all(1).sum()),
        "coordinate_spans": diff.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "centered_rms": float(centered.square().mean().sqrt()),
        "uncentered_rms": float(rows.square().mean().sqrt()),
        "shape": list(codes.shape),
    }


def batch_probe(
    model: SCJEPA,
    objective: TrainingObjective,
    batch: dict[str, Tensor],
    step: int,
    seed: int,
    shift: int | None,
    *,
    detail: bool = False,
) -> dict[str, Any]:
    """Use native forward/backward on one device with an optional forced donor offset."""
    if frozenset(batch) != native.INPUT_KEYS:
        raise ValueError("replay requires only observations and actions")
    device = batch["observations"].device
    params = tuple(model.parameter_encoder.parameters())
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if any(p.grad is not None for _, p in named):
        raise ValueError("expected freshly cleared gradients")
    selected: list[int] = []

    def choose(codes: Tensor, *, training: bool = False) -> Tensor:
        if not training or codes.shape[1] <= 2:
            raise ValueError("expected the training donor law for multiple sibling episodes")
        # Consume the draw even when intervening on its value to keep every other stream identical.
        original = int(torch.randint(1, codes.shape[1], (), device=codes.device))
        offset = original if shift is None else shift
        if not 0 < offset < codes.shape[1]:
            raise ValueError("invalid donor offset")
        selected.append(offset)
        return codes.roll(offset, dims=1)

    with (
        preserve_rng(device),
        patch.object(model_module, "cross_episode_parameters", choose),
    ):
        torch.default_generator.manual_seed((1 << 40) + seed + step)
        if device.type == "cuda":
            torch.cuda.manual_seed((1 << 40) + seed + step)
        loss, terms, output = native.forward_loss(model, batch, objective, forward_seed=seed + step)
        auxiliary = sibling_loss(output.parameters)
        forecast = terms["prediction"]
        native_grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True)
        forecast_grads = torch.autograd.grad(forecast, params, retain_graph=True)
        raw = torch.autograd.grad(auxiliary, params, retain_graph=True)
        stats = add_capped_gradient(forecast, auxiliary, params)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
    if len(selected) != 1:
        raise ValueError("unexpected number of donor draws")
    gp64 = torch.stack([x.double().square().sum() for x in forecast_grads]).sum().sqrt()
    gs64 = torch.stack([x.double().square().sum() for x in raw]).sum().sqrt()
    scale64 = torch.where(
        gs64 > 0, (gp64 / gs64.clamp_min(torch.finfo(torch.float64).tiny)).clamp_max(1), 0.0
    )
    guide = {id(p): g for p, g in zip(params, raw, strict=True)}
    actual: list[Tensor] = []
    uncapped: list[Tensor] = []
    stable: list[Tensor] = []
    for (_, p), base in zip(named, native_grads, strict=True):
        if p.grad is None:
            raise ValueError("incomplete native backward")
        actual.append(p.grad.detach())
        g = guide.get(id(p))
        stable.append(base if g is None else base + (g.double() * scale64).to(base.dtype))
        uncapped.append(base if g is None else base + g)
        p.grad = None
    return {
        "step": step,
        "donor_offset": selected[0],
        "native_loss": scalar(loss.detach()),
        "forecast": scalar(forecast.detach()),
        "sibling_loss": scalar(auxiliary.detach()),
        "code": chart(output.parameters.detach().cpu()),
        "cap_inputs": {
            "forecast_encoder": describe(forecast_grads),
            "sibling_encoder": describe(raw),
        },
        "cap_native_statistics": {k: scalar(v) for k, v in stats.items()},
        "scale_with_fp64_norms": scalar(scale64),
        **(
            {
                "by_parameter": {
                    name: {
                        "native": describe([base]),
                        "capped": describe([g]),
                        "capped_sha256": tensor_fingerprint(g),
                        "raw_sibling": describe([guide[id(p)]]) if id(p) in guide else None,
                    }
                    for (name, p), base, g in zip(named, native_grads, actual, strict=True)
                }
            }
            if detail
            else {}
        ),
        "global_gradients": {
            "native_only": describe(native_grads),
            "guided_native_cap": describe(actual),
            "guided_without_cap": describe(uncapped),
            "guided_fp64_cap": describe(stable),
        },
    }


def model_identity(state: dict[str, Any]) -> str:
    """Use the release fingerprint, including its non-tensor architecture metadata."""
    return canonical_fingerprint(
        {k: tensor_fingerprint(v) if isinstance(v, Tensor) else v for k, v in state.items()}
    )


def load(path: Path, *, weights_only: bool) -> dict[str, Any]:
    """Load a local checkpoint or archived sample without copying tensors to an accelerator."""
    return cast(
        dict[str, Any], torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True)
    )


def bound_failure(history: Path, checkpoint_step: int, log_every: int) -> dict[str, Any]:
    """Bound the historical failure step from durable progress, accounting for log cadence."""
    rows = [json.loads(row) for row in history.read_bytes().splitlines()]
    successful = [
        int(r["train/successful_updates"]) for r in rows if "train/successful_updates" in r
    ]
    lower = max([checkpoint_step, *successful]) + 1
    upper = (max(successful, default=0) // log_every + 1) * log_every
    if upper < lower:
        upper = ((lower - 1) // log_every + 1) * log_every
    return {
        "checkpoint_step": checkpoint_step,
        "last_logged_step": max(int(r["step"]) for r in rows),
        "last_logged_successful_update": max(successful, default=None),
        "earliest_possible_failed_update": lower,
        "latest_possible_failed_update": upper,
        "exact_failed_update_from_archive": lower if lower == upper else None,
    }


def heldout(directory: Path, record: dict[str, Any], digests: dict[str, str]) -> dict[str, Any]:
    """Rescore saved calibration and validation samples with the original fitted probe protocol."""
    samples: dict[str, EvaluationResult] = {}
    rows: dict[str, Any] = {}
    for split in SPLITS:
        for ext in ("pt", "json"):
            file = directory / f"{split}.{ext}"
            digest = sha(file)
            if digest != record["hashes"][file.name]:
                raise ValueError(f"changed outcome {file}")
            digests[str(file)] = digest
        raw = read_json(directory / f"{split}.json")["metrics"]
        saved = load(directory / f"{split}.pt", weights_only=True)
        samples[split] = EvaluationResult(raw, saved["samples"])
    for split, result in samples.items():
        measure = result.samples
        raw = result.metrics
        mse = float((measure["predictions"].double() - measure["targets"].double()).square().mean())
        r2 = native.code_metrics(samples["calibration"], result)["code_r2_min"]
        rows[split] = {
            "code": chart(measure["codes"]),
            "prediction": mse,
            "code_r2_min": r2,
            "state_mcc": raw["state_mcc"],
            "stored_prediction": raw["prediction"],
            "stored_code_r2_min": raw["code_r2_min"],
            "mse_matches": math.isclose(mse, raw["prediction"], rel_tol=2e-6, abs_tol=1e-10),
            "r2_matches": math.isclose(r2, raw["code_r2_min"], rel_tol=2e-6, abs_tol=1e-9),
        }
    return rows


def note(commit: str, archived: str, data_id: str) -> str:
    """Predeclare inputs, reconstruction limits and interpretations before inspecting gradients."""
    module = "ICML_review.code.timing_breakdown"
    command = f".venv/bin/python -m {module} {commit} {archived}"
    return f"""# First-batch adaptation diagnosis, synthetic seed four

Hypothesis: a failure of the projected initializer or single-precision gradient accumulation
can be distinguished from a failure introduced by the online sibling gradient. Compare the
same first TRAIN minibatch, frozen state encoders, initialized weights, objective and RNG
under native gradients, the recorded capped gradient, the uncapped gradient and a cap with
FP64 norm accumulation. No optimizer update. Inspect both the guided and native a0 projected
initializers. Enumerate donor shifts to check CPU/CUDA sampling differences. A CPU failure
across all shifts supports a mechanism at this first batch; it cannot date the original
GPU failure exactly between logged updates. If the native gradient fails before guidance on
the projected state, the proposed cap alone cannot explain the failure. If only guidance fails,
inspect its raw component values and FP32 versus FP64 norm accumulation. If neither reproduces,
leave the accelerator failure unresolved. Evaluate archived calibration and validation sample
receipts to locate loss of parameter retention, rank and forecast quality relative to native a0.
No update, test score or treatment selection occurs in this diagnostic.

Diagnostic checkout {commit}
Archived training checkout {archived}
Data {DATA}, identity {data_id}
Worker command from checkout:
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD:$PWD/src {command}
"""


def diagnose(commit: str, archived: str, *, declare: bool, gpu: bool = False) -> None:
    """Bind the failed run and its matched control, then write a single read-only diagnostic."""
    check_head(commit)
    manifests = read_json(FAILED / "stages.json")
    data = read_json(DATA / "completion.json")
    if (
        manifests["commit"],
        manifests["seed"],
        manifests["schedule"],
        manifests["data_identity"],
    ) != (archived, SEED, "entire", data["identity"]):
        raise ValueError("failed run or data do not match")
    original_note = (FAILED / "NOTE.md").read_text()
    if (
        f"- Checkout `{archived}`" not in original_note
        or sha(DATA / "completion.json") not in original_note
        or data["hashes"]["train.pt"] not in original_note
    ):
        raise ValueError("archived prospective declaration differs from the replay data/source")
    if gpu:
        from .timing_breakdown_gpu import GPU_OUT, gpu_note

        root = GPU_OUT
        text = gpu_note(commit, archived, data["identity"])
    else:
        root = OUT
        text = note(commit, archived, data["identity"])
    path = root / "NOTE.md"
    if declare:
        root.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x") as stream:
                stream.write(text)
    if path.read_text() != text:
        raise ValueError("incorrect diagnostic declaration")
    if declare:
        return
    with file_lock(root / ".lock"):
        if (root / "metrics.json").exists():
            prior = read_json(root / "metrics.json")
            if (prior["commit"], prior["training_commit"], prior["source_sha256"][str(path)]) != (
                commit,
                archived,
                sha(path),
            ):
                raise ValueError("existing diagnostic differs")
            return
        subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                "--quiet",
                archived,
                "--",
                "src",
                "diagnostics/shared",
                "diagnostics/masked_mlp_learned",
                "diagnostics/masked_mlp_oracle",
                "ICML_review/code/timing_synthetic.py",
                "ICML_review/code/timing_loss.py",
                "ICML_review/code/consensus_gradient.py",
            ],
            check=True,
        )
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if gpu:
            from .timing_breakdown_gpu import original_gpu

            out = original_gpu(commit, archived, data, manifests)
        else:
            out = analyze(data, manifests, commit)
        check_head(commit)
        write_json(
            root / "metrics.json",
            out
            | {"commit": commit, "training_commit": archived, "data_identity": data["identity"]},
        )


def analyze(data: dict[str, Any], stages: dict[str, Any], commit: str) -> dict[str, Any]:
    """Perform the stage and first-batch calculations from authenticated upstream outputs."""
    adapt = FAILED / "adapt/results"
    cold = COLD / "adapt/results"
    cfg = read_json(adapt / "config.json")
    raw_cfg = cfg["config"]
    checkpoints = read_json(COLD / "metrics.json")["checkpoint_sha256"]
    original = read_json(PARENT / f"selection-s{SEED}.json")
    a0 = next(x for x in original["candidates"] if x["acquisition_seed"] == 1000 + SEED)
    digests: dict[str, str] = {}

    def bind(p: Path, expected: str | None = None) -> str:
        digest = sha(p)
        if expected is not None and expected != digest:
            raise ValueError(f"changed input {p}")
        digests[str(p)] = digest
        return digest

    for p in (
        OUT / "NOTE.md",
        FAILED / "NOTE.md",
        FAILED / "stages.json",
        COLD / "metrics.json",
        PARENT / f"selection-s{SEED}.json",
        DATA / "completion.json",
        adapt / "config.json",
        adapt / "startup.json",
        adapt / "history.jsonl",
    ):
        bind(p)
    if cfg["data_identity"] != data["identity"] or cfg["code_identity"] != native.code_identity():
        raise ValueError("adaptation source/data differs")
    for filename, expected in cfg["scientific_config"]["sibling_guidance"]["source_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("invalid stored source file")
        bind(Path(__file__).with_name(filename), expected)
    stage_rows: dict[str, Any] = {}
    for name, directory in (
        ("guided_acquire", FAILED / "acquire/results"),
        ("guided_project", FAILED / "project/results"),
        ("native_acquire", PARENT / f"acquisition-0-s{SEED}/results"),
        ("native_project", COLD / "project/results"),
    ):
        check_head(commit)
        policy, stage = name.split("_")
        expected = (
            stages["stages"][stage]["completion_sha256"]
            if policy == "guided"
            else a0["completion_sha256"]
            if stage == "acquire"
            else None
        )
        bind(directory / "completion.json", expected)
        record = read_json(directory / "completion.json")
        if (record["seed"], record["data_identity"], record["code_identity"]) != (
            SEED,
            data["identity"],
            cfg["code_identity"],
        ):
            raise ValueError(f"unmatched stage {directory}")
        if (
            stage == "project"
            and policy == "native"
            and record["hashes"]["last.pt"] != checkpoints["project"]
        ):
            raise ValueError("wrong native projection")
        if stage == "project":
            bind(directory / "projection.json", record["hashes"]["projection.json"])
        stage_rows[name] = heldout(directory, record, digests)
    if (
        raw_cfg["seed"] != SEED
        or raw_cfg["condition"] != "dense"
        or cfg["state_source"] is not None
        or cfg["adaptation_source"]["checkpoint_sha256"]
        != stages["stages"]["project"]["checkpoint_sha256"]
    ):
        raise ValueError("incorrect adaptation specification or source")
    for name in ("initial.pt", "last.pt"):
        bind(adapt / name, cfg["initial_checkpoint_sha256"] if name == "initial.pt" else None)
    guided = load(adapt / "last.pt", weights_only=False)
    if (
        guided["identity"] != cfg["identity"]
        or guided["initial_checkpoint_sha256"] != cfg["initial_checkpoint_sha256"]
        or guided["step"] != 0
    ):
        raise ValueError("expected registered initial adaptation snapshot")
    if guided["optimizer"]["state"]:
        raise ValueError("step-zero snapshot contains updated optimizer buffers")
    initial = load(adapt / "initial.pt", weights_only=False)
    if (
        not model_identity(initial["model"])
        == model_identity(guided["model"])
        == cfg["initial_model_identity"]
    ):
        raise ValueError("step-zero checkpoint weights differ from initial.pt")
    native_cp = read_json(cold / "completion.json")
    bind(cold / "completion.json")
    if native_cp["hashes"]["last.pt"] != checkpoints["adapt"]:
        raise ValueError("wrong native adaptation")
    bind(cold / "initial.pt", native_cp["hashes"]["initial.pt"])
    native_initial = load(cold / "initial.pt", weights_only=False)
    bind(cold / "history.jsonl", native_cp["hashes"]["history.jsonl"])
    for d, cp, expected in (
        (
            FAILED / "project/results",
            guided["model"],
            cfg["adaptation_source"]["checkpoint_sha256"],
        ),
        (COLD / "project/results", native_initial["model"], checkpoints["project"]),
    ):
        bind(d / "last.pt", expected)
        projected = load(d / "last.pt", weights_only=False)["model"]
        if model_identity(projected) != model_identity(cp):
            raise ValueError("adaptation did not copy projected weights exactly")
    bind(DATA / "train.pt", data["hashes"]["train.pt"])
    # Memory mapping dereferences only the two TRAIN input arrays selected by the native forward.
    train = load(DATA / "train.pt", weights_only=True)["tensors"]
    size = len(train["observations"])
    sampler = GroupSampler(size, raw_cfg["batch_size"], SEED)
    sampler.load_state_dict(guided["sampler"])
    indices = sampler.next()
    if not torch.equal(indices, GroupSampler(size, raw_cfg["batch_size"], SEED).next()):
        raise ValueError("native first batch and restored saved sampler differ")
    batch = {k: cast(Tensor, train[k])[indices] for k in native.INPUT_KEYS}
    out: dict[str, Any] = {}
    for name, saved in (("guided", guided), ("native", native_initial)):
        check_head(commit)
        model = native.build_control("dense", SEED, 6, predictor_activation="gelu")
        model.load_state_dict(saved["model"], strict=True)
        model.online_encoder.requires_grad_(False)
        model.train()
        objective = TrainingObjective(objective_config("dense", 0.0), GECOConfig())
        objective.load_state_dict(guided["objective"], strict=True)
        objective.train()
        with preserve_rng(torch.device("cpu")):
            torch.set_rng_state(guided["torch_rng"])
            random.setstate(guided["python_rng"])
            np.random.set_state(guided["numpy_rng"])  # noqa: NPY002
            runs = [
                batch_probe(model, objective, batch, 1, SEED, s)
                for s in (None, *range(1, batch["observations"].shape[1]))
            ]
        out[name] = {"cpu_seeded": runs[0], "all_donor_offsets": runs[1:]}
    attempts = [
        p
        for p in sorted((FAILED / "attempts").iterdir())
        if (p / "trainer_exit_code").exists() and int((p / "trainer_exit_code").read_text()) != 0
    ]
    if len(attempts) != 1:
        raise ValueError("expected one archived failed training attempt")
    log = attempts[0] / "log.txt"
    for f in (log, attempts[0] / "trainer_exit_code", attempts[0] / "exit_code"):
        bind(f)
    trace = log.read_text(errors="replace")
    failure = (
        "clip_grad_norm_" in trace and "total norm of order" in trace and "is non-finite" in trace
    )
    if not failure:
        raise ValueError("archived traceback differs")
    bind(attempts[0] / "declaration.sha256")
    if (attempts[0] / "declaration.sha256").read_text().split()[0] != sha(FAILED / "NOTE.md"):
        raise ValueError("original declaration changed since launch")
    hist = bound_failure(adapt / "history.jsonl", int(guided["step"]), raw_cfg["log_every"])
    cold_history = [json.loads(x) for x in (cold / "history.jsonl").read_text().splitlines()]
    first = next(row for row in cold_history if "train/successful_updates" in row)
    return {
        "source_sha256": digests,
        "heldout": stage_rows,
        "replay": out,
        "history": hist
        | {
            "failed_at": "global_gradient_clip",
            "attempt_log": str(log),
            "native_first_logged_update": {
                k: first[k]
                for k in (
                    "step",
                    "loss/prediction",
                    "train/gradient_norm",
                    "train/successful_updates",
                )
            },
        },
        "reproduction": {
            "device": "cpu",
            "torch": str(torch.__version__),
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "train_group_indices": [int(i) for i in indices],
            "batch_sha256": {k: tensor_fingerprint(v) for k, v in batch.items()},
            "host_rng_restored": True,
            "model_weights_equal_projected": True,
            "empty_optimizer_step_zero": True,
            "archived_cuda_rng_sha256": tensor_fingerprint(guided["cuda_rng"]),
            "cuda_rng_applied": False,
            "forward_seed": SEED + 1,
            "objective_seed": (1 << 40) + SEED + 1,
            "note": (
                "CPU and CUDA kernels/RNG implementations may differ. Every legal donor shift "
                "is examined at the same first batch. No optimizer update is performed."
            ),
        },
    }


def main() -> None:
    """Run at the diagnostic checkout using the separately pinned failed-run revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("training_commit")
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    diagnose(args.commit, args.training_commit, declare=args.declare, gpu=args.gpu)


if __name__ == "__main__":
    main()
