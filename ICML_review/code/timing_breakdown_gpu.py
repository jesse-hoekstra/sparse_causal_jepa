"""Bounded reproduction of the synthetic adaptation clip on its original physical worker."""

import os
import random
import re
import socket
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as native
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.oracle import objective_config
from diagnostics.shared.sampling import GroupSampler
from scjepa.data.datasets import tensor_fingerprint
from scjepa.experiments.storage import file_lock
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.loop import TrainingObjective

from .single_acquisition import DATA, PRIVATE
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .timing_breakdown import (
    FAILED,
    SEED,
    batch_probe,
    bound_failure,
    describe,
    load,
    model_identity,
    scalar,
)
from .timing_loss import add_sibling_gradient

GPU_OUT = PRIVATE / "timing-breakdown-synthetic-gpu-s4"


def gpu_note(commit: str, archived: str, data_id: str) -> str:
    """Declare the exact replay bounds, failure signature and numerical controls in advance."""
    return f"""# Accelerator replay of the failed synthetic adaptation

Hypothesis: the failed global gradient clip can be reproduced from the saved step-zero
adaptation state on the original worker, visible GPU ordinal, architecture and software.
Replay the native optimizer on a disposable in-memory copy. Preserve sampler, CUDA and host
RNG, per-step forward seeds, frozen encoders, native dense loss, capped sibling gradient,
clipping and AdamW. Stop at the first failing update or the end of the archive's unlogged
interval, capped at one logging interval. Compare native-only and capped/uncapped gradients
on the failing pre-update state. Require immutable-parent, TRAIN-input and source identities.
If the native-only norm also overflows, guidance is unnecessary for the immediate failure
on those weights. If only the capped backward fails, distinguish nonfinite elements from
norm overflow and test FP64 norm accumulation in the same gradient arithmetic. A finite
replay through the bound leaves the accelerator failure unreproduced. The experiment writes
only a diagnostic receipt. No held-out data, labels or scientific checkpoints are produced.

Diagnostic checkout {commit}
Archived training checkout {archived}
Data {DATA}, identity {data_id}
Worker command from checkout:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m \\
        ICML_review.code.timing_breakdown {commit} {archived} --gpu
"""


def replay_window(
    model: SCJEPA,
    saved: dict[str, Any],
    train: dict[str, Tensor],
    cfg: dict[str, Any],
    bound: int,
    check: Callable[[], None],
) -> dict[str, Any]:
    """Re-execute bounded native steps and record the unreduced gradients on a failing step."""
    device = next(model.parameters()).device
    seed = cfg["seed"]
    first = int(saved["step"]) + 1
    if not first <= bound < first + cfg["log_every"]:
        raise ValueError("replay must fit inside one archived logging interval")
    if frozenset(train) != native.INPUT_KEYS:
        raise ValueError("expected only native observations and actions")
    model.online_encoder.requires_grad_(False)
    model.train()
    objective = TrainingObjective(objective_config("dense", 0.0), GECOConfig()).to(device)
    objective.load_state_dict(saved["objective"], strict=True)
    objective.train()
    named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    optim = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    optim.load_state_dict(saved["optimizer"])
    if saved["step"] != 0 or optim.state:
        raise ValueError("first-batch replay requires fresh Adam state")
    sampler = GroupSampler(len(train["observations"]), cfg["batch_size"], seed)
    sampler.load_state_dict(saved["sampler"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])  # noqa: NPY002
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
    successes: list[dict[str, Any]] = []
    for step in range(first, bound + 1):
        check()
        index = sampler.next()
        batch = {k: v[index].to(device) for k, v in train.items()}
        optim.zero_grad(set_to_none=True)
        torch.manual_seed((1 << 40) + seed + step)  # pyright: ignore[reportUnknownMemberType]
        loss, terms, output = native.forward_loss(model, batch, objective, forward_seed=seed + step)
        add_sibling_gradient(
            terms["prediction"], output.parameters, tuple(model.parameter_encoder.parameters())
        )
        if not bool(torch.isfinite(loss)):
            return {"status": "nonfinite_forward", "failed_step": step, "successes": successes}
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        try:
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"], error_if_nonfinite=True)
        except RuntimeError as exc:
            if "non-finite" not in str(exc) or "total norm" not in str(exc):
                raise
            gradients = {name: p.grad.detach() for name, p in named if p.grad is not None}
            actual = {k: tensor_fingerprint(v) for k, v in gradients.items()}
            actual_norm = describe(list(gradients.values()))
            optim.zero_grad(set_to_none=True)
            detail = batch_probe(model, objective, batch, step, seed, None, detail=True)
            matched = {
                k: item["capped_sha256"] == actual[k] for k, item in detail["by_parameter"].items()
            }
            return {
                "status": "global_clip_reproduced",
                "failed_step": step,
                "successes": successes,
                "failing_batch_sha256": {k: tensor_fingerprint(v) for k, v in batch.items()},
                "failing_group_indices": [int(i) for i in index],
                "actual_clip_input": actual_norm,
                "actual_by_parameter_sha256": actual,
                "diagnostic_matches_actual_by_parameter": matched,
                "all_counterfactual_replayed_gradients_match": all(matched.values()),
                "failure_gradient_breakdown": detail,
            }
        optim.step()  # pyright: ignore[reportUnknownMemberType]
        successes.append(
            {"step": step, "native_loss": scalar(loss.detach()), "gradient_norm": scalar(norm)}
        )
    return {"status": "finite_through_archive_bound", "failed_step": None, "successes": successes}


def original_gpu(
    commit: str, archived: str, data: dict[str, Any], stages: dict[str, Any]
) -> dict[str, Any]:
    """Check the original accelerator assignment before permitting the bounded replay."""
    folder = FAILED / "adapt/results"
    cfg = read_json(folder / "config.json")
    attempts = [
        p
        for p in (FAILED / "attempts").iterdir()
        if (p / "trainer_exit_code").is_file()
        and (p / "trainer_exit_code").read_text().strip() != "0"
    ]
    if len(attempts) != 1:
        raise ValueError("expected exactly one archived failed attempt")
    log = attempts[0] / "log.txt"
    text = log.read_text(errors="replace")
    label = re.search(r"host=([A-Za-z0-9.-]+) head=([a-f0-9]{40})", text.splitlines()[0])
    host = socket.gethostname().split(".")[0]
    if (
        not label
        or label.groups() != (host, archived)
        or "total norm of order" not in text
        or "is non-finite" not in text
    ):
        raise ValueError("requires original failure, worker and revision")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("requires physical GPU zero as the only visible accelerator")
    inputs = {
        p: sha(p)
        for p in (
            GPU_OUT / "NOTE.md",
            FAILED / "NOTE.md",
            FAILED / "stages.json",
            log,
            attempts[0] / "declaration.sha256",
            folder / "config.json",
            folder / "history.jsonl",
            folder / "initial.pt",
            folder / "last.pt",
            FAILED / "project/results/last.pt",
            DATA / "completion.json",
            DATA / "train.pt",
        )
    }
    if (
        (attempts[0] / "declaration.sha256").read_text().split()[0] != inputs[FAILED / "NOTE.md"]
        or inputs[DATA / "train.pt"] != data["hashes"]["train.pt"]
        or cfg["data_identity"] != data["identity"]
        or cfg["code_identity"] != native.code_identity()
        or cfg["config"]["seed"] != SEED
        or cfg["config"]["adapt_from"] is None
        or cfg["config"]["condition"] != "dense"
        or inputs[folder / "initial.pt"] != cfg["initial_checkpoint_sha256"]
        or inputs[FAILED / "project/results/last.pt"]
        != stages["stages"]["project"]["checkpoint_sha256"]
        or inputs[FAILED / "project/results/last.pt"]
        != cfg["adaptation_source"]["checkpoint_sha256"]
    ):
        raise ValueError("archived input identity mismatch")
    saved = load(folder / "last.pt", weights_only=False)
    if (
        saved["identity"] != cfg["identity"]
        or saved["initial_checkpoint_sha256"] != cfg["initial_checkpoint_sha256"]
    ):
        raise ValueError("incorrect last adaptation snapshot")
    if any(
        model_identity(cp["model"]) != cfg["initial_model_identity"]
        for cp in (
            saved,
            load(folder / "initial.pt", weights_only=False),
            load(FAILED / "project/results/last.pt", weights_only=False),
        )
    ):
        raise ValueError("saved, initial and projected states must agree at step zero")
    bounds = bound_failure(folder / "history.jsonl", saved["step"], cfg["config"]["log_every"])
    train = load(DATA / "train.pt", weights_only=True)["tensors"]
    with file_lock(PRIVATE / "timing-dispatch" / f"{host}.gpu0.log"):

        def smi(query: str) -> str:
            return subprocess.check_output(
                ["nvidia-smi", "--id=0", query, "--format=csv,noheader,nounits"], text=True
            ).strip()

        if (
            smi("--query-compute-apps=pid")
            or int(smi("--query-gpu=memory.free")) < 8192
            or int(smi("--query-gpu=utilization.gpu")) != 0
        ):
            raise RuntimeError("original GPU is currently busy")
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(device)
        if (gpu_name, str(torch.__version__), torch.version.cuda) != (
            cfg["runtime"]["gpu_name"],
            cfg["runtime"]["torch_version"],
            cfg["runtime"]["cuda_version"],
        ):
            raise ValueError("original GPU model and Torch/CUDA software required")
        model = native.build_control("dense", SEED, 6, predictor_activation="gelu").to(device)
        model.load_state_dict(saved["model"], strict=True)
        result = replay_window(
            model,
            saved,
            {k: train[k] for k in native.INPUT_KEYS},
            cfg["config"],
            bounds["latest_possible_failed_update"],
            lambda: check_head(commit),
        )
    return {
        "reproduction": result,
        "historical_failure_bounds": bounds,
        "source_sha256": {str(p): v for p, v in inputs.items()},
        "runtime": {
            "host": host,
            "gpu_name": gpu_name,
            "physical_ordinal": 0,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }
