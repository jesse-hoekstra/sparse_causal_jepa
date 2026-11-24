"""Replay the frozen RGB first-window projection and measure its TRAIN covariance support."""

import argparse
import json
import math
import os
import re
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.vision_dimer import train as release
from scjepa.experiments.storage import file_lock, write_json

from .changed_queries import Inputs, require
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA, OUTPUT, ROOT, checkpoint_hashes, declared, read, sha

SEED = 3
NATIVE = ROOT / f"seed{SEED}"
GUIDED = OUTPUT / f"timing-rgb-entire-s{SEED}"
PLAIN = OUTPUT / f"timing-rgb-sparse-s{SEED}"
OUT = OUTPUT / f"timing-rgb-projection-s{SEED}"


def covariance_record(matrix: Tensor) -> dict[str, Any]:
    """Describe the symmetric spectrum and distinguish positive support from numerical rank."""
    symmetric = (matrix + matrix.T) / 2
    values = cast(Tensor, torch.linalg.eigvalsh(symmetric))  # pyright: ignore[reportUnknownMemberType]
    threshold = max(1e-300, float(values[-1]) * 1e-12)
    rank = cast(Tensor, torch.linalg.matrix_rank(matrix))  # pyright: ignore[reportUnknownMemberType]
    return {
        "eigenvalues": values.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "numeric_matrix_rank": int(rank),
        "positive_resolved_rank": int((values > threshold).sum()),
        "positive_rank_threshold": threshold,
        "matrix": matrix.tolist(),  # pyright: ignore[reportUnknownMemberType]
    }


def replay(checkpoint: Path, data: Path, out: Path, device: str) -> dict[str, Any]:
    """Intercept the released projection after its actual unregularized inverse attempt."""
    original_group = release.group_covariances
    original_inv = cast(Callable[[Tensor], Tensor], torch.linalg.inv)  # pyright: ignore[reportUnknownMemberType]
    linalg_error = cast(type[RuntimeError], torch.linalg.LinAlgError)  # pyright: ignore[reportUnknownMemberType,reportPrivateImportUsage]
    captured: list[Tensor] = []
    outcome: dict[str, Any] = {}

    class Finished(Exception):
        """The native diagnostic has reached the inverse, before fitting new weights."""

    def group(codes: Tensor) -> tuple[Tensor, Tensor]:
        shared, total = original_group(codes)
        require(not captured, "multiple projection code cohorts")
        captured.extend((codes, shared, total))
        return shared, total

    def inverse(matrix: Tensor) -> Tensor:
        if not captured:
            return original_inv(matrix)
        try:
            value = original_inv(matrix)
        except linalg_error as exc:
            outcome.update(inverse_succeeded=False, inverse_error=str(exc))
        else:
            error = float((matrix @ value - torch.eye(len(matrix), dtype=matrix.dtype)).abs().max())
            outcome.update(
                inverse_succeeded=True,
                inverse_finite=bool(torch.isfinite(value).all()),
                inverse_identity_max_abs=error if math.isfinite(error) else None,
            )
        captured.append(matrix)
        raise Finished

    with (
        patch.object(release, "group_covariances", group),
        patch.object(torch.linalg, "inv", inverse),
        patch.object(release, "_save", side_effect=RuntimeError("unexpected projection save")),
    ):
        try:
            release.project(checkpoint.parent, data, out / "unused", device=device)
        except Finished:
            pass
        else:
            raise RuntimeError("native projection did not call the registered inverse")
    codes, shared, total, projected = captured
    rows = codes.flatten(0, 1)
    lows, highs = rows.amin(0), rows.amax(0)
    spans = highs - lows
    return {
        "code_shape": list(codes.shape),
        "coordinate_min": lows.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "coordinate_max": highs.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "coordinate_span": spans.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "exactly_constant_coordinates": (spans == 0).tolist(),  # pyright: ignore[reportUnknownMemberType]
        "all_train_rows_identical": bool((spans == 0).all()),
        "total": covariance_record(total),
        "shared": covariance_record(shared),
        "projected_total": covariance_record(projected),
        **outcome,
    }


def declaration(commit: str, training: str) -> str:
    """Declare the fixed failed and untreated first acquisitions and measurement convention."""
    command = (
        "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/src .venv/bin/python "
        f"-m ICML_review.code.timing_rgb_projection {commit} {training}"
    )
    data_id = sha(DATA / "manifest.json")
    train_id = read(DATA / "manifest.json")["splits"]["train"]["sha256"]
    return (
        "# Frozen RGB first-acquisition projection replay\n\n"
        "Hypothesis: guidance from the first joint update can remove TRAIN code variation "
        "needed by the four-dimensional reconstruction. Change only the frozen first "
        "acquisition endpoint between the two branches. On the worker that ran the guided "
        "projection, run the released projector through the unregularized inverse with its "
        "native first-window batches. Record raw coordinate spans, shared and total covariance "
        "spectra, positive resolved ranks and the inverse result. The relative eigenvalue "
        "threshold is one trillionth of the largest positive eigenvalue, floored at 1e-300. "
        "If guided support is deficient while native support resolves, report which directions "
        "are missing. If both matrices resolve or both fail, the mechanism is unconfirmed. "
        "Read the stored calibration/validation forecast and validation retention and shuffle "
        "diagnostics. Use only TRAIN rendered observations, actions and grouping in the replay.\n\n"
        f"Diagnostic commit {commit}\nTraining commit {training}\nData {DATA}\n"
        f"Manifest SHA-256 {data_id}\nTRAIN file SHA-256 {train_id}\n"
        f"Guided checkpoint {GUIDED / 'acquire/a0/final.pt'}\n"
        f"Native checkpoint {NATIVE / 'acquire/a0/final.pt'}\n\n    {command}\n"
    )


def run(commit: str, training: str, *, declare: bool = False) -> None:
    """Replay the two fixed checkpoints after binding their identity and training lineage."""
    _head(commit)
    require(bool(re.fullmatch(r"[a-f0-9]{40}", training)), "expected full training commit")
    note = OUT / "NOTE.md"
    text = declaration(commit, training)
    with file_lock(OUT / ".lock"):
        if declare:
            declared(note, text)
            return
        require(note.read_text() == text, "missing or changed diagnostic declaration")
        for directory in (GUIDED, PLAIN):
            require(training in (directory / "NOTE.md").read_text(), "wrong training revision")
        ledger = Inputs()
        manifest = ledger.record(DATA / "manifest.json")
        split_ids = {
            s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")
        }
        ledger.bind(DATA / "train.pt", split_ids["train"])
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
        )
        logs = sorted(GUIDED.glob("attempts/*/log.txt"))
        failed = [
            p
            for p in logs
            if (p.parent / "exit_code").is_file()
            and (p.parent / "exit_code").read_text().strip() == "1"
        ]
        require(bool(failed), "missing failed worker attempt")
        log = failed[-1]
        log_text = log.read_text()
        header = log_text.splitlines()[0]
        host = re.search(r"\bhost=(\S+)", header)
        require(
            host is not None
            and host[1] == socket.gethostname().split(".")[0]
            and f"head={training}" in header
            and "in project" in log_text
            and "input matrix is singular" in log_text,
            "failure signature, revision or physical worker differs",
        )
        ledger.bind(log)
        ledger.bind(log.parent / "exit_code")
        require(os.getenv("CUDA_VISIBLE_DEVICES") == "0", "expected GPU ordinal zero")
        _idle_gpu()
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report: dict[str, Any] = {}
        fixed = ledger.record(PLAIN / "candidate.json")
        require(
            fixed["verified"]
            and fixed["candidate_index"] == 0
            and fixed["manifest_sha256"] == ledger[str(DATA / "manifest.json")],
            "uncorroborated native first acquisition",
        )
        for name, folder in (("guided", GUIDED), ("native", NATIVE)):
            _head(commit)
            ckpt = folder / "acquire/a0/final.pt"
            joint = folder / "joint/final.pt"
            if name == "guided":
                completion = ledger.record(ckpt.parent / "completion.json")
                parent = ledger.record(joint.parent / "completion.json")
                require(
                    completion["commit"] == parent["commit"] == training
                    and completion["parent_sha256"] == parent["checkpoint_sha256"],
                    "guided stage lineage differs",
                )
                ledger.bind(joint, parent["checkpoint_sha256"])
                ledger.bind(ckpt.parent / "NOTE.md", completion["note_sha256"])
                expected = completion["checkpoint_sha256"]
            else:
                require(str(ckpt) == fixed["candidate_path"], "unexpected native candidate")
                require(
                    fixed["joint_sha256"] == checkpoint_hashes()[str(joint)], "wrong native joint"
                )
                ledger.bind(joint, fixed["joint_sha256"])
                ledger.bind(
                    ckpt.parent / "final.json", fixed["first_window"]["a0"]["summary_sha256"]
                )
                expected = fixed["final_sha256"]
            ledger.bind(ckpt, expected)
            cfg = release.StageConfig.from_saved(
                ledger.record(ckpt.parent / "config.json")["config"]
            )
            require(
                cfg.stage == "acquire"
                and cfg.seed == SEED
                and cfg.acquisition_seed == 1000 + SEED
                and cfg.steps == 100_000
                and cfg.code_dim == 8,
                "wrong cold acquisition or update budget",
            )
            heldout = ledger.record(ckpt.parent / "final.json")
            eval_log = ckpt.parent / "evaluations.jsonl"
            ledger.bind(eval_log)
            require(
                heldout["step"] == cfg.steps
                and heldout == json.loads(eval_log.read_text().splitlines()[-1]),
                "held-out summary differs from the terminal native evaluation",
            )
            report[name] = {
                "checkpoint": str(ckpt),
                "sha256": expected,
                "native_projection": replay(ckpt, DATA, OUT, "cuda"),
                "archived": {
                    k: heldout[k]
                    for k in (
                        "calibration_prediction_mse",
                        "prediction_mse",
                        "mcc",
                        "parameter_r2_min",
                        "parameter_r2",
                        "code_shuffle_ratio",
                    )
                },
            }
            _head(commit)
        write_json(
            OUT / "metrics.json",
            {
                "commit": commit,
                "training_commit": training,
                "seed": SEED,
                "note_sha256": sha(note),
                "worker": socket.gethostname(),
                "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(0),
                "inputs": ledger,
                "split_ids": split_ids,
                "arms": report,
            },
        )


def main() -> None:
    """Declare or execute the read-only projection diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("training_commit")
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    run(args.commit, args.training_commit, declare=args.declare)


if __name__ == "__main__":
    main()
