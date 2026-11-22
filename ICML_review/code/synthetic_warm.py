"""Continue the synthetic joint head through the original frozen acquisition and sparse stages."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.shared.io import read_json, sha
from scjepa.models.model import SCJEPA

from . import single_acquisition as pipeline

MODULE = "ICML_review.code.synthetic_warm"
POLICY = "synthetic-warm"
TRANSFER = "Use all joint code-encoder and predictor tensors with frozen joint state/EMA encoders"
METHOD = (
    "Hypothesis: retaining the joint code encoder and predictor makes a single frozen acquisition "
    "more reliable. Compare to the original fresh a0 at its same 100000 updates, joint source, "
    "training groups, donor stream, LR, fresh AdamW, twelve codes and fixed state/EMA. The one "
    "controlled change is the initialization of the trainable code encoder and predictor. "
    "Use the original training-only acquisition cutoff and trailing window. No rankings, "
    "restarts or new tolerances. After admission use the native projection, 50000-update "
    "adaptation and both 300000+50000 sparse forks. Report the single cold comparator and "
    "the reported best-of-sixteen control separately. Failed admission or any endpoint gate "
    "ends the seed-zero pilot; an admission alone supports only forecast continuity."
)


def _paths(seed: int) -> tuple[Path, Path, Path]:
    return (
        pipeline.PARENT / f"states-s{seed}/results",
        pipeline.PARENT / f"acquisition-0-s{seed}/results",
        pipeline.PARENT / f"selection-s{seed}.json",
    )


def declaration(seed: int) -> str:
    """Bind the immutable common joint, budget-matched control and original TRAIN cutoff."""
    joint, cold, selector = _paths(seed)
    original = read_json(selector)
    bound = pipeline.PARENT / "calibration/calibration.json"
    tau = read_json(bound)["tau_prediction"]
    # The originally recorded joint checkpoint digest is also verified against its raw bytes.
    sources = {str(d): read_json(d / "completion.json")["hashes"]["last.pt"] for d in (joint, cold)}
    return (
        f"# Single warm synthetic acquisition, seed {seed}\n\n{METHOD}\n\n{pipeline.GATE}\n\n"
        f"Original selector {selector} SHA256 {sha(selector)}, window {original['window']} "
        f"and maximum loss {original['maximum_loss']}. "
        f"Fixed numerical tau {tau} from {bound}, SHA256 {sha(bound)}. "
        f"Inputs and SHA256 {json.dumps(sources, sort_keys=True)}."
    )


def _payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def verify_transfer(folder: Path, parent: Path) -> None:
    """Check the complete initial model and the frozen final state tensors against the joint."""
    initial = _payload(folder / "initial.pt")
    if initial["initialization"]["module_sha256"] != sha(Path(__file__)) or initial[
        "initialization"
    ]["source_checkpoint_sha256"] != sha(parent / "last.pt"):
        raise ValueError("warm initializer identity changed")
    weights = cast(dict[str, Tensor], _payload(parent / "last.pt")["model"])
    torch.testing.assert_close(initial["model"], weights, rtol=0, atol=0)
    last = _payload(folder / "last.pt")
    if not all(
        torch.equal(t, last["model"][k])
        for k, t in weights.items()
        if k.startswith(runner.STATE_ENCODERS)
    ):
        raise ValueError("state or EMA weights changed during frozen acquisition")


def train_warm(config: runner.TrainConfig, *, resume: bool = False) -> dict[str, Any]:
    """Copy the full joint model at initialization only and leave native training intact."""
    if (
        config.states_from is None
        or config.model_code_dim != 12
        or config.model_initializer != "shared_feature"
    ):
        raise ValueError("expected an original twelve-code frozen acquisition")
    parent = Path(config.states_from)
    source = runner.verify_completion(parent, kind="train", require_current_source=True)
    saved = _payload(parent / "last.pt")
    if (
        source["code_dim"] != config.model_code_dim
        or source["seed"] != config.seed
        or source["data_identity"]
        != read_json(Path(config.data_dir) / "completion.json")["identity"]
        or source["state_source"] is not None
        or source["adaptation_source"] is not None
        or source["steps"] != saved["step"]
        or source["identity"] != saved["identity"]
    ):
        raise ValueError("unmatched jointly trained checkpoint")
    weights = cast(dict[str, Tensor], saved["model"])
    parent_sha = source["hashes"]["last.pt"]
    original = runner._scientific_config  # pyright: ignore[reportPrivateUsage]

    def restore(model: SCJEPA, _data: object, _device: torch.device) -> dict[str, Any]:
        for key, value in model.state_dict().items():
            if key.startswith(runner.STATE_ENCODERS) and not torch.equal(value.cpu(), weights[key]):
                raise ValueError("native frozen state initializer does not match joint")
        model.load_state_dict(weights, strict=True)
        return {
            "policy": TRANSFER,
            "source_checkpoint_sha256": parent_sha,
            "source_completion_sha256": sha(parent / "completion.json"),
            "module_sha256": sha(Path(__file__)),
            "copies_optimizer": False,
            "uses_physical_labels": False,
        }

    def description(cfg: runner.TrainConfig) -> dict[str, Any]:
        return original(cfg) | {
            "initialization": TRANSFER,
            "initializer": "warm_joint",
            "protected_configuration_template": cfg.model_initializer,
        }

    with (
        patch.object(runner, "initialize_shared_output", restore),
        patch.object(runner, "_scientific_config", description),
    ):
        result = runner.train(config, resume=resume)
    verify_transfer(Path(config.output_dir), parent)
    return result


def prepare(root: Path, seed: int, commit: str, data_id: str, note: str) -> dict[str, Any]:
    """Train one warm head and score it on exactly the original training-only admission rule."""
    joint, cold, selector = _paths(seed)
    historical = read_json(selector)
    control = pipeline._admission(seed, commit, data_id, False)  # pyright: ignore[reportPrivateUsage]
    path = root / "acquisition/results"
    pipeline._stage(  # pyright: ignore[reportPrivateUsage]
        path.parent, [MODULE, commit, str(seed), "acquire"], commit, data_id, declaration=note
    )
    # Repeat the tensor checks even when a completed stage is reused after an interruption.
    verify_transfer(path, joint)
    admission = select([path], window=historical["window"], maximum_loss=historical["maximum_loss"])
    if (
        admission["common"] != control["common"]
        or admission["selected"]["acquisition_seed"] != control["selected"]["acquisition_seed"]
    ):
        raise ValueError("paired cold control differs beyond the initialization")
    previous = pipeline.PRIVATE / f"single-acquisition-s{seed}"
    comparison: dict[str, Any] = {}
    if (previous / "metrics.json").is_file():
        original = read_json(previous / "admission.json")
        metrics = read_json(previous / "metrics.json")
        if (
            original["checkpoint_sha256"] != control["checkpoint_sha256"]
            or original["selected"]["identity"] != control["selected"]["identity"]
            or original["admitted"] != control["admitted"]
            or metrics["seed"] != seed
        ):
            raise ValueError("previous review control is a different acquisition")
        if metrics.get("validation") is not None:
            if metrics["acquisition_receipt_sha256"] != sha(previous / "admission.json"):
                raise ValueError("previous cold control assessment is unbound")
            for arm in ("full", "baseline"):
                d = previous / f"{arm}-350k" / "results"
                if sha(d / "last.pt") != metrics["checkpoint_sha256"][f"{arm}-350k"]:
                    raise ValueError("previous cold endpoint checkpoint changed")
        comparison = {
            "directory": str(previous),
            "commit": metrics["commit"],
            "passed": metrics["passed"],
            "reason": metrics.get("reason"),
            "validation": metrics.get("validation"),
            "checkpoint_sha256": metrics.get("checkpoint_sha256"),
            "metrics_sha256": sha(previous / "metrics.json"),
            "admission_sha256": sha(previous / "admission.json"),
        }
    return admission | {
        "commit": commit,
        "data_identity": data_id,
        "policy": POLICY,
        "checkpoint_sha256": sha(path / "last.pt"),
        "initial_sha256": sha(path / "initial.pt"),
        "source_sha256": sha(joint / "last.pt"),
        "source_completion_sha256": sha(joint / "completion.json"),
        "acquisition_completion_sha256": sha(path / "completion.json"),
        "selection_sha256": sha(selector),
        "initial_tensors_match": True,
        "frozen_tensors_match": True,
        "cold_control": {
            "directory": str(cold),
            "previous_sparse_comparison": comparison,
            "admitted": control["admitted"],
            "selected": control["selected"],
            "checkpoint_sha256": control["checkpoint_sha256"],
            "completion_sha256": control["completion_sha256"],
        },
    }


def main() -> None:
    """Run preparation, either paired sparse fork or the common final assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=("prepare", "acquire", "baseline", "full", "assess"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "acquire":
        pipeline._head(args.commit)  # pyright: ignore[reportPrivateUsage]
        root = pipeline.PRIVATE / f"{POLICY}-s{args.seed}"
        if args.seed:
            pilot = read_json(pipeline.PRIVATE / f"{POLICY}-s0/metrics.json")
            if pilot["commit"] != args.commit or not pilot["passed"]:
                raise ValueError("pilot has not licensed additional training")
        joint, cold, _ = _paths(args.seed)
        raw = read_json(cold / "config.json")["config"]
        cfg = runner.TrainConfig(**raw)
        if (
            cfg.steps != 100_000
            or cfg.device != "cuda"
            or cfg.seed != args.seed
            or cfg.acquisition_seed != 1000 + args.seed
            or cfg.states_from is None
            or Path(cfg.states_from).resolve() != joint.resolve()
            or Path(cfg.data_dir).resolve() != pipeline.DATA.resolve()
        ):
            raise ValueError("historical cold acquisition does not match the declared design")
        directory = root / "acquisition"
        if (
            not (directory / "NOTE.md").exists()
            or args.commit not in (directory / "NOTE.md").read_text()
        ):
            raise ValueError("run prepare to declare the acquisition first")
        pipeline._idle_gpu()  # pyright: ignore[reportPrivateUsage]
        train_warm(
            replace(
                cfg,
                output_dir=str((directory / "results").resolve()),
                wandb_group=f"icml-review-{POLICY}-s{args.seed}",
            ),
            resume=args.resume,
        )
        pipeline._head(args.commit)  # pyright: ignore[reportPrivateUsage]
        return
    pipeline.run(
        args.commit,
        args.seed,
        args.phase,
        route=pipeline.AcquisitionRoute(POLICY, MODULE, declaration, prepare),
    )


if __name__ == "__main__":
    main()
