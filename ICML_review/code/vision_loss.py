"""Compare a paired deployed-mask forecast constraint in a fixed sparse continuation."""

import argparse
import fcntl
import json
import math
import platform
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from diagnostics.vision_dimer.model import LaggedMaskedPredictor
from ICML_review.code.single_acquisition import (
    _head,  # pyright: ignore[reportPrivateUsage]
    _idle_gpu,  # pyright: ignore[reportPrivateUsage]
)
from scjepa.models.model import SCJEPA, ModelOutput

ARCHIVE = Path("outputs/diagnostics/vision-dimer-rgb-v1-s05all")
OUTPUTS = Path("ICML_review/outputs")
SPLITS = ("train", "calibration", "validation")
SHD = ("state_shd", "action_shd", "parameter_shd")
GATE = (
    "Require deployed calibration and validation MSE ≤ the original numerical tau, "
    "validation Pearson MCC ≥ the reported full seed minus 0.01, minimum retained state "
    "and drag R² ≥ 0.99 and 0.95. On the pilot require total validation SHD at most half "
    "its reported value (rounded down). On expansion require SHD ≤ min(6, reported SHD). "
    "All pilot conditions must hold to launch seeds 1-4. Report every outcome. A failure "
    "leaves the proposed late continuation unestablished."
)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write(path: Path, record: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


def hard_forecast(
    predictor: LaggedMaskedPredictor, out: ModelOutput, actions: Tensor, donor: Tensor
) -> Tensor:
    """Replay the deployed two-lag predictor using its thresholded graph and attached donors."""
    logits = predictor.learned_logits()
    if set(logits) != {"state", "action", "parameter"}:
        raise ValueError("paired constraint requires the full sparse vision predictor")

    def gate(name: str) -> Tensor:
        value = logits[name]
        soft = torch.sigmoid(value / predictor.temperature)
        return (value > 0).to(value.dtype) + (soft - soft.detach())

    state, action, code = (
        gate("state")[None, :, None],
        gate("action")[None, :, None],
        gate("parameter"),
    )
    source = out.prediction_start
    history = out.online_states.flatten(0, 1)[:, source - 1 : source + 1]
    fields = actions.flatten(0, 1)
    steps: list[Tensor] = []
    for time in range(source, source + out.predictions.shape[-2]):
        inputs = torch.cat(
            (
                (history[:, None] * state).flatten(2),
                (fields[:, None, time - 1 : time + 1] * action).flatten(2),
                donor[:, None] * code,
            ),
            -1,
        )
        features = F.gelu(predictor.second(F.gelu(predictor.first(inputs))))
        z = (features * predictor.readout_weight).sum(-1) + predictor.readout_bias
        steps.append(z)
        history = torch.cat((history[:, 1:], z[:, None]), 1)
    return torch.stack(steps, 1).reshape_as(out.predictions)


class PairedBranches:
    """Replay both forecasts with the sampled donor and optionally optimize the worse."""

    def __init__(self, model: SCJEPA, use_max: bool, observe: Callable[[Tensor], None]) -> None:
        """Install hooks only on this instance; archived evaluation runs with both hooks dormant."""
        if not model.autoregressive or not isinstance(model.predictor, LaggedMaskedPredictor):
            raise ValueError("requires the reported autoregressive vision model")
        self.predictor = model.predictor
        self.donor: Tensor | None = None
        self.use_max = use_max
        self.observe = observe
        self.predictor.register_forward_pre_hook(self._capture)
        model.register_forward_hook(self._compare)

    def _capture(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        if self.predictor.training and self.donor is None:
            self.donor = args[1]

    def _compare(self, _module: nn.Module, args: tuple[Any, ...], out: ModelOutput) -> ModelOutput:
        if not self.predictor.training:
            return out
        if self.donor is None:
            raise RuntimeError("stochastic branch did not expose its donor")
        hard = hard_forecast(self.predictor, out, args[1], self.donor)
        self.donor = None
        with torch.no_grad():
            stochastic_mse = (out.predictions - out.targets).square().mean()
            hard_mse = (hard - out.targets).square().mean()
            active = hard_mse > stochastic_mse
            self.observe(
                torch.stack(
                    (
                        stochastic_mse,
                        hard_mse,
                        torch.maximum(stochastic_mse, hard_mse),
                        active.float(),
                    )
                )
            )
        if self.use_max:
            return out._replace(predictions=torch.where(active, hard, out.predictions))
        return out


BRANCHES = ("stochastic_mse", "hard_mse", "maximum_mse", "fraction_hard_active")


def _numbers(vector: Tensor) -> list[float]:
    return [float(x) for x in vector.cpu().unbind()]


def _means(vectors: Tensor) -> dict[str, float]:
    return dict(zip(BRANCHES, _numbers(vectors.double().mean(0)), strict=True))


@torch.no_grad()
def paired_calibration(
    checkpoint: Path, config: release.StageConfig, *, draws: int = 4
) -> dict[str, Any]:
    """Measure paired batch discrepancies on fixed calibration windows and evaluation donors."""
    device = torch.device(config.device)
    source = vdata.RenderedInputs(Path(config.data_dir), "calibration", device)
    model = release.model_from_checkpoint(release.load_checkpoint(checkpoint), phase="full").to(
        device
    )
    model.eval()
    model.predictor.train()  # Fixed eval donor shift; only gate sampling is stochastic.
    values: list[Tensor] = []
    PairedBranches(model, False, values.append)
    batch_sizes: list[int] = []
    # This probe runs before resume, which independently restores all archived RNG states.
    for draw in range(draws):
        torch.manual_seed(config.seed + draw)  # pyright: ignore[reportUnknownMemberType]
        for batch in release.rendered_batches(source, size=config.batch_groups):
            model(batch["observations"], batch["actions"])
            batch_sizes.append(len(batch["actions"]))
    matrix = torch.stack(values).cpu()
    if len(set(batch_sizes)) != 1:
        raise ValueError("unequal calibration batch sizes require weighted summaries")
    difference = matrix[:, 1] - matrix[:, 0]
    return {
        "draws": draws,
        "gate_seeds": list(range(config.seed, config.seed + draws)),
        "contexts_per_batch": batch_sizes[0],
        "batches_per_draw": len(values) // draws,
        "fixed_first_window": True,
        "donor": "evaluation cyclic shift",
        "paired_batch_means": _means(matrix),
        "hard_minus_stochastic_batch_quantiles": dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                _numbers(torch.quantile(difference, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))),
                strict=True,
            )
        ),
        "paired_batches_by_draw": [
            dict(zip(BRANCHES, _numbers(row), strict=True)) for row in matrix.unbind()
        ],
    }


def continue_training(config: release.StageConfig, *, use_max: bool = True) -> dict[str, Any]:
    """Resume the native trainer, changing only the forecast selected for loss and GECO."""
    output = Path(config.output_dir)
    step = int(release.load_checkpoint(output / "last.pt")["step"])
    history = output / "branches.jsonl"
    if history.exists():
        rows = [
            line for line in history.read_text().splitlines() if json.loads(line)["step"] <= step
        ]
        history.write_text("".join(line + "\n" for line in rows))
    values: list[Tensor] = []

    def observe(value: Tensor) -> None:
        nonlocal step
        step += 1
        values.append(value)
        if step % config.log_every == 0 or step == config.steps:
            row = {"step": step, "updates": len(values), **_means(torch.stack(values))}
            with history.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            values.clear()

    original = release.model_from_checkpoint

    def build(payload: dict[str, Any], *, phase: str = "dense") -> SCJEPA:
        model = original(payload, phase=phase)
        PairedBranches(model, use_max, observe)
        return model

    with patch.object(release, "model_from_checkpoint", build):
        final = release.train(config, resume=True)
    if step != config.steps:
        raise RuntimeError("paired branch did not cover the fixed continuation budget")
    return final


def assess(
    seed: int, tau: float, trial: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the predeclared seed-level gate from released calibration/validation metrics."""
    original_shd = sum(int(reference[k]) for k in SHD)
    shd_limit = original_shd // 2 if seed == 0 else min(6, original_shd)
    checks = {
        "calibration_feasible": trial["calibration_prediction_mse"] <= tau,
        "validation_feasible": trial["prediction_mse"] <= tau,
        "shd": sum(int(trial[k]) for k in SHD) <= shd_limit,
        "mcc": trial["mcc"] >= reference["mcc"] - 0.01,
        "state_retention": trial["state_r2_min"] >= 0.99,
        "drag_retention": trial["parameter_r2_min"] >= 0.95,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "shd_limit": shd_limit,
        "tau": tau,
        "validation": trial,
        "reference_validation": reference,
    }


REPLAY_RELATIVE = 1e-4
REPLAY_ABSOLUTE = 1e-8


def compare_replay(measured: dict[str, Any], archived: dict[str, Any]) -> dict[str, Any]:
    """Assess every archived scientific field at the same log step, excluding wall time."""
    if archived["step"] != measured["step"] or archived.keys() != measured.keys():
        raise ValueError("replay log schema or update differs from the archived control")
    differences: dict[str, Any] = {}
    for key in sorted(archived.keys() - {"step", "seconds_per_step"}):
        current, old = float(measured[key]), float(archived[key])
        delta = abs(current - old)
        differences[key] = {
            "replayed": current,
            "archived": old,
            "absolute_error": delta,
            "relative_error": delta / abs(old) if old else None,
            "within_tolerance": math.isclose(
                current, old, rel_tol=REPLAY_RELATIVE, abs_tol=REPLAY_ABSOLUTE
            ),
        }
    return {
        "numerically_matched": all(field["within_tolerance"] for field in differences.values()),
        "identical_logged_values": all(
            field["absolute_error"] == 0 for field in differences.values()
        ),
        "step": measured["step"],
        "fields": differences,
        "rtol": REPLAY_RELATIVE,
        "atol": REPLAY_ABSOLUTE,
    }


def _fork(parent: Path, config: release.StageConfig, digest: str) -> None:
    output = Path(config.output_dir)
    if not (output / "fork.json").exists():
        release.fork(parent, config)
    if _read(output / "fork.json")["source_sha256"] != digest:
        raise ValueError("unexpected continuation parent")


def _runtime() -> dict[str, Any]:
    properties = cast(Any, torch.cuda.get_device_properties(0))  # pyright: ignore[reportUnknownMemberType]
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_model": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def _control_interval(
    parent: Path,
    config: release.StageConfig,
    reference: Path,
    digest: str,
    note: str,
    commit: str,
    start: int,
) -> dict[str, Any]:
    output = Path(config.output_dir) / "control-replay"
    output.mkdir(exist_ok=True)
    replay = replace(
        config,
        output_dir=str(output),
        wandb_group=config.wandb_group + "-control",
        steps=start + config.log_every,
    )
    control_note = (
        "# Unchanged forecast control replay\n\n"
        "Hypothesis: resuming the original stochastic objective reproduces the first archived "
        "logging interval at the common sparse checkpoint. Train exactly one logging interval "
        "with the original objective. Also measure the paired deterministic branch without "
        "feeding it to gradient or GECO. Compare every numeric log field apart from seconds per "
        "step at a common endpoint. Record absolute/relative drift and all-field agreement at "
        f"rtol={REPLAY_RELATIVE:g}, atol={REPLAY_ABSOLUTE:g}. Above this tolerance the archived "
        "comparison remains unresolved and expansion is disallowed. The paired pilot can still "
        "report absolute feasibility. The following parent NOTE gives the data identity, "
        "authenticated source, commit and exact executable command that runs this replay before "
        "starting the independent trial.\n\n" + note
    )
    run_note = output / "NOTE.md"
    if run_note.exists() and run_note.read_text() != control_note:
        raise ValueError("control replay preregistration differs")
    run_note.write_text(control_note)
    _head(commit)
    record_path = output / "replay.json"
    if record_path.exists():
        existing = _read(record_path)
        if (
            existing["commit"] != commit
            or existing["source_sha256"] != digest
            or existing["checkpoint_sha256"] != vdata.sha256(output / "final.pt")
        ):
            raise ValueError("control replay identity changed")
        return existing
    _fork(parent, replay, digest)
    if not (output / "final.json").exists():
        continue_training(replay, use_max=False)
    measured = json.loads((output / "history.jsonl").read_text().splitlines()[-1])
    archived = next(
        r
        for r in map(json.loads, (reference / "history.jsonl").read_text().splitlines())
        if r["step"] == replay.steps
    )
    result = compare_replay(measured, archived)
    result.update(
        {
            "commit": commit,
            "start": start,
            "source_sha256": digest,
            "checkpoint_sha256": vdata.sha256(output / "final.pt"),
            "resumable_sha256": vdata.sha256(output / "last.pt"),
            "runtime": _runtime(),
            "replay_seconds_per_step": measured["seconds_per_step"],
            "archived_seconds_per_step": archived["seconds_per_step"],
        }
    )
    _head(commit)
    _write(output / "replay.json", result)
    return result


def run(commit: str, seed: int, output: Path) -> None:
    """Authenticate the common parent, run paired calibration and training, record the endpoint."""
    _head(commit)
    archive = ARCHIVE / f"seed{seed}"
    parent = archive / "sparse-300k/full/last.pt"
    baseline = archive / "sparse-500k/full"
    current = release.StageConfig.from_saved(_read(baseline / "config.json")["config"])
    if (
        current.stage != "full"
        or current.steps != 500_000
        or current.seed != seed
        or current.device != "cuda"
    ):
        raise ValueError("unexpected archived full sparse configuration")
    cfg = replace(
        current, output_dir=str(output.resolve()), wandb_group=f"icml-review-vision-loss-s{seed}"
    )
    data_dir = Path(cfg.data_dir)
    manifest = _read(data_dir / "manifest.json")
    splits = {s: manifest["splits"][s]["sha256"] for s in SPLITS}
    recorded_data = _read(ARCHIVE / "release-check-20261120/data-manifest-with-test.json")
    if any(splits[s] != recorded_data["splits"][s]["sha256"] for s in SPLITS):
        raise ValueError("data identities differ from the archived release")
    fork = _read(baseline / "fork.json")
    command = (
        "PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.vision_loss "
        f"{shlex.quote(commit)} {seed}"
    )
    note = (
        f"# Paired forecast constraint, vision full seed {seed}\n\n"
        "Hypothesis: constraining the worst of paired stochastic and thresholded two-step "
        "forecast MSE improves deployed feasibility and graph recovery without loss of retention. "
        "Change only the forecast selected for the objective and GECO. A deterministic "
        "threshold uses the sigmoid straight-through derivative. Both branches share the "
        "same states, donor codes, actions, targets, horizon, batch and weights. Train "
        "without labels. Resume archived optimizer, EMA, sampled gates, sampler, windows "
        "and GECO at update 300000 to 500000. Compare archived full endpoint at the same "
        "budget. Before the continuation, characterize both masks on identical first-window "
        "calibration batches and fixed cyclic evaluation donors, four gate draws. Repeat at "
        "the original and changed endpoints. Source history through 300000 is inherited. "
        "A separate 500-update replay of the unchanged objective authenticates the first "
        "logged interval of the original 500k GPU control. Larger discrepancies than "
        f"rtol={REPLAY_RELATIVE:g}, atol={REPLAY_ABSOLUTE:g} flag an unresolved comparison "
        "and disallow expansion. Repeatability over the entire 200k updates is unverified.\n\n"
        f"{GATE}\n\nCommit `{commit}`. Source `{parent}`, SHA256 `{fork['source_sha256']}`. "
        f"Tolerance `{cfg.tolerance()}`. Data `{data_dir.name}`. "
        f"Split SHA256 `{json.dumps(splits, sort_keys=True)}`.\n\n"
        f"```sh\n{command}\n```\n"
    )
    note_path = output / "NOTE.md"
    if note_path.exists() and note_path.read_text() != note:
        raise ValueError("existing run has different preregistration")
    note_path.write_text(note)
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
    )
    if fork["source_step"] != 300_000 or vdata.sha256(parent) != fork["source_sha256"]:
        raise ValueError("resumable source differs from archived fork")
    archive_final = baseline / "final.pt"
    archive_hash = vdata.sha256(archive_final)
    if not any(
        line.split() == [archive_hash, str(archive_final.relative_to("outputs"))]
        for line in Path("outputs/reported-checkpoints.sha256").read_text().splitlines()
    ):
        raise ValueError("control endpoint is absent from reported manifest")
    if (output / "metrics.json").exists():
        result = _read(output / "metrics.json")
        if result["commit"] != commit or result["sha256"] != vdata.sha256(output / "final.pt"):
            raise RuntimeError("completed output identity changed")
        return
    if seed:
        pilot = _read(OUTPUTS / "vision-loss-s0/metrics.json")
        if pilot["commit"] != commit or not pilot["passed"]:
            raise ValueError("pilot did not authorize expansion")
    _idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not (output / "calibration-parent.json").exists():
        _write(output / "calibration-parent.json", paired_calibration(parent, cfg))
    _head(commit)
    if not (output / "calibration-reference.json").exists():
        _write(output / "calibration-reference.json", paired_calibration(archive_final, cfg))
    replay = _control_interval(
        parent, cfg, baseline, fork["source_sha256"], note, commit, fork["source_step"]
    )
    _head(commit)
    _fork(parent, cfg, fork["source_sha256"])
    if not (output / "final.json").exists():
        continue_training(cfg)
    _head(commit)
    final = output / "final.pt"
    if not (output / "calibration-endpoint.json").exists():
        _write(output / "calibration-endpoint.json", paired_calibration(final, cfg))
    result = assess(
        seed, cfg.tolerance(), _read(output / "final.json"), _read(baseline / "final.json")
    )
    result["scientific_gate_passed"] = result["passed"]
    result["passed"] = result["passed"] and replay["numerically_matched"]
    result["control_replay_matched"] = replay["numerically_matched"]
    result["runtime"] = _runtime()
    _head(commit)
    _write(
        output / "metrics.json",
        {
            **result,
            "seed": seed,
            "commit": commit,
            "sha256": vdata.sha256(final),
            "resumable_sha256": vdata.sha256(output / "last.pt"),
            "source": str(parent),
            "source_sha256": fork["source_sha256"],
            "reference": str(archive_final),
            "reference_sha256": archive_hash,
            "splits": splits,
            "command": command,
        },
    )


def main() -> None:
    """Run one seed under an exclusive output lock and leave a process exit marker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    args = parser.parse_args()
    output = OUTPUTS / f"vision-loss-s{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "exit_code"
        marker.unlink(missing_ok=True)
        try:
            run(args.commit, args.seed, output)
        except BaseException:
            marker.write_text("1\n")
            raise
        marker.write_text("0\n")


if __name__ == "__main__":
    main()
