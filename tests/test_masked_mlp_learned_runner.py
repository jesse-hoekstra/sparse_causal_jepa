"""Learned parameter codes: label-free training, dense initialization, code probes, resume."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle.data import PrepareConfig, prepare
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_oracle.model import INITIAL_LOGIT
from diagnostics.oracle_state_learning.core import EvaluationResult
from diagnostics.oracle_state_learning.data import read_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _prepare(root: Path) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(root / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))


def _config(root: Path, condition: runner.Condition, seed: int = 0) -> runner.TrainConfig:
    sparse = condition != "dense"
    return runner.TrainConfig(
        str(root / f"{condition}-s{seed}"),
        str(root / "data"),
        condition,
        profile="smoke",
        seed=seed,
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if sparse else None,
        init_dir=str(root / f"dense-s{seed}") if sparse else None,
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def _dense_and_bound(root: Path) -> None:
    runner.train(_config(root, "dense"))
    runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), str(root / "dense-s0"), profile="smoke")
    )


def test_training_never_feeds_true_theta_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    seen: list[set[str]] = []
    actual = runner.forward_loss

    def observe(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        if model.training:
            seen.append(set(batch))
        return actual(model, batch, objective, forward_seed=forward_seed)

    monkeypatch.setattr(runner, "forward_loss", observe)
    runner.train(_config(tmp_path, "dense"))
    assert seen
    assert all(keys == {"observations", "actions"} for keys in seen)
    model = runner.build_control("dense", 0)
    assert model.parameter_source == "learned"
    with pytest.raises(ValueError, match="oracle"):
        model(
            torch.zeros(2, 10, 12, 6),
            torch.zeros(2, 10, 11, 24),
            oracle_parameters=torch.zeros(2, 6),
        )


def test_sparse_arms_start_from_the_dense_endpoint_and_report_codes(tmp_path: Path) -> None:
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    dense = _checkpoint(tmp_path / "dense-s0/last.pt")["model"]
    dense_initial = _checkpoint(tmp_path / "dense-s0/initial.pt")["model"]
    changed = [
        key
        for key in dense
        if key.startswith("parameter_encoder.") and not torch.equal(dense[key], dense_initial[key])
    ]
    assert changed, "the parameter encoder must learn during dense training"
    for condition in ("baseline", "full"):
        completion = runner.train(_config(tmp_path, condition))
        initial = _checkpoint(tmp_path / f"{condition}-s0/initial.pt")["model"]
        assert set(initial) - set(dense) <= runner.MASK_LOGITS
        for key, value in dense.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(initial[key], value, rtol=0, atol=0)
        for key in set(initial) - set(dense):
            assert bool((initial[key] == INITIAL_LOGIT).all())
        metrics = read_json(tmp_path / f"{condition}-s0/validation.json")["metrics"]
        for key in (
            "state_mcc",
            "block_purity",
            "code_r2_min",
            "code_mcc",
            "code_pair_purity",
            "forecast/code_use_ratio",
            "code_r2/theta_6",
            "geco/relative_prediction",
        ):
            assert key in metrics
        assert completion["dense_endpoint"]["seed"] == 0
        assert runner.verify_completion(tmp_path / f"{condition}-s0") == completion
    rows = runner.summarize([str(tmp_path / f"{c}-s0") for c in ("dense", "baseline", "full")])
    assert [row["condition"] for row in rows] == ["baseline", "dense", "full"]


def test_code_metrics_recover_an_informative_code() -> None:
    generator = torch.Generator().manual_seed(0)
    theta = torch.rand(400, 6, generator=generator, dtype=torch.float64) * 1.5 - 0.75
    pair_mix = torch.eye(6, dtype=torch.float64)
    pair_mix[0, 3] = pair_mix[3, 0] = 0.6

    def result(groups: slice) -> EvaluationResult:
        part = theta[groups]
        codes = (part @ pair_mix.T)[:, None].expand(-1, 3, -1)
        return EvaluationResult({}, {"codes": codes.float(), "theta": part.float()})

    metrics = runner.code_metrics(result(slice(0, 300)), result(slice(300, 400)))
    assert metrics["code_r2_min"] > 0.99
    assert metrics["code_pair_purity"] > 0.99
    assert metrics["code_mcc"] < 0.99


def test_shuffled_codes_route_codes_across_groups() -> None:
    model = runner.build_control("dense", 0)
    original = model.parameter_encoder
    states = torch.randn(3 * 2, 10, 6)
    actions = torch.randn(3 * 2, 9, 24)
    with torch.no_grad():
        codes = original(states, actions).reshape(3, 2, 6)
        with runner.shuffled_codes(model, views=2):
            routed = model.parameter_encoder(states, actions).reshape(3, 2, 6)
    torch.testing.assert_close(routed, codes.roll(1, dims=0))
    assert model.parameter_encoder is original


def test_protocol_is_checked_and_resume_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="dense endpoint"):
        runner.TrainConfig("u", "u", "full", calibration_dir="u")
    with pytest.raises(ValueError, match="one of"):
        runner.TrainConfig("u", "u", "dense", steps=200_000)
    with pytest.raises(ValueError, match="seed"):
        runner.TrainConfig("u", "u", "dense", seed=5)
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    config = _config(tmp_path, "full")
    complete = runner.train(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    actual = runner.forward_loss
    updates = 0

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal updates
        if model.training:
            updates += 1
            if updates == 3:
                raise RuntimeError("deliberate interruption")
        return actual(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            runner.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left = _checkpoint(Path(config.output_dir) / "last.pt")
    right = _checkpoint(tmp_path / "interrupted/last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
