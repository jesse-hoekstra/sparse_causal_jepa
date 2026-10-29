"""Dense-endpoint initialization, endpoint authentication and exact resume of sparse arms."""

import json
import math
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
from diagnostics.masked_mlp_oracle import from_dense
from diagnostics.masked_mlp_oracle import run as runner
from diagnostics.masked_mlp_oracle.model import INITIAL_LOGIT
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


def _train_config(
    root: Path, condition: runner.Condition, seed: int = 0, name: str | None = None
) -> runner.TrainConfig:
    return runner.TrainConfig(
        str(root / (name or f"{condition}-{seed}")),
        str(root / "data"),
        condition,
        profile="smoke",
        seed=seed,
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if condition != "dense" else None,
    )


def _prepare(root: Path, dense_seeds: tuple[int, ...] = (0,)) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(root / "source"),
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))
    for seed in dense_seeds:
        runner.train(_train_config(root, "dense", seed))
    runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), str(root / "dense-0"), profile="smoke")
    )


def _config(
    root: Path, condition: runner.Condition, seed: int = 0, init_seed: int | None = None
) -> from_dense.FromDenseConfig:
    return from_dense.FromDenseConfig(
        _train_config(root, condition, seed, name=f"from-dense-{condition}-{seed}"),
        str(root / f"dense-{seed if init_seed is None else init_seed}"),
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def test_sparse_arms_start_exactly_from_the_seed_matched_dense_endpoint(tmp_path: Path) -> None:
    _prepare(tmp_path)
    dense = _checkpoint(tmp_path / "dense-0/last.pt")["model"]
    dense_validation = read_json(tmp_path / "dense-0/validation.json")["metrics"]
    runs: list[str] = []
    for condition in ("baseline", "full"):
        completion = from_dense.train(_config(tmp_path, condition))
        directory = tmp_path / f"from-dense-{condition}-0"
        runs.append(str(directory))
        initial = _checkpoint(directory / "initial.pt")["model"]
        assert set(initial) - set(dense) <= from_dense.MASK_LOGITS
        for key, value in dense.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(initial[key], value, rtol=0, atol=0)
        for key in set(initial) - set(dense):
            assert bool((initial[key] == INITIAL_LOGIT).all())
        startup = read_json(directory / "startup.json")["metrics"]["validation"]
        assert startup["state_mcc"] == dense_validation["state_mcc"]
        assert (
            completion["dense_endpoint"]["identity"]
            == runner.verify_completion(tmp_path / "dense-0")["identity"]
        )
        assert from_dense.verify_completion(directory, require_current_source=True) == completion
        record = read_json(directory / "config.json")
        assert record["geco"]["tau"] == 1.0
        assert record["geco"]["step_size"] == 0.1
        assert "dense endpoint" in record["scientific_config"]["initialization"]
    rows = from_dense.summarize(runs)
    assert [row["condition"] for row in rows] == ["baseline", "full"]
    assert all(row["initial_state_mcc"] == dense_validation["state_mcc"] for row in rows)


def test_endpoint_seed_condition_and_arm_are_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only baseline or full"):
        from_dense.FromDenseConfig(_train_config(tmp_path, "dense"), "unused")
    with pytest.raises(ValueError, match="geco_step_size"):
        from_dense.FromDenseConfig(_train_config(tmp_path, "full"), "unused", geco_step_size=0)
    with pytest.raises(ValueError, match="lambda_init"):
        from_dense.FromDenseConfig(_train_config(tmp_path, "full"), "unused", geco_lambda_init=0)
    _prepare(tmp_path, dense_seeds=(0, 1))
    with pytest.raises(ValueError, match="seed differs"):
        from_dense.train(_config(tmp_path, "full", seed=0, init_seed=1))
    runner.train(_train_config(tmp_path, "baseline", 1, name="fresh-baseline"))
    sparse_init = from_dense.FromDenseConfig(
        _train_config(tmp_path, "full", 1, name="from-sparse"), str(tmp_path / "fresh-baseline")
    )
    with pytest.raises(ValueError, match="must be a dense endpoint"):
        from_dense.train(sparse_init)


def test_exact_interruption_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(tmp_path)
    config = _config(tmp_path, "full")
    complete = from_dense.train(config)
    interrupted = replace(
        config, train=replace(config.train, output_dir=str(tmp_path / "interrupted"))
    )
    actual = from_dense.forward_loss
    updates = 0

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal updates
        updates += 1
        if updates == 3:
            raise RuntimeError("deliberate interruption")
        return actual(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(from_dense, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            from_dense.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = from_dense.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left = _checkpoint(Path(config.train.output_dir) / "last.pt")
    right = _checkpoint(tmp_path / "interrupted/last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_geco_step_size_is_recorded_and_changes_the_run_identity(tmp_path: Path) -> None:
    _prepare(tmp_path)
    default = from_dense.train(_config(tmp_path, "full"))
    config = _config(tmp_path, "full")
    slow = from_dense.FromDenseConfig(
        replace(config.train, output_dir=str(tmp_path / "slow-geco")),
        config.init_dir,
        geco_step_size=0.001,
        geco_lambda_init=50.0,
    )
    completion = from_dense.train(slow)
    record = read_json(tmp_path / "slow-geco/config.json")
    assert record["geco"]["step_size"] == 0.001
    assert record["geco"]["lambda_init"] == 50.0
    first = next(
        json.loads(line)
        for line in (tmp_path / "slow-geco/history.jsonl").read_text().splitlines()
        if "geco/log10_penalty_weight" in line
    )
    # One update from log(50); the multiplier moves by at most step * |violation|.
    assert abs(first["geco/log10_penalty_weight"] + math.log10(50.0)) < 0.05
    assert completion["identity"] != default["identity"]
    history = (tmp_path / "slow-geco/history.jsonl").read_text().splitlines()
    assert any("geco/log10_penalty_weight" in line for line in history)


def test_sparse_budget_is_independent_of_the_dense_reference_budget(tmp_path: Path) -> None:
    exploratory = runner.TrainConfig("unused", "unused", "full", calibration_dir="unused")
    assert from_dense.FromDenseConfig(exploratory, "unused", sparse_steps=200_000).budget == 200_000
    with pytest.raises(ValueError, match="100000 or 200000"):
        from_dense.FromDenseConfig(exploratory, "unused", sparse_steps=150_000)
    _prepare(tmp_path)
    config = _config(tmp_path, "full")
    longer = from_dense.FromDenseConfig(
        replace(config.train, output_dir=str(tmp_path / "longer")), config.init_dir, sparse_steps=6
    )
    completion = from_dense.train(longer)
    assert completion["steps"] == 6
    assert _checkpoint(tmp_path / "longer/last.pt")["step"] == 6
    record = read_json(tmp_path / "longer/config.json")
    assert record["scientific_config"]["sparse_steps"] == 6
    assert record["scientific_config"]["steps"] == 4
