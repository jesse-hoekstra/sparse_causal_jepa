from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_predictor_transfer_control import transfer_sources as transfer_sources
from torch import Tensor

from diagnostics.joint_acquisition.baseline import predict_matched, response_report
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_use_precision.data import build_validation
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.predictor_transfer_control.run import Config as TransferConfig
from diagnostics.predictor_transfer_control.run import run as run_transfer
from diagnostics.predictor_transfer_control.source import load_inputs as load_transfer
from diagnostics.predictor_transfer_precision import run as runner
from diagnostics.predictor_transfer_precision.evaluation import evaluate
from diagnostics.predictor_transfer_precision.source import Arm, load_inputs
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def precision_sources(
    transfer_sources: tuple[Path, Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path, Path]:
    """Use actual tiny predictor fits and a genuine unchanged-law precision cache."""
    source, endpoint, audit = transfer_sources
    root = tmp_path_factory.mktemp("transfer-precision")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for arm in ("spartan", "mlp"):
            run_transfer(
                TransferConfig(
                    str(source),
                    str(endpoint),
                    str(audit),
                    str(root / arm),
                    arm,
                    seed=7,
                    steps=2,
                    groups=2,
                    bootstrap_samples=16,
                    verification=True,
                )
            )
        original = load_transfer(source, endpoint, audit, 7, "mlp", verification=True)
        data = build_validation(
            original.reference_model,
            original.bundle.evaluation["metadata"],
            groups=5,
            seed=87593,
            batch_size=3,
        )
        precision = root / "precision"
        precision.mkdir()
        torch.save(data, precision / "validation_data.pt")
        protocol = {
            "identity": "genuine-tiny-transfer-precision",
            "code_identity": precision_code_identity(),
            "source_hashes": {
                str((source / name).resolve()): sha(source / name)
                for name in (
                    "protocol.json",
                    "evaluation_data.pt",
                )
            },
        }
        write_json(precision / "protocol.json", protocol)
        write_json(
            precision / "results.json",
            {"identity": protocol["identity"], "scope": "genuine unchanged-law fixture cache"},
        )
        write_json(
            precision / "completed.json",
            {
                "identity": protocol["identity"],
                "hashes": {
                    name: sha(precision / name)
                    for name in (
                        "protocol.json",
                        "results.json",
                        "validation_data.pt",
                    )
                },
            },
        )
        return root / "spartan", root / "mlp", source, precision
    finally:
        torch.set_num_threads(previous)


def test_fixed_sources_restore_all_three_without_train_reads_or_optimizer_fitting(
    precision_sources: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spartan, mlp, source, precision = precision_sources
    original = cast(Callable[..., Any], torch.load)
    seen: list[str] = []

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            seen.append(name)
            assert name not in {"train_inputs.pt", "audit_inputs.pt", "train_codes.pt"}
        return original(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    arms: tuple[Arm, ...] = ("spartan", "mlp", "reference")
    all_inputs = {
        arm: load_inputs(
            spartan if arm == "spartan" else mlp,
            source,
            precision,
            arm=arm,
            checkpoint_sha256=sha((spartan if arm == "spartan" else mlp) / "last.pt"),
            verification=True,
        )
        for arm in arms
    }
    inputs = all_inputs["spartan"]
    models = {arm: value.model for arm, value in all_inputs.items()}
    assert tuple(models) == ("spartan", "mlp", "reference")
    assert set(inputs.original_validation) == {"metadata", "validation"}
    assert set(inputs.precision_data) == {"metadata", "validation"}
    assert inputs.evaluation_seed == 7
    assert inputs.original_bootstrap_samples == 16
    for arm, root in (("spartan", spartan), ("mlp", mlp)):
        saved = cast(dict[str, Any], original(root / "last.pt", weights_only=False))
        expected = {
            name: value.double() if isinstance(value, Tensor) else value
            for name, value in saved["model"].items()
        }
        torch.testing.assert_close(models[arm].state_dict(), expected, rtol=0, atol=0)
        assert all(not p.requires_grad and p.grad is None for p in models[arm].parameters())
        assert not models[arm].training
    for name in ("parameter_encoder", "online_encoder", "target_encoder"):
        torch.testing.assert_close(
            getattr(models["spartan"], name).state_dict(),
            getattr(models["mlp"], name).state_dict(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            getattr(models["reference"], name).state_dict(),
            getattr(models["mlp"], name).state_dict(),
            rtol=0,
            atol=0,
        )
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="SHA"):
        load_inputs(
            spartan,
            source,
            precision,
            arm="spartan",
            checkpoint_sha256="0" * 64,
            verification=True,
        )


@pytest.mark.parametrize("arm", ["spartan", "mlp", "reference"])
def test_chunked_six_code_assay_matches_original_global_group_bootstrap_and_direct_identity(
    precision_sources: tuple[Path, Path, Path, Path],
    arm: Arm,
) -> None:
    spartan, mlp, source, precision = precision_sources
    endpoint = spartan if arm == "spartan" else mlp
    inputs = load_inputs(
        endpoint,
        source,
        precision,
        arm=arm,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        verification=True,
    )
    model = inputs.model
    frozen = model_fingerprint(model)
    progress: list[int] = []
    rng = torch.get_rng_state()
    report, raw, vectors = evaluate(
        model,
        inputs.precision_data,
        chunk_size=2,
        bootstrap_samples=32,
        seed=7,
        progress=progress.append,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert progress == [2, 4, 5]
    assert model_fingerprint(model) == frozen
    batch, _, _ = matched_batch(model, inputs.precision_data, "validation", tuple(range(5)))
    prediction = predict_matched(model, batch)
    expected, expected_raw = response_report(
        prediction.predictions, prediction.changed_predictions, batch, bootstrap_samples=32, seed=7
    )
    verify_replay(report, expected)
    for name, value in expected_raw.items():
        torch.testing.assert_close(raw[name], value, rtol=1e-9, atol=2e-13)
    base, changed, target, changed_target = (
        vectors[name]
        for name in (
            "factual_predictions",
            "counterfactual_predictions",
            "factual_targets",
            "counterfactual_targets",
        )
    )
    learned = changed - base[:, :, None]
    exact = changed_target - target[:, :, None]
    residual = base - target
    axes = (1, 3, 4, 5)
    gain = (2 * learned * exact - learned.square()).mean(axes)
    cross = (2 * residual[:, :, None] * learned).mean(axes)
    benefit = (
        (base[:, :, None] - changed_target).square() - (changed - changed_target).square()
    ).mean(axes)
    torch.testing.assert_close(benefit, gain - cross, rtol=1e-9, atol=2e-13)
    torch.testing.assert_close(raw["predictive_benefit"], benefit, rtol=0, atol=0)
    assert torch.equal(raw["group_indices"], torch.arange(5))
    assert torch.equal(raw["query_indices"], torch.tensor([[5, 6, 7, 8, 9]]))
    replay, _, _ = evaluate(
        model,
        inputs.original_validation,
        chunk_size=1,
        bootstrap_samples=inputs.original_bootstrap_samples,
        seed=inputs.evaluation_seed,
    )
    verify_replay(replay, inputs.archived_report)


def test_readonly_runner_replays_original_validation_and_resumes_without_model_selection(
    precision_sources: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, endpoint, source, precision = precision_sources
    config = runner.Config(
        str(endpoint),
        str(source),
        str(precision),
        str(tmp_path / "full"),
        "mlp",
        sha(endpoint / "last.pt"),
        chunk_size=2,
        bootstrap_samples=16,
        evaluation_seed=7,
        verification=True,
    )
    original = runner.evaluate
    seen: list[int] = []
    interrupt = False

    def checked(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        assert set(data) == {"metadata", "validation"}
        count = len(data["validation"]["theta"])
        seen.append(count)
        assert count in (2, 5)
        assert seed == 7
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        if interrupt and count == 5:
            raise RuntimeError("precision operational interruption")
        return original(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )

    monkeypatch.setattr(runner, "evaluate", checked)
    full = runner.run(config)
    assert seen == [2, 5]
    assert full["arm"] == "mlp"
    completion = cast(dict[str, Any], read_json(tmp_path / "full/completed.json"))
    assert len(completion["hashes"]) == 5
    assert all(
        sha(tmp_path / "full" / name) == value for name, value in completion["hashes"].items()
    )
    saved = cast(dict[str, Any], torch.load(tmp_path / "full/measurements.pt", weights_only=False))
    assert saved["raw"]["group_indices"].tolist() == list(range(5))
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    seen.clear()
    interrupt = True
    with pytest.raises(RuntimeError, match="operational interruption"):
        runner.run(resumed_config)
    assert seen == [2, 5]
    assert not (tmp_path / "resumed/completed.json").exists()
    previous = cast(dict[str, Any], read_json(tmp_path / "resumed/tracking.json"))[
        "last_event_step"
    ]
    interrupt = False
    seen.clear()
    resumed = runner.run(resumed_config, resume=True)
    assert seen == [2, 5]
    assert full["report"] == resumed["report"]
    assert full["chart_provenance"] == resumed["chart_provenance"]
    assert (
        cast(dict[str, Any], read_json(tmp_path / "resumed/tracking.json"))["last_event_step"]
        > previous
    )
    seen.clear()
    runner.run(resumed_config, resume=True)
    assert seen == []
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(endpoint / "bad-output"))
    with pytest.raises(ValueError, match="SHA256"):
        replace(config, checkpoint_sha256="bad-pin")
    with pytest.raises(ValueError, match="fixed"):
        replace(config, verification=False)
    changed = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/measurements.pt", weights_only=False)
    )
    changed["raw"]["predictive_benefit"][0, 0] += 1
    torch.save(changed, tmp_path / "resumed/measurements.pt")
    with pytest.raises(ValueError, match="changed"):
        runner.run(resumed_config, resume=True)
