import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_encoder_gradient_source import refresh_hashes
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.joint_acquisition.baseline import response_report
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_acquisition.evaluation import predict_matched
from diagnostics.overcomplete_acquisition.run import Config as AcquisitionConfig
from diagnostics.overcomplete_acquisition.run import run as acquisition_run
from diagnostics.overcomplete_use_precision import run as runner
from diagnostics.overcomplete_use_precision.data import build_validation, verify_independence
from diagnostics.overcomplete_use_precision.evaluation import (
    direct_decomposition,
    evaluate,
    join_batches,
)
from diagnostics.overcomplete_use_precision.run import Config, load_source, run, verify_replay
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.storage import read_json
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
def source_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("precision-source")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = source_with_initialization(root / "source")
        bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
        model, _ = build_source_model(bundle, torch.device("cpu"))
        model.eval()
        # The shared fixture predates per-split seed metadata. Supply genuinely
        # distinct scientific-law evaluation draws before training the endpoint.
        evaluation: EvaluationData = {}
        for index, name in enumerate(("calibration", "validation", "test")):
            generated = build_validation(
                model,
                bundle.evaluation["metadata"],
                groups=4,
                seed=21731 + 1009 * index,
                batch_size=4,
            )
            if index == 0:
                evaluation["metadata"] = generated["metadata"]
            evaluation[name] = generated["validation"]
        torch.save(evaluation, source / "evaluation_data.pt")
        refresh_hashes(source)
        endpoint = root / "endpoint"
        acquisition_run(
            AcquisitionConfig(
                str(source),
                str(endpoint),
                seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        return source, endpoint
    finally:
        torch.set_num_threads(previous)


def configuration(source_pair: tuple[Path, Path], output: Path) -> Config:
    source, endpoint = source_pair
    return Config(
        str(endpoint),
        str(source),
        str(output),
        groups=4,
        seed=99391,
        chunk_size=2,
        bootstrap_samples=32,
        verification=True,
    )


def test_direct_vector_identity_includes_nonzero_mean_residual_bias() -> None:
    generator = torch.Generator().manual_seed(389)
    target = torch.randn(7, 1, 5, 2, 6, generator=generator, dtype=torch.float64)
    base = target + 0.2
    delta = torch.full((7, 1, 6, 5, 2, 6), 0.01, dtype=torch.float64)
    changed_target, changed = target[:, :, None] + delta, base[:, :, None] + delta
    report = direct_decomposition(base, changed, target, changed_target)
    torch.testing.assert_close(
        report["finite_response_gain"],
        torch.full((7, 6), 0.0001, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-16,
    )
    torch.testing.assert_close(
        report["twice_residual_dot_learned_response"],
        torch.full((7, 6), 0.004, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-16,
    )
    assert bool((report["benefit_direct_squared_error"] < 0).all())
    torch.testing.assert_close(
        report["benefit_direct_squared_error"],
        torch.full((7, 6), -0.0039, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-16,
    )
    # Non-exact learned responses also satisfy the vector-derived identity.
    changed += torch.randn(changed.shape, generator=generator, dtype=torch.float64) * 0.03
    arbitrary = direct_decomposition(base, changed, target, changed_target)
    learned, exact = changed - base[:, :, None], changed_target - target[:, :, None]
    cross = 2 * ((base - target)[:, :, None] * learned).mean((1, 3, 4, 5))
    gain = (exact.square() - (learned - exact).square()).mean((1, 3, 4, 5))
    torch.testing.assert_close(
        arbitrary["twice_residual_dot_learned_response"], cross, rtol=0, atol=0
    )
    torch.testing.assert_close(
        arbitrary["benefit_direct_squared_error"], gain - cross, rtol=1e-12, atol=1e-16
    )


def test_new_validation_exact_generator_law_and_identity_only_test_overlap(
    source_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    model, old, _, _, _ = load_source(configuration(source_pair, tmp_path / "unused"))
    model.double()
    fixed = model_fingerprint(model)
    progress: list[int] = []
    rng = torch.get_rng_state()
    data = build_validation(
        model, old["metadata"], groups=4, seed=99391, batch_size=3, progress=progress.append
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert set(data) == {"metadata", "validation"}
    assert progress == [3, 4]
    generator = torch.Generator().manual_seed(99391)
    theta = 0.75 * (2 * torch.rand(4, 6, dtype=torch.float64, generator=generator) - 1)
    initial = torch.randn(4, 10, 6, dtype=torch.float64, generator=generator)
    gaussian = torch.randn(4, 10, 11, 10, dtype=torch.float64, generator=generator)
    coupling = 2 * torch.rand(4, 10, 11, 6, dtype=torch.float64, generator=generator) - 1
    actions = dgp.encode_controls(torch.cat((gaussian, coupling), -1))
    physical = dgp.simulate(initial, theta[:, None], actions)
    split = data["validation"]
    for name, expected in (
        ("theta", theta),
        ("true_actions", actions),
        ("true_states", physical),
        ("actions", actions.float()),
    ):
        torch.testing.assert_close(split[name], expected, rtol=0, atol=0)
    observations = VectorObservationMap(
        matrix=old["metadata"]["observation_matrix"], dimension=6
    ).forward(physical)
    cache_model = copy.deepcopy(model).float()
    assert split["states_online"].dtype == split["states_target"].dtype == torch.float32
    with torch.no_grad():
        for key, encoder in (
            ("states_online", cache_model.online_encoder),
            ("states_target", cache_model.target_encoder),
        ):
            expected = encoder.encode_sequence(observations.float().flatten(0, 1)).reshape(
                4, 10, 12, 6
            )
            torch.testing.assert_close(split[key], expected, rtol=1e-6, atol=1e-7)
    assert fixed == model_fingerprint(model)
    assert verify_independence(data, old)["exact_theta_overlaps"] == {
        name: 0 for name in ("calibration", "validation", "test")
    }
    repeated = copy.deepcopy(data)
    repeated["metadata"]["seed"] = int(old["metadata"]["seed"]) + 1009
    with pytest.raises(ValueError, match="seed overlaps"):
        verify_independence(repeated, old)
    repeated["metadata"]["seed"] = 99391
    repeated["validation"]["theta"][0].copy_(old["test"]["theta"][0])
    with pytest.raises(ValueError, match="archived test"):
        verify_independence(repeated, old)


def test_chunks_match_single_original_bootstrap_and_direct_vector_aggregation(
    source_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    model, old, _, archived, _ = load_source(configuration(source_pair, tmp_path / "unused"))
    model.double()
    data = build_validation(model, old["metadata"], groups=4, seed=99391, batch_size=2)
    fixed = model_fingerprint(model)
    batch, _, _ = matched_batch(model, data, "validation", (0, 1, 2, 3))
    predictions = predict_matched(model, batch)
    expected, expected_raw = response_report(
        predictions.predictions,
        predictions.changed_predictions,
        batch,
        bootstrap_samples=32,
        seed=7,
    )
    for chunk_size in (1, 2, 3, 4):
        progress: list[int] = []
        report, raw, vectors = evaluate(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=32,
            seed=7,
            progress=progress.append,
        )
        assert progress[-1] == 4
        assert report["group_indices"] == [0, 1, 2, 3]
        assert report["donors"] == [0]
        assert report["query_indices"] == [[5, 6, 7, 8, 9]]
        assert report["bootstrap"] == expected["bootstrap"]
        verify_replay(report, expected)
        indices = torch.randint(4, (32, 4), generator=torch.Generator().manual_seed(7))
        cross = raw["twice_residual_dot_learned_response"]
        bounds = torch.quantile(
            cross[indices].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0
        )
        for factor, row in enumerate(
            report["decomposition_intervals"]["twice_residual_dot_learned_response"]
        ):
            torch.testing.assert_close(
                torch.tensor(row["interval_95"], dtype=torch.float64),
                bounds[:, factor],
                rtol=1e-12,
                atol=1e-15,
            )
        for name, value in expected_raw.items():
            torch.testing.assert_close(raw[name], value, rtol=1e-10, atol=2e-15)
        torch.testing.assert_close(
            vectors["factual_predictions"], predictions.predictions, rtol=1e-12, atol=1e-15
        )
        assert report["direct_vector_identity_max_absolute_error"] < 2e-15
    first, _, _ = matched_batch(model, data, "validation", (0, 1))
    second, _, _ = matched_batch(model, data, "validation", (2, 3))
    joined = join_batches([first, second])
    assert joined.group_indices == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="Repeated group"):
        join_batches([first, first])
    with pytest.raises(ValueError, match="geometry differs"):
        join_batches([first, replace(second, donors=(1,))])
    replay_data = {
        "metadata": old["metadata"],
        "validation": {key: value[:2] for key, value in old["validation"].items()},
    }
    replay, _, _ = evaluate(model, replay_data, chunk_size=1, bootstrap_samples=16, seed=7)
    verify_replay(replay, archived["response"])
    assert fixed == model_fingerprint(model)


def test_audit_authenticates_source_never_optimizes_or_predicts_old_test_and_resumes(
    source_pair: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = configuration(source_pair, tmp_path / "audit")
    _, _, _, _, hashes = load_source(config)
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name not in {"train_inputs.pt", "audit_inputs.pt"}
        return original_load(file, *args, **kwargs)

    def no_optimizer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Read-only audit attempted optimization")

    monkeypatch.setattr(torch, "load", guarded_load)
    monkeypatch.setattr(torch.optim, "Adam", no_optimizer)
    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    original_evaluate = runner.evaluate
    calls: list[int] = []

    def guarded_evaluate(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        assert set(data) == {"metadata", "validation"}
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert seed == 7
        frozen = model_fingerprint(model)
        calls.append(len(data["validation"]["theta"]))
        result = original_evaluate(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )
        assert model_fingerprint(model) == frozen
        return result

    monkeypatch.setattr(runner, "evaluate", guarded_evaluate)
    result = run(config)
    assert calls == [2, 4]
    assert result["groups"] == 4
    replay = cast(dict[str, Any], read_json(tmp_path / "audit/replay.json"))
    assert replay["matched"] is True
    assert all(sha(Path(name)) == digest for name, digest in hashes.items())
    assert run(config, resume=True) == result
    assert calls == [2, 4]
    cache = tmp_path / "audit/validation_data.pt"
    original_bytes = cache.read_bytes()
    try:
        cache.write_bytes(original_bytes + b"tampered")
        with pytest.raises(ValueError, match="[Hh]ash|[Aa]rtifact"):
            run(config, resume=True)
    finally:
        cache.write_bytes(original_bytes)
    source, endpoint = source_pair
    with pytest.raises(ValueError, match="disjoint"):
        Config(str(endpoint), str(source), str(source / "nested"))
