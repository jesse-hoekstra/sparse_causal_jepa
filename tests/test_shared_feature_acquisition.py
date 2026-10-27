import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from test_overcomplete_forward import batch_for, gained_model
from torch import Tensor
from torch.nn import functional as F

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import initialize_overcomplete
from diagnostics.overcomplete_acquisition.run import Config as ParentConfig
from diagnostics.overcomplete_acquisition.run import run as run_parent
from diagnostics.overcomplete_replication.run import Config as ReplicaConfig
from diagnostics.overcomplete_replication.run import run as run_replica
from diagnostics.shared_feature_acquisition import projection as projection_module
from diagnostics.shared_feature_acquisition import run as runner
from diagnostics.shared_feature_acquisition.projection import (
    apply_projection,
    fit_projection,
    hidden_features,
)
from diagnostics.shared_feature_acquisition.source import compare_audit, load_inputs
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.distributed import GlobalBatchSampler


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_projection_matches_independent_group_covariances_and_matched_code_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = gained_model()
    initialize_overcomplete(model, 17)
    encoder = model.parameter_encoder
    assert isinstance(encoder, TransitionParameterEncoder)
    generator = torch.Generator().manual_seed(173)
    shared = torch.randn(64, 12, generator=generator, dtype=torch.float64)
    mixing = torch.randn(12, 128, generator=generator, dtype=torch.float64)
    h = (
        (
            (shared @ mixing)[:, None]
            + 0.7 * torch.randn(64, 10, 128, generator=generator, dtype=torch.float64)
        )
        .float()
        .double()
    )
    train = {
        "online": torch.zeros(64, 10, 12, 6),
        "target": torch.zeros(64, 10, 12, 6),
        "actions": torch.zeros(64, 10, 11, 16),
    }

    def extract(
        given: TransitionParameterEncoder, inputs: TensorSplit, *, batch_size: int = 64
    ) -> Tensor:
        assert given is encoder
        assert inputs is train
        assert batch_size == 64
        return h

    monkeypatch.setattr(projection_module, "hidden_features", extract)
    original = copy.deepcopy(encoder.state_dict())
    fixed = [
        module_fingerprint(module)
        for module in (model.predictor, model.online_encoder, model.target_encoder)
    ]
    rng = torch.get_rng_state()
    projection = fit_projection(encoder, train)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == 1
    torch.testing.assert_close(encoder.state_dict(), original, rtol=0, atol=0)
    # Independent per-group covariance calculation catches missing W/R or using
    # sample counts from individual contexts for group-mean covariance.
    within = torch.stack([torch.cov(group.T) for group in h]).mean(0)
    between = torch.cov(h.mean(1).T)
    total = torch.cov(h.flatten(0, 1).T)
    shared_covariance = between - within / 10
    eigenvalues, eigenvectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(total))  # pyright: ignore[reportUnknownMemberType]
    whitening = (
        eigenvectors / eigenvalues.clamp_min(eigenvalues.max() * 1e-6).sqrt()
    ) @ eigenvectors.T
    values, vectors = cast(
        tuple[Tensor, Tensor],
        torch.linalg.eigh(whitening @ shared_covariance @ whitening),  # pyright: ignore[reportUnknownMemberType]
    )
    modes = vectors[:, -12:].clone()
    for index in range(12):
        if modes[modes[:, index].abs().argmax(), index] < 0:
            modes[:, index].neg_()
    expected_weight = modes.T @ whitening
    old = F.linear(h.float(), original["head.4.weight"], original["head.4.bias"]).double()
    mean = h.mean((0, 1))
    projected = (h - mean) @ expected_weight.T
    expected_weight *= (
        old.flatten(0, 1).var(0, correction=0).mean().sqrt()
        / projected.flatten(0, 1).var(0, correction=0).mean().sqrt()
    )
    expected_bias = old.mean((0, 1)) - expected_weight @ mean
    torch.testing.assert_close(projection.weight, expected_weight.float(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(projection.bias, expected_bias.float(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.tensor(projection.receipt["shared_generalized_eigenvalues"], dtype=torch.float64),
        values,
        rtol=1e-10,
        atol=1e-12,
    )
    after = F.linear(h.float(), projection.weight, projection.bias).double()
    torch.testing.assert_close(after.mean((0, 1)), old.mean((0, 1)), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        after.flatten(0, 1).var(0, correction=0).mean().sqrt(),
        old.flatten(0, 1).var(0, correction=0).mean().sqrt(),
        rtol=1e-6,
        atol=1e-7,
    )
    apply_projection(model, projection)
    for name, value in encoder.state_dict().items():
        if name not in {"head.4.weight", "head.4.bias"}:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert fixed == [
        module_fingerprint(module)
        for module in (model.predictor, model.online_encoder, model.target_encoder)
    ]


def test_hidden_features_use_each_train_context_only_and_reject_labels() -> None:
    model = gained_model()
    initialize_overcomplete(model, 17)
    encoder = model.parameter_encoder
    assert isinstance(encoder, TransitionParameterEncoder)
    train = batch_for(model)
    before = module_fingerprint(encoder)
    h = hidden_features(encoder, train, batch_size=1)
    assert h.shape == (2, 10, 128)
    assert not h.requires_grad
    states, actions = (
        train["online"][:, :, :10].flatten(0, 1),
        train["actions"][:, :, :9].flatten(0, 1),
    )
    expected = (
        encoder.head[:4](
            encoder.transition(torch.cat((states[:, :-1], actions, states[:, 1:]), -1)).mean(1)
        )
        .reshape(2, 10, 128)
        .double()
    )
    torch.testing.assert_close(h, expected, rtol=1e-5, atol=1e-7)
    changed = {name: value.clone() for name, value in train.items()}
    changed["target"].fill_(float("nan"))
    changed["online"][:, :, 10:] += 100
    changed["actions"][:, :, 9:] += 100
    torch.testing.assert_close(hidden_features(encoder, changed, batch_size=1), h, rtol=0, atol=0)
    changed["online"][:, 3, :10] += 10
    other = hidden_features(encoder, changed, batch_size=1)
    indices = [index for index in range(10) if index != 3]
    torch.testing.assert_close(other[:, indices], h[:, indices], rtol=0, atol=0)
    assert not torch.equal(other[:, 3], h[:, 3])
    with pytest.raises(ValueError, match="no theta or IDs"):
        fit_projection(encoder, {**train, "theta": torch.zeros(2, 6)})
    assert module_fingerprint(encoder) == before


def test_projection_rejects_different_encoder_before_mutating_common_modules() -> None:
    model = gained_model()
    initialize_overcomplete(model, 17)
    encoder = model.parameter_encoder
    assert isinstance(encoder, TransitionParameterEncoder)
    projection = fit_projection(encoder, batch_for(model))
    with torch.no_grad():
        next(encoder.parameters()).add_(0.01)
    before = module_fingerprint(encoder)
    with pytest.raises(ValueError, match="different encoder"):
        apply_projection(model, projection)
    assert module_fingerprint(encoder) == before


@pytest.fixture(scope="module")
def parent_sources(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[int, Path]]:
    root = tmp_path_factory.mktemp("shared-feature-parents")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = source_with_initialization(root / "source")
        parents = {7: root / "first", 17: root / "replica"}
        run_parent(
            ParentConfig(
                str(source),
                str(parents[7]),
                seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=8,
                verification=True,
            )
        )
        run_replica(
            ReplicaConfig(
                str(source),
                str(parents[17]),
                source_seed=7,
                seed=17,
                evaluation_seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=8,
                verification=True,
            )
        )
        return source, parents
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("seed", [7, 17])
def test_saved_initial_only_train_projection_fresh_adam_and_exact_resume(
    parent_sources: tuple[Path, dict[int, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    source, parents = parent_sources
    parent = parents[seed]
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            path = Path(file).resolve()
            assert path.name != "audit_inputs.pt", "Training theta was loaded"
            assert path != parent / "last.pt", "Trained parent weights/Adam were loaded"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    inputs = load_inputs(source, parent, seed, source_seed=7, verification=True)
    audit = tmp_path / "audit.pt"
    reference = {
        "seed": seed,
        "train_only": True,
        "source_initial_encoder_sha256": inputs.projection.receipt["source_encoder_sha256"],
        "weight": inputs.projection.weight.clone(),
        "bias": inputs.projection.bias.clone(),
    }
    torch.save(reference, audit)
    comparison = compare_audit(inputs.model, inputs.projection, inputs.bundle, audit, seed)
    assert comparison["code_rms_difference"] == 0
    reference["weight"] += 1
    torch.save(reference, audit)
    with pytest.raises(ValueError, match="numerical tolerance"):
        compare_audit(inputs.model, inputs.projection, inputs.bundle, audit, seed)
    parent_initial = cast(
        dict[str, Any], torch.load(parent / "initial_modules.pt", weights_only=False)
    )
    initial_model = copy.deepcopy(inputs.model.state_dict())
    for name, value in inputs.model.parameter_encoder.state_dict().items():
        if name not in {"head.4.weight", "head.4.bias"}:
            torch.testing.assert_close(
                value, parent_initial["parameter_encoder"][name], rtol=0, atol=0
            )
    torch.testing.assert_close(
        inputs.model.predictor.state_dict(), parent_initial["predictor"], rtol=0, atol=0
    )
    config = runner.Config(
        str(source),
        str(parent),
        str(tmp_path / "full"),
        source_seed=7,
        seed=seed,
        evaluation_seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=8,
        verification=True,
    )
    fixed = runner.fixed_hashes(inputs.model)
    loss = runner.training_loss
    count = 0

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        if count == 1:
            torch.testing.assert_close(model.state_dict(), initial_model, rtol=0, atol=0)
            zero = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
            assert zero["optimizer"]["state"] == {}
        assert set(batch) == {"online", "target", "actions"}
        epoch, position = divmod(count - 1, inputs.bundle.train_groups // inputs.bundle.batch_size)
        selected = list(
            GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, seed, epoch)
        )[position]
        torch.testing.assert_close(
            batch,
            {name: value[selected] for name, value in inputs.bundle.train.items()},
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(seed * 100000 + count).get_state()
        )
        assert runner.fixed_hashes(model) == fixed
        return loss(model, batch, objective)

    monkeypatch.setattr(runner, "training_loss", capture)
    full = runner.run(config)
    assert count == 3
    final = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    assert all(int(value["step"]) == 3 for value in final["optimizer"]["state"].values())
    for prefix in ("parameter_encoder.", "predictor."):
        assert any(
            not torch.equal(value, initial_model[name])
            for name, value in final["model"].items()
            if name.startswith(prefix)
        )
    count = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupt projection training")
        return loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    with pytest.raises(RuntimeError, match="interrupt projection training"):
        runner.run(resumed_config)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", loss)
    resumed = runner.run(resumed_config, resume=True)
    replay = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for name in ("model", "optimizer"):
        torch.testing.assert_close(final[name], replay[name], rtol=0, atol=0)
    torch.testing.assert_close(final["rng"]["torch"], replay["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="overlap|source|disjoint"):
        runner.run(replace(config, output_dir=str(parent / "bad-output")))
    assert not (parent / "bad-output").exists()
    persisted = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/projection.pt", weights_only=False)
    )
    persisted["weight"][0, 0] += 1
    torch.save(persisted, tmp_path / "resumed/projection.pt")
    with pytest.raises(ValueError, match="projection changed"):
        runner.run(resumed_config, resume=True)
