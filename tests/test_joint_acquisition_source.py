import copy
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_encoder_gradient_source import refresh_hashes, write_source_fixture

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.joint_acquisition.provenance import (
    historical_training_identity,
    validate_training_semantics,
)
from diagnostics.joint_acquisition.source import load_primary_source


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_primary_endpoint_and_saved_optimizer_parameter_order_are_enforced(tmp_path: Path) -> None:
    source = write_source_fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="30000-step"):
        load_primary_source(source, 7)
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, optimizer = build_source_model(bundle, torch.device("cpu"))
    before = copy.deepcopy(bundle.checkpoint)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for name, value in before["model"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, bundle.checkpoint["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(
        before["optimizer"]["state"], bundle.checkpoint["optimizer"]["state"], rtol=0, atol=0
    )
    path = source / "seed-7/prediction_1/last.pt"
    payload = cast(dict[str, Any], torch.load(path, weights_only=False))
    payload["optimizer"]["param_groups"][0]["betas"] = (0.5, 0.999)
    torch.save(payload, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="optimizer semantics"):
        load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)


def test_archived_training_identity_allows_only_documentation_changes_in_training(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    archive = tmp_path / "archive"
    for name in ("src", "scripts", "diagnostics"):
        shutil.copytree(
            repository / name, archive / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    source = write_source_fixture(tmp_path / "source")
    bundle = load_primary_source(
        source, 7, expected_step=2, expected_checkpoint_sha256=None, training_source_root=archive
    )
    assert historical_training_identity(archive) == bundle.protocol["code_identity"]
    source_path = archive / "src/scjepa/data/controlled_vector.py"
    source_path.write_text(source_path.read_text() + "\n# Documentation-only difference.\n")
    validate_training_semantics(archive, repository)
    source_path.write_text(source_path.read_text() + "\nchanged_training_constant = 1\n")
    with pytest.raises(ValueError, match="training semantics"):
        validate_training_semantics(archive, repository)


def test_actual_source_encoder_is_required_for_compatible_code_comparison(tmp_path: Path) -> None:
    from test_frozen_predictor_source import write_source_fixture as frozen_fixture

    from diagnostics.frozen_predictor_control.run import Config, prepare
    from diagnostics.frozen_predictor_control.source import load_source_bundle
    from diagnostics.joint_acquisition.source import load_comparison_sources
    from diagnostics.multiquery_control.run import sha
    from scjepa.experiments.storage import write_json

    bundle_dir = frozen_fixture(tmp_path / "frozen")
    bundle = load_source_bundle(bundle_dir)
    original = tmp_path / "frozen/original/washout/last.pt"
    output = tmp_path / "current"
    output.mkdir()
    config = Config(
        source_dir=str(bundle_dir), output_dir=str(output), seed=7, steps=3, batch_size=2
    )
    protocol = prepare(config, bundle, resume=False)
    payload = cast(dict[str, Any], torch.load(original, weights_only=False))
    current = output / "seed-7/last.pt"
    current.parent.mkdir()
    torch.save(
        {
            "identity": protocol["identity"],
            "model": payload["model"],
            "seed": 7,
            "step": 3,
            "frozen_hashes": bundle.manifest["frozen_hashes"],
        },
        current,
    )
    write_json(
        current.parent / "completed.json",
        {
            "identity": protocol["identity"],
            "hashes": {"last.pt": sha(current)},
        },
    )
    rng = torch.get_rng_state().clone()
    sources = load_comparison_sources(
        bundle_dir, original, current, expected_current_step=3, expected_current_sha256=None
    )
    assert torch.equal(rng, torch.get_rng_state())
    for first, second in zip(
        sources.source_model.parameters(), sources.current_model.parameters(), strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    wrong_original = tmp_path / "wrong.pt"
    torch.save({"reference_code_stats": bundle.manifest["reference_code_stats"]}, wrong_original)
    with pytest.raises(ValueError, match="actual all-six|Actual all-six"):
        load_comparison_sources(
            bundle_dir,
            wrong_original,
            current,
            expected_current_step=3,
            expected_current_sha256=None,
        )
