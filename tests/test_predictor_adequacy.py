import copy
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_frozen_predictor_source import write_source_fixture

from diagnostics.frozen_predictor_control.source import (
    build_fresh_model,
    load_source_bundle,
    manifest_identity,
)
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_adequacy.run import Config, fixed_hashes, restore_source, run


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def source_fixture(root: Path) -> tuple[Path, Path]:
    bundle_dir = write_source_fixture(root)
    bundle = load_source_bundle(bundle_dir)
    checkpoint = root / "original/washout/last.pt"
    payload = cast(dict[str, Any], torch.load(checkpoint, weights_only=False))
    model = build_fresh_model(bundle, 0, torch.device("cpu"))
    model.load_state_dict(payload["model"])
    model.predictor.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for state in optimizer.state.values():
        state["step"].fill_(10000)
    payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, checkpoint)
    manifest = bundle.manifest
    manifest["provenance"]["sha256"]["checkpoint"] = sha(checkpoint)
    manifest["identity"] = manifest_identity(manifest)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
    return bundle_dir, checkpoint


def test_fixed_encoder_source_and_adam_moments_survive_predictor_refit(tmp_path: Path) -> None:
    source, checkpoint = source_fixture(tmp_path / "source")
    bundle = load_source_bundle(source)
    model, optimizer = restore_source(bundle, checkpoint, torch.device("cpu"), verification=True)
    before = checkpoint.read_bytes()
    frozen = fixed_hashes(model)
    source_state = copy.deepcopy(optimizer.state_dict())
    original = cast(dict[str, Any], torch.load(checkpoint, weights_only=False))
    torch.testing.assert_close(source_state, original["optimizer"], rtol=0, atol=0)
    source_model = copy.deepcopy(model.state_dict())
    assert all(not p.requires_grad for p in model.parameter_encoder.parameters())
    config = Config(
        source_dir=str(source),
        checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "out"),
        verification=True,
        steps=2,
        groups=2,
        bootstrap_samples=16,
    )
    result = run(config)
    saved = cast(dict[str, Any], torch.load(tmp_path / "out/last.pt", weights_only=False))
    model.load_state_dict(saved["model"])
    assert fixed_hashes(model) == frozen
    assert checkpoint.read_bytes() == before
    q_count = len(tuple(model.parameter_encoder.parameters()))
    for index in range(q_count):
        torch.testing.assert_close(
            saved["optimizer"]["state"][index], source_state["state"][index], rtol=0, atol=0
        )
    assert any(
        not torch.equal(value, saved["model"][name])
        for name, value in source_model.items()
        if name.startswith("predictor.")
    )
    assert {row["step"] for row in result["records"]} == {0, 2}
    assert len(result["records"][-1]["response"]["per_parameter"]) == 6
    assert run(config, resume=True) == result


def test_optimizer_order_and_scientific_budget_are_checked(tmp_path: Path) -> None:
    source, checkpoint = source_fixture(tmp_path / "source")
    payload = cast(dict[str, Any], torch.load(checkpoint, weights_only=False))
    payload["optimizer"]["param_groups"][0]["params"].reverse()
    torch.save(payload, checkpoint)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["provenance"]["sha256"]["checkpoint"] = sha(checkpoint)
    manifest["identity"] = manifest_identity(manifest)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parameter order"):
        restore_source(
            load_source_bundle(source), checkpoint, torch.device("cpu"), verification=True
        )
    with pytest.raises(ValueError, match="10000 updates"):
        Config(
            source_dir=str(source),
            checkpoint=str(checkpoint),
            output_dir=str(tmp_path / "out"),
            steps=20000,
        )
