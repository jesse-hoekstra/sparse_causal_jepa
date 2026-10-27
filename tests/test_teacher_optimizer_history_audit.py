import copy
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import pytest
import test_table_settling_control as fixture_module
import torch
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor

from diagnostics import teacher_optimizer_history_audit as audit
from diagnostics.group_latent_control.run import Config as CurriculumConfig
from diagnostics.group_latent_control.run import run as curriculum_run
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control import run as settling
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def test_inherited_fraction_matches_actual_sparse_adam_and_allows_roundoff() -> None:
    table = audit.embedding(torch.ones(4, 6, dtype=torch.float64))
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    before = torch.zeros_like(table.weight)
    for step in range(4):
        optimizer.zero_grad()
        table(torch.tensor([0, 2])).square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        if step == 0:
            before = optimizer.state_dict()["state"][0]["exp_avg_sq"].clone()
    final = optimizer.state_dict()["state"][0]["exp_avg_sq"]
    # Rows 1/3 are never updated, and are explicitly undefined rather than NaN.
    result = audit.inherited_moments(before[[0, 2]], final[[0, 2]], 3, 0.999)
    torch.testing.assert_close(result["fraction"], before[[0, 2]] * 0.999**3 / final[[0, 2]])
    zero = audit.inherited_moments(before[[1, 3]], final[[1, 3]], 3, 0.999)
    assert zero["summary"]["undefined_zero_coordinates"] == 12
    assert zero["summary"]["mean"] is None
    rounded = audit.inherited_moments(torch.ones(2), torch.full((2,), 0.999 - 1e-7), 1, 0.999)
    assert rounded["summary"]["mean"] > 1
    with pytest.raises(ValueError, match="claimed inherited"):
        audit.inherited_moments(torch.ones(2), torch.zeros(2), 1, 0.999)


@pytest.mark.parametrize("reset", [False, True])
def test_direction_matches_actual_sparse_adam_preserving_sources_and_rng(reset: bool) -> None:
    model = make_model().requires_grad_(False).eval()
    _, batch = make_batch(model)
    values = torch.randn(5, 6, dtype=torch.float64)
    table = audit.embedding(values)
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    objective = settling.TableSettlingObjective(objective_config(), "without_parameter_sigreg")
    settling.train_update(model, table, optimizer, batch, torch.tensor([1, 4]), objective)
    values = table.weight.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    state_before = copy.deepcopy(state)
    frozen = model_fingerprint(model)
    model.eval()
    rng = torch.get_rng_state().clone()
    selected = torch.tensor([1, 4])
    direction, _ = audit.next_direction(
        model,
        values[selected],
        audit.optimizer_slice(state, selected),
        batch,
        objective_config(),
        18,
        reset=reset,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    torch.testing.assert_close(state, state_before, rtol=0, atol=0)
    torch.testing.assert_close(table.weight, values, rtol=0, atol=0)
    if reset:
        optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    torch.manual_seed(18)  # pyright: ignore[reportUnknownMemberType]
    settling.train_update(model, table, optimizer, batch, selected, objective)
    torch.testing.assert_close(table.weight[selected] - values[selected], direction, rtol=0, atol=0)
    torch.testing.assert_close(table.weight[[0, 2, 3]], values[[0, 2, 3]], rtol=0, atol=0)
    assert model_fingerprint(model) == frozen
    assert all(p.grad is None for p in model.parameters())


def test_endpoint_never_fits_new_episodes_and_bootstraps_whole_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model().requires_grad_(False).eval()
    _, batch = make_batch(model)
    train = {k: v.repeat(2, 1, 1, 1) for k, v in batch.items()}
    fresh = {k: v.clone() + 0.05 for k, v in train.items()}
    table = audit.embedding(torch.randn(4, 6, dtype=torch.float64))
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    objective = settling.TableSettlingObjective(objective_config(), "without_parameter_sigreg")
    settling.train_update(model, table, optimizer, batch, torch.tensor([0, 1]), objective)
    state = copy.deepcopy(optimizer.state_dict())
    before = model_fingerprint(model)
    original = audit.next_direction
    calls = 0

    def spy(
        model: SCJEPA,
        values: Tensor,
        state: dict[str, Any],
        batch: TensorSplit,
        config: ObjectiveConfig,
        seed: int,
        *,
        reset: bool,
    ) -> tuple[Tensor, dict[str, float]]:
        nonlocal calls
        calls += 1
        torch.testing.assert_close(batch["actions"], train["actions"][:2], rtol=0, atol=0)
        assert not torch.equal(batch["actions"], fresh["actions"][:2])
        return original(model, values, state, batch, config, seed, reset=reset)

    monkeypatch.setattr(audit, "next_direction", spy)
    rng = torch.get_rng_state().clone()
    raw = audit.endpoint_measurements(
        model,
        table.weight.detach(),
        state,
        train,
        fresh,
        objective_config(),
        batch_size=2,
    )
    assert calls == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert model_fingerprint(model) == before
    assert all(p.grad is None for p in model.parameters())
    assert all(v.grad is None for v in fresh.values())
    for permutation in raw["permutations"]:
        assert bool((permutation != torch.arange(4)).all())
    summary = audit.summarize(raw, 2)
    assert set(summary["updates"]) == {"retained", "reset"}
    # Heterogeneous perfectly correlated within-batch changes must resample whole batches.
    changes = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    for fraction in audit.FRACTIONS:
        for name in raw["baseline"]:
            raw["updates"]["reset"][str(fraction)][name] = (
                raw["updates"]["retained"][str(fraction)][name] + changes
            )
    summary = audit.summarize(raw, 2)
    expected = audit.paired_change(
        torch.zeros(2, 2), changes.reshape(2, 2, 2).mean(1), samples=2000, seed=audit.SEED
    )
    observed = summary["reset_minus_retained"]["1.0"]["cached"]
    torch.testing.assert_close(torch.tensor(observed["ci95"]), torch.tensor(expected["ci95"]))
    # Record sampled units directly so an accidental change to per-row resampling fails.
    sizes: list[int] = []
    paired = audit.paired_change

    def checked(before: Tensor, after: Tensor, *, samples: int, seed: int) -> dict[str, Any]:
        sizes.append(len(before))
        return paired(before, after, samples=samples, seed=seed)

    monkeypatch.setattr(audit, "paired_change", checked)
    audit.summarize(raw, 2)
    assert sizes[:2] == [4, 4]  # permutation paired over systems
    assert set(sizes[2:]) == {2}  # next-step paired over complete batches


def cuda_tag(path: Path) -> None:
    """Give CPU storage a CUDA serialization location without requiring any CUDA hardware."""
    with ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    changed = False
    with ZipFile(path, "w") as archive:
        for info, payload in entries:
            if info.filename.endswith("data.pkl"):
                original = b"X\x03\x00\x00\x00cpu"
                assert original in payload
                payload = payload.replace(original, b"X\x06\x00\x00\x00cuda:0")
                changed = True
            archive.writestr(info, payload)
    assert changed


def refresh_completion(root: Path) -> None:
    completion = cast(dict[str, Any], read_json(root / "completed.json"))
    completion["hashes"] = {name: sha(root / name) for name in completion["hashes"]}
    write_json(root / "completed.json", completion)


def test_completed_fresh_omission_integration_and_tamper_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixture_module, "write_source_fixture", source_with_initialization)

    def fresh_curriculum(config: CurriculumConfig) -> dict[str, Any]:
        return curriculum_run(replace(config, initialization="fresh_zero"))

    monkeypatch.setattr(fixture_module, "curriculum_run", fresh_curriculum)
    curriculum, code_audit = fixture_module.fixture_sources(tmp_path)
    cuda_tag(curriculum / "teacher_final.pt")
    cuda_tag(code_audit / "audit_inputs.pt")
    refresh_completion(curriculum)
    refresh_completion(code_audit)

    def tiny_readout(_table: torch.nn.Embedding, _theta: Tensor) -> dict[str, Any]:
        return {"selected_test_r2": [0.0] * 6}

    monkeypatch.setattr(settling, "readout", tiny_readout)
    endpoint = tmp_path / "settling"
    settling.run(
        settling.Config(
            curriculum_dir=str(curriculum),
            audit_dir=str(code_audit),
            output_dir=str(endpoint),
            steps=2,
            groups=4,
            bootstrap_samples=16,
            verification=True,
            objective_arm="without_parameter_sigreg",
        )
    )
    cuda_tag(endpoint / "evaluation_only.pt")
    manifest = cast(dict[str, Any], read_json(endpoint / "evaluation_manifest.json"))
    manifest["sha256"] = sha(endpoint / "evaluation_only.pt")
    write_json(endpoint / "evaluation_manifest.json", manifest)
    checkpoint = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    checkpoint["evaluation_sha256"] = manifest["sha256"]
    torch.save(checkpoint, endpoint / "last.pt")
    cuda_tag(endpoint / "last.pt")
    refresh_completion(endpoint)
    hashes = {str(p): sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = audit.run(endpoint, tmp_path / "history", verification=True)
    assert result["history"]["updates_per_row"] == 1
    assert (tmp_path / "history/completed.json").exists()
    assert all(sha(Path(name)) == expected for name, expected in hashes.items())
    # A completed joint endpoint cannot be audited using the original frozen source P.
    original_protocol = (endpoint / "protocol.json").read_bytes()
    original_completion = (endpoint / "completed.json").read_bytes()
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    protocol["config"]["train_predictor"] = True
    write_json(endpoint / "protocol.json", protocol)
    refresh_completion(endpoint)
    with pytest.raises(ValueError, match="requires a fixed predictor"):
        audit.run(endpoint, tmp_path / "joint-rejected", verification=True)
    (endpoint / "protocol.json").write_bytes(original_protocol)
    (endpoint / "completed.json").write_bytes(original_completion)
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    payload["table"]["weight"] += 1
    torch.save(payload, endpoint / "last.pt")
    with pytest.raises(ValueError, match="Source artifact changed"):
        audit.run(endpoint, tmp_path / "tampered", verification=True)
