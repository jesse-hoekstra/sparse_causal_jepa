"""Posthoc adapters preserve isolated source identity and the 24-action graph shape."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from diagnostics.harder_oracle.run import verify_completion as verify_parent_completion
from diagnostics.isotropic_oracle import analysis, confirmation, run
from diagnostics.isotropic_oracle.data import PrepareConfig, code_identity, prepare
from diagnostics.oracle_state_learning.data import sha


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_isotropic_routing_counts144_actions_and_rejects_older_action_shape() -> None:
    rows, columns = [2, 4, 0, 5, 1, 3], [5, 2, 1, 0, 3, 4]
    state = torch.zeros(6, 6)
    state[:3, :3] = state[3:, 3:] = 1
    parameter = torch.eye(6)
    parameter[torch.arange(6), [3, 4, 5, 0, 1, 2]] = 1
    learned_state, learned_parameter = torch.empty_like(state), torch.empty_like(parameter)
    learned_state[torch.tensor(rows)[:, None], torch.tensor(columns)[None, :]] = state
    learned_parameter[rows] = parameter
    samples = {
        "graph_state": learned_state[None].repeat(2, 1, 1),
        "graph_parameter": learned_parameter[None].repeat(2, 1, 1),
        "graph_action": torch.ones(2, 6, 24),
    }
    matching = {"online_columns": columns, "target_columns": rows}
    result = analysis.routing_metrics(samples, matching)
    for name, edges in (("state", 18), ("action", 144), ("parameter", 12)):
        assert result[f"routing/{name}_observed_union_edges"] == edges
        assert result[f"routing/{name}_aligned_union_shd"] == 0
    samples["graph_action"] = torch.ones(2, 6, 22)
    with pytest.raises(ValueError, match="Malformed local routing"):
        analysis.routing_metrics(samples, matching)


def test_posthoc_changes_cannot_change_frozen_training_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = code_identity()
    original_read = Path.read_bytes
    root = Path(analysis.__file__).parent
    posthoc_paths = {root / name for name in ("analysis.py", "confirmation.py", "parameter_use.py")}

    def changed_analysis(path: Path) -> bytes:
        value = original_read(path)
        return (
            value + b"\n# hypothetical read-only reporting change\n"
            if path in posthoc_paths
            else value
        )

    monkeypatch.setattr(Path, "read_bytes", changed_analysis)
    assert code_identity() == before


def test_tiny_isotropic_endpoints_use_their_own_schema_and_posthoc_readers(tmp_path: Path) -> None:
    before = code_identity()
    prepare(
        PrepareConfig(
            str(tmp_path / "data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    conditions = ("dense", "mlp", "baseline", "full")
    for condition in conditions:
        if condition == "baseline":
            run.calibrate(
                run.CalibrateConfig(
                    str(tmp_path / "bound"), (str(tmp_path / "dense"),), profile="smoke"
                )
            )
        run.train(
            run.TrainConfig(
                str(tmp_path / condition),
                str(tmp_path / "data"),
                condition,
                profile="smoke",
                steps=2,
                batch_size=2,
                log_every=1,
                evaluate_every=2,
                checkpoint_every=1,
                calibration_dir=str(tmp_path / "bound")
                if condition in {"baseline", "full"}
                else None,
            )
        )
    parents = [tmp_path / condition for condition in conditions]
    with pytest.raises(ValueError, match="schema differs"):
        verify_parent_completion(parents[0])
    with pytest.raises(ValueError, match="Scientific analysis"):
        analysis.analyze(parents, tmp_path / "not-scientific")
    result = analysis.analyze(parents, tmp_path / "validation-audit", draws=10, allow_smoke=True)
    assert result["primary_chart"] == "isotropic"
    assert len(result["comparisons"]) == 4
    assert all(row["rotation_scale"] == 0.23865401979977377 for row in result["arms"])
    assert not any((path / "test.pt").exists() for path in parents)
    source_hashes = {
        str(path): sha(path) for parent in parents for path in parent.iterdir() if path.is_file()
    }
    tests = [tmp_path / f"{condition}-test" for condition in conditions]
    for parent, target in zip(parents, tests, strict=True):
        run.score_test(run.TestConfig(str(target), str(parent), str(tmp_path / "data")))
    confirmed = confirmation.confirm(
        tests, parents, tmp_path / "test-audit", draws=10, allow_smoke=True
    )
    assert confirmed["primary_chart"] == "isotropic"
    assert confirmed["source_identity"] == before
    assert all(pair["split"] == "test" for pair in confirmed["comparisons"])
    assert not confirmed["confirmed_requested_ordering"]
    assert all(sha(Path(path)) == digest for path, digest in source_hashes.items())
    assert code_identity() == before
