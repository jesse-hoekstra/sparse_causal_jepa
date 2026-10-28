"""Read-only audit of four previously computed frozen-endpoint test evaluations."""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.harder_oracle.analysis import analyze, routing_metrics, screen_decision
from diagnostics.harder_oracle.run import verify_completion
from diagnostics.oracle_state_learning.analysis import audit_samples, paired_mcc_interval, read_json
from diagnostics.oracle_state_learning.data import sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics
from scjepa.experiments.storage import write_json


def _load_test(directory: Path, parent: Path, *, allow_smoke: bool) -> dict[str, Any]:
    completion = verify_completion(directory, kind="test", require_current_source=False)
    evidence = read_json(directory / "config.json")
    if evidence != {
        key: value for key, value in completion.items() if key not in {"hashes", "wandb"}
    }:
        raise ValueError("Test configuration and completion disagree")
    if evidence["identity"] != canonical_fingerprint(
        {key: value for key, value in evidence.items() if key != "identity"}
    ):
        raise ValueError("Test provenance fingerprint differs")
    parent_completion = read_json(parent / "completion.json")
    parent_config = read_json(parent / "config.json")
    for key in ("data_identity", "code_identity", "condition", "seed", "steps"):
        if completion[key] != parent_completion[key]:
            raise ValueError(f"Test {key} differs from its authenticated parent")
    if completion["parent_identity"] != parent_completion["identity"]:
        raise ValueError("Test identifies a different parent run")
    if completion["parent_completion_sha256"] != sha(parent / "completion.json") or completion[
        "parent_checkpoint_sha256"
    ] != sha(parent / "last.pt"):
        raise ValueError("Test parent completion or checkpoint hash differs")
    matching = read_json(directory / "matching.json")
    if matching != read_json(parent / "matching.json"):
        raise ValueError("Test changed the frozen calibration assignment")
    groups = int(parent_config["candidate_settings"]["test_groups"])
    if not allow_smoke and (
        groups != 512 or completion["steps"] != 100_000 or completion["seed"] != 0
    ):
        raise ValueError("Confirmation requires 512 test groups and fixed seed-zero 100k parents")
    payload = cast(
        dict[str, Any], torch.load(directory / "test.pt", map_location="cpu", weights_only=True)
    )
    if payload["metrics"] != read_json(directory / "test.json")["metrics"]:
        raise ValueError("Readable test metrics differ from the saved tensor artifact")
    samples = audit_samples(payload, groups)
    test_ids = {int(value) for value in samples["group_ids"]}
    calibration = cast(
        dict[str, Any], torch.load(parent / "calibration.pt", map_location="cpu", weights_only=True)
    )["samples"]
    validation = cast(
        dict[str, Any], torch.load(parent / "validation.pt", map_location="cpu", weights_only=True)
    )["samples"]
    for source in (calibration, validation):
        if test_ids & {int(value) for value in source["group_ids"]}:
            raise ValueError("Test parameter groups overlap calibration or validation")
    for encoder, key in (("online", "state_mcc"), ("target", "target_state_mcc")):
        features, truth = (
            samples[f"{encoder}_states"].flatten(0, 2),
            samples["states"].flatten(0, 2),
        )
        actual = mean_correlation_coefficient(
            CoordinateMatch(tuple(matching[f"{encoder}_columns"])), features, truth
        )
        if not math.isclose(actual, payload["metrics"][key], rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Reported test MCC differs from raw samples and frozen matching")
        # Fit exclusively to the authenticated parent calibration rows. Test
        # labels are used only for scoring this unchanged readout, never fitting.
        probe = fit_state_probe(
            calibration[f"{encoder}_states"].flatten(0, 2),
            calibration["states"].flatten(0, 2),
            ridge=1e-3,
        )
        recovered = state_probe_metrics(probe.predict(features), truth)
        if not math.isclose(
            recovered["r2"],
            payload["metrics"][f"{encoder}_state_probe/r2"],
            rel_tol=1e-9,
            abs_tol=1e-10,
        ):
            raise ValueError("Reported test state readout differs from calibration-only probe")
    variance = samples["targets"].double().flatten(0, -2).var(0, correction=0).mean()
    normalized = float(
        (samples["predictions"].double() - samples["targets"].double()).square().mean() / variance
    )
    if not math.isclose(
        normalized, payload["metrics"]["normalized_prediction_mse"], abs_tol=1e-8, rel_tol=2e-6
    ):
        raise ValueError("Reported normalized test error differs from raw samples")
    payload["metrics"].update(routing_metrics(samples, matching))
    return {
        "directory": str(directory),
        "parent": str(parent),
        "completion": completion,
        "matching": matching,
        "samples": samples,
        "metrics": payload["metrics"],
    }


def confirm(
    directories: Sequence[Path],
    parents: Sequence[Path],
    output: Path,
    *,
    primary_chart: str = "mixed",
    draws: int = 1000,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Authenticate saved test outcomes and their parent links without any inference."""
    if len(directories) != 4 or len(parents) != 4:
        raise ValueError("Require exactly four test directories and four training parents")
    if output.exists():
        raise FileExistsError("Confirmation analysis requires a fresh output directory")
    # This reuses the full parent artifact, counter, initialization, source,
    # calibration and validation audit. It performs no model inference.
    validation = analyze(
        parents,
        output / "validation_audit",
        primary_chart=primary_chart,
        draws=draws,
        allow_smoke=allow_smoke,
    )
    parent_by_identity = {read_json(path / "completion.json")["identity"]: path for path in parents}
    tests: list[dict[str, Any]] = []
    for directory in directories:
        provenance = read_json(directory / "completion.json")
        parent = parent_by_identity.get(provenance.get("parent_identity"))
        if parent is None:
            raise ValueError("Test parent was not supplied and authenticated")
        tests.append(_load_test(directory, parent, allow_smoke=allow_smoke))
    indexed = {item["completion"]["condition"]: item for item in tests}
    if set(indexed) != {"dense", "mlp", "baseline", "full"}:
        raise ValueError("Require exactly one test outcome for each primary condition")
    for key in ("data_identity", "code_identity"):
        if len({item["completion"][key] for item in tests}) != 1:
            raise ValueError(f"Test arms differ in {key}")
    comparisons: list[dict[str, Any]] = []
    for left, right in (
        ("full", "baseline"),
        ("baseline", "dense"),
        ("full", "dense"),
        ("baseline", "mlp"),
    ):
        a, b = indexed[left], indexed[right]
        av, bv = a["samples"], b["samples"]
        if not torch.equal(av["group_ids"], bv["group_ids"]) or not torch.equal(
            av["states"], bv["states"]
        ):
            raise ValueError(
                "Test comparisons require identical addressed parameter groups and states"
            )
        comparisons.append(
            {
                "left": left,
                "right": right,
                "split": "test",
                **paired_mcc_interval(
                    av["online_states"],
                    bv["online_states"],
                    av["states"],
                    a["matching"]["online_columns"],
                    b["matching"]["online_columns"],
                    draws=draws,
                ),
            }
        )
    decision = screen_decision({name: item["metrics"] for name, item in indexed.items()})
    passed = decision.pop("screen_passed")
    decision["confirmation_passed"] = passed
    decision["decision"] = (
        "one_seed_test_effect_targets_met" if passed else "test_effect_targets_not_met"
    )
    source = next(iter(indexed.values()))["completion"]["code_identity"]
    result: dict[str, Any] = {
        "schema": "harder-oracle-confirmation-analysis-v1",
        "profile": "smoke" if allow_smoke else "exploratory",
        "primary_chart": validation["primary_chart"],
        **decision,
        "source_identity": source,
        "validation_screen_passed": validation["screen_passed"],
        "geco_tau": validation["geco_tau"],
        "all_sparse_test_constraints_feasible": all(
            indexed[name]["metrics"]["base"] <= validation["geco_tau"]
            for name in ("baseline", "full")
        ),
        "comparisons": comparisons,
        "arms": [
            {
                "condition": item["completion"]["condition"],
                "directory": item["directory"],
                "parent_directory": item["parent"],
                "parent_identity": item["completion"]["parent_identity"],
                "parent_completion_sha256": item["completion"]["parent_completion_sha256"],
                "parent_checkpoint_sha256": item["completion"]["parent_checkpoint_sha256"],
                "test_completion_sha256": sha(Path(item["directory"]) / "completion.json"),
                "test_file_sha256": item["completion"]["hashes"],
                "metrics": item["metrics"],
                "wandb": item["completion"]["wandb"],
            }
            for item in tests
        ],
        "analysis_source_sha256": {
            str(path): sha(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("analysis.py"),
                Path(__file__).parents[1] / "oracle_state_learning/analysis.py",
            )
        },
        "limitations": [
            "The candidate was developed and selected using validation outcomes; "
            "test uses its frozen final checkpoints.",
            "One training seed is exploratory. Paired group intervals condition on trained "
            "models and do not estimate training-seed variability.",
            "No model training, inference, test-fitted assignment or test-fitted probe "
            "occurs in this audit.",
            "Oracle parameters are supplied; success does not establish learned-parameter "
            "acquisition.",
            "Observed routing unions do not certify globally minimal functional causal "
            "graphs or theorem premises.",
        ],
    }
    write_json(output / "summary.json", result)
    write_json(
        output / "completion.json",
        {
            "schema": result["schema"],
            "source_identity": source,
            "primary_chart": validation["primary_chart"],
            "hashes": {
                "summary.json": sha(output / "summary.json"),
                "validation_audit/summary.json": sha(output / "validation_audit/summary.json"),
            },
        },
    )
    return result


def main() -> None:
    """Audit four saved test evaluations; this command cannot run an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, type=Path, required=True)
    parser.add_argument("--parents", nargs=4, type=Path, required=True)
    parser.add_argument("--primary-chart", choices=("mixed", "physical"), default="mixed")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = confirm(args.runs, args.parents, args.output_dir, primary_chart=args.primary_chart)
    print(
        json.dumps({"decision": result["decision"], "comparisons": result["comparisons"]}, indent=2)
    )


if __name__ == "__main__":
    main()
