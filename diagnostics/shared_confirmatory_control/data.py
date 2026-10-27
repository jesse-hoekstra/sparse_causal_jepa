"""Create a single fresh test sample and verify all previously used group identities."""

import hashlib
from collections.abc import Callable
from typing import Any

import torch

from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.overcomplete_use_precision.data import build_validation, verify_independence
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.random import generator
from scjepa.models.model import SCJEPA


def build_fresh(
    model: SCJEPA,
    metadata: dict[str, Any],
    *,
    groups: int,
    seed: int,
    batch_size: int,
    progress: Callable[[int], None] | None = None,
) -> EvaluationData:
    """Reuse the original exact-law sampler, explicitly labelling its output as test data."""
    result = build_validation(
        model, metadata, groups=groups, seed=seed, batch_size=batch_size, progress=progress
    )
    result["test"] = result.pop("validation")
    result["metadata"].update(
        role="confirmatory_test",
        seed_policy="Direct predeclared fresh generator seed; confirmatory test only",
    )
    return result


def independence(
    fresh: EvaluationData,
    old: EvaluationData,
    precision: EvaluationData,
    train_config: dict[str, Any],
) -> dict[str, Any]:
    """Check exact theta identity only; no old TRAIN or test predictions are computed."""
    if set(fresh) != {"metadata", "test"} or fresh["metadata"].get("role") != "confirmatory_test":
        raise ValueError("Require explicitly labelled confirmatory test groups")
    adapted: EvaluationData = {"metadata": fresh["metadata"], "validation": fresh["test"]}
    receipt = verify_independence(adapted, old)
    seed = int(fresh["metadata"]["seed"])
    if seed == precision["metadata"]["seed"]:
        raise ValueError("Fresh sample repeats precision-validation seed")
    rows = {row.numpy().tobytes() for row in fresh["test"]["theta"].double().cpu()}
    if any(
        row.numpy().tobytes() in rows for row in precision["validation"]["theta"].double().cpu()
    ):
        raise ValueError("Fresh parameter group overlaps precision validation")
    if train_config["family"] != "simplified_gaussian" or train_config["split"] != "train":
        raise ValueError("Require exact original TRAIN generator law")
    digest = hashlib.sha256()
    for index in range(train_config["num_episodes"]):
        stream = generator(
            train_config["root_seed"], "train", index, "simplified_gaussian/parameters"
        )
        if stream.initial_seed() == seed:
            raise ValueError("Fresh direct seed repeats an original TRAIN parameter stream")
        theta = dgp.PARAMETER_BOUND * (2 * torch.rand(6, dtype=torch.float64, generator=stream) - 1)
        row = theta.numpy().tobytes()
        if row in rows:
            raise ValueError("Fresh parameter group overlaps TRAIN")
        digest.update(row)
    return {
        **receipt,
        "fresh_role": "confirmatory_test",
        "precision_seed": precision["metadata"]["seed"],
        "exact_precision_overlaps": 0,
        "exact_train_overlaps": 0,
        "training_parameter_groups": train_config["num_episodes"],
        "train_identity_checked": not train_config.get(
            "verification_fixture_missing_train_identity", False
        ),
        "training_theta_identity_sha256": digest.hexdigest(),
        "scope": (
            "Old TRAIN/test labels reconstructed or accessed only for exact group "
            "identity; probes use old calibration/validation only; fresh sample never "
            "selects or fits."
        ),
    }
