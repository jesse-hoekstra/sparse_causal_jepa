"""One new validation draw from the unchanged scientific generator."""

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA


@torch.no_grad()
def build_validation(
    model: SCJEPA,
    metadata: dict[str, Any],
    *,
    groups: int,
    seed: int,
    batch_size: int,
    progress: Callable[[int], None] | None = None,
) -> EvaluationData:
    """Use the exact existing evaluation draw order, with one predeclared independent seed."""
    if groups < 2 or batch_size < 1 or seed < 0 or model.training:
        raise ValueError("Require positive dimensions and frozen eval-mode model")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    theta = dgp.PARAMETER_BOUND * (
        2 * torch.rand(groups, 6, dtype=torch.float64, generator=generator) - 1
    )
    initial = torch.randn(groups, 10, 6, dtype=torch.float64, generator=generator)
    gaussian = torch.randn(groups, 10, 11, 10, dtype=torch.float64, generator=generator)
    coupling = 2 * torch.rand(groups, 10, 11, 6, dtype=torch.float64, generator=generator) - 1
    actions = dgp.encode_controls(torch.cat((gaussian, coupling), -1))
    states = dgp.simulate(initial, theta[:, None], actions)
    cache_model = copy.deepcopy(model).float().requires_grad_(False).eval()
    observation = VectorObservationMap(matrix=metadata["observation_matrix"], dimension=6)
    online: list[Tensor] = []
    target: list[Tensor] = []
    for start in range(0, groups, batch_size):
        a, b = encode_states(
            cache_model, observation.forward(states[start : start + batch_size]), batch_size * 10
        )
        online.append(a)
        target.append(b)
        if progress:
            progress(min(start + batch_size, groups))
    return {
        "metadata": {
            **metadata,
            "seed": seed,
            "source_state_online_sha256": metadata.get("state_online_sha256"),
            "source_state_target_sha256": metadata.get("state_target_sha256"),
            "state_online_sha256": module_fingerprint(cache_model.online_encoder),
            "state_target_sha256": module_fingerprint(cache_model.target_encoder),
            "seed_policy": "Direct fresh generator seed; validation only; no split offset",
            "generator_version": dgp.GENERATOR_VERSION,
            "cache_encoding_dtype": "float32",
            "simulation_dtype": "float64",
        },
        "validation": {
            "theta": theta,
            "true_states": states,
            "true_actions": actions,
            "actions": actions.float(),
            "states_online": torch.cat(online),
            "states_target": torch.cat(target),
        },
    }


def verify_independence(new: EvaluationData, old: EvaluationData) -> dict[str, Any]:
    """Check split-seed separation and exact theta-group disjointness; no probe fitting."""
    new_seed, old_seed = int(new["metadata"]["seed"]), int(old["metadata"]["seed"])
    old_seeds = [old_seed + 1009 * index for index in range(3)]
    if new_seed in old_seeds:
        raise ValueError("Fresh validation seed overlaps archived evaluation draws")
    rows = {
        row.numpy().tobytes()
        for row in cast(TensorSplit, new["validation"])["theta"].double().cpu()
    }
    if len(rows) != len(new["validation"]["theta"]):
        raise ValueError("Duplicate new parameter groups")
    for name in ("calibration", "validation", "test"):
        if any(
            row.numpy().tobytes() in rows
            for row in cast(TensorSplit, old[name])["theta"].double().cpu()
        ):
            raise ValueError(f"Fresh parameter group overlaps archived {name}")
    return {
        "new_seed": new_seed,
        "archived_split_seeds": old_seeds,
        "exact_theta_overlaps": {name: 0 for name in ("calibration", "validation", "test")},
        "scope": (
            "Distinct seeded draws and no repeated theta groups; archived test "
            "labels checked only for identity, never selection or fitting."
        ),
    }
