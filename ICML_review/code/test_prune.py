"""Calibration gate-selection policy and grouped rollout replay checks."""

from typing import cast

import torch
from torch import Tensor

from diagnostics.vision_dimer.model import LaggedMaskedPredictor, build_model
from scjepa.models.model import ModelOutput

from .prune import cache, greedy, group_errors


def test_deletion_can_rescue_infeasible_model() -> None:
    """Check recovery, budget stopping and original-logit restoration after rejected trials."""
    edges = torch.tensor([[3.0, 2.0]])
    losses = {(True, True): 4.0, (False, True): 1.0, (True, False): 2.0, (False, False): 5.0}

    def score() -> float:
        return losses[bool(edges[0, 0] > 0), bool(edges[0, 1] > 0)]

    path = greedy({"state": edges}, score, 1.5)
    assert [step["accepted"] for step in path] == [True, False]
    assert edges.tolist() == [[-1.0, 2.0]]  # pyright: ignore[reportUnknownMemberType]


def test_cpu_frozen_rollout_smoke() -> None:
    """A fresh random vision model must reproduce grouped forecasts before and after deletion."""
    torch.set_num_threads(1)
    model = build_model(13, phase="full", code_dim=4, context=10).eval()
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 3, 12, 3, 64, 64, generator=generator)
    a = torch.randn(1, 3, 11, 5, generator=generator)
    batch = {"observations": x, "actions": a}
    with torch.no_grad():
        cached = cache(model, [batch])
        logits = cast(LaggedMaskedPredictor, model.predictor).learned_logits()
        for logit in (3.0, -1.0):
            logits["state"][0, 0] = logit
            output = cast(ModelOutput, model(x, a))
            expected: Tensor = (
                (output.predictions.double() - output.targets.double()).square().mean()
            )
            torch.testing.assert_close(group_errors(model, cached).mean(), expected)
