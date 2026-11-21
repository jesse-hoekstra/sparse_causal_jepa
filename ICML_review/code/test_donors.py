"""Panel-routing checks on a predictor that returns the supplied code."""

from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.models.model import SCJEPA

from .donors import risks


class _Predictor:
    history_len = 1


class _Echo:
    predictor = _Predictor()
    context_len = 1

    def rollout(self, h: Tensor, c: Tensor, steps: int, a: Tensor) -> Tensor:
        del h, a
        return c[:, None].expand(-1, steps, -1)


def test_disjoint_support_and_dispersion() -> None:
    """Check finite-panel variance and the other-system negative control."""
    codes = torch.tensor([[[0.0], [2.0], [4.0], [6.0]], [[10.0], [12.0], [14.0], [16.0]]])
    targets = codes[:, :, None, :]
    m = risks(cast(SCJEPA, _Echo()), targets, codes, targets, targets)
    assert m[:, 0].tolist() == pytest.approx([18.0, 18.0])  # pyright: ignore[reportUnknownMemberType]
    assert m[:, 1].tolist() == pytest.approx([17.0, 17.0])  # pyright: ignore[reportUnknownMemberType]
    assert m[:, 2].tolist() == pytest.approx([117.0, 117.0])  # pyright: ignore[reportUnknownMemberType]
    assert torch.equal(m[:, 3], torch.ones(2))
