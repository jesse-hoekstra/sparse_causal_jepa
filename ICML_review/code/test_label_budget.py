"""Check row selection for reduced-label calibration."""

import pytest
import torch

from .label_budget import panel_indices


def test_disjoint_reproducible_panels() -> None:
    """Each system supplies exactly one calibration row across all panels."""
    indices = panel_indices(32, 8)
    assert indices.shape == (4, 8)
    assert torch.equal(indices.flatten().sort().values, torch.arange(32))
    assert torch.equal(indices, panel_indices(32, 8))
    with pytest.raises(ValueError, match="divide"):
        panel_indices(31, 8)
