"""Graph comparison helpers; experiments provide aligned reference graphs.

Reachability indicates available information paths, not verified causal use.
There is no simulator-specific ground-truth construction or alignment here.
"""

from torch import Tensor


def structural_hamming_distance(learned: Tensor, target: Tensor) -> Tensor:
    """Mean Hamming count over the last two (output, input) adjacency axes."""
    if learned.shape != target.shape or learned.ndim < 2:
        raise ValueError("graphs must have matching shapes (..., outputs, inputs)")
    return (learned.bool() != target.bool()).sum(dim=(-2, -1)).float().mean()


__all__ = ["structural_hamming_distance"]
