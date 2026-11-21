"""Sufficient-statistic and policy feasibility checks on tiny generated fixtures."""

import math

import numpy as np

from .noisy_support import Array, Candidate, Moments, Normalizer, choose, loss_fit, observe


def test_factorized_fit_matches_explicit_crossed_donors() -> None:
    """Check the actual CPU fit, its objective and the uncorrected within-group variance."""
    obs = observe(714, 3, 24, 1.0)
    n = Normalizer.from_train(obs)
    stats = Moments.from_data(obs, n)
    w, d = loss_fit(stats, 30)
    z = (obs.second - n.center[0]) / n.scales[0]
    g = (obs.inert - n.center[1]) / n.scales[1]
    code = np.cos(np.deg2rad(30)) * z + np.sin(np.deg2rad(30)) * g[:, None]
    x = np.stack(
        (np.ones_like(code), np.broadcast_to(obs.first[:, None], code.shape), code), axis=-1
    )
    design: Array = np.einsum("gqa,grd->grqad", obs.actions, x).reshape(*code.shape, 4, -1)
    y: Array = np.broadcast_to(obs.outcomes[:, None, :], design.shape[:-1])
    b = np.linalg.lstsq(design.reshape(-1, 6), y.reshape(-1), rcond=None)[0]
    explicit = ((design @ b - y) ** 2).mean(axis=(1, 2))
    np.testing.assert_allclose(explicit, stats.errors(w), atol=1e-12)
    assert math.isclose(d, 0.5 * float(code.var(axis=1).mean() / code.var()))


def test_feasible_selector_uses_reference_risk() -> None:
    """An otherwise attractive nuisance can be ineligible for the constrained rule."""
    w = np.zeros((2, 4))
    reference = Candidate(0, w, 0.6, 0.45, 0.5)
    mixed = Candidate(40, w, 0.3, 0.5, 0.49)
    nuisance = Candidate(85, w, 0.05, 0.6, 0.8)
    selected = choose([reference, mixed, nuisance], 1.0)
    assert selected["forecast"] is reference
    assert selected["additive"] is nuisance
    assert selected["constrained"] is mixed
