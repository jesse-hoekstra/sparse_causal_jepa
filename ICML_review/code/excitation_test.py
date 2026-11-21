"""Structural checks for the analytic excitation control."""

import numpy as np

from .excitation import encode, features, fit, observations


def test_charts_observe_only_designated_support_channels() -> None:
    """Use paired worlds to check the isolated action change and donor nuisance."""
    off = observations(31, 1, 8, 0)
    on = observations(31, 1, 8, 1)
    np.testing.assert_array_equal(off.inert_controls, on.inert_controls)
    np.testing.assert_array_equal(off.support[..., 0], on.support[..., 0])
    np.testing.assert_array_equal(off.support[..., 1], np.zeros_like(off.support[..., 1]))
    shared = encode(on, "system")
    np.testing.assert_array_equal(shared[:, 0], shared[:, 1])
    np.testing.assert_array_equal(encode(on, "episode")[..., 1], on.inert_controls[..., 1])


def test_frozen_whitener_and_affine_features() -> None:
    """Check normalization and the exact action-response law on a tiny CPU fit."""
    data = observations(17, 0, 16, 1)
    model = fit(data, "response")
    codes = (encode(data, "response") - model.center) @ model.whitening
    flat = codes.reshape(-1, 2)
    np.testing.assert_allclose(flat.T @ flat / len(flat), np.eye(2), atol=1e-12)
    predictions = features(codes, data.query_actions) @ model.weights.reshape(-1)
    np.testing.assert_allclose(predictions[:, 0], data.query_responses, atol=1e-12)
