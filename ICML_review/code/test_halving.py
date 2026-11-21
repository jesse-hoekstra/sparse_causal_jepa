"""Check strict replay boundaries and rejection of incomplete acquisition histories."""

import json

import pytest

from ICML_review.code.halving import window_scores


def test_windows() -> None:
    """Exclude the lower endpoint and unrelated evaluation records."""
    train = [
        {"step": s, "prediction": float(s)}
        for s in (*range(45_000, 50_001, 500), *range(95_000, 100_001, 500))
    ]
    train.append({"step": 50_000, "evaluation": 0.0})
    history = "\n".join(json.dumps(row) for row in train).encode()
    assert window_scores(history, "prediction", 500, compensated_sum=False) == (47_750, 97_750)
    with pytest.raises(ValueError, match="grid"):
        window_scores(
            history.replace(b'"step": 45500', b'"step": 45499'),
            "prediction",
            500,
            compensated_sum=False,
        )
