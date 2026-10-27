"""Embed six active offsets into the unchanged twelve-coordinate bounded optimizer."""

from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.global_code_calibration.optimization import Evaluator
from diagnostics.global_code_calibration.optimization import fit as fit


class EmbeddedObjective:
    """Use six code offsets; inactive optimizer coordinates stay identically zero."""

    def __init__(self, evaluator: Evaluator) -> None:
        """Wrap a six-dimensional objective without altering its values or derivatives."""
        self.evaluator = evaluator

    def __call__(self, embedded: Tensor) -> tuple[float, Tensor]:
        """Pad the exact six-offset gradient with zeros for the unchanged bounded LBFGS."""
        if embedded.shape != (12,) or not bool((embedded[6:] == 0).all()):
            raise ValueError("Require six active coordinates and six fixed inactive zeros")
        value, gradient = self.evaluator(embedded[:6])
        if gradient.shape != (6,):
            raise ValueError("Six-offset objective must return six gradient entries")
        return value, torch.cat((gradient, gradient.new_zeros(6)))


def verify_embedding(state: dict[str, Any]) -> None:
    """Check that trial/accepted points and every optimizer direction remain in the six-subspace."""
    for key in ("translation", "accepted_gradient", "initial_gradient"):
        value = state[key]
        if value is not None and (value.shape != (12,) or not bool((value[6:] == 0).all())):
            raise ValueError("Inactive optimizer coordinates or gradients moved")
    for row in state["cache"].values():
        for key in ("point", "gradient"):
            value = row[key]
            if value.shape != (12,) or not bool((value[6:] == 0).all()):
                raise ValueError("Cached inactive optimizer point/gradient moved")
    for row in state["optimizer"]["state"].values():
        for key in ("d", "prev_flat_grad"):
            value = row.get(key)
            if value is not None and not bool((value[6:] == 0).all()):
                raise ValueError("Inactive optimizer direction or history moved")
        for key in ("old_dirs", "old_stps"):
            for value in row.get(key, []):
                if not bool((value[6:] == 0).all()):
                    raise ValueError("Inactive optimizer curvature history moved")


def active_state(state: dict[str, Any]) -> dict[str, Any]:
    """Expose only the six actual code offsets for summaries and model composition."""
    verify_embedding(state)
    return {
        **state,
        **{
            key: cast(Tensor, state[key])[:6]
            for key in ("translation", "accepted_gradient", "initial_gradient")
        },
    }
