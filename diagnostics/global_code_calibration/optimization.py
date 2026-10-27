"""Bounded deterministic LBFGS with durable last-accepted rollback and point caching."""

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor, nn

Evaluator = Callable[[Tensor], tuple[float, Tensor]]


class EvaluationBudgetExhausted(RuntimeError):
    """A line-search candidate would exceed the predeclared full-objective budget."""


def point_key(value: Tensor) -> str:
    """Exact twelve-float key; caching cannot substitute a nearby unevaluated candidate."""
    return value.detach().double().cpu().contiguous().numpy().tobytes().hex()


def fit(
    evaluator: Evaluator,
    *,
    device: torch.device,
    max_iterations: int = 25,
    max_evaluations: int = 50,
    state: dict[str, Any] | None = None,
    on_state: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit only twelve numbers, preserving accepted states when trial evaluations hit the cap."""
    if min(max_iterations, max_evaluations) < 1:
        raise ValueError("Positive fixed optimization budgets required")
    translation = nn.Parameter(torch.zeros(12, dtype=torch.float64, device=device))
    optimizer = torch.optim.LBFGS(
        [translation],
        lr=1.0,
        max_iter=1,
        max_eval=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    cache: dict[str, dict[str, Any]] = {}
    evaluations, accepted, closure_calls = 0, 0, 0
    if state:
        if (state["max_iterations"], state["max_evaluations"]) != (max_iterations, max_evaluations):
            raise ValueError("Resume optimization budgets changed")
        with torch.no_grad():
            translation.copy_(state["translation"].to(translation))
        optimizer.load_state_dict(state["optimizer"])
        cache = copy.deepcopy(state["cache"])
        evaluations, accepted = int(state["full_evaluations"]), int(state["accepted_iterations"])
        closure_calls = int(state["closure_calls"])
    accepted_translation = translation.detach().clone()
    accepted_optimizer = copy.deepcopy(optimizer.state_dict())
    reason = "running"

    def snapshot() -> dict[str, Any]:
        entry = cache.get(point_key(accepted_translation))
        return {
            "translation": accepted_translation.detach().cpu().clone(),
            "optimizer": copy.deepcopy(accepted_optimizer),
            "cache": copy.deepcopy(cache),
            "full_evaluations": evaluations,
            "accepted_iterations": accepted,
            "closure_calls": closure_calls,
            "max_iterations": max_iterations,
            "max_evaluations": max_evaluations,
            "stop_reason": reason,
            "accepted_loss": entry["loss"] if entry else None,
            "accepted_gradient": entry["gradient"].clone() if entry else None,
            "initial_gradient": cache[point_key(torch.zeros(12, dtype=torch.float64))][
                "gradient"
            ].clone()
            if cache
            else None,
            "initial_loss": cache[point_key(torch.zeros(12, dtype=torch.float64))]["loss"]
            if cache
            else None,
            "converged_gradient": bool(entry and float(entry["gradient"].abs().max()) <= 1e-9),
        }

    def persist() -> None:
        if on_state:
            on_state(snapshot())

    def closure() -> Tensor:
        nonlocal evaluations, closure_calls
        closure_calls += 1
        key = point_key(translation)
        if key not in cache:
            if evaluations >= max_evaluations:
                raise EvaluationBudgetExhausted("Full-objective evaluation cap reached")
            loss, gradient = evaluator(translation)
            if (
                gradient.shape != (12,)
                or not bool(torch.isfinite(gradient).all())
                or not torch.isfinite(torch.tensor(loss))
            ):
                raise FloatingPointError("Invalid translation loss/gradient")
            evaluations += 1
            cache[key] = {
                "point": translation.detach().cpu().clone(),
                "loss": loss,
                "gradient": gradient.detach().cpu().clone(),
            }
            persist()
        entry = cache[key]
        translation.grad = cast(Tensor, entry["gradient"]).to(translation).clone()
        return translation.new_tensor(entry["loss"])

    closure()
    try:
        while accepted < max_iterations:
            current = cache[point_key(accepted_translation)]
            if float(current["gradient"].abs().max()) <= 1e-9:
                reason = "gradient_tolerance"
                break
            try:
                optimizer.step(closure)  # pyright: ignore[reportUnknownMemberType]
                # Strong-Wolfe may return an earlier trial, whose saved gradient must be used.
                key = point_key(translation)
                if key not in cache:
                    closure()
            except EvaluationBudgetExhausted:
                reason = "evaluation_budget_rollback"
                with torch.no_grad():
                    translation.copy_(accepted_translation)
                optimizer.load_state_dict(accepted_optimizer)
                break
            if torch.equal(translation.detach(), accepted_translation):
                reason = "no_accepted_displacement"
                break
            accepted_translation = translation.detach().clone()
            accepted_optimizer = copy.deepcopy(optimizer.state_dict())
            accepted += 1
            persist()
        else:
            reason = "accepted_iteration_budget"
    except BaseException:
        # Every persisted checkpoint references the last accepted state, even during line search.
        persist()
        raise
    result = snapshot()
    if on_state:
        on_state(result)
    return result
