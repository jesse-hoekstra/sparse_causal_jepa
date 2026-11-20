"""Float64 numerical checks of the paper's Assumptions 1, 11, 12 and 13 for the dimer DGP.

Assumption 1 (observation map): rank of the render Jacobian, including exact bead overlaps.
Assumptions 11, 12 and 13 (sufficient influence of theta, of the actions and of the state):
the spans of the witness Jacobians. The output ``z`` is held fixed exactly as the theory
requires: witness histories are solved through the exact inverse ``previous_state``
(lag-0, parameter and action witnesses) or a Newton solve for the current state (lag-1
witnesses). Ranks are reported against the required dimensions with the normalized smallest
singular value sigma_req / sigma_1. The temporal check uses designed witness families: fields
up to quarter turns for every bead, drag coefficients at the corners of the prior box, both lags.

usage: python -m diagnostics.vision_dimer.verify_assumptions [--out results.json] [--states N]
"""

import argparse
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.autograd.functional import jacobian  # pyright: ignore[reportUnknownVariableType]

from diagnostics.vision_dimer import dgp

torch.set_default_dtype(torch.float64)


def _jacobian(function: Callable[[Tensor], Tensor], point: Tensor) -> Tensor:
    return cast(Tensor, jacobian(function, point))


def _singular_values(matrix: Tensor) -> Tensor:
    return cast(Tensor, torch.linalg.svdvals(matrix))  # pyright: ignore[reportUnknownMemberType]


def _step(physics: dgp.Physics) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    def step(previous: Tensor, current: Tensor, action: Tensor, theta: Tensor) -> Tensor:
        return dgp.flatten_state(
            dgp.transition(
                dgp.unflatten_state(previous), dgp.unflatten_state(current), action, theta, physics
            )
        )

    return step


def _history(
    z: Tensor, current: Tensor, action: Tensor, theta: Tensor, physics: dgp.Physics
) -> Tensor:
    """Previous state that makes the one-step output exactly ``z``."""
    return dgp.flatten_state(
        dgp.previous_state(
            dgp.unflatten_state(z), dgp.unflatten_state(current), action, theta, physics
        )
    )


def _current_for(z: Tensor, action: Tensor, theta: Tensor, physics: dgp.Physics) -> Tensor:
    """Current state with ``g(z, current) = z`` (previous state fixed at ``z``), by Newton."""
    step = _step(physics)
    current = z.clone()
    for _ in range(100):
        residual = step(z, current, action, theta) - z
        if float(residual.abs().max()) < 1e-13:
            break
        derivative = _jacobian(lambda c: step(z, c, action, theta), current)
        current = current - cast(Tensor, torch.linalg.solve(derivative, residual))  # pyright: ignore[reportUnknownMemberType]
    return current


def _report(rows: list[Tensor], required: int) -> dict[str, Any]:
    values = _singular_values(torch.stack(rows))
    rank = int((values > values[0] * 1e-10).sum())
    ratio = float(values[required - 1] / values[0]) if rank >= required else 0.0
    return {"witnesses": len(rows), "rank": rank, "required": required, "holds": rank == required,
            "sigma_ratio": ratio}  # fmt: skip


def check(physics: dgp.Physics, z: Tensor, generator: torch.Generator) -> dict[str, Any]:
    """Parameter, action and temporal witness spans (Assumptions 11-13) at one output ``z``."""
    graphs = dgp.graphs()
    step = _step(physics)
    theta0 = dgp.sample_theta(1, physics, generator)[0]
    report: dict[str, Any] = {}

    # Assumption 11 (parameter influence): same z and theta; histories vary
    # (one row offset at a time, two fields).
    rows: list[Tensor] = []
    offsets = [torch.zeros(dgp.STATE_DIM)] + [
        torch.eye(dgp.STATE_DIM)[k] * s for k in range(dgp.STATE_DIM) for s in (0.5, -0.7)
    ]
    for offset in offsets:
        for field in (0.0, 0.9):
            action = torch.zeros(dgp.ACTION_DIM)
            action[0] = field
            current = z + offset
            previous = _history(z, current, action, theta0, physics)

            def by_theta(th: Tensor, p: Tensor = previous, c: Tensor = current,
                         a: Tensor = action) -> Tensor:  # fmt: skip
                return step(p, c, a, th)

            rows.append(_jacobian(by_theta, theta0)[graphs["parameter"]])
    report["parameter_influence"] = _report(rows, int(graphs["parameter"].sum()))

    # Assumption 12 (action influence): fixed z; vary actions, theta and histories;
    # one column per action coordinate.
    witnesses: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for _ in range(40):
        action = torch.randn(dgp.ACTION_DIM, generator=generator)
        theta = dgp.sample_theta(1, physics, generator)[0]
        current = z + 0.6 * torch.randn(dgp.STATE_DIM, generator=generator)
        witnesses.append((_history(z, current, action, theta, physics), current, action, theta))
    by_coordinate: dict[str, Any] = {}
    for j in range(dgp.ACTION_DIM):
        children = graphs["action"][:, j]
        columns: list[Tensor] = []
        for p0, c0, a0, th0 in witnesses:

            def by_action(a: Tensor, p: Tensor = p0, c: Tensor = c0, th: Tensor = th0) -> Tensor:
                return step(p, c, a, th)

            columns.append(_jacobian(by_action, a0)[children, j])
        by_coordinate[f"action_{j}"] = _report(columns, int(children.sum()))
    report["action_influence"] = by_coordinate

    # Assumption 13 (temporal influence): designed families. Fields giving
    # tau = sigma*beta*B up to quarter turns for every
    # bead; drag at the 16 corners of the prior box; lag-0 (current = z) and lag-1 (previous = z).
    low, high = physics.drag_range
    corners = [torch.tensor(c) for c in itertools.product((low, high), repeat=dgp.PARAM_DIM)]
    fields = [0.0] + [s * v / physics.field_scale for v in (0.5, 1.0, 2.0, 4.0) for s in (1, -1)]
    lag0: list[Tensor] = []
    lag1: list[Tensor] = []
    for theta in corners:
        for field in fields:
            action = torch.zeros(dgp.ACTION_DIM)
            action[0] = field
            previous = _history(z, z, action, theta, physics)
            current = _current_for(z, action, theta, physics)

            def by_current(c: Tensor, p: Tensor = previous, a: Tensor = action,
                           th: Tensor = theta) -> Tensor:  # fmt: skip
                return step(p, c, a, th)

            def by_previous(p: Tensor, c: Tensor = current, a: Tensor = action,
                            th: Tensor = theta) -> Tensor:  # fmt: skip
                return step(p, c, a, th)

            lag0.append(_jacobian(by_current, z)[graphs["state"]])
            lag1.append(_jacobian(by_previous, z)[graphs["state"]])
    required = int(graphs["state"].sum())
    report["temporal_influence"] = {
        "lag0": _report(lag0, required),
        "lag1": _report(lag1, required),
        "both": _report(lag0 + lag1, required),
    }
    return report


def main() -> None:
    """Run the checks at several stationary output states and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--states", type=int, default=4)
    args = parser.parse_args()
    physics = dgp.Physics()
    generator = torch.Generator().manual_seed(20261115)
    theta = dgp.sample_theta(64, physics, generator)
    states, actions = dgp.simulate(theta, 1, 40, physics, generator)
    pool = states.reshape(-1, dgp.STATE_DIM)
    picks = torch.randint(0, len(pool), (args.states,), generator=generator)
    outputs = [pool[int(i)] for i in picks]
    results = [check(physics, z, generator) for z in outputs]

    renderer = dgp.Renderer()

    def observation_record(z: Tensor, label: str) -> dict[str, Any]:
        jac = _jacobian(lambda s: renderer(s).double().flatten(), z.float()).double()
        values = _singular_values(jac)
        return {"state": label, "rank": int((values > values[0] * 1e-8).sum()),
                "sigma_ratio": float(values[-1] / values[0])}  # fmt: skip

    observation = [observation_record(z, f"sampled {k}") for k, z in enumerate(outputs)]
    # Exact overlaps: every pair of beads (and all four) at one point, at each sampled state.
    names = ("1a", "1b", "2a", "2b")
    for k, z in enumerate(outputs):
        beads = dgp.unflatten_state(z).flatten(-3, -2)
        groups = [(i, j) for i in range(4) for j in range(i + 1, 4)] + [(0, 1, 2, 3)]
        for group in groups:
            moved = beads.clone()
            moved[list(group)] = beads[group[0]]
            label = f"sampled {k}, {'='.join(names[i] for i in group)}"
            observation.append(observation_record(dgp.flatten_state(moved.reshape(2, 2, 2)), label))

    step = _step(physics)
    support = dgp.graphs()["state"]
    magnitudes: list[Tensor] = []
    for k in range(200):
        group, time = k % 64, 10 + k % 25
        q = states[group, 0]

        def by_state(c: Tensor, p: Tensor = q[time - 1], a: Tensor = actions[group, 0, time],
                     th: Tensor = theta[group]) -> Tensor:  # fmt: skip
            return step(p, c, a, th)

        magnitudes.append(_jacobian(by_state, q[time]).abs())
    mean_abs = torch.stack(magnitudes).mean(0)
    off_diagonal = support & ~torch.eye(dgp.STATE_DIM, dtype=torch.bool)
    summary = {
        "physics": physics.as_record(),
        "renderer": renderer.as_record(),
        "per_state": results,
        "observation": observation,
        "observation_min_sigma_ratio": min(a["sigma_ratio"] for a in observation),
        "temporal_mean_abs_entry": {
            "diagonal_min": float(mean_abs.diagonal().min()),
            "off_diagonal_on_support_min": float(mean_abs[off_diagonal].min()),
            "off_diagonal_on_support_max": float(mean_abs[off_diagonal].max()),
            "off_support_max": float(mean_abs[~support].max()),
        },
        "all_hold": all(
            r["parameter_influence"]["holds"]
            and all(v["holds"] for v in r["action_influence"].values())
            and r["temporal_influence"]["both"]["holds"]
            for r in results
        )
        and all(a["rank"] == dgp.STATE_DIM for a in observation),
    }
    text = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
