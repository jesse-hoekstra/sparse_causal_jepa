"""Downstream causal inference: virtual tweezer experiments on trained vision models.

Q1 (tweezer): hold one true coordinate at ``a`` from the last context frame ``t`` on, re-imposed
after every step, and read every coordinate at ``t + 2``; the effect is the least-squares slope
of the outcome in ``a`` (an average causal effect per unit of the hold). Q2 (pulse with the
partner held): hold one bead at rest where it is and kick its dimer's field once at ``t``; read
the partner bead's displacement at ``t + 2`` per unit kick, with and without the hold.

A model is queried with exactly what identification leaves open: on the calibration split each
true coordinate is matched to one latent (the evaluation's Spearman assignment) and isotonic
maps convert between the two in both directions. The truth is the simulator applied to the
same contexts with the same holds and kicks. Only latent states are intervened on.

usage: python -m diagnostics.vision_dimer.downstream --data-dir DATA --root RUNS --split test
       --stored-dir EVALUATIONS --output OUT
(``RUNS`` holds ``seed<s>/<sub>/<arm>/final.pt``; ``EVALUATIONS`` holds the stored test
evaluations ``test-seed<s>-<arm>.json`` written by ``train evaluate --split test``.)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy
import torch
from scipy.optimize import isotonic_regression  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import dgp
from diagnostics.vision_dimer.train import load_checkpoint, model_from_checkpoint
from scjepa.eval.recovery import fit_coordinate_match
from scjepa.models.model import SCJEPA

NAMES = ("u1a", "v1a", "u1b", "v1b", "u2a", "v2a", "u2b", "v2b")
STARTS = (0, 14, 28)  # disjoint 12-frame windows per episode
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
CONTEXT, T = 10, 9  # context frames; the hold starts at window frame T (the last context frame)
BEADS = ((0, 1), (2, 3), (4, 5), (6, 7))  # (u, v) of beads 1a, 1b, 2a, 2b
PARTNER = (1, 0, 3, 2)
BATCH_GROUPS = 16
Hold = dict[int, Tensor]


class Monotone:
    """Isotonic calibration map; linear interpolation, constant extension (as the evaluation)."""

    def __init__(self, x: Tensor, y: Tensor) -> None:
        """Fit in the direction of the calibration rank correlation."""
        order = x.argsort()
        self.xs = x[order].numpy()
        ranks = torch.stack((x.argsort().argsort(), y.argsort().argsort())).double()
        increasing = bool(torch.corrcoef(ranks)[0, 1] >= 0)
        self.fitted = cast(Any, isotonic_regression(y[order].numpy(), increasing=increasing)).x

    def __call__(self, x: Tensor) -> Tensor:
        """Map values through the fitted step function."""
        return torch.tensor(numpy.interp(x.numpy(), self.xs, self.fitted), dtype=torch.float64)


class Queries(Protocol):
    """What Q1 and Q2 need from a world: held rollouts, physical readout and hold values."""

    def rollout(self, hold: Hold, kick: Tensor | None) -> Tensor:
        """State at t+2 (world coordinates) with ``hold`` re-imposed after every step."""
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Physical coordinates ``(N, 8)`` of world states."""
        ...

    def hold_value(self, j: int, a: float) -> Hold:
        """Hold of true coordinate ``j`` at physical value ``a``."""
        ...

    def current(self, bead: int) -> Hold:
        """Hold of both axes of ``bead`` at rest where it is at frame t."""
        ...


class Truth:
    """The simulator on the scored split's contexts."""

    def __init__(self, physics: dgp.Physics, ctx: dict[str, Tensor]) -> None:
        """Contexts: true frames t-1 and t, drags and actions a_{t-1}, a_t, a_{t+1}."""
        self.physics, self.ctx = physics, ctx

    def _step(self, prev: Tensor, cur: Tensor, action: Tensor) -> Tensor:
        state = dgp.transition(
            dgp.unflatten_state(prev),
            dgp.unflatten_state(cur),
            action,
            self.ctx["theta"],
            self.physics,
        )
        return dgp.flatten_state(state)

    def rollout(self, hold: Hold, kick: Tensor | None) -> Tensor:
        """Two simulator steps with the hold re-imposed after each."""
        prev, cur = self.ctx["prev"].clone(), self.ctx["cur"].clone()
        for j, v in hold.items():
            prev[:, j], cur[:, j] = v, v
        a_t = self.ctx["actions"][:, 1] + (0 if kick is None else kick)
        nxt = self._step(prev, cur, a_t)
        for j, v in hold.items():
            nxt[:, j] = v
        last = self._step(cur, nxt, self.ctx["actions"][:, 2])
        for j, v in hold.items():
            last[:, j] = v
        return last

    def decode(self, z: Tensor) -> Tensor:
        """Simulator states are physical coordinates."""
        return z

    def hold_value(self, j: int, a: float) -> Hold:
        """Hold coordinate ``j`` at ``a`` in every context."""
        return {j: torch.full((len(self.ctx["cur"]),), a, dtype=torch.float64)}

    def current(self, bead: int) -> Hold:
        """Hold the bead where it is at t."""
        return {i: self.ctx["cur"][:, i].clone() for i in BEADS[bead]}


class Model:
    """A trained model on the same contexts, named and scaled on the calibration split."""

    def __init__(
        self, model: SCJEPA, lat: dict[str, Tensor], cal: Tensor, cal_truth: Tensor
    ) -> None:
        """Spearman assignment and two-way isotonic maps, fitted on calibration latents/states."""
        self.model, self.lat = model, lat
        self.match = list(
            fit_coordinate_match(cal.float(), cal_truth.float(), method="spearman").columns
        )
        self.decoders = [Monotone(cal[:, self.match[i]], cal_truth[:, i]) for i in range(8)]
        self.encoders = [Monotone(cal_truth[:, i], cal[:, self.match[i]]) for i in range(8)]
        self.predictor = cast(Any, model.predictor)

    def _predict(self, history: Tensor, actions: Tensor) -> Tensor:
        output = self.predictor(history.float(), self.lat["code"].float(), actions.float())
        return cast(Tensor, output.prediction).double()

    @torch.no_grad()
    def rollout(self, hold: Hold, kick: Tensor | None) -> Tensor:
        """Two predictor steps from latent frames t-1, t; the second sees the first prediction."""
        hist = self.lat["online"][:, T - 1 : T + 1].clone()
        for c, v in hold.items():
            hist[:, :, c] = v[:, None]
        acts = self.lat["actions"][:, T - 1 : T + 2].clone()
        if kick is not None:
            acts[:, 1] += kick
        first = self._predict(hist, acts[:, 0:2])
        for c, v in hold.items():
            first[:, c] = v
        second = self._predict(torch.stack((hist[:, 1], first), 1), acts[:, 1:3])
        for c, v in hold.items():
            second[:, c] = v
        return second

    @torch.no_grad()
    def forecast(self) -> Tensor:
        """Unheld two-step forecast ``(N, 2, 8)``, computed as in the evaluation."""
        hist, acts = self.lat["online"][:, T - 1 : T + 1], self.lat["actions"][:, T - 1 : T + 2]
        first = self._predict(hist, acts[:, 0:2])
        return torch.stack(
            (first, self._predict(torch.stack((hist[:, 1], first), 1), acts[:, 1:3])), 1
        )

    def decode(self, z: Tensor) -> Tensor:
        """Read each true coordinate from its matched latent."""
        return torch.stack([self.decoders[i](z[:, self.match[i]]) for i in range(8)], -1)

    def hold_value(self, j: int, a: float) -> Hold:
        """Clamp the latent matched to coordinate ``j`` at the calibration value of ``a``."""
        value = self.encoders[j](torch.tensor([a], dtype=torch.float64))
        return {self.match[j]: value.expand(len(self.lat["code"])).clone()}

    def current(self, bead: int) -> Hold:
        """Clamp the bead's two matched latents at their frame-t values."""
        return {self.match[i]: self.lat["online"][:, T, self.match[i]].clone() for i in BEADS[bead]}


def as_list(t: Tensor) -> list[Any]:
    """JSON-ready nested list."""
    return cast(list[Any], t.tolist())  # pyright: ignore[reportUnknownMemberType]


def forecast_nmse(forecast: Tensor, targets: Tensor, rows: slice) -> float:
    """Two-step forecast MSE over the target variance, the evaluation's NMSE."""
    error = (forecast[rows] - targets[rows]).square().mean()
    return float(error / targets[rows].reshape(-1, 8).var(0).mean())


def slope(grid: Tensor, values: Tensor) -> Tensor:
    """Least-squares slope of ``values (len(grid), k)`` on ``grid``."""
    g = grid - grid.mean()
    return (g[:, None] * (values - values.mean(0))).sum(0) / g.square().sum()


def kicks(dimer: int) -> list[Tensor]:
    """Unit kicks on one dimer's camera-frame field, x then y (actions B, E1x, E1y, E2x, E2y)."""
    rows: list[Tensor] = []
    for axis in range(2):
        k = torch.zeros(dgp.ACTION_DIM, dtype=torch.float64)
        k[1 + 2 * dimer + axis] = 1.0
        rows.append(k)
    return rows


def experiments(world: Queries, grids: list[Tensor]) -> dict[str, Any]:
    """Q1 for every held coordinate and Q2 for every held bead."""
    q1: dict[str, Any] = {}
    for j in range(8):
        means = torch.stack(
            [
                world.decode(world.rollout(world.hold_value(j, float(a)), None)).mean(0)
                for a in grids[j]
            ]
        )
        q1[NAMES[j]] = {
            "grid": as_list(grids[j]),
            "means": as_list(means),
            "slopes": as_list(slope(grids[j], means)),
        }
    q2: dict[str, Any] = {}
    for bead, (u, _) in enumerate(BEADS):
        pu, pv = BEADS[PARTNER[bead]]
        result: dict[str, Any] = {}
        empty: Hold = {}
        for label, hold in (("held", world.current(bead)), ("free", empty)):
            base = world.decode(world.rollout(hold, None))
            columns = [
                (world.decode(world.rollout(hold, k)) - base)[:, [pu, pv]].mean(0)
                for k in kicks(bead // 2)
            ]
            result[label] = as_list(
                torch.stack(columns, 1)
            )  # rows: partner u, v; columns: kick x, y
        q2[NAMES[u][1:]] = result  # bead name, e.g. "2a"
    return {"q1": q1, "q2": q2}


@torch.no_grad()
def encode(
    model: SCJEPA, inputs: vdata.RenderedInputs, starts: tuple[int, ...], groups: int
) -> dict[str, Tensor]:
    """Online latents of each window frame, forecast-frame target latents, codes and actions."""
    out: dict[str, list[Tensor]] = {"online": [], "target": [], "code": [], "actions": []}
    for s in starts:
        for begin in range(0, groups, BATCH_GROUPS):
            idx = torch.arange(begin, min(begin + BATCH_GROUPS, groups))
            batch = inputs.batch(idx, torch.full((len(idx), inputs.views), s, dtype=torch.long))
            x, a = batch["observations"].flatten(0, 1), batch["actions"].flatten(0, 1)
            z = model.online_encoder.encode_sequence(x)
            out["online"].append(z)
            out["target"].append(model.target_encoder.encode_sequence(x[:, T + 1 : T + 3]))
            out["code"].append(
                cast(Tensor, model.parameter_encoder(z[:, :CONTEXT], a[:, : CONTEXT - 1]))
            )
            out["actions"].append(a)
    return {k: torch.cat(v).double() for k, v in out.items()}


def truth_contexts(
    tensors: dict[str, Tensor], starts: tuple[int, ...], groups: int
) -> dict[str, Tensor]:
    """True frames t-1, t, drags and actions a_{t-1}, a_t, a_{t+1}, in the order of ``encode``."""
    states, actions, theta = (
        tensors["states"].double(),
        tensors["actions"].double(),
        tensors["theta"].double(),
    )
    views = states.shape[1]
    rows: dict[str, list[Tensor]] = {"prev": [], "cur": [], "actions": [], "theta": []}
    for s in starts:
        for begin in range(0, groups, BATCH_GROUPS):
            g = slice(begin, min(begin + BATCH_GROUPS, groups))
            rows["prev"].append(states[g, :, s + T - 1].flatten(0, 1))
            rows["cur"].append(states[g, :, s + T].flatten(0, 1))
            rows["actions"].append(actions[g, :, s + T - 1 : s + T + 2].flatten(0, 1))
            rows["theta"].append(theta[g, None].expand(-1, views, -1).flatten(0, 1))
    return {k: torch.cat(v) for k, v in rows.items()}


def main() -> None:
    """Run the declared analysis for each listed run; one JSON per run plus the truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, required=True, help="outputs root holding seed<s>/<sub>/<arm>"
    )
    parser.add_argument("--sub", default="sparse-500k")
    parser.add_argument(
        "--runs", nargs="+", default=[f"{s}/{a}" for s in range(5) for a in ("full", "baseline")]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=10**9, help="tests only")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--stored-dir",
        type=Path,
        help="test: directory of the stored test evaluations test-seed<s>-<arm>.json",
    )
    args = parser.parse_args()
    if args.split == "test" and args.stored_dir is None:
        raise ValueError(
            "--split test needs --stored-dir (final.json holds the validation evaluation)"
        )
    torch.set_num_threads(args.threads)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    config = vdata.load_config(args.data_dir)
    cal_inputs = vdata.RenderedInputs(args.data_dir, "calibration", device)
    val_inputs = vdata.RenderedInputs(args.data_dir, args.split, device)
    cal_groups, val_groups = (
        min(len(cal_inputs), args.max_groups),
        min(len(val_inputs), args.max_groups),
    )
    cal_truth = vdata.load_labels(args.data_dir, "calibration")["states"][
        :cal_groups, :, : config.frames
    ]
    cal_truth = cal_truth.double().reshape(-1, 8)
    validation = vdata._load(args.data_dir, args.split)  # pyright: ignore[reportPrivateUsage]
    ctx = truth_contexts(validation, STARTS, val_groups)
    grids = [
        torch.quantile(cal_truth[:, j], torch.tensor(QUANTILES, dtype=torch.float64))
        for j in range(8)
    ]

    if not (output / "truth.json").exists():
        truth = experiments(Truth(config.physics, ctx), grids)
        truth["observational_slopes"] = as_list(
            torch.cov(cal_truth.T) / cal_truth.var(0)[None, :]
        )  # [i][j]: i on j
        truth["contexts"] = len(ctx["cur"])
        (output / "truth.json").write_text(json.dumps(truth, indent=1) + "\n")
        print("truth written", flush=True)

    for run in args.runs:
        seed, arm = run.split("/")
        clock = time.time()
        path = Path(args.root) / f"seed{seed}" / args.sub / arm / "final.pt"
        payload = load_checkpoint(path)
        model = model_from_checkpoint(
            payload, phase=payload.get("config", {}).get("stage", arm)
        ).to(device)
        model.eval()
        cal = encode(model, cal_inputs, (0,), cal_groups)["online"].reshape(-1, 8)
        world = Model(model, encode(model, val_inputs, STARTS, val_groups), cal, cal_truth)
        result = experiments(world, grids)
        forecast, targets = world.forecast(), world.lat["target"]
        first_window = (
            val_groups * val_inputs.views
        )  # contexts of the evaluation's own window (frames 0-11)

        stored_name = "final.json" if args.split == "validation" else f"test-seed{seed}-{arm}.json"
        stored_path = (path.parent if args.split == "validation" else args.stored_dir) / stored_name
        stored = json.loads(stored_path.read_text())
        if args.split == "test" and stored.get("scored_split") != "test":
            raise ValueError(f"{stored_path} is not a test evaluation")
        result.update(
            split=args.split,
            stored_evaluation=str(stored_path),
            pearson_assignment=stored["assignment"],
            seed=int(seed),
            arm=arm,
            assignment=world.match,
            stored_assignment=stored["spearman_assignment"],
            nmse_two_step=forecast_nmse(forecast, targets, slice(None)),
            nmse_first_window=forecast_nmse(forecast, targets, slice(0, first_window)),
            stored_nmse=stored["nmse"],
            per_coordinate_spearman=stored["per_coordinate_spearman"],
            spearman_mcc=stored["spearman_mcc"],
            contexts=len(world.lat["code"]),
            checkpoint=str(path),
            seconds=time.time() - clock,
        )
        (output / f"seed{seed}-{arm}.json").write_text(json.dumps(result, indent=1) + "\n")
        same = world.match == stored["spearman_assignment"]
        print(
            f"{run}: {time.time() - clock:.0f}s, two-step nmse {result['nmse_two_step']:.5f}, "
            f"first window {result['nmse_first_window']:.5f} "
            f"(stored two-step {stored['nmse']:.5f}), "
            f"assignment {'matches' if same else 'DIFFERS'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
