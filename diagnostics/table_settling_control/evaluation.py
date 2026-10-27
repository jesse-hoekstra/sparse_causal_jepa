"""Independent episodes and retrospective readouts; no evaluation gradients enter fitting."""

from typing import Any

import torch
from torch import Tensor, nn

from diagnostics.group_latent_control.forward import group_teacher_forward
from diagnostics.multiquery_control.data import TensorSplit, encode_states
from diagnostics.multiquery_control.evaluation import parameter_readouts
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


@torch.no_grad()
def new_episodes(model: SCJEPA, theta: Tensor, matrix: Tensor, seed: int) -> TensorSplit:
    """Generate ten new trajectories per already-trained system on a separate RNG stream."""
    generator = torch.Generator().manual_seed(seed)
    shape = (len(theta), 10, 11)
    initial = torch.randn(len(theta), 10, 6, generator=generator, dtype=torch.float64)
    gaussian = torch.randn(*shape, 10, generator=generator, dtype=torch.float64)
    uniform = 2 * torch.rand(*shape, 6, generator=generator, dtype=torch.float64) - 1
    actions = dgp.encode_controls(torch.cat((gaussian, uniform), dim=-1))
    states = dgp.simulate(initial, theta.double()[:, None], actions)
    observation = VectorObservationMap(matrix=matrix, dimension=6).forward(states).float()
    online, target = encode_states(model, observation, 32)
    return {"online": online, "target": target, "actions": actions.float()}


@torch.no_grad()
def forecast_errors(
    model: SCJEPA, table: nn.Embedding, data: TensorSplit, batch_size: int
) -> Tensor:
    """Evaluate per-training-system, per-horizon MSE in eval mode without changing RNG."""
    if set(data) != {"online", "target", "actions"}:
        raise ValueError("Forecast evaluation accepts state/action tensors only")
    reference = next(model.parameters())
    with preserve_rng(reference.device):
        model.eval()
        pieces: list[Tensor] = []
        for start in range(0, len(data["online"]), batch_size):
            batch = {
                key: value[start : start + batch_size].to(reference) for key, value in data.items()
            }
            codes = table.weight[start : start + batch_size]
            output = group_teacher_forward(model, batch, codes)
            pieces.append((output.predictions - output.targets).square().mean(dim=(1, 3)).cpu())
        return torch.cat(pieces)


def paired_change(before: Tensor, after: Tensor, *, samples: int, seed: int) -> dict[str, Any]:
    """Bootstrap paired TRAIN systems; negative changes mean improved prediction."""
    change = (after - before).double()
    draws = torch.randint(
        len(change), (samples, len(change)), generator=torch.Generator().manual_seed(seed)
    )
    bootstrap = change[draws].mean(dim=1)
    return {
        "mean": change.mean(0).tolist(),  # pyright: ignore[reportUnknownMemberType]
        "ci95": torch.quantile(  # pyright: ignore[reportUnknownMemberType]
            bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0
        ).T.tolist(),
    }


@torch.no_grad()
def readout(table: nn.Embedding, theta: Tensor) -> dict[str, Any]:
    """Use a fixed 128/64/64 partition of selected TRAIN systems, not unseen-system acceptance."""
    count = len(theta)
    bounds = (0, count // 2, 3 * count // 4, count)
    values = table.weight.detach().cpu()
    codes: dict[str, Tensor] = {}
    labels: dict[str, dict[str, Tensor]] = {}
    for name, start, end in zip(
        ("calibration", "validation", "test"), bounds[:-1], bounds[1:], strict=True
    ):
        codes[name] = values[start:end, None]
        labels[name] = {"theta": theta[start:end]}
    return parameter_readouts(codes, labels)
