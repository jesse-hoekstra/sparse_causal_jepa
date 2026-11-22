"""Whiten a fixed teacher's chart and add a forecast-norm-capped code-encoder gradient."""

from typing import cast

import torch
from torch import Tensor, nn


def whitening(codes: Tensor) -> Tensor:
    """Return the symmetric full-rank TRAIN inverse square root in float64."""
    if codes.ndim != 3 or codes.shape[1] < 2:
        raise ValueError("expected grouped teacher codes")
    rows = codes.detach().double().flatten(0, 1)
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(covariance))  # pyright: ignore[reportUnknownMemberType]
    if not torch.isfinite(values).all() or values[0] <= (values[-1] * 1e-12).clamp_min(1e-300):
        raise ValueError("frozen TRAIN teacher covariance lacks complete resolved support")
    return (vectors * values.rsqrt()) @ vectors.T


def anchor(student: Tensor, teacher: Tensor, invsqrt: Tensor, mode: str) -> Tensor:
    """Mean squared distance to the same episode or the exclusively other-episode mean."""
    if student.ndim != 3 or student.shape != teacher.shape or teacher.shape[1] < 2:
        raise ValueError("teacher and student must have identical grouped episode axes")
    frozen = teacher.detach().double()
    if mode == "other":
        target = (frozen.sum(1, keepdim=True) - frozen) / (frozen.shape[1] - 1)
    elif mode == "same":
        target = frozen
    else:
        raise ValueError("unknown frozen-teacher target")
    return ((student.double() - target) @ invsqrt.detach()).square().mean()


def add_capped_gradient(
    forecast: Tensor, auxiliary: Tensor, params: tuple[nn.Parameter, ...]
) -> dict[str, Tensor]:
    """Preaccumulate only the bounded auxiliary gradient before ordinary native backward."""
    if not params or any(p.grad is not None for p in params):
        raise ValueError("expected freshly zeroed trainable code-encoder gradients")
    native = torch.autograd.grad(forecast, params, retain_graph=True)
    auxiliary_grad = torch.autograd.grad(auxiliary, params, retain_graph=True)
    nf = torch.stack([g.detach().square().sum() for g in native]).sum().sqrt()
    na = torch.stack([g.detach().square().sum() for g in auxiliary_grad]).sum().sqrt()
    tiny = torch.finfo(nf.dtype).tiny
    scale = torch.where(na > 0, (nf / na.clamp_min(tiny)).clamp_max(1), 0.0)
    for p, grad in zip(params, auxiliary_grad, strict=True):
        p.grad = grad.detach() * scale
    dot = torch.stack(
        [(a.detach() * b.detach()).sum() for a, b in zip(native, auxiliary_grad, strict=True)]
    ).sum()
    return {
        "consensus_gradient_cosine": torch.where(nf * na > 0, dot / (nf * na).clamp_min(tiny), 0.0),
        "consensus_cap_active": (na > nf).to(nf.dtype),
        "consensus_loss": auxiliary.detach(),
        "consensus_forecast_grad_norm": nf,
        "consensus_raw_grad_norm": na,
        "consensus_gradient_scale": scale,
    }


def context_intervals(
    codes: dict[str, Tensor], errors: dict[str, Tensor], seed: int, *, draws: int = 2000
) -> dict[str, object]:
    """Paired percentile intervals for differences and ratios with entire systems kept together."""
    names = ("native", "same", "other")
    shape = codes["native"].shape
    if len(shape) != 3 or any(codes[a].shape != shape for a in names):
        raise ValueError("unaligned system or code dimensions")
    groups, episodes, width = shape
    if min(groups, episodes, draws) < 2 or any(errors[a].shape != (groups,) for a in names):
        raise ValueError("need paired system errors and at least two bootstrap draws")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(groups, (draws, groups), generator=generator)
    weights = torch.zeros((draws, groups), dtype=torch.float64)
    weights.scatter_add_(1, indices, torch.ones_like(weights))
    weights = torch.cat((torch.ones_like(weights[:1]), weights)) / groups
    results: dict[str, tuple[Tensor, Tensor]] = {}
    for name in names:
        code = codes[name].detach().double().cpu()
        code = code - code.mean((0, 1))
        means = code.mean(1)
        within = code - means[:, None]
        w = (within.transpose(1, 2) @ within / episodes).flatten(1)
        tt = (code.transpose(1, 2) @ code / episodes).flatten(1)
        mu = weights @ means
        total = (weights @ tt).reshape(-1, width, width) - mu[:, :, None] * mu[:, None, :]
        eig, vec = cast(tuple[Tensor, Tensor], torch.linalg.eigh(total))  # pyright: ignore[reportUnknownMemberType]
        if (
            eig[:, 0] <= eig[:, -1] * max(groups * episodes, width) * torch.finfo(eig.dtype).eps
        ).any():
            raise ValueError("unresolved code directions in a paired bootstrap draw")
        wh = vec * eig.rsqrt().unsqueeze(-2)
        ratios = wh.transpose(1, 2) @ (weights @ w).reshape(-1, width, width) @ wh
        worst = cast(Tensor, torch.linalg.eigvalsh(ratios))[:, -1]  # pyright: ignore[reportUnknownMemberType]
        results[name] = (worst, weights @ errors[name].detach().double().cpu())
    ans: dict[str, object] = {}
    for reference in names[:2]:
        block: dict[str, object] = {}
        for metric, ref, treatment in zip(
            ("worst", "mse"), results[reference], results["other"], strict=True
        ):
            for suffix, v in (("difference", treatment - ref), ("ratio", treatment / ref)):
                if not torch.isfinite(v).all():
                    raise ValueError("nonfinite bootstrap contrast")
                ci = torch.quantile(v[1:], v.new_tensor([0.025, 0.975]))
                block[f"{metric}_{suffix}"] = {
                    "estimate": float(v[0]),
                    "ci95": [float(ci[0]), float(ci[1])],
                }
        ans[f"other_minus_{reference}"] = block
    return ans
