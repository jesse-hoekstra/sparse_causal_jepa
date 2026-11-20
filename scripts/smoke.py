"""Short CPU forward/backward checks of the published model paths of both experiments."""

import json
from typing import cast

import torch
from torch import nn

from diagnostics.masked_mlp_learned import run as learned
from diagnostics.masked_mlp_oracle.model import build_control as build_oracle
from diagnostics.shared.oracle import forward_loss as oracle_loss
from diagnostics.shared.oracle import objective_config
from diagnostics.vision_dimer import dgp
from diagnostics.vision_dimer.model import build_model as build_vision
from scjepa.models.model import ModelOutput
from scjepa.training.loop import TrainingObjective, seed_everything


def main() -> None:
    """Check attached gradients, finite updates and stopped EMA targets on random inputs."""
    torch.set_num_threads(1)
    seed_everything(7)
    batch = {
        "observations": torch.randn(2, 10, 12, 6),
        "actions": torch.randn(2, 10, 11, 24),
        "oracle_parameters": torch.rand(2, 6) - 0.5,
        "episode_ids": torch.arange(2),
    }
    for mode in ("oracle", "learned"):
        model = build_oracle("full", 7) if mode == "oracle" else learned.build_control("full", 7)
        objective = TrainingObjective(objective_config("full", logit_weight=0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
        forward = oracle_loss if mode == "oracle" else learned.forward_loss
        loss, _, output = forward(model, batch, objective)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        modules: list[nn.Module] = [model.online_encoder, model.predictor]
        if mode == "learned":
            modules.append(model.parameter_encoder)
        assert all(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
            for module in modules
        )
        assert all(p.grad is None for p in model.target_encoder.parameters())
        assert torch.isfinite(loss)
        assert len(output.dynamics) == 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        model.update_target()
        print(json.dumps({"mode": mode, "loss": float(loss.detach()), "two_attached_steps": True}))
    vision()


def vision() -> None:
    """The vision model: RGB frames to two attached prediction steps with every gate."""
    model = build_vision(7, phase="full", code_dim=dgp.PARAM_DIM, context=10)
    observations = torch.rand(2, 3, 12, *dgp.Renderer().shape)
    actions = torch.randn(2, 3, 11, dgp.ACTION_DIM)
    output = cast(ModelOutput, model(observations, actions))
    sparsity = torch.stack(
        [d.sparsity.state + d.sparsity.action + d.sparsity.parameter for d in output.dynamics]
    ).mean()
    loss = (output.predictions - output.targets).square().mean() + 1e-3 * sparsity
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    modules: list[nn.Module] = [model.online_encoder, model.parameter_encoder, model.predictor]
    assert all(
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        for module in modules
    )
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert torch.isfinite(loss)
    assert len(output.dynamics) == 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    model.update_target()
    print(json.dumps({"mode": "vision", "loss": float(loss.detach()), "two_attached_steps": True}))


if __name__ == "__main__":
    main()
