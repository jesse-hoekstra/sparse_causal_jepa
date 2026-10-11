"""Short CPU-only forward/backward validation using random tensors, never a dataset."""

import json

import torch

from scjepa.training import ObjectiveConfig, TrainConfig, Trainer, seed_everything
from scjepa.training.factory import ModelConfig, build_model


def main() -> None:
    """Check image/vector gradients, finite updates, and EMA with small tensors."""
    torch.set_num_threads(1)
    seed_everything(7)
    for kind in ("vector", "image"):
        config = ModelConfig(
            observation_kind=kind,
            state_dim=5,
            param_dim=3,
            action_dim=2,
            context_len=3,
            history_len=2,
            encoder_hidden_dims=(16,),
            image_channels=(8, 16),
            parameter_embed_dim=16,
            spartan_embed_dim=16,
            spartan_mlp_hidden=16,
            spartan_dense=True,
        )
        trainer = Trainer(
            build_model(config),
            TrainConfig(
                batch_size=3,
                accumulation_steps=2,
                objective=ObjectiveConfig(
                    state_regularizer="sigreg",
                    state_regularizer_weight=0.01,
                    parameter_regularizer="sigreg",
                    parameter_regularizer_weight=0.01,
                    regularizer_num_projections=8,
                ),
            ),
        )
        shape = (3, 6, 8) if kind == "vector" else (3, 6, 3, 16, 16)
        batch = {
            "observations": torch.randn(shape),
            "actions": torch.randn(3, 5, 2),
            "episode_ids": torch.arange(3),
        }
        first = trainer.train_step(batch)
        assert first["health/updated"] == 0
        for module in (
            trainer.model.online_encoder,
            trainer.model.parameter_encoder,
            trainer.model.predictor,
        ):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None for p in trainer.model.target_encoder.parameters())
        metrics = trainer.train_step(batch)
        assert metrics["health/updated"] == 1
        assert metrics["health/skipped"] == 0
        assert torch.isfinite(torch.tensor(metrics["loss/total"]))
        print(json.dumps({"encoder": kind, "prediction_shape": [3, 3, 5], **metrics}))


if __name__ == "__main__":
    main()
