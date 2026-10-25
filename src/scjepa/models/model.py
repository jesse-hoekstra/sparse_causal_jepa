"""A framewise online encoder, contextual parameters, sparse dynamics, and EMA targets."""

import copy
from typing import NamedTuple, Self

import torch
from torch import Tensor, nn

from scjepa.models.encoders import FramewiseEncoder
from scjepa.models.parameter_encoder import (
    ParameterEncoder,
    TransitionAttentionParameterEncoder,
    TransitionParameterEncoder,
)
from scjepa.models.spartan import Spartan, SpartanOutput


def cross_episode_parameters(parameters: Tensor, *, training: bool = False) -> Tensor:
    """Assign a different episode's attached code to every recipient in each group.

    Training draws one nonzero cyclic shift per call for R>2; evaluation uses
    shift one. R=2 retains the exact previous swap without consuming RNG state.
    """
    if parameters.ndim != 3 or parameters.shape[1] < 2:
        raise ValueError("cross-episode codes require shape (B, R>=2, d_theta)")
    views = parameters.shape[1]
    shift = (
        int(torch.randint(1, views, (), device=parameters.device).item())
        if training and views > 2
        else 1
    )
    return parameters.roll(shift, dims=1)


class ModelOutput(NamedTuple):
    """Sequence predictions and explicit inputs for configurable objectives."""

    predictions: Tensor
    """Predicted next states, ``(B, W, d_z)`` or grouped ``(B, R, W, d_z)``."""
    targets: Tensor
    """Stopped-gradient EMA next states with the same shape as predictions."""
    online_states: Tensor
    """Independent frame encodings, ``(B, T, d_z)`` or ``(B, R, T, d_z)``."""
    parameters: Tensor
    """Own-context codes or raw oracle values, ``(B, d_theta)`` or ``(B, R, d_theta)``."""
    dynamics: tuple[SpartanOutput, ...]
    """One predictor result per transition; paired graphs flatten the first two axes."""
    prediction_start: int
    """First source index t; targets begin at t+1."""


class SCJEPA(nn.Module):
    """Experiment-independent model with independent context and dynamics lengths."""

    def __init__(
        self,
        encoder: FramewiseEncoder,
        parameter_encoder: (
            ParameterEncoder | TransitionParameterEncoder | TransitionAttentionParameterEncoder
        ),
        predictor: Spartan,
        context_len: int,
        ema_decay: float = 0.996,
        autoregressive: bool = False,
        parameter_source: str = "learned",
    ) -> None:
        """Copy and freeze the complete encoder, including its output head."""
        super().__init__()
        if parameter_source not in {"learned", "oracle"}:
            raise ValueError("parameter_source must be learned or oracle")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not 1 <= context_len <= parameter_encoder.max_history:
            raise ValueError("context_len must lie in [1, parameter_encoder.max_history]")
        if (
            isinstance(
                parameter_encoder, TransitionParameterEncoder | TransitionAttentionParameterEncoder
            )
            and context_len < 2
        ):
            raise ValueError("transition parameter encoder requires context_len>=2")
        if (
            encoder.state_dim != parameter_encoder.state_dim
            or encoder.state_dim != predictor.state_dim
        ):
            raise ValueError(
                "state dimensions of the encoder, parameter encoder and predictor differ"
            )
        if parameter_encoder.param_dim != predictor.param_dim:
            raise ValueError("parameter dimensions of the parameter encoder and predictor differ")
        if parameter_encoder.action_dim != predictor.action_dim:
            raise ValueError("action dimensions of the parameter encoder and predictor differ")
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.parameter_encoder = parameter_encoder
        self.parameter_source = parameter_source
        if self.parameter_source == "oracle":
            self.parameter_encoder.requires_grad_(False)
            self.parameter_encoder.eval()
        self.predictor = predictor
        self.context_len = context_len
        self.ema_decay = ema_decay
        self.autoregressive = autoregressive

    def train(self, mode: bool = True) -> Self:
        """Train online modules while keeping the target encoder in evaluation mode."""
        super().train(mode)
        self.target_encoder.eval()
        if self.parameter_source == "oracle":
            self.parameter_encoder.eval()
        return self

    def get_extra_state(self) -> dict[str, float | int | bool]:
        """Checkpoint EMA decay, context and prediction mode with encoder weights."""
        return {
            "ema_decay": self.ema_decay,
            "context_len": self.context_len,
            "autoregressive": self.autoregressive,
            "oracle_parameters": self.parameter_source == "oracle",
        }

    def set_extra_state(self, state: dict[str, float | int | bool]) -> None:
        """Restore configuration; older checkpoints used observed state histories."""
        if state.get("oracle_parameters", False) != (self.parameter_source == "oracle"):
            raise ValueError("checkpoint parameter_source differs from the model")
        decay, context = float(state["ema_decay"]), int(state["context_len"])
        if not 0 <= decay < 1 or not 1 <= context <= self.parameter_encoder.max_history:
            raise ValueError("invalid checkpoint EMA decay or context length")
        if (
            isinstance(
                self.parameter_encoder,
                TransitionParameterEncoder | TransitionAttentionParameterEncoder,
            )
            and context < 2
        ):
            raise ValueError("transition parameter encoder requires context_len>=2")
        self.ema_decay = decay
        self.context_len = context
        self.autoregressive = bool(state.get("autoregressive", False))
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        if self.parameter_source == "oracle":
            self.parameter_encoder.requires_grad_(False)
            self.parameter_encoder.eval()

    @torch.no_grad()
    def update_target(self) -> None:
        """Apply parameter EMA and copy every buffer after an accepted optimizer step.

        Retains the audited EMA policy of the previous implementation: floating
        and nonfloating buffers are copied exactly, never exponentially averaged.
        The trainer calls this once per successful optimizer update, excluding
        gradient-accumulation microsteps and nonfinite/skipped updates.
        """
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            target.mul_(self.ema_decay).add_(online.detach(), alpha=1 - self.ema_decay)
        for target_buffer, online_buffer in zip(
            self.target_encoder.buffers(), self.online_encoder.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)
        self.target_encoder.eval()

    def forward(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        prediction_start: int | None = None,
        *,
        oracle_parameters: Tensor | None = None,
    ) -> ModelOutput:
        """Predict future states using one parameter inference per episode.

        Observations are ``(B,T,*observation_shape)``. Actions are ``(B,T-1,d_a)``:
        action ``a_t`` advances ``x_t`` to ``x_{t+1}``. Parameter inference sees
        only states ``[:K]`` and actions ``[:K-1]``. A source index ``t`` uses
        states/actions ``[t-L+1:t+1]`` and predicts target state ``t+1``.
        The first source defaults to ``max(K,L)-1`` and the last is ``T-2``.
        Autoregressive mode initializes the history from observed states at the
        first source, then appends attached predictions instead of observations.
        Known actions remain aligned to each transition in either mode.

        Grouped observations/actions add an R>=2 episode axis after the batch axis.
        Each episode uses a distinct episode's attached parameter code while
        retaining its own state history, actions and target. The returned codes
        still correspond to their own contexts; all recipients share one mean loss.
        Explicit oracle mode instead requires raw ``(B, d_theta)`` parameters,
        detached and reused across every view and forecast step of each group.
        """
        if self.parameter_source == "learned" and oracle_parameters is not None:
            raise ValueError("learned parameter_source cannot consume oracle_parameters")
        if self.parameter_source == "oracle" and oracle_parameters is None:
            raise ValueError("oracle parameter_source requires oracle_parameters")
        paired = observations.ndim == self.online_encoder.observation_ndim + 3
        groups = 0
        views = 1
        if paired:
            groups, views = observations.shape[:2]
            if views < 2:
                raise ValueError("paired observations must contain at least two episodes")
            if actions is not None:
                expected = (groups, views, observations.shape[2] - 1, self.predictor.action_dim)
                if actions.shape != expected:
                    raise ValueError("paired actions must have shape (B, R, T-1, action_dim)")
                actions = actions.flatten(0, 1)
            observations = observations.flatten(0, 1)
        elif observations.ndim != self.online_encoder.observation_ndim + 2:
            raise ValueError(
                "observations must have shape (B, T, *observation_shape) "
                "or (B, R, T, *observation_shape)"
            )
        batch, length = observations.shape[:2]
        minimum_start = max(self.context_len, self.predictor.history_len) - 1
        start = minimum_start if prediction_start is None else prediction_start
        if batch < 1 or not minimum_start <= start < length - 1:
            raise ValueError("prediction_start must permit full context/history and a next frame")
        if self.predictor.action_dim:
            if actions is None or actions.shape != (batch, length - 1, self.predictor.action_dim):
                raise ValueError("actions must have shape (B, T-1, action_dim)")
        elif actions is not None:
            raise ValueError("actions must be None when action_dim=0")
        online_states = self.online_encoder.encode_sequence(observations)
        if self.parameter_source == "oracle":
            assert oracle_parameters is not None
            expected = (groups if paired else batch, self.predictor.param_dim)
            if oracle_parameters.shape != expected:
                raise ValueError(f"oracle_parameters must have shape {expected}")
            if (
                not oracle_parameters.is_floating_point()
                or oracle_parameters.dtype != online_states.dtype
                or oracle_parameters.device != online_states.device
            ):
                raise ValueError("oracle_parameters must match the state dtype and device")
            if not bool(torch.isfinite(oracle_parameters).all()):
                raise ValueError("oracle_parameters must be finite")
            parameters = oracle_parameters.detach()
            if paired:
                parameters = parameters[:, None].expand(-1, views, -1).reshape(batch, -1)
            predictor_parameters = parameters
        else:
            context_actions = None if actions is None else actions[:, : self.context_len - 1]
            parameters = self.parameter_encoder(
                online_states[:, : self.context_len], context_actions
            )
            predictor_parameters = (
                cross_episode_parameters(
                    parameters.reshape(groups, views, self.predictor.param_dim),
                    training=self.training,
                ).flatten(0, 1)
                if paired
                else parameters
            )
        with torch.no_grad():
            targets = self.target_encoder.encode_sequence(observations[:, start + 1 :])
        transitions: list[SpartanOutput] = []
        history = online_states[:, start - self.predictor.history_len + 1 : start + 1]
        for source in range(start, length - 1):
            first = source - self.predictor.history_len + 1
            history_actions = None if actions is None else actions[:, first : source + 1]
            if not self.autoregressive:
                history = online_states[:, first : source + 1]
            transition = self.predictor(history, predictor_parameters, history_actions)
            transitions.append(transition)
            if self.autoregressive:
                history = torch.cat((history[:, 1:], transition.prediction[:, None]), dim=1)
        predictions = torch.stack([step.prediction for step in transitions], dim=1)
        if paired:
            predictions = predictions.reshape(
                groups, views, length - start - 1, self.predictor.state_dim
            )
            targets = targets.reshape(groups, views, length - start - 1, self.predictor.state_dim)
            online_states = online_states.reshape(groups, views, length, self.predictor.state_dim)
            parameters = parameters.reshape(groups, views, self.predictor.param_dim)
        return ModelOutput(
            predictions, targets, online_states, parameters, tuple(transitions), start
        )

    def rollout(
        self,
        states_history: Tensor,
        parameters: Tensor,
        steps: int,
        actions: Tensor | None = None,
    ) -> Tensor:
        """Append predicted states and shift a fixed-length history at every step.

        Initial states have shape ``(B,L,d_z)``. For action-conditioned dynamics,
        supply ``(B,L+steps-1,d_a)`` with its first ``L`` actions aligned to the
        initial history. Parameters are reused as the same attached tensor;
        rollout length never changes the model's input lag count.
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        if states_history.ndim != 3 or states_history.shape[1:] != (
            self.predictor.history_len,
            self.predictor.state_dim,
        ):
            raise ValueError("states_history must have shape (B, history_len, state_dim)")
        batch = states_history.shape[0]
        if parameters.shape != (batch, self.predictor.param_dim):
            raise ValueError("parameters must have shape (B, param_dim)")
        if self.predictor.action_dim:
            if actions is None or actions.shape != (
                batch,
                self.predictor.history_len + steps - 1,
                self.predictor.action_dim,
            ):
                raise ValueError("rollout actions must have shape (B, L+steps-1, action_dim)")
        elif actions is not None:
            raise ValueError("actions must be None when action_dim=0")
        history = states_history
        predictions: list[Tensor] = []
        for step in range(steps):
            window_actions = (
                None if actions is None else actions[:, step : step + self.predictor.history_len]
            )
            prediction = self.predictor(history, parameters, window_actions).prediction
            predictions.append(prediction)
            history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
        return torch.stack(predictions, dim=1)


__all__ = ["SCJEPA", "ModelOutput", "cross_episode_parameters"]
