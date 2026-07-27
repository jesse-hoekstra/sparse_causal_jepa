"""State-to-state regime: true object states in, true next state out.

The oracle rung of the ladder — perception, state grounding and slot
correspondence are all given, so a failure here is a failure of causal parameter
identification and nothing else. Row i IS physical object i, so no track
alignment exists anywhere in training or evaluation.

(experiments.pdf §6.2.)

The context encoder is the identity on states plus the parameter encoder
(Eq. 15): E⁽¹⁾_ctx[Z_{0:Tpar-1}] = (Z_{Tpar-1}, θ̂) with θ̂ = P_η(Z_{0:Tpar-1}).
SPARTAN then makes the |I| = K teacher-forced one-step predictions of Eq. 38,

    Ẑ_{t+1} = f_gamma(Z_t, θ̂),    t ∈ I = {Tpar-1, …, T-2},

every one anchored at the OBSERVED current state Z_t (these are 30 one-step
predictions, not a 30-step open-loop rollout), reusing the same θ̂ for all
transitions of the episode. Gates, logits, and adjacencies are resampled per
transition (Eq. 33's fresh Gumbel noise), which falls out of flattening the K
transitions into the batch axis of one SPARTAN call.

The hybrid objective adds a SECOND branch alongside this one (hybrid write-up
§4.2): a dense K-step autoregressive rollout from a sampled start, in which the
predictor consumes its own output and every prefix k = 1..K is supervised. The
teacher-forced branch keeps the mechanism anchored at observed states; the
rollout branch is what forces one fixed θ̂ to remain valid along a generated
trajectory (§4.4(ii)). Both branches share the single θ̂ pooled above.

There is no target encoder and no representation-collapse regularizer; the
targets are the fixed observed states Z_{t+1} (§6.2 "Training uses aligned
raw-state MSE"). A constant or uninformative θ̂ is a failure to identify the
parameters, not a collapsed trivial optimum.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan


class TransitionOutput(NamedTuple):
    """One forward pass over an episode batch, flattened to (B·K, ...) rows.

    The graph quantities (``path_matrix``, ``sparsity``, ``logit_penalty`` and
    the gate diagnostics) are those of the TEACHER-FORCED pass only. The
    rollout branch reuses the same predictor and therefore the same gate
    distribution, but its extra SPARTAN calls do not contribute to the path
    objective — L_path stays the Eq. 11 quantity measured at observed states.
    """

    prediction: Float[Tensor, "bk n k"]
    target: Float[Tensor, "bk n k"]
    causal_params: Float[Tensor, "b n 1"]
    path_matrix: Float[Tensor, "bk m m"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]
    # Hybrid Eqs. 33-35; None when the rollout branch is disabled.
    rollout_prediction: Float[Tensor, "b j n k"] | None = None
    rollout_target: Float[Tensor, "b j n k"] | None = None


class StateToStateModel(nn.Module):
    """Parameter encoder P_η + SPARTAN f_gamma on ground-truth object states."""

    def __init__(self, parameter_encoder: ParameterEncoder, predictor: Spartan) -> None:
        """Compose the two modules of §6.2 — nothing else exists in this model."""
        super().__init__()
        self.parameter_encoder = parameter_encoder
        self.predictor = predictor

    def forward(
        self,
        states: Float[Tensor, "b t n k"],
        context_len: int | None = None,
        rollout_len: int | None = None,
    ) -> TransitionOutput:
        """θ̂ from the context window, then the two branches of the hybrid objective.

        Args:
            states: (B, T, N, k) episode batch; the parameter encoder sees
                observations 0..Tpar-1, and the supervised pairs are
                (Z_{Tpar-1}, Z_{Tpar}), …, (Z_{T-2}, Z_{T-1})  (Eq. 7).
            context_len: Tpar (the state-to-state regime: 30). None -> T-1 (K = 1).
            rollout_len: K for the autoregressive branch (hybrid Eqs. 33-35).
                None disables it, reproducing the pure teacher-forced objective.
        """
        if states.ndim != 4 or states.shape[1] < 2:
            raise ValueError(f"expected (B, T>=2, N, k), got {tuple(states.shape)}")
        length = states.shape[1]
        tpar = context_len if context_len is not None else length - 1
        if not 1 <= tpar < length:
            raise ValueError(f"context_len={tpar} must be in [1, T-1={length - 1}]")
        num_transitions = length - tpar  # K

        causal_params = self.parameter_encoder(states[:, :tpar])  # (B, N, 1), Eq. 26
        anchors = states[:, tpar - 1 : -1].flatten(0, 1)  # true Z_t, t ∈ I_TF
        targets = states[:, tpar:].flatten(0, 1)  # true Z_{t+1}
        params = causal_params.repeat_interleave(num_transitions, dim=0)  # same θ̂ ∀ t
        out = self.predictor(anchors, params)

        rollout_prediction = rollout_target = None
        if rollout_len is not None:
            rollout_prediction, rollout_target = self._rollout(
                states, tpar, rollout_len, causal_params
            )
        return TransitionOutput(
            prediction=out.prediction,
            target=targets,
            causal_params=causal_params,
            path_matrix=out.path_matrix,
            sparsity=out.sparsity,
            logit_penalty=out.logit_penalty,
            mean_abs_logit=out.mean_abs_logit,
            mean_gate_probability=out.mean_gate_probability,
            gate_entropy=out.gate_entropy,
            rollout_prediction=rollout_prediction,
            rollout_target=rollout_target,
        )

    def _rollout(
        self,
        states: Float[Tensor, "b t n k"],
        tpar: int,
        horizon: int,
        causal_params: Float[Tensor, "b n 1"],
    ) -> tuple[Float[Tensor, "b j n k"], Float[Tensor, "b j n k"]]:
        """Hybrid Eqs. 33-34: dense K-step autoregressive rollout.

            Ŝ^[0]_t := S^on_t,   Ŝ^[k]_{t+k} := f_gamma(Ŝ^[k-1]_{t+k-1}, θ̂)

        The chain starts at t = Tpar-1, the first state after the parameter
        context and the same anchor the first teacher-forced transition uses,
        and runs over t+1, …, t+K. There is no sampling of the start: one fixed
        anchor per episode makes train and eval compute the same quantity,
        which matters because τ is calibrated on the eval constraint.

        For k >= 2 the predictor consumes its own previous output, never a
        fresh encoding — this is what forces one fixed θ̂ to stay valid along a
        generated trajectory (§4.4(ii)). The same θ̂ is reused at every step.
        Gates are resampled per step (Eq. 33's fresh noise) because each step
        is a separate SPARTAN call.
        """
        length = states.shape[1]
        if tpar + horizon > length:
            raise ValueError(
                f"rollout_len={horizon} runs past the episode: need Tpar-1+K <= T-1, "
                f"got {tpar - 1}+{horizon} > {length - 1} (T={length}, Tpar={tpar})"
            )
        state = states[:, tpar - 1]  # Ŝ^[0] := S^on_t, t = Tpar-1 (Eq. 33)
        predictions: list[Tensor] = []
        for _ in range(horizon):
            state = self.predictor(state, causal_params).prediction  # Eq. 34
            predictions.append(state)
        # Targets S̄_{t+1}, …, S̄_{t+K}; .detach() is Eq. 35's sg(·).
        target = states[:, tpar : tpar + horizon].detach()
        return torch.stack(predictions, dim=1), target


def build_state_to_state(
    state_dim: int = 4,
    num_slots: int = 5,
    param_encoder_dim: int = 32,
    param_encoder_heads: int = 4,
    max_history: int = 64,
    spartan_layers: int = 3,
    spartan_embed_dim: int = 512,
    spartan_mlp_hidden: int = 512,
    spartan_mlp_layers: int = 3,
    spartan_temperature: float = 1.0,
    spartan_dense: bool = False,
    spartan_identity: bool = False,
) -> StateToStateModel:
    """Build the state-to-state model from plain config values (Hydra-friendly)."""
    return StateToStateModel(
        parameter_encoder=ParameterEncoder(
            state_dim=state_dim,
            dim=param_encoder_dim,
            num_heads=param_encoder_heads,
            max_history=max_history,
        ),
        predictor=Spartan(
            state_dim=state_dim,
            param_dim=1,
            num_slots=num_slots,
            num_layers=spartan_layers,
            embed_dim=spartan_embed_dim,
            mlp_hidden_size=spartan_mlp_hidden,
            mlp_num_layers=spartan_mlp_layers,
            temperature=spartan_temperature,
            dense=spartan_dense,
            identity=spartan_identity,
        ),
    )


__all__ = ["StateToStateModel", "TransitionOutput", "build_state_to_state"]
