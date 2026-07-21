"""Invariant tests for the SPARTAN predictor (hard attention, path matrix, sparsity)."""

from typing import cast

import pytest
import torch

from scjepa.models import Spartan
from scjepa.models.spartan import SpartanLayer

B, N, D = 2, 3, 8
T = 2 * N  # state + parameter tokens


@pytest.fixture
def model() -> Spartan:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16)


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, N, D), torch.randn(B, N, D)


def test_output_shapes(model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    out = model(*inputs)
    assert out.prediction.shape == (B, N, D)
    assert out.path_matrix.shape == (B, T, T)
    assert out.sparsity.shape == ()
    assert torch.isfinite(out.prediction).all()


def test_path_matrix_is_integer_valued(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Hard {0,1} adjacencies (Eq. 3) ⇒ Ā counts paths ⇒ integer entries."""
    for train in (True, False):
        model.train(train)
        path = model(*inputs).path_matrix
        torch.testing.assert_close(path, path.round(), atol=1e-4, rtol=0)
        assert (path >= 0).all()
        # Residual identity (Eq. 5): every token has at least the self path.
        diagonal = path.diagonal(dim1=1, dim2=2)
        assert (diagonal >= 1 - 1e-4).all()


def test_sparsity_equals_decoded_state_row_path_sum(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Only paths ending at decoded state rows belong in the objective."""
    out = model(*inputs)
    torch.testing.assert_close(out.sparsity, out.path_matrix[:, :N].sum(dim=(1, 2)).mean())


def test_identity_sparsity_counts_only_state_residuals(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Undecoded parameter residuals must not add a constant sparsity charge."""
    model = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16, identity=True)
    out = model(*inputs)
    torch.testing.assert_close(out.sparsity, torch.tensor(float(N)))


def test_sparsity_penalty_has_gradients(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Straight-through sampling must let |Ā| gradients reach the q/k projections."""
    model.train()
    model(*inputs).sparsity.backward()  # pyright: ignore[reportUnknownMemberType]
    for layer_index, module in enumerate(model.layers):
        layer = cast(SpartanLayer, module)
        for projection in (layer.project_q, layer.project_k):
            assert projection.weight.grad is not None, f"layer {layer_index}: no grad"
            assert projection.weight.grad.abs().sum() > 0, f"layer {layer_index}: zero grad"


def test_mask_blocks_information_flow(model: Spartan) -> None:
    """THE causal claim: Ā_ij = 0 ⇒ ∂ prediction_i / ∂ token_j = 0 (eval mode)."""
    model.eval()
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    state = torch.randn(1, N, D, requires_grad=True)
    params = torch.randn(1, N, D, requires_grad=True)
    out = model(state, params)
    path = out.path_matrix[0]  # (T, T)

    checked_zero = checked_nonzero = 0
    for i in range(N):  # prediction rows = state-token positions
        grads = torch.autograd.grad(out.prediction[0, i].sum(), (state, params), retain_graph=True)
        token_grads = torch.cat([grads[0][0], grads[1][0]], dim=0)  # (T, d)
        for j in range(T):
            if path[i, j] < 0.5:
                assert token_grads[j].abs().max() < 1e-6, f"leak {j} → prediction {i}"
                checked_zero += 1
            elif token_grads[j].abs().max() > 0:
                checked_nonzero += 1
    assert checked_nonzero > 0  # dependence actually flows where paths exist
    if checked_zero == 0:
        pytest.skip("random init produced a fully connected graph; nothing to check")


def test_slot_permutation_equivariance(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Permuting slots (same permutation for S_t and Ŝ^ph) permutes predictions."""
    model.eval()
    state, params = inputs
    perm = torch.randperm(N)
    with torch.no_grad():
        base = model(state, params).prediction
        permuted = model(state[:, perm], params[:, perm]).prediction
    torch.testing.assert_close(permuted, base[:, perm])


def test_independent_node_embeddings_expose_coordinates(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Node addresses make reassignment visible without specifying i ← i."""
    state, params = inputs
    reassignment = torch.tensor([1, 2, 0])

    torch.manual_seed(14)  # pyright: ignore[reportUnknownMemberType]
    anonymous = Spartan(
        slot_size=D,
        num_layers=1,
        embed_dim=None,
        mlp_hidden_size=16,
        dense=True,
    ).eval()
    torch.manual_seed(14)  # pyright: ignore[reportUnknownMemberType]
    addressed = Spartan(
        slot_size=D,
        num_layers=1,
        embed_dim=None,
        mlp_hidden_size=16,
        dense=True,
        node_embeddings=True,
        num_slots=N,
    ).eval()

    with torch.no_grad():
        anonymous_base = anonymous(state, params).prediction
        anonymous_reassigned = anonymous(state, params[:, reassignment]).prediction
        addressed_base = addressed(state, params).prediction
        addressed_reassigned = addressed(state, params[:, reassignment]).prediction
    torch.testing.assert_close(anonymous_reassigned, anonymous_base)
    assert not torch.allclose(addressed_reassigned, addressed_base)
    assert addressed.state_node_embed is not None
    assert addressed.param_node_embed is not None
    assert not torch.equal(addressed.state_node_embed, addressed.param_node_embed)


def test_node_embeddings_validate_fixed_slot_count(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    with pytest.raises(ValueError, match="positive num_slots"):
        Spartan(
            slot_size=D,
            embed_dim=None,
            mlp_hidden_size=16,
            node_embeddings=True,
        )
    addressed = Spartan(
        slot_size=D,
        embed_dim=None,
        mlp_hidden_size=16,
        node_embeddings=True,
        num_slots=N + 1,
    )
    with pytest.raises(ValueError, match="configured num_slots"):
        addressed(*inputs)


def test_auxiliary_tokens() -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    aux_model = Spartan(slot_size=D, num_layers=1, embed_dim=None, mlp_hidden_size=16, aux_dim=4)
    state, params = torch.randn(B, N, D), torch.randn(B, N, D)
    aux = torch.randn(B, 2, 4)
    out = aux_model(state, params, aux)
    assert out.prediction.shape == (B, N, D)
    assert out.path_matrix.shape == (B, T + 2, T + 2)
    # Aux is optional even when the pathway exists (CLEVRER-style usage).
    assert aux_model(state, params).path_matrix.shape == (B, T, T)


def test_embed_dim_projection() -> None:
    """App. A.1 separates token dim from embedding dim: d -> e -> d round trip."""
    torch.manual_seed(4)  # pyright: ignore[reportUnknownMemberType]
    model = Spartan(slot_size=D, num_layers=1, embed_dim=32, mlp_hidden_size=16)
    out = model(torch.randn(B, N, D), torch.randn(B, N, D))
    assert out.prediction.shape == (B, N, D)  # back in slot space
    assert out.path_matrix.shape == (B, T, T)
    out.prediction.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert not isinstance(model.in_project, torch.nn.Identity)
    assert not isinstance(model.out_project, torch.nn.Identity)


def test_input_guards(model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    state, params = inputs
    with pytest.raises(ValueError, match="params"):
        model(state, torch.randn(B, N + 1, D))
    with pytest.raises(ValueError, match="aux_dim=None"):
        model(state, params, aux=torch.randn(B, 2, 4))


def test_param_size_split_token_dims() -> None:
    """D20: state tokens k-dim, param tokens d-dim, predictions back in k-dim."""
    torch.manual_seed(5)  # pyright: ignore[reportUnknownMemberType]
    model = Spartan(slot_size=4, num_layers=1, embed_dim=32, mlp_hidden_size=16, param_size=D)
    state, params = torch.randn(B, N, 4), torch.randn(B, N, D)
    out = model(state, params)
    assert out.prediction.shape == (B, N, 4)
    assert out.path_matrix.shape == (B, T, T)
    out.prediction.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert model.param_project is not None
    with pytest.raises(ValueError, match="params"):
        model(state, torch.randn(B, N, 4))  # params must be param_size-dim
    # param_size=None keeps the shared projection (pre-D20 state_dict intact).
    shared = Spartan(slot_size=D, num_layers=1, embed_dim=None, mlp_hidden_size=16)
    assert shared.param_project is None
    assert not any("param_project" in k for k in shared.state_dict())


def test_logit_penalty_finite_and_grows_with_logits(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Eq. 11 (Baumgartner): larger attention logits ⇒ larger penalty.

    The penalty must be finite, at least 2 (exp(x) + exp(-x) >= 2), and grow
    when logits are inflated — it exists to keep the softmax gradient alive.
    """
    model.eval()
    out = model(*inputs)
    assert torch.isfinite(out.logit_penalty)
    assert out.logit_penalty >= 2.0 - 1e-4  # exp(x) + exp(-x) >= 2
    layer = cast(SpartanLayer, model.layers[0])
    with torch.no_grad():
        layer.project_q.weight.mul_(3.0)
    inflated = model(*inputs)
    assert inflated.logit_penalty > out.logit_penalty


def test_logit_diagnostics_are_finite_and_bounded(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    out = model(*inputs)
    assert torch.isfinite(out.mean_abs_logit)
    assert torch.isfinite(out.mean_gate_probability)
    assert torch.isfinite(out.gate_entropy)
    assert out.mean_abs_logit >= 0
    assert 0 <= out.mean_gate_probability <= 1
    assert 0 <= out.gate_entropy <= torch.log(torch.tensor(2.0)) + 1e-6


def test_dense_mode_is_fully_connected_and_deterministic(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A≡1 (audit F-8): the tau-calibration reference must be a TRUE dense
    transformer — every path open, no Bernoulli sampling noise in train mode."""
    torch.manual_seed(0)
    dense = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16, dense=True)
    dense.train()
    first = dense(*inputs)
    second = dense(*inputs)
    torch.testing.assert_close(first.prediction, second.prediction)  # no sampling
    assert (first.path_matrix >= 1).all()  # every token reaches every prediction
    torch.manual_seed(0)
    gated = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16)
    gated.train()
    a = gated(*inputs)
    b = gated(*inputs)
    assert not torch.allclose(a.prediction, b.prediction)  # gated model DOES sample


def test_identity_mode_is_mass_blind_and_deterministic(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A≡0 (D16 go/no-go): the mass-blind reference must have NO cross-token
    flow — path matrix exactly I, predictions insensitive to param tokens and
    to every other slot — and no Bernoulli sampling noise in train mode."""
    torch.manual_seed(0)
    model = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16, identity=True)
    model.train()
    state, params = inputs
    first = model(state, params)
    second = model(state, params)
    torch.testing.assert_close(first.prediction, second.prediction)  # no sampling
    eye = torch.eye(2 * N).expand(B, -1, -1)
    torch.testing.assert_close(first.path_matrix, eye)  # residual paths only
    # Mass-blind: perturbing the param tokens must not move any prediction.
    blind = model(state, torch.randn_like(params))
    torch.testing.assert_close(first.prediction, blind.prediction)
    # Slot-local: perturbing slot j must leave every other slot's prediction unchanged.
    poked_state = state.clone()
    poked_state[:, 0] += 1.0
    poked = model(poked_state, params)
    torch.testing.assert_close(first.prediction[:, 1:], poked.prediction[:, 1:])
    assert not torch.allclose(first.prediction[:, 0], poked.prediction[:, 0])


def test_dense_identity_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Spartan(slot_size=D, embed_dim=None, mlp_hidden_size=16, dense=True, identity=True)


def test_gate_noise_reuse_is_deterministic(
    model: Spartan, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """D19: with a fixed noise draw the gated forward is deterministic — a
    chain reusing one draw cannot flicker at unchanged state — while the
    default (no noise passed) still samples freshly per call."""
    model.train()
    noise = model.sample_gate_noise(*inputs)
    assert noise is not None and len(noise) == 2
    assert noise[0].shape == (B, T, T)
    first = model(*inputs, gate_noise=noise)
    second = model(*inputs, gate_noise=noise)
    torch.testing.assert_close(first.prediction, second.prediction)
    torch.testing.assert_close(first.path_matrix, second.path_matrix)
    fresh_a = model(*inputs)
    fresh_b = model(*inputs)
    assert not torch.allclose(fresh_a.prediction, fresh_b.prediction)
    with pytest.raises(ValueError, match="gate_noise"):
        model(*inputs, gate_noise=noise[:1])


def test_gate_noise_none_for_dense_and_identity(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Dense (A≡1) and identity (A≡0) modes sample nothing — no noise to draw."""
    for kwargs in ({"dense": True}, {"identity": True}):
        ref = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16, **kwargs)
        assert ref.sample_gate_noise(*inputs) is None


def test_rollout_draws_gate_noise_once_per_chain(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """D19: a Tp=4 chain consumes exactly as much RNG as a Tp=1 chain — the
    thresholds are drawn once per chain, never per step."""
    from scjepa.models.jepa import rollout_predictions

    state, params = inputs
    anchors = state.unsqueeze(1)  # (B, 1 chain, N, D)
    torch.manual_seed(7)
    predictor = Spartan(slot_size=D, num_layers=2, embed_dim=None, mlp_hidden_size=16)
    predictor.train()
    torch.manual_seed(123)
    rollout_predictions(predictor, anchors, params, None, 1)
    rng_after_short = torch.get_rng_state()
    torch.manual_seed(123)
    rollout_predictions(predictor, anchors, params, None, 4)
    rng_after_long = torch.get_rng_state()
    assert torch.equal(rng_after_short, rng_after_long)
