"""Tests for the SPARTAN sparse transition predictor (experiments.pdf Eqs. 27-37)."""

import pytest
import torch

from scjepa.models import Spartan

N, K, DIM = 3, 4, 32


def tiny(dense: bool = False, identity: bool = False, layers: int = 2) -> Spartan:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return Spartan(
        state_dim=K,
        param_dim=1,
        num_slots=N,
        num_layers=layers,
        embed_dim=DIM,
        mlp_hidden_size=DIM,
        mlp_num_layers=2,
        dense=dense,
        identity=identity,
    )


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(2, N, K), torch.randn(2, N, 1)


def test_output_shapes(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    out = tiny()(*inputs)
    assert out.prediction.shape == (2, N, K)  # Eq. 37: decoded into R^4
    assert out.path_matrix.shape == (2, 2 * N, 2 * N)  # M = 2N tokens
    assert out.sparsity.ndim == 0


def test_path_matrix_is_integer_valued(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Eq. 10: entries of (A^L+I)...(A^1+I) are path COUNTS."""
    out = tiny()(*inputs)
    torch.testing.assert_close(out.path_matrix, out.path_matrix.round())
    assert (out.path_matrix.diagonal(dim1=-2, dim2=-1) >= 1).all()  # residual self-paths


def test_sparsity_covers_only_decoded_state_rows(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Eq. 11: the path objective sums the N decoded rows only."""
    out = tiny()(*inputs)
    expected = out.path_matrix[:, :N].sum(dim=(1, 2)).mean()
    torch.testing.assert_close(out.sparsity, expected)


def test_identity_mode_is_token_local(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A≡0: path matrix exactly I — only each token's residual-MLP path remains."""
    model = tiny(identity=True).eval()
    out = model(*inputs)
    eye = torch.eye(2 * N).expand(2, -1, -1)
    torch.testing.assert_close(out.path_matrix, eye)
    # Parameter values cannot influence predictions.
    state, params = inputs
    out_other = model(state, params + 5.0)
    torch.testing.assert_close(out.prediction, out_other.prediction)


def test_dense_mode_is_fully_connected_and_deterministic(
    inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A≡1 with no gate sampling: rho_path = 1, repeat calls agree in train mode."""
    model = tiny(dense=True)
    out1, out2 = model(*inputs), model(*inputs)
    torch.testing.assert_close(out1.prediction, out2.prediction)
    assert bool((out1.path_matrix >= 1).all())


def test_paper_stated_path_objective_endpoints() -> None:
    """§6.1.3: with ten tokens, dense L_path = 6655 and token-local = 5."""
    state, params = torch.randn(1, 5, K), torch.randn(1, 5, 1)
    dense = Spartan(num_slots=5, num_layers=3, embed_dim=DIM, mlp_hidden_size=DIM, dense=True)
    assert dense(state, params).sparsity.item() == 6655.0
    local = Spartan(num_slots=5, num_layers=3, embed_dim=DIM, mlp_hidden_size=DIM, identity=True)
    assert local(state, params).sparsity.item() == 5.0


def test_eval_gates_are_deterministic(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Eq. 34: eval-mode adjacency is the noiseless threshold 1{g > 0}."""
    model = tiny().eval()
    out1, out2 = model(*inputs), model(*inputs)
    torch.testing.assert_close(out1.prediction, out2.prediction)
    torch.testing.assert_close(out1.path_matrix, out2.path_matrix)


def test_train_gates_resample_fresh_noise(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Eq. 33: every training forward draws fresh Gumbel noise."""
    model = tiny()
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    out1 = model(*inputs)
    out2 = model(*inputs)
    assert not torch.equal(out1.path_matrix, out2.path_matrix)


def test_track_keys_fixed_shared_and_distinct() -> None:
    """Eq. 27: one fixed non-trainable codebook shared by all three modes."""
    sparse, dense, identity = tiny(), tiny(dense=True), tiny(identity=True)
    keys = sparse.track_keys
    assert keys.shape == (1, N, DIM)
    assert not keys.requires_grad
    assert all("track_keys" not in name for name, _ in sparse.named_parameters())
    dists = torch.cdist(keys[0], keys[0]) + torch.eye(N)
    assert bool((dists > 0).all())  # rows pairwise distinct
    torch.testing.assert_close(dense.track_keys, keys)
    torch.testing.assert_close(identity.track_keys, keys)


def test_track_keys_give_tokens_addresses(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Identical-content rows get different outputs: kappa_i is an address."""
    model = tiny(dense=True).eval()
    state, params = inputs
    state = state.clone()
    state[:, 1] = state[:, 0]  # rows 0 and 1 identical in content
    params = params.clone()
    params[:, 1] = params[:, 0]
    out = model(state, params)
    # If keys carried no information the two rows would be exchangeable and
    # their predictions identical; the distinct kappa breaks the tie.
    assert not torch.allclose(out.prediction[:, 0], out.prediction[:, 1])


def test_gates_block_information_flow() -> None:
    """A masked edge admits no information (Eq. 35 masks BEFORE normalizing)."""
    model = tiny(identity=True).eval()  # extreme case: everything masked
    state = torch.randn(1, N, K)
    params = torch.randn(1, N, 1)
    out1 = model(state, params)
    state2 = state.clone()
    state2[:, 2] += 10.0  # perturb another token
    out2 = model(state2, params)
    torch.testing.assert_close(out1.prediction[:, 0], out2.prediction[:, 0])


def test_logit_penalty_minimum_and_growth(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """exp(a)+exp(-a): minimum 2 at a=0, grows with |a|, finite at extremes."""
    out = tiny()(*inputs)
    assert out.logit_penalty.item() >= 2.0
    assert torch.isfinite(out.logit_penalty)
    # The linear continuation keeps huge logits finite.
    model = tiny(dense=True)
    with torch.no_grad():
        model.state_project.weight.mul_(1e4)
    big = model(*inputs)
    assert torch.isfinite(big.logit_penalty)
    assert big.logit_penalty > out.logit_penalty


def test_diagnostics_bounded(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    out = tiny()(*inputs)
    assert 0 <= out.mean_gate_probability.item() <= 1
    assert 0 <= out.gate_entropy.item() <= 0.6932
    assert out.mean_abs_logit.item() >= 0


def test_gradients_reach_gates_and_params(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Straight-through (Eq. 33): prediction + path losses reach q/k and theta."""
    model = tiny()
    state, params = inputs
    params = params.clone().requires_grad_(True)
    out = model(state, params)
    (out.prediction.square().mean() + 1e-6 * out.sparsity).backward()  # pyright: ignore[reportUnknownMemberType]
    assert params.grad is not None
    q_grad = model.layers[0].project_q.weight.grad
    assert q_grad is not None
    assert q_grad.abs().sum() > 0


def test_input_guards(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    model = tiny()
    state, params = inputs
    with pytest.raises(ValueError, match="expected state"):
        model(state[:, :2], params)
    with pytest.raises(ValueError, match="expected state"):
        model(state, params.squeeze(-1))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Spartan(num_slots=N, dense=True, identity=True)
