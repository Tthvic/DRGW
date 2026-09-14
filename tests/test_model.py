"""Numerical and graph invariants required by blind watermark verification."""

import pytest
import torch

from drgw.model import DRGW, GraphAwareINN, node_features


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(17)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph_batch(nodes=9):
    upper = torch.triu((torch.rand(2, nodes, nodes) < 0.4).float(), diagonal=1)
    adj = upper + upper.transpose(-1, -2)
    mask = torch.ones(2, nodes, dtype=torch.bool)
    mask[1, -2:] = False
    adj = adj * (mask.unsqueeze(-1) & mask.unsqueeze(-2))
    return adj, mask


def small_model():
    return DRGW(hidden_dim=16, watermark_dim=8, gin_layers=2, flow_layers=2, editor_chunk_size=5)


def test_flow_exact_inverse_and_logdet_with_nonidentity_parameters():
    adj, mask = graph_batch()
    model = small_model().double()
    for layer in model.inn.layers:
        torch.nn.init.normal_(layer.conditioner.second.weight, std=0.04)
        torch.nn.init.normal_(layer.conditioner.second.bias, std=0.03)
    hs = torch.randn(2, adj.shape[1], 16, dtype=torch.double)
    hw = torch.randn_like(hs)  # Includes nonzero padded coordinates.
    z, logdet = model.flow(hw, hs, adj.double(), mask)
    restored, inverse_logdet = model.flow(z, hs, adj.double(), mask, reverse=True)
    torch.testing.assert_close(restored, hw, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), atol=1e-10, rtol=0)
    torch.testing.assert_close(z[~mask], hw[~mask], rtol=0, atol=0)
    assert logdet.abs().max() > 0.01


def test_flow_logdet_matches_full_jacobian_and_derivative():
    flow = GraphAwareINN(hidden_dim=4, layers=2, scale_limit=0.7).double()
    for layer in flow.layers:
        torch.nn.init.normal_(layer.conditioner.second.weight, std=0.1)
        torch.nn.init.normal_(layer.conditioner.second.bias, std=0.05)
    adj = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.double)
    mask = torch.ones(1, 2, dtype=torch.bool)
    hs = torch.randn(1, 2, 4, dtype=torch.double)
    x = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    _, analytic = flow(x, hs, adj, mask)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: flow(value.reshape_as(x), hs, adj, mask)[0].flatten(),
        x.flatten(),
        create_graph=True,
    )
    sign, numerical = torch.linalg.slogdet(jacobian)
    assert sign > 0
    torch.testing.assert_close(analytic.squeeze(), numerical, rtol=1e-9, atol=1e-9)
    analytical_gradient = torch.autograd.grad(analytic.sum(), x, retain_graph=True)[0]
    numerical_gradient = torch.autograd.grad(numerical, x)[0]
    torch.testing.assert_close(analytical_gradient, numerical_gradient, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("straight_through", [False, True])
def test_embedding_is_binary_symmetric_and_obeys_exact_budget(straight_through):
    adj, mask = graph_batch()
    model = small_model()
    w = torch.randn(2, 8)
    output = model.embed(adj, mask, w, edit_budget=torch.tensor([2, 4]), straight_through=straight_through)
    edited = output["adj"]
    assert torch.all((edited == 0) | (edited == 1))
    assert torch.equal(edited, edited.transpose(-1, -2))
    assert not edited.diagonal(dim1=-2, dim2=-1).any()
    assert not edited[~mask].any()
    actual = (edited != adj).sum(dim=(1, 2)) // 2
    assert actual.tolist() == [2, 4]
    assert torch.equal(actual, output["edit_counts"])
    torch.testing.assert_close(output["target_z_graph"], output["z_graph"] + 0.1 * w)
    target_z, _ = model.flow(output["target_hw"], output["hs"], adj, mask)
    torch.testing.assert_close(model.pooled(target_z, mask)[..., :8], output["target_z_graph"])


def test_default_floor_budget_does_not_force_edits_on_small_graphs():
    adj, mask = graph_batch()
    output = small_model().embed(adj, mask, torch.randn(2, 8), budget_ratio=0.001)
    assert output["requested_budget"].tolist() == [0, 0]
    assert output["edit_counts"].tolist() == [0, 0]
    assert torch.equal(output["adj"], adj)


def test_budget_ratio_counts_undirected_edges_once_and_caps_by_eligible_pairs():
    adj, mask = graph_batch()
    model = small_model()
    output = model.embed(adj, mask, torch.randn(2, 8), budget_ratio=0.4)
    expected = torch.floor(adj.sum((1, 2)) / 2 * 0.4).long()
    assert torch.equal(output["edit_counts"], expected)
    saturated = model.embed(adj, mask, torch.randn(2, 8), edit_budget=10000)
    n = mask.sum(-1)
    assert torch.equal(saturated["edit_counts"], n * (n - 1) // 2)
    assert saturated["requested_budget"].tolist() == [10000, 10000]


def test_detection_and_features_are_invariant_to_node_relabeling():
    adj, mask = graph_batch()
    model = small_model().eval()
    order = torch.randperm(adj.shape[-1])
    relabeled = adj[:, order][:, :, order]
    moved_mask = mask[:, order]
    torch.testing.assert_close(
        node_features(relabeled, moved_mask), node_features(adj, mask)[:, order], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(model.latent(adj, mask), model.latent(relabeled, moved_mask), rtol=1e-5, atol=1e-6)
    hs, hw = model.encode(adj, mask)
    rhs, rhw = model.encode(relabeled, moved_mask)
    torch.testing.assert_close(rhs, hs[:, order], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rhw, hw[:, order], rtol=1e-5, atol=1e-6)


def test_padding_and_empty_graph_are_ignored():
    adj, mask = graph_batch()
    model = small_model().eval()
    padded = torch.nn.functional.pad(adj, (0, 4, 0, 4))
    padded_mask = torch.nn.functional.pad(mask, (0, 4))
    # Even arbitrary values touching padding cannot change current graph features.
    padded[:, -4:, :] = 1
    padded[:, :, -4:] = 1
    torch.testing.assert_close(model.latent(adj, mask), model.latent(padded, padded_mask), rtol=1e-5, atol=1e-6)
    empty = torch.zeros_like(mask)
    assert torch.equal(model.latent(adj, empty), torch.zeros(2, 8))
    output = model.embed(adj, empty, torch.randn(2, 8), edit_budget=2)
    assert output["edit_counts"].tolist() == [0, 0]
    assert not output["adj"].any()


def test_straight_through_editing_trains_editor_and_watermark_channel():
    adj, mask = graph_batch()
    model = small_model()
    w = torch.randn(2, 8, requires_grad=True)
    output = model.embed(adj, mask, w, edit_budget=2, straight_through=True, all_pairs=True)
    detected = model.latent(output["adj"], mask)
    loss = -(detected * w).sum(-1).mean()
    loss.backward()
    for module in (model.encoder, model.inn, model.editor):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_editor_scores_do_not_depend_on_endpoint_order():
    model = small_model()
    hs, hw = torch.randn(7, 16), torch.randn(7, 16)
    pairs = torch.tensor([[0, 2, 5], [1, 6, 3]])
    torch.testing.assert_close(model.editor.score_pairs(hs, hw, pairs), model.editor.score_pairs(hs, hw, pairs.flip(0)))


def test_decoder_edge_logits_are_converted_to_directional_flip_utility():
    adj, mask = graph_batch()
    model = small_model()
    with torch.no_grad():
        for parameter in model.editor.parameters():
            parameter.zero_()
        model.editor.mlp[-1].bias.fill_(2)
    inserted = model.embed(adj, mask, torch.randn(2, 8), edit_budget=2, all_pairs=True)
    assert ((inserted["adj"] - adj).sum((1, 2)) / 2).tolist() == [2, 2]
    with torch.no_grad():
        model.editor.mlp[-1].bias.fill_(-2)
    deleted = model.embed(adj, mask, torch.randn(2, 8), edit_budget=2, all_pairs=True)
    assert ((deleted["adj"] - adj).sum((1, 2)) / 2).tolist() == [-2, -2]


def test_auxiliary_objectives_and_frozen_encoder_reconstruction():
    adj, mask = graph_batch()
    model = small_model()
    losses = model.encoder_losses(adj, mask, adj)
    assert set(losses) == {"invariance", "orthogonality", "feature", "variance"}
    assert losses["invariance"] == 0
    sum(losses.values()).backward()
    assert model.encoder.carrier_decoder.weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.encoder.requires_grad_(False)
    reconstruction = model.reconstruction_loss(adj, mask)
    reconstruction.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.editor.mlp[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("budget", [-1, 0.5])
def test_invalid_explicit_budgets_raise(budget):
    adj, mask = graph_batch()
    with pytest.raises(ValueError, match="edit_budget"):
        small_model().embed(adj, mask, torch.randn(2, 8), edit_budget=budget)
