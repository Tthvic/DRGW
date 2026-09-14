"""Fair-comparison invariants for the newly reconstructed autoencoder control."""

import pytest
import torch

from drgw.baseline import NaiveLatent
from drgw.model import DRGW


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(37)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup():
    upper = torch.triu((torch.rand(2, 8, 8) < 0.35).float(), diagonal=1)
    adj = upper + upper.transpose(-1, -2)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, -2:] = False
    adj *= mask.unsqueeze(-1) & mask.unsqueeze(-2)
    model = NaiveLatent(hidden_dim=16, watermark_dim=8, gin_layers=2, editor_chunk_size=3)
    return model, adj, mask


def test_baseline_is_one_head_without_an_inn_or_disentanglement():
    model, adj, mask = setup()
    assert model.model_kind == "naive_latent"
    assert not hasattr(model, "inn")
    assert not hasattr(model.encoder, "structural_head")
    assert not hasattr(model.encoder, "carrier_head")
    hs, hw = model.encode(adj, mask)
    assert hs is hw
    z, logdet = model.flow(hw, hs, adj, mask)
    assert z is hw
    assert not logdet.any()
    losses = model.encoder_losses(adj, mask, torch.zeros_like(adj))
    assert losses["invariance"] == 0
    assert losses["orthogonality"] == 0


@pytest.mark.parametrize("straight_through", [False, True])
def test_same_exact_budget_and_valid_graph_as_drgw(straight_through):
    model, adj, mask = setup()
    watermark = torch.randn(2, 8)
    result = model.embed(adj, mask, watermark, edit_budget=3, straight_through=straight_through)
    reference = DRGW(hidden_dim=16, watermark_dim=8, gin_layers=2, flow_layers=2)
    reference_result = reference.embed(adj, mask, watermark, edit_budget=3)
    assert torch.equal(result["edit_counts"], reference_result["edit_counts"])
    changed = result["adj"]
    assert torch.equal(changed, changed.transpose(-1, -2))
    assert torch.all((changed == 0) | (changed == 1))
    assert not changed.diagonal(dim1=-1, dim2=-2).any()
    assert not changed[~mask].any()
    assert ((changed != adj).sum((1, 2)) // 2).tolist() == [3, 3]
    torch.testing.assert_close(result["target_z_graph"], result["z_graph"] + 0.1 * watermark)
    assert torch.equal(model.embed(adj, mask, watermark)["adj"], adj)


def test_decoder_logits_choose_removal_or_addition_by_reconstruction_gain():
    model, adj, mask = setup()
    # Constant high edge probability makes insertion favorable and removal bad.
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


def test_single_head_detection_is_relabeling_invariant_and_padding_independent():
    model, adj, mask = setup()
    order = torch.randperm(adj.shape[-1])
    torch.testing.assert_close(
        model.latent(adj, mask), model.latent(adj[:, order][:, :, order], mask[:, order]), rtol=1e-5, atol=1e-6
    )
    padded_adj = torch.nn.functional.pad(adj, (0, 3, 0, 3))
    padded_mask = torch.nn.functional.pad(mask, (0, 3))
    torch.testing.assert_close(model.latent(adj, mask), model.latent(padded_adj, padded_mask), rtol=1e-5, atol=1e-6)


def test_reconstruction_only_objective_trains_encoder_and_decoder():
    model, adj, mask = setup()
    losses = model.encoder_losses(adj, mask)
    _, h = model.encode(adj, mask)
    objective = model.reconstruction_loss(adj, mask, h, h)
    objective = objective + losses["feature"] + losses["variance"] + 0.01 * model.latent_l2(h, mask)
    objective.backward()
    for module in (model.encoder, model.editor):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert model.encoder.feature_decoder.weight.grad.abs().sum() > 0


def test_empty_graph_and_budget_saturation():
    model, adj, mask = setup()
    watermark = torch.randn(2, 8)
    result = model.embed(adj, mask, watermark, edit_budget=10000)
    n = mask.sum(-1)
    assert torch.equal(result["edit_counts"], n * (n - 1) // 2)
    empty_mask = torch.zeros_like(mask)
    assert not model.latent(adj, empty_mask).any()
    empty = model.embed(adj, empty_mask, watermark, edit_budget=3)
    assert not empty["adj"].any()
    assert empty["edit_counts"].tolist() == [0, 0]


def test_invalid_budget_is_rejected():
    model, adj, mask = setup()
    with pytest.raises(ValueError, match="edit_budget"):
        model.embed(adj, mask, torch.randn(2, 8), edit_budget=0.5)
