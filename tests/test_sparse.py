"""Dense/CSR numerical equivalence and whole-graph editing invariants."""

import numpy as np
import pytest
from scipy import sparse as sp
import torch

from drgw.baseline import NaiveLatent
from drgw.model import DRGW, node_features
from drgw.sparse import (SparseGraph, candidate_pairs_sparse, embed_sparse,
                         encode_sparse, flow_sparse, latent_sparse,
                         node_features_sparse, sample_nonedges)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(71)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(kind="drgw"):
    upper = torch.triu((torch.rand(11, 11) < 0.3).double(), diagonal=1)
    adj = upper + upper.T
    mask = torch.ones((1, len(adj)), dtype=torch.bool)
    if kind == "drgw":
        model = DRGW(hidden_dim=16, watermark_dim=8, gin_layers=2, flow_layers=3).double().eval()
        for layer in model.inn.layers:
            torch.nn.init.normal_(layer.conditioner.second.weight, std=0.03)
            torch.nn.init.normal_(layer.conditioner.second.bias, std=0.02)
    else:
        model = NaiveLatent(hidden_dim=16, watermark_dim=8, gin_layers=2).double().eval()
    graph = SparseGraph.from_scipy(sp.csr_matrix(adj.numpy()), dtype=torch.float64)
    return model, adj, mask, graph


@pytest.mark.parametrize("kind", ["drgw", "naive"])
def test_sparse_features_encoder_and_latent_match_dense(kind):
    model, adj, mask, graph = setup(kind)
    torch.testing.assert_close(node_features_sparse(graph), node_features(adj.unsqueeze(0), mask)[0], rtol=1e-12, atol=1e-12)
    dense_hs, dense_hw = model.encode(adj.unsqueeze(0), mask)
    hs, hw = encode_sparse(model, graph, node_chunk_size=3)
    torch.testing.assert_close(hs, dense_hs[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(hw, dense_hw[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(latent_sparse(model, graph, node_chunk_size=3), model.latent(adj.unsqueeze(0), mask)[0], rtol=1e-10, atol=1e-10)
    assert not hs.requires_grad and not hw.requires_grad


def test_sparse_flow_and_inverse_match_nonidentity_dense_flow():
    model, adj, mask, graph = setup()
    hs, hw = model.encode(adj.unsqueeze(0), mask)
    dense_z, dense_logdet = model.flow(hw, hs, adj.unsqueeze(0), mask)
    z, logdet = flow_sparse(model, hw[0], hs[0], graph, node_chunk_size=4)
    torch.testing.assert_close(z, dense_z[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logdet, dense_logdet[0], rtol=1e-10, atol=1e-10)
    recovered, inverse_logdet = flow_sparse(model, z, hs[0], graph, reverse=True, node_chunk_size=4)
    torch.testing.assert_close(recovered, hw[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), rtol=0, atol=1e-10)


@pytest.mark.parametrize("kind", ["drgw", "naive"])
def test_sparse_topk_is_global_and_embedding_obeys_binary_budget(kind):
    model, adj, mask, graph = setup(kind)
    w = torch.randn(8, dtype=torch.double)
    result = embed_sparse(model, graph, w, edit_budget=4, candidate_seed=17, chunk_size=3,
                          node_chunk_size=4, return_latents=True)
    assert result["edit_counts"] == result["requested_budget"] == 4
    assert (result["adj"] != graph.adj).nnz // 2 == 4
    assert result["adj"].dtype == np.uint8 and np.all(result["adj"].data == 1)
    assert not result["adj"].diagonal().any() and (result["adj"] != result["adj"].T).nnz == 0
    np.testing.assert_array_equal(graph.adj.toarray(), adj.numpy())
    torch.testing.assert_close(result["target_z_graph"], result["z_graph"] + 0.1 * w)
    target, _ = flow_sparse(model, result["target_hw"], result["hs"], graph, node_chunk_size=3)
    torch.testing.assert_close(target.mean(0)[:8], result["target_z_graph"], rtol=1e-10, atol=1e-10)
    pairs = candidate_pairs_sparse(graph, model.nonedge_ratio, minimum=4, candidate_seed=17)
    with torch.inference_mode():
        logits = model.editor.score_pairs(result["hs"], result["target_hw"], torch.from_numpy(pairs))
        signs = 1 - 2 * torch.as_tensor(np.asarray(graph.adj[pairs[0], pairs[1]]).ravel(), dtype=torch.double)
        best = (logits * signs).topk(4).values
    np.testing.assert_allclose(np.sort(result["edit_scores"]), np.sort(best.numpy()), rtol=1e-10, atol=1e-10)
    roundtrip = latent_sparse(model, result["graph"], node_chunk_size=4)
    dense_edited = torch.from_numpy(result["adj"].toarray()).double().unsqueeze(0)
    torch.testing.assert_close(roundtrip, model.latent(dense_edited, mask)[0], rtol=1e-10, atol=1e-10)


def test_sparse_zero_budget_does_not_force_an_edit_and_saturation_is_recorded():
    model, _, _, graph = setup()
    w = torch.randn(8, dtype=torch.double)
    zero = embed_sparse(model, graph, w)
    assert zero["graph"] is graph
    assert zero["edit_counts"] == zero["requested_budget"] == 0
    assert zero["edit_pairs"].shape == (0, 2)
    assert "hs" not in zero  # Large node states are released by default.
    full = embed_sparse(model, graph, w, edit_budget=10000, chunk_size=7)
    assert full["edit_counts"] == len(graph.degree) * (len(graph.degree) - 1) // 2
    assert full["requested_budget"] == 10000


def test_sparse_detection_remains_invariant_to_relabeling():
    model, adj, _, graph = setup()
    order = torch.randperm(len(adj))
    moved = SparseGraph.from_scipy(sp.csr_matrix(adj[order][:, order].numpy()), dtype=torch.float64)
    torch.testing.assert_close(latent_sparse(model, graph), latent_sparse(model, moved), rtol=1e-10, atol=1e-10)


def test_nonedges_are_unique_absent_and_seeded_in_sparse_and_dense_regimes():
    for probability in (0.05, 0.95):
        rng = np.random.default_rng(101)
        upper = np.triu((rng.random((40, 40)) < probability).astype(np.uint8), k=1)
        adj = sp.csr_matrix(upper + upper.T)
        count = min(40, 40 * 39 // 2 - adj.nnz // 2)
        sampled = sample_nonedges(adj, count, np.random.default_rng(11))
        repeated = sample_nonedges(adj, count, np.random.default_rng(11))
        assert sampled.shape == (2, count)
        assert (sampled[0] < sampled[1]).all()
        assert len(np.unique(sampled[0] * 40 + sampled[1])) == count
        assert not np.asarray(adj[sampled[0], sampled[1]]).any()
        np.testing.assert_array_equal(sampled, repeated)


def test_sparse_path_handles_thousands_of_nodes_without_dense_adjacency(monkeypatch):
    n = 5000
    u = np.arange(n)
    v = (u + 1) % n
    adj = sp.coo_matrix((np.ones(2*n), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    graph = SparseGraph.from_scipy(adj)
    model = DRGW(hidden_dim=8, watermark_dim=4, gin_layers=1, flow_layers=1).eval()

    def forbidden_dense(*args, **kwargs):
        raise AssertionError("Full graph inference must not create a dense adjacency")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden_dense)
    monkeypatch.setattr(np, "triu_indices", forbidden_dense)
    result = embed_sparse(model, graph, torch.ones(4), budget_ratio=0.001, chunk_size=257, node_chunk_size=311)
    assert result["edit_counts"] == 5
    assert (result["adj"] != graph.adj).nnz == 10
    assert torch.isfinite(latent_sparse(model, result["graph"], node_chunk_size=311)).all()


def test_sparse_empty_graph_and_invalid_inputs():
    graph = SparseGraph.from_scipy(sp.csr_matrix((0, 0)))
    model = DRGW(hidden_dim=8, watermark_dim=4, gin_layers=1, flow_layers=1)
    assert torch.equal(latent_sparse(model, graph), torch.zeros(4))
    assert embed_sparse(model, graph, torch.ones(4), edit_budget=2)["edit_counts"] == 0
    with pytest.raises(ValueError, match="undirected"):
        SparseGraph.from_scipy(sp.csr_matrix([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match="binary"):
        SparseGraph.from_scipy(sp.csr_matrix([[0, 2], [2, 0]]))
    with pytest.raises(ValueError, match="edit_budget"):
        embed_sparse(model, graph, torch.ones(4), edit_budget=0.5)
    with pytest.raises(ValueError, match="4-dimensional"):
        embed_sparse(model, graph, torch.ones(5))
