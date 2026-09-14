import gzip
import json

import numpy as np
import pytest
from scipy import sparse
import torch

from drgw.attacks import edge_flip, node_delete, node_permute
from drgw.data import DATASETS, load_dataset, prepare_banks, sample_subgraphs, sample_subgraphs_with_metadata, split_node_pools
from drgw.metrics import (auroc, calibrate_threshold, degree_assortativity,
                          detection_metrics, dk2_emd, edge_flips, global_transitivity,
                          link_prediction_auc, make_link_prediction_split,
                          node_embedding_cosine, structural_fidelity)


def cycle(n):
    graph = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        graph[i, (i + 1) % n] = graph[(i + 1) % n, i] = 1
    return graph


def test_preprocess_deduplicates_directed_edges_and_retains_provenance(tmp_path):
    path = tmp_path / "raw" / "Facebook" / "facebook_combined.txt.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as handle:
        handle.write("# test edge list\n10 20\n20 10\n20 30\n30 30\n")
    graph, metadata = load_dataset("Facebook", tmp_path, download=False)
    assert graph.shape == (3, 3)
    assert graph.nnz == 4
    assert metadata["raw_edge_rows"] == 4
    assert metadata["raw_self_loop_rows"] == 1
    assert len(metadata["raw_sha256"]) == 64
    assert len(metadata["processed_sha256"]) == 64
    cached, cached_metadata = load_dataset("Facebook", tmp_path, download=False)
    assert (cached != graph).nnz == 0
    assert metadata == cached_metadata
    graph_path = tmp_path / "processed" / "Facebook" / "graph.npz"
    with graph_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        load_dataset("Facebook", tmp_path, download=False)


def test_named_dataset_never_falls_back_to_synthetic(tmp_path):
    assert len(DATASETS) == 18
    with pytest.raises(ValueError, match="unresolved"):
        load_dataset("Patents", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset("Facebook", tmp_path, download=False)


def test_banks_cache_uint8_and_auditable_node_ids(tmp_path):
    raw = tmp_path / "raw" / "Facebook" / "facebook_combined.txt.gz"
    raw.parent.mkdir(parents=True)
    with gzip.open(raw, "wt") as handle:
        for index in range(100):
            handle.write(f"{index} {(index+1)%100}\n")
    config = dict(data_dir=str(tmp_path), datasets=["Facebook"], seed=42,
                  download=False, splits={"train": dict(num_nodes=10, num_graphs=3),
                                          "test": dict(num_nodes=10, num_graphs=2)})
    paths = prepare_banks(config)
    assert prepare_banks({"data": config}) == paths
    with np.load(paths["Facebook"]["train"]) as bank:
        assert bank["adj"].dtype == np.uint8
        assert bank["adj"].shape == (3, 10, 10)
        train_ids = bank["node_ids"]
    with np.load(paths["Facebook"]["test"]) as bank:
        assert not set(bank["node_ids"].ravel()) & set(train_ids.ravel())
    metadata = json.loads(paths["Facebook"]["train"].with_suffix(".json").read_text())
    assert len(metadata["split_pool_sha256"]) == 64
    assert len(metadata["bank_sha256"]) == 64


def test_split_pools_are_disjoint_even_when_sampling_seeds_differ():
    graph = sparse.csr_matrix(cycle(200))
    pools = split_node_pools(graph)
    assert len(set(np.concatenate(list(pools.values())))) == 200
    train, tr_meta = sample_subgraphs_with_metadata(graph, "train", 7, 5, 16)
    val, va_meta = sample_subgraphs_with_metadata(graph, "val", 123, 4, 16)
    test, te_meta = sample_subgraphs_with_metadata(graph, "test", 999, 4, 16)
    sets = [set(meta["source_node_ids"].ravel()) for meta in (tr_meta, va_meta, te_meta)]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    assert np.array_equal(train, sample_subgraphs(graph, "train", 7, 5, 16))
    assert np.array_equal(train, train.transpose(0, 2, 1))
    assert not np.diagonal(train, axis1=1, axis2=2).any()
    assert sample_subgraphs(graph, "test", 1, 1, 1).shape == (1, 1, 1)
    with pytest.raises(ValueError, match="fewer"):
        sample_subgraphs(graph, "test", 1, 1, 100)


def test_fractional_attack_budget_is_edge_based_and_zero_is_zero():
    adj = torch.tensor(cycle(10)).unsqueeze(0)
    mask = torch.ones((1, 10), dtype=torch.bool)
    unchanged, _ = edge_flip(adj, mask, rate=0.001)
    assert torch.equal(adj, unchanged)
    attacked, attacked_mask = edge_flip(adj, mask, rate=0.3, generator=torch.Generator().manual_seed(5))
    assert torch.count_nonzero(attacked != adj).item() == 6  # 3 unordered pairs
    assert torch.equal(attacked, attacked.transpose(-1, -2))
    assert torch.equal(mask, attacked_mask)
    explicit, _ = edge_flip(adj, mask, num_flips=4)
    assert torch.count_nonzero(explicit != adj).item() == 8


def test_deletion_preserves_padding_and_permutation_preserves_isomorphism():
    adj = torch.tensor(cycle(8)).unsqueeze(0)
    mask = torch.ones((1, 8), dtype=torch.bool)
    deleted, deleted_mask = node_delete(adj, mask, rate=0.25, generator=torch.Generator().manual_seed(1))
    assert deleted.shape == adj.shape
    assert deleted_mask.sum() == 6
    assert not deleted[0, ~deleted_mask[0]].any()
    permuted, permuted_mask = node_permute(deleted, deleted_mask, generator=torch.Generator().manual_seed(2))
    assert permuted_mask.sum() == 6
    assert permuted.sum() == deleted.sum()
    np.testing.assert_allclose(np.linalg.eigvalsh(permuted[0].numpy()), np.linalg.eigvalsh(deleted[0].numpy()), atol=1e-6)


def test_auroc_ties_and_calibration_boundary():
    assert auroc([0, 0, 1, 1], [0, 1, 1, 2]) == 0.875
    assert auroc([0, 1], [1, 1]) == 0.5
    assert auroc([0, 1], [0, 1]) == 1
    assert np.isnan(auroc([1, 1], [0, 1]))
    values = np.array([1.0, 2.0, 2.0, 3.0])
    threshold = calibrate_threshold(values, 0.25)
    assert (values >= threshold).mean() == 0.25
    metrics = detection_metrics([4, 5], values, threshold)
    assert metrics["tpr"] == 1 and metrics["fpr"] == 0.25
    float32_values = np.ones(32, dtype=np.float32)
    float32_threshold = calibrate_threshold(float32_values, 0.01)
    assert not (float32_values >= float32_threshold).any()
    assert not (torch.from_numpy(float32_values) >= float32_threshold).any()


def test_transitivity_and_assortativity_have_standard_definitions():
    triangle = np.ones((3, 3)) - np.eye(3)
    assert global_transitivity(triangle) == 1
    assert global_transitivity(cycle(5)) == 0
    star = np.zeros((5, 5))
    star[0, 1:] = star[1:, 0] = 1
    assert degree_assortativity(star) == -1
    assert np.isnan(degree_assortativity(cycle(5)))
    changed = cycle(5)
    changed[0, 2] = changed[2, 0] = 1
    metrics = structural_fidelity(cycle(5), changed)
    assert metrics["edges_flipped"] == 1
    assert metrics["edges_flipped_pct"] == 20
    assert metrics["transitivity_change"] > 0


def test_dk2_detects_joint_degree_changes_with_same_degree_sequence():
    # A degree-preserving 2-switch changes correlations between endpoints.
    left = np.zeros((6, 6))
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (4, 5)]:
        left[u, v] = left[v, u] = 1
    right = left.copy()
    right[0, 3] = right[3, 0] = right[4, 5] = right[5, 4] = 0
    right[0, 4] = right[4, 0] = right[3, 5] = right[5, 3] = 1
    assert np.array_equal(left.sum(axis=1), right.sum(axis=1))
    assert dk2_emd(left, left) == 0
    assert dk2_emd(left, right, bins=8) > 0
    assert np.isclose(dk2_emd(left, right), dk2_emd(right, left))


def test_link_prediction_holdout_uses_fixed_true_nonedges_without_leakage():
    graph = cycle(20)
    train, positives, negatives = make_link_prediction_split(graph, seed=12, fraction=0.2)
    assert np.all(graph[positives[:, 0], positives[:, 1]] == 1)
    assert np.all(graph[negatives[:, 0], negatives[:, 1]] == 0)
    assert not np.asarray(train[positives[:, 0], positives[:, 1]]).any()
    score = link_prediction_auc(train, positives, negatives)
    # Reintroducing all held-out pairs cannot leak adjacency labels to scorer.
    poisoned = train.toarray()
    pairs = np.r_[positives, negatives]
    poisoned[pairs[:, 0], pairs[:, 1]] = poisoned[pairs[:, 1], pairs[:, 0]] = 1
    assert link_prediction_auc(poisoned, positives, negatives) == score
    assert 0 <= score <= 1


def test_embedding_cosine_obeys_mask():
    a = np.eye(3)
    b = a.copy()
    b[1] *= -1
    assert node_embedding_cosine(a, b, [True, False, True]) == 1
    assert np.isclose(node_embedding_cosine(a, b), 1 / 3)
