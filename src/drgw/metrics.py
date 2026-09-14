"""Detection and fidelity metrics with explicit finite-sample conventions."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _simple_graph(adj, mask=None):
    graph = adj.tocsr(copy=True) if sparse.issparse(adj) else sparse.csr_matrix(_numpy(adj))
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Expected one square adjacency matrix")
    if mask is not None:
        valid = np.flatnonzero(_numpy(mask).astype(bool))
        graph = graph[valid][:, valid]
    graph = (graph > 0.5).astype(np.float64)
    if (graph != graph.T).nnz:
        raise ValueError("Metrics require an undirected adjacency matrix")
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def auroc(labels, scores):
    """Mann–Whitney AUROC; tied positive/negative scores count as one half."""
    labels, scores = _numpy(labels).reshape(-1), _numpy(scores).reshape(-1)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("labels and scores must have equal nonzero length")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Scores must be finite and labels binary")
    positives = labels == 1
    n_positive, n_negative = int(positives.sum()), int((~positives).sum())
    if not n_positive or not n_negative:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


roc_auc = auroc


def detection_metrics(positive_scores, negative_scores, threshold):
    """Use one preselected threshold (accept iff score >= threshold)."""
    positive = _numpy(positive_scores).astype(np.float64).ravel()
    negative = _numpy(negative_scores).astype(np.float64).ravel()
    if not len(positive) or not len(negative):
        raise ValueError("Both positive and negative scores are required")
    scores = np.r_[positive, negative]
    if not np.isfinite(scores).all():
        raise ValueError("Detection scores must be finite")
    return dict(auroc=auroc(np.r_[np.ones(len(positive)), np.zeros(len(negative))], scores),
                tpr=float(np.mean(positive >= threshold)), fpr=float(np.mean(negative >= threshold)),
                threshold=float(threshold), n_positive=len(positive), n_negative=len(negative))


def calibrate_threshold(negative_validation_scores, target_fpr=0.01):
    """Empirical conservative threshold using validation negatives only.

    The acceptance rule is >=. Ties at the boundary are rejected by choosing
    the next representable float. This does not guarantee population FPR.
    """
    values = np.sort(_numpy(negative_validation_scores).ravel())
    if not len(values) or not np.isfinite(values).all() or not 0 <= target_fpr < 1:
        raise ValueError("Finite validation scores and 0 <= target_fpr < 1 required")
    allowed = int(np.floor(target_fpr * len(values)))
    boundary = values[len(values) - allowed - 1]
    # Step in the score dtype so downstream torch/NumPy float32 comparisons
    # cannot round the threshold back onto the rejected boundary.
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype("float64")
    return float(np.nextafter(dtype.type(boundary), dtype.type(np.inf)))


def degree_assortativity(adj):
    """Pearson correlation of degrees at oriented edge endpoints (Newman r).

    Returns NaN for graphs with no degree variance or no edges.
    """
    graph = _simple_graph(adj)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    rows, cols = sparse.triu(graph, k=1).nonzero()
    if not len(rows):
        return float("nan")
    x, y = degree[rows], degree[cols]
    mean = (x.mean() + y.mean()) / 2
    denominator = (np.mean(x * x) + np.mean(y * y)) / 2 - mean * mean
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float((np.mean(x * y) - mean * mean) / denominator)


def global_transitivity(adj):
    """3*triangles / connected triples; zero if there are no triples."""
    graph = _simple_graph(adj)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    triples_twice = float(np.sum(degree * (degree - 1)))
    if not triples_twice:
        return 0.0
    triangles_six = float(graph.multiply(graph @ graph).sum())
    return triangles_six / triples_twice


def _joint_degree_histogram(graph, bins):
    degree = np.asarray(graph.sum(axis=1)).ravel()
    # Symmetric, oriented-edge distribution over normalized log endpoint degrees.
    rows, cols = graph.nonzero()
    if not len(rows):
        histogram = np.zeros((bins, bins))
        histogram[0, 0] = 1  # declared empty-graph sentinel
        return histogram.ravel()
    log_degree = np.log1p(degree) / np.log(max(graph.shape[0], 2))
    scaled = np.minimum((log_degree * bins).astype(int), bins - 1)
    histogram = np.bincount(scaled[rows] * bins + scaled[cols], minlength=bins * bins).astype(float)
    return histogram / histogram.sum()


def dk2_emd(original, modified, bins=8):
    """Binned 2-D joint-degree Earth Mover distance, normalized to [0,1].

    Each undirected edge contributes both ordered endpoint-degree pairs. Node
    degrees use log(1+degree)/log(N) and uniform bins on [0,1], preserving
    resolution on sparse graphs. Ground cost is half the Manhattan distance
    between normalized log-degree bin centers. Optimal transport is
    solved exactly for this discrete histogram using a linear program. This
    is a binned dK-2 metric, not a 1-D degree-distribution Wasserstein metric.
    Empty graphs use a unit atom in bin (0,0). Bin resolution is a convention,
    so these values must not be claimed equal to unspecified paper dK values.
    """
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    left, right = _simple_graph(original), _simple_graph(modified)
    if left.shape != right.shape:
        raise ValueError("dK-2 comparison requires aligned vertex sets")
    p, q = _joint_degree_histogram(left, bins), _joint_degree_histogram(right, bins)
    if np.array_equal(p, q):
        return 0.0
    centers = (np.indices((bins, bins)).reshape(2, -1).T + 0.5) / bins
    p_idx, q_idx = np.flatnonzero(p), np.flatnonzero(q)
    p, q = p[p_idx], q[q_idx]
    cost = cdist(centers[p_idx], centers[q_idx], metric="cityblock") / 2
    m, n = len(p), len(q)
    constraints = sparse.vstack([
        sparse.kron(sparse.eye(m), np.ones((1, n)), format="csr"),
        sparse.kron(np.ones((1, m)), sparse.eye(n), format="csr"),
    ], format="csr")
    result = linprog(cost.ravel(), A_eq=constraints, b_eq=np.r_[p, q], bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"dK-2 optimal transport failed: {result.message}")
    return float(max(0.0, result.fun))


def edge_flips(original, modified):
    left, right = _simple_graph(original), _simple_graph(modified)
    if left.shape != right.shape:
        raise ValueError("Edge changes require aligned vertex sets")
    return int(sparse.triu(left != right, k=1).nnz)


def structural_fidelity(original, modified, mask=None, bins=8):
    """Signed/absolute deltas and edits relative to original undirected edges."""
    left, right = _simple_graph(original, mask), _simple_graph(modified, mask)
    edits, original_edges = edge_flips(left, right), left.nnz // 2
    original_r, modified_r = degree_assortativity(left), degree_assortativity(right)
    original_c, modified_c = global_transitivity(left), global_transitivity(right)
    return dict(edges_flipped=edits, original_edges=int(original_edges),
                edges_flipped_pct=100 * edits / original_edges if original_edges else (0.0 if not edits else float("nan")),
                assortativity_original=original_r, assortativity_modified=modified_r,
                assortativity_change=modified_r - original_r,
                assortativity_abs_change=abs(modified_r - original_r),
                transitivity_original=original_c, transitivity_modified=modified_c,
                transitivity_change=modified_c - original_c,
                transitivity_abs_change=abs(modified_c - original_c),
                dk2_emd=dk2_emd(left, right, bins), dk2_bins=bins)


def node_embedding_cosine(original_embeddings, modified_embeddings, mask=None):
    """Mean cosine in a shared encoder coordinate system, with aligned nodes.

    This does not fit/rotate embeddings independently. Zero-norm vectors have
    cosine zero; masked nodes are excluded.
    """
    left, right = _numpy(original_embeddings), _numpy(modified_embeddings)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("Embeddings must have matching [...,N,D] shape")
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.divide(np.sum(left * right, axis=-1), denominator,
                       out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    cosine = np.clip(cosine, -1, 1)
    if mask is not None:
        cosine = cosine[_numpy(mask).astype(bool)]
    return float(cosine.mean()) if cosine.size else float("nan")


def make_link_prediction_split(adj, seed=0, fraction=0.1):
    """Return training graph, held-out positives and fixed true nonedges.

    Choose these once from the unwatermarked graph, remove test pairs before
    embedding or fitting, and reuse the same pairs for every method. No
    connectivity repair or test-based sample selection is applied.
    """
    graph = _simple_graph(adj)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    rows, cols = sparse.triu(graph, k=1).nonzero()
    count = int(np.floor(fraction * len(rows)))
    if count == 0:
        raise ValueError("Too few edges for the requested held-out fraction")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(rows))[:count]
    positives = np.column_stack((rows[selected], cols[selected])).astype(np.int64)
    n = graph.shape[0]
    n_nonedges = n * (n - 1) // 2 - len(rows)
    if n_nonedges < count:
        raise ValueError("Not enough true nonedges for a balanced test set")
    # The public runner uses <= 1024-node subgraphs. Materializing their
    # triangular candidate list avoids unbounded rejection on dense graphs.
    u, v = np.triu_indices(n, 1)
    nonedge_mask = np.asarray(graph[u, v]).ravel() == 0
    candidates = np.flatnonzero(nonedge_mask)
    indices = rng.choice(candidates, count, replace=False)
    negatives = np.column_stack((u[indices], v[indices])).astype(np.int64)
    train = graph.tolil()
    train[positives[:, 0], positives[:, 1]] = 0
    train[positives[:, 1], positives[:, 0]] = 0
    train = train.tocsr()
    train.eliminate_zeros()
    return train, positives, negatives


def link_prediction_auc(adj, positive_edges, negative_edges):
    """Adamic-Adar AUROC on fixed held-out edges; remove all test pairs first.

    This is an explicit topology-only downstream probe, not a learned GNN or
    the manuscript's unspecified downstream link predictor. The watermarking
    caller must also use the train graph returned by make_link_prediction_split
    so it never sees held-out positives during watermark training/embedding.
    """
    graph = _simple_graph(adj).tolil()
    positive, negative = _numpy(positive_edges).astype(int), _numpy(negative_edges).astype(int)
    for edges in (positive, negative):
        if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
            raise ValueError("Held-out edges must be nonempty [E,2] arrays")
        if np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("Held-out edges must be canonical unordered pairs (u < v)")
    pset, nset = set(map(tuple, positive)), set(map(tuple, negative))
    if len(pset) != len(positive) or len(nset) != len(negative) or pset & nset:
        raise ValueError("Held-out positives and negatives must be unique and disjoint")
    pairs = np.concatenate((positive, negative))
    if pairs.min() < 0 or pairs.max() >= graph.shape[0]:
        raise ValueError("Held-out vertex index outside graph")
    graph[pairs[:, 0], pairs[:, 1]] = 0
    graph[pairs[:, 1], pairs[:, 0]] = 0
    graph = graph.tocsr()
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    weights = np.zeros_like(degree)
    weights[degree > 1] = 1 / np.log(degree[degree > 1])
    scores = graph @ sparse.diags(weights) @ graph
    values = np.asarray(scores[pairs[:, 0], pairs[:, 1]]).ravel()
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return auroc(labels, values)
