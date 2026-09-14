"""Memory-bounded whole-graph inference with the existing trained parameters.

These routines replace dense adjacency products with CSR sparse matrix products
and chunk node-wise networks. They introduce no new model weights or graph
features. Floating-point summation order can differ from dense inference. Models
trained on sampled subgraphs are still *transfer* models when used on full graphs.

Inference is for one unpadded, undirected simple graph at a time. Functions are
inference-only; training continues to use the differentiable dense implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse as sp
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class SparseGraph:
    """A validated CPU CSR graph with a corresponding torch CSR on ``device``.

    Node labels are only array indices; no labels are passed to the encoder.
    Do not mutate ``adj`` or ``tensor`` in place after construction.
    """

    adj: sp.csr_matrix
    tensor: Tensor
    degree: Tensor
    _normalized: Tensor | None = field(default=None, repr=False)

    @classmethod
    def from_scipy(cls, adjacency, device="cpu", dtype=torch.float32):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Sparse inference supports float32 or float64")
        adj = sp.csr_matrix(adjacency, copy=True)
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("SparseGraph requires a square adjacency matrix")
        adj.sum_duplicates()
        adj.eliminate_zeros()
        adj.sort_indices()
        if not np.isfinite(adj.data).all() or not np.all(adj.data == 1):
            raise ValueError("SparseGraph requires finite binary edge values")
        if np.any(adj.diagonal()) or (adj != adj.T).nnz:
            raise ValueError("SparseGraph requires an undirected graph without self-loops")
        adj = adj.astype(np.uint8, copy=False)
        tensor = _torch_csr(adj, device, dtype)
        degree = torch.as_tensor(np.diff(adj.indptr).copy(), dtype=dtype, device=device)
        return cls(adj=adj, tensor=tensor, degree=degree)

    @property
    def num_nodes(self):
        return self.adj.shape[0]

    @property
    def num_edges(self):
        return self.adj.nnz // 2

    @property
    def device(self):
        return self.tensor.device

    @property
    def dtype(self):
        return self.tensor.dtype

    def normalized(self):
        """D^-1/2 (A+I) D^-1/2, computed once using torch's target dtype."""
        if self._normalized is None:
            loops = self.adj + sp.eye(self.num_nodes, format="csr", dtype=np.uint8)
            base = _torch_csr(loops, self.device, self.dtype)
            inv = (self.degree + 1).rsqrt()
            rows = torch.repeat_interleave(
                torch.arange(self.num_nodes, device=self.device),
                base.crow_indices().diff(),
                output_size=base.values().numel(),
            )
            values = inv[rows] * inv[base.col_indices()]
            self._normalized = torch.sparse_csr_tensor(
                base.crow_indices(), base.col_indices(), values,
                size=base.shape, device=self.device, dtype=self.dtype, check_invariants=True,
            )
        return self._normalized


def _torch_csr(adj, device, dtype):
    return torch.sparse_csr_tensor(
        torch.as_tensor(adj.indptr.astype(np.int64, copy=False), device=device),
        torch.as_tensor(adj.indices.astype(np.int64, copy=False), device=device),
        torch.ones(adj.nnz, device=device, dtype=dtype),
        size=adj.shape, device=device, dtype=dtype, check_invariants=True,
    )


def _check_model(model, graph):
    parameter = next(model.parameters())
    if parameter.device != graph.device or parameter.dtype != graph.dtype:
        raise ValueError("Move the model and SparseGraph to the same device and dtype")


def _node_map(function, x, chunk_size, output_dim=None):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = x.new_empty((len(x), x.shape[-1] if output_dim is None else output_dim))
    for start in range(0, len(x), chunk_size):
        output[start:start + chunk_size] = function(x[start:start + chunk_size])
    return output


@torch.inference_mode()
def node_features_sparse(graph: SparseGraph) -> Tensor:
    """The same eight current-topology features as ``model.node_features``."""
    n = graph.num_nodes
    degree = graph.degree
    max_degree = max(n - 1, 1)
    normalized = degree / max_degree
    neighbor_degree = torch.sparse.mm(graph.tensor, normalized.unsqueeze(-1)).squeeze(-1)
    neighbor_square = torch.sparse.mm(graph.tensor, normalized.square().unsqueeze(-1)).squeeze(-1)
    density = degree.sum() / max(n * max_degree, 1)
    return torch.stack(
        [
            torch.ones_like(degree),
            normalized,
            degree.log1p() / degree.new_tensor(max(n, 1)).log1p(),
            neighbor_degree / degree.clamp_min(1),
            (neighbor_square / degree.clamp_min(1) + 1e-8).sqrt(),
            neighbor_degree / max_degree,
            density.expand_as(degree),
            degree.new_tensor(n / (n + 100)).expand_as(degree),
        ], dim=-1,
    )


@torch.inference_mode()
def encode_sparse(model, graph: SparseGraph, node_chunk_size=65536) -> tuple[Tensor, Tensor]:
    """Exact architecture reuse; returns unpadded [N, hidden_dim] heads."""
    _check_model(model, graph)
    features = node_features_sparse(graph)
    x = _node_map(lambda part: F.gelu(model.encoder.input(part)), features, node_chunk_size, model.hidden_dim)
    del features
    for layer in model.encoder.layers:
        aggregate = torch.sparse.mm(graph.tensor, x)
        aggregate.add_(x, alpha=1 + float(layer.eps))
        mapped = _node_map(layer.mlp, aggregate, node_chunk_size)
        mapped.add_(x).div_(math.sqrt(2))
        x = mapped
        del aggregate
    if hasattr(model.encoder, "latent_head"):
        h = _node_map(model.encoder.latent_head, x, node_chunk_size)
        return h, h
    hs = _node_map(model.encoder.structural_head, x, node_chunk_size)
    hw = _node_map(model.encoder.carrier_head, x, node_chunk_size)
    return hs, hw


@torch.inference_mode()
def flow_sparse(
    model, hw: Tensor, hs: Tensor, graph: SparseGraph, reverse=False, node_chunk_size=65536
) -> tuple[Tensor, Tensor]:
    """Conditional forward/inverse with the dense model's exact weights.

    The first GCN layer uses ``A_norm @ concat(a, hs) == concat(A_norm @ a,
    A_norm @ hs)`` to avoid a full [N, 1.5D] input allocation.
    """
    _check_model(model, graph)
    expected = (graph.num_nodes, model.hidden_dim)
    if hw.shape != expected or hs.shape != expected:
        raise ValueError(f"Sparse flow expects hw and hs with shape {expected}")
    if node_chunk_size <= 0:
        raise ValueError("node_chunk_size must be positive")
    if not hasattr(model, "inn"):
        return hw, hw.new_zeros(())
    normalized = graph.normalized()
    conditioned_hs = torch.sparse.mm(normalized, hs)
    x = hw
    total_logdet = hw.new_zeros(())
    layers = reversed(model.inn.layers) if reverse else model.inn.layers
    half = model.hidden_dim // 2
    for layer in layers:
        left, right = x.chunk(2, dim=-1)
        a, b = (right, left) if layer.swap else (left, right)
        aggregated_a = torch.sparse.mm(normalized, a)
        intermediate = torch.empty_like(x)
        for start in range(0, len(x), node_chunk_size):
            section = slice(start, start + node_chunk_size)
            joined = torch.cat((aggregated_a[section], conditioned_hs[section]), dim=-1)
            intermediate[section] = F.gelu(layer.conditioner.first(joined))
        del aggregated_a
        aggregate = torch.sparse.mm(normalized, intermediate)
        del intermediate
        raw = _node_map(layer.conditioner.second, aggregate, node_chunk_size)
        del aggregate
        output = torch.empty_like(x)
        for start in range(0, len(x), node_chunk_size):
            section = slice(start, start + node_chunk_size)
            raw_scale, shift = raw[section].chunk(2, dim=-1)
            scale = layer.scale_limit * raw_scale.tanh()
            if reverse:
                changed = (b[section] - shift) * torch.exp(-scale)
                total_logdet -= scale.sum()
            else:
                changed = b[section] * torch.exp(scale) + shift
                total_logdet += scale.sum()
            if layer.swap:
                output[section, :half], output[section, half:] = changed, a[section]
            else:
                output[section, :half], output[section, half:] = a[section], changed
        x = output
        del raw
    return x, total_logdet


@torch.inference_mode()
def latent_sparse(model, graph: SparseGraph, node_chunk_size=65536) -> Tensor:
    """Blind full-graph signal, shape [watermark_dim], with masked-mean convention."""
    hs, hw = encode_sparse(model, graph, node_chunk_size)
    z, _ = flow_sparse(model, hw, hs, graph, node_chunk_size=node_chunk_size)
    return z.sum(0)[:model.watermark_dim] / max(graph.num_nodes, 1)


def sample_nonedges(adj: sp.csr_matrix, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform distinct unordered nonedges, shape [2,count], without NxN arrays."""
    n = adj.shape[0]
    available = n * (n - 1) // 2 - adj.nnz // 2
    if count < 0 or count > available:
        raise ValueError("Nonedge sample count is outside the available range")
    if count == 0:
        return np.empty((2, 0), dtype=np.int64)
    # Rejection sampling is inefficient for nearly complete small graphs.
    if n <= 4096 and available < n * (n - 1) // 4:
        all_ids = []
        for u in range(n):
            present = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
            absent = np.setdiff1d(np.arange(u + 1, n), present, assume_unique=True)
            all_ids.append(u * n + absent)
        ids = np.concatenate(all_ids)
        selected = rng.choice(ids, size=count, replace=False)
        return np.stack((selected // n, selected % n))
    selected = np.empty(0, dtype=np.int64)
    while len(selected) < count:
        proposed_count = min(1_000_000, max(64, 2 * (count - len(selected))))
        endpoints = rng.integers(0, n, size=(2, proposed_count), dtype=np.int64)
        u, v = np.minimum(endpoints[0], endpoints[1]), np.maximum(endpoints[0], endpoints[1])
        eligible = (u != v) & (np.asarray(adj[u, v]).ravel() == 0)
        proposals = np.unique(u[eligible] * n + v[eligible])
        proposals = np.setdiff1d(proposals, selected, assume_unique=True)
        missing = count - len(selected)
        if len(proposals) > missing:
            proposals = rng.choice(proposals, size=missing, replace=False)
        selected = np.union1d(selected, proposals)
    return np.stack((selected // n, selected % n))


def candidate_pairs_sparse(graph: SparseGraph, nonedge_ratio=1.0, minimum=0, candidate_seed=0):
    """All existing edges plus the same nonedge-count policy as dense DRGW."""
    if nonedge_ratio < 0 or minimum < 0:
        raise ValueError("nonedge_ratio and minimum must be nonnegative")
    upper = sp.triu(graph.adj, k=1, format="coo")
    existing = np.stack((upper.row, upper.col)).astype(np.int64, copy=False)
    available = graph.num_nodes * (graph.num_nodes - 1) // 2 - graph.num_edges
    count = min(available, max(32, math.ceil(graph.num_edges * nonedge_ratio), minimum - graph.num_edges))
    nonedges = sample_nonedges(graph.adj, count, np.random.default_rng(candidate_seed))
    return np.concatenate((existing, nonedges), axis=1)


def _edited_csr(adj, pairs):
    if pairs.shape[1] == 0:
        return adj.copy()
    u, v = pairs
    current = np.asarray(adj[u, v]).ravel().astype(np.int8)
    values = 1 - 2 * current
    delta = sp.coo_matrix(
        (np.concatenate((values, values)), (np.concatenate((u, v)), np.concatenate((v, u)))),
        shape=adj.shape, dtype=np.int8,
    ).tocsr()
    edited = adj.astype(np.int8) + delta
    edited.eliminate_zeros()
    return edited.astype(np.uint8)


@torch.inference_mode()
def embed_sparse(
    model,
    graph: SparseGraph,
    w: Tensor,
    alpha=0.1,
    budget_ratio=0.001,
    edit_budget=None,
    candidate_seed=0,
    chunk_size=4096,
    node_chunk_size=65536,
    return_latents=False,
) -> dict:
    """Inject then choose global top-k flip utilities from streamed edge scores.

    Candidate indices require O(E) host memory; node states require O(ND) device
    memory. Pair features and score-selection buffers require O(chunk_size*D+k),
    never O(N²). Returned ``adj`` is SciPy CSR and ``graph`` wraps that graph on
    the original device. ``edit_pairs`` is [K,2], with unordered endpoints.

    The NumPy sampler and torch dense sampler need not choose identical nonedges
    for the same integer seed, but use the same uniform-without-replacement law.
    """
    _check_model(model, graph)
    w = torch.as_tensor(w, device=graph.device, dtype=graph.dtype)
    if w.shape != (model.watermark_dim,) or not bool(torch.isfinite(w).all()):
        raise ValueError(f"w must be one finite {model.watermark_dim}-dimensional vector")
    if not math.isfinite(budget_ratio) or budget_ratio < 0 or chunk_size <= 0:
        raise ValueError("budget_ratio must be finite/nonnegative and chunk_size positive")
    if edit_budget is None:
        requested = math.floor(graph.num_edges * budget_ratio)
    else:
        if not math.isfinite(float(edit_budget)) or edit_budget < 0 or int(edit_budget) != edit_budget:
            raise ValueError("edit_budget must be a nonnegative integer")
        requested = int(edit_budget)
    k = min(requested, graph.num_nodes * (graph.num_nodes - 1) // 2)
    hs, hw = encode_sparse(model, graph, node_chunk_size)
    z, logdet = flow_sparse(model, hw, hs, graph, node_chunk_size=node_chunk_size)
    z_graph = z.sum(0)[:model.watermark_dim] / max(graph.num_nodes, 1)
    target_z_graph = z_graph + alpha * w if graph.num_nodes else z_graph.clone()
    result = dict(
        graph=graph, adj=graph.adj, z_graph=z_graph, target_z_graph=target_z_graph,
        edit_pairs=np.empty((0, 2), dtype=np.int64), edit_counts=0,
        requested_budget=requested, edge_counts=graph.num_edges, logdet=logdet,
    )
    if k == 0 and not return_latents:
        return result
    target_z = z.clone()
    target_z[:, :model.watermark_dim] += alpha * w
    target_hw, _ = flow_sparse(model, target_z, hs, graph, reverse=True, node_chunk_size=node_chunk_size)
    del target_z
    if return_latents:
        result.update(hs=hs, hw=hw, z_node=z, target_hw=target_hw)
    if k == 0:
        return result
    pairs = candidate_pairs_sparse(graph, model.nonedge_ratio, k, candidate_seed)
    best_scores = z.new_empty(0)
    best_pairs = torch.empty((2, 0), dtype=torch.long, device=graph.device)
    for start in range(0, pairs.shape[1], chunk_size):
        cpu_chunk = pairs[:, start:start + chunk_size]
        gpu_chunk = torch.as_tensor(cpu_chunk, device=graph.device)
        logits = model.editor.score_pairs(hs, target_hw, gpu_chunk)
        existing = torch.as_tensor(np.asarray(graph.adj[cpu_chunk[0], cpu_chunk[1]]).ravel(), dtype=graph.dtype, device=graph.device)
        scores = logits * (1 - 2 * existing)
        combined_scores = torch.cat((best_scores, scores))
        combined_pairs = torch.cat((best_pairs, gpu_chunk), dim=1)
        keep = min(k, combined_scores.numel())
        best_scores, positions = combined_scores.topk(keep, sorted=False)
        best_pairs = combined_pairs[:, positions]
    chosen = best_pairs.cpu().numpy()
    modified = _edited_csr(graph.adj, chosen)
    edited_graph = SparseGraph.from_scipy(modified, device=graph.device, dtype=graph.dtype)
    result.update(graph=edited_graph, adj=edited_graph.adj, edit_pairs=chosen.T,
                  edit_counts=chosen.shape[1], edit_scores=best_scores.cpu().numpy())
    return result
