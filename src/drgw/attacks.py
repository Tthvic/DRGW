"""Reproducible topology attacks on padded batches of undirected simple graphs."""

from __future__ import annotations

import math
import torch


def _inputs(adj, mask):
    single = adj.ndim == 2
    if single:
        adj = adj.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    if adj.ndim != 3 or adj.shape[-1] != adj.shape[-2]:
        raise ValueError("adj must have shape [B,N,N] or [N,N]")
    if mask is None:
        mask = torch.ones(adj.shape[:2], dtype=torch.bool, device=adj.device)
    if mask.shape != adj.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B,N] or [N]")
    return adj, mask, single


def _rate(rate):
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be finite and between zero and one")


def _permutation(n, device, generator):
    # A supplied CPU generator also works when the graph is on a GPU.
    rng_device = generator.device if generator is not None else device
    return torch.randperm(n, device=rng_device, generator=generator).to(device)


def _output(adj, mask, single):
    return (adj[0], mask[0]) if single else (adj, mask)


def edge_flip(adj, mask=None, rate=0.1, generator=None, *, num_flips=None):
    """Toggle unique uniformly sampled valid unordered pairs.

    Budget is ``floor(rate * number_of_existing_undirected_edges)``. It is not
    a percentage of all possible pairs. Pairs may be edges or nonedges, so
    this attack generally adds more edges than it removes on sparse graphs.
    An explicit integer ``num_flips`` overrides the fractional budget. Zero
    stays zero, including a 0.1% budget on graphs with fewer than 1,000 edges.
    """
    _rate(rate)
    adj, mask, single = _inputs(adj, mask)
    result = adj.clone()
    if num_flips is not None and (not isinstance(num_flips, int) or num_flips < 0):
        raise ValueError("num_flips must be a nonnegative integer")
    for index in range(len(adj)):
        valid = torch.nonzero(mask[index], as_tuple=True)[0]
        pairs = torch.triu_indices(len(valid), len(valid), offset=1, device=adj.device)
        rows, cols = valid[pairs[0]], valid[pairs[1]]
        edge_count = int((adj[index, rows, cols] > 0.5).sum().item())
        count = num_flips if num_flips is not None else math.floor(rate * edge_count)
        if count > len(rows):
            raise ValueError("Requested more flips than valid unordered pairs")
        chosen = _permutation(len(rows), adj.device, generator)[:count]
        u, v = rows[chosen], cols[chosen]
        toggled = 1 - result[index, u, v]
        result[index, u, v] = toggled
        result[index, v, u] = toggled
    return _output(result, mask.clone(), single)


def node_delete(adj, mask=None, rate=0.1, generator=None):
    """Delete ``floor(rate * valid_nodes)`` uniformly sampled vertices.

    Tensor size stays fixed. Removed vertices become padding (mask=False),
    their incident edges vanish, and surviving isolated vertices remain valid.
    """
    _rate(rate)
    adj, mask, single = _inputs(adj, mask)
    result, new_mask = adj.clone(), mask.clone()
    for index in range(len(adj)):
        valid = torch.nonzero(mask[index], as_tuple=True)[0]
        count = math.floor(rate * len(valid))
        deleted = valid[_permutation(len(valid), adj.device, generator)[:count]]
        new_mask[index, deleted] = False
    result = result * new_mask[:, :, None] * new_mask[:, None, :]
    return _output(result, new_mask, single)


def node_permute(adj, mask=None, rate=1.0, generator=None):
    """Relabel vertices and padding jointly; preserve graph isomorphism exactly.

    ``rate`` selects the floor(rate*N) positions eligible for permutation.
    The default permutes every position. It does not delete or add any edge.
    """
    _rate(rate)
    adj, mask, single = _inputs(adj, mask)
    result, new_mask = adj.clone(), mask.clone()
    for index in range(len(adj)):
        n = adj.shape[-1]
        positions = _permutation(n, adj.device, generator)[:math.floor(rate * n)]
        order = torch.arange(n, device=adj.device)
        order[positions] = positions[_permutation(len(positions), adj.device, generator)]
        result[index] = adj[index][order][:, order]
        new_mask[index] = mask[index, order]
    return _output(result, new_mask, single)


ATTACKS = {"edge_flip": edge_flip, "node_delete": node_delete, "node_permute": node_permute}
