"""DRGW encoder, graph-aware flow and structure-aware editor.

Inputs are batches of dense undirected adjacency matrices and boolean node
masks. Node features are recomputed from the current graph; verification never
uses an original graph, node identifiers, or the watermark as encoder input.

The implementation uses topology-only
input features, masked mean pooling, the first ``watermark_dim`` coordinates as
the watermark subspace, symmetrized endpoint logits converted to flip utility,
and a straight-through relaxation for training the discrete editor. Feature
reconstruction and variance objectives are configurable encoder auxiliaries.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _adjacency(adj: Tensor, mask: Tensor) -> Tensor:
    if adj.ndim != 3 or adj.shape[-1] != adj.shape[-2]:
        raise ValueError("adj must have shape [batch, nodes, nodes]")
    if mask.shape != adj.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [batch, nodes]")
    if not adj.is_floating_point():
        raise ValueError("adj must have a floating-point dtype")
    valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    diagonal = torch.eye(adj.shape[-1], dtype=torch.bool, device=adj.device)
    return adj * (valid & ~diagonal).to(adj.dtype)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Pool nodes, returning zeros for an entirely empty padded graph."""
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1)


def node_features(adj: Tensor, mask: Tensor) -> Tensor:
    """Eight differentiable, permutation-equivariant topology features.

    Columns are a constant, normalized degree, normalized log-degree, mean and
    RMS neighbor degree, two-step walk density, graph density, and bounded graph
    size. Every quantity depends only on the currently supplied graph and mask.
    No triangle counting or dense [B,N,N,D] tensor is required.
    """
    adj = _adjacency(adj, mask)
    n = mask.sum(-1, keepdim=True).to(adj.dtype)
    max_degree = (n - 1).clamp_min(1)
    degree = adj.sum(-1)
    normalized = degree / max_degree
    neighbors_degree = torch.bmm(adj, normalized.unsqueeze(-1)).squeeze(-1)
    neighbor_mean = neighbors_degree / degree.clamp_min(1)
    neighbor_square = torch.bmm(adj, normalized.square().unsqueeze(-1)).squeeze(-1)
    neighbor_rms = (neighbor_square / degree.clamp_min(1) + 1e-8).sqrt()
    density = degree.sum(-1, keepdim=True) / (n * max_degree).clamp_min(1)
    features = torch.stack(
        [
            torch.ones_like(degree),
            normalized,
            degree.clamp_min(0).log1p() / n.clamp_min(1).log1p(),
            neighbor_mean,
            neighbor_rms,
            neighbors_degree / max_degree,
            density.expand_as(degree),
            (n / (n + 100)).expand_as(degree),
        ],
        dim=-1,
    )
    return features * mask.unsqueeze(-1)


class GINLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor) -> Tensor:
        aggregate = (1 + self.eps) * x + torch.bmm(adj, x)
        return ((x + self.mlp(aggregate)) / math.sqrt(2)) * mask.unsqueeze(-1)


class DisentangledEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.input = nn.Linear(8, hidden_dim)
        self.layers = nn.ModuleList(GINLayer(hidden_dim) for _ in range(layers))
        self.structural_head = nn.Linear(hidden_dim, hidden_dim)
        self.carrier_head = nn.Linear(hidden_dim, hidden_dim)
        # Optional auxiliary reconstruction heads prevent constant embeddings.
        self.structural_decoder = nn.Linear(hidden_dim, 8)
        self.carrier_decoder = nn.Linear(hidden_dim, 8)

    def forward(self, adj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        adj = _adjacency(adj, mask)
        x = F.gelu(self.input(node_features(adj, mask))) * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, mask)
        valid = mask.unsqueeze(-1)
        return self.structural_head(x) * valid, self.carrier_head(x) * valid


def _normalized_adjacency(adj: Tensor, mask: Tensor) -> Tensor:
    adj = _adjacency(adj, mask)
    adj = adj + torch.diag_embed(mask.to(adj.dtype))
    inv_sqrt = adj.sum(-1).clamp_min(1e-8).rsqrt()
    return adj * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)


class GraphConditioner(nn.Module):
    """Two GCN layers conditioned on the unchanged carrier half and hs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.first = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.second = nn.Linear(hidden_dim, hidden_dim)
        # Start with an identity flow, then learn nonzero affine parameters.
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, a: Tensor, hs: Tensor, norm_adj: Tensor, mask: Tensor) -> Tensor:
        x = torch.cat((a, hs), dim=-1)
        x = F.gelu(self.first(torch.bmm(norm_adj, x)))
        return self.second(torch.bmm(norm_adj, x)) * mask.unsqueeze(-1)


class AffineCoupling(nn.Module):
    def __init__(self, hidden_dim: int, swap: bool, scale_limit: float):
        super().__init__()
        self.conditioner = GraphConditioner(hidden_dim)
        self.swap = swap
        self.scale_limit = scale_limit

    def forward(
        self, x: Tensor, hs: Tensor, norm_adj: Tensor, mask: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        left, right = x.chunk(2, dim=-1)
        a, b = (right, left) if self.swap else (left, right)
        raw_scale, shift = self.conditioner(a, hs, norm_adj, mask).chunk(2, dim=-1)
        scale = self.scale_limit * raw_scale.tanh()
        if reverse:
            transformed = (b - shift) * torch.exp(-scale)
            logdet = -scale.sum(dim=(1, 2))
        else:
            transformed = b * torch.exp(scale) + shift
            logdet = scale.sum(dim=(1, 2))
        out = torch.cat((transformed, a) if self.swap else (a, transformed), dim=-1)
        # Padded coordinates pass through unchanged, preserving bijectivity.
        return out, logdet


class GraphAwareINN(nn.Module):
    def __init__(self, hidden_dim: int = 256, layers: int = 8, scale_limit: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList(
            AffineCoupling(hidden_dim, bool(i % 2), scale_limit) for i in range(layers)
        )

    def forward(
        self, hw: Tensor, hs: Tensor, adj: Tensor, mask: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        norm_adj = _normalized_adjacency(adj, mask)
        x = hw
        logdet = hw.new_zeros(hw.shape[0])
        for layer in reversed(self.layers) if reverse else self.layers:
            x, increment = layer(x, hs, norm_adj, mask, reverse=reverse)
            logdet = logdet + increment
        return x, logdet


class StructureAwareEditor(nn.Module):
    """Predict unordered edge logits without constructing all pair features.

    Stage 2 supervises these logits with edge-presence BCE. The embedding path
    converts them to flip utility before top-k, so a confident existing edge is
    preserved rather than deleted merely because its existence score is high.
    """

    def __init__(self, hidden_dim: int = 256, chunk_size: int = 4096):
        super().__init__()
        self.chunk_size = chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_pairs(self, hs: Tensor, hw: Tensor, pairs: Tensor) -> Tensor:
        """hs/hw: [N,D], pairs: [2,C]; average both endpoint orders."""
        scores = []
        for chunk in pairs.split(self.chunk_size, dim=1):
            u, v = chunk
            uv = torch.cat((hs[u], hw[u], hs[v], hw[v]), dim=-1)
            vu = torch.cat((hs[v], hw[v], hs[u], hw[u]), dim=-1)
            scores.append(0.5 * (self.mlp(uv) + self.mlp(vu)).squeeze(-1))
        return torch.cat(scores) if scores else hs.new_empty(0)


class DRGW(nn.Module):
    """DRGW encoder, conditional INN, and budget-constrained edge editor.

    ``latent`` is mean-pooled over valid nodes, then restricted to its first
    ``watermark_dim`` coordinates. Node-level Gaussian NLL does not, by itself,
    establish a standard-normal graph-level null; calibrate detection on held-out
    unwatermarked graphs before interpreting Gaussian p-values.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        watermark_dim: int = 128,
        gin_layers: int = 4,
        flow_layers: int = 8,
        scale_limit: float = 1.0,
        editor_chunk_size: int = 4096,
        nonedge_ratio: float = 1.0,
    ):
        super().__init__()
        if hidden_dim < 2 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer")
        if not 1 <= watermark_dim <= hidden_dim:
            raise ValueError("watermark_dim must be in [1, hidden_dim]")
        if gin_layers < 1 or flow_layers < 1 or editor_chunk_size < 1:
            raise ValueError("layer counts and editor_chunk_size must be positive")
        if nonedge_ratio < 0 or scale_limit <= 0:
            raise ValueError("nonedge_ratio must be nonnegative and scale_limit positive")
        self.hidden_dim = hidden_dim
        self.watermark_dim = watermark_dim
        self.nonedge_ratio = nonedge_ratio
        self.encoder = DisentangledEncoder(hidden_dim, gin_layers)
        self.inn = GraphAwareINN(hidden_dim, flow_layers, scale_limit)
        self.editor = StructureAwareEditor(hidden_dim, editor_chunk_size)

    @staticmethod
    def node_features(adj: Tensor, mask: Tensor) -> Tensor:
        return node_features(adj, mask)

    @staticmethod
    def pooled(x: Tensor, mask: Tensor) -> Tensor:
        return masked_mean(x, mask)

    def encode(self, adj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(adj, mask)

    def flow(
        self, hw: Tensor, hs: Tensor, adj: Tensor, mask: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        return self.inn(hw, hs, adj, mask, reverse=reverse)

    def latent(self, adj: Tensor, mask: Tensor) -> Tensor:
        hs, hw = self.encode(adj, mask)
        z, _ = self.flow(hw, hs, adj, mask)
        return masked_mean(z, mask)[..., : self.watermark_dim]

    def _candidate_pairs(
        self,
        adj: Tensor,
        mask: Tensor,
        minimum: int = 0,
        generator: torch.Generator | None = None,
        all_pairs: bool = False,
    ) -> Tensor:
        valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pairs = torch.triu(valid, diagonal=1).nonzero().T
        if all_pairs or pairs.shape[1] == 0:
            return pairs
        existing = adj[pairs[0], pairs[1]] > 0.5
        edges, nonedges = pairs[:, existing], pairs[:, ~existing]
        count = min(
            nonedges.shape[1],
            max(32, math.ceil(edges.shape[1] * self.nonedge_ratio), minimum - edges.shape[1]),
        )
        if count < nonedges.shape[1]:
            chosen = torch.randperm(nonedges.shape[1], device=adj.device, generator=generator)[:count]
            nonedges = nonedges[:, chosen]
        return torch.cat((edges, nonedges), dim=1)

    def embed(
        self,
        adj: Tensor,
        mask: Tensor,
        w: Tensor,
        alpha: float = 0.1,
        budget_ratio: float = 0.001,
        straight_through: bool = False,
        edit_budget: int | Tensor | None = None,
        temperature: float = 0.2,
        generator: torch.Generator | None = None,
        all_pairs: bool = False,
    ) -> dict[str, Tensor]:
        """Inject a latent watermark, then flip exactly the feasible budget.

        Default budget is floor(number of undirected edges * budget_ratio),
        including zero for small graphs. An explicit integer/tensor edit_budget
        overrides that ratio and is returned as ``requested_budget``. If there
        are fewer eligible pairs, edit_counts records the smaller actual count.
        In straight-through mode forward values remain exactly binary; only
        backward derivatives use sigmoid scores normalized to total mass k.
        No implicit minimum edit exists.

        The editor emits edge-presence logits for consistency with stage 2 BCE.
        A nonedge's flip score is +logit and an existing edge's is -logit; this
        conversion supplies the edit affinity used by top-k.

        hs/hw/z_node/z_graph describe the original graph. Call ``latent`` on the
        returned adjacency to extract the signal actually surviving hard edits.
        """
        adj = _adjacency(adj, mask)
        if w.shape != (adj.shape[0], self.watermark_dim):
            raise ValueError("w must have shape [batch, watermark_dim]")
        if budget_ratio < 0 or temperature <= 0:
            raise ValueError("budget_ratio must be nonnegative and temperature positive")
        if not bool(torch.all((adj == 0) | (adj == 1))):
            raise ValueError("embed requires binary input adjacency")
        if not torch.equal(adj, adj.transpose(-1, -2)):
            raise ValueError("embed requires undirected input adjacency")
        counts = adj.sum(dim=(1, 2)).long() // 2
        if edit_budget is None:
            requested = torch.floor(counts.to(torch.float64) * budget_ratio).long()
        else:
            supplied = torch.as_tensor(edit_budget, device=adj.device)
            if bool(torch.any(supplied < 0)) or bool(torch.any(supplied != supplied.floor())):
                raise ValueError("edit_budget must contain nonnegative integers")
            requested = supplied.long().expand(adj.shape[0])

        hs, hw = self.encode(adj, mask)
        z, logdet = self.flow(hw, hs, adj, mask)
        injection = F.pad(alpha * w, (0, self.hidden_dim - self.watermark_dim))
        target_z = z + injection.unsqueeze(1) * mask.unsqueeze(-1)
        target_hw, _ = self.flow(target_z, hs, adj, mask, reverse=True)
        edits, actual = [], []
        for b in range(adj.shape[0]):
            k_requested = int(requested[b].item())
            edit = torch.zeros_like(adj[b])
            if k_requested == 0:
                edits.append(edit)
                actual.append(0)
                continue
            pairs = self._candidate_pairs(adj[b], mask[b], k_requested, generator, all_pairs)
            k = min(k_requested, pairs.shape[1])
            if k == 0:
                edits.append(edit)
                actual.append(0)
                continue
            logits = self.editor.score_pairs(hs[b], target_hw[b], pairs)
            scores = logits * (1 - 2 * adj[b, pairs[0], pairs[1]])
            selected = scores.topk(k, sorted=False).indices
            hard = torch.zeros_like(scores).scatter(0, selected, 1)
            if straight_through and k < scores.numel():
                boundary = scores.topk(k + 1).values[-2:].mean().detach()
                soft = torch.sigmoid((scores - boundary) / temperature)
                # The backward surrogate carries the same total edit mass as
                # the hard plan, avoiding an effective thousands-edge update
                # when the forward budget is only one or two edges.
                soft = soft * (k / soft.sum().clamp_min(1e-8))
                values = hard + (soft - soft.detach())
            else:
                values = hard
            edit = edit.index_put((pairs[0], pairs[1]), values)
            edit = edit + edit.T
            edits.append(edit)
            actual.append(k)
        edit_mask = torch.stack(edits)
        watermarked = adj + (1 - 2 * adj) * edit_mask
        return {
            "adj": watermarked,
            "hs": hs,
            "hw": hw,
            "z_node": z,
            "z_graph": masked_mean(z, mask)[..., : self.watermark_dim],
            "target_z_graph": masked_mean(target_z, mask)[..., : self.watermark_dim],
            "target_hw": target_hw,
            "edit_mask": edit_mask,
            "edit_counts": torch.tensor(actual, dtype=torch.long, device=adj.device),
            "requested_budget": requested,
            "edge_counts": counts,
            "logdet": logdet,
        }

    @staticmethod
    def flow_nll(z: Tensor, logdet: Tensor, mask: Tensor) -> Tensor:
        """Node-density NLL per active scalar, omitting the Gaussian constant."""
        quadratic = 0.5 * (z.square() * mask.unsqueeze(-1)).sum(dim=(1, 2))
        dimensions = mask.sum(-1).clamp_min(1) * z.shape[-1]
        return ((quadratic - logdet) / dimensions).mean()

    def encoder_losses(
        self, adj: Tensor, mask: Tensor, augmented_adj: Tensor, augmented_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Unweighted paper losses plus explicit optional anti-collapse losses.

        ``invariance`` is graph-level MSE and ``orthogonality`` is the squared
        cross-moment of normalized structural/carrier features. ``feature`` and
        ``variance`` are auxiliary terms controlled by the training configuration.
        """
        augmented_mask = mask if augmented_mask is None else augmented_mask
        hs, hw = self.encode(adj, mask)
        hs_aug, _ = self.encode(augmented_adj, augmented_mask)
        invariance = F.mse_loss(masked_mean(hs, mask), masked_mean(hs_aug, augmented_mask))
        active_s, active_w = hs[mask], hw[mask]
        if active_s.shape[0] == 0:
            zero = hs.sum() * 0
            return {"invariance": invariance, "orthogonality": zero, "feature": zero, "variance": zero}
        normalized_s, normalized_w = F.normalize(active_s, dim=-1), F.normalize(active_w, dim=-1)
        cross = normalized_s.T @ normalized_w / active_s.shape[0]
        orthogonality = cross.square().sum()
        features = node_features(adj, mask)[mask].detach()
        feature = F.mse_loss(self.encoder.structural_decoder(active_s), features)
        feature = feature + F.mse_loss(self.encoder.carrier_decoder(active_w), features)
        std_s = (active_s.var(0, unbiased=False) + 1e-4).sqrt()
        std_w = (active_w.var(0, unbiased=False) + 1e-4).sqrt()
        variance = F.relu(0.1 - std_s).mean() + F.relu(0.1 - std_w).mean()
        return {"invariance": invariance, "orthogonality": orthogonality, "feature": feature, "variance": variance}

    def reconstruction_loss(
        self,
        adj: Tensor,
        mask: Tensor,
        hs: Tensor | None = None,
        hw: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Candidate-edge BCE for stage 2 initialization of the editor."""
        if hs is None or hw is None:
            hs, hw = self.encode(adj, mask)
        losses = []
        for b in range(adj.shape[0]):
            pairs = self._candidate_pairs(adj[b], mask[b], generator=generator)
            if pairs.shape[1] == 0:
                continue
            logits = self.editor.score_pairs(hs[b], hw[b], pairs)
            target = adj[b, pairs[0], pairs[1]]
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
        return torch.stack(losses).mean() if losses else hs.sum() * 0
