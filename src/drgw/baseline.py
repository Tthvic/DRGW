"""A reconstruction-only latent graph-autoencoder reference baseline.

The baseline shares DRGW's graph-only features, GIN backbone size,
pooling, candidate sampling, and hard editing budget. It has one representation
head and an adjacency decoder: no representation disentanglement, INN, or
watermark/attack-aware training objective is included.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import DRGW, GINLayer, _adjacency, masked_mean, node_features


class AutoencoderEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int):
        super().__init__()
        self.input = nn.Linear(8, hidden_dim)
        self.layers = nn.ModuleList(GINLayer(hidden_dim) for _ in range(layers))
        self.latent_head = nn.Linear(hidden_dim, hidden_dim)
        self.feature_decoder = nn.Linear(hidden_dim, 8)

    def forward(self, adj: Tensor, mask: Tensor) -> Tensor:
        adj = _adjacency(adj, mask)
        x = F.gelu(self.input(node_features(adj, mask))) * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, mask)
        return self.latent_head(x) * mask.unsqueeze(-1)


class AdjacencyDecoder(nn.Module):
    """Symmetric two-endpoint adjacency logits from one latent representation."""

    def __init__(self, hidden_dim: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_pairs(self, hs: Tensor, hw: Tensor, pairs: Tensor) -> Tensor:
        """Compatibility signature; ``hs`` is intentionally unused."""
        scores = []
        for u, v in pairs.split(self.chunk_size, dim=1):
            uv = torch.cat((hw[u], hw[v]), dim=-1)
            vu = torch.cat((hw[v], hw[u]), dim=-1)
            scores.append(0.5 * (self.mlp(uv) + self.mlp(vu)).squeeze(-1))
        return torch.cat(scores) if scores else hw.new_empty(0)


class NaiveLatent(nn.Module):
    """Graph-autoencoder control with a direct additive latent watermark.

    Training should minimize reconstruction BCE, optional topology-feature and
    variance auxiliaries, and a small latent-L2 penalty. Train neither a matched
    filter nor an attack loss for this baseline. ``model_kind`` is available for
    checkpoints/configuration dispatch.

    The ``encode`` tuple and identity ``flow`` method only preserve the DRGW
    caller interface: both tuple entries are the *same* single-head embedding,
    and no invertible-network parameters/modules are created.
    """

    model_kind = "naive_latent"

    def __init__(
        self,
        hidden_dim: int = 256,
        watermark_dim: int = 128,
        gin_layers: int = 4,
        editor_chunk_size: int = 4096,
        nonedge_ratio: float = 1.0,
    ):
        super().__init__()
        if hidden_dim < 1 or not 1 <= watermark_dim <= hidden_dim:
            raise ValueError("require hidden_dim >= watermark_dim >= 1")
        if gin_layers < 1 or editor_chunk_size < 1 or nonedge_ratio < 0:
            raise ValueError("layer/chunk counts must be positive and nonedge_ratio nonnegative")
        self.hidden_dim = hidden_dim
        self.watermark_dim = watermark_dim
        self.nonedge_ratio = nonedge_ratio
        self.encoder = AutoencoderEncoder(hidden_dim, gin_layers)
        self.editor = AdjacencyDecoder(hidden_dim, editor_chunk_size)

    # These operations need only the public encode/editor interface.
    _candidate_pairs = DRGW._candidate_pairs
    reconstruction_loss = DRGW.reconstruction_loss
    node_features = staticmethod(node_features)
    pooled = staticmethod(masked_mean)

    def encode(self, adj: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(adj, mask)
        return h, h

    @staticmethod
    def flow(
        hw: Tensor, hs: Tensor, adj: Tensor, mask: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Identity adapter, not a learned flow or density model."""
        return hw, hw.new_zeros(hw.shape[0])

    def latent(self, adj: Tensor, mask: Tensor) -> Tensor:
        return masked_mean(self.encoder(adj, mask), mask)[..., : self.watermark_dim]

    @staticmethod
    def latent_l2(h: Tensor, mask: Tensor) -> Tensor:
        """Mean squared active scalar, for a small explicit latent regularizer."""
        return (h.square() * mask.unsqueeze(-1)).sum() / (mask.sum().clamp_min(1) * h.shape[-1])

    def encoder_losses(
        self,
        adj: Tensor,
        mask: Tensor,
        augmented_adj: Tensor | None = None,
        augmented_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Feature/variance auxiliaries; no invariance or orthogonality training."""
        h = self.encoder(adj, mask)
        zero = h.sum() * 0
        active = h[mask]
        if active.shape[0] == 0:
            return {"invariance": zero, "orthogonality": zero, "feature": zero, "variance": zero}
        target = node_features(adj, mask)[mask].detach()
        feature = F.mse_loss(self.encoder.feature_decoder(active), target)
        std = (active.var(0, unbiased=False) + 1e-4).sqrt()
        variance = F.relu(0.1 - std).mean()
        return {"invariance": zero, "orthogonality": zero, "feature": feature, "variance": variance}

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
        """Decode the shifted carrier with exactly the feasible edit budget.

        An absent edge's flip score is its decoded logit; an existing edge's
        score is minus that logit. Top-k therefore chooses the most favorable
        changes under the decoder, including low-confidence reconstruction
        changes. This decoder is not trained to preserve a watermark.
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
        h = self.encoder(adj, mask)
        injection = F.pad(alpha * w, (0, self.hidden_dim - self.watermark_dim))
        target_h = h + injection.unsqueeze(1) * mask.unsqueeze(-1)
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
            logits = self.editor.score_pairs(h[b], target_h[b], pairs)
            scores = logits * (1 - 2 * adj[b, pairs[0], pairs[1]])
            hard = torch.zeros_like(scores).scatter(0, scores.topk(k, sorted=False).indices, 1)
            if straight_through and k < scores.numel():
                boundary = scores.topk(k + 1).values[-2:].mean().detach()
                soft = torch.sigmoid((scores - boundary) / temperature)
                soft = soft * (k / soft.sum().clamp_min(1e-8))
                values = hard + (soft - soft.detach())
            else:
                values = hard
            edit = edit.index_put((pairs[0], pairs[1]), values)
            edits.append(edit + edit.T)
            actual.append(k)
        edit_mask = torch.stack(edits)
        return {
            "adj": adj + (1 - 2 * adj) * edit_mask,
            "hs": h,
            "hw": h,
            "z_node": h,
            "z_graph": masked_mean(h, mask)[..., : self.watermark_dim],
            "target_z_graph": masked_mean(target_h, mask)[..., : self.watermark_dim],
            "target_hw": target_h,
            "edit_mask": edit_mask,
            "edit_counts": torch.tensor(actual, dtype=torch.long, device=adj.device),
            "requested_budget": requested,
            "edge_counts": counts,
            "logdet": h.new_zeros(h.shape[0]),
        }
