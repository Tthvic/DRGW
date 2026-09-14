"""Command-line entry points for datasets, experiments, embedding and verification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .runtime import load_checkpoint, read_config


def read_graph(path, device):
    with np.load(path, allow_pickle=False) as source:
        adj = torch.as_tensor(source["adj"], dtype=torch.float32, device=device)
        if adj.ndim == 2:
            adj = adj.unsqueeze(0)
        mask = torch.as_tensor(source["mask"], dtype=torch.bool, device=device) if "mask" in source else torch.ones(adj.shape[:2], dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
    if adj.ndim != 3 or adj.shape[-1] != adj.shape[-2] or mask.shape != adj.shape[:2]:
        raise ValueError("Expected adj [N,N] or [B,N,N], with optional corresponding mask")
    if not torch.all((adj == 0) | (adj == 1)) or not torch.equal(adj, adj.transpose(-1, -2)):
        raise ValueError("Graph adjacency must be finite, binary and symmetric")
    if torch.count_nonzero(adj.diagonal(dim1=-2, dim2=-1)):
        raise ValueError("Graph adjacency must have a zero diagonal")
    return adj, mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/main.yaml")
        if name != "prepare":
            command.add_argument("--device", default="cpu")
        if name == "train":
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--kind", choices=("drgw", "naive"))
            command.add_argument("--resume", action="store_true")
        if name == "evaluate":
            command.add_argument("--checkpoint", required=True)
    keygen = sub.add_parser("keygen", help="Generate a Gaussian owner signal; keep the file private")
    keygen.add_argument("--output", required=True)
    keygen.add_argument("--dimension", type=int, default=128)
    for name in ("embed", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--graph", required=True, help="NPZ with adj and optional mask")
        command.add_argument("--sparse", action="store_true", help="Read a SciPy CSR NPZ for whole-graph inference")
        command.add_argument("--key", required=True, help="NPY Gaussian watermark vector")
        command.add_argument("--device", default="cpu")
        if name == "embed":
            command.add_argument("--output", required=True)
            command.add_argument("--budget-ratio", type=float, default=0.001)
            command.add_argument("--alpha", type=float, default=0.1)
            command.add_argument("--candidate-seed", type=int, default=0)
        else:
            command.add_argument("--null-scores", help="NPY held-out unwatermarked scores for this owner and attack setting")
    args = parser.parse_args(argv)
    if args.command in ("prepare", "train", "evaluate"):
        config = read_config(args.config)
        if args.command == "prepare":
            from .data import prepare_banks
            result = prepare_banks(config["data"])
            print(json.dumps({k: {s: str(p) for s,p in v.items()} for k,v in result.items()}, indent=2))
        elif args.command == "train":
            from .training import train
            if args.kind:
                config["model"]["kind"] = args.kind
            print(train(config, args.seed, args.device, args.resume))
        else:
            from .evaluation import evaluate
            print(evaluate(config, args.checkpoint, args.device))
        return
    if args.command == "keygen":
        if args.dimension <= 0:
            parser.error("--dimension must be positive")
        # default_rng draws its seed from OS entropy when no seed is supplied.
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.save(handle, np.random.default_rng().standard_normal(args.dimension).astype(np.float32))
        path.chmod(0o600)
        print(path)
        return
    model, _ = load_checkpoint(args.checkpoint, args.device)
    if args.sparse:
        from scipy import sparse
        from .sparse import SparseGraph, embed_sparse, latent_sparse
        graph = SparseGraph.from_scipy(sparse.load_npz(args.graph), args.device)
    else:
        adj, mask = read_graph(args.graph, args.device)
    key = torch.as_tensor(np.load(args.key, allow_pickle=False), dtype=torch.float32, device=args.device)
    if key.shape != (model.watermark_dim,) or not torch.isfinite(key).all() or key.norm() == 0:
        parser.error(f"--key must contain one finite nonzero {model.watermark_dim}-dimensional vector")
    with torch.inference_mode():
        if args.command == "embed":
            if not math.isfinite(args.budget_ratio) or not 0 <= args.budget_ratio <= 1:
                parser.error("--budget-ratio must lie in [0, 1]")
            if not math.isfinite(args.alpha) or args.alpha <= 0:
                parser.error("--alpha must be finite and positive")
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.sparse:
                result = embed_sparse(model, graph, key, alpha=args.alpha, budget_ratio=args.budget_ratio,
                                      candidate_seed=args.candidate_seed)
                with path.open("xb") as handle:
                    sparse.save_npz(handle, result["adj"])
                counts, edges = [result["edit_counts"]], [result["edge_counts"]]
            else:
                generator = torch.Generator(device=args.device).manual_seed(args.candidate_seed)
                result = model.embed(adj, mask, key.expand(len(adj), -1), alpha=args.alpha,
                                     budget_ratio=args.budget_ratio, generator=generator)
                with path.open("xb") as handle:
                    np.savez_compressed(handle, adj=result["adj"].cpu().numpy().astype(np.uint8), mask=mask.cpu().numpy())
                counts, edges = result["edit_counts"].tolist(), result["edge_counts"].tolist()
            print(json.dumps(dict(output=str(path), edges_flipped=counts,
                                  original_edges=edges, budget_ratio=args.budget_ratio)))
        else:
            latent = latent_sparse(model, graph).unsqueeze(0) if args.sparse else model.latent(adj, mask)
            scores = (latent * key).sum(-1).cpu().numpy()
            result = dict(scores=scores.tolist())
            if args.null_scores:
                null = np.asarray(np.load(args.null_scores, allow_pickle=False)).ravel()
                if not len(null) or not np.isfinite(null).all():
                    parser.error("--null-scores must contain finite held-out scores")
                result["empirical_p_values"] = [(1 + int((null >= s).sum())) / (1 + len(null)) for s in scores]
                result["calibration_samples"] = len(null)
            print(json.dumps(result))


if __name__ == "__main__":
    main()
