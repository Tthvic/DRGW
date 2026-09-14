"""Benchmark completed-model sparse inference on complete source graphs.

This is a single-key execution and score diagnostic. Complete source graphs
include training-pool nodes; the output is not held-out full-graph accuracy.
No attacks, AUC, TPR, or detection thresholds are estimated here.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from drgw.data import load_dataset, sha256_file
from drgw.runtime import build_model, config_digest, environment, json_write, read_config, watermark
from drgw.sparse import SparseGraph, embed_sparse, latent_sparse


KEY_SEED = 104729
ALPHA = 0.1
BUDGET_RATIO = 0.001
def validate_final_checkpoint(path):
    """Reject partial runs; exported inference weights retain completion proof."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    kind = config["model"].get("kind", "drgw")
    training = config["training"]
    if kind == "naive":
        stage_steps = [training["naive_steps"]]
    else:
        stage_steps = [training[f"stage{index}_steps"] for index in (1, 2, 3)]
    if any(not isinstance(steps, int) or isinstance(steps, bool) or steps < 0 for steps in stage_steps) or sum(stage_steps) <= 0:
        raise ValueError("Checkpoint training plan must have nonnegative integer stage lengths and at least one update")
    expected = sum(stage_steps)
    final_index = max(index for index, steps in enumerate(stage_steps) if steps)
    final_step = stage_steps[final_index]
    if payload.get("global_step") != expected:
        raise ValueError(f"Whole-graph benchmark requires the completed {expected}-update saved training plan; partial runs are rejected")
    if payload.get("config_digest") != config_digest(config):
        raise ValueError("Checkpoint configuration digest does not match its saved configuration")
    if payload.get("inference_only"):
        origin = payload.get("training_checkpoint_sha256", "")
        if not isinstance(origin, str) or not re.fullmatch(r"[0-9a-f]{64}", origin):
            raise ValueError("Inference export must retain the original training checkpoint SHA256")
        evidence = {"type": "inference_export", "training_checkpoint_sha256": origin,
                    "completed_optimizer_updates": expected}
    else:
        run_path = path.with_name("run.json")
        if not run_path.is_file():
            raise ValueError("Training checkpoint needs an adjacent run.json confirming completion")
        run = json.loads(run_path.read_text())
        if run.get("status") != "complete" or run.get("optimizer_updates") != expected:
            raise ValueError(f"run.json must confirm status=complete and optimizer_updates={expected}")
        if run.get("config_digest") != payload["config_digest"]:
            raise ValueError("run.json and checkpoint configuration digests differ")
        if payload.get("stage_index") != final_index or payload.get("stage_step") != final_step:
            raise ValueError("Checkpoint has not reached the end of its final training stage")
        evidence = {"type": "completed_training_run", "run_json": str(run_path),
                    "run_json_sha256": sha256_file(run_path), "completed_optimizer_updates": expected}
    return payload, evidence


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_and_reset(device):
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)


def _timed(function, device):
    _sync(device)
    start = time.perf_counter()
    value = function()
    _sync(device)
    return value, time.perf_counter() - start


@torch.inference_mode()
def _one_graph(model, name, data_dir, device):
    """All large graph/tensor handles stay local and are released on return."""
    _clear_and_reset(device)
    started = time.perf_counter()
    (adj, metadata), load_seconds = _timed(lambda: load_dataset(name, data_dir, download=False), device)
    metadata_path = Path(data_dir) / "processed" / name / "metadata.json"
    graph_path = metadata_path.with_name("graph.npz")
    if not metadata_path.is_file() or not graph_path.is_file():
        raise ValueError(f"Prepared source files are missing for {name}; run dataset preparation first")
    graph_sha256 = sha256_file(graph_path)
    if graph_sha256 != metadata.get("processed_sha256"):
        raise ValueError(f"Processed graph checksum does not match source metadata for {name}")
    source = {"metadata": metadata, "metadata_sha256": sha256_file(metadata_path),
              "processed_sha256": graph_sha256, "prepared_graph": str(graph_path)}
    graph, csr_seconds = _timed(lambda: SparseGraph.from_scipy(adj, device=device), device)
    key = watermark(KEY_SEED, model.watermark_dim).to(device)
    original, original_seconds = _timed(lambda: latent_sparse(model, graph), device)
    embedded, embed_seconds = _timed(
        lambda: embed_sparse(model, graph, key, alpha=ALPHA, budget_ratio=BUDGET_RATIO,
                             candidate_seed=KEY_SEED), device,
    )
    modified, modified_seconds = _timed(lambda: latent_sparse(model, embedded["graph"]), device)
    expected = math.floor(graph.num_edges * BUDGET_RATIO)
    actual = (embedded["adj"] != graph.adj).nnz // 2
    if actual != expected or embedded["edit_counts"] != expected or embedded["requested_budget"] != expected:
        raise RuntimeError(f"Budget mismatch for {name}: expected={expected}, actual={actual}, reported={embedded['edit_counts']}")
    if (embedded["adj"] != embedded["adj"].T).nnz or embedded["adj"].diagonal().any():
        raise RuntimeError(f"Sparse editing produced an invalid undirected graph for {name}")
    original_values = original.cpu().numpy().astype(np.float64)
    modified_values = modified.cpu().numpy().astype(np.float64)
    key_values = key.cpu().numpy().astype(np.float64)
    if not np.isfinite(original_values).all() or not np.isfinite(modified_values).all():
        raise FloatingPointError(f"Non-finite latent in whole-graph benchmark for {name}")
    original_score = float(original_values @ key_values)
    modified_score = float(modified_values @ key_values)
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    row = {
        "dataset": name, "nodes": graph.num_nodes, "original_edges": graph.num_edges,
        "key_seed": KEY_SEED, "alpha": ALPHA, "budget_ratio": BUDGET_RATIO,
        "expected_edits": expected, "actual_edits": actual,
        "original_score": original_score, "modified_score": modified_score,
        "score_delta": modified_score - original_score,
        "seconds_source_load": load_seconds, "seconds_csr_to_device": csr_seconds,
        "seconds_original_latent": original_seconds, "seconds_embed": embed_seconds,
        "seconds_modified_latent": modified_seconds, "seconds_total": time.perf_counter() - started,
        "peak_allocated_GiB": peak_allocated, "peak_reserved_GiB": peak_reserved,
        "processed_sha256": graph_sha256,
    }
    del original, modified, embedded, key, graph, adj
    _clear_and_reset(device)
    return row, source


def benchmark(config, checkpoint, device="cpu", output="results/main/whole_graph"):
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; use a new benchmark output directory")
    payload, completion = validate_final_checkpoint(checkpoint)
    datasets = config["data"]["datasets"]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Benchmark datasets must be a nonempty list without duplicates")
    device = torch.device(device)
    model = build_model(payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "benchmark": "single-key whole-graph execution and score diagnostic",
        "scope": "Complete source graphs include training-pool nodes; this is not held-out full-graph accuracy.",
        "metrics_scope": "Execution time, device memory, exact edit budget, and one-key raw score change only; no accuracy or attack evaluation.",
        "checkpoint_status": "final", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "completion_evidence": completion, "checkpoint_global_step": payload["global_step"],
        "model_kind": payload["config"]["model"].get("kind", "drgw"), "training_seed": payload["seed"],
        "model_config": payload["config"]["model"], "checkpoint_config_digest": payload["config_digest"],
        "benchmark_config_digest": config_digest(config), "datasets": list(datasets),
        "device": str(device), "dtype": "float32", "environment": environment(),
        "key_seed": KEY_SEED, "key_count": 1, "alpha": ALPHA, "budget_ratio": BUDGET_RATIO,
        "candidate_seed": KEY_SEED,
        "timing": "Wall time with CUDA synchronization before and after each phase; per-dataset peak memory resets after releasing prior graph handles and emptying the cache.",
        "memory_units": "GiB = 2^30 bytes; CPU runs report 0 for CUDA allocator measurements.",
        "created_at": datetime.now(timezone.utc).isoformat(), "completed_datasets": [],
    }
    manifest_path = output / "manifest.json"
    json_write(manifest_path, manifest)
    rows, sources = [], {}
    try:
        for name in datasets:
            row, source = _one_graph(model, name, config["data"]["data_dir"], device)
            rows.append(row)
            sources[name] = source
            manifest["completed_datasets"].append(name)
            json_write(manifest_path, manifest)
            print(f"{name}: nodes={row['nodes']:,} edges={row['original_edges']:,} "
                  f"edits={row['actual_edits']} time={row['seconds_total']:.3f}s "
                  f"peak={row['peak_allocated_GiB']:.3f}GiB", flush=True)
        csv_path = output / "whole_graph.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)
        json_write(output / "sources.json", sources)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                        csv=csv_path.name, csv_sha256=sha256_file(csv_path),
                        sources="sources.json", sources_sha256=sha256_file(output / "sources.json"))
        json_write(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        json_write(manifest_path, manifest)
        raise
    finally:
        del model, payload
        _clear_and_reset(device)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/main/whole_graph")
    args = parser.parse_args(argv)
    print(benchmark(read_config(args.config), args.checkpoint, args.device, args.output))


if __name__ == "__main__":
    main()
