#!/usr/bin/env python3
"""Evaluate a fixed-key Adamic-Adar utility probe on cached test subgraphs.

Relative config, checkpoint, data, and output paths use the repository root.
The downstream predictor uses Adamic-Adar scores on held-out edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from drgw.data import prepare_banks, sha256_file
from drgw.evaluation import link_prediction_utility
from drgw.runtime import config_digest, environment, json_write, load_checkpoint, read_config


PROTOCOL_VERSION = 1
HOLDOUT_BASE_SEED = 2026
HOLDOUT_DATASET_STRIDE = 1000


def write_manifest(path, payload):
    temporary = path.with_suffix(".json.tmp")
    json_write(temporary, payload)
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No utility rows to write to {path}")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["budget_ratio"]), []).append(row)
    summary = []
    for (dataset, budget), group in groups.items():
        result = dict(dataset=dataset, budget_ratio=budget,
                      key_seed=group[0]["key_seed"], holdout_base_seed=group[0]["holdout_base_seed"],
                      graphs=len(group))
        for field in ("auc_original", "auc_marked", "auc_change"):
            values = np.asarray([row[field] for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            result[field] = float(finite.mean()) if len(finite) else float("nan")
            result[field + "_defined_count"] = len(finite)
        result.update(mean_train_edges=float(np.mean([row["train_edges"] for row in group])),
                      mean_edits=float(np.mean([row["edits"] for row in group])),
                      zero_edit_fraction=float(np.mean([row["edits"] == 0 for row in group])))
        summary.append(result)
    return summary


def evaluate_utility(config, checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    cfg = config["evaluation"]
    datasets = list(config["data"]["datasets"])
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Utility evaluation requires a nonempty list of unique datasets")
    budgets = [float(value) for value in cfg["budget_ratios"]]
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in budgets)):
        raise ValueError("budget_ratios must be unique finite fractions in [0,1]")
    alpha = float(cfg.get("alpha", 0.1))
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("evaluation.alpha must be finite and nonnegative")
    key_seed = int(cfg["key_seed"])
    if key_seed < 0:
        raise ValueError("evaluation.key_seed must be nonnegative")
    utility = dict(protocol_version=PROTOCOL_VERSION, key_seed=key_seed, keys=1,
                   budget_ratios=budgets, alpha=alpha, holdout_fraction=0.1,
                   holdout_base_seed=HOLDOUT_BASE_SEED,
                   holdout_dataset_stride=HOLDOUT_DATASET_STRIDE)
    identity = dict(checkpoint_sha256=sha256_file(checkpoint), config_sha256=config_digest(config),
                    utility_config_sha256=config_digest(dict(data=config["data"], utility=utility)),
                    utility_protocol_version=PROTOCOL_VERSION)
    output = checkpoint.parent / "utility"
    manifest_path = output / "manifest.json"
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise FileExistsError("Utility directory belongs to a different checkpoint/configuration; use a separate checkpoint directory")
    elif output.exists() and any(output.glob("*.csv")):
        raise FileExistsError("Utility CSVs exist without a manifest; use a separate checkpoint directory")

    model, payload = load_checkpoint(checkpoint, device)
    banks = prepare_banks(config["data"])
    test_files = {name: banks[name]["test"] for name in datasets}
    bank_hashes = {name: sha256_file(path) for name, path in test_files.items()}
    if old is not None and old.get("test_bank_sha256") != bank_hashes:
        raise FileExistsError("Cached test banks differ from the existing utility manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        **identity, status="running", checkpoint=str(checkpoint),
        checkpoint_training_seed=payload.get("seed"), checkpoint_global_step=payload.get("global_step"),
        model_kind=payload["config"]["model"].get("kind", "drgw"),
        datasets=datasets, utility=utility, data=config["data"], environment=environment(),
        scope="Adamic-Adar topology utility probe on cached test subgraphs; one fixed key",
        statistic="AUROC on balanced fixed held-out edges and true nonedges; delta = marked AUROC minus original AUROC",
        holdout_protocol="dataset base seed = 2026 + dataset_index * 1000; helper derives a per-graph seed independent of method and training seed",
        key_protocol="first evaluation.key_seed only, shared across datasets, budgets, methods and training seeds",
        leakage_control="watermark embedding receives only the edge-heldout graph; every held-out positive and negative pair is removed from both scoring graphs",
        edge_budget="floor(ratio * edge-heldout training graph edge count); zero edits retained",
        undefined_metrics="CSV nan and per-metric finite-only means with defined-count columns; invalid graph/holdout errors fail the run",
        test_bank_sha256=bank_hashes,
        bank_files={name: dict(path=str(path), sha256=bank_hashes[name]) for name, path in test_files.items()},
    )
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    scores = []
    try:
        for dataset_index, dataset in enumerate(datasets):
            with np.load(test_files[dataset], allow_pickle=False) as bank:
                graphs = bank["adj"]
            if not len(graphs):
                raise ValueError(f"Empty test bank for {dataset}")
            dataset_seed = HOLDOUT_BASE_SEED + dataset_index * HOLDOUT_DATASET_STRIDE
            for budget in budgets:
                rows = link_prediction_utility(model, graphs, key_seed=key_seed, budget_ratio=budget,
                                               alpha=alpha, device=device, holdout_seed=dataset_seed)
                if len(rows) != len(graphs):
                    raise RuntimeError(f"Utility helper returned {len(rows)} rows for {len(graphs)} graphs in {dataset}")
                scores.extend(dict(dataset=dataset, holdout_base_seed=dataset_seed, **row) for row in rows)
                write_csv(output / "scores.csv", scores)
                write_csv(output / "summary.csv", summarize(scores))
                print(f"utility dataset={dataset} budget={budget:g} graphs={len(rows)} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        output_sha256={name: sha256_file(output / name) for name in ("scores.csv", "summary.csv")})
        write_manifest(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        elapsed_seconds=time.monotonic() - started,
                        error=f"{type(exc).__name__}: {exc}")
        write_manifest(manifest_path, manifest)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    try:
        output = evaluate_utility(read_config(Path(args.config).expanduser()),
                                  Path(args.checkpoint).expanduser(), args.device)
    except KeyboardInterrupt:
        print("Utility evaluation interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Utility evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
