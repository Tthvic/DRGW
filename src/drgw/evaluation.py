"""Held-out detection, wrong-key controls and graph fidelity evaluation.

Every threshold uses validation nulls for its dataset, verification key and
attack. AUROC is computed within each key before averaging: key-dependent
score offsets never count as successful discrimination.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .attacks import ATTACKS
from .data import prepare_banks, sha256_file
from .metrics import (auroc, calibrate_threshold, detection_metrics,
                      link_prediction_auc, make_link_prediction_split,
                      node_embedding_cosine, structural_fidelity)
from .runtime import config_digest, environment, json_write, load_checkpoint, watermark


def stable_seed(*parts):
    """A process-independent RNG seed from unambiguous JSON-encoded fields."""
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _generator(seed, device):
    return torch.Generator(device=device).manual_seed(seed)


def _attack_id(spec):
    return f"{spec['name']}:{float(spec.get('rate', 0)):g}"


def _score(latent, key):
    return np.sum(latent.astype(np.float64) * np.asarray(key, dtype=np.float64), axis=-1)


def _batches(total, batch_size):
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def _attack_batch(adj, mask, attack, dataset, key_seed, split, first_index):
    name, rate = attack["name"], float(attack.get("rate", 0))
    if name == "clean":
        return adj, mask
    output, masks = [], []
    for local_index in range(len(adj)):
        seed = stable_seed("attack", dataset, key_seed, split, first_index + local_index, _attack_id(attack))
        result, valid = ATTACKS[name](adj[local_index], mask[local_index], rate,
                                     generator=_generator(seed, adj.device))
        output.append(result)
        masks.append(valid)
    return torch.stack(output), torch.stack(masks)


@torch.inference_mode()
def _null_latents(model, array, attack, dataset, key_seed, split, batch_size, device):
    result = []
    for start, end in _batches(len(array), batch_size):
        adj = torch.from_numpy(array[start:end]).to(device=device, dtype=torch.float32)
        mask = torch.ones(adj.shape[:2], device=device, dtype=torch.bool)
        attacked, attacked_mask = _attack_batch(adj, mask, attack, dataset, key_seed, split, start)
        result.append(model.latent(attacked, attacked_mask).cpu().numpy())
    return np.concatenate(result)


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _atomic_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize_scores(rows):
    """Build per-key and macro summaries without pooling key score scales."""
    groups = {}
    for row in rows:
        key = (row["dataset"], row["budget_ratio"], row["attack"], row["attack_rate"], row["key_index"])
        groups.setdefault(key, []).append(row)
    per_key = []
    for (dataset, budget, attack, rate, key_index), group in groups.items():
        pos = np.asarray([r["score_pos"] for r in group])
        null = np.asarray([r["score_null"] for r in group])
        wrong = np.asarray([r["score_wrongkey"] for r in group])
        threshold = group[0]["threshold"]
        wrong_threshold = group[0]["wrongkey_threshold"]
        metrics = detection_metrics(pos, null, threshold)
        row = dict(dataset=dataset, budget_ratio=budget, attack=attack, attack_rate=rate,
                   key_index=key_index, key_seed=group[0]["key_seed"], n_graphs=len(group),
                   auroc=metrics["auroc"], tpr=metrics["tpr"], fpr=metrics["fpr"],
                   wrongkey_acceptance=float(np.mean(wrong >= wrong_threshold)),
                   threshold=threshold, wrongkey_threshold=wrong_threshold,
                   validation_null_count=group[0]["validation_null_count"],
                   mean_original_edges=float(np.mean([r["original_edge_count"] for r in group])),
                   mean_edits=float(np.mean([r["edits"] for r in group])),
                   zero_edit_fraction=float(np.mean([r["edits"] == 0 for r in group])))
        per_key.append(row)
    macro_groups = {}
    for row in per_key:
        identity = (row["dataset"], row["budget_ratio"], row["attack"], row["attack_rate"])
        macro_groups.setdefault(identity, []).append(row)
    summary = []
    for (dataset, budget, attack, rate), group in macro_groups.items():
        row = dict(dataset=dataset, budget_ratio=budget, attack=attack, attack_rate=rate,
                   keys=len(group), graphs_per_key=group[0]["n_graphs"],
                   macro_auroc=_finite_mean([r["auroc"] for r in group]),
                   auroc_key_std=float(np.std([r["auroc"] for r in group])),
                   macro_tpr=_finite_mean([r["tpr"] for r in group]),
                   macro_fpr=_finite_mean([r["fpr"] for r in group]),
                   wrongkey_acceptance=_finite_mean([r["wrongkey_acceptance"] for r in group]),
                   mean_original_edges=_finite_mean([r["mean_original_edges"] for r in group]),
                   mean_edits=_finite_mean([r["mean_edits"] for r in group]),
                   zero_edit_fraction=_finite_mean([r["zero_edit_fraction"] for r in group]),
                   validation_nulls_per_key=group[0]["validation_null_count"])
        summary.append(row)
    return per_key, summary


@torch.inference_mode()
def evaluate(config, checkpoint, device="cpu"):
    """Evaluate one checkpoint and return its ``evaluation/`` output directory.

    A different checkpoint or evaluation/data configuration cannot overwrite
    an existing evaluation directory. Identical reruns atomically replace CSVs.
    All raw scores, thresholds, per-key summaries and provenance are retained.
    """
    checkpoint = Path(checkpoint)
    cfg = config["evaluation"]
    if int(cfg["keys"]) < 2:
        raise ValueError("At least two keys are required for the wrong-key control")
    batch_size = int(cfg.get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    attacks = list(cfg["attacks"])
    identifiers = [_attack_id(attack) for attack in attacks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate evaluation attack conditions")
    for attack in attacks:
        if attack["name"] not in {"clean", *ATTACKS}:
            raise ValueError(f"Unsupported attack {attack['name']!r}")
        rate = float(attack.get("rate", 0))
        if not 0 <= rate <= 1:
            raise ValueError("Attack rates must lie in [0,1]")
    budgets = [float(value) for value in cfg["budget_ratios"]]
    if not budgets or any(not 0 <= value <= 1 for value in budgets):
        raise ValueError("budget_ratios must be a nonempty list in [0,1]")
    output = checkpoint.parent / "evaluation"
    manifest_path = output / "manifest.json"
    identity = dict(checkpoint_sha256=sha256_file(checkpoint),
                    evaluation_config_sha256=config_digest(dict(data=config["data"], evaluation=cfg)))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise FileExistsError("Evaluation directory belongs to a different checkpoint/configuration; use a separate checkpoint directory")
    model, payload = load_checkpoint(checkpoint, device)
    banks = prepare_banks(config["data"])
    key_seeds = [int(cfg["key_seed"]) + index for index in range(int(cfg["keys"]))]
    keys = [watermark(seed, model.watermark_dim).numpy() for seed in key_seeds]
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(**identity, status="running", checkpoint=str(checkpoint),
                    checkpoint_training_seed=payload.get("seed"), checkpoint_global_step=payload.get("global_step"),
                    model_kind=payload["config"]["model"].get("kind", "drgw"),
                    evaluation=cfg, datasets=config["data"]["datasets"], environment=environment(),
                    scope="sampled induced graphs from disjoint train/validation/test node pools",
                    statistic="raw inner product of graph latent and Gaussian watermark",
                    calibration="separate empirical threshold per dataset, verification key and attack; validation nulls only; acceptance >= threshold",
                    calibration_limitation="N validation nulls resolve empirical FPR only in steps of 1/N; an empirical target is not a population FPR guarantee",
                    validation_counts={},
                    auc_aggregation="within-key positive-vs-matched-null AUROC followed by unweighted key mean",
                    wrongkey_control="owner key i embedded; verification key (i+1) modulo K scored against that verification key's own validation threshold",
                    attack_randomness="same per-graph RNG seed for owner-marked and null graph; edge-flip budgets depend on each graph's own edge count",
                    edge_budget="floor(ratio * original undirected edge count), zero retained",
                    dk2="8-bin joint endpoint log-degree distribution; log(1+d)/log(N); exact discrete 2D optimal transport with normalized L1/2 ground cost",
                    undefined_metrics="CSV nan for undefined degree assortativity, with finite-only means and defined-count columns",
                    bank_files={name: {split: dict(path=str(path), sha256=sha256_file(path)) for split, path in split_paths.items()}
                                for name, split_paths in banks.items()})
    json_write(manifest_path, manifest)
    started = time.monotonic()
    score_rows, fidelity_rows, calibration_rows = [], [], []
    for dataset in config["data"]["datasets"]:
        with np.load(banks[dataset]["val"]) as bank:
            validation = bank["adj"]
        with np.load(banks[dataset]["test"]) as bank:
            test = bank["adj"]
        if not len(validation) or not len(test):
            raise ValueError("Evaluation requires nonempty validation and test banks")
        manifest["validation_counts"][dataset] = dict(null_graphs_per_key=len(validation),
                                                    minimum_nonzero_empirical_fpr=1 / len(validation))
        thresholds, null_test = {}, {}
        # Nulls do not depend on embedding budget. Evaluate once and reuse
        # across the budget grid, while retaining per-key attack RNG plans.
        for key_index, (key_seed, key) in enumerate(zip(key_seeds, keys)):
            for attack in attacks:
                attack_id = _attack_id(attack)
                val_latent = _null_latents(model, validation, attack, dataset, key_seed, "val", batch_size, device)
                val_scores = _score(val_latent, key)
                threshold = calibrate_threshold(val_scores, float(cfg["target_fpr"]))
                thresholds[key_index, attack_id] = threshold
                for graph_index, score in enumerate(val_scores):
                    calibration_rows.append(dict(dataset=dataset, key_index=key_index, key_seed=key_seed,
                                                 attack=attack["name"], attack_rate=float(attack.get("rate", 0)),
                                                 graph_index=graph_index, score_null=float(score), threshold=threshold,
                                                 empirical_acceptance=int(score >= threshold)))
                null_test[key_index, attack_id] = _null_latents(
                    model, test, attack, dataset, key_seed, "test", batch_size, device)
        for budget in budgets:
            for key_index, (key_seed, key) in enumerate(zip(key_seeds, keys)):
                wrong_index = (key_index + 1) % len(keys)
                wrong_key = keys[wrong_index]
                for start, end in _batches(len(test), batch_size):
                    adj = torch.from_numpy(test[start:end]).to(device=device, dtype=torch.float32)
                    mask = torch.ones(adj.shape[:2], dtype=torch.bool, device=device)
                    # Candidate sampling is seeded per graph and independent of
                    # batch size. Reuse the candidate RNG across owner keys and
                    # budgets so arbitrary candidate changes are not key signals.
                    embedded_list, hs_list, edits_list = [], [], []
                    for local_index in range(len(adj)):
                        candidate_seed = stable_seed("editor_candidates", dataset, int(cfg["key_seed"]), start + local_index)
                        embedded = model.embed(
                            adj[local_index:local_index + 1], mask[local_index:local_index + 1],
                            torch.from_numpy(key).to(device).unsqueeze(0), alpha=float(cfg.get("alpha", 0.1)),
                            budget_ratio=budget, generator=_generator(candidate_seed, device))
                        embedded_list.append(embedded["adj"])
                        hs_list.append(embedded["hs"])
                        edits_list.append(int(embedded["edit_counts"][0].item()))
                    marked = torch.cat(embedded_list)
                    original_hs = torch.cat(hs_list)
                    marked_hs, _ = model.encode(marked, mask)
                    marked_numpy = marked.cpu().numpy()
                    for local_index, edits in enumerate(edits_list):
                        graph_index = start + local_index
                        fidelity = structural_fidelity(test[graph_index], marked_numpy[local_index])
                        cosine = node_embedding_cosine(original_hs[local_index], marked_hs[local_index], mask[local_index])
                        fidelity_rows.append(dict(dataset=dataset, budget_ratio=budget, key_index=key_index,
                                                  key_seed=key_seed, graph_index=graph_index,
                                                  node_embedding_cosine=cosine, **fidelity))
                    for attack in attacks:
                        attack_id = _attack_id(attack)
                        attacked, attacked_mask = _attack_batch(marked, mask, attack, dataset, key_seed, "test", start)
                        pos_latent = model.latent(attacked, attacked_mask).cpu().numpy()
                        null_latent = null_test[key_index, attack_id][start:end]
                        # Identical zero-edit graphs carry no watermark signal.
                        # This also prevents batch floating-point variation from
                        # producing artificial rank differences in this case.
                        zero_edit = np.asarray(edits_list) == 0
                        pos_latent[zero_edit] = null_latent[zero_edit]
                        pos_scores, null_scores = _score(pos_latent, key), _score(null_latent, key)
                        wrong_scores = _score(pos_latent, wrong_key)
                        for local_index, edits in enumerate(edits_list):
                            graph_index = start + local_index
                            original_edges = int(test[graph_index].sum() // 2)
                            marked_edges = int(marked_numpy[local_index].sum() // 2)
                            attack_name, rate = attack["name"], float(attack.get("rate", 0))
                            scale_pos = marked_edges if attack_name == "edge_flip" else test.shape[-1]
                            scale_null = original_edges if attack_name == "edge_flip" else test.shape[-1]
                            score_rows.append(dict(
                                dataset=dataset, budget_ratio=budget, attack=attack_name, attack_rate=rate,
                                key_index=key_index, key_seed=key_seed, wrong_key_seed=key_seeds[wrong_index],
                                graph_index=graph_index, original_edge_count=original_edges,
                                marked_edge_count=marked_edges, requested_edits=math.floor(budget * original_edges), edits=edits,
                                score_pos=float(pos_scores[local_index]), score_null=float(null_scores[local_index]),
                                score_wrongkey=float(wrong_scores[local_index]),
                                threshold=thresholds[key_index, attack_id], wrongkey_threshold=thresholds[wrong_index, attack_id],
                                validation_null_count=len(validation),
                                attack_budget_pos=math.floor(rate * scale_pos) if attack_name != "clean" else 0,
                                attack_budget_null=math.floor(rate * scale_null) if attack_name != "clean" else 0,
                                attacked_valid_nodes=int(attacked_mask[local_index].sum().item()),
                                attack_seed=stable_seed("attack", dataset, key_seed, "test", graph_index, attack_id)))
            print(f"evaluation dataset={dataset} budget={budget:g} complete elapsed={time.monotonic()-started:.1f}s", flush=True)
        # Write complete datasets incrementally; manifest remains running until
        # every dataset finishes, so partial CSVs cannot be mistaken for final.
        _atomic_csv(output / "scores.csv", score_rows)
        _atomic_csv(output / "calibration.csv", calibration_rows)
        _atomic_csv(output / "fidelity.csv", fidelity_rows)
    per_key, summary = summarize_scores(score_rows)
    _atomic_csv(output / "per_key.csv", per_key)
    _atomic_csv(output / "summary.csv", summary)
    fidelity_summary = []
    for dataset in config["data"]["datasets"]:
        for budget in budgets:
            group = [row for row in fidelity_rows if row["dataset"] == dataset and row["budget_ratio"] == budget]
            row = dict(dataset=dataset, budget_ratio=budget, graphs_times_keys=len(group))
            for field in ("edges_flipped", "edges_flipped_pct", "assortativity_change", "assortativity_abs_change",
                          "transitivity_change", "transitivity_abs_change", "dk2_emd", "node_embedding_cosine"):
                row[field] = _finite_mean([item[field] for item in group])
                row[field + "_defined_count"] = int(np.isfinite([item[field] for item in group]).sum())
            fidelity_summary.append(row)
    _atomic_csv(output / "fidelity_summary.csv", fidelity_summary)
    manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                    output_sha256={path.name: sha256_file(path) for path in sorted(output.glob("*.csv"))})
    json_write(manifest_path, manifest)
    return output


@torch.inference_mode()
def link_prediction_utility(model, graphs, key_seed, budget_ratio, alpha=0.1, device="cpu", holdout_seed=2026):
    """Independent fixed-holdout Adamic-Adar utility probe, returning CSV rows.

    Positives/negatives are chosen once from each unmarked graph. Watermark
    embedding sees only its edge-heldout train adjacency. Both downstream
    evaluations remove every test pair before scoring. This helper is separate
    from detection evaluation and does not silently reuse the full test graph.
    """
    rows = []
    key = watermark(key_seed, model.watermark_dim).to(device).unsqueeze(0)
    for index, graph in enumerate(graphs):
        seed = stable_seed("link_holdout", holdout_seed, index)
        train, positive, negative = make_link_prediction_split(graph, seed=seed)
        adj = torch.from_numpy(train.toarray()).to(device=device, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(adj.shape[:2], device=device, dtype=torch.bool)
        embedded = model.embed(adj, mask, key, alpha=alpha, budget_ratio=budget_ratio,
                               generator=_generator(stable_seed("link_editor", holdout_seed, index), device))
        marked = embedded["adj"][0].cpu().numpy()
        original_auc = link_prediction_auc(train, positive, negative)
        marked_auc = link_prediction_auc(marked, positive, negative)
        rows.append(dict(graph_index=index, key_seed=key_seed, budget_ratio=budget_ratio,
                         holdout_seed=seed, heldout_positives=len(positive), heldout_negatives=len(negative),
                         train_edges=int(train.nnz // 2), edits=int(embedded["edit_counts"][0]),
                         auc_original=original_auc, auc_marked=marked_auc, auc_change=marked_auc-original_auc))
    return rows
