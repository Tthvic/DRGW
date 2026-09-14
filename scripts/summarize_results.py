#!/usr/bin/env python3
"""Validate and export DRGW main results without checkpoints or dataset files.

Run from any directory:
    python scripts/summarize_results.py --input results/main --output results/summary

Only completed runs are accepted. --allow-partial explicitly lists skipped runs;
it never permits malformed, duplicated or internally inconsistent result rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_VERSION = 1
DETECTION_METRICS = ("macro_auroc", "macro_tpr", "macro_fpr", "wrongkey_acceptance",
                     "mean_original_edges", "mean_edits", "zero_edit_fraction")
FIDELITY_METRICS = ("edges_flipped", "edges_flipped_pct", "assortativity_change",
                    "assortativity_abs_change", "transitivity_change",
                    "transitivity_abs_change", "dk2_emd", "node_embedding_cosine")
UTILITY_METRICS = ("auc_original", "auc_marked", "auc_change", "mean_train_edges", "mean_edits", "zero_edit_fraction")
REQUIRED_CSV = ("summary.csv", "fidelity_summary.csv", "scores.csv", "calibration.csv",
                "per_key.csv", "fidelity.csv")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_read(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in rows)


def _number(row, key, *, finite=True):
    try:
        value = float(row[key])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Missing/non-numeric field {key!r}") from error
    if finite and not math.isfinite(value):
        raise ValueError(f"Non-finite field {key!r}")
    return value


def _identity(row, fidelity=False):
    identity = (row["dataset"], _number(row, "budget_ratio"))
    return identity if fidelity else identity + (row["attack"], _number(row, "attack_rate"))


def _unique_conditions(rows, *, fidelity=False):
    result = {}
    for row in rows:
        identity = _identity(row, fidelity)
        if identity in result:
            raise ValueError(f"Duplicate result condition {identity}")
        result[identity] = row
    return result


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)


def validate_run(run_dir, kind, seed):
    """Read and validate one complete run, including aggregation and hashes."""
    run_dir = Path(run_dir)
    evaluation = run_dir / "evaluation"
    required = [run_dir / "run.json", run_dir / "training.jsonl", evaluation / "manifest.json"]
    required += [evaluation / filename for filename in REQUIRED_CSV]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{kind}/seed-{seed}: missing {', '.join(missing)}")
    run = json.loads((run_dir / "run.json").read_text())
    manifest = json.loads((evaluation / "manifest.json").read_text())
    if run.get("status") != "complete" or manifest.get("status") != "complete":
        raise FileNotFoundError(f"{kind}/seed-{seed}: training or evaluation is not complete")
    if run.get("seed") != seed or manifest.get("checkpoint_training_seed") != seed:
        raise ValueError(f"Seed identity mismatch in {kind}/seed-{seed}")
    if manifest.get("model_kind") != kind or run["config"]["model"].get("kind", "drgw") != kind:
        raise ValueError(f"Model identity mismatch in {kind}/seed-{seed}")
    if not manifest.get("checkpoint_sha256") or not manifest.get("evaluation_config_sha256"):
        raise ValueError(f"Missing checkpoint/configuration hash in {kind}/seed-{seed}")
    declared_hashes = manifest.get("output_sha256", {})
    for filename in REQUIRED_CSV:
        if declared_hashes.get(filename) != _hash(evaluation / filename):
            raise ValueError(f"Result checksum mismatch: {kind}/seed-{seed}/{filename}")
    summary = _csv_read(evaluation / "summary.csv")
    fidelity = _csv_read(evaluation / "fidelity_summary.csv")
    per_key = _csv_read(evaluation / "per_key.csv")
    conditions, fidelity_conditions = _unique_conditions(summary), _unique_conditions(fidelity, fidelity=True)
    settings = manifest["evaluation"]
    datasets = list(manifest["datasets"])
    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate datasets in evaluation manifest")
    expected = {(dataset, float(budget), attack["name"], float(attack.get("rate", 0)))
                for dataset in datasets for budget in settings["budget_ratios"] for attack in settings["attacks"]}
    expected_fidelity = {(dataset, float(budget)) for dataset in datasets for budget in settings["budget_ratios"]}
    if set(conditions) != expected or set(fidelity_conditions) != expected_fidelity:
        raise ValueError(f"Incomplete or unexpected condition grid in {kind}/seed-{seed}")
    keys = int(settings["keys"])
    per_key_groups = {}
    for row in per_key:
        identity, key_index = _identity(row), int(row["key_index"])
        per_key_groups.setdefault(identity, {})
        if key_index in per_key_groups[identity]:
            raise ValueError(f"Duplicate verification key row in {kind}/seed-{seed}")
        per_key_groups[identity][key_index] = row
    if set(per_key_groups) != expected:
        raise ValueError(f"Incomplete per-key result grid in {kind}/seed-{seed}")
    mappings = dict(macro_auroc="auroc", macro_tpr="tpr", macro_fpr="fpr",
                    wrongkey_acceptance="wrongkey_acceptance", mean_original_edges="mean_original_edges",
                    mean_edits="mean_edits", zero_edit_fraction="zero_edit_fraction")
    for identity, group in per_key_groups.items():
        if set(group) != set(range(keys)):
            raise ValueError(f"Missing verification keys in {kind}/seed-{seed}/{identity}")
        row = conditions[identity]
        if int(row["keys"]) != keys:
            raise ValueError("Summary key count disagrees with manifest")
        for metric, source_metric in mappings.items():
            expected_mean = statistics.mean(_number(item, source_metric) for item in group.values())
            if not _close(_number(row, metric), expected_mean):
                raise ValueError(f"Summary {metric} is not the per-key mean in {kind}/seed-{seed}/{identity}")
    for row in fidelity:
        for metric in FIDELITY_METRICS:
            _number(row, metric, finite=False)
    return dict(kind=kind, seed=seed, run_dir=run_dir, manifest=manifest, run=run,
                summary=summary, fidelity=fidelity)


def _stats(values):
    finite = [value for value in values if math.isfinite(value)]
    return (statistics.mean(finite) if finite else None,
            statistics.stdev(finite) if len(finite) >= 2 else None,
            len(finite))


def validate_utility(run):
    """Validate an optional fixed-holdout probe separately from detection."""
    path = run["run_dir"] / "utility"
    files = [path / name for name in ("scores.csv", "summary.csv", "manifest.json")]
    absent = [item.name for item in files if not item.is_file()]
    if absent:
        raise FileNotFoundError("utility missing " + ", ".join(absent))
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise FileNotFoundError("utility is not complete")
    if manifest.get("model_kind") != run["kind"] or manifest.get("checkpoint_training_seed") != run["seed"]:
        raise ValueError("Utility model/seed identity mismatch")
    if manifest.get("checkpoint_sha256") != run["manifest"]["checkpoint_sha256"]:
        raise ValueError("Utility uses a different checkpoint from detection")
    if not manifest.get("utility_config_sha256"):
        raise ValueError("Utility is missing its probe configuration hash")
    for filename in ("scores.csv", "summary.csv"):
        if manifest.get("output_sha256", {}).get(filename) != _hash(path / filename):
            raise ValueError(f"Utility result checksum mismatch: {run['kind']}/seed-{run['seed']}/{filename}")
    summary, scores = _csv_read(path / "summary.csv"), _csv_read(path / "scores.csv")
    conditions = _unique_conditions(summary, fidelity=True)
    datasets = manifest.get("datasets", run["manifest"]["datasets"])
    if list(datasets) != list(run["manifest"]["datasets"]):
        raise ValueError("Utility and detection use different datasets")
    expected_banks = {dataset: run["manifest"]["bank_files"][dataset]["test"]["sha256"] for dataset in datasets}
    if manifest.get("test_bank_sha256") != expected_banks:
        raise ValueError("Utility and detection use different test banks")
    expected = {(dataset, float(budget)) for dataset in datasets for budget in manifest["utility"]["budget_ratios"]}
    if set(conditions) != expected:
        raise ValueError("Incomplete or unexpected utility condition grid")
    groups = {}
    for row in scores:
        identity = _identity(row, fidelity=True)
        group = groups.setdefault(identity, {})
        graph_index = int(row["graph_index"])
        if graph_index in group:
            raise ValueError("Duplicate utility graph within a condition")
        group[graph_index] = row
    if set(groups) != expected:
        raise ValueError("Incomplete raw utility condition grid")
    for identity, indexed in groups.items():
        group, row = list(indexed.values()), conditions[identity]
        if int(row["graphs"]) != len(group):
            raise ValueError("Utility graph count does not match raw scores")
        if int(row["key_seed"]) != int(manifest["utility"]["key_seed"]):
            raise ValueError("Utility verification key disagrees with its manifest")
        for metric in ("auc_original", "auc_marked", "auc_change"):
            mean, _, defined = _stats([_number(item, metric, finite=False) for item in group])
            actual = _number(row, metric, finite=False)
            if (mean is None and math.isfinite(actual)) or (mean is not None and not _close(actual, mean)):
                raise ValueError(f"Utility {metric} does not match the raw-score mean")
            if int(row[metric + "_defined_count"]) != defined:
                raise ValueError(f"Utility {metric} has an incorrect defined count")
        for metric, source_field in (("mean_train_edges", "train_edges"), ("mean_edits", "edits")):
            if not _close(_number(row, metric), statistics.mean(_number(item, source_field) for item in group)):
                raise ValueError(f"Utility {metric} does not match raw graph values")
        if not _close(_number(row, "zero_edit_fraction"), statistics.mean(_number(item, "edits") == 0 for item in group)):
            raise ValueError("Utility zero-edit fraction does not match raw graph values")
    return dict(manifest=manifest, summary=summary)


def aggregate(runs, metrics, *, fidelity=False):
    """Use training seeds as independent replicates; sample standard deviation."""
    groups = {}
    identities = set()
    for run in runs:
        run_identity = (run["kind"], run["seed"])
        if run_identity in identities:
            raise ValueError(f"Duplicate training seed: {run_identity}")
        identities.add(run_identity)
        source_rows = run["fidelity"] if fidelity else run["summary"]
        _unique_conditions(source_rows, fidelity=fidelity)
        for row in source_rows:
            identity = (run["kind"],) + _identity(row, fidelity)
            groups.setdefault(identity, []).append((run["seed"], row))
    result = []
    for identity, group in sorted(groups.items()):
        kind, dataset, budget, *attack = identity
        row = dict(method=kind, dataset=dataset, budget_ratio=budget)
        if not fidelity:
            row.update(attack=attack[0], attack_rate=attack[1])
        row.update(n_seeds=len(group), seeds=";".join(str(seed) for seed, _ in sorted(group)))
        for metric in metrics:
            values = [_number(item, metric, finite=not fidelity) for _, item in group]
            mean, std, defined = _stats(values)
            row.update({f"{metric}_mean": mean, f"{metric}_std": std,
                        f"{metric}_defined_seeds": defined})
            count_field = metric + "_defined_count"
            if fidelity:
                row[count_field + "_total"] = sum(int(item.get(count_field, "0")) for _, item in group)
        result.append(row)
    return result


def dataset_macro(runs):
    """Equal-weight datasets within a training seed, then mean/std across seeds."""
    reduced = []
    for run in runs:
        groups = {}
        for row in run["summary"]:
            identity = (_number(row, "budget_ratio"), row["attack"], _number(row, "attack_rate"))
            groups.setdefault(identity, []).append(row)
        rows = []
        for (budget, attack, rate), group in groups.items():
            row = dict(dataset="all_datasets_equal_weight", budget_ratio=budget, attack=attack, attack_rate=rate)
            row.update({metric: statistics.mean(_number(item, metric) for item in group) for metric in DETECTION_METRICS})
            rows.append(row)
        reduced.append(dict(kind=run["kind"], seed=run["seed"], summary=rows))
    return aggregate(reduced, DETECTION_METRICS)


def _relative_path(value):
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return _relative_path(value)
    return value


def export_raw(runs, output):
    """Copy only public result text; normalize absolute JSON paths recursively."""
    copied = []
    for run in runs:
        source = run["run_dir"]
        destination = output / "raw" / run["kind"] / f"seed-{run['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        files = [(source / "run.json", "run.json"), (source / "training.jsonl", "training.jsonl")]
        files += [(source / "evaluation" / name, name) for name in (*REQUIRED_CSV, "manifest.json")]
        if "utility" in run:
            files += [(source / "utility" / name, f"utility/{name}")
                      for name in ("scores.csv", "summary.csv", "manifest.json")]
        for source_path, name in files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            original_hash = _hash(source_path)
            if source_path.suffix == ".json":
                content = _sanitize(json.loads(source_path.read_text()))
                content["result_export"] = dict(source_sha256=original_hash,
                                                note="Absolute path values normalized for sharing; original hashes and environment retained.")
                target.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
            elif source_path.suffix == ".jsonl":
                with target.open("w") as handle:
                    for line in source_path.read_text().splitlines():
                        if line.strip():
                            handle.write(json.dumps(_sanitize(json.loads(line)), allow_nan=False) + "\n")
            else:
                shutil.copyfile(source_path, target)
            copied.append(dict(path=target.relative_to(output).as_posix(),
                               source_sha256=original_hash, exported_sha256=_hash(target),
                               bytes=target.stat().st_size))
    return copied


def _display(mean, std, scale=1):
    if mean is None:
        return "NA"
    error = f"{std * scale:.4f}" if std is not None else "NA"
    return f"{mean * scale:.4f} ± {error}"


def write_markdown(path, report):
    lines = ["# Measured results", "",
             "Results from DRGW and the graph-autoencoder baseline on the configured sampled subgraphs. "
             "The run manifests specify the sampling protocol.", "",
             f"Status: **{report['status']}**. Completed runs: {len(report['completed_runs'])}/{len(report['expected_runs'])}.", "",
             "Values are mean ± sample standard deviation across training seeds (ddof=1). "
             "AUROC is averaged over verification keys within each dataset before aggregating seeds. "
             "NA marks undefined values or insufficient seeds.", "",
             "Test subgraphs can overlap, and the same graphs are reused across keys and training seeds. "
             "The standard deviation describes training-seed variation on this shared benchmark; "
             "it is not a confidence interval treating graphs or graph–key pairs as independent observations.", "",
             "Detection thresholds come from validation nulls for each dataset, key and attack. "
             "Finite validation banks do not establish a population FPR guarantee. "
             "Wrong-key acceptance uses that key's own threshold; zero-edit cases remain in the results.", ""]
    if report["missing_runs"]:
        lines += ["## Missing runs", ""]
        lines += [f"- `{item['method']}/seed-{item['seed']}`: {item['reason']}" for item in report["missing_runs"]]
        lines.append("")
    lines += ["## Detection without attacks", "",
              "Rates and edit budgets are fractions. Full results for every dataset and attack are in [detection.csv](detection.csv).", "",
              "| Method | Dataset | Budget | Attack | Seeds | AUROC | TPR | FPR | Wrong key | Zero edits |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["detection"]:
        if row["attack"] != "clean":
            continue
        attack = row["attack"] if row["attack"] == "clean" else f"{row['attack']} {row['attack_rate']:g}"
        cells = [row["method"], row["dataset"], f"{row['budget_ratio']:g}", attack, str(row["n_seeds"])]
        cells += [_display(row[f"{metric}_mean"], row[f"{metric}_std"])
                  for metric in ("macro_auroc", "macro_tpr", "macro_fpr", "wrongkey_acceptance", "zero_edit_fraction")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Robustness across datasets", "",
              "Each dataset has equal weight within a seed; the table then aggregates training seeds.", "",
              "| Method | Budget | Attack | Seeds | AUROC | TPR | FPR | Wrong key |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["dataset_macro"]:
        if row["attack"] == "clean":
            continue
        cells = [row["method"], f"{row['budget_ratio']:g}", f"{row['attack']} {row['attack_rate']:g}", str(row["n_seeds"])]
        cells += [_display(row[f"{metric}_mean"], row[f"{metric}_std"])
                  for metric in ("macro_auroc", "macro_tpr", "macro_fpr", "wrongkey_acceptance")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Fidelity", "",
              "Edge edits are divided by the original edge count. Assortativity and transitivity changes are absolute. "
              "dK2 is an 8-bin two-dimensional joint endpoint log-degree transport distance. "
              "Cosine uses node-aligned features from the same encoder. Undefined assortativity values are excluded "
              "from means, with defined counts retained in CSV/JSON.", "",
              "| Method | Dataset | Budget | Seeds | Edits | Edges flipped (%) | Δ assortativity | Δ transitivity | dK2 EMD | Cosine |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["fidelity"]:
        cells = [row["method"], row["dataset"], f"{row['budget_ratio']:g}", str(row["n_seeds"])]
        cells += [_display(row[f"{metric}_mean"], row[f"{metric}_std"])
                  for metric in ("edges_flipped", "edges_flipped_pct", "assortativity_abs_change",
                                 "transitivity_abs_change", "dk2_emd", "node_embedding_cosine")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Link-prediction utility", "",
              f"Probe status: **{report['utility_status']}** "
              f"({len(report['utility_available_runs'])}/{len(report['expected_runs'])} runs). "
              "This separate Adamic–Adar probe compares original and marked graphs using fixed held-out edges.", ""]
    if report["utility"]:
        lines += ["| Method | Dataset | Budget | Seeds | Original AUC | Marked AUC | AUC change |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in report["utility"]:
            cells = [row["method"], row["dataset"], f"{row['budget_ratio']:g}", str(row["n_seeds"])]
            cells += [_display(row[f"{metric}_mean"], row[f"{metric}_std"])
                      for metric in ("auc_original", "auc_marked", "auc_change")]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if report["utility_missing_runs"]:
        lines.append("Unavailable probes: " + ", ".join(
            f"`{item['method']}/seed-{item['seed']}`" for item in report["utility_missing_runs"]) + ".")
        lines.append("")
    lines += ["", "## Files", "",
              "- [Detection across seeds](detection.csv)", "- [Fidelity across seeds](fidelity.csv)",
              "- [Equal-weight macro results across datasets](dataset_macro.csv)",
              "- [Machine-readable report and export hashes](report.json)", "",
              "`raw/<method>/seed-<seed>/` contains the original score, calibration, per-key and fidelity CSVs, "
              "plus training logs and provenance. JSON path fields are normalized for sharing; the export records "
              "both original and exported file hashes. Dataset files and model checkpoints are excluded.", ""]
    if report["utility"]:
        lines += ["[Utility results across seeds](utility.csv). Raw probe files are under each run's `utility/` folder.", ""]
    path.write_text("\n".join(lines))


def summarize(input_dir, output_dir, *, kinds=("drgw", "naive"), seeds=(0, 1, 2), allow_partial=False):
    kinds, seeds = list(kinds), list(seeds)
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in {"drgw", "naive"} for kind in kinds):
        raise ValueError("Methods must be a nonempty, duplicate-free subset of drgw and naive")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input and output trees must be separate")
    runs, missing, resolved, checkpoint_hashes = [], [], set(), set()
    for kind in kinds:
        for seed in seeds:
            source = input_dir / kind / f"seed-{seed}"
            if source.resolve() in resolved:
                raise ValueError(f"Duplicate run directory for {kind}/seed-{seed}")
            resolved.add(source.resolve())
            try:
                run = validate_run(source, kind, seed)
            except FileNotFoundError as error:
                missing.append(dict(method=kind, seed=seed, reason=str(error)))
                continue
            checkpoint_hash = run["manifest"]["checkpoint_sha256"]
            if checkpoint_hash in checkpoint_hashes:
                raise ValueError(f"Duplicate checkpoint across purportedly independent runs: {kind}/seed-{seed}")
            checkpoint_hashes.add(checkpoint_hash)
            runs.append(run)
    if missing and not allow_partial:
        raise FileNotFoundError("Incomplete main experiment. " + "; ".join(item["reason"] for item in missing)
                                + ". Use --allow-partial only for an explicitly partial report.")
    if not runs:
        raise ValueError("No completed result runs to summarize")
    evaluation_hashes = {run["manifest"]["evaluation_config_sha256"] for run in runs}
    if len(evaluation_hashes) != 1:
        raise ValueError("Runs use different evaluation/data configurations and cannot be aggregated")
    detection = aggregate(runs, DETECTION_METRICS)
    fidelity = aggregate(runs, FIDELITY_METRICS, fidelity=True)
    macro = dataset_macro(runs)
    utility_runs, utility_missing = [], []
    for run in runs:
        try:
            utility = validate_utility(run)
        except FileNotFoundError as error:
            utility_missing.append(dict(method=run["kind"], seed=run["seed"], reason=str(error)))
            continue
        run["utility"] = utility
        utility_runs.append(dict(kind=run["kind"], seed=run["seed"], fidelity=utility["summary"],
                                 manifest=utility["manifest"]))
    utility_missing += [dict(method=item["method"], seed=item["seed"], reason="detector run unavailable") for item in missing]
    utility_hashes = {run["manifest"]["utility_config_sha256"] for run in utility_runs}
    if len(utility_hashes) > 1:
        raise ValueError("Utility runs use different probe/data configurations")
    utility_rows = aggregate(utility_runs, UTILITY_METRICS, fidelity=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(exporter="drgw-summarize-results", export_version=EXPORT_VERSION,
                  status="partial" if missing else "complete", input=_relative_path(str(input_dir)),
                  expected_runs=[dict(method=kind, seed=seed) for kind in kinds for seed in seeds],
                  completed_runs=[dict(method=run["kind"], seed=run["seed"],
                                       checkpoint_sha256=run["manifest"]["checkpoint_sha256"]) for run in runs],
                  missing_runs=missing, evaluation_config_sha256=next(iter(evaluation_hashes)),
                  scope="Sampled-subgraph experiments under the saved run configurations",
                  aggregation="AUROC per verification key -> mean within dataset -> mean/sample std across training seeds; dataset macro first averages datasets within each seed",
                  standard_deviation="Sample standard deviation across training seeds (ddof=1); null if fewer than two finite seeds",
                  dependence="Test subgraphs may overlap and are reused across verification keys and training seeds; no graph-level independent-sample confidence interval is reported",
                  detection=detection, fidelity=fidelity, dataset_macro=macro,
                  utility_status="unavailable" if not utility_runs else ("partial" if utility_missing else "complete"),
                  utility=utility_rows, utility_config_sha256=next(iter(utility_hashes), None),
                  utility_available_runs=[dict(method=run["kind"], seed=run["seed"]) for run in utility_runs],
                  utility_missing_runs=utility_missing,
                  raw_files=export_raw(runs, output_dir))
    _csv_write(output_dir / "detection.csv", detection)
    _csv_write(output_dir / "fidelity.csv", fidelity)
    _csv_write(output_dir / "dataset_macro.csv", macro)
    _csv_write(output_dir / "utility.csv", utility_rows)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(output_dir / "README.md", report)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "results" / "main", help="Root containing method/seed-N runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "summary", help="Shareable result export directory")
    parser.add_argument("--kinds", nargs="+", default=["drgw", "naive"], choices=["drgw", "naive"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allow-partial", action="store_true", help="Clearly label a report containing only completed runs")
    args = parser.parse_args(argv)
    try:
        output = summarize(args.input, args.output, kinds=args.kinds, seeds=args.seeds, allow_partial=args.allow_partial)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
