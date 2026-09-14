import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
_SPEC = importlib.util.spec_from_file_location("summarize_results", _SCRIPT)
summaries = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summaries)


def detection_row(auc, dataset="fixture"):
    return dict(dataset=dataset, budget_ratio=0.001, attack="clean", attack_rate=0,
                keys=2, graphs_per_key=1, macro_auroc=auc, macro_tpr=1.0,
                macro_fpr=0.0, wrongkey_acceptance=0.0,
                mean_original_edges=1000.0, mean_edits=1.0, zero_edit_fraction=0.0)


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, seed=0, kind="drgw"):
    path = root / kind / f"seed-{seed}"
    evaluation = path / "evaluation"
    evaluation.mkdir(parents=True)
    run = dict(status="complete", seed=seed, config=dict(model=dict(kind=kind),
               output_dir="/private/lijiasen/projects/DRGW/results/main"),
               environment=dict(torch="test-version", gpus=["test GPU"]))
    (path / "run.json").write_text(json.dumps(run))
    (path / "training.jsonl").write_text(json.dumps(dict(global_step=1, loss=0.5)) + "\n")
    _write_csv(evaluation / "summary.csv", [detection_row(1.0)])
    per_key, scores = [], []
    for index, pos, null in [(0, 1.0, 0.0), (1, -100.0, -101.0)]:
        per_key.append(dict(dataset="fixture", budget_ratio=0.001, attack="clean", attack_rate=0,
                            key_index=index, auroc=1.0, tpr=1.0, fpr=0.0, wrongkey_acceptance=0.0,
                            mean_original_edges=1000.0, mean_edits=1.0, zero_edit_fraction=0.0))
        scores.append(dict(key_index=index, score_pos=pos, score_null=null))
    _write_csv(evaluation / "per_key.csv", per_key)
    _write_csv(evaluation / "scores.csv", scores)
    _write_csv(evaluation / "calibration.csv", [dict(score_null=0.0, threshold=0.1)])
    fidelity = dict(dataset="fixture", budget_ratio=0.001, graphs_times_keys=2)
    for metric in summaries.FIDELITY_METRICS:
        fidelity[metric] = float("nan") if "assortativity" in metric else 0.0
        fidelity[metric + "_defined_count"] = 0 if "assortativity" in metric else 2
    _write_csv(evaluation / "fidelity_summary.csv", [fidelity])
    _write_csv(evaluation / "fidelity.csv", [dict(edges_flipped=1)])
    manifest = dict(status="complete", model_kind=kind, checkpoint_training_seed=seed,
                    checkpoint_sha256=hashlib.sha256(f"{kind}/{seed}".encode()).hexdigest(),
                    checkpoint="/private/lijiasen/projects/DRGW/checkpoint.pt",
                    evaluation_config_sha256="e" * 64, datasets=["fixture"],
                    evaluation=dict(keys=2, budget_ratios=[0.001], attacks=[dict(name="clean", rate=0)]),
                    environment=dict(torch="test-version", gpus=["test GPU"]),
                    bank_files={"fixture": {"test": dict(path="/private/lijiasen/data/test.npz", sha256="f" * 64)}},
                    output_sha256={filename: summaries._hash(evaluation / filename) for filename in summaries.REQUIRED_CSV})
    (evaluation / "manifest.json").write_text(json.dumps(manifest))
    # These must never be included in the shareable result export.
    (path / "checkpoint.pt").write_bytes(b"not a real model checkpoint")
    (path / "data.npz").write_bytes(b"not a real dataset")
    return path


def add_utility(run_dir, seed=0, kind="drgw"):
    folder = run_dir / "utility"
    folder.mkdir()
    original = json.loads((run_dir / "evaluation" / "manifest.json").read_text())
    scores = []
    for index, before, after in [(0, 0.75, 0.8 + 0.2 * seed), (1, 0.25, 0.4 + 0.2 * seed)]:
        scores.append(dict(dataset="fixture", budget_ratio=0.001, key_seed=71, holdout_base_seed=2026,
                           graph_index=index, auc_original=before, auc_marked=after, auc_change=after-before,
                           train_edges=1000, edits=1))
    _write_csv(folder / "scores.csv", scores)
    row = dict(dataset="fixture", budget_ratio=0.001, key_seed=71, holdout_base_seed=2026, graphs=2,
               auc_original=0.5, auc_marked=0.6 + 0.2 * seed, auc_change=0.1 + 0.2 * seed,
               auc_original_defined_count=2, auc_marked_defined_count=2, auc_change_defined_count=2,
               mean_train_edges=1000, mean_edits=1, zero_edit_fraction=0)
    _write_csv(folder / "summary.csv", [row])
    manifest = dict(status="complete", model_kind=kind, checkpoint_training_seed=seed,
                    checkpoint_sha256=original["checkpoint_sha256"], utility_config_sha256="u" * 64,
                    config_sha256="c" * 64, utility_protocol_version=1, datasets=["fixture"],
                    utility=dict(key_seed=71, budget_ratios=[0.001], holdout_base_seed=2026),
                    test_bank_sha256={"fixture": "f" * 64},
                    bank_files={"fixture": dict(path="/private/lijiasen/data/test.npz", sha256="f" * 64)},
                    output_sha256={name: summaries._hash(folder / name) for name in ("scores.csv", "summary.csv")})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_training_seed_sample_standard_deviation_is_not_key_or_graph_variance():
    runs = [dict(kind="drgw", seed=0, summary=[detection_row(0.2)]),
            dict(kind="drgw", seed=1, summary=[detection_row(0.8)])]
    row = summaries.aggregate(runs, summaries.DETECTION_METRICS)[0]
    assert row["macro_auroc_mean"] == 0.5
    assert math.isclose(row["macro_auroc_std"], math.sqrt(0.18))
    assert row["n_seeds"] == 2
    assert row["seeds"] == "0;1"
    assert row["macro_auroc_std"] != 0.3  # population standard deviation
    with pytest.raises(ValueError, match="Duplicate training seed"):
        summaries.aggregate([runs[0], runs[0]], summaries.DETECTION_METRICS)


def test_dataset_macro_averages_datasets_within_seed_before_std():
    runs = [dict(kind="drgw", seed=0, summary=[detection_row(0.2, "A"), detection_row(0.8, "B")]),
            dict(kind="drgw", seed=1, summary=[detection_row(0.8, "A"), detection_row(0.2, "B")])]
    row = summaries.dataset_macro(runs)[0]
    assert row["macro_auroc_mean"] == 0.5
    assert row["macro_auroc_std"] == 0


def test_export_preserves_macro_auc_and_sanitizes_only_paths(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    make_run(source, 0)
    make_run(source, 1)
    result = summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1])
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "complete"
    # Pooled raw key scores would yield 0.75; per-key macro is 1.0.
    assert report["detection"][0]["macro_auroc_mean"] == 1.0
    assert report["detection"][0]["macro_auroc_std"] == 0.0
    assert report["fidelity"][0]["assortativity_abs_change_mean"] is None
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("*.npz"))
    manifest_path = output / "raw" / "drgw" / "seed-0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "/private/lijiasen" not in manifest_path.read_text()
    assert manifest["checkpoint"] == "external/checkpoint.pt"
    assert manifest["environment"] == dict(torch="test-version", gpus=["test GPU"])
    copied = report["raw_files"]
    assert all(summaries._hash(output / item["path"]) == item["exported_sha256"] for item in copied)
    assert (output / "README.md").exists()
    assert report["utility_status"] == "unavailable"
    assert len(report["utility_missing_runs"]) == 2
    assert "Test subgraphs can overlap" in (output / "README.md").read_text()
    assert "confidence interval" in report["dependence"]


def test_missing_runs_are_errors_unless_explicitly_partial(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    make_run(source, 0)
    with pytest.raises(FileNotFoundError, match="Incomplete main experiment"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1, 2])
    assert not output.exists()
    summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1, 2], allow_partial=True)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "partial"
    assert [item["seed"] for item in report["missing_runs"]] == [1, 2]
    assert report["detection"][0]["macro_auroc_std"] is None
    assert "**partial**" in (output / "README.md").read_text()


def test_duplicate_seeds_seed_identity_and_bad_macro_cannot_be_hidden(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    run = make_run(source, 0)
    with pytest.raises(ValueError, match="distinct"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 0])
    summary_path = run / "evaluation" / "summary.csv"
    _write_csv(summary_path, [detection_row(0.75)])  # incorrectly pooled key scores
    manifest_path = run / "evaluation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["summary.csv"] = summaries._hash(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not the per-key mean"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0], allow_partial=True)
    run_json = run / "run.json"
    details = json.loads(run_json.read_text())
    details["seed"] = 99
    run_json.write_text(json.dumps(details))
    with pytest.raises(ValueError, match="Seed identity"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0], allow_partial=True)


def test_duplicate_checkpoints_do_not_count_as_independent_seeds(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    first, second = make_run(source, 0), make_run(source, 1)
    original = json.loads((first / "evaluation" / "manifest.json").read_text())
    path = second / "evaluation" / "manifest.json"
    duplicate = json.loads(path.read_text())
    duplicate["checkpoint_sha256"] = original["checkpoint_sha256"]
    path.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="Duplicate checkpoint"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1])


def test_utility_is_aggregated_separately_and_exported_with_hashes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    for seed in [0, 1]:
        add_utility(make_run(source, seed), seed)
    summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["utility_status"] == "complete"
    assert not report["utility_missing_runs"]
    assert report["detection"][0]["macro_auroc_mean"] == 1
    row = report["utility"][0]
    assert math.isclose(row["auc_marked_mean"], 0.7)
    assert math.isclose(row["auc_marked_std"], math.sqrt(0.02))
    assert math.isclose(row["auc_change_mean"], 0.2)
    assert (output / "utility.csv").exists()
    copied = output / "raw" / "drgw" / "seed-0" / "utility"
    assert (copied / "scores.csv").exists()
    assert "/private/lijiasen" not in (copied / "manifest.json").read_text()
    assert len([item for item in report["raw_files"] if "/utility/" in item["path"]]) == 6


def test_missing_utility_does_not_make_complete_detection_partial(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    add_utility(make_run(source, 0), 0)
    make_run(source, 1)
    summaries.summarize(source, output, kinds=["drgw"], seeds=[0, 1])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["utility_status"] == "partial"
    assert report["utility_available_runs"] == [dict(method="drgw", seed=0)]
    assert report["utility_missing_runs"][0]["seed"] == 1
    assert report["utility"][0]["auc_change_std"] is None


def test_corrupt_or_mismatched_utility_is_rejected(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    folder = add_utility(make_run(source, 0), 0)
    with (folder / "scores.csv").open("a") as handle:
        handle.write("corrupt\n")
    with pytest.raises(ValueError, match="Utility result checksum"):
        summaries.summarize(source, output, kinds=["drgw"], seeds=[0])
