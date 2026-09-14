"""Offline CPU fixtures for completed-checkpoint benchmark publishing gates."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp
import torch

from drgw.runtime import build_model, config_digest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_whole_graph.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_whole_graph", _SCRIPT)
benchmarking = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmarking)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(81)
    monkeypatch.setattr(benchmarking, "environment", lambda: {"test_fixture": True, "device": "cpu"})
    yield
    torch.set_num_threads(previous)


def checkpoint_fixture(tmp_path, *, updates=6000, status="complete", exported=False):
    config = {
        "model": {"kind": "drgw", "hidden_dim": 8, "watermark_dim": 4, "gin_layers": 1, "flow_layers": 1},
        "training": {"stage1_steps": 1000, "stage2_steps": 2000, "stage3_steps": 3000, "naive_steps": 6000},
        "data": {"data_dir": str(tmp_path / "data"), "datasets": ["CliqueFixture", "CycleFixture"]},
    }
    # Deliberately tiny random model: these tests validate artifact handling,
    # not accuracy or an actual six-thousand-update training run.
    payload = {"config": config, "config_digest": config_digest(config), "seed": 0,
               "model": build_model(config["model"]).state_dict(), "global_step": updates,
               "stage_index": 2, "stage_step": updates - 3000}
    checkpoint = tmp_path / "checkpoint.pt"
    if exported:
        payload.update(inference_only=True, training_checkpoint_sha256="a" * 64)
        payload.pop("stage_index")
        payload.pop("stage_step")
    else:
        run = {"status": status, "optimizer_updates": updates, "config_digest": payload["config_digest"]}
        checkpoint.with_name("run.json").write_text(json.dumps(run))
    torch.save(payload, checkpoint)
    return config, checkpoint


def source_fixtures(config, monkeypatch):
    matrices = {
        "CliqueFixture": sp.csr_matrix(np.ones((48, 48), dtype=np.uint8) - np.eye(48, dtype=np.uint8)),
        "CycleFixture": sp.csr_matrix(np.eye(10, k=1, dtype=np.uint8) + np.eye(10, k=-1, dtype=np.uint8)),
    }
    metadata = {}
    for name, adj in matrices.items():
        directory = Path(config["data"]["data_dir"]) / "processed" / name
        directory.mkdir(parents=True)
        graph_path = directory / "graph.npz"
        sp.save_npz(graph_path, adj)
        metadata[name] = {"name": name, "nodes": adj.shape[0], "edges": adj.nnz // 2,
                          "source": {"test_fixture": True}, "raw_sha256": "b" * 64,
                          "processed_sha256": benchmarking.sha256_file(graph_path)}
        (directory / "metadata.json").write_text(json.dumps(metadata[name]))
    calls = []

    def load(name, data_dir, download):
        assert download is False
        assert str(data_dir) == config["data"]["data_dir"]
        calls.append(name)
        return matrices[name].copy(), dict(metadata[name])

    monkeypatch.setattr(benchmarking, "load_dataset", load)
    return calls, matrices


@pytest.mark.parametrize("updates,status", [(4500, "running"), (6000, "running")])
def test_partial_checkpoint_cannot_produce_a_final_benchmark(tmp_path, updates, status):
    config, checkpoint = checkpoint_fixture(tmp_path, updates=updates, status=status)
    output = tmp_path / "must-not-publish"
    with pytest.raises(ValueError, match="6000|complete"):
        benchmarking.benchmark(config, checkpoint, output=output)
    assert not output.exists()


def test_full_checkpoint_requires_matching_config_and_last_stage(tmp_path):
    _, checkpoint = checkpoint_fixture(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["stage_step"] = 2500
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="final training stage"):
        benchmarking.validate_final_checkpoint(checkpoint)
    payload["stage_step"] = 3000
    payload["config"]["model"]["watermark_dim"] = 3
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="digest"):
        benchmarking.validate_final_checkpoint(checkpoint)


def test_completed_inference_export_is_supported_without_run_json(tmp_path):
    _, checkpoint = checkpoint_fixture(tmp_path, exported=True)
    assert not checkpoint.with_name("run.json").exists()
    _, evidence = benchmarking.validate_final_checkpoint(checkpoint)
    assert evidence["type"] == "inference_export"
    assert evidence["completed_optimizer_updates"] == 6000
    payload = torch.load(checkpoint, weights_only=True)
    payload.pop("training_checkpoint_sha256")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="SHA256"):
        benchmarking.validate_final_checkpoint(checkpoint)


@pytest.mark.parametrize("steps,last_index,last_step", [([1, 1, 1], 2, 1), ([2, 1, 0], 1, 1)])
def test_completion_validation_respects_custom_training_plan(tmp_path, steps, last_index, last_step):
    _, checkpoint = checkpoint_fixture(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    for index, count in enumerate(steps, 1):
        payload["config"]["training"][f"stage{index}_steps"] = count
    payload.update(global_step=sum(steps), stage_index=last_index, stage_step=last_step)
    payload["config_digest"] = config_digest(payload["config"])
    torch.save(payload, checkpoint)
    checkpoint.with_name("run.json").write_text(json.dumps({
        "status": "complete", "optimizer_updates": sum(steps), "config_digest": payload["config_digest"],
    }))
    _, evidence = benchmarking.validate_final_checkpoint(checkpoint)
    assert evidence["completed_optimizer_updates"] == sum(steps)


def test_cpu_whole_graph_benchmark_writes_exact_budget_diagnostics_and_sources(tmp_path, monkeypatch):
    config, checkpoint = checkpoint_fixture(tmp_path)
    calls, matrices = source_fixtures(config, monkeypatch)
    output = benchmarking.benchmark(config, checkpoint, device="cpu", output=tmp_path / "benchmark")
    assert calls == config["data"]["datasets"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["checkpoint_status"] == "final"
    assert manifest["key_count"] == 1 and manifest["key_seed"] == 104729
    assert "training-pool" in manifest["scope"] and "not held-out" in manifest["scope"]
    assert manifest["completed_datasets"] == calls
    csv_path = output / manifest["csv"]
    assert manifest["csv_sha256"] == benchmarking.sha256_file(csv_path)
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert not {"auroc", "auc", "tpr", "fpr", "accuracy"} & rows[0].keys()
    assert [int(row["actual_edits"]) for row in rows] == [1, 0]
    for row in rows:
        assert int(row["expected_edits"]) == int(row["actual_edits"])
        assert int(row["original_edges"]) == matrices[row["dataset"]].nnz // 2
        assert float(row["score_delta"]) == pytest.approx(float(row["modified_score"]) - float(row["original_score"]))
        assert np.isfinite(float(row["score_delta"]))
        assert all(float(value) >= 0 for key, value in row.items() if key.startswith("seconds_"))
        assert float(row["peak_allocated_GiB"]) == 0
    sources = json.loads((output / "sources.json").read_text())
    for name in calls:
        assert len(sources[name]["metadata_sha256"]) == 64
        assert sources[name]["processed_sha256"] == sources[name]["metadata"]["processed_sha256"]
        assert sources[name]["metadata"]["raw_sha256"] == "b" * 64
    assert not list(output.rglob("*.npz"))  # Provenance is copied; graphs are not.
    with pytest.raises(FileExistsError, match="not empty"):
        benchmarking.benchmark(config, checkpoint, output=output)


def test_failed_source_validation_leaves_failed_manifest_and_no_result_csv(tmp_path, monkeypatch):
    config, checkpoint = checkpoint_fixture(tmp_path)
    source_fixtures(config, monkeypatch)
    source = Path(config["data"]["data_dir"]) / "processed" / "CliqueFixture" / "graph.npz"
    with source.open("ab") as handle:
        handle.write(b"corrupt")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="checksum"):
        benchmarking.benchmark(config, checkpoint, output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (output / "whole_graph.csv").exists()
