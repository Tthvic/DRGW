import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drgw import evaluation
from drgw.metrics import auroc, calibrate_threshold, make_link_prediction_split
from drgw.runtime import build_model


def _graph(n, extra):
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1
    for j in range(2, 2 + extra):
        adj[0, j] = adj[j, 0] = 1
    return adj


def _read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def test_macro_auc_never_pools_key_offsets():
    rows = []
    for key, pos, null in [(0, 1.0, 0.0), (1, -100.0, -101.0)]:
        rows.append(dict(dataset="fixture", budget_ratio=0.1, attack="clean", attack_rate=0,
                         key_index=key, key_seed=key, original_edge_count=10, edits=1,
                         score_pos=pos, score_null=null, score_wrongkey=null,
                         threshold=pos, wrongkey_threshold=pos, validation_null_count=32))
    per_key, macro = evaluation.summarize_scores(rows)
    assert macro[0]["macro_auroc"] == 1.0
    pooled = auroc([1, 1, 0, 0], [1, -100, 0, -101])
    assert pooled == 0.75
    assert all(row["auroc"] == 1 for row in per_key)


def test_evaluate_exports_controls_calibrates_on_validation_and_preserves_zero_budget(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    model_cfg = dict(kind="drgw", hidden_dim=8, watermark_dim=4, gin_layers=1, flow_layers=1)
    config = dict(model=model_cfg, data=dict(datasets=["fixture"]), evaluation=dict(
        keys=2, key_seed=71, batch_size=2, target_fpr=0.01,
        budget_ratios=[0.0, 0.1], alpha=0.1,
        attacks=[dict(name="clean", rate=0), dict(name="edge_flip", rate=0.3),
                 dict(name="node_delete", rate=0.3), dict(name="node_permute", rate=1.0)]))
    banks = {"fixture": {}}
    for split, size in [("train", 2), ("val", 5), ("test", 3)]:
        path = tmp_path / f"{split}.npz"
        np.savez_compressed(path, adj=np.stack([_graph(16, i + 1) for i in range(size)]))
        banks["fixture"][split] = path
    monkeypatch.setattr(evaluation, "prepare_banks", lambda _: banks)
    model = build_model(model_cfg)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(config=config, model=model.state_dict(), seed=0, global_step=0), checkpoint)
    output = evaluation.evaluate(config, checkpoint)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["output_sha256"]) == 6
    scores = _read_csv(output / "scores.csv")
    assert len(scores) == 2 * 2 * 4 * 3
    calibration = _read_csv(output / "calibration.csv")
    assert len(calibration) == 2 * 4 * 5
    for row in scores:
        if float(row["budget_ratio"]) == 0:
            assert row["score_pos"] == row["score_null"]
            assert int(row["edits"]) == 0
        # Verification under another key uses that key's own threshold.
        matching = [item for item in calibration
                    if item["key_seed"] == row["wrong_key_seed"]
                    and item["attack"] == row["attack"]
                    and item["attack_rate"] == row["attack_rate"]]
        assert float(row["wrongkey_threshold"]) == float(matching[0]["threshold"])
        owner = [item for item in calibration
                 if item["key_seed"] == row["key_seed"] and item["attack"] == row["attack"]
                 and item["attack_rate"] == row["attack_rate"]]
        expected = calibrate_threshold([float(item["score_null"]) for item in owner], 0.01)
        assert float(row["threshold"]) == expected
    summary = _read_csv(output / "summary.csv")
    zero = [row for row in summary if float(row["budget_ratio"]) == 0]
    assert all(float(row["macro_auroc"]) == 0.5 for row in zero)
    # Same checkpoint/config can rerun; changed evaluation cannot overwrite it.
    assert evaluation.evaluate(config, checkpoint) == output
    changed = {**config, "evaluation": {**config["evaluation"], "target_fpr": 0.2}}
    with pytest.raises(FileExistsError, match="different"):
        evaluation.evaluate(changed, checkpoint)


def test_rng_plans_are_stable_and_paired():
    adj = torch.from_numpy(np.stack([_graph(20, 3), _graph(20, 4)])).float()
    mask = torch.ones(adj.shape[:2], dtype=torch.bool)
    attack = dict(name="edge_flip", rate=0.3)
    first = evaluation._attack_batch(adj, mask, attack, "fixture", 7, "test", 0)
    second = evaluation._attack_batch(adj, mask, attack, "fixture", 7, "test", 0)
    assert torch.equal(first[0], second[0])
    split = [evaluation._attack_batch(adj[i:i+1], mask[i:i+1], attack, "fixture", 7, "test", i)[0]
             for i in range(2)]
    assert torch.equal(first[0], torch.cat(split))
    assert evaluation.stable_seed("ab", "c") != evaluation.stable_seed("a", "bc")


def test_utility_embeddings_only_see_heldout_training_graph():
    graph = _graph(20, 4)
    holdout_seed = evaluation.stable_seed("link_holdout", 19, 0)
    train, positives, negatives = make_link_prediction_split(graph, seed=holdout_seed)

    class CaptureModel:
        watermark_dim = 4

        def embed(self, adj, mask, key, **kwargs):
            assert np.array_equal(adj[0].numpy(), train.toarray())
            assert not adj[0, positives[:, 0], positives[:, 1]].any()
            return {"adj": adj.clone(), "edit_counts": torch.tensor([0])}

    rows = evaluation.link_prediction_utility(CaptureModel(), [graph], key_seed=7,
                                               budget_ratio=0, holdout_seed=19)
    assert rows[0]["edits"] == 0
    assert rows[0]["heldout_positives"] == len(positives)
    assert rows[0]["auc_original"] == rows[0]["auc_marked"]
