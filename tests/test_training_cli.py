"""CPU-only training/checkpoint/CLI checks; graph banks are test fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drgw.__main__ import main, read_graph
from drgw.runtime import build_model, load_checkpoint
from drgw import training


@pytest.fixture(autouse=True)
def _cpu_execution():
    torch.manual_seed(67)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(tmp_path, kind="drgw"):
    return {
        "output_dir": str(tmp_path / "runs"),
        "model": {"kind": kind, "hidden_dim": 16, "watermark_dim": 8, "gin_layers": 2, "flow_layers": 2},
        "data": {"datasets": ["SyntheticTestFixture"]},
        "training": {
            "stage1_steps": 1,
            "stage2_steps": 1,
            "stage3_steps": 1,
            "naive_steps": 2,
            "batch_size": 2,
            "lr": 0.001,
            "min_lr": 0.00001,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "alpha": 0.1,
            "edit_budget": 2,
            "augmentation_rate": 0.1,
            "robustness_rates": [0.1, 0.3],
            "feature_weight": 1.0,
            "variance_weight": 1.0,
            "orthogonality_weight": 0.1,
            "nll_weight": 5.0,
            "cycle_weight": 10.0,
            "stage3_aux_weight": 0.1,
            "log_every": 1,
            "checkpoint_every": 1,
        },
    }


def graph_fixture():
    generator = np.random.default_rng(23)
    upper = np.triu((generator.random((6, 10, 10)) < 0.35).astype(np.uint8), k=1)
    return upper + upper.transpose(0, 2, 1)


def fake_banks(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-training-bank.npz"
    np.savez_compressed(path, adj=graph_fixture())
    calls = []

    def prepare(config):
        calls.append(config)
        return {"SyntheticTestFixture": {"train": path}}

    monkeypatch.setattr(training, "prepare_banks", prepare)
    monkeypatch.setattr(training, "environment", lambda: {"test_fixture": True, "device": "cpu"})

    def seed_cpu(seed):
        torch.manual_seed(seed)
        torch.set_num_threads(1)

    monkeypatch.setattr(training, "seed_everything", seed_cpu)
    return calls


def test_stage2_freezes_encoder_but_trains_flow_and_editor_with_finite_gradients(tmp_path):
    config = tiny_config(tmp_path)
    model = build_model(config["model"])
    model.encoder.requires_grad_(False)
    before = {name: tensor.clone() for name, tensor in model.encoder.state_dict().items()}
    adj = torch.from_numpy(graph_fixture()[:2]).float()
    mask = torch.ones(adj.shape[:2], dtype=torch.bool)
    objective, parts = training.training_objective(model, adj, mask, "stage2", config["training"], step=0)
    assert torch.isfinite(objective)
    assert {"nll", "edits", "reconstruction", "cycle"}.issubset(parts)
    objective.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    for module in (model.inn, model.editor):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in model.encoder.state_dict().items())


def test_stage_transition_removes_old_encoder_gradients_before_clipping(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    fake_banks(tmp_path, monkeypatch)
    original_clip = torch.nn.utils.clip_grad_norm_
    inspected = []

    def inspect_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        frozen = [parameter for parameter in parameters if not parameter.requires_grad]
        if frozen:
            # Stage 1 has just populated encoder grads. Stage 2 must clear them
            # even though its optimizer excludes the newly frozen parameters.
            assert all(parameter.grad is None for parameter in frozen)
            active = [parameter.grad for parameter in parameters if parameter.requires_grad and parameter.grad is not None]
            assert active and all(torch.isfinite(gradient).all() for gradient in active)
            inspected.append(len(frozen))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", inspect_clip)
    training.train(config, seed=3, device="cpu")
    assert len(inspected) == 1 and inspected[0] > 0


@pytest.mark.parametrize("kind, updates", [("drgw", 3), ("naive", 2)])
def test_train_checkpoint_loads_with_weights_only_and_preserves_model(tmp_path, monkeypatch, kind, updates):
    config = tiny_config(tmp_path, kind)
    bank_calls = fake_banks(tmp_path, monkeypatch)
    checkpoint = training.train(config, seed=5, device="cpu")
    assert checkpoint.is_file() and len(bank_calls) == 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["global_step"] == updates
    assert payload["config"] == config
    assert payload["device_type"] == "cpu"
    assert payload["optimizer"]["state"]
    reloaded, loaded_payload = load_checkpoint(checkpoint, device="cpu")
    assert not reloaded.training
    assert loaded_payload["config_digest"] == payload["config_digest"]
    reference = build_model(config["model"])
    reference.load_state_dict(payload["model"])
    adj = torch.from_numpy(graph_fixture()[:2]).float()
    mask = torch.ones(adj.shape[:2], dtype=torch.bool)
    torch.testing.assert_close(reloaded.latent(adj, mask), reference.latent(adj, mask), rtol=0, atol=0)
    run = json.loads(checkpoint.with_name("run.json").read_text())
    assert run["status"] == "complete" and run["optimizer_updates"] == updates
    rows = [json.loads(line) for line in checkpoint.with_name("training.jsonl").read_text().splitlines()]
    assert len(rows) == updates
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]) for row in rows)
    with pytest.raises(FileExistsError, match="resume"):
        training.train(config, seed=5, device="cpu")


@pytest.fixture
def cli_assets(tmp_path):
    model_config = tiny_config(tmp_path)["model"]
    model = build_model(model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": model_config}, "model": model.state_dict()}, checkpoint)
    graph = tmp_path / "graph.npz"
    np.savez_compressed(graph, adj=graph_fixture()[0])
    return model, checkpoint, graph


def test_cli_generated_key_zero_budget_embed_verify_roundtrip(tmp_path, cli_assets, capsys):
    model, checkpoint, graph = cli_assets
    key_path = tmp_path / "owner-key.npy"
    main(["keygen", "--output", str(key_path), "--dimension", "8"])
    assert Path(capsys.readouterr().out.strip()) == key_path
    key = np.load(key_path, allow_pickle=False)
    assert key.shape == (8,) and np.isfinite(key).all()
    assert key_path.stat().st_mode & 0o777 == 0o600
    output_path = tmp_path / "watermarked.npz"
    shared = ["--checkpoint", str(checkpoint), "--key", str(key_path)]
    main(["embed", *shared, "--graph", str(graph), "--output", str(output_path), "--budget-ratio", "0"])
    embedded = json.loads(capsys.readouterr().out)
    assert embedded["edges_flipped"] == [0]
    assert embedded["budget_ratio"] == 0
    with np.load(graph) as source, np.load(output_path) as changed:
        assert np.array_equal(changed["adj"][0], source["adj"])
        assert changed["mask"].all()
    main(["verify", *shared, "--graph", str(output_path)])
    verified = json.loads(capsys.readouterr().out)
    adj, mask = read_graph(graph, "cpu")
    expected = (model.latent(adj, mask) * torch.from_numpy(key)).sum(-1).detach().numpy()
    np.testing.assert_allclose(verified["scores"], expected, rtol=1e-6)
    assert "empirical_p_values" not in verified
    null_path = tmp_path / "null-scores.npy"
    null = np.array([expected[0] - 1, expected[0], expected[0] + 1])
    np.save(null_path, null)
    main(["verify", *shared, "--graph", str(output_path), "--null-scores", str(null_path)])
    calibrated = json.loads(capsys.readouterr().out)
    assert calibrated["calibration_samples"] == 3
    assert calibrated["empirical_p_values"] == [0.75]


@pytest.mark.parametrize("command", ["embed", "verify"])
def test_cli_wrong_key_dimension_has_explicit_error(tmp_path, cli_assets, capsys, command):
    _, checkpoint, graph = cli_assets
    wrong_key = tmp_path / "wrong-key.npy"
    np.save(wrong_key, np.ones(9, dtype=np.float32))
    args = [command, "--checkpoint", str(checkpoint), "--graph", str(graph), "--key", str(wrong_key)]
    if command == "embed":
        args += ["--output", str(tmp_path / "must-not-exist.npz")]
    with pytest.raises(SystemExit) as raised:
        main(args)
    assert raised.value.code == 2
    assert "8-dimensional vector" in capsys.readouterr().err
    assert not (tmp_path / "must-not-exist.npz").exists()


def test_cli_reports_invalid_graph_shape_and_does_not_overwrite_private_key(tmp_path):
    graph = tmp_path / "invalid-graph.npz"
    np.savez_compressed(graph, adj=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Expected adj"):
        read_graph(graph, "cpu")
    key = tmp_path / "key.npy"
    np.save(key, np.arange(8))
    before = key.read_bytes()
    with pytest.raises(FileExistsError):
        main(["keygen", "--output", str(key), "--dimension", "8"])
    assert key.read_bytes() == before
