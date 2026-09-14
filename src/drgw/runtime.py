"""Small shared helpers for configuration, checkpointing and provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
import yaml


def read_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def watermark(key, dimension):
    """Stable public integer seed -> N(0,I) test signal (not a secret-key KDF)."""
    return torch.from_numpy(np.random.default_rng(int(key)).standard_normal(dimension).astype(np.float32))


def build_model(config):
    kwargs = dict(config)
    kind = kwargs.pop("kind", "drgw")
    if kind == "drgw":
        from .model import DRGW
        return DRGW(**kwargs)
    if kind == "naive":
        from .baseline import NaiveLatent
        kwargs.pop("flow_layers", None)
        return NaiveLatent(**kwargs)
    raise ValueError(f"Unknown model kind {kind!r}")


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def environment():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, cuda=torch.version.cuda, git_revision=revision,
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def load_checkpoint(path, device="cpu"):
    # Checkpoints contain state dictionaries and primitive metadata only.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    payload = torch.load(path, map_location=device, weights_only=True)
    model = build_model(payload["config"]["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
