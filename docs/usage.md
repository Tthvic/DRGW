# Embedding and blind verification

The detector needs the trained checkpoint, the candidate graph and the owner's
watermark vector. It does not need the original graph or source node IDs.

## Hardware and installation

Training and inference support CPU and NVIDIA CUDA GPUs. Data downloads,
SciPy graph preprocessing, subgraph sampling, and statistical metric aggregation
use CPU. When running sparse inference on a GPU, candidate-pair sampling remains
on CPU and the graph network runs on the selected GPU.

The README gives the GPU installation path. For a CPU environment, create the
same Python environment and replace its PyTorch installation command with:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Use `--device cpu` with `train`, `evaluate`, `embed` and `verify`. GPU commands
use `--device cuda:0` (or another CUDA device index). CPU execution is useful
for tests and small examples; GPU training is recommended for the main setup.
Check GPU availability after installation with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

CUDA packages require a compatible NVIDIA driver. Other supported builds are
listed in the [official installation instructions](https://pytorch.org/get-started/previous-versions/).

## Input format

Store an undirected binary adjacency matrix in an NPZ file with an `adj` array
of shape `[N, N]` (one graph) or `[B, N, N]` (a batch). Diagonal entries must be
zero. An optional boolean `mask` identifies valid nodes in padded batches.
Node features are computed from the supplied adjacency; external attributes
are not used by this implementation.

For example, export one already prepared test sample:

```python
import numpy as np
from drgw.data import prepare_banks
from drgw.runtime import read_config

banks = prepare_banks(read_config("configs/main.yaml")["data"])
with np.load(banks["Facebook"]["test"]) as bank:
    np.savez_compressed("results/example.npz", adj=bank["adj"][0])
```

## Create a watermark and embed it

```bash
python -m drgw keygen --output results/owner-key.npy --dimension 128
python -m drgw embed \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph results/example.npz --key results/owner-key.npy \
  --output results/watermarked.npz --budget-ratio 0.001 --device cuda:0
```

Keep `owner-key.npy` private. The reproducible integer seeds in the experiments
are public test signals. `keygen` instead samples a fresh Gaussian vector using
OS-provided entropy. Key and output creation refuse to overwrite existing files.

The reported edit count is `floor(original_edges * budget_ratio)`. In particular,
0.1% permits zero changes on a graph with fewer than 1,000 undirected edges.
The same trained encoder supports varying graph sizes, although generalization
outside the training distribution must be evaluated.

## Verify

```bash
python -m drgw verify \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph results/watermarked.npz --key results/owner-key.npy --device cuda:0
```

This returns the raw matched-filter score. A large raw score alone is not a
calibrated ownership decision. To obtain an empirical one-sided p-value, collect
scores from held-out, unwatermarked graphs for the same owner key and deployment
condition, save them as a one-dimensional NPY array, and add:

```bash
--null-scores results/heldout-null-scores.npy
```

The reported p-value is `(1 + count(null >= observed)) / (1 + n_null)`.
It requires exchangeable null and candidate graphs for statistical calibration;
its smallest possible value is `1 / (n_null + 1)`. The experiment runner saves
the separate validation thresholds and raw calibration scores it actually uses.
It does not infer a Gaussian graph-level null from the node-level INN loss.

## Sparse whole graphs

For larger graphs, use a SciPy CSR file (`scipy.sparse.save_npz`) and add
`--sparse` to both commands. Prepared complete datasets already use this format:

```bash
python -m drgw embed --sparse \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph data/processed/roadNet-TX/graph.npz --key results/owner-key.npy \
  --output results/road-watermarked.npz --device cuda:0
python -m drgw verify --sparse \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph results/road-watermarked.npz --key results/owner-key.npy --device cuda:0
```

Sparse inference uses the same learned parameters, CSR graph products, chunked
node networks and global Top-k over streamed candidate-edge scores. It stores
`O(ND + E)` data rather than an `N × N` adjacency matrix. Dense and sparse
nonedge samplers use different RNG implementations; their sampling distribution
is the same, but a shared integer seed need not produce identical edits.
Applying a subgraph-trained checkpoint to a complete source graph is a transfer
experiment. Successful execution does not establish full-graph detection accuracy.

## Experiment commands

Run multiple training seeds across available GPUs:

```bash
python scripts/run_main.py --config configs/main.yaml \
  --devices cuda:0 cuda:1 --seeds 0 1 2 --kinds drgw naive
```

Train the single-carrier autoencoder baseline with `--kind naive`. The launcher
supports `--resume`, `--train-only`, and `--evaluate-only`.

Run the separate link-prediction utility probe and summarize completed runs:

```bash
python scripts/evaluate_utility.py --config configs/main.yaml \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt --device cuda:0
python scripts/summarize_results.py --input results/main --output results/summary
```

For development checks:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
