# DRGW

PyTorch implementation of **DRGW: Learning Disentangled Representations for
Robust Graph Watermarking** (WWW 2026).

[Paper](https://doi.org/10.1145/3774904.3792543) ·
[arXiv](https://arxiv.org/abs/2601.13569) ·
[Method](docs/method.md) · [Usage](docs/usage.md)

DRGW combines a disentangled GIN encoder, a graph-aware invertible network,
and a structure-aware edge editor. A Gaussian watermark is injected into the
carrier representation and translated into budgeted edge edits. Verification
uses the candidate graph and the owner's watermark, without the original graph.

The implementation includes three-stage training, dense and sparse inference,
random edge and node attacks, and a single-carrier autoencoder baseline.

## Installation

Use Python 3.10 or later. Install [PyTorch](https://pytorch.org/get-started/locally/)
for your device, then install this package:

```bash
git clone https://github.com/Tthvic/DRGW.git
cd DRGW
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU installation; select a CUDA build from the link above for GPU training.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

## Getting started

```bash
python -m drgw prepare --config configs/main.yaml
python -m drgw train --config configs/main.yaml --seed 0 --device cuda:0
python -m drgw evaluate --config configs/main.yaml \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt --device cuda:0
```

Use `--device cpu` for CPU execution, `--resume` to continue training, or
`--kind naive` to train the autoencoder baseline. A small configuration is
provided in `configs/smoke.yaml` for checking the complete workflow.

`configs/main.yaml` defines six source datasets, node-disjoint subgraph splits,
and the optimizer-update count for each training stage. Data preparation records
source URLs, preprocessing and checksums. See the [experiment protocol](docs/protocol.md)
for sampling, attack budgets and metric definitions.

To run multiple seeds across devices:

```bash
python scripts/run_main.py --config configs/main.yaml \
  --devices cuda:0 cuda:1 --seeds 0 1 2 --kinds drgw naive
```

## Embed and verify

```bash
python -m drgw keygen --output results/owner-key.npy
python -m drgw embed \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph graph.npz --key results/owner-key.npy \
  --output results/watermarked.npz --budget-ratio 0.001 --device cuda:0
python -m drgw verify \
  --checkpoint results/main/drgw/seed-0/checkpoint.pt \
  --graph results/watermarked.npz --key results/owner-key.npy --device cuda:0
```

Dense inputs use an NPZ `adj` array. Add `--sparse` for complete graphs stored as
SciPy CSR files. See [usage](docs/usage.md) for input formats and score calibration.
Datasets, checkpoints and generated outputs stay in local, Git-ignored folders.

## Tests

```bash
python -m pytest -q
```

## Citation

```bibtex
@inproceedings{li2026drgw,
  title     = {DRGW: Learning Disentangled Representations for Robust Graph Watermarking},
  author    = {Li, Jiasen and Liu, Yanwei and Shang, Zhuoyi and Gu, Xiaoyan and Wang, Weiping},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages     = {3263--3274},
  year      = {2026},
  doi       = {10.1145/3774904.3792543}
}
```

Code is released under the [MIT license](LICENSE).
