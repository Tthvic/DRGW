# Default experiment protocol

This document describes the sampling, training and evaluation settings in
`configs/main.yaml`. All generated outputs are stored under `results/`.

## Data and samples

The default mixture contains Facebook, DBLP, YAGO3-10, Amazon, web-Stanford, and
roadNet-TX. They represent the six categories in the paper. Preprocessing makes
simple undirected graphs: self-loops and duplicates are removed, and direction
and relation labels are discarded. Source and processed checksums are recorded.

For each source graph, a deterministic BFS traversal partitions vertices into
60% training, 20% validation, and 20% test pools. Each sample is an induced graph
selected by randomized BFS within its pool, restarting when necessary to reach
the requested size. Cross-pool edges are omitted. Train/validation/test source
vertices are disjoint; samples within one pool can overlap.

Each dataset contributes 128 training samples of 128 nodes, 32 validation samples
of 512 nodes, and 32 test samples of 512 nodes. The data seed is 2026. The model
trains on the union of the six equal-sized training banks, sampling batches of
32 with replacement. Source node IDs are retained for audits and never used as
model inputs. Topology features are recomputed from the graph being encoded.

## Model and editing

The backbone is a four-layer GIN with width 256. Separate linear heads produce
structural and carrier representations. Eight affine coupling layers use
two-layer graph conditioners; their log scales are bounded. A three-layer GELU
MLP predicts pairwise edge-presence logits, averaging the two endpoint orders.

A fresh 128-dimensional Gaussian watermark is multiplied by 0.1 and added to
the first 128 flow coordinates of every valid node. The remaining coordinates
receive no direct injection. Extraction recomputes the encoder and flow from
the candidate graph, then mean-pools the first 128 coordinates. The detection
score is their inner product with the watermark. Published integer seeds make
the experiments reproducible; they are not a production secret-key mechanism.

Editor candidates contain existing edges and sampled nonedges. The default
nonedge count matches the edge count, with at least 32 when available. An
existing edge's flip score is minus its presence logit; a nonedge's flip score
is its presence logit. The largest scores select the hard edge flips. During
training, a sigmoid relaxation around the top-k boundary supplies gradients;
its total mass is normalized to k. Forward graphs remain binary. This is a
straight-through approximation to discrete optimization.

Training uses exactly two feasible edge flips per graph. Evaluation separately
uses `floor(0.001 * |E|)` and `floor(0.01 * |E|)`, corresponding to 0.1% and 1%
of original undirected edges. There is no minimum of one evaluation edit. A
small graph may receive no watermark edits, and those samples remain in the
results. Requested and actual edit counts must accompany detection metrics.

## Training objectives

The three stages run for **1000, 2000, and 3000 optimizer updates**. They are not
whole-dataset epochs. Each stage starts a fresh AdamW optimizer with betas
`(0.9, 0.999)`, weight decay `1e-4`, and a cosine learning rate from `1e-3` to
`1e-5`. Gradient norm clipping uses a limit of 5.

1. **Encoder learning.** Invariance is MSE between the mean-pooled structural
   representations of the original graph and a graph with 10% random edge
   flips. This implements the squared-distance objective in paper Eq. 4, using
   a mean over coordinates. Orthogonality uses node vectors normalized to unit
   length, then penalizes the squared Frobenius norm of their mean cross-moment,
   weighted by 0.1. Additional feature reconstruction and variance-floor losses
   each have weight 1. Both heads reconstruct the eight topology features; the
   variance penalty encourages each active-node coordinate's standard deviation
   to reach 0.1. These auxiliary terms are implementation choices.
2. **Editor and flow initialization.** The encoder is frozen. The objective is
   candidate-edge reconstruction BCE plus `5 * NLL + 10 * cycle`. NLL is the
   node-level Gaussian flow loss per active scalar. The cycle term compares the
   edited graph's re-extracted signal with the detached injected target using
   MSE. It includes re-encoding after hard graph edits and measures a different
   property from the INN's algebraic forward/inverse round trip.
3. **Robustness training.** All modules are trainable. With
   `robust_control_variate: true`, the robustness term is the negative mean of
   `score(z_marked, w) - score(z_null, w)`. Marked and original graphs share the
   same attack seed, and both branches retain gradients. The unmarked branch
   does not depend on the fresh zero-mean Gaussian watermark, so its score and
   parameter gradient have expectation zero. Subtracting it preserves the
   expected robustness objective and its raw gradient, with the aim of reducing
   common graph-dependent sampling noise. This is a training control variate; detection
   still uses only the candidate graph and the verification key, without an
   original-graph branch. The full objective adds `5 * NLL` and `0.1` times the
   stage-one objective. Edge flips and node deletion alternate, cycling through
   rates 10%, 30%, and 50%.

The inverse flow uses the same conditioning graph and structural features as
the forward map. Verification re-encodes the candidate graph and calibrates
the pooled score using held-out null observations.

## Evaluation and control

For each dataset and edit budget, the default design evaluates 32 held-out test
graphs with four seeded Gaussian watermark keys. Validation uses a separate
32-graph bank. Each dataset, verification key, and attack has its own threshold,
calibrated on validation negatives without test labels. Each threshold therefore
has 32 null scores, not 128. Empirical FPR changes in steps of 1/32 (3.125%); the
1% target selects a threshold with zero validation accepts. Evaluation reports
test TPR and FPR separately. AUROC is calculated within each key and then
averaged over keys. Wrong-key acceptance scores each marked graph with the next
key, using that verification key's own validation threshold. Shared keys and
overlapping graphs mean the graph/key scores are not independent observations.

Conditions are clean detection; random edge flips at 10%, 30%, and 50%; random
node deletion at those same rates; and a full node permutation. Edge-flip counts
are a fraction of existing edges, while selected pairs are uniform over all
valid unordered pairs, so sparse graphs receive mostly additions. Deleted nodes
become masked padding and lose incident edges. A permutation relabels adjacency
and mask together. No owner watermark is used to choose these attacks.

The `naive` control has one GIN representation head and an adjacency decoder,
with no disentanglement or INN. It trains for 6000 updates on reconstruction,
feature and variance auxiliaries, and a `0.001` latent-L2 penalty. Watermarks are
added directly to its latent coordinates before budgeted decoding. It has no
watermark or attack training loss.

Checkpoints store model, optimizer, stage position, configuration, and PyTorch
RNG state. Use the same configuration and compatible execution device when
resuming. Retain the sampled-bank metadata, training log, evaluation predictions,
software revision, and environment with each reported result.

`scripts/evaluate_utility.py` separately runs an Adamic-Adar topology utility
probe with one fixed evaluation key and shared per-dataset edge holdouts; its
link-prediction AUROC is measured before and after watermark embedding.
