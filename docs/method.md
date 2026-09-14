# Method and implementation

DRGW maps graph structure to a continuous carrier, injects a watermark, and
uses a learned editor to return to a discrete graph. The detector reuses the
encoder and flow with frozen weights.

## Encoder

The default encoder contains four GIN layers with hidden dimension 256 and two
256-dimensional projection heads: structural features `h_s` and carrier
features `h_w`. Its inputs are eight topology features computed from the current
graph: a constant, degree, log-degree, two neighborhood-degree summaries,
two-step walk density, graph density and a bounded graph-size feature.

Structural invariance uses the squared distance between mean-pooled structural
representations of original and augmented graphs. The orthogonality objective
is the squared cross-moment of L2-normalized active structural and carrier
features. Configurable feature reconstruction and variance terms preserve
information in the encoder heads.

## Graph-aware invertible network

Eight affine coupling layers alternately transform half of the carrier
coordinates. Each layer predicts a bounded log-scale and a translation using
a two-layer GCN conditioned on the unchanged coordinates and `h_s`. The
conditional forward map has an analytic inverse and log determinant.

Training minimizes a Gaussian node-density negative log-likelihood, reduced
over active node coordinates. The detector averages node latents and uses the
first 128 coordinates as its graph-level carrier.

## Watermark and editor

The owner samples `w ~ N(0, I)` in 128 dimensions. The embedding strength is
`alpha = 0.1` by default. The same `alpha * w` is added to the first 128 latent
coordinates at each valid node, followed by the inverse flow.

The three-layer GELU editor scores unordered node pairs using structural and
modified carrier features. Endpoint-order logits are averaged for symmetry.
Candidates comprise existing edges and a random sample of nonedges. The
adjacency decoder's logit is multiplied by `1 - 2A_uv` to obtain a flip utility;
the global Top-k utilities select additions and deletions through XOR.

For a fractional budget, `k = floor(budget_ratio * original_edges)`. Zero is a
valid budget. Training can instead specify an integer edit count. A normalized
sigmoid straight-through estimator provides gradients while the forward graph
remains binary, symmetric and within budget.

## Training and verification

Training first fits the encoder, then freezes it while fitting the flow and
editor with reconstruction, density and latent-cycle losses. The final stage
jointly trains the model under graph attacks. Its matched-filter objective can
subtract an unmarked-graph control variate with the same attack randomness;
the subtracted term has zero expectation over independent zero-mean keys.

Verification extracts the graph latent from the candidate graph and computes
its inner product with the owner's watermark. Validation null scores determine
the empirical detection threshold. The optional command-line p-value uses the
finite-sample rank of the candidate score among held-out null scores.

## Sparse inference

The CSR inference path shares all learned parameters with the dense model.
It uses sparse matrix products and chunked node networks, and retains a global
Top-k while streaming candidate-edge scores. Storage scales with `ND + E`.
The sparse and dense nonedge samplers use the same distribution with different
RNG implementations. See the [protocol](protocol.md) for data and evaluation
configuration and [usage](usage.md) for commands.
