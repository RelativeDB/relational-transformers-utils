# Experimental causal discovery

`relational_transformers_utils.causal` compares information-theoretic explanations
of categorical observations. It works without a transformer or ablation runs.
The output is a **causal-model preference under assumptions**, not a contribution
score for a prediction or a measured intervention effect.

## Pairwise direction

Pass paired observations as strings, integers, or booleans. Categories are encoded
internally; continuous values and missing values must be handled explicitly by the
caller. Rows must refer to the same observations in both variables.

```python
from relational_transformers_utils.causal import entropic_direction

# Synthetic paired observations; replace with your categorical measurements.
customer_segment = [0, 1, 2, 3] * 50
purchase_category = [segment // 2 for segment in customer_segment]

result = entropic_direction(
    customer_segment,
    purchase_category,
    names=("segment", "purchase"),
    criterion="exogenous",
    bootstrap=100,
    seed=42,
)
print(result.direction)             # (cause, effect), or None for a score tie
print(result.margin_bits)           # positive favors segment -> purchase
print(result.bootstrap_fractions)   # forward, reverse, unresolved
print(result.margin_interval_bits)  # percentile 95% bootstrap interval
print(result.min_context_count)     # smallest observed category count
```

For `X -> Y`, the method fits `Y = f(X, E)` with `E` independent of `X`.
The entropy of the smallest such noise is a minimum-entropy coupling problem over
all observed conditional distributions `P(Y | X=x)`.

- `criterion="exogenous"` compares approximate `H(E)` in both directions. This is
  the paper's theoretical pairwise criterion and is the default.
- `criterion="total"` adds the marginal entropy of the proposed cause. It compares
  approximate `H(X) + H(E)` against `H(Y) + H(reverse_noise)`.

All entropies are in **bits**. `margin_bits = reverse_bits - forward_bits`.
`tolerance` defaults to `1e-9` bits; differences within it are unresolved.
`forward_noise_bits` and `reverse_noise_bits` remain available with either criterion.
In the example, the default criterion favors `segment -> purchase` by 1 bit;
`criterion="total"` ties at 2 bits in both directions. Bootstrap is disabled by
default (`bootstrap=0`), leaving both bootstrap result fields as `None`.

The score uses **greedy approximate** minimum-entropy coupling. It is an upper
bound, not the exact minimum. Comparing two upper bounds does not certify which
exact minimum is smaller. In particular, this is not ordinary conditional entropy
or mutual information. The utility also exposes `greedy_coupling_entropy(marginals)`
for callers that already have conditional distributions.

## Orient a small known skeleton

```python
from relational_transformers_utils.causal import orient_graph

customer_segment = [0, 1, 2, 3] * 50
purchase_category = [segment // 2 for segment in customer_segment]
return_category = ["kept" if category == 0 else "returned" for category in purchase_category]

result = orient_graph(
    {"segment": customer_segment, "purchase": purchase_category, "return": return_category},
    skeleton=[("segment", "purchase"), ("purchase", "return")],
    required_edges=[("purchase", "return")],
)
print(result.edges)       # directions shared by all best-scoring candidates
print(result.unresolved)  # skeleton edges whose best-scoring directions disagree
for candidate in result.candidates:
    print(candidate.edges, candidate.score_bits)
```

This implements the paper's **heuristic entropic enumeration**: enumerate acyclic
orientations consistent with the supplied skeleton and required directions, then
rank by the sum of each node's greedy noise entropy given its parents. Roots
contribute their marginal entropy. The skeleton is never learned or pruned;
conditional independence testing and the paper's peeling algorithm are outside
this initial API. Supply a scientifically justified skeleton, not arbitrary
correlations or a relational database's foreign-key graph.

Candidates are sorted by ascending score. `edges` contains only directions shared
by every candidate within `tolerance` of the minimum; no random tie-breaking is
used. `unresolved` reflects score ties, not statistical confidence or a Markov
equivalence class. Required directions constrain the result; they are not evidence
learned from the observations.

Enumeration defaults to at most 4,096 orientations (`2**unfixed_edges`, before
rejecting cycles). Conditional tables default to at most 1,000,000 entries.
Exceeding either bound raises `ValueError`. Per-node parent-set scores are cached
within a call. The graph result's `min_context_count` is the smallest observed
parent-configuration count across all scored candidates. Only observed parent
configurations are included, with no smoothing. Large sparse tables and rare
categories can make rankings unreliable even below the resource caps.

## Interpretation and assumptions

The entropic framework assumes acyclic causal structure, causal sufficiency (no
unobserved confounders), no selection bias, and low-entropy independent exogenous
noise. The paper's identifiability results also impose support and random-function
conditions, asymptotically in the number of categorical states. This experimental
implementation does not verify those assumptions or inherit an unconditional
identifiability guarantee.

Neither pairwise criterion tests independence. Independent variables with unequal
marginal entropies can produce a noise-only direction preference. Even with total
entropy, finite-sample errors can break an independence tie. A pair preference
also does not distinguish a direct effect from an ancestral relationship.

Bootstrap fractions measure stability under paired-row resampling, **not the
probability that a causal claim is true**. The bootstrap interval describes score
variation under that resampling procedure; it is not an interval for a causal
effect. Repeated observations of the same entity, time dependence, retrieval
selection, and shared context require an appropriate sampling design instead of
this IID bootstrap. For graph stability, resample appropriate units externally
and rerun `orient_graph` with the same skeleton and constraints.

Continuous transformer activations require an explicit discretization strategy;
deterministic computation alone does not establish the assumptions. Ablation
measures model sensitivity to removing context. Entropic discovery proposes
relationships among observations. Neither automatically measures real-world
intervention effects or allocates credit for a specific prediction.

## Sources and implementation reuse

- [Entropic Causal Inference: Graph Identifiability, §§3–6](https://arxiv.org/html/2509.16463v1).
- [Paper implementation](https://github.com/SpencerCompton/entropic-causality-for-graphs/tree/bc2998f43288c2421022e9b0d665097d8f91727b):
  used as a behavioral reference; its checkout has no license file, so its code
  is not vendored here.
- [ssokota/mec](https://github.com/ssokota/mec/tree/e56ad6619498914fe289d91516febb1fa3be050f):
  the NumPy greedy coupling algorithm is adapted in `_greedy_mec.py` with its MIT
  copyright and license retained. The adaptation stores only coupling masses,
  uses strict probability validation, returns entropy in bits, and omits Rust
  dispatch and Cartesian coupling storage. No additional dependency is required.
