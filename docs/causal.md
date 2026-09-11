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

## Select feature groups and join paths

`CausalFeatureSelector` uses a graph to propose subsets and a validation metric to
choose between them. Each named group can represent columns, cells, or an entire
join path. Selection is for prediction, not for identifying an intervention target
or a valid causal adjustment set.

For `stock_shortage -> fulfillment_delay -> cancellation`, inventory information
may be available at order creation while the future fulfillment delay is not.
Declare availability explicitly; the selector cannot infer event-time availability
from column names or audit your retrieval query.

```python
from relational_transformers_utils.causal import CausalFeatureSelector, orient_graph

# Fit this graph on a discovery/training partition, separate from validation.
# Historical outcomes may be used for discovery, but not as prediction inputs.
graph = orient_graph(
    discovery_columns,  # stock_shortage, fulfillment_delay, cancellation, region
    skeleton=[("stock_shortage", "fulfillment_delay"),
              ("fulfillment_delay", "cancellation")],
)
selector = CausalFeatureSelector(
    graph,
    target="cancellation",
    feature_groups={
        "inventory_join": ["stock_shortage"],
        "order_delay": ["fulfillment_delay"],
        "customer": ["region"],
    },
    # At order creation: delay is unavailable. It will never reach the evaluator.
    available_groups=["inventory_join", "customer"],
    greater_is_better=False,  # Brier score: smaller is better
    score_tolerance=0.002,   # prefer fewer groups within 0.002 of the best score
    max_evaluations=32,
)
```

Provide `evaluate(selected_groups) -> float`. This callback builds the selected
contexts, runs the model, and scores predictions on a **fixed validation set**.
The selector neither retrains the model nor accesses labels itself. For fixed cell
positions in existing `RelationalBatch` inputs:

```python
from relational_transformers_utils.metrics import brier_score

# Illustrative positions only: adapt them to your batch's actual cell layout.
# Target cells and mandatory structural context must not be listed here.
group_positions = {
    "inventory_join": [1, 2],
    "order_delay": [3],
    "customer": [4],
}
all_positions = {p for positions in group_positions.values() for p in positions}

def evaluate(selected_groups):
    keep = {p for name in selected_groups for p in group_positions[name]}
    remove = sorted(all_positions - keep)
    inputs = [batch.ablate(remove) if remove else batch for batch in validation_inputs]
    predictions = model.predict(inputs)
    return brier_score(predictions, validation_labels)

result = selector.select(evaluate)
print(result.selected_groups)
print(result.validation_score)
print(result.excluded_unavailable)
print(result.budget_exhausted)
for trial in result.evaluations:
    print(trial.groups, trial.score, trial.proposal)
```

For variable-length rows or retrieval-based contexts, implement the callback by
constructing each context from the selected join paths. The returned group names
are the selection plan; applying that plan to SQL, retrieval, encoding, or future
batches remains the caller's responsibility. An unavailable group must be removed
from every evaluated context, not merely ignored when scoring. A group containing
both available and unavailable columns must be split or excluded as a whole.
Overlapping groups are allowed: retain the union of selected cells/joins.

### Selection rules

1. Evaluate the empty-group baseline and all available groups.
2. Evaluate available groups intersecting target parents and target ancestors.
   Use the **union across candidate graphs** within `graph_tolerance` bits of the
   minimum, rather than arbitrarily choosing one tied orientation. The selector
   reads `graph.candidates`; it does not interpret `graph.edges` as the only
   possible graph. Its graph tolerance is independent of the tolerance originally
   used by `orient_graph`.
3. Prefer fewer groups among evaluated subsets within `score_tolerance` of the
   best validation metric. Equal-size ties are resolved alphabetically.
4. Try single-group deletions from that preferred subset and repeat while the
   preferred subset changes. Stop when no progress is possible or the evaluation
   budget is exhausted. Each distinct subset is evaluated once per `select` call.

`score_tolerance` is always measured from the **best score observed across all
trials**, so allowed losses cannot accumulate across removal steps. Set it to zero
for strict best-score selection. `max_evaluations` must cover the distinct initial
proposals; otherwise construction raises `ValueError` before running a callback.
If the budget prevents remaining deletions, `budget_exhausted` is true. Metrics
must be finite scalars; undefined AUROC on a single-class validation set raises
an error rather than silently influencing the selection.

The empty subset still contains mandatory context that the callback preserves.
The full baseline deliberately permits available non-ancestor features: a flawed
causal graph must not automatically exclude a useful predictor. The algorithm is
a bounded greedy search, not exhaustive subset optimization; feature interactions
can create better subsets it never visits. Group count is a simplicity preference,
not a measure of join cost, latency, or number of cells.

Use discovery/training data for graph fitting, validation data for subset selection,
and untouched test data for the final performance estimate. Repeated evaluation
can overfit a validation set. Keep entity/time splits and retrieval cutoffs intact.
Ablating a fixed model can create unusual contexts; if deployment involves fitting
a new head for each subset, the callback must fit on training data only and then
score on the same validation split. Causal scores guide proposals; they are never
used as feature-importance scores or substituted for validation performance.

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
