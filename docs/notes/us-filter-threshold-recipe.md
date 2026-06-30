# US Filter Threshold Recipe

## Default Threshold

The calibrated US filter produces probability scores in [0, 1]. The default decision threshold is **0.5**: articles with `us_score >= 0.5` are classified as US events.

## CCA-Consumer Recall Recipe

When using the US filter as a pre-filter to protect downstream CCA classification recall, apply the following recipe:

1. **Choose a target recall level** (e.g., 0.98 = protect 98% of DoCA-matched articles).
2. **Compute `doca_recall` at multiple thresholds** using the `doca_recall()` function from `src.validation.doca_recall`.
3. **Select the largest threshold** whose `doca_recall >= target`.
4. **Record the chosen threshold and target** alongside the artifact (weights + config + calibration sidecars).

### Example

For a target `doca_recall = 0.98`:

```python
from src.validation.doca_recall import doca_recall

# Compute recall at several thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for t in thresholds:
    result = doca_recall(scored_df, threshold=t)
    print(f"threshold={t}: recall={result['recall']:.3f}, n={result['n']}")

# Select largest threshold with recall >= 0.98
# Record both the chosen threshold and the 0.98 target
```

### Rationale

- **Default 0.5**: Operates at the calibrated probability midpoint; suitable for general-purpose US event detection.
- **Recall-targeted recipe**: For CCA consumers using the US filter as a pre-filter, a lower threshold (e.g., 0.3) preserves more US events (higher recall) at the cost of including borderline cases. The recipe ensures explicit control of this tradeoff.
- **Topic-skew caveat**: The `doca_recall` diagnostic is biased toward DoCA topics (civil disobedience, strikes, etc.); see `doca_recall.DOCA_TOPIC_SKEW_CAVEAT` for details.

## References

- `src.validation.doca_recall` — Recall diagnostic over DoCA-matched articles
- `src.validation.artifact_check.reload_and_score` — Artifact triple reload (verify reproducibility)
