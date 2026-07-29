# pattern: Functional Core
"""Guard against silently overwriting production model artifacts.

Trainers and calibrators (`run_us_features.py`, `run_cca_doca.py`,
`run_relevance.py`, `calibrate_us_filter.py`, `calibrate_cca.py`,
`calibrate_relevance.py`) all take an embedding-cache suffix and a weights
output/target path as independent CLI knobs. When someone points a script at
a non-production cache (e.g. one of the `*_tuned` caches from the rel-first
sequential encoder-unfreeze arc) but leaves the weights path at its default,
the script silently trains-on-tuned-data-writes-over-production-weights: the
production `.weights.h5`, its `.config.json`, and (for calibrators) its
`.calibration.json` sidecar all get clobbered with an artifact nobody intended
to ship.

`check_no_production_overwrite` is the single guard shared by all six call
sites (three trainers + three calibrators): non-default cache suffix paired
with the production weights path is refused up front, before any I/O happens.
"""

from __future__ import annotations

from pathlib import Path


def check_no_production_overwrite(
    *,
    cache_suffix: str,
    production_cache_suffix: str,
    weights_path: Path | str,
    production_weights_path: Path | str,
    artifact_label: str,
) -> None:
    """Raise if a non-default cache suffix is paired with the production weights path.

    Args:
        cache_suffix: the embedding-cache suffix this run was invoked with.
        production_cache_suffix: the suffix that identifies the production cache
            (the script's own default, e.g. "train250k").
        weights_path: the weights path this run resolved to (after applying
            any CLI default).
        production_weights_path: the path identifying the production artifact
            for this head (e.g. `config.CCA_DOCA_WEIGHTS`).
        artifact_label: human-readable name for the error message (e.g. "CCA").

    Raises:
        ValueError: cache_suffix != production_cache_suffix AND weights_path
            resolves to production_weights_path. The two must move together:
            pointing at a non-production cache requires also pointing at a
            non-production weights path.
    """
    weights_path = Path(weights_path)
    production_weights_path = Path(production_weights_path)
    if cache_suffix != production_cache_suffix and weights_path == production_weights_path:
        raise ValueError(
            f"refusing to run {artifact_label} on cache suffix {cache_suffix!r} "
            f"(production is {production_cache_suffix!r}) while writing to the "
            f"production weights path {production_weights_path}. Pass an "
            f"explicit --out/--weights pointing at a distinct path for this "
            f"experiment (e.g. a '_tuned' suffix on the filename)."
        )
