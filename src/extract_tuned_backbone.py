# pattern: Imperative Shell (default_out_path / layer_diff_summary /
#   expected_tuned_groups / _group_sort_key are the pure Functional Core;
#   everything else -- model reconstruction, weight load/save, verification
#   against the original DAPT backbone -- is I/O)
"""
Extract a fine-tuned RoBERTa backbone out of a text-mode rel training artifact.

Part of the "rel-first sequential" encoder-unfreeze plan
(`docs/notes/encoder-unfreeze-strategy.md`): after text-mode training tunes
the shared DAPT backbone via the rel head's loss (top-N unfreeze,
discriminative LR), the tuned backbone needs to stand alone as a
`load_dapt_backbone`-compatible `.weights.h5` file so `embed_corpus.py` can
re-embed the corpora with it (see its `--backbone-weights` flag).

Reconstruction mirrors `scripts/eval_rel_text_artifact.py:score_text_model`'s
Pattern-2 approach: rebuild the exact model structure the artifact was saved
from (fresh DAPT backbone + a `rel`-named `ClassificationHead`, sized from the
artifact's `cca_config.RunConfig` sidecar), then `load_weights` over it by
structure. The backbone instance is shared in-process with the head (Pattern
A -- see `docs/notes/engineering-patterns.md`), so once `load_weights` runs,
`backbone.save_weights(...)` persists exactly the fine-tuned encoder.

Verification reloads the saved file FRESH via `load_dapt_backbone` (proving
the save/load round-trip, not just the in-process object) and diffs it
against a fresh original DAPT backbone, per top-level backbone group
("embeddings", "embeddings_layer_norm", "transformer_layer_0" .. "_11").

**Empirical finding (2026-07-29), documented because it contradicts the
naive "only the multiplier=1 layers move" mental model**: the escalation
path used to train `job8823087` sets `freeze_encoder=False` and relies
entirely on `LayerLRModel`'s per-variable *gradient* multiplier (0.0 for the
"encoder_frozen" group) to hold non-tuned layers still -- `backbone.trainable`
stays `True` for ALL backbone variables, "frozen" ones included. But Keras's
`AdamW.apply_gradients` calls `_apply_weight_decay` on every trainable
variable UNCONDITIONALLY (`var -= lr * weight_decay * var`), independent of
that variable's gradient value -- this codebase never registers an
`exclude_from_weight_decay` list. So "frozen" (multiplier=0-gradient) layers
still shrink slightly every step from decoupled weight decay, and are NOT
bit-identical to the original DAPT backbone. They're just *orders of
magnitude* smaller than the deliberately-tuned top-N layer(s)' movement
(which get a full gradient signal plus the same decay). Verification below
checks that ordering (tuned >> frozen), not exact-zero equality on the frozen
side -- see `_verify_expected_groups`.

Usage (from project root):
    uv run python -m src.extract_tuned_backbone
    uv run python -m src.extract_tuned_backbone --weights relevance/relevance_text.job8823087.weights.h5 \
        --out relevance/tuned_backbone.job8823087.weights.h5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import src.cca_config as cca_config
import src.config as config
from src.model_setup.assembly import build_inference_model
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead

DEFAULT_WEIGHTS = config.PROJECT_ROOT / "relevance" / "relevance_text.job8823087.weights.h5"
# "Clearly moved" vs. "weight-decay noise floor": the ratio the smallest
# expected-tuned-group delta must exceed the largest non-tuned-group delta by.
# Not a physical constant -- a sanity margin chosen well above the ~1-2 orders
# of magnitude weight-decay drift observed empirically (see module docstring),
# so a real bug (e.g. loading the wrong sidecar) would trip it.
_MIN_TUNED_TO_FROZEN_RATIO = 10.0


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def default_out_path(weights_path: Path) -> Path:
    """Pure: derive `tuned_backbone.<jobtag>.weights.h5` from an artifact path.

    Expects a filename of the form `<name>.<jobtag>.weights.h5` (e.g.
    `relevance_text.job8823087.weights.h5` -> jobtag `job8823087`, out path
    `tuned_backbone.job8823087.weights.h5` in the same directory). Raises
    `ValueError` if the filename doesn't carry the expected `.weights.h5`
    suffix plus a `.`-delimited jobtag segment before it.
    """
    weights_path = Path(weights_path)
    name = weights_path.name
    suffix = ".weights.h5"
    if not name.endswith(suffix):
        raise ValueError(
            f"expected a '*{suffix}' filename to derive a jobtag from; got {name!r}"
        )
    stem = name[: -len(suffix)]
    if "." not in stem:
        raise ValueError(
            f"expected '<name>.<jobtag>{suffix}'; got {name!r} (no jobtag segment "
            f"before {suffix})"
        )
    jobtag = stem.rsplit(".", 1)[-1]
    if not jobtag:
        raise ValueError(f"empty jobtag segment parsed from {name!r}")
    return weights_path.parent / f"tuned_backbone.{jobtag}.weights.h5"


def layer_diff_summary(
    paths: list[str], weights_a: list[np.ndarray], weights_b: list[np.ndarray]
) -> dict[str, float]:
    """Pure: max |delta| per top-level backbone group between two weight lists.

    `paths` / `weights_a` / `weights_b` must be parallel (same order, same
    variable identity) -- e.g. `[w.path for w in backbone.weights]` and the
    matching `.numpy()` arrays for two backbones built from the same
    `keras_hub.models.Backbone.from_preset(...)` call, so structure/order is
    guaranteed identical between the two. Groups by the first `/`-delimited
    path segment (`"embeddings"`, `"embeddings_layer_norm"`,
    `"transformer_layer_0"` .. `"transformer_layer_11"` for `roberta_base_en`).
    """
    if not (len(paths) == len(weights_a) == len(weights_b)):
        raise ValueError(
            f"paths/weights_a/weights_b must be equal length; got "
            f"{len(paths)}/{len(weights_a)}/{len(weights_b)}"
        )
    if not paths:
        raise ValueError("paths must be non-empty")
    out: dict[str, float] = {}
    for path, a, b in zip(paths, weights_a, weights_b):
        group = path.split("/")[0]
        delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        out[group] = max(out.get(group, 0.0), delta)
    return out


def expected_tuned_groups(unfreeze_top_n: int, n_layers: int = 12) -> set[str]:
    """Pure: the `transformer_layer_*` group names the escalation path unfreezes.

    Mirrors `src.validation.escalation.top_n_group_fn`'s top-index selection
    (top_indices = n_layers-1 down to n_layers-unfreeze_top_n) without needing
    a `Variable`-shaped input -- this module only has group names, not
    `Variable` objects, once weights are loaded as plain numpy arrays.
    """
    if not (0 <= unfreeze_top_n <= n_layers):
        raise ValueError(f"unfreeze_top_n must be in [0, {n_layers}]; got {unfreeze_top_n}")
    return {f"transformer_layer_{n_layers - 1 - i}" for i in range(unfreeze_top_n)}


def _group_sort_key(group: str) -> tuple[int, str]:
    """Pure: sort embeddings groups first, then transformer_layer_N numerically."""
    m = re.fullmatch(r"transformer_layer_(\d+)", group)
    if m:
        return (1, f"{int(m.group(1)):03d}")
    return (0, group)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def build_tuned_backbone(weights_path: Path):
    """Rebuild the text-mode rel artifact's model structure; return the
    (now fine-tuned, in-process) backbone plus its `RunConfig` sidecar.

    Pattern-2 reconstruction (fresh backbone + a fresh `rel`-named head sized
    from the sidecar), same shape as `eval_rel_text_artifact.score_text_model`.
    The backbone starts from the DAPT weights (`config.DAPT_BACKBONE_WEIGHTS`
    -- the platform-resolved constant, not the sidecar's possibly-cluster-only
    `backbone_weights_path`, same reasoning as `score_text_model`), and
    `inference_model.load_weights` then overwrites it in place with the
    artifact's fine-tuned values (Pattern A in-process sharing).
    """
    weights_path = Path(weights_path)
    run_config = cca_config.RunConfig.from_json(
        cca_config.config_path_for_weights(weights_path)
    )
    head_cfg = run_config.heads[0]

    backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)
    run_config.validate_against_backbone(backbone)

    head = ClassificationHead(hidden_dim=head_cfg.hidden_dim, name=head_cfg.name)
    inference_model = build_inference_model(
        backbone=backbone, heads={head_cfg.name: head}, seq_length=run_config.seq_length
    )
    inference_model.load_weights(str(weights_path), skip_mismatch=False)
    return backbone, run_config


def _verify_expected_groups(summary: dict[str, float], unfreeze_top_n: int) -> None:
    """Raise if the diff summary doesn't show tuned-group movement clearly
    separated from non-tuned-group movement.

    Does NOT require non-tuned groups to be exactly 0.0 -- see the module
    docstring's weight-decay finding. Instead requires: every expected-tuned
    group's delta is non-zero, and the SMALLEST tuned delta exceeds the
    LARGEST non-tuned delta by at least `_MIN_TUNED_TO_FROZEN_RATIO`.
    """
    tuned_groups = expected_tuned_groups(unfreeze_top_n)
    missing = tuned_groups - summary.keys()
    if missing:
        raise ValueError(f"expected tuned groups not present in summary: {sorted(missing)}")
    frozen_groups = set(summary) - tuned_groups

    tuned_deltas = [summary[g] for g in tuned_groups]
    if any(d <= 0.0 for d in tuned_deltas):
        raise ValueError(
            f"expected tuned group(s) {sorted(tuned_groups)} show zero movement "
            f"vs. original DAPT -- fine-tuning may not have touched the backbone: "
            f"{summary}"
        )
    if not frozen_groups:
        return  # unfreeze_top_n == n_layers: nothing left to compare against.

    min_tuned = min(tuned_deltas)
    max_frozen = max(summary[g] for g in frozen_groups)
    if max_frozen > 0.0 and min_tuned < _MIN_TUNED_TO_FROZEN_RATIO * max_frozen:
        raise ValueError(
            f"tuned-group movement ({min_tuned:.3e}) is not clearly separated "
            f"from non-tuned-group movement ({max_frozen:.3e}, ratio "
            f"{min_tuned / max_frozen:.1f}x < required {_MIN_TUNED_TO_FROZEN_RATIO}x) "
            f"-- unexpected. Full summary: {summary}"
        )


def extract_and_verify(weights_path: Path, out_path: Path) -> dict:
    """Extract the fine-tuned backbone from `weights_path`, save it to
    `out_path`, then verify by a FRESH reload against the original DAPT
    backbone. Prints a per-group max-|delta| summary. Returns a result dict
    (weights/out paths, `unfreeze_top_n`, the diff summary) for callers/tests.
    """
    weights_path = Path(weights_path)
    out_path = Path(out_path)

    backbone, run_config = build_tuned_backbone(weights_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backbone.save_weights(str(out_path))
    print(f"saved tuned backbone: {out_path}")  # LOG

    # Reload FRESH (not the in-process instance) -- proves the save/load
    # round-trip, not just that the in-memory object happens to hold the
    # right values.
    tuned = load_dapt_backbone(out_path)
    original = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)

    tuned_paths = [w.path for w in tuned.weights]
    original_paths = [w.path for w in original.weights]
    if tuned_paths != original_paths:
        raise ValueError(
            "tuned/original backbone weight ordering diverged -- structural mismatch "
            "between the two Backbone.from_preset() constructions"
        )

    tuned_arrays = [np.asarray(w.numpy()) for w in tuned.weights]
    original_arrays = [np.asarray(w.numpy()) for w in original.weights]
    summary = layer_diff_summary(tuned_paths, tuned_arrays, original_arrays)

    print("Per-group max |delta| vs. original DAPT backbone:")  # LOG
    for group in sorted(summary, key=_group_sort_key):
        tag = " <- TUNED" if group in expected_tuned_groups(run_config.unfreeze_top_n) else ""
        print(f"  {group:24s} {summary[group]:.3e}{tag}")  # LOG

    _verify_expected_groups(summary, run_config.unfreeze_top_n)
    print(
        f"verified: tuned group(s) {sorted(expected_tuned_groups(run_config.unfreeze_top_n))} "
        f"clearly separated from non-tuned movement (>= {_MIN_TUNED_TO_FROZEN_RATIO}x)"
    )  # LOG

    return {
        "weights_path": str(weights_path),
        "out_path": str(out_path),
        "unfreeze_top_n": run_config.unfreeze_top_n,
        "layer_diff_summary": summary,
    }


def main(weights_path: Path | str = DEFAULT_WEIGHTS, out_path: Path | str | None = None) -> dict:
    weights_path = Path(weights_path)
    if out_path is None:
        out_path = default_out_path(weights_path)
    return extract_and_verify(weights_path, Path(out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract a fine-tuned RoBERTa backbone from a text-mode rel artifact."
    )
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                     help="text-mode rel artifact .weights.h5 (its .config.json sidecar "
                          "must sit alongside it)")
    ap.add_argument("--out", type=str, default=None,
                     help="output backbone .weights.h5 path (default: "
                          "tuned_backbone.<jobtag>.weights.h5 next to --weights)")
    args = ap.parse_args()
    main(weights_path=args.weights, out_path=args.out)
