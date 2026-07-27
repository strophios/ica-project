# pattern: Imperative Shell
"""Relevance text-mode encoder-unfreeze smoke test.

Level-1 mechanical check for the "rel-first sequential" text-mode training
path (docs/notes/encoder-unfreeze-strategy.md): 1 epoch, ~30 steps,
unfreeze_top_n=1, on the REAL DAPT backbone + REAL relevance_text_table.
Verifies end-to-end mechanics including that encoder-group gradients
ACTUALLY FLOW under the escalation branch -- that's the whole point of this
smoke test, not just "does it run."

Prerequisite: `uv run python -m src.build_relevance_text_table` must have been
run first (writes `relevance/relevance_text_table.parquet`).

Run from project root (module form -- `python scripts/rel_unfreeze_smoke.py`
directly fails with `ModuleNotFoundError: No module named 'src'`, because
Python sets `sys.path[0]` to the script's own directory rather than the CWD;
this bites every `scripts/*.py` invoked that way, e.g. `scripts/
tier5_short_run.py`'s own docstring instruction -- pre-existing, out of scope
here, flagged for a follow-up doc pass):
    uv run python -m scripts.rel_unfreeze_smoke
"""

import dataclasses
import time

from src.run_relevance_text import DEFAULT_REL_TEXT_CONFIG, main

# If MPS memory forces it, shrink this (the tf.data cache is batch-size-
# independent -- see run_relevance_text.main's batch_size docstring).
BATCH_SIZE = 32
MAX_STEPS = 30

if __name__ == "__main__":
    smoke_cfg = dataclasses.replace(
        DEFAULT_REL_TEXT_CONFIG,
        epochs=1,
        unfreeze_top_n=1,
        freeze_encoder=False,
    )

    t0 = time.monotonic()
    model, inference, history = main(
        run_config=smoke_cfg, max_steps=MAX_STEPS, batch_size=BATCH_SIZE
    )
    elapsed = time.monotonic() - t0
    n_steps = len(history.history.get("loss", []))
    per_step = elapsed / max(n_steps, 1)
    print(f"smoke run: {elapsed:.1f}s total, ~{per_step:.2f}s/epoch-record "
          f"(max_steps={MAX_STEPS}, batch_size={BATCH_SIZE})")  # LOG

    # The point of this smoke test: prove encoder-group gradients actually
    # flow under unfreeze_top_n=1, not just that freeze_encoder=False was set.
    tracker_key = "grad_norm/encoder_top/mean"
    if tracker_key not in history.history:
        raise AssertionError(
            f"{tracker_key!r} not found in history.history "
            f"(keys: {sorted(history.history.keys())}); diagnostics wiring "
            f"did not produce the expected per-group gradient tracker"
        )
    final_value = history.history[tracker_key][-1]
    print(f"{tracker_key} (final epoch) = {final_value}")  # LOG
    if not (final_value > 0):
        raise AssertionError(
            f"{tracker_key} = {final_value} -- encoder_top gradients did NOT "
            f"flow. Either unfreeze_top_n wiring is broken, or the top layer "
            f"genuinely received zero gradient this run (investigate before "
            f"trusting the escalation path)."
        )
    print("PASS: encoder_top gradients flowed under unfreeze_top_n=1.")  # LOG
