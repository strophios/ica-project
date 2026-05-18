# pattern: Imperative Shell
"""Tier 5 short stress-test run: full diagnostic stack on real cca_set/,
one epoch capped at a few hundred steps. Reproducible level-1 mechanical
check. Run from project root: python scripts/tier5_short_run.py"""

import dataclasses

from src import cca_config
from src.run_cca_classification import main

if __name__ == "__main__":
    short_cfg = dataclasses.replace(cca_config.DEFAULT_CCA_CONFIG, epochs=1)
    main(run_config=short_cfg, max_steps=200)
