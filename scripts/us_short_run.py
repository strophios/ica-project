# pattern: Imperative Shell
"""US filter short stress-test run: 1 epoch, 200 max steps on local float32.

Reproducible level-1 mechanical check. Run from project root:
  uv run python scripts/us_short_run.py
"""

import dataclasses

from src import us_config
from src.run_us_classification import main

if __name__ == "__main__":
    short_cfg = dataclasses.replace(us_config.DEFAULT_US_CONFIG, epochs=1)
    main(run_config=short_cfg, max_steps=200)
