"""Tests for the `ICA_DTYPE_POLICY` env override in `src/config.py`.

Default policy is platform-conditional (`mixed_float16` on cluster, `float32`
locally). The override exists so a cluster job can produce float32 CLS caches
uniform with the locally-produced production caches (the 2026-08-04 finding:
`full` part-1 / `ldc_9507` are local fp32; appending fp16 shards would put a
precision seam exactly on the 1975/76 era boundary).

Each test reloads `src.config` under a controlled env and restores the real
env + module state afterward, so other tests see the unmodified config.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_config_with(env_value: str | None):
    """Reload src.config with ICA_DTYPE_POLICY set (or absent) and return it."""
    if env_value is None:
        os.environ.pop("ICA_DTYPE_POLICY", None)
    else:
        os.environ["ICA_DTYPE_POLICY"] = env_value
    import src.config
    return importlib.reload(src.config)


@pytest.fixture(autouse=True)
def _restore_config():
    """Restore the real env + module state after each test."""
    saved = os.environ.get("ICA_DTYPE_POLICY")
    yield
    if saved is None:
        os.environ.pop("ICA_DTYPE_POLICY", None)
    else:
        os.environ["ICA_DTYPE_POLICY"] = saved
    import src.config
    importlib.reload(src.config)


def test_default_policy_is_platform_conditional():
    """Without the env var, the platform rule stands (float32 locally)."""
    config = _reload_config_with(None)
    expected = "mixed_float16" if config.IS_CLUSTER else "float32"
    assert config.DTYPE_POLICY == expected


def test_env_override_float32():
    config = _reload_config_with("float32")
    assert config.DTYPE_POLICY == "float32"


def test_env_override_mixed_float16():
    config = _reload_config_with("mixed_float16")
    assert config.DTYPE_POLICY == "mixed_float16"


def test_invalid_override_raises():
    """A typo'd policy must fail loudly at import, not silently fall back."""
    with pytest.raises(ValueError, match="ICA_DTYPE_POLICY"):
        _reload_config_with("float16")
