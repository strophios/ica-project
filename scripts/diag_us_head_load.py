# pattern: Imperative Shell
"""One-off diagnostic (2026-08-04): does _build_embed_model's us head match the file?

Context: cluster and local produce bit-identical CLS but shifted us_logit from
the same us_classifier.weights.h5 (md5-identical) — implying the runtime head
weights differ on the cluster despite identical file+code. Locally the four
runtime head arrays match the file exactly. Run this ON THE CLUSTER
(`uv run python -m scripts.diag_us_head_load`, CPU is fine) and compare.

Reads:  runtime head arrays vs the h5's classification_head datasets.
Tells:  match booleans + norms (trained bias norms are 0.2804 / 0.0044;
        EXACTLY-ZERO bias norms mean the head never left its zeros-init),
        plus the keras_hub preset cache fingerprint (the one per-machine
        downloaded artifact outside uv.lock).
Dumps:  us_head_runtime_diag.npz (the four runtime arrays) in the CWD for
        exact off-machine comparison.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import keras

import src.config as config


HEAD_DATASETS = [
    "layers/classification_head/dense/vars/0",
    "layers/classification_head/dense/vars/1",
    "layers/classification_head/logits/vars/0",
    "layers/classification_head/logits/vars/1",
]


def main() -> None:
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)
    from src.embed_corpus import _build_embed_model

    print("DTYPE_POLICY:", config.DTYPE_POLICY)
    model, us_cfg, backbone_path = _build_embed_model(config.US_FILTER_CLASSIFIER_WEIGHTS)
    print("backbone_path:", backbone_path)
    head = next(layer for layer in model.layers if layer.name == us_cfg.head.name)
    runtime = head.get_weights()

    with h5py.File(str(config.US_FILTER_CLASSIFIER_WEIGHTS), "r") as f:
        for arr, ds in zip(runtime, HEAD_DATASETS):
            file_arr = f[ds][()]
            match = bool(np.allclose(file_arr, arr, atol=1e-6))
            print(
                f"{ds}\n  match={match}  file_norm={np.linalg.norm(file_arr):.6f}"
                f"  runtime_norm={np.linalg.norm(arr):.6f}"
            )

    np.savez("us_head_runtime_diag.npz", **{f"arr{i}": a for i, a in enumerate(runtime)})
    print("dumped us_head_runtime_diag.npz (cwd)")

    # keras_hub preset cache fingerprint: the per-machine downloaded artifact.
    kdir = os.path.expanduser(os.environ.get("KERAS_HOME", "~/.keras"))
    print("preset cache under:", kdir)
    for root, _dirs, files in os.walk(kdir):
        if "roberta" in root.lower():
            for fn in sorted(files):
                p = os.path.join(root, fn)
                print(f"  {p}  {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
