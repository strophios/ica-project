# pattern: Imperative Shell
"""IcaModel: multi-head inference artifact for assembled ICA scoring.

Loads the three head configs (US, CCA, relevance), transfers trained weights via
Pattern 2 (temporary single-head models), assembles them into one model (Pattern A
in-process sharing), loads calibrators and fusion config, and composes:
  1. Gate: survivors = calib_us >= tau_us
  2. Combine: product-AND or logistic regression on CCA + rel
  3. Apply composed-Platt if calibrated
  4. ICA score: composed score for survivors, 0.0 for gated-out rows

Prediction paths:
  - predict_ica_from_features(features): input shape (n, 768) CLS embeddings
  - predict_ica_from_text(texts): optional; input shape (n,) raw text strings
"""

from __future__ import annotations

import logging
import numpy as np

import keras

import src.config as config
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model
from src.calibration.sidecar import (
    calibration_path_for_weights,
    load_calibration,
)
from src.fusion.sidecar import load_fusion
from src.fusion.combiner import combine_and, apply_logistic_combiner
from src.calibration.calibrator import platt_transform
from src.us_config import UsRunConfig
from src.cca_config import RunConfig


logger = logging.getLogger(__name__)


class IcaModel:
    """Multi-head ICA inference artifact (features-mode primary path).

    Constructor loads three head configs, constructs heads, transfers weights via
    Pattern 2, assembles into one model, and loads calibrators + fusion config.

    Public methods:
      - predict_ica_from_features(features): dict with per-head probs + ica_score
      - predict_ica_from_text(texts): optional text-mode prediction

    Attributes:
      - self.model: assembled keras.Model, features input → per-head logits output
      - self.us_head, self.cca_head, self.rel_head: ClassificationHead instances
      - self.us_calibrator, self.cca_calibrator, self.rel_calibrator: PlattCalibrator
      - self.fusion_config: FusionConfig with gate_threshold, combine method, etc.
    """

    def __init__(
        self,
        us_weights_path=None,
        cca_weights_path=None,
        rel_weights_path=None,
        fusion_path=None,
        head_feature_sources=None,
    ):
        """Load configs, construct heads, transfer weights, assemble model.

        Args:
            us_weights_path: path to US head weights (features-mode, head-only).
                None (default) resolves to `config.US_FILTER_FULL_WEIGHTS`,
                read from `src.config` AT CALL TIME (not baked in at import
                time) -- so `mock.patch("src.assemble_ica.config")` works.
            cca_weights_path: path to CCA head weights (features-mode, head-only).
                None (default) resolves to `config.CCA_DOCA_WEIGHTS`.
            rel_weights_path: path to relevance head weights (features-mode,
                head-only). None (default) resolves to `config.RELEVANCE_DOCA_WEIGHTS`.
            fusion_path: path to fusion config (.fusion.json); if None, defaults to
                         config.CCA_DOCA_DIR / "ica_fusion.fusion.json"
            head_feature_sources: optional `dict[str, str]`, head name -> CLS
                source tag (e.g. `{"us": "base", "cca": "base", "rel": "rel_branch"}`
                for the branched-encoder apply path,
                `docs/design-plans/2026-08-18-stage4-joint-finetune.md`). `None`
                (default) = legacy single shared feature matrix. When set, the
                fusion sidecar's own `head_feature_sources` (if present) is
                cross-checked against this value.

        Raises:
            ValueError: if head configs don't load, weight transfer fails, or
                head_feature_sources doesn't match the fusion sidecar's record
            FileNotFoundError: if weights, config sidecars, or fusion config don't exist
        """
        from src.cca_config import config_path_for_weights

        # Resolve None sentinels from src.config AT CALL TIME. Prior versions
        # bound these as literal default-argument expressions (evaluated once
        # at module-import time), which silently defeated
        # `mock.patch("src.assemble_ica.config")` in tests -- a bare
        # `IcaModel()` call would load the real production artifacts instead
        # of whatever the test pointed `config` at.
        if us_weights_path is None:
            us_weights_path = config.US_FILTER_FULL_WEIGHTS
        if cca_weights_path is None:
            cca_weights_path = config.CCA_DOCA_WEIGHTS
        if rel_weights_path is None:
            rel_weights_path = config.RELEVANCE_DOCA_WEIGHTS

        # Convert to Path for consistency
        us_weights_path = str(us_weights_path)
        cca_weights_path = str(cca_weights_path)
        rel_weights_path = str(rel_weights_path)

        # Default fusion_path if not provided
        if fusion_path is None:
            fusion_path = config.CCA_DOCA_DIR / "ica_fusion.fusion.json"
        fusion_path = str(fusion_path)

        self.head_feature_sources = head_feature_sources

        logger.info("Constructing IcaModel: loading configs and heads")

        # ====================================================================
        # Load configs
        # ====================================================================
        us_config = UsRunConfig.from_json(
            config_path_for_weights(us_weights_path)
        )
        cca_config = RunConfig.from_json(
            config_path_for_weights(cca_weights_path)
        )
        rel_config = RunConfig.from_json(
            config_path_for_weights(rel_weights_path)
        )

        logger.info(f"Loaded configs: US (hidden_dim={us_config.head.hidden_dim}), "
                   f"CCA (hidden_dim={cca_config.heads[0].hidden_dim}), "
                   f"rel (hidden_dim={rel_config.heads[0].hidden_dim})")

        # ====================================================================
        # Construct heads (loss_fn is required by constructor but ignored in inference)
        # ====================================================================
        # US head uses BCE loss (defined at construction time)
        self.us_head = ClassificationHead(
            hidden_dim=us_config.head.hidden_dim,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="us",
        )

        # CCA and rel use their configured loss (FLPU for CCA, typically)
        cca_loss_config = cca_config.heads[0].loss
        self.cca_head = ClassificationHead(
            hidden_dim=cca_config.heads[0].hidden_dim,
            loss_fn=_instantiate_loss_from_config(cca_loss_config),
            name="cca",
        )

        rel_loss_config = rel_config.heads[0].loss
        self.rel_head = ClassificationHead(
            hidden_dim=rel_config.heads[0].hidden_dim,
            loss_fn=_instantiate_loss_from_config(rel_loss_config),
            name="rel",
        )

        # ====================================================================
        # Transfer weights via Pattern 2 (temporary single-head models)
        # ====================================================================
        logger.info("Transferring weights via Pattern 2 (temporary single-head models)")

        # US head weights
        temp_us_model = build_feature_inference_model(
            {"us": self.us_head}, hidden_dim=us_config.head.hidden_dim
        )
        temp_us_model.load_weights(us_weights_path, skip_mismatch=False)
        logger.info(f"Loaded US weights: {us_weights_path}")

        # CCA head weights
        temp_cca_model = build_feature_inference_model(
            {"cca": self.cca_head}, hidden_dim=cca_config.heads[0].hidden_dim
        )
        temp_cca_model.load_weights(cca_weights_path, skip_mismatch=False)
        logger.info(f"Loaded CCA weights: {cca_weights_path}")

        # Relevance head weights
        temp_rel_model = build_feature_inference_model(
            {"rel": self.rel_head}, hidden_dim=rel_config.heads[0].hidden_dim
        )
        temp_rel_model.load_weights(rel_weights_path, skip_mismatch=False)
        logger.info(f"Loaded relevance weights: {rel_weights_path}")

        # ====================================================================
        # Assemble multi-head model (Pattern A in-process sharing)
        # ====================================================================
        logger.info("Assembling multi-head inference model (Pattern A)")
        self.model = build_feature_inference_model(
            {
                "us": self.us_head,
                "cca": self.cca_head,
                "rel": self.rel_head,
            },
            hidden_dim=768,  # Shared CLS feature dimension
            feature_sources=head_feature_sources,
        )

        # ====================================================================
        # Load calibrators (per-head)
        # ====================================================================
        logger.info("Loading per-head calibrators")
        self.us_calibrator = load_calibration(
            calibration_path_for_weights(us_weights_path)
        )
        self.cca_calibrator = load_calibration(
            calibration_path_for_weights(cca_weights_path)
        )
        self.rel_calibrator = load_calibration(
            calibration_path_for_weights(rel_weights_path)
        )

        logger.info("Per-head calibrators loaded (Platt fit populations: "
                   f"US={self.us_calibrator.fit_population}, "
                   f"CCA={self.cca_calibrator.fit_population}, "
                   f"rel={self.rel_calibrator.fit_population})")

        # ====================================================================
        # Load fusion config
        # ====================================================================
        logger.info("Loading fusion config")
        self.fusion_config = load_fusion(fusion_path)

        logger.info(
            f"Fusion config loaded: combine={self.fusion_config.combine}, "
            f"gate_threshold={self.fusion_config.gate_threshold}, "
            f"score_space={self.fusion_config.score_space}, "
            f"composed_platt={'fitted' if self.fusion_config.composed_platt else 'not fitted'}"
        )

        # Cross-check: when the constructor was given explicit sources AND the
        # fusion sidecar records its own (non-None), they must agree -- a
        # branched-encoder apply must score with the same sources the fusion
        # combiner/calibration was fit on. Fusion without the field (None) =
        # no check (back-compat with fusion sidecars predating this feature).
        if head_feature_sources is not None and self.fusion_config.head_feature_sources is not None:
            if head_feature_sources != self.fusion_config.head_feature_sources:
                raise ValueError(
                    f"head_feature_sources mismatch: IcaModel was constructed "
                    f"with {head_feature_sources!r}, but the fusion sidecar "
                    f"({fusion_path}) records {self.fusion_config.head_feature_sources!r}. "
                    f"The two must agree -- the fusion combiner/calibration was "
                    f"fit on scores produced by its recorded sources."
                )

        logger.info("IcaModel construction complete")

    def _prepare_model_inputs(self, features) -> tuple[dict, int]:
        """Validate `features` and build the model's `.predict()` input dict.

        Legacy mode (`self.head_feature_sources is None`): `features` must be
        a `(n, 768)` ndarray; returns `{"features": features}`.

        Sources mode (`self.head_feature_sources` set): `features` must be a
        `dict[source_tag, (n, 768) ndarray]` covering every tag the model
        needs; returns `{"features_<tag>": arr, ...}`.

        Returns:
            (model_inputs, n_rows)

        Raises:
            ValueError: wrong container type for the configured mode, a
                missing source tag, a non-2-D array, a hidden-dim mismatch,
                or mismatched row counts across sources.
        """
        if self.head_feature_sources is None:
            if isinstance(features, dict):
                raise ValueError(
                    "predict_ica_from_features received a dict, but this "
                    "IcaModel has no head_feature_sources configured (legacy "
                    "shared-feature mode expects a single (n, 768) array)."
                )
            features = np.asarray(features, dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != 768:
                raise ValueError(
                    f"features must have shape (n, 768), got {features.shape}"
                )
            return {"features": features}, features.shape[0]

        if not isinstance(features, dict):
            raise ValueError(
                "predict_ica_from_features requires a dict[source_tag, "
                "(n, 768) array] because this IcaModel was constructed with "
                f"head_feature_sources={self.head_feature_sources!r}; got "
                f"{type(features).__name__}."
            )

        needed_tags = sorted(set(self.head_feature_sources.values()))
        missing = [t for t in needed_tags if t not in features]
        if missing:
            raise ValueError(
                f"features dict is missing required source tag(s) {missing} "
                f"(head_feature_sources={self.head_feature_sources!r})"
            )

        model_inputs = {}
        n_rows = None
        for tag in needed_tags:
            arr = np.asarray(features[tag], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 768:
                raise ValueError(
                    f"features[{tag!r}] must have shape (n, 768), got {arr.shape}"
                )
            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError(
                    f"features dict has mismatched row counts across sources: "
                    f"tag {tag!r} has {arr.shape[0]} rows, expected {n_rows} "
                    f"(all sources must be row-aligned)"
                )
            model_inputs[f"features_{tag}"] = arr

        return model_inputs, n_rows

    def predict_ica_from_features(
        self, features, gate_override: np.ndarray | None = None
    ) -> dict:
        """Score cached CLS features, returning per-head probs + composed ICA score.

        Args:
            features: legacy mode (head_feature_sources=None, the default):
                shape (n, 768) float32 array of CLS embeddings. Sources mode
                (head_feature_sources set): dict[source_tag, (n, 768) array],
                one entry per distinct tag the model's heads need, all
                row-aligned.
            gate_override: optional shape (n,) boolean array override for the US gate.
                When provided, use this as the survivor mask instead of (calib_us >= tau_us).
                When None (default), use the ML US gate.

        Returns:
            dict with keys:
              - "us": (n,) calibrated US probabilities in [0, 1]
              - "cca": (n,) calibrated CCA probabilities in [0, 1]
              - "rel": (n,) calibrated relevance probabilities in [0, 1]
              - "ica_score": (n,) composed ICA score in [0, 1],
                  0.0 for gated-out rows, composed score for survivors
        """
        model_inputs, n_rows = self._prepare_model_inputs(features)

        # Run inference: dict of logits keyed by head name
        logits_dict = self.model.predict(model_inputs, verbose=0)

        # Extract and calibrate per-head logits
        us_logits = logits_dict["us"].ravel()
        cca_logits = logits_dict["cca"].ravel()
        rel_logits = logits_dict["rel"].ravel()

        calib_us = self.us_calibrator.transform(us_logits)
        calib_cca = self.cca_calibrator.transform(cca_logits)
        calib_rel = self.rel_calibrator.transform(rel_logits)

        # Gate: survivors (ML or overridden)
        if gate_override is not None:
            gate_override = np.asarray(gate_override, dtype=bool)
            if gate_override.shape[0] != n_rows:
                raise ValueError(
                    f"gate_override shape {gate_override.shape[0]} does not match features row count {n_rows}"
                )
            survivors = gate_override
        else:
            tau_us = self.fusion_config.gate_threshold
            survivors = calib_us >= tau_us

        # Combine on survivors
        combined = self._combine_scores(calib_cca, calib_rel)

        # Apply composed-Platt calibration if fitted
        if self.fusion_config.composed_platt is not None:
            composed_score = self._apply_composed_platt(combined)
        else:
            composed_score = combined

        # Gate-out non-survivors
        ica_score = np.where(survivors, composed_score, 0.0)

        return {
            "us": calib_us,
            "cca": calib_cca,
            "rel": calib_rel,
            "ica_score": ica_score,
        }

    def predict_ica_from_text(self, texts: list[str]) -> dict:
        """Optional text-mode prediction.

        Not implemented; see Phase 6.
        """
        raise NotImplementedError(
            "Text-mode prediction (predict_ica_from_text) not yet implemented. "
            "Use predict_ica_from_features with pre-computed CLS embeddings."
        )

    def _combine_scores(self, calib_cca: np.ndarray, calib_rel: np.ndarray) -> np.ndarray:
        """Apply fusion combiner: product-AND or logistic regression.

        Args:
            calib_cca: calibrated CCA probabilities
            calib_rel: calibrated relevance probabilities

        Returns:
            Combined score (shape matching inputs)
        """
        if self.fusion_config.combine == "product":
            return combine_and(calib_cca, calib_rel)
        elif self.fusion_config.combine == "logreg":
            # CRITICAL: LR was fit on PROBABILITY features (not logits).
            # The coefs encode the learned linear combination in probability space.
            # Input features are [p_cca, p_rel] (calibrated probabilities)
            # coefs = [slope_cca, slope_rel, intercept] (last element is intercept)
            scores = np.column_stack([calib_cca, calib_rel])
            if self.fusion_config.coefs is None:
                raise ValueError("coefs must be set for logreg combiner")
            if len(self.fusion_config.coefs) < 3:
                raise ValueError(
                    f"logreg coefs must have ≥3 elements (slopes+intercept), got {len(self.fusion_config.coefs)}"
                )
            # Type cast coefs list to tuple for apply_logistic_combiner
            # Format: (slopes_array, intercept)
            coefs_tuple: tuple[np.ndarray, float] = (
                np.asarray(self.fusion_config.coefs[:-1]),
                float(self.fusion_config.coefs[-1]),
            )
            return apply_logistic_combiner(coefs_tuple, scores)
        else:  # pragma: no cover  # unreachable: FusionConfig.combine is Literal["product", "logreg"]
            raise ValueError(
                f"unknown combine method: {self.fusion_config.combine}"
            )

    def _apply_composed_platt(self, combined_score: np.ndarray) -> np.ndarray:
        """Apply composed-Platt calibration to combined score.

        CRITICAL: Matches the score-space transformation from fit_fusion.py:
          1. combined_score is in [0, 1] (probability space)
          2. Clip to (1e-10, 1-1e-10) to avoid log(0)
          3. Convert to logit space
          4. Apply Platt transform (sigmoid of A*logit + B)
          5. Return probability in [0, 1]

        Args:
            combined_score: shape (n,) combined score in [0, 1]

        Returns:
            Platt-calibrated score in [0, 1]
        """
        if self.fusion_config.composed_platt is None:
            raise ValueError("composed_platt must be fitted before calling _apply_composed_platt")
        A, B = self.fusion_config.composed_platt

        # Step 1: Clip combined score (probability) to avoid log(0)
        combined_clip = np.clip(combined_score, 1e-10, 1.0 - 1e-10)

        # Step 2: Convert probability to logit
        logits = np.log(combined_clip / (1.0 - combined_clip))

        # Step 3: Apply Platt transform
        calibrated = platt_transform(logits, A, B)

        return calibrated


def _instantiate_loss_from_config(loss_config):
    """Instantiate a loss function from a loss config (Functional Core).

    Handles FLPULoss and standard losses like BCE.

    Args:
        loss_config: a loss config object with a get_loss_fn() method or similar

    Returns:
        Instantiated loss function

    Raises:
        ValueError: if config type is unknown
    """
    from src.loss_functions.loss import FLPULoss

    # Check if it's an FLPULoss config
    if hasattr(loss_config, "prior"):
        # FLPULoss config
        return FLPULoss(prior=loss_config.prior)
    elif hasattr(loss_config, "get_loss_fn"):
        # Generic loss config with getter
        return loss_config.get_loss_fn()
    else:
        raise ValueError(
            f"unknown loss config type: {type(loss_config).__name__}. "
            f"Loss config must have prior (FLPULoss) or get_loss_fn() method."
        )
