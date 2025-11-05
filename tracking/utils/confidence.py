from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..core.interfaces import FramePrediction


BBox = Tuple[float, float, float, float]


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _bbox_diag(bbox: BBox) -> float:
    _, _, w, h = bbox
    return math.hypot(w, h)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


@dataclass(frozen=True)
class ConfidenceSignals:
    """Optionally supplied internal tracker signals for confidence estimation."""

    raw_logit: Optional[float] = None
    token_vectors: Optional[np.ndarray] = None  # shape (4, C)
    edge_distributions: Optional[Mapping[str, np.ndarray]] = None  # per-edge probability vectors
    attention_distribution: Optional[np.ndarray] = None  # shape (4, N)
    attention_focus: Optional[float] = None


@dataclass(frozen=True)
class ConfidenceConfig:
    """Hyper-parameters controlling the confidence estimator behaviour."""

    distribution_logit_weight: float = 1.0
    drift_logit_weight: float = 0.8
    token_logit_weight: float = 0.4

    token_gamma: float = 1.2
    distribution_beta: float = 2.0
    smoothing_alpha: float = 0.7

    drift_normalizer: float = 2.5
    default_score: float = 0.6


@dataclass
class ConfidenceState:
    frame_index: int
    confidence: float
    raw_components: Dict[str, float]
    previous_confidence: Optional[float]


class ConfidenceEstimator:
    """Fuse tracker signals into a conservative confidence estimate."""

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.cfg = config or ConfidenceConfig()
        self.reset()

    def reset(self) -> None:
        self._anchor_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None
        self._prev_confidence: Optional[float] = None
        self._frame_index: int = -1

    def update_from_prediction(self, prediction: FramePrediction, *, signals: Optional[ConfidenceSignals] = None) -> ConfidenceState:
        return self.update(
            frame_index=int(prediction.frame_index),
            bbox=prediction.bbox,
            raw_score=prediction.score,
            signals=signals,
        )

    def update(
        self,
        frame_index: int,
        bbox: BBox,
        raw_score: Optional[float],
        *,
        signals: Optional[ConfidenceSignals] = None,
    ) -> ConfidenceState:
        return self.evaluate(
            frame_index=frame_index,
            bbox=bbox,
            raw_score=raw_score,
            signals=signals,
            commit=True,
        )

    def evaluate(
        self,
        *,
        frame_index: int,
        bbox: BBox,
        raw_score: Optional[float],
        signals: Optional[ConfidenceSignals] = None,
        commit: bool = True,
    ) -> ConfidenceState:
        anchor_bbox = self._anchor_bbox if self._anchor_bbox is not None else bbox
        prev_bbox = self._prev_bbox if self._prev_bbox is not None else bbox
        prev_confidence = self._prev_confidence

        token_component = self._token_component(signals)
        distribution_component = self._distribution_component(signals)
        drift_component = self._drift_component(bbox, anchor_bbox if self._anchor_bbox is not None else None)

        logit = 0.0
        raw_components: Dict[str, float] = {}
        if distribution_component is not None:
            distribution_component = max(0.0, min(1.0, float(distribution_component)))
            raw_components["distribution"] = distribution_component
            logit += self.cfg.distribution_logit_weight * distribution_component
        if drift_component is not None:
            drift_component = max(0.0, min(1.0, float(drift_component)))
            raw_components["drift"] = drift_component
            logit += self.cfg.drift_logit_weight * drift_component
        if token_component is not None:
            token_component = max(0.0, min(1.0, float(token_component)))
            raw_components["token"] = token_component
            logit += self.cfg.token_logit_weight * token_component

        blended = 1.0 / (1.0 + math.exp(-logit))
        if prev_confidence is None:
            confidence = blended
        else:
            alpha = self.cfg.smoothing_alpha
            confidence = (1.0 - alpha) * prev_confidence + alpha * blended
        confidence = max(0.0, min(1.0, confidence))
        raw_components["blended"] = blended
        raw_components["logit"] = logit

        state = ConfidenceState(
            frame_index=frame_index,
            confidence=confidence,
            raw_components=raw_components,
            previous_confidence=prev_confidence,
        )

        if commit:
            self._anchor_bbox = anchor_bbox
            self._prev_bbox = bbox
            self._prev_confidence = confidence
            self._frame_index = frame_index
        return state

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------
    def _token_component(self, signals: Optional[ConfidenceSignals]) -> Optional[float]:
        if signals is None or signals.token_vectors is None:
            return None
        tokens = np.asarray(signals.token_vectors, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            return None

        tokens = np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Normalise token energy to prevent extremely large magnitudes from dominating variance.
        per_token_energy = np.sum(tokens * tokens, axis=1)
        mean_energy = float(np.mean(per_token_energy))
        if mean_energy > 0.0:
            scale = math.sqrt(mean_energy / float(max(tokens.shape[1], 1)))
            if scale > 1e-6:
                tokens = tokens / scale

        mean_vec = tokens.mean(axis=0, keepdims=True)
        diff = tokens - mean_vec
        variance = float(np.mean(np.sum(diff * diff, axis=1)))
        variance = variance / max(tokens.shape[1], 1)
        value = math.exp(-self.cfg.token_gamma * variance)
        return max(0.0, min(1.0, value))

    def _distribution_component(self, signals: Optional[ConfidenceSignals]) -> Optional[float]:
        if signals is None or signals.edge_distributions is None:
            return None
        entropies = []
        for dist in signals.edge_distributions.values():
            probs = np.asarray(dist, dtype=np.float32)
            if probs.ndim != 1 or probs.size == 0:
                continue
            probs = np.maximum(probs, 1e-8)
            probs = probs / probs.sum()
            entropy = float(-(probs * np.log(probs)).sum())
            max_entropy = math.log(float(probs.size)) if probs.size > 1 else 0.0
            if max_entropy <= 0.0:
                norm_entropy = 0.0
            else:
                norm_entropy = min(1.0, max(0.0, entropy / max_entropy))
            entropies.append(norm_entropy)
        if not entropies:
            return None
        mean_entropy = float(np.mean(entropies))
        value = math.exp(-self.cfg.distribution_beta * mean_entropy)
        return max(0.0, min(1.0, value))

    def _drift_component(self, bbox: BBox, anchor_bbox: Optional[BBox]) -> float:
        if anchor_bbox is None:
            return 1.0
        anchor_center = _bbox_center(anchor_bbox)
        current_center = _bbox_center(bbox)
        drift = math.hypot(current_center[0] - anchor_center[0], current_center[1] - anchor_center[1])
        anchor_diag = max(_bbox_diag(anchor_bbox), 1e-6)
        normalized = drift / (anchor_diag * self.cfg.drift_normalizer)
        if normalized >= 1.0:
            return 0.0
        return max(0.0, 1.0 - normalized)


__all__ = [
    "ConfidenceSignals",
    "ConfidenceConfig",
    "ConfidenceEstimator",
    "ConfidenceState",
]
