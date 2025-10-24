from __future__ import annotations

from typing import Dict, Sequence, Any, Protocol

from ..core.interfaces import FramePrediction


class TrajectoryFeatureExtractor(Protocol):
    """Extracts fixed-length feature dictionaries from trajectory samples."""

    name: str

    def extract_video(self, samples: Sequence[FramePrediction]) -> Dict[str, float]:
        """Compute features for a single video trajectory."""

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-video features into a subject-level feature dictionary."""

    def feature_order(self, level: str = "video") -> Sequence[str]:
        """Return deterministic feature key ordering used for vectorisation at a given level."""


class SubjectClassifier(Protocol):
    """Fits a subject-level classifier and performs inference."""

    name: str

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001 - numpy arrays
        """Train classifier and return training metadata."""

    def predict(self, X):  # noqa: ANN001
        """Predict class labels for given feature matrix."""

    def predict_proba(self, X):  # noqa: ANN001
        """Predict class probabilities for given feature matrix."""

    def save(self, path: str) -> None:
        """Persist classifier state to disk (optional)."""

    def load(self, path: str) -> None:
        """Load classifier state from disk (optional)."""
