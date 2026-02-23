from __future__ import annotations

from typing import Dict, Optional

try:  # pragma: no cover - optional dependency guard
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except Exception as exc:  # noqa: BLE001
    accuracy_score = None  # type: ignore
    balanced_accuracy_score = None  # type: ignore
    confusion_matrix = None  # type: ignore
    precision_recall_fscore_support = None  # type: ignore
    roc_auc_score = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def ensure_metrics_available() -> None:
    if accuracy_score is None:
        raise RuntimeError(
            "scikit-learn is required for classification metrics."
            + (f" Import error: {_IMPORT_ERROR}" if _IMPORT_ERROR else "")
        )


def summarise_classification(
    y_true,
    y_pred,
    y_prob: Optional[list[float]] = None,
    positive_label: int = 1,
) -> Dict[str, float]:  # noqa: ANN001 - numpy or list inputs
    ensure_metrics_available()
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        metrics["balanced_accuracy"] = metrics["accuracy"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_label], zero_division=0
    )
    metrics["precision_positive"] = float(precision[0]) if len(precision) else 0.0
    metrics["recall_positive"] = float(recall[0]) if len(recall) else 0.0
    metrics["f1_positive"] = float(f1[0]) if len(f1) else 0.0
    metrics["support_positive"] = float(support[0]) if len(support) else 0.0
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["tn"] = float(cm[0, 0]) if cm.shape == (2, 2) else 0.0
        metrics["fp"] = float(cm[0, 1]) if cm.shape == (2, 2) else 0.0
        metrics["fn"] = float(cm[1, 0]) if cm.shape == (2, 2) else 0.0
        metrics["tp"] = float(cm[1, 1]) if cm.shape == (2, 2) else 0.0
    except Exception:
        metrics.update({"tn": 0.0, "fp": 0.0, "fn": 0.0, "tp": 0.0})
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
    # Friendly short-key aliases
    metrics["precision"] = metrics["precision_positive"]
    metrics["recall"] = metrics["recall_positive"]
    metrics["f1"] = metrics["f1_positive"]
    return metrics
