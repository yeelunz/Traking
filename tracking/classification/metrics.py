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


def _brier_decomposition(
    y_true_bin: list[float],
    y_prob: list[float],
    *,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Murphy decomposition for binary Brier score.

    BS = Reliability - Resolution + Uncertainty
    """
    n = len(y_true_bin)
    if n == 0 or n != len(y_prob):
        nan = float("nan")
        return {
            "reliability": nan,
            "resolution": nan,
            "uncertainty": nan,
        }

    n_bins = max(1, int(n_bins))
    clipped = [max(0.0, min(1.0, float(p))) for p in y_prob]
    o_bar = sum(float(o) for o in y_true_bin) / float(n)

    groups: list[list[int]] = [[] for _ in range(n_bins)]
    for idx, p in enumerate(clipped):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        groups[bin_idx].append(idx)

    reliability = 0.0
    resolution = 0.0
    for indices in groups:
        if not indices:
            continue
        nk = float(len(indices))
        fk = sum(clipped[i] for i in indices) / nk
        ok = sum(y_true_bin[i] for i in indices) / nk
        wk = nk / float(n)
        reliability += wk * (fk - ok) ** 2
        resolution += wk * (ok - o_bar) ** 2

    uncertainty = o_bar * (1.0 - o_bar)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


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
        # Derive labels dynamically so TP/FP/FN/TN stay consistent with
        # the ``positive_label`` parameter used for precision/recall/F1.
        neg_label = 0 if positive_label != 0 else 1
        cm = confusion_matrix(y_true, y_pred, labels=[neg_label, positive_label])
        metrics["tn"] = float(cm[0, 0])
        metrics["fp"] = float(cm[0, 1])
        metrics["fn"] = float(cm[1, 0])
        metrics["tp"] = float(cm[1, 1])
    except Exception:
        metrics.update({"tn": 0.0, "fp": 0.0, "fn": 0.0, "tp": 0.0})
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")  # not computable (single-class fold)
        try:
            y_true_bin = [1.0 if int(y) == int(positive_label) else 0.0 for y in y_true]
            y_prob_list = [float(p) for p in y_prob]
            if len(y_true_bin) == len(y_prob_list) and len(y_true_bin) > 0:
                brier = sum((max(0.0, min(1.0, p)) - o) ** 2 for p, o in zip(y_prob_list, y_true_bin)) / float(len(y_true_bin))
                metrics["brier_score"] = float(brier)
                dec = _brier_decomposition(y_true_bin, y_prob_list, n_bins=10)
                metrics["reliability"] = dec["reliability"]
                metrics["resolution"] = dec["resolution"]
                metrics["uncertainty"] = dec["uncertainty"]
            else:
                metrics["brier_score"] = float("nan")
                metrics["reliability"] = float("nan")
                metrics["resolution"] = float("nan")
                metrics["uncertainty"] = float("nan")
        except Exception:
            metrics["brier_score"] = float("nan")
            metrics["reliability"] = float("nan")
            metrics["resolution"] = float("nan")
            metrics["uncertainty"] = float("nan")
    # Friendly short-key aliases
    metrics["precision"] = metrics["precision_positive"]
    metrics["recall"] = metrics["recall_positive"]
    metrics["f1"] = metrics["f1_positive"]
    return metrics
