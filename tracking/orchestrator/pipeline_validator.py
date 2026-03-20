from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..classification.fusion_modules import is_learnable_fusion_module


NON_DIFFERENTIABLE_CLASSIFIERS = {
    "random_forest",
    "decision_tree",
    "svm",
    "xgboost",
    "lightgbm",
    "tabpfn_v2",
    "tabpfn_2_5",
    "tabpfn25",
    "tabpfn2_5",
    "multirocket",
}

# Known learnable classifier families (used for explicit detection hints).
LEARNABLE_CLASSIFIERS = {
    "v3pro_fusion",
    "fusion_mlp",
    "fusion_gating_mlp",
    "patchtst",
    "timemachine",
    "mlp_linear_head",
    "mlp_head",
    "mlp",
    "linear_head",
    "transformer",
}

LEARNABLE_FUSION_KEYWORDS = (
    "attention",
    "gating",
    "cross_attention",
    "fusion",
)

LEARNABLE_BACKBONE_KEYWORDS = (
    "convnext",
    "resnet",
    "efficientnet",
    "vit",
    "swin",
    "backbone",
)


class PipelineValidationError(RuntimeError):
    """Raised when pipeline validation fails under strict mode."""


@dataclass
class PipelineValidationResult:
    experiment_name: str
    has_learnable_feature: bool
    is_non_differentiable_classifier: bool
    strict: bool
    classifier_name: str
    details: List[str] = field(default_factory=list)

    @property
    def incompatible(self) -> bool:
        return self.has_learnable_feature and self.is_non_differentiable_classifier


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _get_validation_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg1 = config.get("pipeline_validation")
    if isinstance(cfg1, dict):
        return cfg1

    cfg2 = config.get("validation")
    if isinstance(cfg2, dict):
        pv = cfg2.get("pipeline_validator")
        if isinstance(pv, dict):
            return pv

    clf = config.get("classification")
    if isinstance(clf, dict):
        pv = clf.get("pipeline_validator")
        if isinstance(pv, dict):
            return pv

    return {}


def _iter_named_nodes(node: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield next_path, value
            yield from _iter_named_nodes(value, next_path)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            next_path = f"{path}[{idx}]"
            yield next_path, value
            yield from _iter_named_nodes(value, next_path)


def detect_classifier_type(classifier_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    name = str((classifier_cfg or {}).get("name", "")).strip().lower()
    is_non_diff = name in NON_DIFFERENTIABLE_CLASSIFIERS
    return is_non_diff, name


def detect_learnable_modules(classification_cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    details: List[str] = []
    classification_cfg = classification_cfg or {}

    feature_extractor = classification_cfg.get("feature_extractor")
    classifier = classification_cfg.get("classifier")
    fusion_module_cfg = classification_cfg.get("fusion_module") if isinstance(classification_cfg.get("fusion_module"), dict) else None
    fe_params = (feature_extractor or {}).get("params", {}) if isinstance(feature_extractor, dict) else {}
    clf_params = (classifier or {}).get("params", {}) if isinstance(classifier, dict) else {}

    classifier_name = str((classifier or {}).get("name", "")).strip().lower() if isinstance(classifier, dict) else ""
    fe_name = str((feature_extractor or {}).get("name", "")).strip().lower() if isinstance(feature_extractor, dict) else ""

    if classifier_name in LEARNABLE_CLASSIFIERS:
        details.append(f"classifier={classifier_name} (differentiable classifier family)")

    if fusion_module_cfg is not None:
        fm_name = str(fusion_module_cfg.get("name", "")).strip().lower()
        if fm_name:
            if is_learnable_fusion_module(fm_name):
                details.append(f"fusion_module={fm_name} (learnable)")

    if fe_name:
        if any(k in fe_name for k in ("v3pro", "fusion", "deep", "nn")):
            details.append(f"feature_extractor={fe_name} (likely learnable feature pathway)")

    def _record_if_true(path: str, value: Any) -> None:
        if isinstance(value, bool) and value:
            details.append(f"{path}=True")

    for path, value in _iter_named_nodes(classification_cfg):
        path_lc = path.lower()

        # Explicit trainable / gradient switches.
        if any(k in path_lc for k in ("trainable", "requires_grad", "learnable")):
            _record_if_true(path, value)

        # Backbone hints.
        if any(k in path_lc for k in ("backbone", "encoder")):
            if isinstance(value, str) and any(tok in value.lower() for tok in LEARNABLE_BACKBONE_KEYWORDS):
                details.append(f"{path}={value}")

        # Projection hints (linear/mlp projections).
        if "projection" in path_lc or "proj" in path_lc:
            if isinstance(value, bool) and value:
                details.append(f"{path}=True")
            elif isinstance(value, (int, float)) and value > 0:
                details.append(f"{path}={value}")
            elif isinstance(value, str) and value.strip():
                details.append(f"{path}={value}")

        # Learnable fusion modes.
        if "fusion_mode" in path_lc and isinstance(value, str):
            fusion_mode = value.strip().lower()
            if any(k in fusion_mode for k in LEARNABLE_FUSION_KEYWORDS):
                details.append(f"fusion_mode={value}")

        if path_lc.endswith("fusion_module.name") and isinstance(value, str):
            fm_name = value.strip().lower()
            if is_learnable_fusion_module(fm_name):
                details.append(f"fusion_module={value} (learnable)")

        # Explicit texture mode switch.
        if "texture_mode" in path_lc and isinstance(value, str):
            if value.strip().lower() == "learnable":
                details.append("texture_mode=learnable")

    # De-duplicate while preserving order.
    dedup_details: List[str] = []
    seen = set()
    for item in details:
        if item in seen:
            continue
        seen.add(item)
        dedup_details.append(item)

    return len(dedup_details) > 0, dedup_details


def _resolve_experiment_items(config: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    exps = config.get("experiments")
    if isinstance(exps, list) and len(exps) > 0:
        return [e for e in exps if isinstance(e, dict)]
    return [{"name": "default"}]


def validate_pipeline(config: Dict[str, Any]) -> List[PipelineValidationResult]:
    validation_cfg = _get_validation_cfg(config)
    strict = bool(validation_cfg.get("strict", False))

    top_level_clf_cfg = config.get("classification") if isinstance(config.get("classification"), dict) else {}
    results: List[PipelineValidationResult] = []

    for idx, exp in enumerate(_resolve_experiment_items(config), start=1):
        exp_name = str(exp.get("name") or f"experiment#{idx}")
        exp_clf_cfg = exp.get("classification") if isinstance(exp.get("classification"), dict) else {}
        effective_clf_cfg = _deep_merge_dict(top_level_clf_cfg or {}, exp_clf_cfg or {})

        has_learnable_feature, learnable_details = detect_learnable_modules(effective_clf_cfg)
        is_non_diff, classifier_name = detect_classifier_type(effective_clf_cfg.get("classifier", {}) or {})

        detail_lines = list(learnable_details)
        if classifier_name:
            detail_lines.append(f"classifier={classifier_name}")

        results.append(
            PipelineValidationResult(
                experiment_name=exp_name,
                has_learnable_feature=has_learnable_feature,
                is_non_differentiable_classifier=is_non_diff,
                strict=strict,
                classifier_name=classifier_name,
                details=detail_lines,
            )
        )

    return results


def format_pipeline_warning(result: PipelineValidationResult) -> str:
    lines = [
        "[Warning] Detected incompatible pipeline configuration:",
        f"- experiment = {result.experiment_name}",
        "",
        "You are using learnable feature modules (e.g., backbone / projection / fusion),",
        "but the selected classifier does not support backpropagation.",
        "",
        "As a result:",
        "- The backbone will NOT be trained",
        "- The fusion module will NOT learn",
        "- The feature extractor will behave as a fixed feature generator",
        "",
        "This may significantly limit performance and waste model capacity.",
        "",
        "Suggestions:",
        "1. Use a differentiable classifier (e.g., MLP, Transformer, PatchTST)",
        "2. OR freeze all feature extractors explicitly and treat them as fixed features",
        "3. OR remove learnable fusion/backbone modules",
    ]
    if result.details:
        lines.extend([
            "",
            "Detected sources:",
        ])
        for item in result.details:
            lines.append(f"- {item}")
    return "\n".join(lines)


def enforce_or_collect_warnings(config: Dict[str, Any]) -> List[str]:
    """Validate config and either raise (strict) or return warning messages.

    Returns
    -------
    warnings : list[str]
        Formatted warning messages for all incompatible experiment items.
    """
    results = validate_pipeline(config)
    warning_messages = [format_pipeline_warning(r) for r in results if r.incompatible]

    if warning_messages and any(r.strict for r in results):
        raise PipelineValidationError(
            "\n\n".join(
                [
                    "Pipeline validator (strict mode) rejected incompatible configuration.",
                    *warning_messages,
                ]
            )
        )

    return warning_messages
