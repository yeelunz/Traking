from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ...core.registry import register_classifier
from ..interfaces import SubjectClassifier


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _extract_missing_module_name(exc: Exception) -> Optional[str]:
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


@register_classifier("limix")
class LimiXSubjectClassifier(SubjectClassifier):
    """LimiX-based subject-level classifier.

    Notes
    -----
    - Keeps the existing pipeline contract (fit / predict / predict_proba).
    - LimiX inference is transductive (needs train + test features at inference).
      Therefore this classifier stores the fitted training matrix in memory.
    - Width/height tracking or trajectory logic is unrelated and untouched.
    """

    name = "LimiX"

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_path": "ckpt/limix/LimiX-16M.ckpt",
        "inference_config_path": "libs/LimiX/config/cls_default_noretrieval.json",
        "device": "auto",
        "mix_precision": True,
        "softmax_temperature": 0.9,
        "outlier_remove_std": 12.0,
        "seed": 42,
        "inference_with_DDP": False,
        "categorical_features_indices": None,
        "auto_copy_ckpt_to_default": True,
        "auto_install_dependencies": True,
        "dependency_packages": [
            "pandas",
            "scipy",
            "einops",
            "kditransform",
            "hyperopt",
        ],
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._cfg = cfg
        self._limix_import_error: Optional[Exception] = None
        self._predictor_cls: Any = None
        self._torch: Any = None
        self._predictor: Any = None

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _resolve_path(self, p: str) -> Path:
        pp = Path(str(p))
        return pp if pp.is_absolute() else (self._repo_root() / pp)

    def _default_ckpt_target(self) -> Path:
        return self._repo_root() / "ckpt" / "limix" / "LimiX-16M.ckpt"

    def _find_ckpt_candidate(self) -> Optional[Path]:
        configured = self._resolve_path(str(self._cfg.get("model_path", "")))
        candidates = [configured]

        home = Path.home()
        candidates.extend(
            [
                self._default_ckpt_target(),
                self._repo_root() / "LimiX-16M.ckpt",
                home / "Downloads" / "LimiX-16M.ckpt",
                home / "Desktop" / "LimiX-16M.ckpt",
            ]
        )

        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _ensure_ckpt(self) -> Path:
        ckpt_found = self._find_ckpt_candidate()
        if ckpt_found is None:
            tried = [
                str(self._resolve_path(str(self._cfg.get("model_path", "")))),
                str(self._default_ckpt_target()),
                str(Path.home() / "Downloads" / "LimiX-16M.ckpt"),
                str(Path.home() / "Desktop" / "LimiX-16M.ckpt"),
            ]
            raise FileNotFoundError(
                "LimiX checkpoint not found. Please place LimiX-16M.ckpt at one of: "
                + " | ".join(tried)
            )

        target = self._default_ckpt_target()
        target.parent.mkdir(parents=True, exist_ok=True)

        if bool(self._cfg.get("auto_copy_ckpt_to_default", True)):
            if ckpt_found.resolve() != target.resolve():
                shutil.copy2(str(ckpt_found), str(target))
            return target

        return ckpt_found

    def _ensure_limix_import(self) -> None:
        if self._predictor_cls is not None and self._torch is not None:
            return

        limix_root = self._repo_root() / "libs" / "LimiX"
        if not limix_root.exists():
            raise RuntimeError(
                f"LimiX source not found at {limix_root}. Please clone it into libs/LimiX first."
            )

        root_str = str(limix_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        self._ensure_limix_dependencies()

        try:
            import torch  # noqa: PLC0415
            from inference.predictor import LimiXPredictor  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            missing_mod = _extract_missing_module_name(exc)
            dep_names = {
                str(pkg).strip()
                for pkg in (self._cfg.get("dependency_packages", self.DEFAULT_CONFIG["dependency_packages"]) or [])
                if str(pkg).strip()
            }
            # If import fails due a specific missing module, attempt a targeted
            # one-shot install in the same interpreter and retry import once.
            if (
                missing_mod
                and bool(self._cfg.get("auto_install_dependencies", True))
                and missing_mod in dep_names
            ):
                self._ensure_limix_dependencies(force_install=[missing_mod])
                try:
                    import torch  # noqa: PLC0415
                    from inference.predictor import LimiXPredictor  # noqa: PLC0415
                except Exception as exc_retry:  # noqa: BLE001
                    self._limix_import_error = exc_retry
                    raise RuntimeError(
                        "Failed to import LimiX predictor after dependency repair. "
                        "Install required deps (e.g. pandas/scipy/einops/kditransform/hyperopt). "
                        f"Active interpreter: {sys.executable}. Original error: {exc_retry}"
                    ) from exc_retry
                self._torch = torch
                self._predictor_cls = LimiXPredictor
                return
            self._limix_import_error = exc
            raise RuntimeError(
                "Failed to import LimiX predictor. "
                "Install required deps (e.g. pandas/scipy/einops/kditransform/hyperopt). "
                f"Active interpreter: {sys.executable}. Original error: {exc}"
            ) from exc

        self._torch = torch
        self._predictor_cls = LimiXPredictor

    def _ensure_limix_dependencies(self, force_install: Optional[Sequence[str]] = None) -> None:
        deps = self._cfg.get("dependency_packages", self.DEFAULT_CONFIG["dependency_packages"])
        if isinstance(deps, (str, bytes)) or not isinstance(deps, Sequence):
            return

        force_set = {
            str(pkg).strip()
            for pkg in (force_install or [])
            if str(pkg).strip()
        }

        missing = [
            str(pkg).strip()
            for pkg in deps
            if str(pkg).strip() and (str(pkg).strip() in force_set or not _module_importable(str(pkg).strip()))
        ]
        if not missing:
            return
        if not bool(self._cfg.get("auto_install_dependencies", True)):
            raise RuntimeError(
                "Missing LimiX dependencies: "
                + ", ".join(missing)
                + ". Install them in the active Python environment and retry."
            )
        cmd = [sys.executable, "-m", "pip", "install", *missing]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or "unknown pip failure"
            raise RuntimeError(
                "Failed to install missing LimiX dependencies "
                f"({', '.join(missing)}). pip output: {detail}"
            )

        still_missing = [pkg for pkg in missing if not _module_importable(pkg)]
        if still_missing:
            raise RuntimeError(
                "LimiX dependencies remain unavailable after pip install "
                f"({', '.join(still_missing)}). "
                f"Active interpreter: {sys.executable}"
            )

    def _resolve_device(self) -> Any:
        self._ensure_limix_import()
        assert self._torch is not None
        pref = str(self._cfg.get("device", "auto")).strip().lower()
        if pref == "auto":
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        return self._torch.device(pref)

    def _build_predictor(self) -> Any:
        self._ensure_limix_import()
        model_path = self._ensure_ckpt()

        cfg_path = self._resolve_path(str(self._cfg.get("inference_config_path", "")))
        if not cfg_path.exists():
            raise FileNotFoundError(f"LimiX inference config not found: {cfg_path}")

        assert self._predictor_cls is not None
        predictor = self._predictor_cls(
            device=self._resolve_device(),
            model_path=str(model_path),
            inference_config=str(cfg_path),
            mix_precision=bool(self._cfg.get("mix_precision", True)),
            outlier_remove_std=float(self._cfg.get("outlier_remove_std", 12.0)),
            softmax_temperature=float(self._cfg.get("softmax_temperature", 0.9)),
            mask_prediction=False,
            categorical_features_indices=self._cfg.get("categorical_features_indices", None),
            inference_with_DDP=bool(self._cfg.get("inference_with_DDP", False)),
            seed=int(self._cfg.get("seed", 42)),
        )
        return predictor

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        Xn = np.asarray(X, dtype=np.float32)
        yn = np.asarray(y)
        if Xn.ndim != 2:
            raise ValueError("LimiXSubjectClassifier expects a 2D feature matrix.")
        if len(Xn) == 0:
            raise ValueError("No training data provided for LimiX classifier.")

        classes = np.unique(yn)
        if classes.size < 2:
            raise ValueError("LimiXSubjectClassifier requires at least two classes.")

        self.classes_ = classes
        self._X_train = Xn.copy()
        self._y_train = yn.copy()
        self._predictor = None

        # validate dependencies early
        self._ensure_limix_import()
        ckpt_path = self._ensure_ckpt()

        return {
            "backend": "limix",
            "checkpoint": str(ckpt_path),
            "inference_config_path": str(self._resolve_path(str(self._cfg.get("inference_config_path", "")))),
            "n_train": int(Xn.shape[0]),
            "n_features": int(Xn.shape[1]),
            "n_classes": int(classes.size),
        }

    def predict_proba(self, X):  # noqa: ANN001
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("LimiX classifier is not fitted.")

        Xn = np.asarray(X, dtype=np.float32)
        if Xn.ndim != 2:
            raise ValueError("LimiXSubjectClassifier expects a 2D feature matrix.")

        if self._predictor is None:
            self._predictor = self._build_predictor()

        probs = self._predictor.predict(
            self._X_train,
            self._y_train,
            Xn,
            task_type="Classification",
        )

        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise RuntimeError(f"LimiX predict output must be 2D, got shape={probs.shape}")
        return probs

    def predict(self, X):  # noqa: ANN001
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        if self.classes_ is None:
            return pred_idx.astype(np.int64)
        return self.classes_[pred_idx]

    def save(self, path: str) -> None:
        # We intentionally do not persist train matrix by default to avoid
        # huge cache files in persistent classifier cache.
        raise RuntimeError("LimiXSubjectClassifier cache persistence is disabled by design.")

    def load(self, path: str) -> None:
        # Keep explicit failure so engine falls back to fit() cleanly.
        raise RuntimeError("LimiXSubjectClassifier cache loading is disabled by design.")
