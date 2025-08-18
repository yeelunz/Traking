from __future__ import annotations
from typing import Dict, Type, Any

PREPROC_REGISTRY: Dict[str, Any] = {}
MODEL_REGISTRY: Dict[str, Any] = {}
EVAL_REGISTRY: Dict[str, Any] = {}


def register_preproc(name: str):
    def deco(cls):
        PREPROC_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_evaluator(name: str):
    def deco(cls):
        EVAL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco
