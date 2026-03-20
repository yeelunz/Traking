from __future__ import annotations
from typing import Dict, Type, Any

PREPROC_REGISTRY: Dict[str, Any] = {}
MODEL_REGISTRY: Dict[str, Any] = {}
EVAL_REGISTRY: Dict[str, Any] = {}
FEATURE_EXTRACTOR_REGISTRY: Dict[str, Any] = {}
CLASSIFIER_REGISTRY: Dict[str, Any] = {}
FUSION_MODULE_REGISTRY: Dict[str, Any] = {}
SEGMENTATION_MODEL_REGISTRY: Dict[str, Any] = {}


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


def register_feature_extractor(name: str):
    def deco(cls):
        FEATURE_EXTRACTOR_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_classifier(name: str):
    def deco(cls):
        CLASSIFIER_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_fusion_module(name: str):
    def deco(cls):
        FUSION_MODULE_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_segmentation_model(name: str):
    def deco(cls):
        SEGMENTATION_MODEL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco
