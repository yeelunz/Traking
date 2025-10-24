from tracking.orchestrator.runner import PipelineRunner
from tracking.core.registry import MODEL_REGISTRY, CLASSIFIER_REGISTRY
from tracking.classification import classifiers as _cls  # noqa: F401


def test_imports():
    cfg = {
        "dataset": {"root": "."},
        "experiments": [
            {"name": "t", "pipeline": [
                {"type": "preproc", "name": "CLAHE", "params": {}},
                {"type": "model", "name": "TemplateMatching", "params": {}},
            ]}
        ],
        "evaluation": {"evaluator": "BasicEvaluator"},
        "output": {"results_root": "./results"}
    }
    runner = PipelineRunner(cfg)
    assert runner is not None


def test_strongsort_registered():
    assert "StrongSORT" in MODEL_REGISTRY


def test_tamos_registered():
    assert "TaMOs" in MODEL_REGISTRY


def test_classifiers_registered():
    assert "random_forest" in CLASSIFIER_REGISTRY
