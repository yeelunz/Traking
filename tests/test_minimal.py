from tracking.orchestrator.runner import PipelineRunner


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
