from tracking.orchestrator.runner import PipelineRunner


def _base_cfg(**overrides):
    cfg = {
        "dataset": {"root": "."},
        "experiments": [
            {
                "name": "smoke",
                "pipeline": [
                    {"type": "model", "name": "TemplateMatching", "params": {}},
                ],
            }
        ],
        "evaluation": {"evaluator": "BasicEvaluator"},
        "output": {"results_root": "./results"},
    }
    cfg.update(overrides)
    return cfg


def test_detection_only_uses_global_flag_when_experiment_not_set():
    runner = PipelineRunner(_base_cfg(detection_only=True))
    assert runner._resolve_detection_only_mode({}) is True


def test_detection_only_experiment_flag_overrides_global_flag():
    runner = PipelineRunner(_base_cfg(detection_only=True))
    assert runner._resolve_detection_only_mode({"detection_only": False}) is False


def test_detection_only_accepts_mode_keyword():
    runner = PipelineRunner(_base_cfg())
    assert runner._resolve_detection_only_mode({"mode": "detection_only"}) is True
    assert runner._resolve_detection_only_mode({"mode": "detector_only"}) is True


def test_detection_only_accepts_string_bool_values():
    runner = PipelineRunner(_base_cfg())
    assert runner._resolve_detection_only_mode({"detection_only": "yes"}) is True
    assert runner._resolve_detection_only_mode({"detection_only": "off"}) is False
