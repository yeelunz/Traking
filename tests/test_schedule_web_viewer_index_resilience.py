import json
from pathlib import Path

from tools.schedule_web_viewer.app import ExperimentIndex


def _write_metadata(exp_path: Path, name: str) -> None:
    exp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": {"name": name, "pipeline": []},
        "dataset": {"split": {"method": "holdout"}},
    }
    (exp_path / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_refresh_skips_broken_experiment_without_crashing(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    bad_exp = results_root / "exp_bad"
    good_exp = results_root / "exp_good"
    _write_metadata(bad_exp, "exp_bad")
    _write_metadata(good_exp, "exp_good")

    def _fake_has_segmentation_visuals(exp_path: Path) -> bool:
        if exp_path.name == "exp_bad":
            raise OSError("simulated broken path")
        return False

    monkeypatch.setattr(
        ExperimentIndex,
        "_has_segmentation_visuals",
        staticmethod(_fake_has_segmentation_visuals),
    )

    index = ExperimentIndex(results_root)

    assert "exp_good" in index.entries
    assert index.entries["exp_good"]["name"] == "exp_good"
