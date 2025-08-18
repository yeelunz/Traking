# Video Tracking Framework (Skeleton)

This is a minimal, extensible framework to run tracking experiments against videos and annotations exported by the label tool in `labelTool/`.

Key pieces:
- tracking/core: Interfaces and registries
- tracking/data: Dataset manager for COCO-VID-like JSON next to videos
- tracking/preproc: Example CLAHE module
- tracking/models: Example Template Matching tracker (no training)
- tracking/eval: Basic evaluator computing IoU and center error
- tracking/orchestrator: Pipeline runner to glue everything

Quick start:
1. Place videos and `<video>.json` exported from the label tool in a dataset root.
2. Edit `pipeline.example.yaml` to point to your dataset root and results path.
3. Install dependencies (see below) and run:

```bat
python run_pipeline.py --config pipeline.example.yaml
```

Dependencies:
- Python 3.9+
- opencv-python
- numpy
- pyyaml (if using YAML configs)
 - PySide6 (optional, for the simple UI)
 - matplotlib (for plots and k-fold aggregates)

Notes on trackers:
- CSRT tracker requires `opencv-contrib-python` (cv2.legacy). If you want to use CSRT, uninstall opencv-python and install the contrib build instead.

Install:
```bat
pip install -r requirements.txt
```

Notes:
- The TemplateMatching model is a simple baseline, mainly to validate the pipeline.
- Extend by registering new preproc or model classes using the registries.

UI (optional):
```bat
python ui.py
```
Use the UI to load/edit a YAML/JSON config and run the pipeline with live logs.

How splitting works:
- Provide only train/test ratios (defaults to [0.8, 0.2]).
- If k_fold > 1, k-fold validation is performed within the training split for model selection/validation.
- After k-fold, the chosen model is trained on the full training split and evaluated on the test split.
