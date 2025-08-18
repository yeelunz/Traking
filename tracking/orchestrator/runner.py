from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Callable, Optional

from ..core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
from ..utils.annotations import load_coco_vid
from ..data.dataset_manager import COCOJsonDatasetManager, SimpleDataset
# import built-in plugins to populate registries
from ..preproc import clahe  # noqa: F401
from ..models import template_matching  # noqa: F401
from ..models import csrt  # noqa: F401
from ..models import optical_flow_lk  # noqa: F401
from ..eval import evaluator  # noqa: F401
from ..utils.env import capture_env
from ..utils.seed import set_seed


class PipelineRunner:
    def __init__(self, config: Dict[str, Any], logger: Optional[Callable[[str], None]] = None):
        self.cfg = config
        self.seed = int(config.get("seed", 0))
        dataset_cfg = config.get("dataset", {})
        self.dataset_root = dataset_cfg.get("root", ".")
        self.results_root = config.get("output", {}).get("results_root", os.path.join(os.getcwd(), "results"))
        os.makedirs(self.results_root, exist_ok=True)
        self._logger = logger
        self._log_file: Optional[str] = None

    def _timestamp_dir(self, name: str) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        d = os.path.join(self.results_root, f"{ts}_{name}")
        os.makedirs(d, exist_ok=True)
        return d

    def _log(self, msg: str):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    def run(self):
        set_seed(self.seed)
        dm = COCOJsonDatasetManager(self.dataset_root)
        ds_cfg = self.cfg.get("dataset", {})
        split_cfg = (ds_cfg or {}).get("split", {})
        method = split_cfg.get("method", "video_level")
        ratios = split_cfg.get("ratios", [0.8, 0.2])
        k_fold = int(split_cfg.get("k_fold", 1) or 1)
        self._log(f"Dataset root: {self.dataset_root}")
        # accept [train,test] or [train,val,test]; always map to (train, 0.0, test)
        if isinstance(ratios, list) and len(ratios) == 2:
            train_r, test_r = float(ratios[0]), float(ratios[1])
        elif isinstance(ratios, list) and len(ratios) == 3:
            train_r, test_r = float(ratios[0]), float(ratios[2])
        else:
            train_r, test_r = 0.8, 0.2
        split_tt = dm.split(method=method, seed=self.seed, ratios=(train_r, 0.0, test_r))
        train_ds = split_tt["train"]
        test_ds = split_tt["test"]
        self._log("Dataset split created (train/test).")

        # orchestrate experiments
        for exp in self.cfg.get("experiments", []):
            exp_name = exp.get("name", "exp")
            out_dir = self._timestamp_dir(exp_name)
            meta = {"config": exp, "seed": self.seed, "env": capture_env()}
            with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # init log file for this experiment
            logs_dir = os.path.join(out_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            self._log_file = os.path.join(logs_dir, "run.log")
            self._log(f"Started experiment: {exp_name}")

            # build preproc chain
            preprocs = []
            for step in exp.get("pipeline", []):
                if step.get("type") == "preproc":
                    cls = PREPROC_REGISTRY[step["name"]]
                    preprocs.append(cls(step.get("params", {})))
            self._log(f"Preprocs: {[type(p).__name__ for p in preprocs]}")

            # factory for model(s) to ensure fresh instance per fold/final
            model_steps = [step for step in exp.get("pipeline", []) if step.get("type") == "model"]
            def build_models():
                ms = []
                for step in model_steps:
                    cls = MODEL_REGISTRY[step["name"]]
                    m = cls(step.get("params", {}))
                    if hasattr(m, "preprocs"):
                        setattr(m, "preprocs", preprocs)
                    ms.append((step["name"], m))
                return ms
            models = build_models()
            self._log(f"Models: {[name for name,_ in models]}")

            # evaluator
            eval_name = self.cfg.get("evaluation", {}).get("evaluator", "BasicEvaluator")
            eval_cls = EVAL_REGISTRY.get(eval_name)
            evaluator = eval_cls() if eval_cls else None

            def run_on_dataset(dataset, out_dir_base: str, models_list=None):
                predictions_all = {}
                # aggregate across entire dataset (all videos) per model
                dataset_agg = {}
                met_dir = os.path.join(out_dir_base, "metrics")
                os.makedirs(met_dir, exist_ok=True)
                for video_item in dataset:
                    vp = video_item["video_path"]
                    gt_json = os.path.splitext(vp)[0] + ".json"
                    gt = load_coco_vid(gt_json) if os.path.exists(gt_json) else {"frames": {}}
                    self._log(f"Predicting video: {os.path.basename(vp)}")
                    pv_predictions = {}
                    use_models = models_list or models
                    for model_name, model in use_models:
                        try:
                            preds = model.predict(vp)
                        except Exception as e:
                            # Log the failure and continue to next model/video
                            self._log(f"[ERROR] Predict failed | model={model_name} video={os.path.basename(vp)} error={e}")
                            preds = []
                        pv_predictions[model_name] = preds
                        predictions_all.setdefault(model_name, []).extend(preds)
                    if evaluator:
                        # per-video evaluation directory to avoid overwriting
                        vid_stem = os.path.splitext(os.path.basename(vp))[0]
                        vid_met_dir = os.path.join(met_dir, vid_stem)
                        os.makedirs(vid_met_dir, exist_ok=True)
                        res = evaluator.evaluate(pv_predictions, gt, vid_met_dir)
                        # accumulate for dataset-level summary
                        for model_name, sm in res.items():
                            agg = dataset_agg.setdefault(model_name, {
                                "count": 0,
                                "sum_iou": 0.0, "sum_iou_sq": 0.0,
                                "sum_ce": 0.0, "sum_ce_sq": 0.0,
                                # detection aggregation
                                "tp_50": 0, "fp_50": 0, "fn_50": 0,
                                "tp_75": 0, "fp_75": 0, "fn_75": 0,
                            })
                            agg["count"] += int(sm.get("count", 0))
                            agg["sum_iou"] += float(sm.get("sum_iou", 0.0))
                            agg["sum_iou_sq"] += float(sm.get("sum_iou_sq", 0.0))
                            agg["sum_ce"] += float(sm.get("sum_ce", 0.0))
                            agg["sum_ce_sq"] += float(sm.get("sum_ce_sq", 0.0))
                            agg["tp_50"] += int(sm.get("tp_50", 0))
                            agg["fp_50"] += int(sm.get("fp_50", 0))
                            agg["fn_50"] += int(sm.get("fn_50", 0))
                            agg["tp_75"] += int(sm.get("tp_75", 0))
                            agg["fp_75"] += int(sm.get("fp_75", 0))
                            agg["fn_75"] += int(sm.get("fn_75", 0))
                        self._log(f"Evaluated metrics written to: {vid_met_dir}")
                # dataset-level summary across all videos
                if evaluator and dataset_agg:
                    summary_out = {}
                    for model_name, a in dataset_agg.items():
                        n = max(1, int(a.get("count", 0)))
                        i_mu = a["sum_iou"] / n
                        # E[X^2] - (E[X])^2, guard non-negative
                        i_var = max(0.0, a["sum_iou_sq"] / n - i_mu * i_mu)
                        i_sd = i_var ** 0.5
                        c_mu = a["sum_ce"] / n
                        c_var = max(0.0, a["sum_ce_sq"] / n - c_mu * c_mu)
                        c_sd = c_var ** 0.5
                        # micro-precision as AP at single threshold
                        tp50, fp50 = int(a.get("tp_50", 0)), int(a.get("fp_50", 0))
                        tp75, fp75 = int(a.get("tp_75", 0)), int(a.get("fp_75", 0))
                        map50 = (tp50 / (tp50 + fp50)) if (tp50 + fp50) > 0 else 0.0
                        map75 = (tp75 / (tp75 + fp75)) if (tp75 + fp75) > 0 else 0.0
                        summary_out[model_name] = {
                            "frames_count": n,
                            "iou_mean": i_mu, "iou_std": i_sd,
                            "ce_mean": c_mu, "ce_std": c_sd,
                            "mAP_50": map50, "mAP_75": map75,
                        }
                    with open(os.path.join(met_dir, "summary.json"), "w", encoding="utf-8") as f:
                        json.dump(summary_out, f, ensure_ascii=False, indent=2)
                pred_dir = os.path.join(out_dir_base, "predictions")
                os.makedirs(pred_dir, exist_ok=True)
                for model_name, preds in predictions_all.items():
                    with open(os.path.join(pred_dir, f"{model_name}.json"), "w", encoding="utf-8") as f:
                        json.dump([
                            {"frame_index": p.frame_index, "bbox": list(p.bbox), "score": p.score} for p in preds
                        ], f, ensure_ascii=False, indent=2)

            # k-fold within training for validation
            if k_fold > 1:
                self._log(f"Running {k_fold}-Fold validation on training set…")
                # build folds from train_ds items
                train_vids = [train_ds[i]["video_path"] for i in range(len(train_ds))]
                # reproducible shuffle
                import random
                rng = random.Random(self.seed)
                vids = train_vids[:]
                rng.shuffle(vids)
                fold_size = max(1, len(vids) // k_fold)
                fold_summaries = []
                for fi in range(k_fold):
                    val_vids = vids[fi * fold_size:(fi + 1) * fold_size]
                    trn_vids = [v for v in vids if v not in val_vids]
                    val_ds = SimpleDataset(val_vids, dm.ann_by_video)
                    trn_ds = SimpleDataset(trn_vids, dm.ann_by_video)
                    fold_dir = os.path.join(out_dir, f"fold_{fi+1}")
                    os.makedirs(fold_dir, exist_ok=True)
                    # fresh models per fold
                    fold_models = build_models()
                    # optional training step
                    for model_name, model in fold_models:
                        if hasattr(model, "train"):
                            try:
                                model.train(trn_ds, val_ds, seed=self.seed, output_dir=os.path.join(fold_dir, "train"))
                            except Exception:
                                pass
                    run_on_dataset(val_ds, fold_dir, models_list=fold_models)
                    try:
                        with open(os.path.join(fold_dir, "metrics", "summary.json"), "r", encoding="utf-8") as f:
                            fold_summaries.append(json.load(f))
                    except Exception:
                        pass
                # aggregate stats across folds
                agg = {}
                for sm in fold_summaries:
                    for model_name, m in sm.items():
                        agg.setdefault(model_name, {"iou_mean": [], "ce_mean": [], "mAP_50": [], "mAP_75": []})
                        agg[model_name]["iou_mean"].append(m.get("iou_mean", 0.0))
                        agg[model_name]["ce_mean"].append(m.get("ce_mean", 0.0))
                        if "mAP_50" in m:
                            agg[model_name]["mAP_50"].append(m.get("mAP_50", 0.0))
                        if "mAP_75" in m:
                            agg[model_name]["mAP_75"].append(m.get("mAP_75", 0.0))
                comp_dir = os.path.join(out_dir, "comparison")
                os.makedirs(comp_dir, exist_ok=True)
                agg_out = {}
                for model_name, vals in agg.items():
                    def mean_std(arr):
                        if not arr:
                            return (0.0, 0.0)
                        mu = sum(arr) / len(arr)
                        sd = (sum((x - mu) ** 2 for x in arr) / len(arr)) ** 0.5
                        return (mu, sd)
                    i_mu, i_sd = mean_std(vals["iou_mean"])
                    c_mu, c_sd = mean_std(vals["ce_mean"])
                    agg_out[model_name] = {
                        "iou_mean_mean": i_mu, "iou_mean_std": i_sd,
                        "ce_mean_mean": c_mu, "ce_mean_std": c_sd,
                        "mAP_50_mean": mean_std(vals.get("mAP_50", []))[0],
                        "mAP_50_std": mean_std(vals.get("mAP_50", []))[1],
                        "mAP_75_mean": mean_std(vals.get("mAP_75", []))[0],
                        "mAP_75_std": mean_std(vals.get("mAP_75", []))[1],
                    }
                with open(os.path.join(comp_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(agg_out, f, ensure_ascii=False, indent=2)
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                    labels = list(agg_out.keys())
                    # IoU bar
                    i_means = [agg_out[k]["iou_mean_mean"] for k in labels]
                    i_stds = [agg_out[k]["iou_mean_std"] for k in labels]
                    plt.figure(); plt.bar(labels, i_means, yerr=i_stds, capsize=5)
                    plt.ylabel("IoU mean (±std)"); plt.title("K-Fold Aggregate IoU")
                    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                    plt.savefig(os.path.join(comp_dir, "kfold_iou_bar.png")); plt.close()
                    # CE bar
                    c_means = [agg_out[k]["ce_mean_mean"] for k in labels]
                    c_stds = [agg_out[k]["ce_mean_std"] for k in labels]
                    plt.figure(); plt.bar(labels, c_means, yerr=c_stds, capsize=5)
                    plt.ylabel("Center Error (px) mean (±std)"); plt.title("K-Fold Aggregate Center Error")
                    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                    plt.savefig(os.path.join(comp_dir, "kfold_ce_bar.png")); plt.close()
                except Exception:
                    pass
                self._log("K-Fold validation finished. Aggregates saved.")

            # final training on full train and evaluate on test
            final_models = build_models()
            final_train_dir = os.path.join(out_dir, "train_full")
            os.makedirs(final_train_dir, exist_ok=True)
            for model_name, model in final_models:
                if hasattr(model, "train"):
                    try:
                        model.train(train_ds, None, seed=self.seed, output_dir=final_train_dir)
                    except Exception:
                        pass
            test_dir = os.path.join(out_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            run_on_dataset(test_ds, test_dir, models_list=final_models)
