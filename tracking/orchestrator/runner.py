from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, List, Callable, Optional
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from ..core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
from ..utils.annotations import load_coco_vid
from ..core.interfaces import FramePrediction
from ..data.dataset_manager import COCOJsonDatasetManager, SimpleDataset
# import built-in plugins to populate registries
from ..preproc import clahe  # noqa: F401
from ..models import template_matching  # noqa: F401
from ..models import csrt  # noqa: F401
from ..models import optical_flow_lk  # noqa: F401
from ..models import faster_rcnn  # noqa: F401
from ..models import yolov11  # noqa: F401
from ..models import fast_speckle  # noqa: F401
from ..eval import evaluator  # noqa: F401
from ..utils.env import capture_env
from ..utils.seed import set_seed
import traceback as _tb


class PipelineRunner:
    def __init__(self, config: Dict[str, Any], logger: Optional[Callable[[str], None]] = None, progress_cb: Optional[Callable[[str,int,int,Dict[str,Any]], None]] = None):
        self.cfg = config
        self.seed = int(config.get("seed", 0))
        dataset_cfg = config.get("dataset", {})
        self.dataset_root = dataset_cfg.get("root", ".")
        out_cfg = config.get("output", {}) or {}
        # If output missing or lacks results_root, default to project folder /results
        try:
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception:
            proj_root = os.getcwd()
        self.results_root = out_cfg.get("results_root") or os.path.join(proj_root, "results")
        os.makedirs(self.results_root, exist_ok=True)
        self._logger = logger
        self._log_file: Optional[str] = None
        self._progress_cb = progress_cb

    # lightweight wrapper to emit progress events
    def _progress(self, stage: str, current: int, total: int, extra: Optional[Dict[str,Any]] = None):
        if self._progress_cb:
            try:
                self._progress_cb(stage, int(current), int(total), extra or {})
            except Exception:
                pass

    def _timestamp_dir(self, name: str) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        d = os.path.join(self.results_root, f"{ts}_{name}")
        os.makedirs(d, exist_ok=True)
        return d

    def _log(self, msg: str, to_console: bool = True):
        # keep file logging always; console logging can be suppressed when tqdm bars are active
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        if to_console and self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    def run(self):
        # Only need reproducible dataset splits, not full deterministic ops (which can break CuBLAS).
        set_seed(self.seed, deterministic=False)
        dm = COCOJsonDatasetManager(self.dataset_root)
        ds_cfg = self.cfg.get("dataset", {})
        split_cfg = (ds_cfg or {}).get("split", {})
        method = split_cfg.get("method", "video_level")
        ratios = split_cfg.get("ratios", [0.8, 0.2])
        k_fold = int(split_cfg.get("k_fold", 1) or 1)
        self._log(f"Dataset root: {self.dataset_root}")
        self._log(f"Results root: {self.results_root}")
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
            # 允許在 config.output.skip_pip_freeze = true 時跳過 pip freeze 以避免卡住
            out_cfg = self.cfg.get("output", {}) or {}
            skip_freeze = bool(out_cfg.get("skip_pip_freeze", False))
            meta = {"config": exp, "seed": self.seed, "env": capture_env(skip_freeze=skip_freeze)}
            with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # init log file for this experiment
            logs_dir = os.path.join(out_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            self._log_file = os.path.join(logs_dir, "run.log")
            self._log(f"Started experiment: {exp_name}")
            self._log(f"Experiment folder: {out_dir}")

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
                    try:
                        m = cls(step.get("params", {}))
                        if hasattr(m, "preprocs"):
                            setattr(m, "preprocs", preprocs)
                        # --- 新增：記錄模型實際使用的參數快照，方便偵錯 UI 傳參是否正確 ---
                        try:
                            # 可能的參數名稱集合（通用 + FasterRCNN 常見）
                            cand_keys = {
                                'device','pretrained','num_classes','epochs','batch_size','lr','weight_decay','momentum',
                                'optimizer_name','optimizer','step_size','gamma','grad_clip','detect_anomaly','include_empty_frames',
                                'fallback_last_prediction','score_thresh','adamw_betas','adamw_eps','inference_batch','num_workers','pin_memory'
                            }
                            snap = {}
                            for k in sorted(cand_keys):
                                if hasattr(m, k):
                                    v = getattr(m, k)
                                    # 避免張量 / 模型本體被序列化
                                    if not hasattr(v, 'parameters') and not isinstance(v, (type(m.model),)):
                                        snap[k] = v
                            self._log(f"[ModelConfig] model={step['name']} params={snap}")
                        except Exception:
                            pass
                        ms.append((step["name"], m))
                    except Exception as _e_inst:
                        self._log(f"[ERROR] Model init failed | model={step['name']} error={_e_inst}\n{_tb.format_exc()}")
                        # fallback placeholder to keep pipeline running and produce result files
                        class _UnavailableModel:
                            def __init__(self, reason: str):
                                self.name = step["name"]
                                self.reason = reason
                                self.preprocs = []
                            def train(self, *a, **k):
                                return {"status": "unavailable", "reason": self.reason}
                            def predict(self, *a, **k):
                                return []
                            def load_checkpoint(self, *a, **k):
                                pass
                        ms.append((step["name"], _UnavailableModel(str(_e_inst))))
                return ms
            models = build_models()
            # attach model-level progress callbacks (per-epoch)
            for name, m in models:
                try:
                    setattr(m, 'progress_callback', lambda stage, cur, tot, model=name: self._progress(stage, cur, tot, {'model': model}))
                except Exception:
                    pass
            self._log(f"Models: {[name for name,_ in models]}")

            # evaluator
            eval_name = self.cfg.get("evaluation", {}).get("evaluator", "BasicEvaluator")
            eval_cls = EVAL_REGISTRY.get(eval_name)
            evaluator = eval_cls() if eval_cls else None

            def run_on_dataset(dataset, out_dir_base: str, models_list=None, phase: str = 'eval'):
                # initialize containers and ensure per-model predictions files are always created
                predictions_all = {}
                # aggregate across entire dataset (all videos) per model
                dataset_agg = {}
                met_dir = os.path.join(out_dir_base, "metrics")
                os.makedirs(met_dir, exist_ok=True)
                eval_cfg = self.cfg.get("evaluation", {}) or {}
                restrict_to_gt_frames = bool(eval_cfg.get("restrict_to_gt_frames", True))
                viz_cfg = eval_cfg.get("visualize", {}) or {}
                viz_enabled = bool(viz_cfg.get("enabled", False))
                viz_samples = int(viz_cfg.get("samples", 10) or 10)
                # Determine models used and initialize prediction collectors
                use_models = models_list or models
                for model_name, _ in use_models:
                    predictions_all.setdefault(model_name, [])
                # iterate videos with optional progress bar
                iterable = dataset
                total_videos = None
                try:
                    total_videos = len(dataset)  # type: ignore
                except Exception:
                    total_videos = None
                if tqdm is not None:
                    iterable = tqdm(dataset, total=total_videos, desc="Videos", unit="vid")
                idx_video = 0
                for video_item in iterable:
                    idx_video += 1
                    if total_videos:
                        self._progress('eval_video', idx_video, total_videos, {'phase': phase, 'exp': exp_name})
                    vp = video_item["video_path"]
                    gt_json = os.path.splitext(vp)[0] + ".json"
                    gt = load_coco_vid(gt_json) if os.path.exists(gt_json) else {"frames": {}}
                    # reduce noisy console logs during progress; keep in file only when tqdm is present
                    self._log(f"Predicting video: {os.path.basename(vp)}", to_console=(tqdm is None))
                    pv_predictions = {}
                    # per-video per-model bar
                    model_iter = use_models
                    if tqdm is not None and len(use_models) > 1:
                        model_iter = tqdm(use_models, desc=f"Models@{os.path.basename(vp)}", unit="mdl", leave=False)
                    for model_name, model in model_iter:
                        try:
                            # If we only evaluate GT frames and the model supports sparse inference,
                            # request predictions only on those frames (useful for detection-based ML models)
                            if restrict_to_gt_frames and hasattr(model, 'predict_frames'):
                                gt_frames_sorted = sorted([int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes])
                                preds = model.predict_frames(vp, gt_frames_sorted)  # type: ignore[attr-defined]
                            else:
                                preds = model.predict(vp)
                        except Exception as e:
                            # Log the failure and continue to next model/video
                            self._log(f"[ERROR] Predict failed | model={model_name} video={os.path.basename(vp)} error={e}")
                            preds = []
                        pv_predictions[model_name] = preds
                        predictions_all.setdefault(model_name, []).extend(preds)
                    if evaluator:
                        # Filter predictions to GT frames only for evaluation (avoid penalizing unannotated frames)
                        if restrict_to_gt_frames:
                            gt_frames_set = {int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes}
                            eval_pv_predictions = {}
                            if gt_frames_set:
                                min_gt = min(gt_frames_set)
                            for mn, pl in pv_predictions.items():
                                # original filter (no shift)
                                direct = [p for p in pl if int(getattr(p, 'frame_index', -1)) in gt_frames_set]
                                best = direct
                                # try alignment by constant offset: delta = min_gt - min_pred
                                try:
                                    if gt_frames_set and pl:
                                        min_pred = int(min(int(getattr(p, 'frame_index', 0)) for p in pl))
                                        delta = int(min_gt) - int(min_pred)
                                        if delta != 0:
                                            shifted = [FramePrediction(int(p.frame_index) + delta, p.bbox, p.score) for p in pl]
                                            shifted_f = [p for p in shifted if int(getattr(p, 'frame_index', -1)) in gt_frames_set]
                                            # choose the one with higher coverage on GT frames
                                            if len(shifted_f) > len(direct):
                                                best = shifted_f
                                except Exception:
                                    pass
                                eval_pv_predictions[mn] = best
                        else:
                            eval_pv_predictions = pv_predictions
                        # per-video evaluation directory to avoid overwriting
                        vid_stem = os.path.splitext(os.path.basename(vp))[0]
                        vid_met_dir = os.path.join(met_dir, vid_stem)
                        os.makedirs(vid_met_dir, exist_ok=True)
                        res = evaluator.evaluate(eval_pv_predictions, gt, vid_met_dir)
                        # accumulate for dataset-level summary
                        for model_name, sm in res.items():
                            agg = dataset_agg.setdefault(model_name, {
                                "count": 0,
                                "sum_iou": 0.0, "sum_iou_sq": 0.0,
                                "sum_ce": 0.0, "sum_ce_sq": 0.0,
                                # detection aggregation
                                "tp_50": 0, "fp_50": 0, "fn_50": 0,
                                "tp_75": 0, "fp_75": 0, "fn_75": 0,
                                # EAO aggregation (average across videos)
                                "sum_EAO": 0.0, "videos": 0,
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
                            # EAO per-video scalar average
                            if "EAO" in sm:
                                agg["sum_EAO"] += float(sm.get("EAO", 0.0))
                                agg["videos"] += 1
                        self._log(f"Evaluated metrics written to: {vid_met_dir}", to_console=(tqdm is None))

                        # Optional visualization: draw up to N GT frames per video with GT/pred boxes
                        if viz_enabled:
                            try:
                                import cv2  # type: ignore
                                vis_dir = os.path.join(out_dir_base, "visualizations", vid_stem)
                                os.makedirs(vis_dir, exist_ok=True)
                                gt_frames_sorted = sorted([int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes])
                                if viz_samples > 0:
                                    gt_frames_sorted = gt_frames_sorted[:min(viz_samples, len(gt_frames_sorted))]
                                # preload predictions by frame for each model
                                eval_pred_maps = {}
                                for mn, pl in (eval_pv_predictions.items() if restrict_to_gt_frames else pv_predictions.items()):
                                    m = {}
                                    for p in pl:
                                        m[int(p.frame_index)] = p.bbox
                                    eval_pred_maps[mn] = m
                                cap = cv2.VideoCapture(vp)
                                for fi in gt_frames_sorted:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                                    ok, frame = cap.read()
                                    if not ok:
                                        continue
                                    # draw GT boxes in green
                                    for gtb in gt.get("frames", {}).get(fi, []) or []:
                                        x, y, w, h = map(float, gtb)
                                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                                    # draw per-model pred in red (one per model)
                                    for mn, fmap in eval_pred_maps.items():
                                        pb = fmap.get(fi)
                                        if pb is None:
                                            continue
                                        x, y, w, h = map(float, pb)
                                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)
                                        cv2.putText(frame, mn, (int(x), max(0, int(y)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                                    out_path = os.path.join(vis_dir, f"frame_{fi:06d}.jpg")
                                    try:
                                        cv2.imwrite(out_path, frame)
                                    except Exception:
                                        pass
                                cap.release()
                                self._log(f"Visualization images saved: {vis_dir}", to_console=(tqdm is None))
                            except Exception as _e_viz:
                                self._log(f"[Warn] Visualization failed: {_e_viz}", to_console=(tqdm is None))
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
                        # Average EAO across videos (if any present)
                        vcnt = max(1, int(a.get("videos", 0)))
                        eao_mean = float(a.get("sum_EAO", 0.0)) / vcnt if int(a.get("videos", 0)) > 0 else 0.0
                        summary_out[model_name] = {
                            "frames_count": n,
                            "iou_mean": i_mu, "iou_std": i_sd,
                            "ce_mean": c_mu, "ce_std": c_sd,
                            "mAP_50": map50, "mAP_75": map75,
                            "EAO": eao_mean,
                        }
                    summary_path = os.path.join(met_dir, "summary.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary_out, f, ensure_ascii=False, indent=2)
                    self._log(f"Dataset metrics summary saved: {summary_path}", to_console=(tqdm is None))
                pred_dir = os.path.join(out_dir_base, "predictions")
                os.makedirs(pred_dir, exist_ok=True)
                for model_name, preds in predictions_all.items():
                    outp = os.path.join(pred_dir, f"{model_name}.json")
                    with open(outp, "w", encoding="utf-8") as f:
                        json.dump([
                            {"frame_index": p.frame_index, "bbox": list(p.bbox), "score": p.score} for p in preds
                        ], f, ensure_ascii=False, indent=2)
                    self._log(f"Predictions saved: {outp}", to_console=(tqdm is None))

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
                fold_iter = range(k_fold)
                if tqdm is not None:
                    fold_iter = tqdm(range(k_fold), total=k_fold, desc="K-Fold", unit="fold")
                for fi in fold_iter:
                    self._progress('kfold_fold', fi+1, k_fold, {'exp': exp_name})
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
                            self._log(f"[Train] Fold {fi+1}/{k_fold} | model={model_name} | train_videos={len(trn_ds)} val_videos={len(val_ds)}")
                            try:
                                ret = model.train(trn_ds, val_ds, seed=self.seed, output_dir=os.path.join(fold_dir, "train"))
                                if isinstance(ret, dict):
                                    self._log(f"[Train] Result: {ret}", to_console=(tqdm is None))
                            except Exception as e:
                                self._log(f"[ERROR] Training failed | model={model_name} fold={fi+1} error={e}\n{_tb.format_exc()}")
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
                        agg.setdefault(model_name, {"iou_mean": [], "ce_mean": [], "mAP_50": [], "mAP_75": [], "EAO": []})
                        agg[model_name]["iou_mean"].append(m.get("iou_mean", 0.0))
                        agg[model_name]["ce_mean"].append(m.get("ce_mean", 0.0))
                        if "mAP_50" in m:
                            agg[model_name]["mAP_50"].append(m.get("mAP_50", 0.0))
                        if "mAP_75" in m:
                            agg[model_name]["mAP_75"].append(m.get("mAP_75", 0.0))
                        if "EAO" in m:
                            agg[model_name]["EAO"].append(m.get("EAO", 0.0))
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
                        "EAO_mean": mean_std(vals.get("EAO", []))[0],
                        "EAO_std": mean_std(vals.get("EAO", []))[1],
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
                self._log("K-Fold validation finished. Aggregates saved.", to_console=(tqdm is None))

            # final training on full train and evaluate on test
            final_models = build_models()
            final_train_dir = os.path.join(out_dir, "train_full")
            os.makedirs(final_train_dir, exist_ok=True)
            for model_name, model in final_models:
                if hasattr(model, "train"):
                    self._log(f"[Train] Full train | model={model_name} | train_videos={len(train_ds)}")
                    try:
                        ret = model.train(train_ds, None, seed=self.seed, output_dir=final_train_dir)
                        if isinstance(ret, dict):
                            self._log(f"[Train] Result: {ret}", to_console=(tqdm is None))
                            if ret.get("status") == "no_data":
                                        self._log("[Warn] No training samples found (check that each video has a matching .json annotation next to it).", to_console=(tqdm is None))
                    except Exception as e:
                        self._log(f"[ERROR] Training failed | model={model_name} error={e}\n{_tb.format_exc()}")
            test_dir = os.path.join(out_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            run_on_dataset(test_ds, test_dir, models_list=final_models)
