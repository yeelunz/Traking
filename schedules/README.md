# Schedules（排程檔）撰寫規格

本文件描述專案「排程 YAML/JSON」的可用結構與規則，目標是：
- 讓 UI 的 Queue（多組排程批次執行）與 CLI（單次執行）都能寫出**可被程式正確讀取**的配置。
- 明確定義本專案的前處理作用域控制：**僅使用 `preproc_scheme`（A/B/C）**，不支援 per-step `scope`。

> 來源：`tracking/orchestrator/runner.py`（PipelineRunner）與 `runner_ui/mixins/queue_mixin.py`（UI Queue 匯入）。

---

## 1) 兩種檔案形態

### A. 單一 config（CLI 直接跑）

`run_pipeline.py` 期待你提供的是「單一 config dict」，例如：

```yaml
seed: 42
dataset:
  root: C:\path\to\dataset
  split:
    method: loso
    loso: true
experiments:
  - name: my_exp
    preproc_scheme: A
    pipeline:
      - type: model
        name: YOLOv11
        params: {weights: models/detection/yolo11l.pt}
evaluation:
  evaluator: BasicEvaluator
segmentation:
  model:
    name: nnunet
    params: {}
output:
  results_root: results
```

CLI 執行方式：

```bash
python run_pipeline.py --config schedules/standard_exper_1217_loso_scheme12.yaml
```

> 注意：上面這個檔案本身是 Queue 格式（見下），CLI 若直接讀整份 `queue:` 會拿到 dict 但 runner 期待頂層是 config；
> 因此 CLI 使用時，請改成「只存單一 config」的檔案，或把 `queue` 其中某個 item 的 `config` 另存成單檔。


### B. Queue schedule（UI 批次匯入/跑多組）

UI 匯入排程檔時，會讀取：

```yaml
queue:
  - label: "01_xxx"
    config: { ...單一 config... }
  - label: "02_yyy"
    config: { ...單一 config... }
```

Queue item 的 `config` **就是**上面 A 的「單一 config」。

額外規則（UI 專用）：
- 如果某個 `config.experiments` 有多於 1 個 experiment，UI 會自動拆成多個 queue item（每次只跑 1 個 experiment）。
- `label` 只是顯示與辨識用，不影響 runner 行為。

參考範例：`schedules/standard_exper_1217_loso_scheme12.yaml`

---

## 2) Config 頂層 schema（runner 會讀的 key）

以下為常見、且確定會被 `PipelineRunner` 使用到的欄位：

- `seed: int`
- `dataset: { root: str, split: {...} }`
- `experiments: [ { name, preproc_scheme, pipeline } , ... ]`
- `detector: {...}`（通常是給 YOLOv11 的 `params` 用的 anchor；runner 不會自動套用，需你在 `pipeline` 內引用）
- `segmentation: {...}`
- `evaluation: {...}`
- `classification: {...}`（若要 subject classification）
- `output: { results_root: str, skip_pip_freeze?: bool }`

最小可跑（只跑 detection+eval）的必要條件通常是：
- `dataset.root` 指到資料集
- `experiments[0].pipeline` 至少含一個 `type: model`

---

## 3) dataset / split（特別是 LOSO）

### dataset
- `dataset.root: str`
  - 指向資料集根目錄（內含影片與同名 `.json` 標註）。

### split
Runner 會讀：
- `dataset.split.method: str`（預設 `video_level`）
- `dataset.split.loso: bool`（可直接設 `true`）
- `dataset.split.ratios: [train, test]` 或 `[train, val, test]`
- `dataset.split.k_fold: int`（>1 會啟用 detection 的 k-fold 訓練評估流程）

LOSO 推薦寫法（兩者擇一即可）：

```yaml
dataset:
  split:
    method: loso
```

或：

```yaml
dataset:
  split:
    loso: true
```

> 備註：對 LOSO 而言，`ratios` 幾乎不影響 fold 的切法；主要由資料集的 subject 分組決定。

---

## 4) experiments / pipeline

### experiments
`experiments` 是 list，每個 experiment 都會在每個 fold 內各跑一次（LOSO 時是每個 subject 一次）。

每個 experiment 支援的重點欄位：
- `name: str`（用於輸出資料夾命名）
- `preproc_scheme: str`（**必用，且只支援 A/B/C 或 GLOBAL/ROI/HYBRID**）
- `pipeline: list[step]`

### preproc_scheme（重要）
本專案目前採 **scheme-only**，不支援 per-step `scope`。

- `A` / `GLOBAL`：前處理套在 full frame（影響 detection + segmentation）
- `B` / `ROI`：前處理只套在 ROI crop 後（只影響 segmentation）
- `C` / `HYBRID`：
  - detection 使用 global-preproc frame 來提升 bbox
  - segmentation crop 來源仍以 raw frame 為主，再套 ROI preproc

範例：

```yaml
preproc_scheme: C
```

### pipeline step
每個 step 都是 dict：
- `type: preproc | model`
- `name: str`（必須存在於 registry）
- `params: dict`（選用；依模組而定）

示例：

```yaml
pipeline:
  - type: preproc
    name: CLAHE
    params:
      clipLimit: 1.5
      tileGridSize: [8, 8]
  - type: model
    name: YOLOv11
    params:
      weights: models/detection/yolo11l.pt
      conf: 0.25
      iou: 0.5
      imgsz: 640
      train_enabled: true
      epochs: 30
```

---

## 5) evaluation

Runner 會讀：
- `evaluation.evaluator: str`（預設 `BasicEvaluator`）
- `evaluation.restrict_to_gt_frames: bool`（預設 `true`）
- `evaluation.visualize.enabled: bool`（預設 `true`）
- `evaluation.visualize.samples: int`
- `evaluation.visualize.include_detection: bool`
- `evaluation.visualize.include_segmentation: bool`

當 `restrict_to_gt_frames: true`：
- 沒有 GT frame 的影片會跳過 metrics（避免把未標註 frame 當成 FN/FP）。

---

## 6) segmentation

Segmentation 的 schema 由 `tracking/segmentation/workflow.py` 的 `SegmentationConfig.from_dict()` 解析。

常用欄位：
- `segmentation.model.name: str`
- `segmentation.model.params: dict`
- `segmentation.padding_min / padding_max / padding_inference: float`
- `segmentation.target_size: [H, W]`（也接受 `resize` / `crop_size`）
- `segmentation.epochs: int`
- `segmentation.batch_size: int`
- `segmentation.lr / weight_decay: float`
- `segmentation.threshold: float`
- `segmentation.val_ratio: float`
- `segmentation.seed: int`
- `segmentation.device: auto|cpu|cuda...`
- `segmentation.train: bool`（也接受舊 key：`enabled`）
- `segmentation.inference_checkpoint: str|null`（也接受 `checkpoint`/`weights`）
- `segmentation.pretrained_external: str|null`
- `segmentation.auto_pretrained: bool`
- `segmentation.jitter: float`（也接受 `jitter_translate`）

> 注意：`model_name` 會被轉成小寫比對 registry（例如 `"deeplabv3+"`）。

---

## 7) 合法名稱（registries）

### 為什麼要看 registry？
`pipeline[*].name`、`segmentation.model.name`、`evaluation.evaluator` 都是用 registry 查表。
寫錯名稱會在 runtime 報錯（或找不到對應 class）。

### 目前已知的 registry keys（會隨程式更新而變）
你可以用下面方式在本機輸出當前 keys（Windows 友善、不用 heredoc）：

```bash
python -c "from tracking.core.registry import MODEL_REGISTRY, PREPROC_REGISTRY, SEGMENTATION_MODEL_REGISTRY, EVAL_REGISTRY; print('MODEL_REGISTRY', sorted(MODEL_REGISTRY.keys())); print('PREPROC_REGISTRY', sorted(PREPROC_REGISTRY.keys())); print('SEGMENTATION_MODEL_REGISTRY', sorted(SEGMENTATION_MODEL_REGISTRY.keys())); print('EVAL_REGISTRY', sorted(EVAL_REGISTRY.keys()))"
```

---

## 8) 各 step 的 params 速查（常用）

> 這一段是「排程檔作者」最常需要的：每個 `pipeline[*].params` 到底能放什麼。

### Preproc：`type: preproc`

#### `CLAHE`
- `clipLimit`（或別名 `clip`）
- `tileGridSize`（或別名 `grid`；例如 `[8, 8]`）

#### `TGC`
- `mode`: `linear` | `exp` | `custom`
- `gain_start`, `gain_end`
- `exp_k`
- `custom_points`: `[[y_norm, gain], ...]`
- `per_channel`: bool
- `clip`: bool

#### `SRAD`
- `iterations`: int
- `lambda`: float（時間步長，建議 $(0, 0.25]$）
- `eps`: float
- `convert_gray`: bool

#### `LOG_DR`
- `method`: `log` | `gamma`
- `gamma`: float（method=gamma 時用）
- `clip_percentile`: float
- `eps`: float
- `per_channel`: bool

#### `AUGMENT`
- `hflip_prob`, `vflip_prob`
- `translate_frac`
- `rotate_max_deg`
- `brightness`, `contrast`
- `noise_std`
- `seed`

### Model：`type: model`

#### `YOLOv11`（Ultralytics）
以下 key 會被 `tracking/models/yolov11.py` 讀取：
- Inference: `weights`, `conf`, `iou`, `imgsz`, `device`, `classes`, `max_det`, `fallback_last_prediction`, `include_empty_frames`
- Training: `train_enabled`, `epochs`, `batch`, `lr0`, `patience`, `workers`

> 備註：你在 schedules 裡常看到的 `force_cpu_if_no_cuda` 並不是 `YOLOv11Model.DEFAULT_CONFIG` 的欄位，但目前 runner 在建立 model 時會把 params 原封不動傳進去；
> 「是否會生效」取決於 model 類別本身是否有讀該 key。若你要確保生效，請以各 model 的 `DEFAULT_CONFIG` / `__init__` 為準。

#### `FasterRCNN`（torchvision）
以下 key 會被 `tracking/models/faster_rcnn.py` 讀取：
- Inference: `score_thresh`, `device`, `pretrained`, `num_classes`, `fallback_last_prediction`, `include_empty_frames`, `inference_batch`
- Training: `epochs`, `batch_size`, `lr`, `weight_decay`, `momentum`, `step_size`, `gamma`, `num_workers`, `pin_memory`
- Optimizer/穩定性: `optimizer`, `adamw_betas`, `adamw_eps`, `grad_clip`, `detect_anomaly`

---

## 9) 常見錯誤 / 規則

- **不要再寫 `scope`**：runner 已移除 per-step scope 支援；一律靠 `preproc_scheme` 決定 routing。
- `preproc_scheme` 寫錯（例如 `D`）會直接丟 `ValueError`。
- Windows 路徑建議：
  - YAML 可直接用 `C:\\path\\to\\data`（雙反斜線）
  - 或用 `/`：`C:/path/to/data`
- `queue:` 檔案主要給 UI 用；CLI 若要跑單次，建議另存單一 config 檔。

---

## 10) 更新這份文件的建議流程

當你新增/修改模組（例如新增 preproc / model / segmentation model）：
1) 先跑一次上面的 registry dump 指令確認 key。
2) 更新本文件的「常用欄位」與「常見錯誤」章節（保持與 runner 行為一致）。
3) 若你改了 runner 的解析規則（例如新增新的 split method），務必同步更新「dataset/split」段落。
