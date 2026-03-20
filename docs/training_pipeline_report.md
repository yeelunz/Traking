# 訓練流水線資料流報告

> 本文件對照 `runner.py`、`dataset_manager.py`、`workflow.py`、`engine.py` 的**實際程式碼**重新撰寫，供 Code Review 使用。
> 標註 ⚠️ 的段落表示**實際行為與可能的設計預期存在差異**。

---

## 目錄

1. [總覽：流水線階段順序](#1-總覽流水線階段順序)
2. [資料載入與分割](#2-資料載入與分割)
3. [前處理鏈 (Preprocessing Chain)](#3-前處理鏈-preprocessing-chain)
4. [Detection 訓練](#4-detection-訓練)
5. [Segmentation 訓練](#5-segmentation-訓練)
6. [Classification 訓練資料準備](#6-classification-訓練資料準備)
7. [三層持久化快取機制](#7-三層持久化快取機制)
8. [潛在不一致 / Review 重點](#8-潛在不一致--review-重點)

---

## 1. 總覽：流水線階段順序

```
run_pipeline.py  →  PipelineRunner(config).run()

for fold in (LOSO folds | single split):
  for exp in experiments:
    ┌─ Stage 0: validate config (enforce_or_collect_warnings)
    ├─ Stage 1: build preproc chain (Scheme A/B/C)
    ├─ Stage 2: build model instances
    │
    ├─ Stage 3: [可選] K-Fold Detection CV  (k_fold > 1)
    │     訓練 + 驗證在 train set 的子拆分上
    │
    ├─ Stage 4: detector_train_full
    │     在完整 train set 上訓練 → 持久化快取
    │
    ├─ Stage 5: detector_eval
    │     在 test set 上推論 → test_predictions
    │
    ├─ Stage 6: trajectory_filter (預設啟用)
    │     Hampel + S-G 平滑 → 取代 test_predictions
    │
    ├─ Stage 6b: filtered_detection_accuracy
    │     重新計算濾波後 IoU / CE / SR
    │
    ├─ Stage 7: segmentation_train (有快取)
    ├─ Stage 8: segmentation_infer (test set 推論)
    │
    └─ Stage 9: classification (有快取)
          含 GT 軌跡平滑、Youden 閾值校準
```

---

## 2. 資料載入與分割

### 2.1 Dataset 掃描 (`COCOJsonDatasetManager._scan`)

```
dataset_root/
├── <subject_dir>/
│   ├── video_A.avi
│   ├── video_A.json          ← COCO-VID JSON annotation
│   └── video_A_masks/frame_*.png  ← (segmentation masks, optional)
```

- 遞迴掃描 `.mp4/.avi/.mov/.mkv/.wmv`
- 每個影片旁尋找同名 `.json` 作為標註
- `video_subjects` 字典：根據資料夾路徑推導 subject ID
  - 優先策略：取 `relpath(video, root)` 的第一層目錄名稱
  - 回退：正規化取檔名前導數字

### 2.2 分割策略 (`runner.py L286-371`)

| 策略 | 觸發方式 | 行為 |
|------|---------|------|
| `video_level` | 預設 | shuffle → ratio 切分 |
| `subject_level` | `method: subject_level` | 按 subject 分組 → ratio 切分 (subject 不跨 bucket) |
| `loso` | `method: loso` 或 `split.loso: true` | 每個 subject 輪流做 test set |

LOSO 支援過濾旋鈕（`runner.py L321-358`）：
- `subjects` / `max_folds`：限制 fold 數量
- `max_train_videos` / `max_test_videos`：截斷每折影片列表

### 2.3 K-Fold 內部交叉驗證 (`runner.py L1136-1300`)

- 在 **LOSO 每折的訓練集內部**再做一次 K-Fold
- `train_ds` 的影片 `shuffle` → `round-robin` 分配到 K 個 bin
- 每折：一個 bin 為 validation，其餘為 training
- 對每折：重新 `build_models()` → 訓練 → 驗證
- 最後聚合 K 折的 `iou_mean`, `ce_mean`, `success_rate`, `success_auc`, `fps`, `drift_rate`

> ⚠️ **K-Fold 的 round-robin 分配**：使用 `Random(seed).shuffle → 手動 slice`，與 `dataset_manager.py.k_fold()` 裡的 round-robin 不同。兩套 K-Fold 邏輯共存（runner 自行實作 vs dm.k_fold()），runner 的 K-Fold **不使用** `COCOJsonDatasetManager.k_fold()`。

---

## 3. 前處理鏈 (Preprocessing Chain)

### 3.1 現有前處理模組

| 名稱 | 檔案 | 說明 |
|------|------|------|
| `CLAHE` | `preproc/clahe.py` | 自適應直方圖均衡 |
| `SRAD` | `preproc/srad.py` | 散斑降噪 (Speckle Reducing Anisotropic Diffusion) |
| `TGC` | `preproc/tgc.py` | 時間增益補償 |
| `LogDR` | `preproc/logdr.py` | 對數動態範圍壓縮 |
| `Augment` | `preproc/augment.py` | 訓練限定增強 (train_only=True) |

### 3.2 Scheme 路由（`runner.py L669-698`）

```
YAML config:  preproc_scheme: A | B | C  (或 GLOBAL / ROI / HYBRID)

Scheme A (Global):
  preprocs = [所有 preproc 步驟]    ← detector & segmentation 全局使用
  preprocs_roi = []

Scheme B (ROI):
  preprocs = []                     ← detector 看原始幀
  preprocs_roi = [所有 preproc]     ← segmentation ROI 裁切後才使用

Scheme C (Hybrid):
  preprocs = [所有 preproc]          ← detector 使用全域前處理
  preprocs_roi = [所有 preproc]      ← 同時 segmentation ROI 也使用
```

### 3.3 Scheme 在各階段的作用域

| 階段 | Scheme A | Scheme B | Scheme C |
|------|---------|---------|---------|
| Detector 訓練/推論 | 使用 `preprocs` | 不使用 (原始幀) | 使用 `preprocs` |
| Segmentation 全域 | 使用 `preprocs` | **不使用** | **不使用** |
| Segmentation ROI | 不使用 | 使用 `preprocs_roi` | 使用 `preprocs_roi` |

> ⚠️ **Scheme C 的 Segmentation 全域前處理被清空**：`runner.py L1726-1729` 在 Scheme B 和 C 下都把 `seg_global_preprocs = []`。這意味著 Scheme C 下 segmentation 從**原始幀**裁切 ROI，再在 ROI 上做前處理 — 與直覺上「Hybrid = detector 用全域 + seg 也用全域」可能有差異。這是設計意圖（確保 seg 看到無失真的原始裁切），但值得在 Review 中確認。

---

## 4. Detection 訓練

### 4.1 Model 實例化 (`build_models`, L706-748)

- 從 `MODEL_REGISTRY` 查表建構
- 將 `preprocs` 掛到 `model.preprocs`
- 建立失敗時以 `_UnavailableModel` placeholder 替代（predict 回空結果）
- 記錄模型實際參數快照到 log

### 4.2 快取重用檢查 (`runner.py L1317-1406`)

```
_detector_signature = SHA-256(
    detector_models + params,
    detector_preprocs (Scheme A/C 才有),
    preproc_scheme,
    dataset_root,
    sorted(train_video_paths),
    subject,
    fold_index
)
```

流程：
1. 查 `detector_reuse_cache[sig_key][model_name]`
2. 命中 → `model.load_checkpoint(cached_ckpt)` → 跳過訓練
3. 未命中 → 正常 `model.train(train_ds, None, seed, output_dir)`
4. 訓練完 → 取 `best_ckpt` → 存入 in-memory + persistent cache (`detector_cache.json`)

### 4.3 訓練呼叫介面

```python
model.train(
    dataset,       # SimpleDataset
    val_ds,        # None (full train) 或 SimpleDataset (k-fold)
    seed=seed,
    output_dir=output_dir,
)
→ 回傳 dict: {"status", "best_ckpt" | "checkpoint", ...}
```

> ⚠️ **train_enabled / should_train 雙重控制**：`runner.py L1358-1371` 先檢查 `model.train_enabled`（靜態屬性），再呼叫 `model.should_train(train_ds, val_ds)`（動態回呼）。兩者都可以取消訓練，但行為是 AND 關係（任一為 False 就跳過），而非 OR。文件過程應確認各 model 的 `should_train` 實作是否符合預期。

---

## 5. Segmentation 訓練

### 5.1 架構 (`SegmentationWorkflow`)

- 配置由 `SegmentationConfig.from_dict()` 解析
- Model 從 `SEGMENTATION_MODEL_REGISTRY` 查表
- 支援模型：UNet, UNet++, DeepLabV3+, MedNeXt, MedSAM, nnU-Net

### 5.2 訓練資料準備 (`SegmentationCropDataset`)

```
1. 遍歷 train_video_paths → 載入 COCO-VID JSON
2. 取得每幀的 GT bbox + mask 路徑
3. __getitem__:
   ├─ 讀取影像
   ├─ _apply_preprocs_frame(global preprocs)
   ├─ 隨機 padding 擴展 bbox (padding_min ~ padding_max)
   │   └─ jitter: 隨機偏移 ΔX, ΔY
   │   └─ redundancy: 每幀產生 N 個不同 jitter 的樣本
   ├─ crop_with_bbox → ROI 影像 + mask
   ├─ roi preprocs (apply_to_frame_and_mask)
   ├─ resize → target_size (default 256×256)
   │   ├─ 影像: INTER_LINEAR
   │   └─ mask: INTER_NEAREST
   └─ 歸一化 → Tensor
```

### 5.3 Validation Split（`runner.py L1780-1789`）

```python
if val_ratio > 0 and len(train_targets) > 1:
    rng = Random(seg_seed)
    shuffle → val_videos = first val_count items
    train_targets = remaining (若為空則用全部)
```

> ⚠️ **val_videos 為空時回退行為**：`runner.py L1789` `train_targets = shuffled[val_count:] or shuffled` — 如果 `val_count >= len(shuffled)`，`shuffled[val_count:]` 為空列表，`or` 觸發使得 `train_targets = shuffled`（全部影片），此時 val 和 train 完全重疊。

### 5.4 Loss Function

```
loss = bce_weight × BCE_with_logits(logits, masks)
     + dice_weight × DiceLoss(logits, masks)
```

預設：`bce_weight=1.0`, `dice_weight=1.0`

### 5.5 快取機制

```
_segmentation_signature = SHA-256(
    seg_config, model_name,
    preproc_scheme,
    global_preprocs (class names),
    roi_preprocs (class names),
    dataset_root, sorted(train_videos),
    seed, subject, fold
)
```

- 存在 `results/segmentation_cache.json`
- 命中 → `seg_workflow.load_checkpoint(cached_ckpt)` → 跳過訓練
- 訓練完 → 存入 persistent cache + 更新 in-memory cache

---

## 6. Classification 訓練資料準備

### 6.1 完整流程 (`engine.py`)

```
Train Dataset (video_paths)
  │
  ├─ 載入 COCO JSON 標註
  │   └─ _load_annotations_for_videos()
  │
  ├─ 建立 FramePrediction（從 GT annotation）
  │   └─ _annotation_to_predictions()
  │       → frames[fi][0] → (x,y,w,h) → FramePrediction
  │
  ├─ 掛載 GT segmentation masks
  │   └─ attach_ground_truth_segmentation()
  │
  ├─ GT 軌跡平滑（skip_hampel=True, S-G only）
  │   ├─ bbox_strategy: "independent" (預設)
  │   ├─ sg_window: 7 (比推論路徑的 11 更小)
  │   └─ sg_polyorder: 1 (比推論路徑的 2 更小)
  │
  ├─ [可選] Texture Pretrain 自動 bootstrap
  │   └─ _ensure_texture_pretrain_ckpt()
  │
  ├─ 特徵提取
  │   ├─ video_level: extract_video(samples, video_path)
  │   ├─ finalize_batch(fit=True) → PCA/Z-score
  │   └─ subject_level:
  │       └─ aggregate_subject(video_feats[])
  │
  ├─ 向量化
  │   └─ FeatureVectoriser.transform(features)
  │       → Dict[str, float] → np.ndarray (N, D)
  │
  └─ Youden's Index 閾值校準
      └─ Leave-One-Subject-Out on TRAIN SET
```

### 6.2 Youden 閾值校準 (`engine.py L496-593`)

1. 遍歷訓練集每個 subject → LOSO
2. 對 held-out 以外的 entities 訓練一個臨時 classifier
3. 對 held-out entities 預測機率
4. 收集所有 out-of-sample (y_true, prob)
5. `_find_youden_threshold()`: 掃描所有 unique probability 中點 → 最大化 `J = Sensitivity + Specificity - 1`

> ⚠️ **深度學習分類器的 LOO epoch 上限**：`engine.py L532-545` 對有 epochs 的分類器（PatchTST, TimeMachine），LOO fold 的 `epochs` 被 cap 在 `min(原始, 25)`、`patience` 被 cap 在 `min(原始, 8)`。這會導致 calibration 時分類器的訓練程度與最終分類器不一致，可能影響閾值品質。

### 6.3 分類器持久化快取 (`runner.py L2113-2161`)

```
_classification_signature = SHA-256(
    feature_extractor config,
    classifier config,
    label_file,
    seg_enabled, seg_model_name,
    dataset_root, sorted(train_videos),
    subject, fold
)

model_key = "{fe_name}_{clf_name}"
```

- 存在 `results/classification_cache.json`
- 命中 → `classifier.load(cached_ckpt)` → 跳過訓練
- 訓練完 → 存到 `classifier.pkl` + persistent cache

---

## 7. 三層持久化快取機制

| 快取 | 檔案 | Signature 考量 | 存儲內容 |
|------|------|---------------|---------|
| Detector | `detector_cache.json` | preproc scheme + model params + train videos + subject/fold | checkpoint 路徑 |
| Segmentation | `segmentation_cache.json` | seg config + preproc scheme + preproc classes + train videos + seed + subject/fold | checkpoint 路徑 |
| Classification | `classification_cache.json` | FE config + classifier config + label file + seg enabled/model + train videos + subject/fold | classifier.pkl 路徑 |

共通行為：
- 載入時檢查 checkpoint 檔案是否存在（刪除了的不算命中）
- in-memory cache 優先於 disk cache
- 寫入錯誤不會中斷 pipeline (best-effort)

> ⚠️ **Segmentation signature 只記錄 preproc 類別名稱**：`_segmentation_signature` 用 `type(p).__name__` 表示前處理模組（`runner.py L497-498`），不含參數。若同一 preproc 更換了參數（例如 CLAHE 的 clipLimit），signature 不變 → 錯誤命中快取。

---

## 8. 潛在不一致 / Review 重點

### 8.1 K-Fold 兩套實作

`runner.py L1136-1200` 自行實作 K-Fold（手動 shuffle + slice），但 `dataset_manager.py` 也有 `k_fold()` 方法（round-robin 分配）。runner 的版本**不呼叫** dm 的方法，兩者的分配策略不同：
- runner: `vids[fi * fold_size : (fi+1) * fold_size]` — 連續切片，最後一折可能更大
- dm: `bins[idx % k]` — round-robin 均勻分配

### 8.2 Scheme C 的語意

Scheme C 被命名為 "Hybrid"，但 segmentation 的全域前處理被清空（`seg_global_preprocs = []`），這代表 segmentation 從原始幀裁切 ROI。若原意是讓 Segmentation 也看到全域處理過的幀，則此行為不符預期。

### 8.3 Segmentation val 回退

當 `val_ratio` 設定導致 `val_count >= len(train_targets)` 時，`train_targets = shuffled[val_count:] or shuffled` 造成 train/val 完全重疊。

### 8.4 Segmentation signature 不含 preproc 參數

只記錄 class name → 不同參數不會產生不同 signature。

### 8.5 Classification GT 軌跡平滑的 bbox_strategy

`engine.py L718-719` 使用 `(config.get("trajectory_filter", {}) or {}).get("bbox_strategy", "independent")`，但 runner 的 test trajectory filter stage（`runner.py L1445`）使用 `"hampel_only"` 作為預設值。兩者的預設 `bbox_strategy` 不同 — GT 路徑用 `"independent"`，推論路徑用 `"hampel_only"`。

### 8.6 Detector train_enabled + should_train 的互動

兩者是 AND 關係，但命名不直觀 — `train_enabled=True` 不代表一定會訓練，`should_train()` 可再否決。

### 8.7 DiceLoss 的 smooth 常數

`workflow.py` 的 `_dice_loss` 用 `eps=1e-6`，但部分 segmentation_models_pytorch 的 DiceLoss 用 `smooth=1.0`。自訂實現(eps=1e-6)更易受小分母影響而產生梯度不穩定。
