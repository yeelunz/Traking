# 訓練流水線資料流詳細報告

> 本文件完整記錄從原始資料到訓練完成的每一步資料轉換與處理邏輯。

---

## 目錄

1. [總覽：訓練階段架構](#1-總覽訓練階段架構)
2. [資料載入與分割](#2-資料載入與分割)
3. [前處理鏈（Preprocessing Chain）](#3-前處理鏈preprocessing-chain)
4. [Detection 訓練](#4-detection-訓練)
5. [Segmentation 訓練](#5-segmentation-訓練)
6. [Classification 訓練](#6-classification-訓練)
7. [快取機制](#7-快取機制)
8. [Scheme A/B/C 對訓練的影響](#8-scheme-abc-對訓練的影響)

---

## 1. 總覽：訓練階段架構

整個訓練流程在 `PipelineRunner.run()` 中按以下順序執行：

```
LOSO Fold 迴圈（或單次 Split）
  └─ Experiment 迴圈
       ├── Stage 1: detector_kfold        — 可選的 K-Fold 交叉驗證（在訓練集內部）
       ├── Stage 2: detector_train_full   — 在完整訓練集上訓練 Detector
       ├── Stage 3: detector_eval         — 在測試集上推論（產生 test_predictions）
       ├── Stage 4: trajectory_filter     — 軌跡平滑（僅修改 test_predictions）
       ├── Stage 5: segmentation_train    — 訓練分割模型
       ├── Stage 6: segmentation_infer    — 在測試集上推論分割
       └── Stage 7: classification        — 訓練與評估分類器
```

**重點**：Stage 3/4 雖然是「推論」但在 LOSO 每折的主迴圈中執行，因為 Classification 需要測試集的 detection 結果作為輸入。

---

## 2. 資料載入與分割

### 2.1 Dataset 掃描

`COCOJsonDatasetManager(dataset_root)` 執行以下步驟：

1. **遞迴掃描** `dataset_root` 下所有影片檔（`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`）
2. 對每支影片尋找同目錄同名 `.json` 標註檔（COCO-VID 格式）
3. **Subject 推導**：取影片相對路徑的第一層目錄名，用正則 `^(\d+)` 提取數字前綴作為 Subject ID
   - 例如：`n001/D.avi` → Subject = `n001`

### 2.2 LOSO（Leave-One-Subject-Out）

1. `dm.loso()` 產生 generator，每次 yield 一個 fold：
   - `train`：除了該 subject 以外的所有影片
   - `test`：該 subject 的所有影片
2. 支援過濾參數：
   - `subjects` / `subject_ids`：只跑指定 subject
   - `max_folds`：限制最大折數
   - `max_train_videos` / `max_test_videos`：截斷影片數（用於 smoke test）
3. 建立 `SimpleDataset(train_videos)` 和 `SimpleDataset(test_videos)`
   - `SimpleDataset` 只包含**有標註**的影片
   - `__getitem__` 回傳 `{"video_path": str, "annotation": dict}`

### 2.3 K-Fold（Detection 內部交叉驗證）

- 在 LOSO 每折的**訓練集內部**再做一次 K-Fold
- 將 `train_ds` 的影片 shuffle → round-robin 分配到 K 個 bin
- 每折：一個 bin 為 validation，其餘為 training
- 用途：評估 detector 在不同訓練子集上的穩定性

---

## 3. 前處理鏈（Preprocessing Chain）

### 3.1 中央路由

Pipeline 中的每個 `step`（type = `"preproc"`）會根據 **Scheme** 路由到不同的列表：

| Scheme | `preprocs`（全域） | `preprocs_roi`（ROI 局部） |
|--------|-------------------|--------------------------|
| **A** (Global) | ✅ 加入 | ❌ |
| **B** (ROI) | ❌ | ✅ 加入 |
| **C** (Hybrid) | ✅ 加入 | ✅ 也加入 |

### 3.2 可用前處理模組

| 模組 | Registry 名 | 功能 | 參數 |
|------|------------|------|------|
| CLAHE | `CLAHE` | 自適應直方圖均衡化（LAB L 通道） | `clipLimit=2.0`, `tileGridSize=[8,8]` |
| Augment | `AUGMENT` | 幾何+光度增強 | hflip, vflip, rotate, translate, brightness, contrast, noise |
| SRAD | `SRAD` | 散斑抑制各向異性擴散 | — |
| LogDR | `LOGDR` | 對數動態範圍壓縮 | — |
| TGC | `TGC` | 時間增益補償 | — |

### 3.3 統一前處理管線

**所有模型**（Detection / Segmentation）在訓練與推論時都走同一個前處理管線：

```
輸入影像 (BGR, uint8)
  → cvtColor(BGR → GRAY)
  → cvtColor(GRAY → BGR)        ← 強制灰階一致性
  → cvtColor(BGR → RGB)
  → 依序執行 preproc chain 的 apply_to_frame()
  → cvtColor(RGB → BGR)
輸出影像 (BGR, uint8)
```

**訓練限定增強**：preproc 可標記 `train_only=True`（如 Augment），推論時自動跳過。

### 3.4 帶 BBox 的前處理（Detection 訓練專用）

YOLOv11 / FasterRCNN 訓練時，若 preproc 支援 `apply_to_frame_and_bboxes(frame, bboxes)`，會同時更新 GT bbox 座標（例如翻轉增強時需要鏡像 bbox）。

---

## 4. Detection 訓練

### 4.1 YOLOv11 訓練資料準備

1. **遍歷 Dataset** 中每支影片的 COCO-like JSON 標註
2. 對每個有 GT 的幀：
   - 從影片讀取該幀影像
   - 執行前處理管線（含 bbox 同步更新）
   - 儲存為 `.jpg`（影像）+ `.txt`（YOLO 格式標籤）
3. **COCO → YOLO 格式轉換**：
   ```
   (x, y, w, h) → (xc/W, yc/H, nw/W, nh/H)  # 歸一化中心座標
   ```
   - 自動 clamp 到 [0, 1]
4. 自動建立 `data.yaml`（Ultralytics 格式）
5. `include_empty_frames=False`（預設）：跳過無 GT 標註的幀

### 4.2 YOLOv11 訓練迴圈

| 項目 | 設定 |
|------|------|
| 引擎 | Ultralytics YOLO API (`model.train()`) |
| Epochs | 5（預設） |
| Batch Size | 8 |
| 學習率 | lr0=0.01 |
| 圖片大小 | imgsz=640 |
| Patience | 50 |
| Loss | CIoU + BCE Classification + DFL (Distribution Focal Loss) |
| Workers | 0（Windows 安全） |
| 輸出 | `{output_dir}/YOLOv11/weights/best.pt` |

### 4.3 FasterRCNN 訓練資料準備

1. `_build_train_index()`：展開 Dataset 為 frame-level 索引
   - 每筆：`{video_path, frame_index, bboxes}`
2. **BBox 格式轉換**：COCO `(x,y,w,h)` → PyTorch `(x1,y1,x2,y2)`
   - 自動偵測歸一化座標（全 ≤ 1.0）→ 乘回像素座標
   - NaN / 極端越界 bbox 被過濾
3. `FrameDataset` + `DataLoader`：
   - 自訂 `collate_fn`
   - `_VideoFrameCache`：LRU 式 VideoCapture cache（max_open=4）

### 4.4 FasterRCNN 訓練迴圈

| 項目 | 設定 |
|------|------|
| 架構 | `fasterrcnn_resnet50_fpn` + ImageNet 預訓練 |
| Optimizer | AdamW (lr=0.0025, weight_decay=0.0005) |
| Scheduler | StepLR(step_size=3, gamma=0.1) |
| Loss | loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg |
| NaN 防護 | 逐 batch 檢查 `torch.isfinite(losses)`，非有限則跳過 |
| 梯度裁剪 | `clip_grad_norm_(max_norm=5.0)` |
| Epochs | 1（預設） |
| Checkpoint | `{model_name}.pth`（state_dict）+ `_history.json` |

### 4.5 TemplateMatching / OpticalFlowLK / FASTSpeckle

**無訓練**。這些模型不需要訓練階段，`train()` 直接回傳空結果。

### 4.6 Detector Signature & 快取

訓練前計算 **Detector Signature**（SHA-256 雜湊）：
- 輸入：preproc chain 參數 + model 參數 + 訓練影片列表 + scheme
- 快取命中：載入已存 checkpoint → 跳過訓練
- 快取未命中：正常訓練 → 存入持久化快取 (`detector_cache.json`)

---

## 5. Segmentation 訓練

### 5.1 訓練資料準備（SegmentationCropDataset）

**初始化流程：**

1. 遍歷每個 `video_path`，載入 COCO-VID JSON 標註
2. 取得每幀的 GT bbox 和 mask 路徑
3. **過濾空 mask**：`_mask_has_content()` → 讀取 mask → 若 size=0 或 nonzero=0 則跳過
4. **Redundancy 重複取樣**：每個有效 sample 重複 `redundancy` 次，每次隨機取不同的 padding
   - `padding_train_min=0.10`, `padding_train_max=0.15`
5. 建立 `SegmentationSampleDescriptor`：`(video_path, frame_index, roi_bbox, mask_path, original_bbox)`

**`__getitem__` 資料轉換管線（12 步）：**

```
步驟 1:  _load_frame(video_path, frame_index)
         → cv2.VideoCapture 讀取 BGR 影像

步驟 2:  _load_mask(video_path, mask_path)
         → cv2.imread(GRAYSCALE) → 二值化
         → fill_holes() → keep_largest_component()

步驟 3:  全域前處理 (self.preprocs)
         → BGR → Gray → BGR → 逐一執行 preproc chain
         → 支援 apply_to_frame_mask_bbox() 同時更新 mask & bbox

步驟 4:  Jitter 增強（若 jitter > 0）
         → 隨機平移 bbox ±(jitter × bbox寬高)
         → clamp 到影像邊界

步驟 5:  crop_with_bbox(frame, roi_bbox)
         → 從全幀裁切 ROI 影像

步驟 6:  crop_with_bbox(mask, roi_bbox)
         → 從全幀裁切 ROI mask

步驟 7:  ROI 前處理 (self.roi_preprocs)
         → 支援 apply_to_frame_and_mask() 同時處理

步驟 8:  cv2.resize(target_size)
         → 影像: INTER_LINEAR
         → mask: INTER_NEAREST    ← 避免插值汙染二值 mask

步驟 9:  灰階檢查
         → 若 2D → cvtColor(GRAY→BGR)

步驟 10: 歸一化
         → float32 / 255.0

步驟 11: mask 二值化
         → (mask > 0).astype(float32)

步驟 12: 維度轉換
         → 影像: HWC → CHW tensor
         → mask: 添加 channel dim (H,W) → (1,H,W)
```

### 5.2 Scheme 對 Segmentation 訓練前處理的影響

| Scheme | `self.preprocs`（全域，步驟 3） | `self.roi_preprocs`（ROI，步驟 7） |
|--------|-------------------------------|-----------------------------------|
| **A** | 有（如 CLAHE） | 無 |
| **B** | **無**（強制清空） | 有（如 CLAHE） |
| **C** | **無**（強制清空） | 有（如 CLAHE） |

**核心邏輯**：Scheme B/C 時，`seg_global_preprocs = []`，確保 Segmentation 從 RAW 原圖裁切 ROI。

### 5.3 訓練迴圈

```python
def train(train_videos, val_videos=None, seed=None):
```

1. **前置檢查**：
   - auto_mask 模式 → 跳過（僅推論用）
   - MedSAM inference-only → 跳過
   - `train=False` → 跳過
2. **外部預訓練**：若 `pretrained_external` 路徑存在 → `load_checkpoint()` warm-start
3. **建立 Dataset**：`SegmentationCropDataset(train_videos, preprocs, roi_preprocs, ...)`
4. **建立 DataLoader**：`shuffle=True, pin_memory=True(if CUDA)`
5. **驗證集**（若有 `val_videos`）：用 `padding_inference`（固定 0.15）、`redundancy=1`、`jitter=0`

### 5.4 Loss Function

```
loss = bce_weight × BCE_with_logits(logits, masks)
     + dice_weight × DiceLoss(logits, masks)
```

| Loss | 公式 |
|------|------|
| BCE | `F.binary_cross_entropy_with_logits(logits, masks)` |
| Dice | $1 - \frac{2 \cdot \sum(\sigma(\text{logits}) \cdot \text{target}) + \epsilon}{\sum \sigma(\text{logits}) + \sum \text{target} + \epsilon}$ |

- 預設權重：`bce_weight=1.0`, `dice_weight=1.0`
- Dice Loss 在 (B,C,H,W) 的 dim=(1,2,3) 上計算

### 5.5 Optimizer & Scheduler

| 項目 | 設定 |
|------|------|
| Optimizer | AdamW(lr=1e-3, weight_decay=1e-5) |
| MedSAM 特殊 | 只取 `model.get_trainable_parameters()`（mask decoder 參數） |
| Scheduler | 無 |

### 5.6 Checkpoint 策略

| 情況 | 策略 |
|------|------|
| 有 validation | `segmentation_best.pt`（val loss 最低時存檔） |
| 無 validation | `segmentation_last.pt`（每 epoch 覆寫） |
| MedSAM | 只存 mask_decoder state_dict |

### 5.7 Segmentation 模型架構

| 模型 | Registry Key | 架構 | 特色 |
|------|-------------|------|------|
| UNet | `unet` | segmentation_models_pytorch | encoder=ResNet34, ImageNet pretrained |
| UNet++ | `unetpp` | segmentation_models_pytorch | 同上，密集跳接 |
| DeepLabV3+ | `deeplabv3+` | segmentation_models_pytorch | ASPP 空洞卷積 |
| MedNeXt | `mednext` | MedNeXt V1 2D | S/B/M/L 尺寸可配 |
| MedSAM | `medsam` | SAM + box/point prompt | 只微調 mask_decoder |
| nnU-Net | `nnunet` | PlainConvUNet (nnUNetv2) | 5 stage, [32,64,128,256,512] |
| FCN-ResNet50 | `torchvision_fcn_resnet50` | TorchVision FCN | 預設 fallback（推論 only） |

### 5.8 Segmentation Signature & 快取

- 計算 `_segmentation_signature()`：SHA-256(seg config + model name + preprocs + train paths)
- 快取命中 → `seg_workflow.load_checkpoint(cached_ckpt)` → 跳過訓練
- 快取未命中 → 正常訓練 → 存入 `segmentation_cache.json`

---

## 6. Classification 訓練

### 6.1 訓練資料準備全流程

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
  ├─ 附加 GT 分割遮罩
  │   └─ attach_ground_truth_segmentation()
  │       → 讀取 mask → compute_mask_stats()
  │       → 寫入 FramePrediction.segmentation
  │
  ├─ GT 軌跡平滑
  │   └─ filter_detections(skip_hampel=True)
  │       → 僅做雙向 S-G（sg_window=7, polyorder=1）
  │       → 不做 Hampel 離群值移除
  │       → 重建 FramePrediction 列表
  │
  ├─ 特徵提取
  │   └─ feature_extractor.extract_video(samples, video_path)
  │       → 各 extractor 獨立邏輯（詳見特徵提取報告）
  │
   ├─ 批次後處理（PCA/降維 + 全域標準化）
  │   └─ feature_extractor.finalize_batch(feat_dicts, fit=True)
   │       → 訓練時 fit 降維器參數（如 PCA）
   │       → 並 fit 全域標準化參數（mean/std）
  │
  ├─ Subject 聚合（若 level="subject"）
  │   └─ feature_extractor.aggregate_subject(video_features)
  │       → Non-TSC: mean/std/min/max 統計
  │       → TSC: 逐元素取 mean
  │
  ├─ 標籤讀取
  │   └─ _load_subject_labels()
  │       → 從 ann.txt 讀取 subject → 0/1 映射
  │
  └─ 向量化
      └─ FeatureVectoriser.transform(features)
          → Dict[str, float] → np.ndarray (N, D)
          → 缺失 key 填 0.0
```

### 6.2 Youden's Index 閾值校準

**在分類器訓練之後、正式預測之前**執行：

1. **方法**：Leave-One-Subject-Out (LOSO) 校準
2. **流程**：
   - 對訓練集每個 subject 進行 LOSO：
     - 剩餘 subjects → 訓練臨時分類器
     - 預測 held-out subject 的正類機率
   - 收集所有 out-of-sample 機率
3. **閾值選擇**：
   - 掃描所有候選閾值（unique 機率值的中點）
   - 選出使 $J(t) = \text{Sensitivity} + \text{Specificity} - 1$ 最大化的閾值
4. **深度學習分類器特殊處理**：LOO 中 epoch 上限 cap 到 25、patience cap 到 8

### 6.3 分類器一覽

| Registry 名稱 | 核心模型 | 超參數（預設） | 類型 |
|--------------|---------|------------|------|
| `random_forest` | sklearn RandomForest | n_estimators=200, class_weight=balanced | 表格式 |
| `decision_tree` | sklearn DecisionTree | max_depth=5, class_weight=balanced | 表格式 |
| `svm` | sklearn SVC | C=1.0, rbf kernel, probability=True | 表格式 |
| `lightgbm` | LGBMClassifier | n_estimators=200, lr=0.1 | 表格式 |
| `xgboost` | XGBClassifier | n_estimators=200, max_depth=4, lr=0.05 | 表格式 |
| `tabpfn_v2` | TabPFN (in-context learning) | — (失敗 fallback → XGBoost → RF) | 表格式 |
| `multirocket` | MultiRocket | 84 kernels + 4 pooling types | TSC |
| `patchtst` | PatchTST Transformer | — | TSC |
| `timemachine` | TimeMachine | — | TSC |

> 補充：`multirocket` 的 C-axis 降維可用 `channel_reduction_method: pca|lda|umap`，
> T-axis 則可選 `dim_reduction: autoencoder`。

### 6.4 分類器訓練流程

```python
classifier.fit(X_train, y_train) → dict
```

- **無內建超參搜索**：每次執行只訓練 config 指定的一個分類器
- 超參搜索由外部 YAML schedule 負責多組參數
- 訓練後 `classifier.save(model_path)` 持久化為 pickle

### 6.5 Classification Signature & 快取

- 計算分類簽名 → 查詢 `classification_cache.json`
- 快取命中 → 直接載入 `classifier.pkl` → 跳過訓練
- 快取未命中 → 正常訓練 → 存入快取

---

## 7. 快取機制

### 7.1 三層持久化快取

| 快取 | 檔案 | Signature 組成 | 存儲內容 |
|------|------|---------------|---------|
| Detector | `detector_cache.json` | preproc + model params + train videos + scheme | checkpoint 路徑 |
| Segmentation | `segmentation_cache.json` | seg config + model name + preprocs + train paths | checkpoint 路徑 |
| Classification | `classification_cache.json` | clf config + features + classifier + train entities | classifier 路徑 |

### 7.2 快取驗證

載入快取時，會驗證 checkpoint 檔案是否仍然存在，不存在的條目自動清除。

### 7.3 同一 Schedule 內的 Detector 重用

同一 schedule 中，若多個 experiment 有相同的 detector signature（例如只改分類器/分割模型），第一次訓練後的 checkpoint 可被後續 experiment 重用：
- 記憶體快取 `_detector_reuse_cache[sig][model_name] = checkpoint_path`
- 磁碟持久化快取也會更新

---

## 8. Scheme A/B/C 對訓練的影響

### 8.1 完整影響矩陣

| 環節 | Scheme A | Scheme B | Scheme C |
|------|----------|----------|----------|
| **Detector 訓練影像** | 經前處理（如 CLAHE） | 原始影像 | 經前處理 |
| **Detector Signature 含前處理** | ✅ 是 | ❌ 否 | ✅ 是 |
| **Segmentation 全域前處理** | ✅ 有 | ❌ 無 | ❌ 無 |
| **Segmentation ROI 前處理** | ❌ 無 | ✅ 有 | ✅ 有 |
| **Seg 裁切源影像** | 前處理後全幀 | RAW 原圖 | RAW 原圖 |
| **Classification 特徵來源** | 前處理後的 detection + seg | RAW detection + ROI-前處理 seg | 前處理 detection + ROI-前處理 seg |

### 8.2 設計理念

- **Scheme A（Global）**：統一增強，detector 和 segmentation 看到相同影像
- **Scheme B（ROI）**：detector 看原始影像，只在 ROI 裁切後才增強 → 測試「局部增強」效果
- **Scheme C（Hybrid）**：detector 用增強影像提升 bbox 品質，但 segmentation 從 RAW 裁切再做 ROI 增強

---

## 附錄：訓練階段完整時序圖

```
┌────────────────────────────────────────────────────────────────────┐
│ Pipeline 初始化                                                     │
│   ・set_seed(seed)                                                  │
│   ・COCOJsonDatasetManager 掃描影片+標註                             │
│   ・載入持久化快取 (detector/seg/clf)                                 │
│   ・建立 LOSO folds 列表                                            │
└────────────┬───────────────────────────────────────────────────────┘
             │
    ┌────────▼────────┐
    │ For each LOSO   │
    │ fold            ├──────────────────────────────────────────┐
    │ (train/test)    │                                          │
    └────────┬────────┘                                          │
             │                                                   │
    ┌────────▼────────┐                                          │
    │ For each        │                                          │
    │ experiment      ├───────────────────────────────┐          │
    │                 │                               │          │
    └────────┬────────┘                               │          │
             │                                        │          │
    ┌────────▼────────────────────────────────┐       │          │
    │ 1. build_models()                        │       │          │
    │    → 建立模型 + 附加 preprocs              │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 2. detector_train_full                   │       │          │
    │    → sig 比對 → 快取 or 訓練              │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 3. detector_eval (on test_ds)            │       │          │
    │    → test_predictions 產出                │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 4. trajectory_filter                     │       │          │
    │    → test_predictions 就地替換為平滑版    │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 5. segmentation_train                    │       │          │
    │    → sig 比對 → 快取 or 訓練              │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 6. segmentation_infer (on test_ds)       │       │          │
    │    → seg masks + metrics 產出             │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 7. classification                        │       │          │
    │    → 特徵提取(train+test)                 │       │          │
    │    → fit classifier                       │       │          │
    │    → Youden 校準                          │       │          │
    │    → predict & evaluate                   │       │          │
    ├──────────────────────────────────────────┤       │          │
    │ 8. metadata finalization                 │       │          │
    │    → 儲存所有 stage metrics               │       │          │
    └──────────────────────────────────────────┘       │          │
             │                                        │          │
             └────── next experiment ─────────────────┘          │
                                                                 │
             └────── next fold ──────────────────────────────────┘
```
