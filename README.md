# Video Tracking & Analysis Framework

超音波影像追蹤、分割與分類的端到端實驗框架。透過 YAML 設定檔驅動完整的 **偵測 → 分割 → 分類** 流水線，支援多種模型後端、前處理方案、資料分割策略與排程批次執行。

---

## 目錄

- [快速開始](#快速開始)
- [安裝與環境設定](#安裝與環境設定)
- [Pipeline 完整流程](#pipeline-完整流程)
- [YAML 設定檔結構](#yaml-設定檔結構)
- [資料集格式與分割方式](#資料集格式與分割方式)
- [前處理模組](#前處理模組)
- [偵測器（Detection）](#偵測器detection)
- [分割模型（Segmentation）](#分割模型segmentation)
- [分類階段（Classification）](#分類階段classification)
- [評估指標（Evaluation）](#評估指標evaluation)
- [可視化設定](#可視化設定)
- [UI 操作指南](#ui-操作指南)
- [排程系統](#排程系統)
- [輸出結果目錄結構](#輸出結果目錄結構)
- [擴充機制](#擴充機制)

---

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 準備資料集：將影片與對應的 <video>.json 標註放在同一目錄
# 3. 編輯 YAML 設定檔，指定 dataset.root
# 4. 執行流水線
python run_pipeline.py --config pipeline.yolov11_unetpp.yaml
```

也可以透過圖形化介面操作：

```bash
python ui.py
```

---

## 安裝與環境設定

### 基本依賴

| 套件 | 用途 |
|------|------|
| `opencv-python` / `opencv-contrib-python` | 影像處理、CSRT 追蹤器需要 contrib 版 |
| `numpy` | 數值運算 |
| `pyyaml` | 解析 YAML 設定檔 |
| `torch` + `torchvision` | 深度學習模型推論與訓練 |
| `ultralytics>=8.3.0` | YOLOv11 偵測器 |
| `segmentation-models-pytorch>=0.3.3` | UNet / UNet++ / DeepLabV3+ 分割模型 |
| `scikit-learn` | 分類器、評估指標 |
| `matplotlib` | 圖表繪製 |
| `PySide6` | GUI 介面（可選） |
| `tqdm` | 進度列 |
| `timm>=0.9.12` | 視覺 Backbone（ToMP/TaMOs 必要） |

### 可選依賴

| 套件 | 用途 |
|------|------|
| `segment-anything` | MedSAM 分割模型 |
| `xgboost` | XGBoost 分類器 |
| `lightgbm` | LightGBM 分類器 |
| `ocsort` | OC-SORT 追蹤器 |
| `tensorboard`, `tensorboardX`, `easydict`, `lmdb`, `einops` | MixFormerV2 依賴 |
| `scikit-image` | auto_mask 中的 MGAC 演算法 |
| `libs/nnUNet`（以 `-e` 安裝） | nnU-Net v2 分割模型 |

```bash
pip install -r requirements.txt
```

---

## Pipeline 完整流程

整個流水線由 `PipelineRunner`（位於 `tracking/orchestrator/runner.py`）驅動，依序執行以下階段：

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 資料集載入與分割                                              │
│     COCOJsonDatasetManager 掃描 dataset.root                    │
│     依 method 進行 video_level / subject_level / LOSO 分割       │
├─────────────────────────────────────────────────────────────────┤
│  2. 前處理鏈建立                                                 │
│     依 preproc_scheme (A/B/C) 決定前處理作用域                    │
├─────────────────────────────────────────────────────────────────┤
│  3. 偵測器訓練（可選）                                            │
│     k-fold 交叉驗證 → 全量訓練                                   │
│     具備偵測器快取機制（跨執行階段重用已訓練模型）                    │
├─────────────────────────────────────────────────────────────────┤
│  4. 偵測推論                                                     │
│     在測試集上執行偵測，生成 per-frame bounding box                │
├─────────────────────────────────────────────────────────────────┤
│  5. 分割訓練（可選）+ 分割推論                                     │
│     使用偵測 bbox 作為 ROI，裁切後進行遮罩推論                     │
│     前處理 scheme B/C 時從原始影格裁切                             │
├─────────────────────────────────────────────────────────────────┤
│  6. 分類（可選）                                                  │
│     從追蹤軌跡提取特徵 → 訓練/評估分類器                           │
├─────────────────────────────────────────────────────────────────┤
│  7. 評估與可視化                                                  │
│     計算 IoU / 中心誤差 / 成功率 / 漂移率等指標                    │
│     生成視覺化圖片與指標 JSON                                     │
├─────────────────────────────────────────────────────────────────┤
│  8. 輸出 metadata.json                                           │
│     記錄完整的執行資訊、指標、工件路徑                              │
└─────────────────────────────────────────────────────────────────┘
```

### 詳細流程說明

#### 階段 1：資料集載入

`COCOJsonDatasetManager` 會遞迴掃描 `dataset.root` 下所有 `.mp4`/`.avi`/`.mov`/`.mkv` 影片，並嘗試載入同名的 `.json` 標註檔。標註格式為 COCO-VID-like JSON，由標註工具匯出。

#### 階段 2：前處理鏈

每個實驗可定義多個前處理步驟（`type: preproc`），它們會依 `preproc_scheme` 被分配到不同作用域（詳見[前處理模組](#前處理模組)）。

#### 階段 3–4：偵測訓練與推論

若 `train_enabled: true`，偵測器會在訓練集上進行訓練。框架會計算訓練簽名（包含模型參數、前處理配置、訓練資料列表）的 SHA-256 雜湊，若已有相同簽名的已訓練權重，會自動跳過訓練。此快取同時存在於記憶體和磁碟（`results/detector_cache.json`），可跨執行階段重用。

#### 階段 5：分割

分割階段以偵測 bbox 為 ROI，裁切影格後送入分割模型。支援多種後端（詳見[分割模型](#分割模型segmentation)）。訓練完成後權重儲存於 `results/segmentation/`。

#### 階段 6：分類

選擇性階段。從偵測/追蹤軌跡中提取特徵向量，訓練分類器進行受試者/影片層級的分類。

---

## YAML 設定檔結構

完整的設定檔頂層結構如下：

```yaml
seed: 42                    # 全域隨機種子

dataset:                    # 資料集設定
  root: "path/to/dataset"
  split:
    method: video_level     # video_level | subject_level | loso
    ratios: [0.8, 0.2]     # 訓練/測試比例
    k_fold: 1               # k-fold 交叉驗證（1 = 不啟用）

experiments:                # 實驗列表
  - name: "experiment_name"
    preproc_scheme: "A"     # A | B | C
    pipeline:
      - type: preproc       # 前處理步驟
        name: CLAHE
        params: { clipLimit: 2.0, tileGridSize: [8, 8] }
      - type: model          # 偵測模型
        name: YOLOv11
        params: { ... }

evaluation:                 # 評估設定
  evaluator: BasicEvaluator
  restrict_to_gt_frames: true
  visualize:
    enabled: true
    samples: 10
    strategy: even_spread
    include_detection: true
    include_segmentation: true
    ensure_first_last: true

segmentation:               # 分割設定
  model:
    name: unetpp
    params: { encoder_name: resnet34 }
  train: false
  epochs: 5
  # ... 更多選項見下方

classification:             # 分類設定（可選）
  enabled: false
  feature_extractor: { name: basic }
  classifier: { name: random_forest }

output:                     # 輸出設定
  results_root: results
  skip_pip_freeze: false
```

---

## 資料集格式與分割方式

### 資料集目錄結構

```
dataset_root/
├── subject_001/
│   ├── video1.avi
│   ├── video1.json        ← COCO-VID 標註
│   ├── video2.mp4
│   └── video2.json
├── subject_002/
│   ├── ...
```

或扁平結構（受試者 ID 從檔名前綴推斷）：

```
dataset_root/
├── 001Rest.avi
├── 001Rest.json
├── 001Grasp.avi
├── 001Grasp.json
├── 002Rest.avi
├── 002Rest.json
```

> **注意**：受試者 ID 推斷規則——若目錄結構有子資料夾，以第一層資料夾名稱為受試者 ID；否則從影片檔名開頭的連續數字作為 ID（例如 `001Rest.avi` → 受試者 `001`）。

### 分割方式

| 方法 | `method` 值 | 說明 |
|------|------------|------|
| **影片層級** | `video_level` | 隨機將影片分為訓練/測試集（預設） |
| **受試者層級** | `subject_level` | 以受試者為單位分割，同一受試者的所有影片在同一集合 |
| **留一受試者法** | `loso` | Leave-One-Subject-Out：每個受試者輪流作為測試集 |

#### LOSO 進階選項

```yaml
dataset:
  split:
    method: loso
    subjects: ["001", "002"]    # 僅對指定受試者執行（可選）
    max_folds: 3                # 最多執行幾個 fold（除錯用）
    max_train_videos: 10        # 每 fold 最多訓練影片數
    max_test_videos: 5          # 每 fold 最多測試影片數
```

#### K-Fold 交叉驗證

```yaml
dataset:
  split:
    k_fold: 5    # 在訓練集內進行 5-fold 交叉驗證
```

當 `k_fold > 1` 時，框架會在訓練集內進行 k-fold 交叉驗證以評估模型表現（結果存於 `comparison/kfold_summary.json`），之後仍會在完整訓練集上訓練最終模型並在測試集上評估。

---

## 前處理模組

### 前處理作用域方案（Preproc Scheme）

前處理的作用範圍由 `preproc_scheme`（或 `preprocessing_scheme` / `preproc_mode`）控制，分三種方案：

| 方案 | 值 | 偵測器看到的影格 | 分割裁切來源 | 分割 ROI 前處理 |
|------|---|----------------|-------------|----------------|
| **Global** | `A` | 前處理後 | 前處理後 | 無 |
| **ROI** | `B` | 原始影格 | 原始影格 | 套用前處理 |
| **Hybrid** | `C` | 前處理後 | **原始影格** | 套用前處理 |

- **方案 A**（預設）：前處理套用到全幀，偵測器與分割模型都看到前處理後的影像。
- **方案 B**：偵測器看到原始影格；分割模型在 ROI 裁切後才套用前處理。
- **方案 C**：偵測器看到前處理後影格以獲得更好的 bbox；分割模型從原始影格裁切，再於 ROI 上套用前處理。

### 內建前處理模組

#### CLAHE（自適應直方圖均衡化）

```yaml
- type: preproc
  name: CLAHE
  params:
    clipLimit: 2.0           # 對比度限制閾值（預設 2.0）
    tileGridSize: [8, 8]     # 網格大小（預設 [8, 8]）
```

在 LAB 色彩空間的 L 通道上執行 CLAHE，增強局部對比度。適用於超音波影像中結構不清晰的場景。也接受別名 `clip` 和 `grid`。

#### SRAD（斑點降噪各向異性擴散）

```yaml
- type: preproc
  name: SRAD
  params:
    iterations: 10           # 擴散迭代次數（預設 10）
    lambda: 0.15             # 時間步長，穩定範圍 (0, 0.25]（預設 0.15）
    eps: 1.0e-6              # 避免除以零的小值（預設 1e-6）
    convert_gray: true       # RGB 輸入時轉換為灰階處理再還原（預設 true）
```

基於 Yu & Acton (2002) 的簡化實作。使用局部變異係數調節擴散係數，保留邊緣的同時平滑斑點雜訊。特別適合超音波影像。

#### LOG_DR（動態範圍壓縮）

```yaml
- type: preproc
  name: LOG_DR
  params:
    method: log              # 壓縮方法：log | gamma（預設 log）
    gamma: 0.5               # gamma 方法的指數（預設 0.5，<1 提亮暗區）
    clip_percentile: 99.5    # 裁剪高亮值的百分位（預設 99.5）
    eps: 1.0e-6              # 避免 log(0)（預設 1e-6）
    per_channel: true        # RGB 時是否逐通道處理（預設 true）
```

壓縮影像動態範圍以揭示低強度結構。`log` 方法使用 log(1+x) 壓縮；`gamma` 方法使用冪次方曲線。

#### TGC（時間增益補償）

```yaml
- type: preproc
  name: TGC
  params:
    mode: linear             # 增益模式：linear | exp | custom（預設 linear）
    gain_start: 1.0          # 頂部（淺層）增益倍數（預設 1.0）
    gain_end: 2.0            # 底部（深層）增益倍數（預設 2.0）
    exp_k: 1.0               # exp 模式的指數因子（預設 1.0）
    custom_points: null      # custom 模式的 (y_norm, gain) 控制點列表
    per_channel: true        # RGB 時是否逐通道處理（預設 true）
    clip: true               # 是否裁剪至 [0, 255]（預設 true）
```

模擬超音波的深度增益曲線，補償深層回波衰減。三種模式：
- `linear`：從 `gain_start` 線性增長至 `gain_end`
- `exp`：指數增長曲線
- `custom`：使用者自定義分段線性曲線（需提供 `custom_points`）

#### AUGMENT（資料增強）

```yaml
- type: preproc
  name: AUGMENT
  params:
    hflip_prob: 0.5          # 水平翻轉機率（預設 0.5）
    vflip_prob: 0.0          # 垂直翻轉機率（預設 0.0）
    rotate_max_deg: 8.0      # 最大旋轉角度（預設 8.0）
    translate_frac: 0.0      # 最大平移比例（預設 0.0）
    brightness: 0.08         # 亮度抖動幅度（預設 0.08）
    contrast: 0.08           # 對比度抖動幅度（預設 0.08）
    noise_std: 0.0           # 高斯雜訊標準差（預設 0.0）
```

支援幾何變換（翻轉、旋轉、平移）與光度變換（亮度、對比度、雜訊）。同時變換 bbox 和遮罩以保持一致性。

---

## 偵測器（Detection）

目前主要使用 **YOLOv11**（透過 Ultralytics）作為逐幀偵測器。策略為每幀執行偵測，取最高信心度的 bbox。

### 完整參數

```yaml
- type: model
  name: YOLOv11
  params:
    # === 推論參數 ===
    weights: models/detection/yolo11l.pt   # 預訓練/微調權重路徑
    conf: 0.25               # 物件信心度閾值（預設 0.25）
    iou: 0.5                 # NMS IoU 閾值（預設 0.5）
    imgsz: 640               # 推論影像尺寸（預設 640）
    device: cuda              # 裝置：cpu / cuda / cuda:0（預設 auto）
    classes: null             # 限制偵測類別 ID 列表（null = 所有類別）
    max_det: 100              # 每幀最大偵測數（預設 100）
    fallback_last_prediction: true  # 無偵測時沿用上一幀 bbox（預設 true）

    # === 訓練參數 ===
    train_enabled: true       # 是否啟用訓練（預設 true）
    epochs: 30                # 訓練週期數
    batch: 8                  # 批次大小
    lr0: 0.01                 # 初始學習率
    patience: 50              # Early stopping 耐心值
    workers: 0                # 資料載入工作執行緒數
    include_empty_frames: false  # 是否包含無標註的空幀（預設 false）
    force_cpu_if_no_cuda: true   # CUDA 不可用時自動降級到 CPU
```

### 偵測器快取機制

框架會計算每次偵測器訓練的「簽名」（基於模型參數、前處理配置、訓練資料列表等），並將已訓練的權重路徑存入 `results/detector_cache.json`。後續執行時若簽名相同，會自動載入已訓練權重而跳過訓練。此機制跨程式執行階段有效。

### 其他內建追蹤模型

| 模型 | 註冊名稱 | 說明 |
|------|---------|------|
| Template Matching | `TemplateMatching` | 簡單基線，用於驗證流水線 |
| CSRT | `CSRT` | 需要 `opencv-contrib-python` |
| Optical Flow LK | `OpticalFlowLK` | 光流追蹤 |
| Faster R-CNN | `FasterRCNN` | 兩階段偵測器 |
| FastSpeckle/NCC | `FASTSpeckle` | 基於 NCC 的斑點追蹤 |
| OC-SORT | `OC-SORT` | 結合 YOLOv11 偵測的 OC-SORT |
| StrongSORT | `StrongSORT` | 結合外觀特徵的追蹤 |
| StrongSORT++ | `StrongSORT++` | 加上軌跡補點與高斯平滑 |
| ToMP | `ToMP` | pytracking 的 ToMP 追蹤器 |
| TaMOs | `TaMOs` | pytracking 的 TaMOs 追蹤器 |
| MixFormerV2 | `MixFormerV2` | Transformer 追蹤器（需 CUDA） |

> 部分追蹤器支援 `low_confidence_reinit`：當追蹤信心度低於閾值時，自動呼叫 YOLOv11 偵測器重新初始化。

---

## 分割模型（Segmentation）

分割階段以偵測 bbox 為 ROI，在裁切區域上生成像素級遮罩。支援訓練與推論兩種模式。

### 通用分割參數

```yaml
segmentation:
  model:
    name: unetpp             # 模型名稱（見下方支援列表）
    params: { ... }          # 模型特定參數
  train: true                # 是否啟用訓練（預設 false）
  epochs: 5                  # 訓練週期數
  batch_size: 8              # 批次大小
  num_workers: 0             # 載入資料的工作執行緒
  lr: 0.001                  # 學習率
  weight_decay: 1.0e-5       # 權重衰減
  threshold: 0.5             # 分割閾值（logit → binary）
  dice_weight: 1.0           # Dice Loss 權重
  bce_weight: 1.0            # BCE Loss 權重
  val_ratio: 0.0             # 從訓練集分出驗證集的比例
  seed: 0                    # 分割階段的隨機種子
  device: auto               # auto / cpu / cuda
  redundancy: 1              # 每張影格重複採樣次數（資料增強用）
  target_size: [256, 256]    # ROI 裁切尺寸 [H, W]
  padding_min: 0.1           # 訓練時 ROI 最小外擴比例
  padding_max: 0.15          # 訓練時 ROI 最大外擴比例
  padding_inference: 0.15    # 推論時 ROI 外擴比例
  jitter: 0.05               # 訓練時 bbox 抖動幅度
  inference_checkpoint: null  # 推論用的權重路徑（訓練後自動使用最佳權重）
```

### 支援的分割模型

#### UNet (`unet`)

```yaml
model:
  name: unet
  params:
    encoder_name: resnet34       # 編碼器骨幹（見 segmentation_models_pytorch）
    encoder_weights: imagenet    # 預訓練權重（null = 隨機初始化）
    in_channels: 3               # 輸入通道數
    classes: 1                   # 輸出類別數
```

#### UNet++ (`unetpp`)

```yaml
model:
  name: unetpp
  params:
    encoder_name: resnet34
    encoder_weights: imagenet
```

與 UNet 參數相同，但使用密集跳躍連接的 UNet++ 架構。

#### DeepLabV3+ (`deeplabv3+`)

```yaml
model:
  name: "deeplabv3+"
  params:
    encoder_name: resnet34
    encoder_weights: imagenet
```

使用 DeepLabV3+ decoder，具有較大的感受野，適合需要更多全域上下文的場景。

#### MedSAM (`medsam`)

```yaml
model:
  name: medsam
  params:
    checkpoint: models/seg/medsam_vit_b.pth   # 必須下載官方權重
    model_type: vit_b                          # 與 checkpoint 對應
    use_box_prompt: true                       # 使用追蹤 bbox 作為 SAM box prompt
    multimask_output: false                    # 輸出單一遮罩
```

基於 Segment Anything Model（SAM）的醫學影像分割。需安裝 `segment-anything` 套件並下載[官方權重](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)。

**微調模式**：啟用 `train: true` 時，會凍結 image encoder 與 prompt encoder，僅訓練 mask decoder。

#### nnU-Net (`nnunet`)

```yaml
model:
  name: nnunet
  params:
    plans_path: path/to/nnUNetPlans.json       # nnUNet 規劃檔（可選）
    configuration: "2d"                         # 規劃中的配置名稱
    architecture_path: path/to/arch.json        # 自訂架構 JSON（可選）
    return_highres_only: true                   # 僅返回最高解析度 head
```

使用官方 nnU-Net v2 動態架構。需透過 `pip install -e ./libs/nnUNet` 安裝。

#### MedNeXt (`mednext`)

nnU-Net 的 MedNeXt 變體，同樣透過 `libs/MedNeXt` 安裝。

#### auto_mask（弱監督遮罩）

```yaml
model:
  name: auto_mask
  params:
    margin: 0.15             # bbox 外擴比例
    num_iter: 120            # GrabCut 迭代次數
    canny_low: 25            # Canny 邊緣偵測低閾值
    canny_high: 70           # Canny 邊緣偵測高閾值
    guided_radius: 8         # 導向濾波半徑
    guided_eps: 0.01         # 導向濾波 epsilon
```

不需要標註資料的弱監督方法。整合 GrabCut + MGAC（需 scikit-image）+ 導向濾波，在追蹤 bbox 周圍生成遮罩。**僅支援推論**，框架會自動關閉訓練。

#### Torchvision FCN (`torchvision_fcn_resnet50`)

使用 torchvision 官方 FCN-ResNet50，屬推論專用基線。

---

## 分類階段（Classification）

分類為選擇性階段，在偵測/追蹤完成後執行。從軌跡中提取特徵向量，訓練分類器進行受試者層級或影片層級的二元分類（如健康 vs 疾病）。

### 設定

```yaml
classification:
  enabled: true                       # 啟用分類
  label_file: "ann.txt"              # 標籤檔路徑（相對於 dataset.root）
  target_level: "video"              # video | subject
  source_model: null                  # 指定使用哪個偵測模型的預測（多模型時）

  feature_extractor:
    name: "texture_hybrid"
    params: { ... }

  classifier:
    name: "random_forest"
    params: { n_estimators: 300, random_state: 42 }
```

### 標籤檔格式（ann.txt）

```
001 0
002 1
003 0
```

每行 `<受試者ID> <標籤>`（0 = 健康, 1 = 疾病）。影片會根據檔名前綴自動繼承受試者標籤。

### 特徵擷取器

| 名稱 | 說明 |
|------|------|
| `basic` | 基本 bbox 特徵（質心、大小、速度等） |
| `motion_only` | 運動特徵（含卡爾曼平滑）：位移統計、曲率、角度變化、面積動態 |
| `motion_texture` | 運動特徵 + 灰階直方圖紋理 |
| `texture_hybrid` | 運動 + 紋理描述子（梯度/灰階直方圖 + GLCM + LBP） |
| `backbone_texture` | CNN 骨幹嵌入（MobileNetV2/ResNet34/DenseNet121/EfficientNetB2）+ 隨機投影降維 |

#### `texture_hybrid` 參數

```yaml
feature_extractor:
  name: "texture_hybrid"
  params:
    dynamic_params:
      aggregate_stats: ["mean", "std", "max"]
    texture_patch_size: 96       # bbox patch 裁切尺寸
    texture_hist_bins: 16        # 灰階直方圖 bin 數
    max_texture_frames: 3        # 每支影片取樣幀數
```

#### `backbone_texture` 參數

```yaml
feature_extractor:
  name: "backbone_texture"
  params:
    backbone: "EfficientNetB2"    # MobileNetV2 / ResNet34 / DenseNet121 / EfficientNetB2
    pretrained: false             # 是否使用預訓練權重
    reduction_method: "random_projection"   # 降維方法
    reduced_dim: 32               # 投影後維度（預設 64）
    pool_stats: ["mean", "std"]   # 池化聚合方式
    max_texture_frames: 2         # 取樣幀數
    device: "cpu"                 # 推論裝置
    zscore_patch: true            # 進骨幹前做 z-score 標準化
    dynamic_params:
      aggregate_stats: ["mean", "std"]
```

#### CSA 特徵

當影片有分割遮罩結果時，特徵擷取器會自動提取 **Cross-Sectional Area (CSA)** 靜態特徵（首幀/末幀的面積、周長、等效直徑、圓度），附加到特徵向量中。

### 分類器

| 名稱 | 註冊名稱 | 主要參數 |
|------|---------|---------|
| 隨機森林 | `random_forest` | `n_estimators`, `max_depth`, `class_weight`, `random_state` |
| 決策樹 | `decision_tree` | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| SVM | `svm` | `C`, `kernel`, `gamma`, `probability` |
| LightGBM | `lightgbm` | `num_leaves`, `learning_rate`, `n_estimators`, `objective` |
| XGBoost | `xgboost` | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |

### 分類輸出

執行後會在實驗目錄下的 `classification/` 產生：

| 檔案 | 內容 |
|------|------|
| `summary.json` | 準確率、平衡準確率、混淆矩陣、ROC AUC |
| `predictions.json` | 每個實體的預測機率 |
| `classifier.pkl` | 序列化的分類器物件 |

---

## 評估指標（Evaluation）

由 `BasicEvaluator` 計算。

### 偵測/追蹤指標

| 指標 | 說明 |
|------|------|
| **IoU Mean** | 所有匹配幀的平均 Intersection over Union |
| **Center Error Mean** | 預測與真值 bbox 中心點的平均歐幾里得距離（像素） |
| **Success Rate@50** | IoU ≥ 0.50 的幀比例（或 mAP@50） |
| **Success Rate@75** | IoU ≥ 0.75 的幀比例（或 mAP@75） |
| **Success AUC** | 不同 IoU 閾值下成功率的曲線下面積 |
| **FPS** | 推論速度（每秒幀數） |
| **Drift Rate** | 連續幀中 bbox 漂移超過閾值的比例 |

### 分割指標

| 指標 | 說明 |
|------|------|
| **Dice/F1** | 遮罩與真值的 Dice 係數 |
| **IoU** | 遮罩的 Intersection over Union |
| **面積** | 預測遮罩的像素面積 |
| **FPS** | 分割推論速度 |

---

## 可視化設定

```yaml
evaluation:
  visualize:
    enabled: true              # 是否啟用可視化
    samples: 10                # 每支影片可視化幀數
    strategy: even_spread      # 取樣策略：even_spread（均勻分佈）
    include_detection: true    # 是否繪製偵測 bbox
    include_segmentation: true # 是否繪製分割遮罩
    ensure_first_last: true    # 確保包含首幀與末幀
```

可視化結果儲存於 `test/detection/visualizations/` 和 `test/segmentation/predictions/` 中。

---

## UI 操作指南

```bash
python ui.py
```

### 設計原則

- **即時雙向同步**：Builder（左側表單）和 Raw YAML（右側編輯器）雙向同步
  - Builder 改動 → 即時生成 Raw YAML
  - Raw 編輯 → 停止輸入 ~0.8s 後自動解析回 Builder
- **無需手動存檔**：執行永遠使用當下 UI 狀態
- **防滾輪誤觸**：SpinBox 需點選後才能調整

### 基本操作流程

1. **選擇 Dataset Root** → 指定資料集路徑
2. **選擇/排序前處理** → 設定 preproc_scheme 與各模組參數
3. **選擇偵測模型** → 設定 YOLOv11 或其他追蹤器參數
4. **設定分割模型** → 選擇後端與訓練/推論模式
5. **設定分類**（可選）→ 勾選啟用並配置特徵擷取器/分類器
6. **設定評估/可視化** → 調整指標與可視化選項
7. **執行** 或 **加入排程隊列**

### 匯入現有設定檔

1. 在上方輸入或瀏覽 `*.yaml` / `*.json` 路徑
2. 點「載入設定檔」
3. Raw 顯示內容 → 自動解析回 Builder → 可繼續圖形化調整

---

## 排程系統

### 排程 YAML 格式

排程檔使用 `queue` 列表，每個項目包含一組完整的設定：

```yaml
queue:
  - label: "00_defs"           # 共用定義（可用 YAML 錨點）
    config:
      seed: 42
      dataset: &dataset_cfg
        root: "path/to/data"
        split: { method: loso }
      experiments: []           # 空 = 僅定義不執行

  - label: "01_experiment_a"
    config:
      seed: 42
      dataset: *dataset_cfg    # 引用共用定義
      experiments:
        - name: "exp_a"
          pipeline:
            - type: model
              name: YOLOv11
              params: { ... }
      evaluation: { ... }
      segmentation: { ... }
```

### CLI 執行排程

```bash
# 執行整份排程
python run_pipeline.py --config schedules/my_schedule.yaml

# 僅執行指定項目
python run_pipeline.py --config schedules/my_schedule.yaml --queue-label "01_experiment_a"
python run_pipeline.py --config schedules/my_schedule.yaml --queue-index 1
```

### UI 排程功能

1. 在 Builder 中配置好參數
2. 點「加入排程」→ 設定快照加入排程隊列
3. 可自訂實驗名稱
4. 點「執行排程」→ 依序執行

多實驗排程會自動建立 `results/yyyy-mm-dd_HH-MM-SS_schedule_<N>exp/` 目錄。

---

## 輸出結果目錄結構

```
results/
└── 2026-02-23_15-30-00_experiment_name/
    ├── metadata.json                    ← 完整執行記錄
    ├── logs/
    │   └── run.log                      ← 執行日誌
    ├── train_full/
    │   ├── detection/                   ← 偵測器訓練工件
    │   │   └── fold_1/ ... fold_k/      ← k-fold 訓練結果
    │   └── segmentation/                ← 分割訓練工件
    ├── test/
    │   ├── detection/
    │   │   ├── metrics/
    │   │   │   ├── summary.json         ← 偵測指標彙總
    │   │   │   └── <video>/summary.json ← 逐影片指標
    │   │   ├── predictions/
    │   │   │   └── <model>.json         ← 偵測預測結果
    │   │   └── visualizations/
    │   │       └── <video>/frame_*.jpg  ← 可視化圖片
    │   └── segmentation/
    │       ├── metrics_summary.json     ← 分割指標彙總
    │       └── predictions/
    │           └── <seg_model>/<det_model>/<video>/
    │               ├── frame_*.png      ← 分割遮罩
    │               ├── metrics.json     ← 逐影片分割指標
    │               └── roi_trace.json   ← ROI 來源追蹤
    ├── comparison/
    │   └── kfold_summary.json           ← k-fold 指標彙總
    ├── classification/
    │   ├── summary.json                 ← 分類指標
    │   ├── predictions.json             ← 預測結果
    │   └── classifier.pkl               ← 分類器模型
    └── predictions_by_video/
        └── <video>/<model>.json         ← 逐影片逐模型預測
```

### metadata.json

每次實驗執行都會生成 `metadata.json`，包含：
- 建立時間、隨機種子
- 資料集資訊（影片數、標註數、分割方式）
- 完整的實驗設定
- 環境資訊（Python 版本、套件列表）
- 各階段的執行記錄（開始/結束時間、狀態、指標）
- 所有工件的路徑

---

## 擴充機制

框架使用 **註冊表裝飾器** 模式，可輕鬆擴充各元件：

### 新增前處理模組

```python
# tracking/preproc/my_preproc.py
from ..core.registry import register_preproc
from ..core.interfaces import PreprocessingModule

@register_preproc("MyPreproc")
class MyPreproc(PreprocessingModule):
    name = "MyPreproc"
    def __init__(self, config):
        self.param = config.get("param", 1.0)
    def apply_to_frame(self, frame):
        return frame  # 自訂處理
    def apply_to_video(self, video_path):
        return video_path
```

### 新增追蹤模型

```python
from ..core.registry import register_model
from ..core.interfaces import TrackingModel

@register_model("MyTracker")
class MyTracker(TrackingModel):
    name = "MyTracker"
    def train(self, train_dataset, val_dataset=None, seed=0, output_dir=None):
        return {}
    def predict(self, video_path):
        return []  # 回傳 List[FramePrediction]
    def load_checkpoint(self, ckpt_path):
        pass
```

### 新增特徵擷取器

```python
from ..core.registry import register_feature_extractor

@register_feature_extractor("my_extractor")
class MyExtractor:
    def extract(self, samples, video_path=None):
        return {"feature_1": 0.5, "feature_2": 1.0}
```

### 新增分類器

```python
from ..core.registry import register_classifier

@register_classifier("my_classifier")
class MyClassifier:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def save(self, path): ...
    def load(self, path): ...
```

### 新增分割模型

```python
from ..core.registry import register_segmentation_model

@register_segmentation_model("my_seg_model")
class MySegModel:
    # 實作所需介面
    pass
```

註冊後新元件會自動出現在 YAML 設定與 GUI 中。

---

## Tools Workbench

```bash
python tools\tools_workbench.py
```

| 分頁 | 功能 |
|------|------|
| **排程結果瀏覽** | 排序、複製或匯出實驗指標 |
| **信心診斷** | 掃描追蹤預測計算平滑信心值，顯示 P10/P05、最低值、連續低信心段落等統計 |

信心診斷低信心門檻預設 0.6，可從分頁右上角調整。若追蹤分數長期鎖在 1.0 或漂移組件偏低，建議啟用 `low_confidence_reinit` 或檢查初始標註品質。

