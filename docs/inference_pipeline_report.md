# 推論流水線資料流詳細報告

> 本文件完整記錄從測試集輸入到最終分類結果的每一步推論資料轉換與處理邏輯。

---

## 目錄

1. [總覽：推論階段架構](#1-總覽推論階段架構)
2. [Detection 推論](#2-detection-推論)
3. [軌跡濾波（Trajectory Filter）](#3-軌跡濾波trajectory-filter)
4. [Segmentation 推論](#4-segmentation-推論)
5. [Classification 推論](#5-classification-推論)
6. [評估指標計算](#6-評估指標計算)
7. [資料流全局圖](#7-資料流全局圖)

---

## 1. 總覽：推論階段架構

推論階段在每個 LOSO fold 的 experiment 中，按以下順序執行：

```
Stage 3: detector_eval          → 產出 test_predictions
Stage 4: trajectory_filter      → 平滑 test_predictions（就地替換）
Stage 6: segmentation_infer     → 使用平滑後的 bbox 做 ROI 裁切 + 分割
Stage 7: classification (test)  → 使用 detection + segmentation 結果做特徵提取 → 分類預測
```

**關鍵資料結構**：

```python
test_predictions: Dict[str, Dict[str, List[FramePrediction]]]
#                 └ model_name  └ video_path  └ 每幀預測

FramePrediction:
  frame_index: int                          # 幀編號
  bbox: (x, y, w, h)                        # 像素座標
  score: Optional[float]                    # 偵測信心分數
  confidence: Optional[float]               # 複合信心（FASTSpeckle 專用）
  confidence_components: Optional[Dict]     # 信心各分量
  segmentation: Optional[SegmentationData]  # 分割結果（後填）
  is_fallback: bool                         # 是否為 fallback 填充
  bbox_source: str                          # "detector" / "prev_bbox" / "segmentation_bootstrap" 等
```

---

## 2. Detection 推論

### 2.1 總流程（`run_on_dataset`）

對測試集每支影片、每個模型執行：

```
for video in test_ds:
    for model in models:
        predictions = model.predict(video_path)     # 或 model.predict_frames(vp, gt_indices)
        → List[FramePrediction]
```

### 2.2 YOLOv11 推論

#### 逐幀處理管線

```
步驟 1:  cv2.VideoCapture(video_path) 開啟影片

步驟 2:  逐幀讀取（或按 gt_frame_indices 跳讀）

步驟 3:  前處理 _apply_preprocs_np(frame):
         BGR → Gray → BGR → RGB
         → 依序執行 preproc chain（跳過 train_only=True 的模組）
         → RGB → BGR

步驟 4:  批次推論（每 inference_batch=4 幀一批）:
         BGR → RGB → self.model.predict(batch, conf, iou, imgsz, device, classes, max_det)

步驟 5:  結果解析:
         取每幀最高信心的偵測結果
         boxes.xyxy → (x1,y1,x2,y2) → (x, y, w, h)
         score = float(confs[best])

步驟 6:  Fallback（若該幀無偵測結果）:
         fallback_last_prediction=True → 複製上一幀 bbox
         is_fallback=True, score=None, bbox_source="detector"

步驟 7:  FPS 計時:
         runner 層級在 predict 前後計時
```

#### 推論模式

| 模式 | 觸發條件 | 行為 |
|------|---------|------|
| `predict(video_path)` | 正常模式 | 處理所有幀 |
| `predict_frames(vp, frame_indices)` | `restrict_to_gt_frames=True` | 僅處理 GT 有標註的幀（跳讀） |

### 2.3 FasterRCNN 推論

```
步驟 1:  cv2.VideoCapture 讀取

步驟 2:  前處理 _apply_preprocs_np (同 YOLO)

步驟 3:  批次推論（inference_batch=4）:
         BGR → RGB → float32 / 255.0 → tensor → model (eval mode, no_grad)

步驟 4:  結果篩選:
         先篩 scores >= score_thresh (0.5)，取最高信心
         若全部低於閾值 → 取全域最高
         boxes (x1,y1,x2,y2) → (x,y,w,h)

步驟 5:  Fallback:
         同 YOLO — 無偵測 → 上一幀 bbox
```

### 2.4 TemplateMatching 推論

```
步驟 1:  首幀初始化:
         resolve_first_frame_bbox() → 取 GT 或 detector 的 bbox
         → 從灰階幀擷取 template patch

步驟 2:  後續幀:
         定義搜尋 ROI = 前一幀位置 ± search_margin
         cv2.matchTemplate(roi, template, TM_CCOEFF_NORMED)
         minMaxLoc 取最佳位置 → 更新 bbox（固定尺寸）

步驟 3:  Confidence = 固定 1.0（所有幀）
```

### 2.5 OpticalFlowLK 推論

```
步驟 1:  首幀初始化:
         resolve_first_frame_bbox() → bbox 區域
         goodFeaturesToTrack(max_corners=50, qualityLevel=0.01) → 特徵點

步驟 2:  後續幀:
         calcOpticalFlowPyrLK(prev_gray, gray, pts, winSize, maxLevel=3)
         → 篩選 status==1 的追蹤成功點
         → 若 ≥ 3 點 → 中位數位移更新 bbox（固定尺寸）
         → 若全失 → 沿用前一幀 bbox

步驟 3:  早停:
         stop_at_last_gt=True → 超過最後 GT 幀 +2 後停止
         max_eval_frames → 限制最大幀數

步驟 4:  Confidence = 固定 1.0
```

### 2.6 FASTSpeckle 推論

```
步驟 1:  首幀初始化:
         resolve_first_frame_bbox() → bbox ROI
         FAST 角點偵測(threshold=20, max_features=200)
         → 失敗 → GFTT → 失敗 → goodFeaturesToTrack

步驟 2:  每 reinit_interval=10 幀或特徵 < min_features=30:
         在當前 bbox ROI 重新偵測特徵

步驟 3:  LK 追蹤:
         calcOpticalFlowPyrLK(winSize=21, maxLevel=3)
         → 中位數位移更新 bbox

步驟 4:  Confidence 計算（最精細的模型）:
         ratio = valid_tracked / base_count
         err_factor = clip(median_LK_error / (win² × (maxLevel+1)), 0, 1)
         confidence = clip(ratio × (1 - err_factor), 0, 1)

步驟 5:  低信心重新初始化:
         若 confidence < threshold=0.3 且距上次 reinit ≥ min_interval=15:
         → detect_bbox_on_frame() 用 YOLO 重新偵測
         → 成功 → 重設 bbox + 特徵點, confidence=1.0
```

### 2.7 各模型 Confidence 比較

| 模型 | Confidence 類型 | 值域 | 說明 |
|------|----------------|------|------|
| YOLOv11 | objectness × class_prob | [0, 1] | YOLO 原生信心 |
| FasterRCNN | classification score | [0, 1] | RPN + head 輸出 |
| TemplateMatching | 固定值 | 1.0 | 無品質評估 |
| OpticalFlowLK | 固定值 | 1.0 | 無品質評估 |
| FASTSpeckle | 追蹤品質複合分數 | [0, 1] | ratio × (1 - err_factor) |

---

## 3. 軌跡濾波（Trajectory Filter）

### 3.1 觸發條件

- `trajectory_filter.enabled=True`（預設開啟）
- `test_predictions` 非空

### 3.2 完整濾波管線

對每個 model 的每支影片的 predictions：

```
步驟 1:  去重（Deduplicate）
         按 frame_index 去重，保留第一個出現的預測

步驟 2:  抽取軌跡陣列
         cx = x + w/2,  cy = y + h/2
         widths,  heights,  frame_indices

步驟 3:  計算 Before 指標（詳見 §3.5）

步驟 4:  平滑質心 — smooth_trajectory_2d(cx, cy)
         對 x, y 分別:
         ├── 多尺度 Hampel 濾波
         │   ├── Mirror padding（反射填充）
         │   ├── Stage 1 (macro): half_window = max(5, N×0.15/2), σ=3.0
         │   │   → 移除大範圍長連續異常
         │   └── Stage 2 (micro): half_window=7, σ=3.0
         │       → 移除點狀離群值
         │   └── OR 合併兩階段 outlier mask
         │
         └── 雙向 Savitzky-Golay
             ├── 前向 savgol_filter(values, window, polyorder)
             ├── 反向 savgol_filter(reversed, window, polyorder)[::-1]
             └── 取平均 (fwd + bwd) / 2   ← 消除 phase shift

步驟 5:  BBox 尺寸策略
         ┌─ "hampel_only"（預設）
         │   w, h 僅做 Hampel 去離群，不做 S-G 平滑
         ├─ "none"
         │   保留原始 w, h，不做任何平滑
         ├─ "independent"
         │   w, h 各自 Hampel + S-G，clamp ≥ 1.0
         ├─ "fixed_global_roi"
         │   全序列取 P95 的 w/h 作為固定尺寸
         └─ "area_constraint"
             面積幀間變化率 >20% → NaN → cubic spline 插值 → S-G 平滑

> ℹ️ 預設從 `"none"` 改為 `"hampel_only"`，原因是保留 Hampel 去除極端 w/h 離群值，
> 同時避免 S-G 對尺寸造成過度平滑，降低 IoU。

步驟 6:  計算 After 指標

步驟 7:  重建 FramePrediction
         平滑後的 (cx, cy, w, h) → 反算 (x, y, w, h)
         保留原始 score, confidence, segmentation, is_fallback, bbox_source

步驟 8:  test_predictions 就地替換
         ⚠️ 所有下游（segmentation、classification）使用的是平滑後版本
```

### 3.3 時間感知自適應 S-G（非均勻間距）

若幀間距不均勻（稀疏推論）：
1. Cubic spline 重採樣到均勻 1-frame 網格
2. 依密度比例縮放 S-G 視窗大小
3. 在均勻網格上做 bidirectional S-G
4. Cubic spline 回到原始 frame 位置
5. 設有 `_MAX_DENSE=10000` 記憶體保護

### 3.4 GT 軌跡平滑（Classification 訓練用）

| 項目 | Detection 推論 | GT 訓練集 |
|------|---------------|----------|
| Hampel | ✅ 多尺度 | ❌ 跳過（`skip_hampel=True`） |
| S-G window | 11 | 7 |
| S-G polyorder | 2 | 1 |
| 理由 | detector 有離群值噪聲 | GT 無離群值，只需輕量平滑 |

### 3.5 Before/After 指標

| 指標 | 公式 |
|------|------|
| `jitter_cx/cy` | $\text{std}(\Delta c / \Delta t)$ — 速度標準差 |
| `jitter_w/h` | $\text{std}(\Delta s / \Delta t)$ — 尺寸變化速度標準差 |
| `smoothness_cx/cy` | $\text{mean}(\|\Delta^2 c / \Delta t^2\|)$ — 平均加速度 |
| `area_stability` | $1 - \sigma(A)/\bar{A}$ |
| `path_length` | $\sum \sqrt{(\Delta cx)^2 + (\Delta cy)^2}$ |

### 3.6 Filtered Detection Accuracy

濾波後額外重新評估 bbox vs GT：
- 使用同樣的 IoU / Center Error / SR@0.5 / SR@0.75 定義
- 結果存入 `trajectory_filter/filtered_detection_summary.json`
- 供 Viewer 顯示「濾波後 Detection 彙總」

---

## 4. Segmentation 推論

### 4.1 輸入

- 來自 **Stage 4 軌跡濾波後**的 `test_predictions[model_name][video_path]`
- 每個 `FramePrediction` 提供 bbox 作為 ROI 裁切依據

### 4.2 ROI Fallback 機制（三層策略）

| 優先順序 | 策略 | 條件 | padding | bbox_source |
|----------|------|------|---------|-------------|
| **Best** | 當幀 detector bbox | `w ≥ 2 && h ≥ 2` | `padding_inference` (0.15) | `"detector"` |
| **1** | 上一幀 bbox | `last_bbox_raw` 有效 | `max(padding_inference, 0.15)` | `"prev_bbox"` |
| **0** | 全幀 ROI | 前兩者都不可用 | N/A（整張影像） | `"full_frame"` |

**Segmentation Bootstrap (Strategy 2)**：
- 全幀分割後 → 若 mask 非空 → 從 mask_stats.bbox 提取 bbox
- 寫入 `last_bbox_raw` → 後續幀可用 Strategy 1
- `bbox_source = "segmentation_bootstrap"`

**設計重點**：`last_bbox_raw` 只儲存**未擴展的原始 bbox**，防止 padding 累積放大。

### 4.3 每幀推論管線（DNN 模型路徑）

```
步驟 1:  cap.read(frame_idx) → BGR 影像

步驟 2:  全域前處理 _apply_preprocs_frame(frame, self.preprocs)
         → BGR → Gray → BGR → preproc chain → 結果
         ⚠️ Scheme B/C 下 self.preprocs = []（空），不做全域前處理

步驟 3:  ROI 選擇（3 層 fallback）
         → 選定 use_bbox_raw

步驟 4:  expand_bbox(use_bbox_raw, pad_fraction, frame_shape)
         → 以 bbox 中心向四邊擴展 pad_fraction × (w, h)
         → clamp 到影像邊界

步驟 5:  crop_with_bbox(frame, expanded_bbox)
         → 裁切 ROI 影像

步驟 6:  ROI 前處理 _apply_preprocs_frame(roi, self.roi_preprocs)
         → 跳過 train_only=True 的模組
         ⚠️ Scheme A 下 self.roi_preprocs = []（空）

步驟 7:  [MedSAM 專用] 投影 bbox 到 ROI 座標空間
         → 建構 box prompt + center point prompt

步驟 8:  cv2.resize(roi, target_size)
         → (256, 256)，INTER_LINEAR

步驟 9:  歸一化 + tensor 化
         → roi / 255.0 → CHW → unsqueeze(0) → to(device)

步驟 10: 模型推論（帶 CUDA fallback）
         → logits = model(roi_tensor)
         → CUDA 失敗 → 自動降回 CPU 重試

步驟 11: sigmoid(logits) → 機率圖
         → (probs > threshold) × 255 → 二值 mask

步驟 12: 後處理
         → keep_largest_component(mask)    ← 保留最大連通區域
         → fill_holes(mask)                ← 形態學填充孔洞

步驟 13: 恢復原始尺寸
         → cv2.resize(mask, orig_roi_size, INTER_NEAREST)
         → 再做一次 keep_largest_component + fill_holes

步驟 14: 放回全幀座標
         → place_mask_on_canvas(frame_shape, mask_roi, bbox)

步驟 15: 計算 mask 統計
         → compute_mask_stats(full_mask)
         → area_px, bbox, centroid, perimeter, equivalent_diameter

步驟 16: [若有 GT] 評估
         → 讀取 GT mask → fill_holes + keep_largest_component
         → dice, iou, centroid_distance

步驟 17: 存檔
         → cv2.imwrite(full_mask) → PNG

步驟 18: 回寫 FramePrediction
         → pred.segmentation = SegmentationData(stats, ...)
```

### 4.4 MedSAM 特殊推論路徑

| 步驟 | 說明 |
|------|------|
| Prompt 建構 | detector bbox 投影到 ROI 座標 → box prompt + center point |
| 推論 | `SamPredictor.predict(box, point_coords, point_labels)` |
| Mask 選擇 | 在多個候選 mask 中選 score 最高的 |
| 後處理 | 同 DNN 路徑 |

### 4.5 Auto-Mask 路徑（傳統 CV）

```
步驟 1:  _extract_roi_with_reflect(frame, bbox, expand_ratio)
步驟 2:  _ensure_gray(roi)
步驟 3:  GrabCut 初始分割 → 失敗 → 橢圓 fallback
步驟 4:  Canny edge → edge_weight = exp(-edges × β)
步驟 5:  [可選] morphological geodesic active contour
步驟 6:  morphologyEx: OPEN + CLOSE
步驟 7:  guided filter(gray, mask, radius, eps)
步驟 8:  [可選] dilate
步驟 9:  strip_padding → remove_boundary → largest_component
步驟 10: place on full-frame canvas
```

### 4.6 ROI Fallback Rate 統計

推論完成後計算：
- `roi_fallback_rate` = 非 `"detector"` bbox_source 的幀數 / 總幀數
- 注入 detection metrics JSON → 供 Viewer 顯示

### 4.7 視覺化

**`_render_segmentation_visualizations()`**：
- 灰階 ROI 上疊加 GT (綠) / Pred (紅) / Overlap (黃)
- Error map：FP=橙色, FN=藍色
- 均等間隔選取最多 N 幀

---

## 5. Classification 推論

### 5.1 測試資料準備

```
test_predictions[source_model]
  │
  ├─ 每支影片的 List[FramePrediction]
  │   （已含 trajectory filter 後的 bbox + segmentation 結果）
  │
  ├─ feature_extractor.extract_video(samples, video_path)
  │   → 每支影片 → 特徵字典
  │
  ├─ feature_extractor.finalize_batch(feat_dicts, fit=False)
  │   → 使用訓練時 fit 的降維器與全域標準化參數（不重新 fit）
  │
  ├─ [若 level=subject] aggregate_subject(video_features)
  │
  ├─ _filter_entities() → 只保留有標籤的實體
  │
  └─ FeatureVectoriser.transform(features) → X_test (N, D)
```

### 5.2 預測流程

```
步驟 1:  classifier.predict_proba(X_test) → 正類機率

步驟 2:  用校準後的 decision_threshold 決定預測:
         y_pred = np.where(prob >= threshold, 1, 0)

步驟 3:  若所有機率 == 0（無效機率）:
         回退到 classifier.predict(X_test) 直接結果

步驟 4:  summarise_classification(y_test, y_pred, prob)
```

### 5.3 TSC 分類器特殊處理

TSC 分類器（MultiRocket, PatchTST, TimeMachine）接收 `(N, C×T)` 的展平向量，內部會：

| 分類器 | 內部處理 |
|--------|---------|
| MultiRocket | reshape → `(N, C, T)` → （可選）C-axis ChannelReducer(`pca/lda/umap`) → （可選）T-axis Autoencoder → 84 kernels × dilations × 4 pooling |
| PatchTST | reshape → `(N, C, T)` → （可選）C-axis learned channel projection → （可選）T-axis learned temporal projection → Transformer encoder → classification head |
| TimeMachine | reshape → `(N, C, T)` → （可選）C-axis learned channel projection → （可選）T-axis learned temporal projection → TimeMachine 模型 → classification head |

---

## 6. 評估指標計算

### 6.1 Detection 指標（BasicEvaluator）

| 指標 | 計算方式 | 說明 |
|------|---------|------|
| **IoU Mean ± Std** | 所有有 GT 且有 pred 的幀之 bbox IoU | 標準交並比 |
| **Center Error (pix)** | $\sqrt{(cx_p - cx_{gt})^2 + (cy_p - cy_{gt})^2}$ | 中心歐氏距離 |
| **SR@0.5** | $\frac{TP_{0.5}}{TP_{0.5} + FP_{0.5}}$ | IoU ≥ 0.5 的精確率 |
| **SR@0.75** | $\frac{TP_{0.75}}{TP_{0.75} + FP_{0.75}}$ | IoU ≥ 0.75 的精確率 |
| **Success AUC** | IoU 閾值 0.00~1.00（步長 0.01），每個計算 success rate，取平均 | OTB 風格 |
| **Drift Rate** | $\frac{1}{N-1}\sum \|CE_i - CE_{i-1}\|$ | 相鄰幀 CE 的平均變化 |
| **FPS** | runner 計時（predict 前後） | 純推論速度 |
| **AUROC** | sklearn `roc_auc_score(gt_iou_binary, confidence)` | 信心分數品質 |
| **ROI Fallback** | 非 detector bbox_source 幀數 / 總幀數 | 偵測失敗率 |

#### Detection-style 計數規則

| GT | Pred | IoU ≥ thr | 結果 |
|----|------|-----------|------|
| ✅ | ✅ | ✅ | **TP** |
| ✅ | ✅ | ❌ | **FP** |
| ✅ | ❌ | — | **FN** |
| ❌ | ✅ | — | **FP** |

### 6.2 Segmentation 指標

| 指標 | 公式 | 空 mask 處理 |
|------|------|-------------|
| **Dice** | $\frac{2\|P \cap T\|}{\|P\| + \|T\| + \epsilon}$ | 兩者皆空 → 1.0 |
| **IoU** | $\frac{\|P \cap T\|}{\|P \cup T\| + \epsilon}$ | 兩者皆空 → 1.0 |
| **Centroid (pix)** | $\sqrt{(cx_p - cx_{gt})^2 + (cy_p - cy_{gt})^2}$ | 任一 area=0 → None |
| **FPS** | 推論計時 | — |

**GT 比對**：GT mask → `fill_holes + keep_largest_component`，與預測 mask 在全幀座標系比較。

### 6.3 Classification 指標

| 指標 | 計算 |
|------|------|
| `accuracy` | `sklearn.accuracy_score` |
| `balanced_accuracy` | `sklearn.balanced_accuracy_score` |
| `precision_positive` | `precision_recall_fscore_support(labels=[1])` |
| `recall_positive` | 同上 |
| `f1_positive` | 同上 |
| `tn, fp, fn, tp` | `confusion_matrix` |
| `roc_auc` | `roc_auc_score(y_true, y_prob)` |

---

## 7. 資料流全局圖

### 7.1 端到端資料流

```
測試影片 (video_path)
  │
  ├──▶ Detection 推論
  │    ┌─────────────────────────────────────────────────┐
  │    │ VideoCapture → 逐幀讀取                          │
  │    │ → 前處理管線 (BGR→Gray→BGR→RGB→chain→BGR)       │
  │    │ → 模型推論 (YOLO/FRCNN/Template/LK/FAST)       │
  │    │ → Fallback: 無偵測 → 上一幀 bbox                │
  │    │ → List[FramePrediction]                         │
  │    └─────────────────────────────────────────────────┘
  │    ↓ test_predictions
  │
  ├──▶ 軌跡濾波
  │    ┌─────────────────────────────────────────────────┐
  │    │ cx, cy 提取                                     │
  │    │ → 多尺度 Hampel (macro+micro)                   │
  │    │ → 雙向 S-G 平滑                                 │
  │    │ → bbox 尺寸策略 (none/hampel_only/independent/fixed/area) │
  │    │ → 重建 FramePrediction                          │
  │    │ → ⚠️ test_predictions 就地替換                    │
  │    │ → filtered_detection_summary.json               │
  │    └─────────────────────────────────────────────────┘
  │    ↓ 平滑後 test_predictions
  │
  ├──▶ Segmentation 推論
  │    ┌─────────────────────────────────────────────────┐
  │    │ for 每幀:                                       │
  │    │   全域前處理(Scheme A)或跳過(B/C)               │
  │    │   ROI 選擇 (3 層 fallback)                      │
  │    │   expand_bbox → crop ROI                        │
  │    │   ROI 前處理(Scheme B/C)或跳過(A)               │
  │    │   resize → model(logits) → sigmoid → threshold  │
  │    │   → keep_largest → fill_holes → resize back     │
  │    │   → place on canvas → compute_mask_stats        │
  │    │   → 比對 GT → dice/iou/centroid                 │
  │    │   → 寫回 FramePrediction.segmentation           │
  │    └─────────────────────────────────────────────────┘
  │    ↓ 帶 segmentation 的 test_predictions
  │
  └──▶ Classification 推論
       ┌─────────────────────────────────────────────────┐
       │ for 每支影片:                                    │
       │   extract_video(predictions)                     │
       │   → 運動特徵 + 分割特徵 + 紋理/ResNet特徵       │
       │                                                  │
       │ finalize_batch(fit=False)                        │
      │   → 降維 + 全域標準化轉換（用訓練時 fit 的參數） │
       │                                                  │
       │ [subject level] aggregate_subject()              │
       │   → mean/std/min/max 或逐元素 mean              │
       │                                                  │
       │ FeatureVectoriser → X_test (N, D)               │
       │                                                  │
       │ classifier.predict_proba() → prob                │
       │ apply threshold → y_pred                         │
       │ summarise_classification → metrics               │
       └─────────────────────────────────────────────────┘
```

### 7.2 不同 Scheme 下的推論前處理差異

| 推論環節 | Scheme A | Scheme B | Scheme C |
|---------|----------|----------|----------|
| Detection 前處理 | CLAHE 等 | 無 | CLAHE 等 |
| Seg 全域前處理 | CLAHE 等 | 無 | 無 |
| Seg ROI 前處理 | 無 | CLAHE 等 | CLAHE 等 |
| Detection 看到的影像 | 增強後 | 原始 | 增強後 |
| Seg 裁切來源 | 增強後全幀 | 原始全幀 | 原始全幀 |
| Seg ROI 輸入 | 原始 crop | 增強 crop | 增強 crop |

### 7.3 關鍵設計觀察

1. **Trajectory Filter 是單向的**：只平滑 test_predictions，不影響訓練資料
2. **Segmentation 使用的是平滑後的 bbox**：因為 test_predictions 已被就地替換
3. **Classification 使用的是完整 pipeline 結果**：detection bbox + segmentation mask 都會反映在特徵中
4. **降維器/標準化器不重新 fit**：測試時使用訓練時建立的映射（如 PCA/UMAP/LDA/AE + Z-score），確保特徵空間一致
5. **Decision threshold 來自訓練集 LOSO 校準**：非固定 0.5 閾值
6. **ROI Fallback 會傳播**：若連續多幀無偵測，segmentation bootstrap 可以「自救」
