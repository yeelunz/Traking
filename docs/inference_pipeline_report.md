# 推論階段 (Inference Pipeline) 取樣及資料流報告

> 本文件完整記錄測試集（Test Set）從輸入影像到最終分類結果的推論資料流。
這份文件針對代碼審查（Code Review）設計，提供大架構的檢核以及揭露與預期不符的潛在設計盲點。

---

## 1. 總覽：推論階段架構

在每個 LOSO (Leave-One-Subject-Out) fold 中，推論會順序執行以下階段：

1. **Stage 3: `detector_eval`** → 在測試影片上進行物件偵測，產出 `test_predictions`
2. **Stage 4: `trajectory_filter`** → 對偵測軌跡進行離群值過濾與雙向 S-G 平滑（就地替換）
3. **Stage 6: `segmentation_infer`** → 基於平滑後的 bbox 進行 ROI 裁切，產生分割 Mask
4. **Stage 7: `classification (test)`** → 利用 Detection + Segmentation 結果提取特徵並進行分類

**核心資料結構 (`test_predictions`)**:
各 Stage 會在這份結構中逐步「填入」其處理結果：
```python
test_predictions: Dict[str, Dict[str, List[FramePrediction]]]
#                 └ model_name  └ video_path  └ 逐幀 FramePrediction
```

---

## 2. Detection 推論 (Stage 3)

### 2.1 執行流程
1. `runner.py` 讀取 Video，可選用跳讀模式（若 `restrict_to_gt_frames=True`）
2. **全域前處理**: 將各影像套入 Config 制定的前處理邏輯（例如 `clahe`, `gray` 等）
3. **模型推論**: YOLOv11、FasterRCNN、FASTSpeckle 等模型分別對 Batch 執行推論。
4. **Fallback (後備機制)**: 當幀若完全無偵測框，系統預設 `is_fallback=True`，沿用**上一幀**的 bbox。

**Confidence 各模型定義差異**:
- YOLOv11 / FasterRCNN: 原生 Objectness 或 Class Probability `[0, 1]`
- Template / LK: 固定值 `1.0` (無品質評估)
- FASTSpeckle: LK 追蹤品質複合分數 (成功追蹤特徵點比例 與 位移誤差 的組合)

---

## 3. Trajectory Filter 軌跡濾波 (Stage 4)

推論時因為 Detector 通常伴隨較大的晃動與誤判，必須加入時序濾波。

### 3.1 濾波管線
1. **去重疊**: 依據 `frame_index` 去除同幀重複的預測框。
2. **中心點平滑 (cx, cy)**: 
   - 雙階段 Hampel Filter (Macro + Micro): 移除連續漂移與點狀離群值。
   - 雙向 Savitzky-Golay (S-G) 平滑: 結合前後幀走向，無相位延遲。
3. **尺寸策略 (w, h)**: 
   - **預設為 `hampel_only`**: 保留 Hampel 去除極端長寬（如面積爆增的偵測失誤），**不做 S-G 平滑**。這是避免 S-G 將實際目標長寬變化給平滑掉。

### 3.2 自適應網格補間 (Non-Uniform Frames)
若推論是跳幀執行（非連續），S-G 濾波會透過 Cubic Spline 重採樣至均勻網格，依密度放大 S-G Window，平滑後再採樣對回原時間點。

---

## 4. Segmentation 推論 (Stage 6)

### 4.1 ROI Fallback 機制（三層策略）
Segmentation 推論必須裁切出 ROI，策略優先度為：
1. 本幀 Detector BBox (需長寬 ≥ 2)
2. 上幀 BBox (透過 Segmentation Bootstrap 反求上一張有效 Mask bbox)
3. Full Frame (全幀裁切 fallback)

### 4.2 特殊的 Scheme 前處理行為
依據 Pipeline Scheme 設計：
- **Scheme A**: 裁切前對全幀做全域前處理（`roi_preprocs` 為空）。
- **Scheme B/C**: **全域前處理清空 (`seg_global_preprocs = []`)**，讓分割模型直接從**原始 RAW 影格**擷取 ROI，擷取後再進行 `roi_preprocs` 處理。

### 4.3 預測與後處理
- 取 ROI 送入 DNN (MedSAM 則提供中心與 Box Prompt 映射)。
- 輸出 Logits 經 Sigmoid 閾值過濾，執行 `keep_largest_component` 與 `fill_holes`，最終還原回全幀座標存放為 PNG。
- 寫入 `area_px`, `centroid`, `perimeter` 至 `FramePrediction.segmentation`。

---

## 5. Classification 推論 (Stage 7)

1. **特徵提取 (`feature_extractor`)**: 以平滑後的 Tracking Bbox、Segmentation Stats 抽取 `motion` / `static` 特徵，並裁切 `texture` 影像輸入 ConvNeXt/ResNet 等架構提取。
2. **向量化化**: 透過降維 `PCA/LDA` 以及標準化 `Z-score`。**（此階段 `fit=False`，保證僅套用 Training Phase Fit 得到的轉換矩陣參數）**。
3. **決策閾值**: 利用訓練集 LOO 預測後用 Youden's Index 計算出最佳 `decision_threshold` 設定二元分類 0 / 1，而非固定的 `0.5`。

---

## 6. Review 重點與潛在不一致 (⚠️ Review Points)

提供給 Code Reviewer 需特別確認的大架構與邏輯議題：

### 6.1 Trajectory Filter 的「非即時性」 (Non-Causal)
`trajectory_filter` 採用**雙向 Savitzky-Golay (forward + backward 取平均)**。這意味著系統推論時需要獲取「未來幀」才能平滑「當前幀」。
- **影響**: 這套 Pipeline **不支援真正的即時 Streaming**（需要 Buffer 延遲或整段影片錄製完再推論）。

### 6.2 訓練與推論的 BBox Smoothing 策略差異
- **訓練集 (GT Base)**: Classification 抽取 GT 特徵時，對於長寬 `w, h` 是採用 `skip_hampel=True`（不去除離群）但套用 S-G Smoothing (為保證 C2 一階與二階導數連續性)，稱為 `"independent"` mode。
- **推論集 (Pred Base)**: Trajectory filter 預設使用 `"hampel_only"`，即去除離群後**不使用 S-G 處理長寬**，以最大程度吻合 GT 長寬以取得較高的 IoU。
- **影響**: 訓練時模型看到的長寬是高度 S-G 全平滑的，而推論時看到的長寬則是只有過濾極端值的粗糙階梯狀。此差異可能會影響 Motion 特徵中的 $\Delta W, \Delta H$ 分佈。

### 6.3 追蹤丟失與 ROI Fallback 的累積放大
Seg 落後 fallback 使用了 "Segmentation Bootstrap (Strategy 2)" 機制，拿上一張正確的 seg mask bouding box 作為本張 ROI 來源。
- **好處**: 避免 Detector 連環出錯。
- **隱患**: Code 雖有設計 `last_bbox_raw` 來儲存沒有 expand 過的 bbox，但連續多張皆無 Detector 而只依賴 Tracker 盲追時，仍可能造成漂移。

### 6.4 Segmentation Scheme B/C 無全域前處理
這是符合預期的行為，因為 `runner.py` 中明確：
```python
if scheme in {"B", "C"}:
    seg_global_preprocs = []
```
若 Detector 需要看 CLAHE 等影像才能框出 BBox，Segmentation 會直接在「沒做過 CLAHE 的原圖 ROI」中進行推理。這保證了 Segmentation Model 的資料流是沒有被全域失真干擾的。此為設計上的巧思。

### 6.5 Detector 評價範圍（全集 vs GT 標記影格）
在 Trajectory Filter 計算 metrics 時：
`restrict_to_gt_frames = bool(_tf_eval_cfg.get("restrict_to_gt_frames", True))`
這代表推論與平滑雖然對**原測試影片的所有影格**進行，但計分基準強制拉回到僅算**有 GT 標記的幀數**，這保證了計分的公正對齊。
