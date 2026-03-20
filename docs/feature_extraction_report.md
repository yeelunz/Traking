# 特徵提取階段 (Feature Extraction) 架構與維度報告

> 本文件完整記錄測試與訓練階段的特徵提取器架構、計算方式、特徵維度，以及為 Code Review 準備的設計盲點與潛在不一致揭露。

---

## 1. 總覽：特徵提取器家族 (Feature Extractors)

特徵提取依據分類器類型，嚴格分為 **Non-TSC (表格型)** 與 **TSC (時間序列型)** 兩大派系。

| 提取器名稱 | 輸出形狀 / 維度 | 適用分類器 (Classifier) | 說明 |
|-----------|----------------|------------------------|------|
| `DELETED_motion_only` | 31 D | 表格型 (RF / XGB) | 基礎運動特徵 |
| `DELETED_motion_texture` | 78 D | 表格型 | 運動 + CSA + 傳統紋理(PCA) |
| `tab_v2` | 132 D | 表格型 | 運動 + 臨床靜態 + ResNet(PCA) |
| `tab_v2_extend`| 180 D | 表格型 | V1 增強版 (新增24維統計) |
| **`tab_v3_lite`** | **33 D** | 表格型 | ⭐ **V3-Lite 主力** (無 GPU) |
| **`tab_v3_pro`** | **33 D (預設)** | `fusion_mlp` (`v3pro_fusion` 相容別名) | ⭐ **V3-Pro 主力** (精簡多模態, 自動紋理降維分流) |
| `tsc_v2` | `(18, 256)` | TSC (PatchTST 等) | 幾何通道 + ResNet紋理(PCA) |
| `tsc_v2_extend` | `(24, 256)` | TSC | V1 增強版 (新增物理通道) |
| **`tsc_v3_lite`**| **`(12, 256)`** | TSC | ⭐ **V3-Lite-TSC 主力** (GLCM) |
| **`tsc_v3_pro`** | **`(12, 256)`** | TSC | ⭐ **V3-Pro-TSC 主力** (ConvNeXt) |

---

## 2. 核心架構解析：V3 世代 (Lite & Pro)

### 2.1 Non-TSC: `tab_v3_lite` (33 D)
純 CPU 提取器，無深度學習參與：
- **Motion (13 D)**: 速度、加速度、從軌跡中位數的偏移量、航向角變化 (捨棄了絕對空間座標)。
- **Static (10 D)**: 分割面積 (CSA)、等效直徑、圓度、長寬比的平均及標準差。
- **Texture (10 D)**: 從 5 個均勻採樣幀的 BBox 提取 GLCM (Contrast/Homogeneity/Energy/Correlation) 與灰階統計。

### 2.2 Non-TSC: `tab_v3_pro` (33 D, 預設)
`tab_v3_pro` 已改為與 Lite 同量級的緊湊向量表示，避免紋理維度淹沒幾何/運動訊號：
- **Motion (13 D)**: 同 Lite。
- **Static (10 D)**: 同 Lite。
- **Texture (`texture_dim`, 預設 10 D)**: 使用 texture backbone 取得紋理嵌入後，自動依 `texture_mode` 分流降維。

> **總維度**: `13 + 10 + texture_dim`，預設為 **33 D**。

`texture_mode` 自動分流策略如下：
- **`freeze`**: 走固定特徵路徑，於 `finalize_batch` 以 PCA 將原始 backbone 紋理嵌入壓至 `texture_dim`。
- **`learnable`**: 走可訓練投影路徑，直接輸出 `texture_dim` 緊湊向量。
- **`pretrain`**: 與 `learnable` 相同走投影路徑，但權重來自 Stage-1 預訓練；主流程可在缺少 ckpt 時自動觸發 Stage-1。

### 2.3 TSC: `tsc_v3_lite` (12 ch × 256 steps)
將不定長度、不均勻採樣的追蹤結果，統一對齊至 256 步時序通道：
- **Motion (ch 0~5)**: `speed`, `accel`, `dx_median`, `dy_median`, `heading_sin`, `heading_cos`。
- **Static (ch 6~8)**: `csa_norm`, `eq_diam_norm`, `aspect_ratio`。
- **Texture (ch 9~11)**: 每幀 GLCM `contrast`, `homogeneity`, `gray_mean`。
> **時序建構法**: 在原始離散幀上計算 -> Hampel 濾離群 -> Cubic Spline 插值到完整影片長度 -> S-G 雙向平滑 -> 均勻重採樣至 256 步。

### 2.4 TSC: `tsc_v3_pro` (12 ch × 256 steps)
沿用 `v3lite` 的前 9 個幾何/運動通道，但將紋理替換為 **ConvNeXt 時序特徵**：
- **Texture (ch 9~11)**: 對 256 步中的指定採樣點提取 ConvNeXt (32D) 特徵，再透過 PCA 壓縮至 **3 通道** `ts_tex_convnext_pca0~2`。

---

## 3. 降維機制 (Dimension Reduction)

為解決大維度帶來的維度災難，系統內建了不同層級的降維法：

### 3.1 PCA (Channel 壓縮)
於 `finalize_batch(fit=True)` 時建立。
所有的 PCA 與 Standard Scaler 狀態都會綁定訓練集，並在 `finalize_batch(fit=False)` 時嚴格只做 Transform。

### 3.2 針對 TSC 的 `dim_reduction` 子模組
在 `tracking/classification/dim_reduction.py` 中負責處理二維時序 `(N, C, T)`，遵循「通道軸與時間軸獨立壓縮」的嚴格原則：
- **Stage A (C 軸)**: `ChannelReducer` (PCA/LDA) 或 `LearnedChannelProjection` (針對 PatchTST)。
- **Stage B (T 軸)**: `TSAutoEncoderReducer` (Conv1d AE) 或 `LearnedProjection`。

### 3.3 Texture Backbone 全域自動分流規則（統一準則）
所有使用 texture backbone 的提取器（包含 `tab_v2` / `_v2`、`tsc_v2` / `_v2`、`tab_v3_pro`、`tsc_v3_pro`）都使用同一條降維分流規則：
- **`texture_mode=freeze`**：視為不可訓練紋理路徑，統一走 **PCA**（訓練集 fit、驗證/測試集 transform）。
- **`texture_mode=learnable` 或 `pretrain`**：視為可訓練紋理路徑，統一走 **projection** 路徑做維度校正。

此規則與 extractor 名稱無關，只要該 extractor 使用 texture backbone 即自動套用。

另外，GLCM 等手工紋理（例如 `tab_v3_lite`、`tsc_v3_lite`）也視為 **non-learnable texture**，同樣走 PCA 路徑。

---

## 4. 重點機制說明設計巧思

過去開發中曾針對以下幾個特殊情境進行討論，最終收斂出目前的架構：

### 4.1 BBox Aspect Ratio 刻意不平滑 (TSC 特徵)
在 `tsc_v3_lite` 及 `_v3pro` 中：
- **實作現狀**: 面積 (`csa`) 與等效直徑 (`eq_diam`) 皆在提取器內部執行了 S-G 平滑；然而長寬比 (`aspect_ratio`) 依賴上游傳入的 BBox $W / H$。
- **設計巧思**: 上游推論時 `trajectory_filter` 預設策略為 `"hampel_only"`（刻意不平滑 $W, H$）。**這是正確的行為設計**，因為對長寬進行 S-G 平滑在物理上會改變面積，導致追蹤框（BBox）變形膨脹而與真實影像特徵不符。

### 4.2 訓練與測試的 PCA/Scaler 狀態隔離
- **架構原則**: `finalize_batch(fit=True)` 負責建立 PCA `components_` 以及 Scaler `mean`, `std`。這保證了測試資料（Validation/Test）絕對不會污染全域統計矩陣，完美符合嚴格的實驗要求。Pipeline 中也有 SHA256 簽核機制守護，這部分運作穩定。

### 4.3 `CubicSpline` 插值的外推回彈保護
在建立 TSC 的 256 步矩陣時，若遇到長時間追蹤丟失造成的標記中斷（大段 NaN），常規的 `scipy.interpolate.CubicSpline` 會在真空區間進行多項式外推而導致數據暴衝。
- **防護實作**: 系統內已經完善了數值保護，實作透過 `np.clip(result, v_min, v_max)` 控制了外推的上下界極限，避免了神經網路收到暴雷數據。

### 4.4 Non-TSC 紋理取樣策略改為 Top-5 Confidence
在 `tab_v3_lite` 等 Non-TSC 特徵中提取紋理時：
- **實作現狀**: 過去採用時間軸上等距切分的 `_evenly_spaced_indices` 取 5 幀。為了更穩定地獲取具代表性的 ROI，最新實作**已全面更改為優先尋找 Detector 信心分數 (Score) 最強的 Top-5 幀** 進行灰階與 GLCM 計算。這保證了提取出的紋理能最大程度地貼合物件特徵，減少模糊段落的干擾。