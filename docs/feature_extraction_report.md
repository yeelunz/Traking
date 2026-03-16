# 特徵提取方法詳細報告（TSC vs Non-TSC）

> 本文件完整記錄所有 6 種特徵提取器的架構、計算方式、特徵維度、以及 TSC / Non-TSC 分類器各自的處理差異。

---

## 目錄

1. [總覽：特徵提取器家族](#1-總覽特徵提取器家族)
2. [共用基礎：運動特徵 (31 D)](#2-共用基礎運動特徵-31-d)
3. [Non-TSC 特徵提取器](#3-non-tsc-特徵提取器)
   - 3.1 motion_only (31 D)
   - 3.2 motion_texture (78 D)
   - 3.3 motion_texture_static (132 D)
   - 3.4 motion_texture_static_v2 (180 D)
   - 3.5 **motion_static_lite (33 D)** ⭐ V3-Lite
4. [TSC 特徵提取器](#4-tsc-特徵提取器)
   - 4.1 time_series (4,608 D)
   - 4.2 time_series_v2 (6,144 D)
   - 4.3 **time_series_v3lite (3,072 D)** ⭐ V3-Lite
5. [PCA 降維機制](#5-pca-降維機制)
6. [dim_reduction 降維模組](#6-dim_reduction-降維模組)
7. [ResNet-18 紋理管線](#7-resnet-18-紋理管線)
8. [時間序列建構管線](#8-時間序列建構管線)
9. [Subject 聚合](#9-subject-聚合)
10. [分類器與特徵提取器匹配表](#10-分類器與特徵提取器匹配表)
11. [完整特徵 Key 列表](#11-完整特徵-key-列表)
12. [V3-Lite 設計理念與對比](#12-v3-lite-設計理念與對比)

---

## 1. 總覽：特徵提取器家族

| 提取器名稱 | 維度 | 適用分類器 | 來源檔 |
|-----------|------|-----------|--------|
| `motion_only` | 31 D | 任意 tabular | `feature_extractors.py` |
| `motion_texture` | 78 D | 任意 tabular | `feature_extractors.py` |
| `motion_texture_static` | 132 D | RF / XGB / LGBM / TabPFN | `feature_extractors_ext.py` |
| `motion_texture_static_v2` | 180 D | RF / XGB / LGBM / TabPFN | `feature_extractors_ext.py` |
| **`motion_static_lite`** | **33 D** | RF / XGB / LGBM / TabPFN | `feature_extractors_v3lite.py` |
| `time_series` | 4,608 D | MultiRocket / PatchTST / TimeMachine | `feature_extractors_ext.py` |
| `time_series_v2` | 6,144 D | MultiRocket / PatchTST / TimeMachine | `feature_extractors_ext.py` |
| **`time_series_v3lite`** | **3,072 D** | MultiRocket / PatchTST / TimeMachine | `feature_extractors_v3lite.py` |

**核心設計哲學**：

- **Non-TSC（表格型）**：將整支影片壓縮成一個固定維度向量（統計量），適合傳統 ML 分類器
- **TSC（時間序列分類）**：保留每幀的多通道時序結構 `(C, T)`，由 TSC 分類器自行學習時序模式

---

## 2. 共用基礎：運動特徵 (31 D)

所有 6 種提取器都以 `_compute_motion_features()` 為基礎（Non-TSC 直接使用；TSC 則另建時序通道但運動特徵概念相通）。

### 2.1 輸入

```python
samples: Sequence[FramePrediction]
# 每個 FramePrediction 提供:
#   .frame_index   → 時間戳
#   .center        → (cx, cy) 質心座標（已由 trajectory_filter 平滑）
#   .bbox          → (x, y, w, h) 邊框
#   .segmentation  → SegmentationData（含 mask_stats）
```

**重要設計**：上游 trajectory_filter 已對 test predictions 執行 Hampel + S-G 平滑，feature extractor **不再二次平滑**，直接使用輸入的 center/bbox。

### 2.2 完整 31 維特徵定義

| # | Key | 公式 | 物理意義 |
|---|-----|------|---------|
| 0 | `num_points` | $N$ | 該影片的追蹤幀數 |
| 1 | `duration_frames` | $f_{last} - f_{first}$ | 追蹤持續時間（幀數） |
| 2 | `displacement_x` | $cx_N - cx_1$ | X 方向淨位移 |
| 3 | `displacement_y` | $cy_N - cy_1$ | Y 方向淨位移 |
| 4 | `net_displacement` | $\sqrt{\Delta x^2 + \Delta y^2}$ | 起止點直線距離 |
| 5 | `span_x` | $\max(cx) - \min(cx)$ | X 方向最大跨度 |
| 6 | `span_y` | $\max(cy) - \min(cy)$ | Y 方向最大跨度 |
| 7 | `path_length` | $\sum_{i=1}^{N-1}\sqrt{(cx_{i+1}-cx_i)^2+(cy_{i+1}-cy_i)^2}$ | 軌跡總路徑長 |
| 8 | `straightness_ratio` | $\frac{\text{net\_displacement}}{\text{path\_length}}$ | 軌跡直線度 (0~1) |
| 9 | `mean_speed` | $\bar{v} = \text{mean}\left(\frac{\|d_i\|}{\Delta t_i}\right)$ | 平均速度 |
| 10 | `max_speed` | $\max(v_i)$ | 最大速度 |
| 11 | `std_speed` | $\sigma(v_i)$ | 速度標準差 |
| 12 | `median_speed` | $\text{median}(v_i)$ | 速度中位數 |
| 13 | `p95_speed` | $P_{95}(v_i)$ | 速度 95 百分位 |
| 14 | `p5_speed` | $P_5(v_i)$ | 速度 5 百分位 |
| 15 | `mean_acc` | $\bar{a} = \text{mean}(\|\Delta v_i\|)$ | 平均加速度 |
| 16 | `max_acc` | $\max(\|\Delta v_i\|)$ | 最大加速度 |
| 17 | `std_acc` | $\sigma(\|\Delta v_i\|)$ | 加速度標準差 |
| 18 | `mean_jerk` | $\bar{j} = \text{mean}(\|\Delta a_i\|)$ | 平均急動度 |
| 19 | `max_jerk` | $\max(\|\Delta a_i\|)$ | 最大急動度 |
| 20 | `std_jerk` | $\sigma(\|\Delta a_i\|)$ | 急動度標準差 |
| 21 | `curvature_mean` | $\bar{\kappa} = \text{mean}\left(\frac{\|v_i \times v_{i+1}\|}{\|v_i\|^3}\right)$ | 平均曲率 |
| 22 | `curvature_std` | $\sigma(\kappa_i)$ | 曲率標準差 |
| 23 | `angular_change_mean` | $\text{mean}(\|\Delta\theta_i\|)$ | 平均角度變化 |
| 24 | `angular_change_std` | $\sigma(\|\Delta\theta_i\|)$ | 角度變化標準差 |
| 25 | `area_mean` | $\bar{A} = \text{mean}(w_i \times h_i)$ | 平均 bbox 面積 |
| 26 | `area_std` | $\sigma(A_i)$ | 面積標準差 |
| 27 | `area_range` | $\max(A) - \min(A)$ | 面積全距 |
| 28 | `area_change_mean` | $\text{mean}(\|\Delta A_i\|)$ | 平均面積變化 |
| 29 | `area_change_std` | $\sigma(\|\Delta A_i\|)$ | 面積變化標準差 |

> 速度定義：$v_i = \frac{\|center_{i+1} - center_i\|}{frame_{i+1} - frame_i}$
> 曲率最小速度門檻：$\|v_i\| > 1.0$ px/frame 才計算，否則視為 0

---

## 3. Non-TSC 特徵提取器

### 3.1 `motion_only` (31 D)

**最簡單的提取器**，直接輸出 `_compute_motion_features()` 的 31 維向量。

```
extract_video(samples) → motion features (31 D)
finalize_batch(fit=True)  → 以訓練集做 global Z-score
finalize_batch(fit=False) → 套用訓練集統計（不重新 fit）
aggregate_subject()    → 每個 key 取 [mean, std, min, max] → 124 D
```

| 環節 | 輸出維度 |
|------|---------|
| Video level | 31 D |
| Subject level | 31 × 4 = 124 D |

### 3.2 `motion_texture` (78 D)

**運動 + CSA 靜態 + 傳統紋理（PCA 壓縮）**

```
extract_video(samples, video_path):
    motion = _compute_motion_features(samples)         → 31 D
    csa    = _compute_csa_features(samples)             → 8 D
    texture = _extract_first_last_texture(samples, vp)  → 42 D raw
                                                          (21 per frame × 2 frames)
    → 合併 motion + csa + texture placeholder

finalize_batch(fit=True):   PCA(42 → 39 D)
                           + global Z-score
finalize_batch(fit=False):  PCA transform only
                           + global Z-score transform

→ 31 + 8 + 39 = 78 D
```

#### CSA 靜態特徵 (8 D)

| # | Key | 說明 |
|---|-----|------|
| 0 | `csa_first_area` | 首幀分割 mask 面積（px²） |
| 1 | `csa_last_area` | 末幀分割 mask 面積 |
| 2 | `csa_first_perimeter` | 首幀 mask 周長 |
| 3 | `csa_last_perimeter` | 末幀 mask 周長 |
| 4 | `csa_first_eq_diameter` | 首幀等效直徑 |
| 5 | `csa_last_eq_diameter` | 末幀等效直徑 |
| 6 | `csa_first_circularity` | 首幀圓度 $\frac{4\pi A}{P^2}$ |
| 7 | `csa_last_circularity` | 末幀圓度 |

#### 傳統紋理特徵 (21 D × 2 幀 = 42 D raw → PCA → 39 D)

對首幀和末幀各計算以下 21 維特徵：

| # | Key | 計算方式 |
|---|-----|---------|
| 0 | `tex_mean` | 灰階平均值 |
| 1 | `tex_std` | 灰階標準差 |
| 2 | `tex_skewness` | 灰階偏度 |
| 3 | `tex_kurtosis` | 灰階峰度 |
| 4 | `tex_grad_mean` | Sobel 梯度幅值平均 |
| 5 | `tex_grad_std` | Sobel 梯度幅值標準差 |
| 6 | `tex_glcm_contrast` | GLCM 對比度（16 級量化，水平方向） |
| 7 | `tex_glcm_dissimilarity` | GLCM 不相似性 |
| 8 | `tex_glcm_homogeneity` | GLCM 同質性 |
| 9 | `tex_glcm_energy` | GLCM 能量 |
| 10 | `tex_glcm_correlation` | GLCM 相關性 |
| 11–20 | `tex_lbp_00`~`tex_lbp_09` | LBP 直方圖 10 bin（8 鄰居 circular LBP） |

> PCA 降維目標：$\min(31 + 8, 42) = 39$ D（等於 motion + csa 維度之和）

### 3.3 `motion_texture_static` (132 D)

**運動 + CTS 臨床靜態 + ResNet-18 紋理（PCA 壓縮）**

```
extract_video(samples, video_path):
    motion = _compute_motion_features(samples)          → 31 D
    static = _compute_cts_static_features(samples, vp)  → 35 D
    resnet = ResNet18(first_frame, last_frame)           → 1024 D raw (2×512)

finalize_batch(fit=True):   PCA(1024 → 66 D)
                           + global Z-score
finalize_batch(fit=False):  PCA transform only
                           + global Z-score transform

→ 31 + 35 + 66 = 132 D
```

#### CTS 靜態特徵 (35 D)

| 類別 | Keys | 維度 | 說明 |
|------|------|------|------|
| **CSA 統計** | `cts_csa_min/max/mean/std/range/cv` | 6 | 橫截面積序列統計 |
| **Swelling** | `cts_csa_first/last`, `cts_swelling_ratio` | 3 | 腫脹率 = max/min |
| **Flattening** | `cts_flat_mean/std/max/min` | 4 | 扁平指數 = w/h |
| **Circularity** | `cts_circularity_first/last/mean/std` | 4 | 分割 mask 圓度 |
| **Eq Diameter** | `cts_eq_diam_mean/std/first/last` | 4 | 等效直徑 |
| **Compactness** | `cts_compact_mean/std` | 2 | 緊湊度（= 圓度公式） |
| **Echo** | `cts_echo_mean_first/std_first/mean_last/std_last/delta` | 5 | 灰階回音強度 |
| **Displacement** | `cts_nerve_net_disp/path_len/stability_x/stability_y` | 4 | 神經移位量 |
| **Aspect** | `cts_aspect_first/last` | 2 | 長寬比 |
| | | **35** | |

#### ResNet-18 紋理 (66 D)

- 首幀 + 末幀 → 各自從 bbox ROI 擷取 → **加入 segmentation mask**（mask 外像素歸零）
- 送入截斷的 ResNet-18（ImageNet pretrained，去掉 FC 層）→ GAP → 512 D
- 兩幀串接 → 2 × 512 = 1,024 D
- PCA 降維 → 66 D（= 31 motion + 35 static）

### 3.4 `motion_texture_static_v2` (180 D)

**V1 的增強版，新增 24 個臨床統計特徵 + 更多 PCA 紋理維度**

```
extract_video(samples, video_path):
    motion = _compute_motion_features(samples)             → 31 D
    static = _compute_cts_static_features_v2(samples, vp)  → 59 D (35 V1 + 24 V2)
    resnet = ResNet18(first_frame, last_frame)              → 1024 D raw

finalize_batch(fit=True):   PCA(1024 → 90 D)
                           + global Z-score
finalize_batch(fit=False):  PCA transform only
                           + global Z-score transform

→ 31 + 59 + 90 = 180 D
```

#### V2 新增的 24 維 CTS 靜態特徵

| 類別 | Keys | 維度 | 說明 |
|------|------|------|------|
| **CSA 百分位** | `cts_csa_q25/q75/iqr/skew/kurt` | 5 | 面積分佈形狀 |
| **Circ 百分位** | `cts_circ_q25/q75/iqr` | 3 | 圓度分佈 |
| **EqDiam 百分位** | `cts_eq_diam_q25/q75/iqr` | 3 | 等效直徑分佈 |
| **Flat 百分位** | `cts_flat_q25/q75/iqr` | 3 | 扁平指數分佈 |
| **時序梯度** | `cts_csa_grad_mean/std`, `cts_flat_grad_mean/std` | 4 | 一階差分統計 |
| **位移統計** | `cts_disp_median_mean/max`, `cts_temporal_energy`, `cts_radial_range` | 4 | 偏離中位數的分析 |
| **方向效率** | `cts_mean_angular_change`, `cts_path_efficiency` | 2 | 軌跡行為描述 |
| | | **24** | |

#### PCA 紋理維度

PCA 目標 = $31 + 59 = 90$ D（motion + static 維度之和）

### 3.5 `motion_static_lite` (33 D) ⭐ V3-Lite

**輕量非 TSC 提取器 — 無深度學習、無 PCA、無 GPU**

```
extract_video(samples, video_path):
    motion  = _compute_motion_lite(samples)              → 13 D
    static  = _compute_static_lite(samples)              → 10 D
    texture = _compute_texture_lite(samples, video_path)  → 10 D

finalize_batch(fit=True)  → global Z-score（訓練集）
finalize_batch(fit=False) → global Z-score（測試集 transform only）

→ 13 + 10 + 10 = 33 D
```

#### 運動特徵 (13 D) — 無絕對位置

| # | Key | 類別 | 說明 |
|---|-----|------|------|
| 0 | `num_points` | 基本 | 追蹤幀數 |
| 1 | `duration_frames` | 基本 | 持續時間 |
| 2 | `path_length` | 軌跡 | 總路徑長度（位置不變量） |
| 3 | `straightness_ratio` | 軌跡 | 直線度比 |
| 4 | `mean_speed` | 速度 | 平均速度 |
| 5 | `std_speed` | 速度 | 速度標準差 |
| 6 | `median_speed` | 速度 | 速度中位數 |
| 7 | `mean_acc` | 加速度 | 平均加速度 |
| 8 | `std_acc` | 加速度 | 加速度標準差 |
| 9 | `disp_median_mean` | 相對中位數 | 平均偏離中位數距離 |
| 10 | `disp_median_std` | 相對中位數 | 偏離距離標準差 |
| 11 | `disp_median_max` | 相對中位數 | 最大偏離距離 |
| 12 | `mean_heading_change` | 方向 | 平均航向變化角 |

#### 靜態特徵 (10 D) — CSA / 直徑 / 長寬比

| # | Key | 說明 |
|---|-----|------|
| 0 | `csa_mean` | CSA 平均（分割 mask 面積） |
| 1 | `csa_std` | CSA 標準差 |
| 2 | `csa_strain_rate` | CSA 應變率 $(\max A - \min A) / A_0$ |
| 3 | `swelling_ratio` | 腫脹率 = max/min |
| 4 | `eq_diam_mean` | 等效直徑平均 |
| 5 | `eq_diam_strain_rate` | 直徑應變率 $(\max D - \min D) / D_0$ |
| 6 | `circularity_mean` | 圓度平均 |
| 7 | `circularity_std` | 圓度標準差 |
| 8 | `aspect_ratio_mean` | 長寬比平均 (w/h) |
| 9 | `aspect_ratio_std` | 長寬比標準差 |

#### 紋理特徵 (10 D) — GLCM 基礎，無 ResNet

從影片中均勻取樣 5 幀（可配置），每幀計算 GLCM + 灰階統計，再取跨幀的 mean/std 聚合：

| # | Key | 說明 |
|---|-----|------|
| 0 | `tex_glcm_contrast_mean` | GLCM 對比度跨幀平均 |
| 1 | `tex_glcm_contrast_std` | GLCM 對比度跨幀標準差 |
| 2 | `tex_glcm_homogeneity_mean` | GLCM 同質性跨幀平均 |
| 3 | `tex_glcm_homogeneity_std` | GLCM 同質性跨幀標準差 |
| 4 | `tex_glcm_energy_mean` | GLCM 能量跨幀平均 |
| 5 | `tex_glcm_correlation_mean` | GLCM 相關性跨幀平均 |
| 6 | `tex_gray_mean` | 灰階平均值跨幀平均 |
| 7 | `tex_gray_std` | 灰階標準差跨幀平均 |
| 8 | `tex_grad_mean` | Sobel 梯度幅值跨幀平均 |
| 9 | `tex_grad_std` | Sobel 梯度幅值跨幀標準差 |

> GLCM 計算方式：16 級灰階量化，水平方向共現矩陣，不依賴 skimage

---

## 4. TSC 特徵提取器

### 4.1 `time_series` (4,608 D = 18 ch × 256 steps)

**保留時序結構的多通道特徵**

```
extract_video(samples, video_path):
    geo, resample_idx = _extract_ts_geo_channels(samples)      → (10, 256)
    raw_resnet = ResNet18(256 frames along trajectory)          → (256, 512)
    → 存入 _raw_tex_ts metadata

finalize_batch(fit=True):
    PCA(512 → 8 D per timestep)
    → channels 10-17 filled
    → 再做 channel-wise global Z-score（18 通道）

finalize_batch(fit=False):
    PCA transform only
    → 套用訓練集 channel-wise Z-score

→ (18, 256) → flatten → 4,608 D
```

#### Geometric Channels (10 D per timestep)

| Ch | Key | 定義 | 提取層輸出 |
|----|-----|------|-----------|
| 0 | `ts_cx` | 質心 X 座標 | 原始 `px` |
| 1 | `ts_cy` | 質心 Y 座標 | 原始 `px` |
| 2 | `ts_bw` | BBox 寬度 | 原始 `px` |
| 3 | `ts_bh` | BBox 高度 | 原始 `px` |
| 4 | `ts_area` | BBox 面積 | 原始 `px^2` |
| 5 | `ts_seg_area` | 分割 mask 面積 | 原始 `px^2` |
| 6 | `ts_circularity` | $4\pi A / P^2$ | 固有 [0,1] |
| 7 | `ts_eq_diam` | 等效直徑 | 原始 `px` |
| 8 | `ts_speed` | 幀間位移速度 | 原始 `px/frame`，**從插值後質心重新計算** |
| 9 | `ts_flat` | 扁平指數 w/h | 無歸一化 |

#### Texture Channels (8 D per timestep)

| Ch | Key | 來源 |
|----|-----|------|
| 10–17 | `ts_tex_pca_0` ~ `ts_tex_pca_7` | ResNet-18 512D → PCA 8D |

#### 時序建構管線（詳見 §7）

```
稀疏標註幀 → 放到真實時間軸 → Hampel 去離群(僅 seg 通道)
→ cubic spline 插值到完整 T 幀 → S-G 平滑(僅 seg 通道)
→ 重新計算 speed → 均勻重採樣到 256 步 → (10, 256)
```

### 4.2 `time_series_v2` (6,144 D = 24 ch × 256 steps)

**V1 的增強版，新增 6 個物理/運動學通道**

```
extract_video(samples, video_path):
    geo_v2, resample_idx = _extract_ts_geo_channels_v2(samples)  → (16, 256)
    raw_resnet = ResNet18(256 frames)                             → (256, 512)
    → 存入 _raw_tex_ts metadata

finalize_batch(fit=True):   PCA(512 → 8 D per timestep) → ch 16-23
                           + channel-wise global Z-score（24 通道）
finalize_batch(fit=False):  PCA transform only
                           + channel-wise global Z-score transform

→ (24, 256) → flatten → 6,144 D
```

#### V2 Geometric Channels (16 D per timestep)

| Ch | Key | V1→V2 變化 | 說明 |
|----|-----|-----------|------|
| 0 | `ts_dx` | cx → **偏離中位數** | $cx - \text{median}(cx)$（原始 `px`） |
| 1 | `ts_dy` | cy → **偏離中位數** | $cy - \text{median}(cy)$（原始 `px`） |
| 2 | `ts_bw` | 不變 | 同 V1 |
| 3 | `ts_bh` | 不變 | 同 V1 |
| 4 | `ts_area` | 不變 | 同 V1 |
| 5 | `ts_seg_area` | 不變 | 同 V1 |
| 6 | `ts_circularity` | 不變 | 同 V1 |
| 7 | `ts_eq_diam` | 不變 | 同 V1 |
| 8 | `ts_speed` | 不變 | 同 V1 |
| 9 | `ts_flat` | 不變 | 同 V1 |
| 10 | `ts_accel` | **新增** | $\|d(\text{speed})/dt\|$（原始 `px/frame^2`） |
| 11 | `ts_heading_sin` | **新增** | $\sin(\theta)$，$\theta = \arctan2(\Delta cy, \Delta cx)$ |
| 12 | `ts_heading_cos` | **新增** | $\cos(\theta)$ |
| 13 | `ts_d_area` | **新增** | $d(\text{area})/dt$（原始 `px^2/frame`） |
| 14 | `ts_radial_dist` | **新增** | $\sqrt{dx^2 + dy^2}$ 離中位數距離 |
| 15 | `ts_curvature` | **新增** | 軌跡曲率（叉積近似） |

#### Texture Channels (8 D per timestep)

| Ch | Key |
|----|-----|
| 16–23 | `ts_tex_pca_0` ~ `ts_tex_pca_7` |

#### V2 通道設計理由

| V2 改進 | 理由 |
|---------|------|
| 位移偏離中位數 (dx, dy) | 移除絕對位置偏差，保留動態行為 |
| 加速度通道 | 捕捉運動狀態的突然變化 |
| 航向角 sin/cos | 描述運動方向的連續旋轉，避免角度跳躍 |
| 面積變化率 | 反映形狀的時序動態 |
| 徑向距離 | 量化偏離靜息位置的程度 |
| 曲率通道 | 量化軌跡彎曲程度 |

### 4.3 `time_series_v3lite` (3,072 D = 12 ch × 256 steps) ⭐ V3-Lite

**輕量 TSC 提取器 — 無 ResNet、無 PCA、無 GPU（GLCM 紋理直接內建）**

```
extract_video(samples, video_path):
    ts = _extract_ts_channels_lite(samples, video_path)  → (12, 256)

finalize_batch(fit=True):
    在訓練集所有影片上計算每個通道的 global mean/std

finalize_batch(fit=False):
    套用訓練集通道統計做 Z-score（不重新 fit）

→ (12, 256) → flatten → 3,072 D
```

#### Channel Layout (12 通道)

| Ch | Key | 類別 | 說明 | 提取層輸出 |
|----|-----|------|------|--------|
| 0 | `ts_speed` | 運動 | 幀間速度 | 原始 `px/frame` |
| 1 | `ts_accel` | 運動 | 加速度 $\|d(v)/dt\|$ | 原始 `px/frame^2` |
| 2 | `ts_dx_median` | 運動 | X 偏離中位數 | 原始 `px` |
| 3 | `ts_dy_median` | 運動 | Y 偏離中位數 | 原始 `px` |
| 4 | `ts_heading_sin` | 運動 | $\sin(\theta)$ | [-1, 1] |
| 5 | `ts_heading_cos` | 運動 | $\cos(\theta)$ | [-1, 1] |
| 6 | `ts_csa_norm` | 靜態 | 分割 CSA（key 名保留） | 原始 `px^2` |
| 7 | `ts_eq_diam_norm` | 靜態 | 等效直徑（key 名保留） | 原始 `px` |
| 8 | `ts_aspect_ratio` | 靜態 | 長寬比 w/h | 無 |
| 9 | `ts_glcm_contrast` | 紋理 | GLCM 對比度 | 原始值 |
| 10 | `ts_glcm_homogeneity` | 紋理 | GLCM 同質性 | 固有 [0,1] |
| 11 | `ts_gray_mean` | 紋理 | 灰階平均值 | 原始值 |

> 註：提取層保留原始量綱；真正的尺度對齊在 `finalize_batch` 以 **通道級 global Z-score** 完成。

#### 紋理通道建構

```
步驟 1:  在已知稀疏幀位置讀取 video bbox patch (64×64 灰階)
步驟 2:  每幀計算 GLCM contrast + homogeneity + gray mean → 3 值
步驟 3:  Hampel 去離群 → cubic spline 插值至完整 T 幀
步驟 4:  Bidirectional S-G 平滑（不做局部 max 正規化）
步驟 5:  均勻重採樣至 256 步（預設；可透過 params.n_steps 調整）
```

> 紋理通道與幾何/運動通道共享同一個插值+重採樣管線，
> 額外 I/O 僅在每個 video 開一次 `cv2.VideoCapture` 遍歷已知幀。

#### 防止雙重平滑

| 通道 | 上游 trajectory_filter 已平滑？ | v3lite 內 Hampel | v3lite 內 S-G |
|------|-------------------------------|------------------|---------------|
| speed, accel (ch 0-1) | — 從 cx/cy 推導 | — | — |
| dx_median, dy_median (ch 2-3) | — 從 cx/cy 推導 | — | — |
| heading (ch 4-5) | — 從 cx/cy 推導 | — | — |
| csa_norm (ch 6) | ❌ | ✅ | ✅ |
| eq_diam_norm (ch 7) | ❌ | ✅ | ✅ |
| aspect_ratio (ch 8) | ✅ (上游 w/h 已平滑) | ❌ | ❌ |
| glcm_contrast (ch 9) | ❌ | ✅ | ✅ |
| glcm_homogeneity (ch 10) | ❌ | ✅ | ✅ |
| gray_mean (ch 11) | ❌ | ✅ | ✅ |

---

## 5. PCA 降維機制

> 補充：目前所有提取器在 `finalize_batch` 階段都會執行「訓練集擬合、測試集僅 transform」的全域標準化；
> 有 PCA 的提取器先做 PCA，再做 global Z-score。

### 5.1 訓練 vs 測試

```
Training:
    finalize_batch(features_list, fit=True)
    → SVD fit: X_centered = X - mean
    → U, S, Vt = svd(X_centered)
    → components = Vt[:target_dim]
    → reduced = X_centered @ components.T
    → 儲存 (mean, components) 到 self._pca_mean, self._pca_components

Testing:
    finalize_batch(features_list, fit=False)
    → reduced = (X - saved_mean) @ saved_components.T
    ⚠️ 若 PCA state 不存在 → RuntimeError（不會靜默重 fit）
```

### 5.2 各提取器的 PCA 目標維度

| 提取器 | Raw 維度 | PCA 目標 | 目標計算方式 |
|--------|---------|---------|-------------|
| `motion_texture` | 42 (21×2) | 39 | motion(31) + csa(8) |
| `motion_texture_static` | 1,024 (512×2) | 66 | motion(31) + static(35) |
| `motion_texture_static_v2` | 1,024 (512×2) | 90 | motion(31) + static(59) |
| `time_series` | 512 per step | 8 per step | 固定 `N_TEX_PCA_TS=8` |
| `time_series_v2` | 512 per step | 8 per step | 固定 `N_TEX_PCA_TS_V2=8` |
| `motion_static_lite` | — | — | **無 PCA**（GLCM 直接輸出 10 D） |
| `time_series_v3lite` | — | — | **無 PCA**（GLCM 3 ch 直接內嵌） |

### 5.3 PCA State 序列化

`motion_texture_static` / `_v2` / `time_series` / `_v2` 提供：
- `get_pca_state() → Dict[str, np.ndarray]`（包含 mean + components）
- `set_pca_state(state)` → 用於 global pre-fit 等場景

---

## 6. dim_reduction 降維模組

> 來源: `tracking/classification/dim_reduction.py`

### 6.0 核心設計哲學：二維獨立壓縮

TSC 特徵的形狀是 **(N, C, T)**（樣本 × 通道 × 時間步），
**絕對不能** 先 flatten 成 `(N, C×T)` 再做聯合 PCA：
這樣做會混淆特徵軸與時間軸，破壞時序結構（PCA 主成分會同時跨越通道和時間維度）。

正確做法是 **分兩個獨立階段** 壓縮：

```
Input (N, C, T)
    ↓
Stage A ── Channel reduction (C → C_target)
           每個時間步獨立看: 把 (N·T, C) 的樣本做 PCA 或 Linear 投影
           → 移除通道間共線性，保留時序結構完整
    ↓
(N, C_target, T)
    ↓
Stage B ── Temporal reduction (T → T_target)
           每個通道獨立看: 把 (N·C_target, T) 做 Linear 投影或 AE 壓縮
           → 時序降採樣，保留通道軸完整
    ↓
Output (N, C_target, T_target)
```

兩個階段的數學意義完全不同：
- **Stage A (C-axis)**：學習通道間線性投影，類似 per-timestep PCA
- **Stage B (T-axis)**：學習時序降採樣，類似 per-channel 線性插值

### 6.1 策略總覽

| 策略 | 軸 | 適用分類器 | 核心思路 | YAML 設定鍵 |
|------|---|---------|---------|------------|
| **UMAP** | 整體 | Tabular（RF, SVM, LightGBM）| 非線性流形，保持鄰域拓撲 | `dim_reduction: umap` |
| **LDA** | 整體 | Tabular | 監督式線性投影，最大化類間/類內變異比 | `dim_reduction: lda` |
| **ChannelReducer (PCA/LDA/UMAP)** | C-axis | MultiRocket | 在 (N·T, C) 上做監督/非監督通道投影 | `channel_reduction_target: 8` + `channel_reduction_method: pca|lda|umap` |
| **Autoencoder** | T-axis | MultiRocket | Conv1d channel-independent AE，壓縮時間軸 | `dim_reduction: autoencoder` |
| **LearnedChannelProjection** | C-axis | PatchTST / TimeMachine | `nn.Linear(C → C_target)`，per-timestep end-to-end | `channel_reduction_target: 8` |
| **LearnedProjection** | T-axis | PatchTST / TimeMachine | `nn.Linear(T → T_target)`，per-channel end-to-end | `dim_reduction: learned_projection` |

> **Tabular 分類器** 接收展平向量，無需考慮二維結構，UMAP/LDA 即可。

### 6.2 Tabular 降維：UMAP / LDA

用於 Non-TSC 分類器（RF、DT、SVM、LightGBM）：

```yaml
classifier:
  name: random_forest
  params:
    dim_reduction: umap      # or "lda"
    dim_reduction_params:
      n_components: 16       # UMAP default=16, LDA default=min(n_classes-1, D)
```

**流程**:
```
fit:     X_train → reducer.fit_transform(X_train, y) → X_reduced → model.fit
predict: X_test  → reducer.transform(X_test) → X_reduced → model.predict
```

- **UMAPReducer**: 內部使用 `umap-learn`，metric="euclidean"，min_dist=0.1
- **LDAReducer**: 內部使用 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`(svd solver)

### 6.3 TSC 降維（MultiRocket）：ChannelReducer + Autoencoder

MultiRocket 使用固定隨機 kernels，**無法反向傳播**，因此用預訓練方法做兩軸獨立壓縮：

```yaml
classifier:
  name: multirocket
  params:
    n_vars: 18                      # C
    n_steps: 256                    # T
    channel_reduction_target: 8     # Stage A: C 16 → 8（可選）
    channel_reduction_method: pca   # pca | lda | umap
    dim_reduction: autoencoder      # Stage B: T 256 → 128（可選）
    dim_reduction_target: 128
```

**Stage A — ChannelReducer（C-axis）**:
```
fit_transform:
  X (N, C, T) → transpose → (N, T, C) → reshape → (N·T, C)
    reducer.fit_transform → (N·T, C_target)
  reshape → (N, T, C_target) → transpose → (N, C_target, T)
```
每個時間步被視為「C 維特徵空間」中的一個獨立觀測。

可選方法：
- `pca` → `ChannelPCAReducer`（非監督）
- `lda` → `ChannelLDAReducer`（監督式，fit 時需要 `y`）
- `umap` → `ChannelUMAPReducer`（非線性流形）

**Stage B — TSAutoEncoderReducer**（T-axis，Conv1d AE）:
```
Encoder:
  Conv1d(1, hidden, k=7) → ReLU
  Conv1d(hidden, 1, k=5) → ReLU
  AdaptiveAvgPool1d(target_len)

Decoder:
  Upsample(size=seq_len)
  Conv1d(1, hidden, k=5) → ReLU
  Conv1d(hidden, 1, k=7)
```
- 每個 channel 共享同一組權重（channel-independent）
- 訓練: MSE reconstruction loss, Adam, 50 epochs
- fit 接收 (N, C, T) → reshape (N·C, 1, T) → 訓練 AE

### 6.4 TSC 降維（PatchTST / TimeMachine）：LearnedChannelProjection + LearnedProjection

可微分模型兩軸均用 `nn.Linear` **在模型內部** end-to-end 聯合訓練：

```yaml
classifier:
  name: patchtst    # 或 timemachine
  params:
    n_vars: 18
    n_steps: 256
    channel_reduction_target: 8     # Stage A: C 18 → 8（可選）
    dim_reduction: learned_projection
    dim_reduction_target: 128       # Stage B: T 256 → 128（可選）
```

**模型內部執行順序**（以 PatchTST 為例）:
```
Input X (N, C, T)
  ── RevIN normalize (on original C channels)
  ── Stage A: LearnedChannelProjection (optional)
       Linear(C, C_target, bias=False) applied per-timestep:
       (N, C, T) → permute → (N, T, C) → Linear → (N, T, C_target) → permute → (N, C_target, T)
  ── Stage B: LearnedProjection (optional)
       Linear(T, T_target, bias=False) applied per-channel:
       (N, C_target, T) → Linear → (N, C_target, T_target)
  ── PatchTST patching / TimeMachine Mamba blocks (on C_target × T_target)
  ── Classification head
```

- 兩層均 Xavier 初始化
- 梯度直接從分類 loss 回傳
- `RevIN` 始終作用於原始 C 通道（channel proj 之前），確保歸一化統計量正確

### 6.5 Save / Load 整合

所有降維器狀態均透過各自分類器的 `save()` / `load()` 自動序列化：

| 分類器類型 | C-axis 降維 | T-axis 降維 | 序列化方式 |
|-----------|------------|------------|-----------|
| Tabular (RF, SVM, etc.) | — | UMAPReducer / LDAReducer | `get_state()` → pickle dict |
| MultiRocket | ChannelPCAReducer / ChannelLDAReducer / ChannelUMAPReducer | TSAutoEncoderReducer | `get_state()` → pickle dict |
| PatchTST / TimeMachine | LearnedChannelProjection | LearnedProjection | 內建於 model `state_dict()` |

### 6.6 向後相容

- `channel_reduction_target` 和 `dim_reduction` 預設均為 `None` → **現有 YAML 完全不受影響**
- `channel_reduction_method` 預設為 `pca`
- 兩個軸各自獨立，可只開其中一個，也可同時開啟
- `channel_reduction_method=lda` 需要在 fit 階段提供標籤 `y`
- Tabular classifiers 的 `load()` 自動偵測新/舊 pickle 格式
- TSC classifiers 的 `n_vars`/`n_steps` 預設 `None` 時觸發自動偵測

---

## 7. ResNet-18 紋理管線

### 7.1 模型結構

```
MaskedROIResNetExtractor:
    backbone = torchvision.models.resnet18(pretrained=True)
    截斷: 去掉 fc + avgpool → 使用 layer4 輸出
    加上 AdaptiveAvgPool2d(1) → squeeze → 512 D
```

### 7.2 Non-TSC 路徑（首+末幀）

```
步驟 1:  選取 samples[0] 和 samples[-1]

步驟 2:  VideoCapture seek 到指定幀

步驟 3:  裁切 bbox ROI

步驟 4:  若有 segmentation mask → mask 外像素歸零
         (保留目標區域紋理，消除背景干擾)

步驟 5:  resize → (224, 224)

步驟 6:  ImageNet normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

步驟 7:  forward → 512 D

步驟 8:  兩幀串接 → 1,024 D raw vector
```

### 7.3 TSC 路徑（256 個時間步）

```
步驟 1:  利用 resample_frame_idx（256 個幀索引）

步驟 2:  為每個時間步找到最近的標註幀 → 使用其 bbox/mask

步驟 3:  batch 式 VideoCapture → 256 幀 ROI

步驟 4:  各自 mask + resize → ResNet → 512 D

步驟 5:  → (256, 512) 矩陣

步驟 6:  finalize_batch 中跨所有 training video 做 PCA:
         將所有影片的 (256, 512) 合併 → (N×256, 512)
         PCA fit → (N×256, 8)
         reshape 回 (N, 256, 8) → 填入 channels 10-17（或 16-23）
```

---

## 8. 時間序列建構管線

### 8.1 完整流程

```
稀疏輸入:
    samples[0].frame_index = 10   (第 10 幀有標註)
    samples[1].frame_index = 25   (第 25 幀有標註)
    ...
    samples[K].frame_index = 300  (第 300 幀有標註)

步驟 1:  建立全長時間軸 T = max(frame_index) + 1

步驟 2:  按 frame_index 排序 samples

步驟 3:  歸一化常數計算（從已知幀):
         max_w, max_h, max_area, max_seg_area, max_eq_diam

步驟 4:  建構稀疏通道 (K, 10):
         每個已知幀填入 cx/cy/bw/bh/area/seg_area/circ/eq_diam/flat
         channel 8 (speed) 跳過

步驟 5:  Pre-interpolation 去離群:
         僅對 seg 衍生通道 (ch 6, 7) 執行 multiscale Hampel
         ⚠️ 不對 ch 0-4, 9 做 Hampel（已由上游 trajectory_filter 平滑）

步驟 6:  Cubic spline 插值到完整 T 幀:
         每通道獨立: CubicSpline(t_known, v_known) → t_all
         邊界外 clamp 到已知值範圍（防止 cubic 發散）
         fallback: numpy linear interpolation

步驟 7:  Post-interpolation 平滑:
         僅對 seg 衍生通道 (ch 5, 6, 7) 執行 bidirectional S-G
         Window=7, polyorder=2

步驟 8:  重新計算 speed 通道:
         從插值後的 cx, cy 反算幀間距離
         → 歸一化到 [0, 1]

步驟 9:  NaN/Inf 清理:
         np.nan_to_num(timeline, nan=0, posinf=0, neginf=0)

步驟 10: 均勻重採樣:
         np.linspace(0, T-1, 256) → 256 個幀索引
         → 從 T-step timeline 取子集 → (10, 256)
```

### 8.2 V2 額外步驟

在步驟 8 之後、步驟 9 之前：

```
步驟 8a: 位移偏離中位數:
         dx = cx_norm - median(cx_norm)
         dy = cy_norm - median(cy_norm)

步驟 8b: 加速度:
         accel = |d(speed)/dt| → 歸一化

步驟 8c: 航向角:
         heading = arctan2(d(cy), d(cx))
         → sin(heading), cos(heading)

步驟 8d: 面積變化率:
         d_area = diff(area_norm) → 歸一化

步驟 8e: 徑向距離:
         radial = sqrt(dx² + dy²)

步驟 8f: 曲率:
         cross = |dx[i] × dy[i+1] - dy[i] × dx[i+1]|
         curvature = cross / (|v|³ + ε)
```

### 8.3 防止雙重平滑的設計

| 通道來源 | 上游已平滑？ | 提取器內 Hampel | 提取器內 S-G |
|---------|------------|----------------|-------------|
| cx, cy (ch 0-1) | ✅ trajectory_filter | ❌ 跳過 | ❌ 跳過 |
| bw, bh (ch 2-3) | ✅ trajectory_filter | ❌ 跳過 | ❌ 跳過 |
| area (ch 4) | ✅ trajectory_filter | ❌ 跳過 | ❌ 跳過 |
| seg_area (ch 5) | ❌ 無上游平滑 | ❌ | ✅ S-G |
| circularity (ch 6) | ❌ 無上游平滑 | ✅ Hampel | ✅ S-G |
| eq_diam (ch 7) | ❌ 無上游平滑 | ✅ Hampel | ✅ S-G |
| flat (ch 9) | ✅ trajectory_filter | ❌ 跳過 | ❌ 跳過 |

---

## 9. Subject 聚合

### 9.1 Non-TSC 提取器

當 `level=subject` 時，同一受試者的多支影片需要聚合：

```python
def _aggregate_video_features(video_features, video_keys, stats):
    # stats = ["mean", "std", "min", "max"]
    for key in video_keys:
        values = [vf[key] for vf in video_features]
        for stat in stats:
            result[f"{key}_{stat}"] = stat_func(values)
```

| 提取器 | Video 維度 | Subject 維度 | 計算 |
|--------|-----------|-------------|------|
| `motion_only` | 31 | 124 | 31 × 4 stats |
| `motion_texture` | 78 | 312 | 78 × 4 stats |
| `motion_texture_static` | 132 | 528 | 132 × 4 stats |
| `motion_texture_static_v2` | 180 | 720 | 180 × 4 stats |
| `motion_static_lite` | 33 | 132 | 33 × 4 stats |

### 9.2 TSC 提取器

TSC 的 subject 聚合不同 — 逐元素取 mean：

```python
# time_series / time_series_v2 / time_series_v3lite
def aggregate_subject(video_features):
    mats = [v["_ts_matrix"] for v in video_features]  # list of (C, T)
    avg = np.mean(np.stack(mats), axis=0)              # (C, T) 逐元素平均
    → flatten → feature dict
```

| TSC 提取器 | (C, T) | Subject 聚合 |
|-----------|--------|-------------|
| `time_series` | (18, 256) | element-wise mean |
| `time_series_v2` | (24, 256) | element-wise mean |
| `time_series_v3lite` | (12, 256) | element-wise mean |

---

## 10. 分類器與特徵提取器匹配表

| 分類器 | 類型 | 推薦提取器 | 輸入形狀 | 可用降維 | 備註 |
|--------|------|-----------|---------|---------|------|
| `random_forest` | Tabular | motion_texture_static / _v2 | (N, D) | umap, lda | scikit-learn |
| `decision_tree` | Tabular | motion_texture_static / _v2 | (N, D) | umap, lda | scikit-learn |
| `svm` | Tabular | motion_texture_static / _v2 | (N, D) | umap, lda | scikit-learn SVC |
| `lightgbm` | Tabular | motion_texture_static / _v2 | (N, D) | umap, lda | GBDT |
| `xgboost` | Tabular | motion_texture_static / _v2 | (N, D) | — | GBDT |
| `tabpfn_v2` | Tabular | motion_texture_static / _v2 | (N, D) | — | 預訓練 transformer |
| `multirocket` | **TSC** | time_series / _v2 / **_v3lite** | (N, C×T) → (N,C,T) | autoencoder | 84 kernels × dilations |
| `patchtst` | **TSC** | time_series / _v2 / **_v3lite** | (N, C×T) → (N,C,T) | learned_projection | Patch Transformer |
| `timemachine` | **TSC** | time_series / _v2 / **_v3lite** | (N, C×T) → (N,C,T) | learned_projection | State-space model |
| `random_forest` | Tabular | **motion_static_lite** | (N, 33) | umap, lda | 輕量 GLCM |
| `lightgbm` | Tabular | **motion_static_lite** | (N, 33) | umap, lda | 輕量 GLCM |

### 10.1 TSC 分類器內部 reshape

TSC 分類器接收展平的 `(N, C×T)` 向量（與 FeatureVectoriser 統一介面），內部會：

```python
# time_series:       (N, 4608) → (N, 18, 256)
# time_series_v2:    (N, 6144) → (N, 24, 256)
# time_series_v3lite:(N, 3072) → (N, 12, 256)
X = X.reshape(N, C, T)
```

---

## 11. 完整特徵 Key 列表

### 11.1 MOTION_FEATURE_KEYS (31 D)

```
num_points, duration_frames,
displacement_x, displacement_y, net_displacement,
span_x, span_y, path_length, straightness_ratio,
mean_speed, max_speed, std_speed, median_speed, p95_speed, p5_speed,
mean_acc, max_acc, std_acc,
mean_jerk, max_jerk, std_jerk,
curvature_mean, curvature_std,
angular_change_mean, angular_change_std,
area_mean, area_std, area_range, area_change_mean, area_change_std
```

### 11.2 CSA_FEATURE_KEYS (8 D)

```
csa_first_area, csa_last_area,
csa_first_perimeter, csa_last_perimeter,
csa_first_eq_diameter, csa_last_eq_diameter,
csa_first_circularity, csa_last_circularity
```

### 11.3 _RAW_TEXTURE_KEYS_PER_FRAME (21 D × 2)

```
tex_mean, tex_std, tex_skewness, tex_kurtosis,
tex_grad_mean, tex_grad_std,
tex_glcm_contrast, tex_glcm_dissimilarity, tex_glcm_homogeneity,
tex_glcm_energy, tex_glcm_correlation,
tex_lbp_00 ~ tex_lbp_09
```

### 11.4 CTS_STATIC_FEATURE_KEYS (35 D)

```
cts_csa_min, cts_csa_max, cts_csa_mean, cts_csa_std, cts_csa_range, cts_csa_cv,
cts_csa_first, cts_csa_last, cts_swelling_ratio,
cts_flat_mean, cts_flat_std, cts_flat_max, cts_flat_min,
cts_circularity_first, cts_circularity_last, cts_circularity_mean, cts_circularity_std,
cts_eq_diam_mean, cts_eq_diam_std, cts_eq_diam_first, cts_eq_diam_last,
cts_compact_mean, cts_compact_std,
cts_echo_mean_first, cts_echo_std_first, cts_echo_mean_last, cts_echo_std_last, cts_echo_delta,
cts_nerve_net_disp, cts_nerve_path_len, cts_nerve_stability_x, cts_nerve_stability_y,
cts_aspect_first, cts_aspect_last
```

### 11.5 CTS_STATIC_V2_EXTRA_KEYS (24 D)

```
cts_csa_q25, cts_csa_q75, cts_csa_iqr, cts_csa_skew, cts_csa_kurt,
cts_circ_q25, cts_circ_q75, cts_circ_iqr,
cts_eq_diam_q25, cts_eq_diam_q75, cts_eq_diam_iqr,
cts_flat_q25, cts_flat_q75, cts_flat_iqr,
cts_csa_grad_mean, cts_csa_grad_std, cts_flat_grad_mean, cts_flat_grad_std,
cts_disp_median_mean, cts_disp_median_max, cts_temporal_energy, cts_radial_range,
cts_mean_angular_change, cts_path_efficiency
```

### 11.6 TS_GEO_CHANNEL_KEYS V1 (10 ch)

```
ts_cx, ts_cy, ts_bw, ts_bh, ts_area,
ts_seg_area, ts_circularity, ts_eq_diam, ts_speed, ts_flat
```

### 11.7 TS_GEO_CHANNEL_KEYS_V2 (16 ch)

```
ts_dx, ts_dy, ts_bw, ts_bh, ts_area,
ts_seg_area, ts_circularity, ts_eq_diam, ts_speed, ts_flat,
ts_accel, ts_heading_sin, ts_heading_cos, ts_d_area, ts_radial_dist, ts_curvature
```

### 11.8 TS_TEX_CHANNEL_KEYS (8 ch)

```
ts_tex_pca_0 ~ ts_tex_pca_7
```

### 11.9 LITE_MOTION_KEYS (13 D)

```
num_points, duration_frames, path_length, straightness_ratio,
mean_speed, std_speed, median_speed,
mean_acc, std_acc,
disp_median_mean, disp_median_std, disp_median_max,
mean_heading_change
```

### 11.10 LITE_STATIC_KEYS (10 D)

```
csa_mean, csa_std, csa_strain_rate, swelling_ratio,
eq_diam_mean, eq_diam_strain_rate,
circularity_mean, circularity_std,
aspect_ratio_mean, aspect_ratio_std
```

### 11.11 LITE_TEXTURE_KEYS (10 D)

```
glcm_contrast_mean, glcm_contrast_std,
glcm_homogeneity_mean, glcm_homogeneity_std,
glcm_energy_mean, glcm_correlation_mean,
gray_mean_mean, gray_mean_std,
gray_grad_mean, gray_grad_std
```

### 11.12 TS_LITE_CHANNEL_KEYS (12 ch)

```
ts_speed, ts_accel, ts_dx_median, ts_dy_median,
ts_heading_sin, ts_heading_cos,
ts_csa_norm, ts_eq_diam_norm, ts_aspect_ratio,
ts_glcm_contrast, ts_glcm_homogeneity, ts_gray_mean
```

---

## 附錄 A：維度匯總

| 提取器 | Motion | Static | Texture (PCA) | 時序 | 總維度 |
|--------|--------|--------|--------------|------|--------|
| motion_only | 31 | — | — | — | **31** |
| motion_texture | 31 | 8 (CSA) | 39 (trad PCA) | — | **78** |
| motion_texture_static | 31 | 35 | 66 (ResNet PCA) | — | **132** |
| motion_texture_static_v2 | 31 | 59 | 90 (ResNet PCA) | — | **180** |
| time_series | — | — | 8 ch (ResNet PCA) | 10 geo ch | **18×256=4,608** |
| time_series_v2 | — | — | 8 ch (ResNet PCA) | 16 geo ch | **24×256=6,144** |
| **motion_static_lite** | 13 | 10 | 10 (GLCM) | — | **33** |
| **time_series_v3lite** | 6 ch | 3 ch | 3 ch (GLCM) | 12 ch 合併 | **12×256=3,072** |

## 附錄 B：V1 vs V2 改進對比

| 面向 | V1 | V2 |
|------|----|----|
| 位置表示 (TSC) | 絕對座標 cx, cy | 偏離中位數 dx, dy |
| 運動學通道 (TSC) | 10 ch | 16 ch (+accel, heading, d_area, radial, curvature) |
| 靜態特徵 (Non-TSC) | 35 D | 59 D (+百分位, IQR, 偏度, 峰度, 梯度, 路徑效率) |
| PCA 紋理維度 (Non-TSC) | 66 D | 90 D |
| 總維度 (Non-TSC) | 132 D | 180 D |
| 總維度 (TSC) | 4,608 D | 6,144 D |

## 附錄 C：V3-Lite 設計理念與對比

### 設計目標

1. **無深度學習紋理** — 以 GLCM 取代 ResNet-18，不需 GPU / 預訓練權重
2. **無絕對座標** — 運動通道僅保留速度、加速度、偏離中位數、航向
3. **通道精簡** — Non-TSC 33 D 對比 V2 的 180 D；TSC 12 ch（對比 V2 的 24 ch），但 steps 提升至 256 使總維度同為 3,072 D
4. **無 PCA** — finalize_batch 不需 fit/transform，減少跨 fold 外洩風險

### V1 vs V2 vs V3-Lite

| 面向 | V1 | V2 | V3-Lite |
|------|----|----|--------|
| 位置表示 (TSC) | 絕對 cx, cy | 偏離中位數 dx, dy | 偏離中位數 dx, dy |
| 紋理方法 | ResNet-18 → PCA | ResNet-18 → PCA | **GLCM（無 PCA）** |
| PCA 依賴 | ✅ | ✅ | ❌ |
| GPU 依賴 | ✅ ResNet | ✅ ResNet | ❌ 純 numpy |
| Non-TSC 維度 | 132 D | 180 D | **33 D** |
| TSC 維度 | 4,608 D (18×256) | 6,144 D (24×256) | **3,072 D (12×256)** |
| TSC 通道數 | 18 (10 geo + 8 tex PCA) | 24 (16 geo + 8 tex PCA) | **12 (6 motion + 3 static + 3 texture)** |
| 靜態統計量 | 35 D (first/last/min/max/mean…) | 59 D (+百分位/IQR/梯度) | **10 D (mean/std/range/ratio)** |
| 運動特徵 | 31 D (含 jerk/curvature) | 31 D | **13 D (速度/加速度/偏離/航向)** |
