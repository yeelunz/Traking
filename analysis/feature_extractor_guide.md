# Feature Extractor Guide

本文件整理目前專案中所有主要 `tab_*` / `tsc_*` 特徵提取器的設計差異、輸出結構、依賴、適用情境與選型建議。

重點原則：

- `tab_*` 給 tabular classifier 用，例如 RF / XGBoost / LightGBM / TabPFN。
- `tsc_*` 給 time-series classifier 用，例如 MultiRocket / PatchTST / TimeMachine。
- 這份文件以「實際程式行為」為準，而不是只看 class docstring。少數舊 docstring 的步數或維度描述已經過時。

## 1. 目前可用 extractor 總覽

### Tabular 家族

| 名稱 | 影片層級輸出維度 | Motion | Static | Texture | 深度模型依賴 | 單位 |
| --- | ---: | --- | --- | --- | --- | --- |
| `tab_v2` | 132 | 31D hand-crafted | 35D CTS static | 66D ResNet-18/PCA | 是 | raw pixel |
| `tab_v2_extend` | 180 | 31D hand-crafted | 59D CTS static v2 | 90D ResNet-18/PCA | 是 | raw pixel |
| `tab_v3_lite` | 33 | 13D lite motion | 10D lite static | 10D GLCM/gray/grad | 否 | raw pixel |
| `tab_v3_pro` | 33 | 13D lite motion | 10D lite static | 10D ConvNeXt | 是 | raw pixel |
| `tab_v4` | 36 | 15D motion v4 | 10D static v4 | 11D ConvNeXt | 是 | raw pixel |
| `tab_v5` | 33 | 12D MOMENT motion | 10D static v4 | 11D ConvNeXt | 是 | raw pixel |
| `tab_v5_lite` | 30 | 15D MOMENT motion | 無 | 15D ConvNeXt | 是 | raw pixel |
| `tab_v6` | 33 | 12D MOMENT motion | 10D static v4 | 11D ConvNeXt | 是 | motion/static 轉 cm |
| `tab_v6_lite` | 30 | 15D MOMENT motion | 無 | 15D ConvNeXt | 是 | motion 轉 cm |

### Time-Series 家族

| 名稱 | 實際輸出形狀 | 平坦化維度 | Motion/Geometry | Texture | 深度模型依賴 | 單位 |
| --- | --- | ---: | --- | --- | --- | --- |
| `tsc_v2` | 18 channels x 256 steps | 4608 | 幾何/運動 10ch | ResNet-18 PCA 8ch | 是 | raw pixel |
| `tsc_v2_extend` | 24 channels x 256 steps | 6144 | 幾何/運動 16ch | ResNet-18 PCA 8ch | 是 | raw pixel |
| `tsc_v3_lite` | 12 channels x 256 steps | 3072 | lite motion/static 9ch | hand-crafted texture 3ch | 否 | raw pixel |
| `tsc_v3_pro` | 12 channels x 256 steps | 3072 | lite motion/static 9ch | ConvNeXt texture 3ch | 是 | raw pixel |
| `tsc_v4` | 12 channels x 256 steps | 3072 | lite motion/static 9ch | ConvNeXt texture 3ch | 是 | base channels 轉 cm |

## 2. 選型先看這一段

### 如果你最在意可解釋性

- 首選 `tab_v4` 或 `tab_v6`
- 次選 `tab_v3_lite`

原因：

- `tab_v4` / `tab_v6` 的 static 特徵仍然是明確的 CSA、equivalent diameter、circularity、aspect ratio
- `tab_v3_lite` 則連 texture 都是 GLCM 與灰階統計，比 deep embedding 更容易解讀

### 如果你最在意跨影片尺度一致性

- 首選 `tab_v6`
- 次選 `tab_v6_lite`

原因：

- `tab_v6` / `tab_v6_lite` 會對 pixel 幾何量做 pixel/cm 轉換
- 目前這個「尺度校正」只做在 `tab_v6` 系列，`tsc_*` 還沒有對應的 cm 版

### 如果你最在意 motion representation 的表達能力

- 首選 `tab_v5` / `tab_v6`
- 若你走 TSC，選 `tsc_v4`

原因：

- `tab_v5` / `tab_v6` 的 motion 不再是手刻 summary，而是先把 `(dx, dy)` 序列丟進 MOMENT，再用 PCA 壓成低維嵌入
- `tsc_v4` 保留整條時間序列，對 temporal pattern 更完整，且 base 幾何量先轉成 cm

### 如果你最在意部署簡單、不想碰 GPU 或大型模型

- Tabular: `tab_v3_lite`
- TSC: `tsc_v3_lite`

原因：

- 不需要 ResNet / ConvNeXt / MOMENT
- texture 直接從 patch 算 GLCM 與灰階統計

### 如果你資料量不大，又想穩定

- Tabular: `tab_v4` 或 `tab_v6`
- TSC: `tsc_v3_lite`

原因：

- `tab_v4` / `tab_v6` 維度中等，訊號明確，通常比 `tab_v2_extend` 更不容易過擬合
- `tsc_v3_lite` 比 `tsc_v2_extend` 更輕、更乾淨

### 如果你資料量夠大，想把 texture 也做深一點

- Tabular: `tab_v5` / `tab_v6`
- TSC: `tsc_v4`

## 3. 共同設計差異

### 3.1 `tab_*` 與 `tsc_*` 的根本差別

`tab_*` 會把一段影片總結成一個固定長度的影片向量。這種設計假設「重要的是整段影片的 summary」，例如平均速度、最大位移、平均 CSA、代表性 texture embedding。適合資料量不大、希望模型容易訓練、容易做 feature importance 的場景。

`tsc_*` 則保留時間軸，把每個 channel 在固定步數上展開。這種設計假設「重要的是時間演化的形狀」，例如先位移後回彈、速度尖峰、面積變化的節律。適合 temporal pattern 很重要的任務。

### 3.2 手工特徵與深度特徵

- `v2` 系列：motion/static 多為手工統計，texture 用 ResNet-18
- `v3_lite`：全部都偏手工
- `v3_pro` / `v4`：motion/static 偏手工，texture 用 ConvNeXt
- `v5` / `v6`：motion 換成 MOMENT，texture 用 ConvNeXt

### 3.3 subject aggregation 的差異

- `tab_v2` / `tab_v2_extend`：subject 層級不是單純平均，而是對每個影片特徵再做 `mean/std/min/max` 聚合，subject 向量會更大
- `tab_v3_pro` / `tab_v4` / `tab_v5` / `tab_v6`：subject 層級基本上就是影片向量逐維平均，維度不變
- `tsc_*`：subject 層級也是逐維平均，維度與影片層級相同

### 3.4 `texture_mode` 的意義

這一段以「實作行為」為準，對照程式可看：

- `tracking/classification/texture_backbone.py`
- `tracking/classification/feature_extractors/v3pro.py`
- `tracking/classification/feature_extractors/v4.py`
- `tracking/classification/feature_extractors/v5.py`
- `tracking/classification/feature_extractors/v3pro_tsc.py`
- `tracking/texture_pretrain/trainer.py`

先講共通骨架：

- `TextureBackboneWrapper` 只接受 `freeze | learnable | pretrain` 三種模式。
- wrapper 一律包含 backbone 與 projection (`self.proj`)；forward 會先做 ImageNet normalize，再跑 backbone，最後過 projection。

各 mode 的真實差異如下。

#### 3.4.1 `freeze`

- wrapper 層：
  - backbone 設為不可訓練（`requires_grad_(False)`）並切到 eval。
  - `_forward_backbone` 走 `torch.no_grad()`。
- extractor 層（`tab_v3_pro / tab_v4 / tab_v5 / tab_v6 / tsc_v3_pro / tsc_v4`）：
  - 先取 raw backbone 特徵（非 projection 輸出）。
  - 會把 raw 特徵暫存為 `_raw_texture_backbone_batch`（或 TSC 的 `_raw_convnext_ts`）。
  - 在 `finalize_batch()` 以訓練集 fit PCA，推論集只 transform。
- 結果：texture 維度由 PCA 決定（例如 tab v4/v5 常見 11D，`tab_v5_lite`/`tab_v6_lite` 是 15D；`tsc_v3_pro` 會落到 3 個 texture channels）。

#### 3.4.2 `learnable`

- wrapper 層：
  - backbone 設為可訓練（`requires_grad_(True)`）。
  - `_forward_backbone` 不包 `no_grad`。
- 但在目前 classification extractor 實際路徑：
  - 建立 wrapper 後會 `wrapper.eval()`。
  - 抽 feature 時仍以 `with torch.no_grad(): feat = wrapper(xb)` 執行。
  - `finalize_batch()` 不走 PCA，直接使用 projection 後低維輸出。
- 結果：在目前離線特徵抽取流程中，`learnable` 代表「用 projection 輸出」，不代表此流程會真的更新權重。

#### 3.4.3 `pretrain`

- wrapper 層：
  - 初始化時強制需要 `texture_pretrain_ckpt`。
  - 會載入 checkpoint 的 backbone 權重，若有 `projection_state_dict` 也會載入 projection。
  - mode 套用後 backbone 仍是 frozen（不是 learnable）。
- extractor 層：
  - 與 `learnable` 一樣走 projection 低維輸出，不走 PCA。
  - 同樣在 `eval + no_grad` 路徑執行。
- 結果：`pretrain` 在分類特徵抽取時是「載入已訓練 texture 表徵後固定使用」。

#### 3.4.4 最常見誤解：`pretrain` 會不會更新權重？

- 在 classification 的實際使用階段：不更新（eval + no_grad，且 pretrain mode backbone 也 frozen）。
- 在 Stage-1 texture pretrain 訓練階段：會更新。
  - `tracking/texture_pretrain/trainer.py` 的 `_run_epoch(training=True)` 會做 `backward()` 與 `optimizer.step()`。

#### 3.4.5 哪些 extractor 真正吃 `texture_mode`，哪些只是相容參數

- 真正使用上述三種 mode 的主線：`tab_v3_pro`、`tab_v4`、`tab_v5`、`tab_v5_lite`、`tab_v6`、`tab_v6_lite`、`tsc_v3_pro`、`tsc_v4`。
- `tab_v3_lite` 與 `tsc_v3_lite` 是 hand-crafted texture（GLCM/灰階統計），即使外部傳 `learnable/pretrain` 也會被強制改回 `freeze`。
- `tab_v2` / `tab_v2_extend` 的 texture 是 ResNet-18 + PCA 舊路線，並非 `TextureBackboneWrapper` 這套 mode 行為。

### 3.5 正規化策略

- `tab_v2` / `tab_v2_extend` / `tab_v3_lite` / `tab_v3_pro` / `tab_v4` / `tab_v5` / `tab_v6`
  - 影片特徵最後都會做 global z-score
  - deep texture 維度通常會跳過 z-score 或採特別處理
- `tsc_*`
  - 以 channel 為單位做 global z-score
  - 訓練集 fit，驗證/測試集 transform

## 4. 各 extractor 詳解

## 4.1 `tab_v2`

### 核心設計

這是最早期的完整 tabular 版。它把影片拆成三塊：

- 31D motion summary
- 35D CTS 診斷導向 static
- 66D ResNet-18 texture

總共 132D。

### Motion

直接用 `base.py` 的 `MOTION_FEATURE_KEYS`，屬於 hand-crafted 軌跡統計。這一支的設計重點是從軌跡本身萃取大量 summary，而不是學習 sequence embedding。

### Static

35D CTS static 內容包括：

- CSA min/max/mean/std/range/cv
- swelling proxy
- flattening index
- circularity
- equivalent diameter
- compactness
- echo intensity
- nerve displacement
- aspect ratio

這一支很 CTS-diagnostic oriented。

### Texture

- 只抓第一幀與最後一幀
- 從 segmentation-masked ROI 取 ResNet-18 embedding
- 2 x 512 = 1024 raw texture
- `finalize_batch()` 再用 PCA 壓成 66D

### 優點

- static 很完整
- 對診斷型特徵工程友善
- 適合 tabular baseline

### 缺點

- texture 只看首尾兩幀，時間內的 texture 變化幾乎沒保留
- motion 與 static 維度不少，subject aggregation 後維度更大
- 全部仍是 raw pixel 尺度

### 適合何時用

- 你想做老派、可解釋、診斷導向 baseline
- 想保留完整 CTS static 但不需要最新 motion embedding

## 4.2 `tsc_v2`

### 實際輸出

- 18 channels x 256 steps
- 實際平坦化維度是 4608

注意：舊 docstring 還寫成 `18 x 128 = 2304`，但實作常數 `N_TS_STEPS=256`，以程式為準。

### Channel 組成

幾何/運動 10ch：

- `cx`
- `cy`
- `bw`
- `bh`
- `area`
- `seg_area`
- `circularity`
- `eq_diam`
- `speed`
- `flatness`

texture 8ch：

- ResNet-18 raw frame embedding 經 PCA 壓成 8 個 per-frame channels

### 優點

- 保留完整時間軸
- 幾何與 texture 都能隨時間變化

### 缺點

- 還保留絕對座標 `cx/cy`，跨受試者位置偏差較大
- 維度高
- 依賴 ResNet + PCA

### 適合何時用

- 你想保留完整時間序列
- 願意接受早期設計中絕對位置偏差

## 4.3 `tab_v2_extend`

### 核心設計

這是 `tab_v2` 的強化版：

- motion 仍是 31D
- static 從 35D 擴到 59D
- texture 從 66D 擴到 90D

總共 180D。

### 比 `tab_v2` 多了什麼

新增 24D static，包括：

- CSA / circularity / equivalent diameter / flattening 的 q25、q75、IQR
- CSA 的 skew / kurtosis
- temporal gradient 統計
- displacement from median
- temporal energy
- radial range
- mean angular change
- path efficiency

### 優點

- CTS static 訊息最完整
- 能更細緻描述形狀分布與時序變化

### 缺點

- 維度很高，容易在小資料過擬合
- texture 仍然只取首尾兩幀 ResNet
- subject aggregation 後更大

### 適合何時用

- 你非常在意手工診斷特徵完整性
- 你有足夠資料量，或下游模型本來就擅長高維 tabular

## 4.4 `tsc_v2_extend`

### 實際輸出

- 24 channels x 256 steps
- 實際平坦化維度是 6144

注意：舊 docstring 還寫成 `24 x 128 = 3072`，但實作已經是 256 steps。

### 相對 `tsc_v2` 的關鍵改進

1. 前兩個位置 channel 不再是絕對 `cx/cy`，改成 `displacement from median`
2. 額外加入 6 個衍生動態 channel

### Channel 組成

位移 2ch：

- `ts_dx`
- `ts_dy`

大小/形狀 6ch：

- `ts_bw`
- `ts_bh`
- `ts_area`
- `ts_seg_area`
- `ts_circularity`
- `ts_eq_diam`

原本運動 2ch：

- `ts_speed`
- `ts_flat`

新增動態 6ch：

- `ts_accel`
- `ts_heading_sin`
- `ts_heading_cos`
- `ts_d_area`
- `ts_radial_dist`
- `ts_curvature`

texture 8ch：

- ResNet PCA channels

### 優點

- 相比 `tsc_v2`，更聚焦「動作模式」而不是絕對位置
- 對 temporal classifier 更合理

### 缺點

- 維度非常高
- 依賴 ResNet + PCA
- 仍然是 raw pixel 尺度

### 適合何時用

- 你想用 TSC
- 想保留早期手工 kinematics 但避開絕對位置偏差

## 4.5 `tab_v3_lite`

### 核心設計

這是專案裡最輕量、最乾淨、最少依賴的 tabular extractor。

輸出：

- motion 13D
- static 10D
- texture 10D

總共 33D。

### Motion

設計哲學是「盡量 position-invariant」，不使用絕對座標，重點放在：

- path length
- straightness ratio
- mean/std/median speed
- mean/std acceleration
- displacement from median
- heading change

### Static

只保留簡潔的形狀摘要：

- CSA mean/std/strain_rate/swelling_ratio
- eq_diam mean/strain_rate
- circularity mean/std
- aspect_ratio mean/std

### Texture

不走 deep model，而是從 patch 直接算：

- GLCM contrast/homogeneity/energy/correlation
- gray mean/std
- gradient mean/std

### 優點

- 不需要 GPU
- 特徵最容易理解
- 維度低
- 小資料通常很穩

### 缺點

- texture representation 上限比 deep extractor 低
- motion 還是手工 summary，不像 `v5/v6` 用 sequence foundation model

### 適合何時用

- 你要做輕量 baseline
- 你要跑大量實驗
- 你想優先看可解釋性與穩定性

## 4.6 `tsc_v3_lite`

### 實際輸出

- 12 channels x 256 steps
- 平坦化維度 3072

### Channel 組成

motion 6ch：

- speed
- accel
- dx from median
- dy from median
- heading_sin
- heading_cos

static/shape 3ch：

- csa_norm
- eq_diam_norm
- aspect_ratio

texture 3ch：

- glcm_contrast
- glcm_homogeneity
- gray_mean

### 設計特徵

- 沒有 deep model
- texture 先在 sparse frames 上計算，再內插到整條 timeline
- `finalize_batch()` 會對 texture channels 做 PCA，然後對所有 channels 做 channel-wise z-score

### 優點

- TSC 中最輕量
- 不依賴 ResNet / ConvNeXt
- 比 `tsc_v2` 系列更乾淨

### 缺點

- texture 表徵能力有限
- 沒有深度時序 embedding

### 適合何時用

- 你要 TSC baseline
- 硬體有限
- 你想先驗證 temporal pattern 是否重要

## 4.7 `tab_v3_pro`

### 核心設計

可以把它理解成：

- `tab_v3_lite` 的 motion/static
- 換成 ConvNeXt 的 texture

所以總維度仍然是 33D：

- motion 13D
- static 10D
- texture 10D

### 關鍵差異

相對 `tab_v3_lite`，只有 texture branch 升級。motion/static 邏輯完全沿用 lite 思路。

### Texture

- 從多個 ROI frame 抽 patch
- 用 `TextureBackboneWrapper`
- `freeze` 模式下先取 raw backbone，再在 `finalize_batch()` 做 PCA
- `learnable/pretrain` 模式下直接用 wrapper projection

### 優點

- 保留 v3_lite 的乾淨 motion/static
- texture 能力比 GLCM 強
- 維度不高

### 缺點

- 仍然沒有 v4 的曲線特徵
- 也沒有 v5/v6 的 MOMENT motion

### 適合何時用

- 你喜歡 v3_lite 的 feature philosophy
- 但希望 texture 不要太弱

## 4.8 `tsc_v3_pro`

### 核心設計

這是 `tsc_v3_lite` 的 deep texture 升級版。

base channels 保持 12 個：

- 前 9 個來自 v3_lite 的 motion/static
- 最後 3 個 texture channels 改成 ConvNeXt 低維投影

### 實際輸出

- 12 channels x 256 steps
- 平坦化維度 3072

### 特別注意

`texture_dim` 雖然預設可寫 32，但當 `texture_mode != freeze` 時，實作會強制把 texture channels 壓成 3 個，因為整體 channel layout 固定就是 12。

### 優點

- 保留完整時間軸
- texture 比 `tsc_v3_lite` 強
- 維度仍然比 `tsc_v2_extend` 精簡很多

### 缺點

- 還是 raw pixel 尺度
- 需要 deep backbone

### 適合何時用

- 你要 TSC 主線模型
- 你想保留時序資訊，同時提升 texture branch 品質

## 4.9 `tab_v4`

### 核心設計

`tab_v4` 是一個很平衡的版本：

- motion 15D
- static 10D
- texture 11D

總共 36D。

### Motion

比 v3_lite 多一些曲線型特徵：

- duration
- path_length
- straightness
- speed mean/std
- acceleration mean/std
- relative displacement mean/std/max
- heading change mean/std
- curve amplitude
- curve curvature
- curve fit r2

這一版的 motion 比 v3_lite 更像「軌跡幾何 summary」。

### Static

使用 10D static v4：

- csa_mean/std/strain/swelling_ratio
- eq_diam_mean/strain
- circularity_mean/std
- aspect_ratio_mean/std

### Texture

- ConvNeXt wrapper
- 多幀 ROI 抽樣
- freeze 模式走 PCA
- learnable/pretrain 模式走投影 head
- 預設 texture dim = 11

### 優點

- 可解釋性與表現力平衡很好
- motion 比 v3_lite 更成熟
- 維度不大

### 缺點

- motion 仍然是手工統計，不是 learned sequence representation
- 沒有 cm 校正

### 適合何時用

- 你想要目前最穩的 hand-crafted + deep texture 折衷版
- 作為 tabular 主線 baseline 很合適

## 4.10 `tab_v5`

### 核心設計

`tab_v5` 最大變化是把 motion branch 從 hand-crafted summary 改成 MOMENT。

輸出：

- motion 12D
- static 10D
- texture 11D

總共 33D。

### Motion

流程是：

1. 對 trajectory center 做 resample
2. 把位置轉成 step displacement `(dx, dy)`
3. 形成 2 x `moment_input_steps` 的序列
4. 丟進 `MOMENTPipeline`
5. 取 embedding 後再用 PCA 壓成 12D

也就是說，`tab_v5` 的 motion 已不是人工定義 feature，而是 foundation time-series embedding。

### Static

仍然是 v4 的 10D static。

### Texture

與 v4 同一家族，仍然是 ConvNeXt texture。

### 優點

- motion representation 更強
- 維度仍然控制得很小
- 適合 tabular classifier

### 缺點

- 需要 `momentfm`
- motion 變得比較不直觀
- 仍然是 raw pixel 尺度

### 適合何時用

- 你想把 motion branch 升級
- 你接受可解釋性下降一些，換取更好的 sequence representation

## 4.11 `tab_v5_lite`

### 核心設計

這不是 `tab_v5` 的「弱化版 texture」，而是把結構改成：

- motion 15D MOMENT
- 沒有 static
- texture 15D ConvNeXt

總共 30D。

### 與 `tab_v5` 的真正差異

- `tab_v5` 有 static、motion 12D、texture 11D
- `tab_v5_lite` 沒有 static、motion 15D、texture 15D

因此它比較像「把容量留給 motion + texture 的雙分支版本」。

### 優點

- 表示能力集中在 learned motion 與 deep texture
- 維度低

### 缺點

- 完全捨棄 static 幾何摘要
- 若你的分類訊號很依賴 CSA / circularity / aspect ratio，會吃虧

### 適合何時用

- 你懷疑 static 不是主訊號
- 你想把容量集中給 motion 與 texture

## 4.12 `tab_v6`

### 核心設計

`tab_v6` 幾乎可以視為 `tab_v5 + pixel/cm 校正`。

輸出仍然是：

- motion 12D MOMENT
- static 10D
- texture 11D

總共 33D。

### 和 `tab_v5` 的關鍵差別

只要有 pixel 幾何量，`tab_v6` 會先做 pixel/cm 轉換，再進 motion/static 分支。

包括：

- bbox 的 `x/y/w/h`
- segmentation stats 的 `bbox`
- segmentation centroid
- perimeter
- equivalent diameter
- area 會按平方比例轉換
- MOMENT motion input 的 `dx/dy` 也會先轉成 cm

### 什麼沒有轉

- texture branch 仍然直接從原始影像 patch 抽 deep feature
- 因為 texture 本身不是幾何量，不存在 pixel-to-cm 的數值轉換

### 優點

- 能跨不同探頭深度／不同尺規做幾何一致化
- 保留 `tab_v5` 的強 motion branch

### 缺點

- 尺度校正目前是靠右側深度尺單幀解析
- 假設 x、y 方向共用同一個比例
- TSC 家族目前沒有對應 v6

### 適合何時用

- 你的資料來自多種深度尺／多種探頭視野
- 你希望 bbox/centroid/displacement 不再只是 pixel
- 目前這是最推薦的 tabular 主線 extractor

## 4.13 `tab_v6_lite`

### 核心設計

`tab_v6_lite` 幾乎等於 `tab_v5_lite + pixel/cm 校正`。

輸出：

- motion 15D MOMENT
- texture 15D ConvNeXt

總共 30D。

### 和 `tab_v5_lite` 的關鍵差別

- MOMENT 的輸入 `dx/dy` 先做 cm 轉換
- 沒有 static branch，所以尺度校正影響主要在 motion

### 適合何時用

- 你想用輕一點的 v6 版本
- 但又不想完全回到 raw pixel motion

## 4.14 `tsc_v4`

### 核心設計

`tsc_v4` 可以直接理解成：

- `tsc_v3_pro`
- 再加上 per-video 的 pixel/cm 尺度校正

輸出形狀不變：

- 12 channels x 256 steps
- 平坦化 3072 維

### 和 `tsc_v3_pro` 的關鍵差別

在建立 base time-series channels 前，會先把 sample 裡所有 pixel 幾何量換成 cm，再交給 `v3_lite` 的 channel builder。也就是說，以下這些 base channels 會受到尺度校正影響：

- `speed`
- `accel`
- `dx_median`
- `dy_median`
- `csa_norm`
- `eq_diam_norm`

而這些不會受影響：

- `heading_sin`
- `heading_cos`
- `aspect_ratio`
- texture 3 channels

### 轉換方式

- bbox `x/y/w/h` 先換成 cm
- segmentation 的 `area / centroid / perimeter / equivalent_diameter / bbox` 都先換成 cm 對應值
- 然後再重建時間序列
- texture timeline 仍然從原始影像 ROI 抽 ConvNeXt 特徵，不做 cm 數值轉換

### 優點

- 對不同深度尺、不同探頭設定的影片更一致
- 保留 `tsc_v3_pro` 的時間序列表達能力
- 和 `tab_v6` 的設計哲學一致

### 缺點

- 仍然依賴單幀解析右側 depth ruler
- 目前只對 base 幾何 channels 做單位校正，texture 本身還是影像 embedding

### 適合何時用

- 你要跑 TSC 主線
- 又希望不同影片的 displacement / size / area 能以實際尺度比較

## 5. 實際推薦路線

### 推薦 1：一般 tabular 主線

選 `tab_v6`

原因：

- motion 已升級成 MOMENT
- static 還在，保留幾何與形狀資訊
- texture 也是現代 backbone
- 多資料來源時能做 pixel/cm 一致化

### 推薦 2：如果你想先做最穩 baseline

選 `tab_v4`

原因：

- 維度中等
- 沒有 MOMENT 的額外依賴
- feature 結構很平衡

### 推薦 3：如果你只想要一個超輕量可解釋 baseline

選 `tab_v3_lite`

### 推薦 4：如果你要完整保留時間軸

選 `tsc_v4`

原因：

- 比 `tsc_v2_extend` 精簡
- 比 `tsc_v3_lite` texture 強
- 多了尺度校正，跨影片更一致

### 推薦 5：如果你只想做不吃 GPU 的 TSC baseline

選 `tsc_v3_lite`

### 不太建議優先從哪裡開始

- `tab_v2_extend`
  - 太重、太手工、維度高，較容易過擬合
- `tsc_v2`
  - 還保留絕對位置
- `tsc_v2_extend`
  - 訊號很多，但維度非常高，成本也高

## 6. 你可以怎麼選

如果你的目標是：

- 最終正式主線 tabular 模型：`tab_v6`
- 不做尺度校正但要穩：`tab_v4`
- 超輕量 baseline：`tab_v3_lite`
- 最終正式主線 TSC 模型：`tsc_v4`
- 輕量 TSC baseline：`tsc_v3_lite`

如果你更在意：

- 可解釋性：`tab_v4` > `tab_v3_lite` > `tab_v6` > `tab_v5`
- 深度時序表徵：`tab_v6` / `tab_v5` > `tab_v4` / `tab_v3_pro`
- 少依賴：`tab_v3_lite` / `tsc_v3_lite`
- 跨尺度一致性：`tab_v6` / `tab_v6_lite` / `tsc_v4`

## 7. 目前不存在的版本

目前沒有：

- `tsc_v5`
- `tsc_v6`

也就是說，目前 TSC 這條線裡，帶 pixel/cm 尺度校正的是 `tsc_v4`。

## 8. 歷史版本

程式裡還有兩個標成 `DELETED_*` 的舊 extractor：

- `DELETED_motion_only`
- `DELETED_motion_texture`

這兩支是歷史相容用途，不建議再作為正式實驗主線。

## 9. 總結

如果只給一句話建議：

- Tabular 主線用 `tab_v6`
- TSC 主線用 `tsc_v4`
- 最穩 baseline 用 `tab_v4`
- 最省資源 baseline 用 `tab_v3_lite` / `tsc_v3_lite`

如果你的任務很依賴不同影片之間的實際位移、深度、大小可比性，那目前最合理的主線會是 `tab_v6` 與 `tsc_v4`，因為它們都會先把 pixel 幾何量轉成 cm 再抽特徵。
