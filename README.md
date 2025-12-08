# Video Tracking Framework (Skeleton)

This is a minimal, extensible framework to run tracking experiments against videos and annotations exported by the label tool in `labelTool/`.

Key pieces:
- tracking/core: Interfaces and registries
- tracking/data: Dataset manager for COCO-VID-like JSON next to videos
- tracking/preproc: Example CLAHE module
- tracking/models: Built-in trackers (Template Matching, FASTSpeckle/NCC, YOLOv11, OC-SORT wrapper, StrongSORT wrapper, ToMP via pytracking, TaMOs via pytracking, MixFormerV2)
- tracking/eval: Basic evaluator computing IoU and center error
- tracking/orchestrator: Pipeline runner to glue everything

Quick start:
1. Place videos and `<video>.json` exported from the label tool in a dataset root.
2. Edit `pipeline.example.yaml` to point to your dataset root and results path.
3. Install dependencies (see below) and run:

```bat
python run_pipeline.py --config pipeline.example.yaml
```

Dependencies:
- Python 3.9+
- opencv-python
- numpy
- pyyaml (if using YAML configs)
 - PySide6 (optional, for the simple UI)
 - matplotlib (for plots and k-fold aggregates)
 - scikit-learn (for subject-level classification metrics)

Notes on trackers:
- CSRT tracker requires `opencv-contrib-python` (cv2.legacy). If you want to use CSRT, uninstall opencv-python and install the contrib build instead.
- OC-SORT wrapper relies on Ultralytics YOLOv11 for detections (`ultralytics` package) and the `ocsort` PyPI package. Both are listed in `requirements.txt`.
- StrongSORT wrapper reuses the cloned `libs/StrongSORT` repo plus Ultralytics YOLOv11; make sure `ultralytics`, `torch`, and OpenCV are installed (see `requirements.txt`).
- ToMP integration uses the bundled `libs/pytracking` checkout. Download the pretrained weights (e.g. `tomp50.pth.tar`) into `libs/pytracking/pytracking/networks/` and ensure `pytracking/pytracking/evaluation/local.py` points `network_path` there. Requires PyTorch (GPU optional) and will fall back to CPU when `device: "cpu"` or when CUDA is unavailable and `force_cpu_if_no_cuda: true`. Install `timm` (already listed in `requirements.txt`) because several ToMP backbones rely on it. When you need to trigger the original pytracking fine-tuning scripts, set `fine_tune.enabled: true` on the ToMP model and provide a `fine_tune.command` (string or list) that runs the desired training entrypoint (for example, a call into `ltr.run_training`). Use `{output_dir}` placeholder inside the command if you want to reuse the framework's train folder, and point `fine_tune.checkpoint` to the generated `.pth.tar` so the wrapper reloads it automatically.
- TaMOs integration also relies on the bundled `libs/pytracking` checkout. Grab checkpoints such as `tamos_resnet50.pth.tar` or `tamos_swin_base.pth.tar` from the [official Google Drive](https://drive.google.com/drive/folders/1i_hegsfhSd-7F6lhNAYw_Kx17TZUq_zy) and place them under `libs/pytracking/pytracking/networks/`. The wrapper defaults to the ResNet-50 preset, respects the same `device`/`force_cpu_if_no_cuda` flags, and supports overriding pytracking parameter attributes through the YAML config. If you want to hand off training to pytracking's tooling, mirror the ToMP instructions: enable `fine_tune`, provide a `command`, and point `checkpoint` to the produced weights so the wrapper switches to them for inference.
- MixFormerV2 integration example: copy `pipeline.mixformerv2.yaml`, set `dataset.root`, and download one of the official checkpoints (e.g. `mixformerv2_base.pth.tar`). Place it under `libs/MixFormerV2/models/` so the wrapper can resolve it. MixFormerV2 currently requires a CUDA-capable GPU because the upstream implementation runs preprocessing and inference on CUDA tensors. Each video still needs a `<video>.json` with the first-frame bounding box for initialisation。請另外安裝 `tensorboard`, `tensorboardX`, `easydict`, `lmdb`, `einops`（已列在 `requirements.txt`）以滿足官方專案在匯入環境設定時的依賴。由於官方 checkpoint 會序列化自訂類別，框架會在載入時強制 `torch.load(..., weights_only=False)`；請僅使用可信來源的權重檔。
-  建議將 `online_size` 保持在 `1`（範例管線已設定），避免原始 MixFormer 維護多個線上模板時在自訂資料集上產生維度不合的錯誤。

### Segmentation workflow

`tracking.segmentation` 提供 ROI 為核心的遮罩推論、可視化與指標彙整，所有結果皆寫入 `results/segmentation/<video>/`。在 pipeline 的 `segmentation` 區塊可以切換不同模型：

> Subject-level split 提醒：若 `dataset.root` 沒有以受試者資料夾分層，系統會嘗試從影片檔名開頭的連續數字推斷受試者 ID（例如 `001Rest.avi`、`001Rest post.avi` 會被視為 subject `001`）。建議資料夾仍以 `root/<subject>/<video>` 結構為主，或以數字前綴命名影片，才能確保 subject-level 分割正確運作。

- `model.name: "unet"`／`"unetpp"`／`"deeplabv3+"`：使用 `segmentation_models_pytorch`，支援 `encoder_name`、`encoder_weights` 等參數，可進行訓練；`deeplabv3+` 會建立 DeepLabV3+ decoder 以獲得較大的感受野。
- `model.name: "torchvision_fcn_resnet50"`：採用 torchvision 官方 FCN，屬推論專用 baseline。
- `model.name: "auto_mask"`（**新增**）：整合 `segmentation_annotator.auto_mask` 的弱監督遮罩流程（GrabCut + MGAC + 導向濾波），直接在 tracker bbox 周圍生成遮罩。此模式僅支援推論，框架會自動關閉訓練並沿用「最大連通元件 + 補洞」後處理，最終遮罩仍限制在 ROI 內。可透過 `model.params` 調整 `margin`、`num_iter`、`canny_low`／`canny_high`、`guided_radius`、`guided_eps` 等參數；若環境缺少 scikit-image，會自動退回無 MGAC 的變體。
- `model.name: "medsam"`：整合 [MedSAM](https://github.com/bowang-lab/MedSAM) / SAM 的推論與微調流程。此模式使用 segment-anything 套件，需要先安裝（已列入 `requirements.txt`）。請到官方 Google Drive 下載 `medsam_vit_b.pth`（[下載連結](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)）後放置到 `models/seg/medsam_vit_b.pth`（可自訂路徑）。
- `model.name: "nnunet"`（**新增**）：使用官方 nnU-Net v2 動態架構（透過 `libs/nnUNet/`）建立 ROI 分割模型。請先執行 `pip install -r requirements.txt` 以安裝 `-e ./libs/nnUNet` 及其依賴（包含 `dynamic-network-architectures`）。常用參數：
	- `plans_path`: 指向 nnUNet 規劃出的 `nnUNetPlans.json`，並透過 `configuration`（預設 `"2d"`）挑選其中一個設定。
	- `architecture_path`: 若無 plans，可直接提供自訂架構 JSON（格式同 `plans['architecture']`）。未提供時會回退到 2D PlainConvUNet 的預設拓撲。
	- `return_highres_only`: 預設 `true`，只回傳最高解析度 head 的 logits；若要使用其他深度監督輸出，可改成 `false` 取得最後一個 head（仍以單一張量餵入訓練迴圈）。
	UI 會在選擇 nnU-Net 後顯示額外欄位，可直接輸入 plans/architecture 路徑與 configuration 名稱；留下空白則使用預設架構。

  **基本推論模式**（預設，`train: false`）：
  ```yaml
  segmentation:
    model:
      name: medsam
      params:
		checkpoint: models/seg/medsam_vit_b.pth
        model_type: vit_b           # 與官方 checkpoint 對應
        use_box_prompt: true        # 使用追蹤 bbox 作為 SAM box prompt
        multimask_output: false     # 輸出單一遮罩
    train: false
  ```

  **微調模式**（`train: true`）：啟用後會凍結 image encoder 與 prompt encoder，只對 **mask decoder** 進行訓練：
  ```yaml
  segmentation:
    model:
      name: medsam
      params:
		checkpoint: models/seg/medsam_vit_b.pth
        model_type: vit_b
    train: true                     # 啟用微調
    epochs: 10                       # 訓練輪數
    lr: 0.0001                       # 學習率（建議較小）
    batch_size: 4                    # 依 GPU 記憶體調整
    target_size: [256, 256]          # ROI 裁切大小
  ```

  微調完成後權重會儲存至 `results/segmentation/segmentation_best.pt`（或 `_last.pt`），包含 mask decoder 的 state_dict，後續推論可透過 `inference_checkpoint` 指向此檔案加載。

典型設定：

```yaml
segmentation:
	enabled: true
	model:
		name: auto_mask
		params:
			margin: 0.15
			num_iter: 120
			canny_low: 25
			canny_high: 70
```

執行完會在 `metrics.json` 多一個 `fps` 欄位（每秒推論幀數），可用來比較不同遮罩流程的效能。

Install:
```bat
pip install -r requirements.txt
```

Notes:
- The TemplateMatching model is a simple baseline, mainly to validate the pipeline.
- Extend by registering new preproc or model classes using the registries.
- OCSort integration example: copy `pipeline.ocsort.yaml`, point `dataset.root` to your data, and run `python run_pipeline.py --config pipeline.ocsort.yaml`. Adjust the `params` section to swap detector weights or tweak OC-SORT hyper-parameters. The wrapper emits a single-object trajectory by sticking to the most consistent OC-SORT track (by ID/IoU, with score fallback).
- StrongSORT integration example: copy `pipeline.strongsort.yaml`, update `dataset.root`, and run `python run_pipeline.py --config pipeline.strongsort.yaml`. Parameters mirror the OC-SORT wrapper (detector tuning + tracker hyper-parameters) with additional appearance bins for the lightweight colour histogram features bundled here.
- StrongSORT++ integration example: copy `pipeline.strongsortpp.yaml`, update `dataset.root`, and run `python run_pipeline.py --config pipeline.strongsortpp.yaml`. 這個變體會在 StrongSORT 的結果上套用內建的軌跡補點與高斯平滑，可減少短暫漏追與抖動；可透過 `enable_gsi`、`gsi_interval`、`enable_gaussian_smoothing` 等參數調整行為。
- ToMP integration example: copy `pipeline.tomp.yaml`, set `dataset.root`, drop the pretrained `tomp50.pth.tar` (or `tomp101.pth.tar`) into `libs/pytracking/pytracking/networks/`, and run `python run_pipeline.py --config pipeline.tomp.yaml`. Make sure each video has a matching `<video>.json` with the first-frame bounding box because ToMP requires it to initialize.
- TaMOs integration example: copy `pipeline.tamos.yaml`, set `dataset.root`, and ensure at least one TaMOs checkpoint (e.g. `tamos_resnet50.pth.tar`) is in `libs/pytracking/pytracking/networks/`. Each video still needs a `<video>.json` with the first-frame bounding box; the wrapper will optionally recycle the previous box if the presence score dips below your threshold. For fine-tuning, set `fine_tune.enabled: true`, fill in the training command (for example call into `python -m ltr.run_training tracking tamos_resnet50`), and point `fine_tune.checkpoint` at the resulting `.pth.tar`—the wrapper will adopt it for subsequent predictions.
- NCC (FASTSpeckle), TaMOs, and MixFormerV2 now share an optional `low_confidence_reinit` block in their configs. Enable it to have the pipeline call the bundled YOLOv11 detector whenever the tracker’s confidence (presence score for TaMOs, MixFormerV2; feature-based ratio for NCC) drops below `threshold`. You can override detector parameters via `low_confidence_reinit.detector`, set a `min_interval` between refreshes, and require a minimum detection confidence. NCC also surfaces its confidence score in `FramePrediction`, making downstream gating or analytics easier.
- If you previously installed dependencies before the ToMP/TaMOs support landed, run `pip install timm>=0.9.12` to pull in the missing backbone dependency.

### Classification stage (optional)
- Place an `ann.txt` file in your dataset root. Each line is `<subject_id> <label>` (0 = healthy, 1 = diseased). Videos whose file name begins with the subject ID (e.g. `001Rest.mp4`, `001Grasp.mp4`) will inherit that label automatically.
- Add a `classification` block to your pipeline config to enable the stage after tracking evaluation (defaults to **video-level** classification to preserve sample count; switch to subject-level later by setting `target_level: "subject"`)。分類階段會自動沿用 pipeline 中第一個模型的預測結果；若同時排程多個 tracker，可手動在 Raw config 加入 `source_model` 指定來源（UI 不再提供選項，以免混淆）。

	```yaml
	classification:
		enabled: true
		label_file: "C:/dataset/ann.txt"   # optional, defaults to <dataset.root>/ann.txt
		target_level: "video"               # "video" (default) or "subject"
		feature_extractor:
			name: "texture_hybrid"           # 新增：結合動態 + 紋理特徵
			params:
				dynamic_params:
					aggregate_stats: ["mean", "std", "max"]
				texture_patch_size: 96          # bbox patch resize 尺寸（像素）
				texture_hist_bins: 16           # 灰階直方圖 bin 數
				max_texture_frames: 3           # 每支影片取樣幀數（平均切分）
		classifier:
			name: "random_forest"
			params:
				n_estimators: 300
				random_state: 42
	```
	- 想要測試不同 CNN backbone 對紋理特徵的影響時，可將 `feature_extractor.name` 換成 `"backbone_texture"`，支援 **MobileNetV2 / ResNet34 / DenseNet121 / EfficientNetB2**，並提供隨機投影降維以避免高維嵌入蓋過動態特徵：

		```yaml
		feature_extractor:
			name: "backbone_texture"
			params:
				backbone: "EfficientNetB2"      # 可改為 MobileNetV2 / ResNet34 / DenseNet121
				pretrained: false                # 無網路環境可設 false；若 GPU/網路可用可改 true
				reduction_method: "random_projection"
				reduced_dim: 32                  # 投影後的維度，預設 64
				pool_stats: ["mean", "std"]      # 聚合方式，支援 mean/std/min/max
				max_texture_frames: 2            # 每支影片取樣的幀數
				device: "cpu"                    # 自動回退；若要用 GPU 可改成 "cuda" 或指定編號
				zscore_patch: true               # 進骨幹前對 patch 做 z-score 標準化
				dynamic_params:
					aggregate_stats: ["mean", "std"]
		classifier:
			name: "random_forest"
			params:
				n_estimators: 300
				random_state: 42
		```
	- 隨機投影會在每支影片第一次執行時建立固定的投影矩陣（預設亂數種子 `1337`），同一個 `BackboneTextureFeatureExtractor` 物件之後的影片皆會沿用相同映射，確保特徵維度一致。若 `reduced_dim` 大於 backbone 原生維度，系統會自動停用降維直接輸出原始全局平均池化向量。
- Ground-truth trajectories from the training split are used to train the classifier. Inference is performed on tracker predictions from the held-out test split to mimic clinical deployment. Outputs (`classification/summary.json`, `predictions.json`, `classifier.pkl`, etc.) are written inside each experiment folder. Each prediction row records the `entity_id` (video when `target_level: "video"`, subject when `"subject"`), the owning `subject_id`, and the originating video path(s).
- Feature extractors and classifiers can be extended via the registries in `tracking/classification/feature_extractors.py` and `tracking/classification/classifiers.py`.
- 目前內建的分類器包含：
	- `random_forest`（預設）：支援 `n_estimators`、`max_depth`、`class_weight` 等常見參數。
	- `logistic_regression`：可設定 `penalty`（含 `elasticnet`）、`C`、`l1_ratio`、`solver`，預設採 L2 正則與 `lbfgs`。
	- `svm`：包裝 `sklearn.svm.SVC`，預設 `probability: true` 以輸出機率；可調整 `kernel`、`gamma`、`C` 等。
	- `xgboost`：需要安裝 `xgboost`，支援 `n_estimators`、`max_depth`、`learning_rate`、`scale_pos_weight`、`tree_method` 等設定。
	- `lightgbm`：需要安裝 `lightgbm`，支援 `num_leaves`、`learning_rate`、`n_estimators`、`objective` 等設定。

	範例（SVM + backbone 紋理）：
	```yaml
	classification:
		enabled: true
		feature_extractor:
			name: "backbone_texture"
			params:
				backbone: "MobileNetV2"
				reduced_dim: 16
				pool_stats: ["mean", "std"]
		classifier:
			name: "svm"
			params:
				C: 1.0
				kernel: "rbf"
				gamma: "scale"
				probability: true
	```

UI (optional):
```bat
python ui.py
```
Use the UI to load/edit a YAML/JSON config and run the pipeline with live logs.

### 🔍 Tracking Tools Workbench

若想快速檢視排程成果與信心診斷，使用新的「Tools Workbench」桌面應用：

```bat
python tools\tools_workbench.py
```

功能概覽：
- **排程結果瀏覽**：原本的 `schedule_results_viewer` 介面已整合為第一個分頁，可以排序、複製或匯出實驗指標。
- **信心診斷**：第二個分頁會掃描排程資料夾底下的 `test/predictions/*.json`，使用 `ConfidenceEstimator` 得到平滑化的信心值，並計算 P10/P05、最低值、連續低信心段落長度、平均 IoU/漂移等統計。選取任一列即可檢視 Top 8 最低信心影格及建議，並可透過「複製重點指標」按鈕一次複製所有實驗的關鍵信心指標以供後續分析。
- **更多工具**：第三個分頁為占位，後續可在此加入新的分析或除錯功能。

信心診斷的低信心門檻預設為 0.6，可在分頁右上角調整並重新分析。若原始追蹤分數長期鎖在 1.0、漂移組件偏低或低信心段落過長，建議啟用/調整 `low_confidence_reinit` 或檢查初始標註品質。

### 排程多組實驗
- 左側面板底部新增「排程隊列」，可以用「加入排程」把目前的設定快照加入待執行清單。
- 任何時候都可以在 Builder 區塊的「實驗名稱」欄位自訂實驗名稱；結果資料夾會沿用這個名稱，留空則依前處理 / 模型自動產生。
- 點「執行排程」後，UI 會依序執行每一組設定，並在進度列上顯示第幾項正在跑。
- 若隊列包含兩組以上的實驗，框架會在 `results/` 底下自動建立一個 `yyyy-mm-dd_HH-MM-SS_schedule_<N>exp/` 目錄，所有該批次的實驗結果都會放在裡面，方便回溯。
- 排程執行中會鎖定所有排程與執行相關按鈕，等整批完成或失敗後才釋放。
- 若排程途中失敗，剩餘的項目會保留在列表中，修正問題後可直接再次執行。

## 🆕 UI 重構（Always Bi-Directional）

新版 `ui.py` 進一步簡化：不再有「高級模式」或「存檔」按鈕，預設即為雙向同步。

### 設計重點
1. 即時雙向：
	- Builder（左）：任何改動立即產生並覆寫右側 Raw YAML/JSON。
	- Raw（右）：直接編輯，停止輸入 ~0.8s 後自動解析；成功則更新 Builder；失敗不覆蓋，欄位紅框並在 Logs 顯示錯誤。
2. 單一來源策略：邏輯上以 Builder 為主；Raw 僅在成功解析後才回填。
3. 無需手動存檔：若要保留設定，直接複製右側 Raw YAML 內容存到檔案即可（或使用外部編輯器）。
4. 參數即時保存：表單元件 value 改變 / editingFinished 即寫回內部狀態。
5. 防止滑鼠滾輪誤觸：所有數值類 SpinBox / DoubleSpinBox 取消 wheel 事件（需刻意點選後用鍵盤或箭頭調整）。
6. 狀態列：顯示當前同步方向（Builder → Raw / Raw → Builder / 解析失敗）。
7. 執行前檢查：若 Raw 尚在「等待解析」或處於錯誤紅框，禁止執行。

### 基本流程
```
選 Dataset Root → 選擇/排序 Preproc → 編輯各參數 → 選 Model + 參數 → 選 Evaluator / Visualization → (必要時改 Raw) → 執行
```

若需要分類評估，可在 Builder 左側勾選「Classification (可選)」，設定 `ann.txt` 路徑、`target_level`（預設 `video`）以及特徵擷取器 / 分類器參數；執行排程或單次實驗時會一併產出 `classification/` 結果。`texture_hybrid` 會自動裁切 tracker/GT 的 bbox patch 計算灰階統計、梯度與直方圖，再與動態特徵拼接；若僅需動態特徵，可改回 `basic`。

### Raw 編輯行為
| 狀態 | 描述 |
|------|------|
| 等待解析 | 使用者剛輸入，計時器尚未觸發；狀態列顯示「等待解析…」 |
| 解析失敗 | 紅框 + Logs 錯誤訊息；Builder 保持舊值，不被覆寫 |
| 解析成功 | Builder 被更新 → 重新序列化回美化後 YAML |

### 為什麼移除「存檔」按鈕？
- 避免使用者以為需要先存檔才執行；執行永遠使用當下 UI 狀態建構的 config。
- Raw 區域已是最終文字表示；直接複製即可分享或另存。

### 匯入現有設定檔
1. 上方輸入或瀏覽選擇 `*.yaml` / `*.json`
2. 點「載入設定檔」
3. Raw 顯示內容 → 自動解析回 Builder → 可繼續圖形化調整

### 安全策略
- Raw 解析失敗時永不覆蓋 Builder，避免破壞可執行狀態。
- Builder 修改會覆寫 Raw，除非 Raw 尚未成功解析（使用者正在編輯）。

### 複製設定
直接在 Raw 全選複製，另存為檔案（UTF-8）。

### 已解決的舊痛點
- 不再需要「產生 YAML / 同步 / 儲存參數」的多餘動作。
- 滾輪誤觸數值問題消失。
- 高級模式心智模型移除，行為更統一。

---
整體目標：把使用者注意力集中在「調整策略 + 執行」，而不是 GUI 操作成本。


How splitting works:
- Provide only train/test ratios (defaults to [0.8, 0.2]).
- If k_fold > 1, k-fold validation is performed within the training split for model selection/validation.
- After k-fold, the chosen model is trained on the full training split and evaluated on the test split.
