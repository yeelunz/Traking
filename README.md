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

Notes on trackers:
- CSRT tracker requires `opencv-contrib-python` (cv2.legacy). If you want to use CSRT, uninstall opencv-python and install the contrib build instead.
- OC-SORT wrapper relies on Ultralytics YOLOv11 for detections (`ultralytics` package) and the `ocsort` PyPI package. Both are listed in `requirements.txt`.
- StrongSORT wrapper reuses the cloned `libs/StrongSORT` repo plus Ultralytics YOLOv11; make sure `ultralytics`, `torch`, and OpenCV are installed (see `requirements.txt`).
- ToMP integration uses the bundled `libs/pytracking` checkout. Download the pretrained weights (e.g. `tomp50.pth.tar`) into `libs/pytracking/pytracking/networks/` and ensure `pytracking/pytracking/evaluation/local.py` points `network_path` there. Requires PyTorch (GPU optional) and will fall back to CPU when `device: "cpu"` or when CUDA is unavailable and `force_cpu_if_no_cuda: true`. Install `timm` (already listed in `requirements.txt`) because several ToMP backbones rely on it. When you need to trigger the original pytracking fine-tuning scripts, set `fine_tune.enabled: true` on the ToMP model and provide a `fine_tune.command` (string or list) that runs the desired training entrypoint (for example, a call into `ltr.run_training`). Use `{output_dir}` placeholder inside the command if you want to reuse the framework's train folder, and point `fine_tune.checkpoint` to the generated `.pth.tar` so the wrapper reloads it automatically.
- TaMOs integration also relies on the bundled `libs/pytracking` checkout. Grab checkpoints such as `tamos_resnet50.pth.tar` or `tamos_swin_base.pth.tar` from the [official Google Drive](https://drive.google.com/drive/folders/1i_hegsfhSd-7F6lhNAYw_Kx17TZUq_zy) and place them under `libs/pytracking/pytracking/networks/`. The wrapper defaults to the ResNet-50 preset, respects the same `device`/`force_cpu_if_no_cuda` flags, and supports overriding pytracking parameter attributes through the YAML config. If you want to hand off training to pytracking's tooling, mirror the ToMP instructions: enable `fine_tune`, provide a `command`, and point `checkpoint` to the produced weights so the wrapper switches to them for inference.
- MixFormerV2 integration example: copy `pipeline.mixformerv2.yaml`, set `dataset.root`, and download one of the official checkpoints (e.g. `mixformerv2_base.pth.tar`). Place it under `libs/MixFormerV2/models/` so the wrapper can resolve it. MixFormerV2 currently requires a CUDA-capable GPU because the upstream implementation runs preprocessing and inference on CUDA tensors. Each video still needs a `<video>.json` with the first-frame bounding box for initialisation。請另外安裝 `tensorboard`, `tensorboardX`, `easydict`, `lmdb`, `einops`（已列在 `requirements.txt`）以滿足官方專案在匯入環境設定時的依賴。由於官方 checkpoint 會序列化自訂類別，框架會在載入時強制 `torch.load(..., weights_only=False)`；請僅使用可信來源的權重檔。
-  建議將 `online_size` 保持在 `1`（範例管線已設定），避免原始 MixFormer 維護多個線上模板時在自訂資料集上產生維度不合的錯誤。

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

UI (optional):
```bat
python ui.py
```
Use the UI to load/edit a YAML/JSON config and run the pipeline with live logs.

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
