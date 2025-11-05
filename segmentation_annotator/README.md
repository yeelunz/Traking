# Segmentation Annotator

這個資料夾提供一個專門針對神經追蹤資料的遮罩標註工具，可與原本的 Bounding Box 標註器並行使用。它支援筆刷與橡皮擦操作、改良後的 YOLO + Chan–Vese 自動遮罩輔助，以及完整的 COCO-VID 匯出。

## 主要功能

- **遮罩編輯**：滑鼠左鍵繪製、右鍵橡皮擦，滾輪調整筆刷半徑，支援 Ctrl+Z 復原。
- **遮罩清理**：每次儲存會自動保留面積最大的連通區，避免橡皮擦遺留噪聲碎片。
- **單一遮罩策略**：每個影格只保留一個遮罩並沿用同一個 track ID，避免同幀多個 mask 造成 downstream 模型混淆。
- **遮罩視覺化控制**：提供透明度滑桿與 5 種色票，可即時切換並微調遮罩顯示強度。
- **筆刷控制**：UI 內建筆刷大小滑桿與數值框，並顯示滑鼠懸停時的筆刷/橡皮擦覆蓋範圍。
- **遮罩預覽**：按下空白鍵可暫時加強遮罩顏色以利檢視。
- **自動建議**：整合現有的 YOLO 權重 (`best.pt`) 進行檢測，取單一高分偵測結果後以鏡像 padding 擴張 ROI、橢圓初始化，再透過局部 Chan–Vese / MorphACWE + MGAC、導向濾波與形態學後處理生成乾淨遮罩。
- **擴充資料結構**：匯出的 JSON 同時保留原本的 bbox、mask 路徑以及推算出的質心、位移距離等運動量指標。
- **與舊流程相容**：匯出的資料可直接轉為 COCO-VID 標註；每個標註含有 `bbox` 與遮罩相對路徑，tracker/detector 仍可取得 bounding box，同時額外的 `motion` 欄位描述神經在連續幀間的位移。

## 安裝額外依賴

為了方便把這個工具獨立打包，我們另外準備了最精簡的需求檔 `segmentation_annotator/requirements.txt`，僅包含：

- `numpy`
- `opencv-python`
- `PySide6`
- `scikit-image`
- `ultralytics>=8.3.0`

醫師的工作站只要執行：

```cmd
pip install -r segmentation_annotator/requirements.txt
```

即可具備執行所需的全部元件。YOLOv11 推論預設以 CPU 進行，無需安裝 CUDA；如果現場有 GPU 需要加速，再在 `YOLO+ChanVese` 按鈕前設定 `AutoMaskGenerator(device="cuda:0")` 即可。

> 注意：ultralytics 會在第一次使用時自動下載模型定義或權重，請確保網路環境允許。

## 使用方式

1. 執行 `python -m segmentation_annotator.main`。
2. 點「檔案 → 開啟資料夾」，選擇含有影片的資料夾；右側清單會列出該資料夾中的所有支援格式 (`.mp4`, `.avi`, `.mov`, `.mkv`, ...)，點選即可切換影片。
3. 若資料夾內沒有 `labels.txt`，程式會自動建立並帶入預設類別 `median_nerve`。
4. 每次繪製、套用自動遮罩或刪除遮罩時，系統會立即自動儲存結果；生成的 JSON 與遮罩檔會放在同一資料夾下（`seg_masks/` 子資料夾），無需額外匯出按鈕。
   - 右下方可調整遮罩透明度/顏色，左側筆刷列提供滑桿與數值框直接調整筆刷大小，滑鼠懸停時會顯示筆刷半徑預覽。
   - 透過「說明 → 快捷鍵說明」可隨時查看滑鼠與鍵盤操作提醒。
5. 如需 YOLO + Chan–Vese 輔助，設定好 `best.pt` 權重的路徑（可放在與影片相同資料夾），按下 `YOLO+ChanVese` 按鈕即可產生候選遮罩；演算法會自動擴張 ROI、鏡像 padding、以局部能量模型處理亮度不均並移除貼邊假影，只留下面積最大的遮罩。
6. 完成後資料夾會包含：
   - `seg_masks/<video_name>/track_xxxx/frame_00000001.png` 形式的遮罩影像
   - `<video_name>.json`：COCO-VID 格式的標註檔，含 `bbox`、`mask_path`、`metadata` (質心、周長、等效直徑)、`motion` (dx/dy/距離)

## 分割格式說明

- `annotations[].bbox`：由遮罩自動推算的外接框，維持舊有流程所需的輸入。
- `annotations[].mask_path`：遮罩相對路徑，讀取後可重建 binary mask。
- `annotations[].metadata`: 包含 `centroid`、`perimeter_px`、`equivalent_diameter_px`。
- `annotations[].motion`: 以 centroid 差值計算的 `(dx, dy, distance)`，提供神經移動量的衡量指標。
- `tracks[]`: 每個 track 的運動時間序列彙整，便於後續分析。

## 與現有流程整合

- `segmentation_annotator/data.py` 中的 `SegmentationProject.export_dataset` 會輸出與原 pipeline 相容的 JSON，既可在 `tracking` 模組中載入 bounding box，也能視需求透過 `mask_path` 取得遮罩。不需依賴 `mmtracking` 套件或任何外部 C++ 擴充套件。
- 未來若需於 `tracking` 匯入遮罩，可透過 `mask_path` 重新讀入二值影像，再透過 `MaskMetadata.from_mask` 提供的資訊取得 bbox / 面積 / 質心等特徵。

歡迎依照實際工作流程再行擴充使用者介面或資料格式。
