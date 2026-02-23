# Segmentation Annotator

本資料夾提供一套針對神經追蹤資料的遮罩標註工具，主打即時筆刷編輯、單幀單遮罩策略與自動遮罩輔助，並輸出含 bbox + mask + motion 的 COCO-VID 相容標註。

以下內容已依程式碼現況重新檢核並整理。

## 功能摘要

- **遮罩編輯**：左鍵繪製、右鍵橡皮擦、滾輪調整筆刷，支援 `Ctrl+Z`。
- **遮罩清理**：任何新增/更新遮罩都會自動只保留面積最大的連通區。
- **單一遮罩策略**：每個影格只保留一個遮罩並沿用單一 `track_id`，避免同幀多個 mask 造成 downstream 模型混淆。
- **顯示控制**：透明度、色票、背景亮度、縮放與縮圖導覽。
- **自動遮罩**：YOLO 偵測 → ROI 反射擴張 → GrabCut 初始化 → MGAC(若可用) → 形態學與導向濾波 → 邊界清理與最大連通區保留。
- **擴充資訊**：輸出 JSON 同時保留 `bbox`、`mask_path`、`metadata`(質心/周長/等效直徑) 與 `motion`(dx/dy/距離)。

## 安裝需求

本工具可獨立安裝，請使用 [segmentation_annotator/requirements.txt](segmentation_annotator/requirements.txt)。

```cmd
pip install -r segmentation_annotator/requirements.txt
```

> 註：ultralytics 初次使用會自動下載模型定義/權重，請確保網路可用。

## 啟動方式

```cmd
python -m segmentation_annotator.main
```

## 輸入資料夾結構（選取的資料夾）

使用「檔案 → 開啟資料夾」選取包含影片的資料夾，支援副檔名：`.mp4`, `.avi`, `.mov`, `.mkv`, `.mpg`, `.mpeg`, `.wmv`。

程式會自動確保該資料夾有 `labels.txt`：

```
<dataset_root>/
   ├─ labels.txt                # 類別清單，若不存在會自動建立（預設 median_nerve）
   ├─ video_001.mp4
   ├─ video_002.avi
   └─ ...
```

若同資料夾內已存在 `<video_stem>.json` 與 `seg_masks/`，啟動時會自動載入既有遮罩。

## 輸出檔案結構（自動儲存）

每次繪製、刪除遮罩或套用自動遮罩都會觸發自動儲存；輸出會寫在「開啟的資料夾」內（即 `dataset_root`）：

```
<dataset_root>/
   ├─ labels.txt
   ├─ video_001.mp4
   ├─ video_001.json            # COCO-VID 相容標註（自動更新）
   ├─ seg_masks/
   │   └─ video_001/
   │       ├─ frame_00000001.png
   │       ├─ frame_00000002.png
   │       └─ ...
   └─ ...
```

### 遮罩檔命名規則

- 單遮罩模式（預設）：`frame_00000001.png`
- 若同幀存在多個 track（雖然目前 UI 會強制只留一個）：`frame_00000001_track_0001.png`

## JSON 規格（COCO-VID 相容）

輸出 JSON 由 [segmentation_annotator/data.py](segmentation_annotator/data.py) 的 `SegmentationProject.export_dataset()` 產生。

### Top-level

```text
info                # 資訊與 mask 根目錄
videos              # 影片資訊
images              # 影格資訊（含 frame_index）
annotations         # bbox + mask + metadata + motion
categories          # 類別清單
tracks              # 依 track 彙整的運動序列
```

### info

- `description`: 固定描述文字
- `annotation_type`: `bbox+mask`
- `mask_root`: 預設 `seg_masks`

### videos[]

- `id`: 影片 ID（固定 1）
- `name`: 影片檔名
- `height`, `width`, `fps`, `total_frames`

### images[]

- `id`: 影格 ID（從 1 起算）
- `video_id`: 固定 1
- `frame_index`: 0-based 影格序號
- `file_name`: `<video_stem>/<frame+1:08d>.jpg`（僅作為索引，程式不輸出影格影像）
- `height`, `width`

### annotations[]

- `id`, `image_id`, `category_id`, `track_id`
- `bbox`: `[x, y, w, h]`（由遮罩自動計算）
- `area`: mask 像素面積
- `iscrowd`: 固定 0
- `mask_path`: 遮罩相對路徑（相對 `mask_root`）
- `metadata`:
   - `centroid`: `[cx, cy]`（像素座標）
   - `perimeter_px`
   - `equivalent_diameter_px`
- `motion`:
   - `dx`, `dy`, `distance`（與同 track 前一筆的質心差；首筆為 0）

### categories[]

- `id`, `name`

### tracks[]

- `track_id`, `category_id`
- `samples[]`：
   - `frame_index`
   - `centroid`
   - `displacement`: `[dx, dy]`
   - `distance`
- `total_path_length`
- `average_speed_per_frame`

## UI 區塊說明

### 左側（畫布與播放控制）

- **畫布**：影格與遮罩疊合顯示，支援縮放/平移與縮圖導覽。
- **上一幀/下一幀**：切換影格；也可用上下鍵。
- **進度滑桿 + 影格輸入框**：快速定位到指定幀。
- **清除遮罩**：清除當前幀所有遮罩。
- **YOLO+ChanVese**：執行自動遮罩（實作為 YOLO + GrabCut + MGAC）。

### 右側（狀態與工具列）

- **影片清單**：顯示資料夾內所有影片與已標註幀數。
- **類別清單**：來自 `labels.txt`。
- **筆刷設定**：滑桿/數值框/復原按鈕，並顯示筆刷半徑。
- **遮罩顯示**：透明度與色票切換。
- **背景亮度**：便於在暗/亮背景下檢視。
- **檢視**：縮放百分比與重設檢視。
- **已標註幀**：可快速跳轉。
- **狀態列**：顯示自動儲存或錯誤訊息。

## 自動遮罩權重搜尋路徑

啟動自動遮罩時會依序尋找 `best.pt`：

1. 專案根目錄的 `best.pt`
2. 目前工作目錄的 `best.pt`
3. `dataset_root/best.pt`
4. 影片所在資料夾的 `best.pt`

找不到權重時會提示並中止自動遮罩。

## 快捷鍵

- 左鍵：筆刷填色
- 右鍵：橡皮擦
- 滾輪：調整筆刷大小
- `Ctrl+滾輪`：縮放畫布
- 中鍵拖曳：平移畫面
- 空白鍵：暫時加強遮罩預覽
- `Ctrl+Z`：復原
- 上/下鍵：切換影格

---

若要在其他流程中讀取遮罩，可用 `mask_path` 還原二值影像，再透過 `MaskMetadata.from_mask()` 取得 bbox、面積與質心等特徵。歡迎依實際流程擴充。 
