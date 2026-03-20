# Texture Pretraining (Stage-1)

## 1) 資料格式

支援 `csv` / `json` / `jsonl`。

- `csv` 欄位至少需包含：`image_path`, `label`
- `json` 範例：

```json
[
  {"image_path": "roi/img_0001.png", "label": 0},
  {"image_path": "roi/img_0002.png", "label": 1}
]
```

## 2) 執行

```bash
python -m tracking.texture_pretrain.train_texture --config schedules/texture_pretrain_example.yaml
```

主流程（classification stage）也支援自動觸發：

- 若特徵提取器 `texture_mode=pretrain`
- 且 `texture_pretrain_ckpt` 未提供/無效

主流程會自動執行 Stage-1（含 cache），完成後回填 checkpoint 給 Stage-2 使用。

預設會使用官方 pretrained 權重做 Stage-1 fine-tune（`pretrained_imagenet: true`）。
若 `force_official_pretrained: true` 且你把 `pretrained_imagenet` 關掉，程式會直接報錯，避免誤用隨機初始化。

## 3) 輸出 checkpoint

- best: `texture_pretrain.save_path`
- last: 在同路徑自動加上 `.last` 後綴

payload 會至少包含：

- `backbone_state_dict`
- `backbone_name`
- `num_classes`
- `input_size`
- `history`

當 `save_backbone_only: false` 時，額外包含：

- `model_state_dict`
- `head_state_dict`

## 4) Stage-2 使用

在 `feature_extractor` 或 `classifier` 設定：

```yaml
texture_mode: pretrain
texture_backbone: convnext_tiny
texture_pretrain_ckpt: ckpt/convnext_tex_pretrain.pth
```

`texture_mode=pretrain` 會只載入 backbone，並自動 frozen（`requires_grad=False`）。

## 5) Cache（跨實驗重用）

可啟用：

```yaml
cache_enabled: true
cache_dir: ckpt/texture_pretrain_cache
```

當以下條件都相同時，會直接命中 cache，不再重跑 Stage-1 訓練：

- backbone / num_classes / input_size
- head_type / hidden_dim / dropout
- pretrained_imagenet
- train_manifest 與 val_manifest 的內容雜湊（SHA-256，與路徑無關）

命中時會把 cache checkpoint 複製到 `save_path`，可直接接 Stage-2 使用。
