import json
import os

pred_path = r'results\2026-03-20_22-41-11_schedule_6exp\2026-03-22_03-02-34_mednext_clahe_global_tscv3pro_multirocket_concat_loso_002\test\detection\predictions_by_video\002\Rest post\YOLOv11.json'
if os.path.exists(pred_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    print(f'Total predictions: {len(preds)}')
else:
    print(f'File not found: {pred_path}')
