import json
import os

trace_path = r'results\2026-03-20_22-41-11_schedule_6exp\2026-03-22_03-02-34_mednext_clahe_global_tscv3pro_multirocket_concat_loso_002\test\segmentation\predictions\mednext\YOLOv11\002\Rest post\roi_trace.json'
if os.path.exists(trace_path):
    with open(trace_path, 'r', encoding='utf-8') as f:
        trace = json.load(f)
    print(f'Total frames in trace: {len(trace)}')
    masked = sum(1 for v in trace.values() if isinstance(v, dict) and v.get('mask_path'))
    print(f'Frames with mask_path: {masked}')
else:
    print(f'File not found: {trace_path}')
