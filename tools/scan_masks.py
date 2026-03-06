"""Scan all mask PNGs in dataset/merged and report area statistics + outliers."""
import os, cv2, numpy as np
from pathlib import Path

dataset_root = r"C:\Users\User\Desktop\code\Traking\dataset\merged"
mask_files = list(Path(dataset_root).rglob("*.png"))

areas = []
records = []
for p in mask_files:
    if p.stat().st_size == 0:
        continue
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    area = int(np.count_nonzero(m > 0))
    areas.append(area)
    records.append({"path": str(p), "area": area})

areas_np = np.array(areas)
mean = float(areas_np.mean())
std  = float(areas_np.std())
low_thresh  = mean - 3 * std
high_thresh = mean + 3 * std

print(f"Total masks : {len(areas)}")
print(f"Mean area   : {mean:.1f} px")
print(f"Std area    : {std:.1f} px")
print(f"Min area    : {int(areas_np.min())} px")
print(f"Max area    : {int(areas_np.max())} px")
print(f"Threshold low  (mean-3σ) : {low_thresh:.1f} px")
print(f"Threshold high (mean+3σ) : {high_thresh:.1f} px")
print()

outliers = [r for r in records if r["area"] < low_thresh or r["area"] > high_thresh]
print(f"Outliers (>3σ from mean): {len(outliers)}")
for r in sorted(outliers, key=lambda x: x["area"]):
    tag = "LOW " if r["area"] < low_thresh else "HIGH"
    print(f"  [{tag}] {r['area']:7d} px  {r['path']}")
