from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Tuple


def load_coco_vid(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build frame -> list[bbox]
    frames: Dict[int, List[Tuple[float, float, float, float]]] = {}
    # Map image_id -> frame_index
    img_to_frame = {}
    for img in data.get("images", []):
        # Prefer 'frame_index' if present, else try 'id' order
        fi = img.get("frame_index")
        if fi is None:
            # fallback: assume consecutive
            fi = img.get("id", 0) - 1
        img_to_frame[img["id"]] = fi
        frames.setdefault(fi, [])
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if img_id is None or bbox is None:
            continue
        fi = img_to_frame.get(img_id)
        if fi is None:
            continue
        frames.setdefault(fi, []).append(tuple(bbox))
    # Categories
    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    return {
        "frames": frames,  # Dict[int, List[(x,y,w,h)]]
        "categories": cat_map,
        "raw": data,
    }
