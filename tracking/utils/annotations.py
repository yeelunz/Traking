from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Tuple


def load_coco_vid(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build frame -> list[bbox]
        frames: Dict[int, List[Tuple[float, float, float, float]]] = {}
        frame_annotations: Dict[int, List[Dict[str, Any]]] = {}
    # Map image_id -> frame_index
    img_to_frame = {}
    for img in data.get("images", []):
        # Prefer 'frame_index' if present, else try 'id' order
        fi = img.get("frame_index")
        if fi is None:
            # fallback: assume consecutive
            fi = img.get("id", 0) - 1
        img_to_frame[img["id"]] = fi
        # make sure mapping has entries for this frame
        frames.setdefault(fi, [])
        frame_annotations.setdefault(fi, [])
    # if there were no images, ensure variables exist
    if data.get("images") is None or not data.get("images"):
        fi = 0
        frames.setdefault(fi, [])
        frame_annotations.setdefault(fi, [])
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if img_id is None or bbox is None:
            continue
        fi = img_to_frame.get(img_id)
        if fi is None:
            continue
        frames.setdefault(fi, []).append(tuple(bbox))
        frame_annotations.setdefault(fi, []).append(
            {
                "bbox": tuple(bbox) if isinstance(bbox, (list, tuple)) else bbox,
                "category_id": ann.get("category_id"),
                "track_id": ann.get("track_id"),
                "area": ann.get("area"),
                "metadata": ann.get("metadata"),
                "motion": ann.get("motion"),
                "mask_path": ann.get("mask_path"),
                "iscrowd": ann.get("iscrowd", 0),
                "id": ann.get("id"),
            }
        )
    # Categories
    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
    return {
            "frames": frames,  # Dict[int, List[(x,y,w,h)]]
            "frame_annotations": frame_annotations,
        "categories": cat_map,
            "mask_root": data.get("info", {}).get("mask_root"),
        "raw": data,
    }
