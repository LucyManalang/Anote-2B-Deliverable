"""
upload(dataset_name: str, data_path: str, split: str) -> dict

Converts a unified JSONL manifest into directory & label structures expected by common detectors.
- Accepts the JSONL manifest rows with fields:
  "image_path": str
  "labels": [str]
  "bboxes": [[x1,y1,x2,y2,"class"], ...]
- Outputs:
  - saves a copy of the manifest per-split under datasets/{dataset_name}/{split}/manifest.jsonl
  - optionally emits YOLO text files under datasets/{dataset_name}/{split}/labels/
  - emits a COCO-style JSON file for Faster R-CNN if requested (basic)
"""
from pathlib import Path
import os
import json
from typing import Dict, List
from .utils import load_jsonl, save_jsonl, ensure_dir, coco_box_from_xyxy


def upload(dataset_name: str, data_path: str, split: str, out_root: str = "datasets") -> Dict:
    manifest = load_jsonl(data_path)
    base = Path(out_root) / dataset_name / split
    images_dir = base / "images"
    labels_dir = base / "labels"
    coco_out = base / "coco.json"
    ensure_dir(images_dir.as_posix())
    ensure_dir(labels_dir.as_posix())

    # Save manifest copy
    manifest_out = base / "manifest.jsonl"
    save_jsonl(manifest, manifest_out.as_posix())

    # Prepare class mapping (gather unique class names)
    classes = []
    for row in manifest:
        for c in row.get("labels", []):
            if c not in classes:
                classes.append(c)

    class_to_id = {c: i for i, c in enumerate(classes)}

    # Emit YOLO label files and basic image copy instructions (assume images already in place)
    for row in manifest:
        image_path = Path(row["image_path"])
        img_name = image_path.name
        label_name = labels_dir / f"{img_name}.txt"
        # If bboxes provided, convert to YOLO normalized format (x_center, y_center, w, h) normalized
        w = row.get("width", None)
        h = row.get("height", None)
        lines = []
        for bbox in row.get("bboxes", []):
            x1, y1, x2, y2, cls = bbox
            if w is None or h is None:
                # If width/height not provided, we cannot normalize - skip YOLO file
                continue
            x_center = ((x1 + x2) / 2.0) / w
            y_center = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{class_to_id[cls]} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            with open(label_name, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # Create a basic COCO-like structure (instances) for convenience
    coco = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1
    for i, row in enumerate(manifest, start=1):
        image_id = i
        filename = Path(row["image_path"]).name
        coco["images"].append({"id": image_id, "file_name": filename})
        for bbox in row.get("bboxes", []):
            x1, y1, x2, y2, cls = bbox
            coco_bbox = coco_box_from_xyxy(x1, y1, x2, y2)
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_to_id[cls],
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    for cls, cid in class_to_id.items():
        coco["categories"].append({"id": cid, "name": cls})

    with open(coco_out, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return {
        "status": "ok",
        "dataset": dataset_name,
        "split": split,
        "manifest_out": manifest_out.as_posix(),
        "labels": labels_dir.as_posix(),
        "coco": coco_out.as_posix(),
        "classes": classes,
        "count": len(manifest)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    print(upload(args.dataset_name, args.data_path, args.split))
