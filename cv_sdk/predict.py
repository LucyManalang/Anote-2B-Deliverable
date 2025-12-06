"""
predict(model_type: str, test_data: str, labels: list[str], model_id: str | None = None, confidence_threshold: float = 0.5) -> list[dict]

Outputs a list of prediction dicts per image:
{
  "image_id": "...",
  "boxes": [[x1,y1,x2,y2], ...],
  "labels": [...],
  "confidence": [...]
}
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
from .utils import load_jsonl
import os

def predict(model_type: str, test_data: str, labels: List[str], model_id: Optional[str] = None, confidence_threshold: float = 0.5) -> List[Dict]:
    model_type = model_type.lower()
    if model_type == "yolov8":
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("Ultralytics is required for prediction: pip install ultralytics")
        # Determine model to use
        if model_id:
            model = YOLO(model_id)
        else:
            # Use latest run if present
            runs = list(Path("runs/detect").glob("*"))
            if runs:
                latest = sorted(runs, key=os.path.getmtime)[-1]
                candidate = latest / "weights" / "best.pt"
                if candidate.exists():
                    model = YOLO(candidate.as_posix())
                else:
                    raise RuntimeError("No model_id provided and no trained weights found.")
            else:
                raise RuntimeError("No model_id provided and no runs found.")
        # test_data can be image file, folder, or JSONL manifest
        if Path(test_data).is_file() and test_data.lower().endswith(".jsonl"):
            manifest = load_jsonl(test_data)
            image_paths = [r["image_path"] for r in manifest]
        elif Path(test_data).is_dir():
            image_paths = [str(p) for p in Path(test_data).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        else:
            image_paths = [test_data]

        results = []
        for img in image_paths:
            res = model(img)[0]  # ultralytics returns a Results object, take first
            boxes = []
            labels_out = []
            scores = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                    conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                    cls = int(b.cls[0]) if hasattr(b, "cls") else None
                    lbl = labels[cls] if cls is not None and cls < len(labels) else str(cls)
                    if conf < confidence_threshold:
                        continue
                    boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    labels_out.append(lbl)
                    scores.append(conf)
            results.append({
                "image_id": img,
                "boxes": boxes,
                "labels": labels_out,
                "confidence": scores
            })
        return results
    else:
        raise NotImplementedError("Prediction for model_type not implemented in minimal SDK.")
