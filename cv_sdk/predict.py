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
import torch
from PIL import Image

def _predict_faster_rcnn(test_data, labels, model_id, confidence_threshold):
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torchvision.transforms as T
    except ImportError:
        raise RuntimeError("torchvision is required for Faster R-CNN prediction")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load checkpoint
    if not model_id or not Path(model_id).exists():
        # Try to find latest
        out_dir = Path("runs/detect/faster_rcnn")
        if out_dir.exists():
            model_id = str(out_dir / "model.pth")
    
    if not model_id or not Path(model_id).exists():
        raise RuntimeError("No model_id provided and no trained weights found for Faster R-CNN.")

    checkpoint = torch.load(model_id, map_location=device)
    
    # Reconstruct model
    # We need to know num_classes. Checkpoint might have it.
    saved_classes = checkpoint.get('classes', labels)
    num_classes = len(saved_classes) + 1 # +1 background
    
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare data
    if Path(test_data).is_file() and test_data.lower().endswith(".jsonl"):
        manifest = load_jsonl(test_data)
        image_paths = [r["image_path"] for r in manifest]
    elif Path(test_data).is_dir():
        image_paths = [str(p) for p in Path(test_data).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    else:
        image_paths = [test_data]

    transform = T.Compose([T.ToTensor()])
    results = []

    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).to(device)
                predictions = model([img_tensor])[0]
                
                boxes = []
                labels_out = []
                scores = []
                
                pred_boxes = predictions['boxes'].cpu().numpy()
                pred_labels = predictions['labels'].cpu().numpy()
                pred_scores = predictions['scores'].cpu().numpy()
                
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score < confidence_threshold:
                        continue
                    
                    # label is 1-based index into saved_classes (0 is background)
                    class_name = saved_classes[label-1] if 0 < label <= len(saved_classes) else str(label)
                    
                    boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                    labels_out.append(class_name)
                    scores.append(float(score))
                
                results.append({
                    "image_id": img_path,
                    "boxes": boxes,
                    "labels": labels_out,
                    "confidence": scores
                })
            except Exception as e:
                print(f"Error predicting on {img_path}: {e}")

    return results

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
    elif model_type == "faster_rcnn":
        return _predict_faster_rcnn(test_data, labels, model_id, confidence_threshold)
    else:
        raise NotImplementedError("Prediction for model_type not implemented in minimal SDK.")
