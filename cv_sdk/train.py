"""
train(task_type: int, model_type: str, train_dataset: str, validation_dataset: str | None = None) -> str

This implementation uses Ultralytics YOLOv8 if available. If not installed,
it raises a helpful error. For Faster R-CNN you can integrate torchvision training later.

Returns a model_id (path to saved weights).
"""
from typing import Optional
import os
from pathlib import Path

def _ensure_ultralytics():
    try:
        import ultralytics
        return ultralytics
    except Exception as e:
        raise RuntimeError(
            "Ultralytics YOLO is required for training. Install with: pip install ultralytics"
        )


def train(task_type: int, model_type: str, train_dataset: str, validation_dataset: Optional[str] = None, epochs: int = 20, imgsz: int = 640, batch: int = 8) -> str:
    """
    Trains a detection model and returns model_id (path to weights).

    model_type: "yolov8" | "faster_rcnn" | "grounding_dino"
    train_dataset: path to dataset manifest or dataset name in datasets/
    validation_dataset: optional validation manifest/dataset
    """
    model_type = model_type.lower()
    if model_type not in ("yolov8", "faster_rcnn", "grounding_dino"):
        raise ValueError("Unsupported model_type. Choose 'yolov8', 'faster_rcnn' or 'grounding_dino'")

    if model_type == "yolov8":
        ultralytics = _ensure_ultralytics()
        from ultralytics import YOLO
        # Determine YAML/data config - ultralytics supports a data YAML specifying train/val paths and names
        data_yaml = {
            "train": train_dataset,
            "val": validation_dataset if validation_dataset else train_dataset,
            "nc": None,  # will be set below
            "names": None
        }
        # If train_dataset points to datasets/<name>/train/manifest.jsonl, ultralytics can parse a dataset or you supply a proper data YAML.
        # For simplicity, we'll assume the user passed a path to a directory with images and labels (YOLO format).
        model = YOLO("yolov8n.pt")  # start from nano weights (modify as needed)
        print("Starting training - this will print ultralytics progress to stdout...")
        results = model.train(data=train_dataset, epochs=epochs, imgsz=imgsz, batch=batch)
        # results.params contains info; weights saved at project/runs/detect/train/weights/best.pt
        # Find the latest run's best weights
        run_root = Path("runs/detect")
        last_run = None
        if run_root.exists():
            last_run = sorted(run_root.iterdir(), key=os.path.getmtime)[-1]
            best = last_run / "weights" / "best.pt"
            if best.exists():
                model_id = best.as_posix()
                print(f"Training complete. Best weights: {model_id}")
                return model_id
        raise RuntimeError("Training finished but could not find saved weights in runs/detect/*/weights/best.pt")
    else:
        raise NotImplementedError("Training for model_type other than 'yolov8' is not implemented in this minimal SDK. You can extend this function to support Faster R-CNN or Grounding DINO.")
