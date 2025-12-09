"""
train(task_type: int, model_type: str, train_dataset: str, validation_dataset: str | None = None) -> str

This implementation uses Ultralytics YOLOv8 if available. If not installed,
it raises a helpful error. For Faster R-CNN you can integrate torchvision training later.

Returns a model_id (path to saved weights).
"""
from typing import Optional, List, Dict
import os
from pathlib import Path
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def _ensure_ultralytics():
    try:
        import ultralytics
        return ultralytics
    except Exception as e:
        raise RuntimeError(
            "Ultralytics YOLO is required for training. Install with: pip install ultralytics"
        )

class FasterRCNNDataset(Dataset):
    def __init__(self, root, manifest_path, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        
        # Build class map
        self.classes = sorted(list(set([l for row in self.data for l in row.get("labels", [])])))
        self.class_to_id = {c: i+1 for i, c in enumerate(self.classes)} # 0 is background

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        # If relative path, try to resolve against root or current dir
        if not os.path.isabs(img_path):
             # Try root/img_path first
             p = self.root / img_path
             if not p.exists():
                 p = Path(img_path) # try relative to cwd
             img_path = str(p)
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for bbox in item.get("bboxes", []):
            # bbox is [x1, y1, x2, y2, class_name]
            x1, y1, x2, y2, label = bbox
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[label])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.data)

def _collate_fn(batch):
    return tuple(zip(*batch))

def _convert_manifest_to_yolo_yaml(train_manifest, val_manifest=None):
    """Convert JSONL manifest to YOLO YAML format"""
    import yaml
    
    train_path = Path(train_manifest)
    
    # Read manifest to extract classes
    with open(train_path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    
    # Get all unique classes from manifest
    classes = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            classes.update(data.get("labels", []))
    
    classes = sorted(list(classes))
    
    # Create YOLO labels directory structure
    yolo_dir = train_path.parent / "yolo_format"
    yolo_dir.mkdir(exist_ok=True)
    
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    if val_manifest:
        (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Convert train manifest
    _convert_manifest_to_yolo_labels(train_manifest, yolo_dir / "labels" / "train", yolo_dir / "images" / "train", classes)
    
    if val_manifest:
        _convert_manifest_to_yolo_labels(val_manifest, yolo_dir / "labels" / "val", yolo_dir / "images" / "val", classes)
    
    # Create YAML config
    yaml_config = {
        "path": str(yolo_dir.absolute()),
        "train": "images/train",
        "val": "images/val" if val_manifest else "images/train",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f)
    
    print(f"Converted to YOLO format: {yaml_path}")
    return str(yaml_path)


def _convert_manifest_to_yolo_labels(manifest_path, labels_dir, images_dir, classes):
    """Convert JSONL manifest to YOLO label format"""
    import shutil
    from PIL import Image
    
    class_to_id = {name: i for i, name in enumerate(classes)}
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            img_path = Path(data["image_path"])
            
            # Copy image to YOLO images dir
            if img_path.exists():
                shutil.copy(img_path, images_dir / img_path.name)
                
                # Get image dimensions
                img = Image.open(img_path)
                img_w, img_h = img.size
                
                # Convert bboxes to YOLO format (normalized center coords)
                label_lines = []
                for bbox in data.get("bboxes", []):
                    x1, y1, x2, y2, cls = bbox
                    
                    # Convert to center x, center y, width, height (normalized)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    class_id = class_to_id.get(cls, 0)
                    label_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                # Write label file
                label_file = labels_dir / f"{img_path.stem}.txt"
                with open(label_file, "w") as lf:
                    lf.writelines(label_lines)


def _train_faster_rcnn(train_dataset_path, val_dataset_path, epochs, batch_size):
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torchvision.transforms as T
    except ImportError:
        raise RuntimeError("torchvision is required for Faster R-CNN training")

    # Resolve paths
    # Assuming train_dataset_path is a folder containing manifest.jsonl or the manifest itself
    train_path = Path(train_dataset_path)
    if train_path.is_dir():
        manifest_file = train_path / "manifest.jsonl"
        root_dir = train_path
    else:
        manifest_file = train_path
        root_dir = train_path.parent

    # Basic transforms
    transform = T.Compose([T.ToTensor()])

    dataset = FasterRCNNDataset(root_dir, manifest_file, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)

    # Model setup
    num_classes = len(dataset.classes) + 1 # +1 for background
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    print(f"Starting Faster R-CNN training on {device} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

    # Save model
    out_dir = Path("runs/detect/faster_rcnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "model.pth"
    
    # Save state dict and class mapping
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, save_path)
    
    return str(save_path)


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
        
        # YOLOv8 requires a YAML config file, not JSONL
        # Convert manifest to YOLO format if needed
        train_path = Path(train_dataset)
        if train_path.suffix == ".jsonl":
            # Need to convert to YOLO format
            yaml_path = _convert_manifest_to_yolo_yaml(train_dataset, validation_dataset)
            train_dataset = yaml_path
        
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
    
    elif model_type == "faster_rcnn":
        return _train_faster_rcnn(train_dataset, validation_dataset, epochs, batch)
        
    else:
        raise NotImplementedError("Training for model_type other than 'yolov8' and 'faster_rcnn' is not implemented in this minimal SDK.")
