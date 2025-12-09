"""
Computer Vision SDK package for dataset upload, training, prediction, and evaluation.

This package provides simple wrappers to:
 - convert a JSONL manifest to YOLO/COCO formats
 - train a YOLOv8 model (via ultralytics) if available
 - run predictions using a model file or the latest trained model
 - evaluate predictions vs. ground truth and save metrics/artifacts

Note: ultralytics (YOLO) is recommended for quick experiments:
    pip install ultralytics
"""

from .upload import upload
from .train import train
from .predict import predict
from .evaluate import evaluate

__all__ = ["upload", "train", "predict", "evaluate", "utils"]
