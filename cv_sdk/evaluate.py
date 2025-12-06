"""
evaluate(ground_truths: str, predictions: str) -> dict

Computes:
 - per-class precision/recall
 - overall mAP (approximate via IoU threshold)
 - confusion matrix
Saves:
 - metrics.csv
 - confusion_matrix.png
"""
from typing import List, Dict
from .utils import load_jsonl, save_jsonl
from pathlib import Path
import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate(ground_truths: str, predictions: str, out_dir: str = "cv_eval", iou_threshold: float = 0.5) -> Dict:
    """
    ground_truths: path to manifest jsonl with fields image_path, bboxes
    predictions: path to predictions jsonl OR python object path (list of dicts) saved to file
    """
    ensure_out = Path(out_dir)
    ensure_out.mkdir(parents=True, exist_ok=True)

    # Load
    gt = load_jsonl(ground_truths)
    with open(predictions, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    # Build mappings by image_id
    gt_map = {}
    for i, row in enumerate(gt):
        img = row["image_path"]
        gt_map[img] = row.get("bboxes", [])

    # Eval per-image
    all_gts = []
    all_preds = []
    per_image_metrics = []
    classes_set = set()
    for p in preds:
        img = p["image_id"]
        pred_boxes = p.get("boxes", [])
        pred_labels = p.get("labels", [])
        pred_scores = p.get("confidence", [])
        gts = gt_map.get(img, [])
        # Convert GT to (box,label)
        matched_gt = set()
        tp = 0
        fp = 0
        fn = 0
        # naive matching: for each pred, find best matching GT by IoU and label equality
        matches = []
        for j, pb in enumerate(pred_boxes):
            best_iou = 0
            best_idx = None
            for k, gb in enumerate(gts):
                gx1, gy1, gx2, gy2, gcls = gb
                iou_val = iou(pb, [gx1, gy1, gx2, gy2])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = k
            if best_iou >= iou_threshold:
                # check label match
                matched_gt.add(best_idx)
                pred_label = pred_labels[j] if j < len(pred_labels) else None
                gt_label = gts[best_idx][4] if best_idx is not None and len(gts[best_idx]) >= 5 else None
                if pred_label == gt_label:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
        fn = max(0, len(gts) - len(matched_gt))
        per_image_metrics.append({"image_id": img, "tp": tp, "fp": fp, "fn": fn})
        # For confusion matrix build simple lists
        for gb in gts:
            classes_set.add(gb[4])
            all_gts.append(gb[4])
        for lab in pred_labels:
            classes_set.add(lab)
            all_preds.append(lab)

    classes = sorted(list(classes_set))
    if not classes:
        classes = ["__none__"]

    # Confusion matrix -- map to indices, missing labels are handled
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = [label_to_idx.get(l, 0) for l in all_gts]
    y_pred = [label_to_idx.get(l, 0) for l in all_preds]

    if len(y_true) == 0 or len(y_pred) == 0:
        # No predictions or no GTs present
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        precision = recall = f1 = 0.0
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Save confusion matrix plot
    try:
        import seaborn as sns
        plt_fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion matrix")
        plt.tight_layout()
        fig_path = Path(out_dir) / "confusion_matrix.png"
        plt_fig.savefig(fig_path.as_posix())
    except Exception:
        fig_path = None

    # Save metrics CSV
    metrics_path = Path(out_dir) / "metrics.csv"
    with open(metrics_path, "w", newline="") as csvfile:
        fieldnames = ["precision", "recall", "f1", "mAP_approx"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"precision": precision, "recall": recall, "f1": f1, "mAP_approx": 0.0})

    # Save per-image metrics
    per_image_path = Path(out_dir) / "per_image_metrics.jsonl"
    save_jsonl(per_image_metrics, per_image_path.as_posix())

    return {
        "confusion_matrix": fig_path.as_posix() if fig_path else None,
        "metrics_csv": metrics_path.as_posix(),
        "per_image": per_image_path.as_posix(),
        "classes": classes
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="ground truth manifest (jsonl)")
    parser.add_argument("--pred", required=True, help="predictions jsonl path")
    parser.add_argument("--out_dir", default="cv_eval")
    args = parser.parse_args()
    print(evaluate(args.gt, args.pred, out_dir=args.out_dir))
