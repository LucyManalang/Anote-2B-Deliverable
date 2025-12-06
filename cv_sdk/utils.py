import os
import json
from typing import List, Dict, Tuple
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(rows: List[Dict], out_path: str):
    ensure_dir(Path(out_path).parent.as_posix())
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def coco_box_from_xyxy(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
