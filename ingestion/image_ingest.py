import json
import os
from typing import List, Dict

from PIL import Image

# Optional BLIP import (only used if no manual caption available)
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load manual override captions
def load_manual_captions(json_path="data/image_captions.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


# Initialize BLIP lazily only if needed
_blip_processor = None
_blip_model = None

def load_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        print("-> Loading BLIP caption model (fallback)...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model



def ingest_images(image_paths: List[str], manual_captions_path="data/image_captions.json") -> List[Dict]:
    """
    Ingest images by:
      1. Checking manual captions (recommended for demo)
      2. Falling back to BLIP model captioning
    """

    manual_captions = load_manual_captions(manual_captions_path)
    docs = []

    for path in image_paths:
        # Always use absolute path key if needed
        key1 = path
        key2 = os.path.abspath(path)

        if key1 in manual_captions:
            caption = manual_captions[key1]
            print(f"[OK] Using MANUAL caption for {path}")
        elif key2 in manual_captions:
            caption = manual_captions[key2]
            print(f"[OK] Using MANUAL caption for {path}")
        else:
            # Fallback to BLIP captioning
            processor, model = load_blip()
            image = Image.open(path).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            print(f"[BLIP] caption for {path}: {caption}")

        docs.append({
            "id": path,
            "text": caption,
            "metadata": {
                "modality": "image",
                "file_path": path
            }
        })

    return docs
