# SyntheticDataGen Module

Synthetic data generation API for the Anote-2B project.

## Overview

This module provides APIs to generate synthetic datasets for:
- **Object Detection** (images with bounding box annotations)
- **RAG Evaluation** (Q&A pairs with gold answers)
- **Text Documents** (for multimodal RAG testing)
- **Audio/Video Metadata** (transcripts and temporal annotations)

## Installation

No additional dependencies required beyond the main project requirements.

Optional for image generation:
```bash
pip install Pillow
```

## Quick Start

### Generate Object Detection Dataset

```python
from synthetic_data import generate

result = generate(
    task_type="image",
    prompt="Undersea species with varied lighting",
    num_rows=300,
    columns=["image_path", "bboxes", "classes", "split"],
    params={
        "resolution": "1024x1024",
        "classes": ["fish", "jellyfish", "crab", "urchin"],
        "bbox_augment": True
    },
    media_dir="outputs/undersea"
)

print(f"Generated {result['num_generated']} images")
print(f"Manifest: {result['manifest_path']}")
```

### Generate RAG Evaluation Set

```python
from synthetic_data import generate_rag_eval_set

eval_path = generate_rag_eval_set(
    num_samples=50,
    output_path="rag_eval.json"
)
```

### Generate Complete CV Dataset

```python
from synthetic_data import generate_cv_dataset

manifests = generate_cv_dataset(
    classes=["cat", "dog", "bird"],
    num_images=100,
    resolution="640x480",
    output_dir="cv_dataset"
)

# Returns:
# {
#     "train": "cv_dataset/train_manifest.jsonl",
#     "validation": "cv_dataset/val_manifest.jsonl",
#     "test": "cv_dataset/test_manifest.jsonl"
# }
```

## API Reference

### `generate(task_type, prompt, num_rows, columns, params, media_dir)`

Main API for generating synthetic data.

**Parameters:**
- `task_type` (str): Type of data - "image", "text", "audio", "video"
- `prompt` (str): Description of the data to generate
- `num_rows` (int): Number of samples to generate
- `columns` (list): Required columns in output
- `params` (dict): Task-specific parameters
- `media_dir` (str): Output directory for generated files

**Returns:** Dict with generation status and file paths

### `generate_rag_eval_set(num_samples, output_path)`

Generate evaluation dataset for RAG system testing.

**Returns:** Path to generated JSON file

### `generate_cv_dataset(classes, num_images, resolution, output_dir)`

Generate complete object detection dataset with train/val/test splits.

**Returns:** Dict with paths to split manifests

## Task-Specific Parameters

### Image Generation

```python
params = {
    "resolution": "1024x1024",  # Image dimensions
    "classes": ["class1", "class2"],  # Object classes
    "bbox_augment": True  # Add noise to bbox coordinates
}
```

### Text Generation

```python
params = {
    "topics": ["science", "history"],  # Document topics
    "doc_length": "medium"  # short/medium/long
}
```

### Audio Generation

```python
params = {
    "duration_range": [5, 30],  # Min/max seconds
    "add_noise": True  # Simulate background noise
}
```

### Video Generation

```python
params = {
    "fps": 30,  # Frames per second
    "duration_range": [10, 60]  # Min/max seconds
}
```

## Output Formats

### Image Manifest (JSONL)

```json
{
  "image_path": "outputs/synthetic_0001.jpg",
  "labels": ["fish", "coral"],
  "bboxes": [[120, 80, 340, 260, "fish"], [400, 150, 580, 320, "coral"]],
  "split": "train"
}
```

### RAG Evaluation Set (JSON)

```json
[
  {
    "id": "1",
    "question": "What animal appears in the images?",
    "gold_answer": "A tiger laying on the ground",
    "gold_context_ids": [1, 2, 5]
  }
]
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic data generation
- Integration with CV SDK
- Full training pipeline
- RAG evaluation workflow

## Production Usage

This is a **local mock implementation** for development. For production:

1. Obtain API key from https://anote.ai/syntheticdata
2. Pass API key to `generate()`:

```python
result = generate(
    task_type="image",
    prompt="...",
    api_key="your-api-key-here"
)
```

The implementation will automatically switch to API mode when a key is provided.

## Notes

- Images are generated as placeholders with colored rectangles
- Audio/video files are metadata-only (no actual media files)
- For real production use, integrate with Anote AI API
- All manifests are in COCO-compatible format
