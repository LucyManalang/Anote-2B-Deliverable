# BTT‑Anote‑2B — Multimodal RAG Chatbot + Computer Vision Fine‑Tuning SDK

A student‑friendly, production‑style project to build a **multimodal Retrieval‑Augmented Generation (RAG) chatbot** and a **Computer Vision (CV) fine‑tuning SDK** for object detection. The chatbot extends the **Autonomous Intelligence** baseline to support **text, images, audio, and video**, while **SyntheticDataGen** supplies **training, stress‑test, and evaluation datasets** to measure progress. The CV SDK provides APIs to **upload data, fine‑tune YOLO / Faster R‑CNN / Grounding DINO**, run **predictions**, and **evaluate** results with standard metrics.


## Research, Code, and Talks

- **Research paper (Object Detection Benchmarking):**
  https://drive.google.com/file/d/1IiBsuG1BwwGDGVslAs7gZttqMwvkjwYv/view

- **Source code (benchmarking):**
  https://github.com/nv78/OpenAnote/tree/main/materials/research/researchcode/Benchmarking-ObjectDetection

- **Talks:**
  Neha Naveen — https://www.youtube.com/watch?v=mOrear19fX4

  Anya Ross — https://www.youtube.com/watch?v=ZTL56FpMRec

  Spurthi Setty — https://www.youtube.com/watch?v=2GI5aFOx1BA

- **Product demo (image tasks):**
  https://www.youtube.com/watch?v=Qj653H5hvIw

- **Datasets & generation services:**
  https://anote.ai/syntheticdata

---

## Why Multimodal RAG and Synthetic Data?

- **Real tasks are multimodal.** Images, audio, video, and text must be fused to answer questions with complete context.
- **Long‑tail coverage.** Edge cases (noisy speech, blur, rare objects) are scarce. **SyntheticDataGen** programmatically fills gaps with controlled difficulty, noise, and distribution shifts.
- **Faster iteration.** Replace weeks of manual collection/labeling with scripted generation; track improvements objectively.
- **Privacy‑aware.** Use PII‑safe synthetic surrogates instead of sensitive source data.
- **Reproducible & benchmarkable.** Seeds, manifests, and versioned artifacts support fair comparisons across model variants.

**Datasets & services:** https://anote.ai/syntheticdata

---

## Product Scope (Multimodal Version of Anote)

Everything we do for **text** now extends to **images** (and other modalities):

- **Image classification** (single/multi‑label)
- **Object detection with images** (bbox annotations)
- **Q&A per image** (ask questions about a single image)
- **Q&A across images** (retrieve related images and answer)
- **Chatbot with image upload** (drag‑and‑drop image → ask questions; grounded citations)

**Tasks supported for other modalities:**
- **Audio:** ASR transcripts (e.g., Whisper), Q&A over transcripts, cross‑modal joins (e.g., “what was said when the laptop appeared?”)
- **Video:** key‑frame extraction, captioning/summarization, temporal grounding, cross‑modal Q&A

**Example video demo:** https://www.youtube.com/watch?v=Qj653H5hvIw

---

## Architecture Overview

**Baseline code:** start from the *Autonomous Intelligence* repo (instructor‑provided). Add multimodal ingestion + a unified RAG layer.

1. **Ingestion / Pre‑processing**
   - Text: Markdown/PDF parsers → chunker
   - Images: OCR (optional), captioning (VLM), object detection (optional)
   - Audio: ASR (Whisper) → timestamped transcripts
   - Video: key‑frames + segment captions/summaries (+ timestamps)

2. **Representations & Indexing**
   - Normalize all modalities into **textual representations** with **rich metadata** (modality, file path, timecodes, bbox/frame IDs)
   - Hybrid retrieval (**BM25 + dense embeddings**) with modality filters; vector store (FAISS/Chroma/pgvector)

3. **Query & Fusion**
   - Retrieve top‑k per modality; fuse via RRF or a learned re‑ranker
   - Generator (LLM) answers with **citations** (frame/timecode, bbox IDs, filenames)

4. **Evaluation & Telemetry**
   - Integrate **SyntheticDataGen** to generate **dev/test** sets per modality
   - Metrics: Retrieval (Recall@k, nDCG@k), Generation (EM/F1/ROUGE), Audio (WER), CV (mAP), System (latency, grounding rate)

---

## Computer Vision API — Object Detection SDK

### CV Project SDK (Workflow)

**Choose Models**
- Options: **YOLO**, **Grounding DINO**, **Faster R‑CNN** (more models in future)
- Select any/all to compare

**Upload Dataset**
- **Labeled data required** (initial release); **format varies by model**
- *Future:* Unlabeled uploads supported via Anote “smart annotation”

**Train**
- Click **Begin Training** to launch job; live plots (e.g., Weights & Biases)
- Optional email on training completion

**Eval**
- Standardized evaluation with formatted metrics and comparisons
- Report **best epoch** per model

**Suggest**
- System **recommends best model + epoch** for deployment
- One‑click export of model bundle
- *Future:* Interactive prediction on selected images inside UI

> The SDK exposes **upload → train → predict → evaluate** methods to script the full lifecycle.

---

### Methods Overview

- `upload()` — register dataset splits and metadata
- `train()` — fine‑tune a selected model family (task_type=5 for detection)
- `predict()` — run inference on images/folders/manifests
- `evaluate()` — score predictions vs. ground truth

#### `upload()`
**Description:** Uploads and registers a dataset for training/evaluation.

**Signature:**
```python
upload(dataset_name: str, data_path: str, split: str) -> dict
```
- `dataset_name`: unique identifier for the dataset
- `data_path`: path to a JSON/JSONL manifest with fields:
  - `image_path`: absolute or repo‑relative image URI
  - `labels`: list of class names present in image
  - `bboxes`: list of `[x1, y1, x2, y2, class_name]`
- `split`: `"train" | "validation" | "test"`

**Returns:** status dict (confirmation + counts)

**Example manifest row (`.jsonl`):**
```json
{"image_path":"data/imgs/0001.jpg",
 "labels":["laptop","person"],
 "bboxes":[[140,220,420,410,"laptop"], [30,100,120,360,"person"]]}
```

> *Notes on formats:*
> - **YOLO** expects per‑image `.txt` label files (class, x_center, y_center, w, h) in normalized coords; the SDK can convert from the JSONL manifest.
> - **Faster R‑CNN** commonly uses **COCO JSON**; the SDK can emit COCO from the manifest.
> - **Grounding DINO** supports COCO‑like boxes; for phrase grounding, prompts can be passed at `predict()` time.

#### `train()`
**Description:** Trains an object detection model on a registered dataset.

**Signature:**
```python
train(
  task_type: int,
  model_type: str,
  train_dataset: str,
  validation_dataset: str | None = None
) -> str
```
- `task_type`: **5** for object detection
- `model_type`: `"yolov8" | "faster_rcnn" | "grounding_dino"`
- `train_dataset`: dataset name or path to training JSON/JSONL/COCO
- `validation_dataset` (optional): path/name for validation split

**Returns:** `model_id` (string)

#### `predict()`
**Description:** Runs inference on images using a trained model.

**Signature:**
```python
predict(
  model_type: str,
  test_data: str,
  labels: list[str],
  model_id: str | None = None,
  confidence_threshold: float = 0.5
) -> list[dict]
```
- `model_type`: `"yolov8" | "faster_rcnn" | "grounding_dino"`
- `test_data`: path to image file, folder, or JSON/JSONL manifest
- `labels`: class label list used for training
- `model_id` (optional): specific trained model; else latest for `model_type`
- `confidence_threshold` (optional): filter low‑confidence predictions

**Returns:** list of predictions, per image:
```json
[
  {
    "image_id": "data/imgs/0001.jpg",
    "boxes": [[140,220,420,410], [30,100,120,360]],
    "labels": ["laptop","person"],
    "confidence": [0.94, 0.88]
  }
]
```

#### `evaluate()`
**Description:** Compares predictions to ground truth and computes metrics.

**Signature:**
```python
evaluate(
  ground_truths: str,
  predictions: str
) -> dict
```
- `ground_truths`: path to ground truth JSON/JSONL/COCO
- `predictions`: path to predictions JSON/JSONL

**Returns:** paths to artifacts + metrics:
- `confusion_matrix.png` — confusion matrix visualization
- `metrics.csv` — Precision, Recall, Accuracy, F1‑score, **mIoU**, **mAP**

---

## Example: End‑to‑End CV SDK (Python)

```python
from cvsdk import upload, train, predict, evaluate

# 1) Upload (train/val/test)
for split in ["train", "validation", "test"]:
    upload(
        dataset_name=f"undersea_{split}",
        data_path=f"data/undersea_{split}.jsonl",
        split=split
    )

# 2) Train
model_id = train(
    task_type=5,
    model_type="yolov8",
    train_dataset="undersea_train",
    validation_dataset="undersea_validation"
)

# 3) Predict
preds = predict(
    model_type="yolov8",
    test_data="data/undersea_test.jsonl",
    labels=["fish","jellyfish","crab","urchin","starfish","coral","seaweed"],
    model_id=model_id,
    confidence_threshold=0.4
)

# 4) Evaluate
report = evaluate(
    ground_truths="data/undersea_test.jsonl",
    predictions="outputs/preds_undersea.jsonl"
)
print(report)  # paths to metrics.csv, confusion_matrix.png
```

---

## Multimodal RAG Evaluation (with SyntheticDataGen)

- Generate **training** and **eval** sets for: captions, transcripts, detection boxes, and adversarial/noisy variants.
- Measure: **Retrieval** (Recall@k, nDCG@k), **Generation** (EM/F1/ROUGE), **Audio** (WER), **CV** (mAP), **System** (latency, grounding rate).
- Report a **leaderboard** across configs (embedder, top‑k, fusion method, re‑ranker on/off, chunk size).

**API snippet (SyntheticDataGen):**
```python
from anotegenerate import generate
gen_images = generate(
  task_type="image",
  prompt="Undersea species with varied turbidity/lighting",
  num_rows=300,
  columns=["image_path","bboxes","classes","split"],
  params={"resolution":"1024x1024",
          "classes":["fish","jellyfish","crab","urchin","starfish","coral","seaweed"],
          "bbox_augment": True},
  media_dir="examples/examples_data"
)
```


## Deliverables & Demo Script

**Deliverables**
- Multimodal RAG chatbot (text+image+audio+video) with citations and modality filters
- CV SDK scripts + README (upload/train/predict/evaluate)
- SyntheticDataGen‑based eval harness + leaderboard report
- Short error analysis (missed detections, ASR errors, retrieval misses)

**5–7 min demo**
1. Upload a small image set; draw a couple of bboxes in the UI (or show manifest).
2. Launch one YOLO and one Faster R‑CNN training job; show live training curves.
3. Predict on 3 test images; open **confusion_matrix.png** and **metrics.csv**.
4. Ingest an image + a short video + a 5‑sec audio clip; index into RAG.
5. Ask a cross‑modal question; answer includes **frame/timecode** + **bbox** citations.
6. Show SyntheticDataGen test run and leaderboard table.

---

## Notes on Data Formats

- **Unified JSONL manifest** (`image_path`, `labels`, `bboxes`) is the **source of truth**.
- SDK provides converters for:
  - **YOLO** label files (normalized center‑format)
  - **COCO** JSON for Faster R‑CNN / Grounding DINO
- For **Grounding DINO** phrase grounding, pass prompts at `predict()` time (e.g., “laptop”, “person with backpack”).

---

## Glossary

- **RAG**: Retrieval‑Augmented Generation; LLM answers grounded in retrieved context
- **ASR**: Automatic Speech Recognition (audio → text)
- **OCR**: Optical Character Recognition (image/PDF text extraction)
- **VLM**: Vision‑Language Model (captioning; vision encoders with LLMs)
- **mAP, mIoU, WER, EM/F1, ROUGE**: Common evaluation metrics
