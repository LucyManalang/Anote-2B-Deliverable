"""
SyntheticDataGen API Implementation

This provides a mock/local implementation of the Anote AI SyntheticDataGen service.
For production use, replace with actual API calls to https://anote.ai/syntheticdata
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
from datetime import datetime
import numpy as np

# Try to import image generation libraries (optional)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class SyntheticDataGenerator:
    """
    Generates synthetic datasets for multimodal RAG and CV training.
    
    For real production use, this should make API calls to Anote's service.
    This local version generates placeholder data for development/testing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            api_key: API key for Anote AI service (optional for mock mode)
        """
        self.api_key = api_key
        self.mock_mode = api_key is None
        
    def generate(
        self,
        task_type: str,
        prompt: str,
        num_rows: int = 100,
        columns: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        media_dir: str = "synthetic_data/outputs"
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on task type.
        
        Args:
            task_type: Type of data to generate ("image", "text", "audio", "video")
            prompt: Description of the data to generate
            num_rows: Number of samples to generate
            columns: Required columns in the output
            params: Additional parameters (resolution, classes, augmentations, etc.)
            media_dir: Directory to save generated media files
            
        Returns:
            Dict containing generated data and metadata
        """
        params = params or {}
        columns = columns or []
        
        # Create output directory
        Path(media_dir).mkdir(parents=True, exist_ok=True)
        
        if task_type == "image":
            return self._generate_images(prompt, num_rows, columns, params, media_dir)
        elif task_type == "text":
            return self._generate_text(prompt, num_rows, columns, params, media_dir)
        elif task_type == "audio":
            return self._generate_audio(prompt, num_rows, columns, params, media_dir)
        elif task_type == "video":
            return self._generate_video(prompt, num_rows, columns, params, media_dir)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    
    def _generate_images(
        self,
        prompt: str,
        num_rows: int,
        columns: List[str],
        params: Dict[str, Any],
        media_dir: str
    ) -> Dict[str, Any]:
        """Generate synthetic images with bounding box annotations."""
        
        if not HAS_PIL:
            print("Warning: PIL not available. Generating metadata only.")
        
        resolution = params.get("resolution", "640x480")
        width, height = map(int, resolution.split("x"))
        classes = params.get("classes", ["object1", "object2"])
        bbox_augment = params.get("bbox_augment", False)
        
        manifest_path = Path(media_dir) / "manifest.jsonl"
        results = []
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for i in range(num_rows):
                # Generate image filename
                img_name = f"synthetic_{i:04d}.jpg"
                img_path = str(Path(media_dir) / img_name)
                
                # Generate random bounding boxes
                num_objects = random.randint(1, min(5, len(classes)))
                selected_classes = random.sample(classes, num_objects)
                bboxes = []
                
                for cls in selected_classes:
                    # Random bbox coordinates
                    x1 = random.randint(0, width // 2)
                    y1 = random.randint(0, height // 2)
                    x2 = random.randint(x1 + 50, width)
                    y2 = random.randint(y1 + 50, height)
                    
                    if bbox_augment:
                        # Add noise to bbox coordinates
                        noise = 10
                        x1 += random.randint(-noise, noise)
                        y1 += random.randint(-noise, noise)
                        x2 += random.randint(-noise, noise)
                        y2 += random.randint(-noise, noise)
                    
                    # Ensure valid coordinates
                    x1 = max(0, min(x1, width))
                    x2 = max(x1 + 10, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(y1 + 10, min(y2, height))
                    
                    bboxes.append([x1, y1, x2, y2, cls])
                
                # Determine split
                split_rand = random.random()
                if split_rand < 0.7:
                    split = "train"
                elif split_rand < 0.85:
                    split = "validation"
                else:
                    split = "test"
                
                # Create data row
                row = {
                    "image_path": img_path,
                    "labels": [b[4] for b in bboxes],
                    "bboxes": bboxes
                }
                
                if "split" in columns:
                    row["split"] = split
                
                f.write(json.dumps(row) + "\n")
                results.append(row)
                
                # Generate placeholder image if PIL available
                if HAS_PIL:
                    self._create_placeholder_image(img_path, width, height, bboxes)
        
        return {
            "status": "success",
            "task_type": "image",
            "num_generated": num_rows,
            "manifest_path": str(manifest_path),
            "media_dir": media_dir,
            "classes": classes,
            "prompt": prompt
        }
    
    def _create_placeholder_image(
        self,
        img_path: str,
        width: int,
        height: int,
        bboxes: List[List]
    ):
        """Create a simple placeholder image with bounding boxes drawn."""
        
        # Generate random colored background
        color = tuple(random.randint(100, 255) for _ in range(3))
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2, cls = bbox
            # Random color for each box
            box_color = tuple(random.randint(0, 200) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            # Draw label text
            try:
                draw.text((x1, y1 - 15), cls, fill=box_color)
            except:
                pass  # If font not available
        
        img.save(img_path)
    
    def _generate_text(
        self,
        prompt: str,
        num_rows: int,
        columns: List[str],
        params: Dict[str, Any],
        media_dir: str
    ) -> Dict[str, Any]:
        """Generate synthetic text documents for RAG evaluation."""
        
        topics = params.get("topics", ["science", "history", "technology"])
        doc_length = params.get("doc_length", "medium")  # short, medium, long
        
        if doc_length == "short":
            sentences_per_doc = (3, 5)
        elif doc_length == "long":
            sentences_per_doc = (15, 25)
        else:
            sentences_per_doc = (8, 12)
        
        manifest_path = Path(media_dir) / "text_manifest.jsonl"
        results = []
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for i in range(num_rows):
                topic = random.choice(topics)
                num_sentences = random.randint(*sentences_per_doc)
                
                # Generate synthetic text
                text = f"{topic.title()} Document {i}: "
                text += " ".join([
                    f"This is sentence {j} about {topic} with relevant information."
                    for j in range(num_sentences)
                ])
                
                # Create text file
                text_file = Path(media_dir) / f"doc_{i:04d}.txt"
                with open(text_file, "w", encoding="utf-8") as tf:
                    tf.write(text)
                
                row = {
                    "doc_id": i,
                    "text_path": str(text_file),
                    "topic": topic,
                    "text_preview": text[:100] + "..."
                }
                
                f.write(json.dumps(row) + "\n")
                results.append(row)
        
        return {
            "status": "success",
            "task_type": "text",
            "num_generated": num_rows,
            "manifest_path": str(manifest_path),
            "media_dir": media_dir
        }
    
    def _generate_audio(
        self,
        prompt: str,
        num_rows: int,
        columns: List[str],
        params: Dict[str, Any],
        media_dir: str
    ) -> Dict[str, Any]:
        """Generate synthetic audio transcripts with metadata."""
        
        duration_range = params.get("duration_range", [5, 30])  # seconds
        add_noise = params.get("add_noise", False)
        
        manifest_path = Path(media_dir) / "audio_manifest.jsonl"
        results = []
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for i in range(num_rows):
                duration = random.uniform(*duration_range)
                
                # Generate synthetic transcript
                transcript = f"This is synthetic audio clip {i}. "
                transcript += "It contains spoken content for testing ASR systems. "
                
                if add_noise:
                    transcript += "[background noise] "
                
                row = {
                    "audio_id": i,
                    "audio_path": f"synthetic_audio_{i:04d}.wav",
                    "transcript": transcript,
                    "duration": round(duration, 2),
                    "has_noise": add_noise
                }
                
                f.write(json.dumps(row) + "\n")
                results.append(row)
        
        return {
            "status": "success",
            "task_type": "audio",
            "num_generated": num_rows,
            "manifest_path": str(manifest_path),
            "note": "Audio files not generated (metadata only)"
        }
    
    def _generate_video(
        self,
        prompt: str,
        num_rows: int,
        columns: List[str],
        params: Dict[str, Any],
        media_dir: str
    ) -> Dict[str, Any]:
        """Generate synthetic video metadata with frame annotations."""
        
        fps = params.get("fps", 30)
        duration_range = params.get("duration_range", [10, 60])
        
        manifest_path = Path(media_dir) / "video_manifest.jsonl"
        results = []
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            for i in range(num_rows):
                duration = random.uniform(*duration_range)
                num_frames = int(duration * fps)
                
                # Generate keyframe timestamps
                num_keyframes = random.randint(3, 8)
                keyframes = sorted(random.sample(range(num_frames), num_keyframes))
                
                row = {
                    "video_id": i,
                    "video_path": f"synthetic_video_{i:04d}.mp4",
                    "duration": round(duration, 2),
                    "fps": fps,
                    "keyframes": keyframes,
                    "caption": f"Synthetic video {i} showing various scenes."
                }
                
                f.write(json.dumps(row) + "\n")
                results.append(row)
        
        return {
            "status": "success",
            "task_type": "video",
            "num_generated": num_rows,
            "manifest_path": str(manifest_path),
            "note": "Video files not generated (metadata only)"
        }


# Global generator instance
_generator = None

def generate(
    task_type: str,
    prompt: str,
    num_rows: int = 100,
    columns: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    media_dir: str = "synthetic_data/outputs",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main API function for generating synthetic data.
    
    This is the primary interface matching the Anote AI SyntheticDataGen API.
    
    Example:
        >>> gen_images = generate(
        ...     task_type="image",
        ...     prompt="Undersea species with varied turbidity",
        ...     num_rows=300,
        ...     columns=["image_path", "bboxes", "classes", "split"],
        ...     params={
        ...         "resolution": "1024x1024",
        ...         "classes": ["fish", "jellyfish", "crab"],
        ...         "bbox_augment": True
        ...     }
        ... )
    """
    global _generator
    
    if _generator is None:
        _generator = SyntheticDataGenerator(api_key=api_key)
    
    return _generator.generate(
        task_type=task_type,
        prompt=prompt,
        num_rows=num_rows,
        columns=columns,
        params=params,
        media_dir=media_dir
    )


def generate_rag_eval_set(
    num_samples: int = 50,
    output_path: str = "synthetic_data/rag_eval.json"
) -> str:
    """
    Generate an evaluation dataset for RAG system testing.
    
    Returns path to generated JSON file with format:
    [
        {
            "id": "1",
            "question": "What appears in image 0001?",
            "gold_answer": "A tiger laying on the ground",
            "gold_context_ids": [1, 2]
        },
        ...
    ]
    """
    samples = []
    
    # Sample questions for multimodal RAG
    question_templates = [
        "What animal appears in the images?",
        "Describe the objects in the document.",
        "What is the main theme of the content?",
        "What sounds can be heard in the audio?",
        "What happens in the video at timestamp {time}?",
        "How many {object} are visible?",
        "What color is the {object}?",
        "Where is the {object} located?"
    ]
    
    for i in range(num_samples):
        template = random.choice(question_templates)
        question = template.format(
            time=f"{random.randint(0, 60)}s",
            object=random.choice(["tiger", "mug", "person", "car", "tree"])
        )
        
        sample = {
            "id": str(i),
            "question": question,
            "gold_answer": f"Sample answer {i} based on retrieved context.",
            "gold_context_ids": [random.randint(1, 10) for _ in range(random.randint(1, 3))]
        }
        samples.append(sample)
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Generated {num_samples} RAG evaluation samples at {output_path}")
    return output_path


def generate_cv_dataset(
    classes: List[str],
    num_images: int = 100,
    resolution: str = "640x480",
    output_dir: str = "synthetic_data/cv_dataset"
) -> Dict[str, str]:
    """
    Generate a complete CV dataset for object detection training.
    
    Returns paths to train/val/test manifests.
    """
    result = generate(
        task_type="image",
        prompt=f"Object detection dataset with classes: {', '.join(classes)}",
        num_rows=num_images,
        columns=["image_path", "bboxes", "labels", "split"],
        params={
            "resolution": resolution,
            "classes": classes,
            "bbox_augment": True
        },
        media_dir=output_dir
    )
    
    # Split manifest into train/val/test files
    manifest_path = result["manifest_path"]
    
    train_manifest = Path(output_dir) / "train_manifest.jsonl"
    val_manifest = Path(output_dir) / "val_manifest.jsonl"
    test_manifest = Path(output_dir) / "test_manifest.jsonl"
    
    with open(manifest_path, "r") as f:
        lines = f.readlines()
    
    train_lines = []
    val_lines = []
    test_lines = []
    
    for line in lines:
        data = json.loads(line)
        split = data.get("split", "train")
        if split == "train":
            train_lines.append(line)
        elif split == "validation":
            val_lines.append(line)
        else:
            test_lines.append(line)
    
    # Write split files
    with open(train_manifest, "w") as f:
        f.writelines(train_lines)
    
    with open(val_manifest, "w") as f:
        f.writelines(val_lines)
    
    with open(test_manifest, "w") as f:
        f.writelines(test_lines)
    
    print(f"Generated CV dataset:")
    print(f"  Train: {len(train_lines)} images -> {train_manifest}")
    print(f"  Val: {len(val_lines)} images -> {val_manifest}")
    print(f"  Test: {len(test_lines)} images -> {test_manifest}")
    
    return {
        "train": str(train_manifest),
        "validation": str(val_manifest),
        "test": str(test_manifest),
        "media_dir": output_dir
    }
