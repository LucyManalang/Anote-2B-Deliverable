"""
Example script demonstrating SyntheticDataGen API usage.

This shows how to:
1. Generate synthetic CV datasets for training
2. Generate RAG evaluation sets
3. Use generated data with the CV SDK and RAG pipeline
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data import generate, generate_rag_eval_set, generate_cv_dataset


def example_1_generate_cv_dataset():
    """Example 1: Generate synthetic object detection dataset"""
    print("\n=== Example 1: Generate CV Dataset ===")
    
    result = generate(
        task_type="image",
        prompt="Undersea species with varied turbidity/lighting",
        num_rows=50,
        columns=["image_path", "bboxes", "classes", "split"],
        params={
            "resolution": "640x480",
            "classes": ["fish", "jellyfish", "crab", "urchin", "starfish", "coral", "seaweed"],
            "bbox_augment": True
        },
        media_dir="synthetic_data/outputs/undersea"
    )
    
    print(f"Status: {result['status']}")
    print(f"Generated: {result['num_generated']} images")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Classes: {result['classes']}")


def example_2_generate_rag_eval():
    """Example 2: Generate RAG evaluation dataset"""
    print("\n=== Example 2: Generate RAG Evaluation Set ===")
    
    eval_path = generate_rag_eval_set(
        num_samples=20,
        output_path="synthetic_data/outputs/rag_eval.json"
    )
    
    print(f"Generated evaluation set at: {eval_path}")
    
    # Show a few samples
    import json
    with open(eval_path, "r") as f:
        samples = json.load(f)
    
    print(f"\nFirst 3 samples:")
    for sample in samples[:3]:
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['gold_answer']}")
        print()


def example_3_generate_split_cv_dataset():
    """Example 3: Generate CV dataset with automatic train/val/test splits"""
    print("\n=== Example 3: Generate Split CV Dataset ===")
    
    manifests = generate_cv_dataset(
        classes=["cat", "dog", "bird"],
        num_images=30,
        resolution="512x512",
        output_dir="synthetic_data/outputs/animals"
    )
    
    print(f"\nGenerated manifests:")
    for split, path in manifests.items():
        if split != "media_dir":
            print(f"  {split}: {path}")


def example_4_generate_text_dataset():
    """Example 4: Generate synthetic text documents"""
    print("\n=== Example 4: Generate Text Dataset ===")
    
    result = generate(
        task_type="text",
        prompt="Technical documentation about AI and machine learning",
        num_rows=20,
        columns=["doc_id", "text_path", "topic"],
        params={
            "topics": ["machine_learning", "computer_vision", "nlp", "robotics"],
            "doc_length": "medium"
        },
        media_dir="synthetic_data/outputs/text_docs"
    )
    
    print(f"Generated: {result['num_generated']} documents")
    print(f"Manifest: {result['manifest_path']}")


def example_5_generate_audio_metadata():
    """Example 5: Generate audio transcript metadata"""
    print("\n=== Example 5: Generate Audio Metadata ===")
    
    result = generate(
        task_type="audio",
        prompt="Podcast conversations about technology",
        num_rows=15,
        params={
            "duration_range": [10, 60],
            "add_noise": True
        },
        media_dir="synthetic_data/outputs/audio"
    )
    
    print(f"Generated: {result['num_generated']} audio metadata entries")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Note: {result.get('note', '')}")


def example_6_full_pipeline_demo():
    """Example 6: Full pipeline - Generate data and use with CV SDK"""
    print("\n=== Example 6: Full Pipeline Demo ===")
    
    # Generate dataset
    print("Step 1: Generating synthetic dataset...")
    manifests = generate_cv_dataset(
        classes=["tiger", "lion", "leopard"],
        num_images=20,
        resolution="640x480",
        output_dir="synthetic_data/outputs/big_cats"
    )
    
    # Train with CV SDK
    print("\nStep 2: Training with CV SDK...")
    try:
        from cv_sdk import upload, train
        
        # Upload
        upload("synthetic_big_cats", manifests["train"], "train")
        upload("synthetic_big_cats", manifests["validation"], "validation")
        
        # Train (just 1 epoch for demo)
        model_id = train(
            task_type=5,
            model_type="yolov8",
            train_dataset=manifests["train"],
            validation_dataset=manifests["validation"],
            epochs=1,
            batch=4
        )
        
        print(f"Training complete! Model: {model_id}")
        
    except Exception as e:
        print(f"CV SDK training skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SyntheticDataGen API Examples")
    print("=" * 60)
    
    # Run examples
    example_1_generate_cv_dataset()
    example_2_generate_rag_eval()
    example_3_generate_split_cv_dataset()
    example_4_generate_text_dataset()
    example_5_generate_audio_metadata()
    
    # Uncomment to run full pipeline demo (requires CV SDK)
    # example_6_full_pipeline_demo()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
