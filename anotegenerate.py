"""
anotegenerate - Anote AI SyntheticDataGen API wrapper

This module provides the exact API interface specified in READMEANOTE2B.md.
It wraps the synthetic_data module to match the expected import path.

Usage (as per README):
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
"""

# Import from synthetic_data implementation
from synthetic_data import generate, generate_rag_eval_set, generate_cv_dataset

__all__ = ["generate", "generate_rag_eval_set", "generate_cv_dataset"]
