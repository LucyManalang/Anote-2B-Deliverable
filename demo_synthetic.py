"""
Demo script with SyntheticDataGen integration.

This demonstrates the full pipeline:
1. Generate synthetic training data
2. Train CV models
3. Generate synthetic evaluation data
4. Evaluate the full system
"""

import os
import json
from pathlib import Path

# Import SyntheticDataGen
from synthetic_data import generate, generate_rag_eval_set, generate_cv_dataset

# Import CV SDK
from cv_sdk import upload, train, predict, evaluate

# Import RAG components
from main import main as run_rag_pipeline


def demo_synthetic_cv_training():
    """Demo: Generate synthetic data and train CV models"""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Data Generation + CV Training")
    print("=" * 60)
    
    # Step 1: Generate synthetic CV dataset
    print("\n[Step 1] Generating synthetic object detection dataset...")
    manifests = generate_cv_dataset(
        classes=["tiger", "lion", "leopard", "cheetah"],
        num_images=30,
        resolution="640x480",
        output_dir="synthetic_data/outputs/big_cats"
    )
    
    print(f"\nGenerated datasets:")
    print(f"  Train: {manifests['train']}")
    print(f"  Val: {manifests['validation']}")
    print(f"  Test: {manifests['test']}")
    
    # Step 2: Upload to CV SDK
    print("\n[Step 2] Uploading to CV SDK...")
    try:
        upload("synthetic_big_cats", manifests["train"], "train")
        upload("synthetic_big_cats", manifests["validation"], "validation")
        print("[OK] Upload complete")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return
    
    # Step 3: Train YOLOv8
    print("\n[Step 3] Training YOLOv8 (1 epoch for demo)...")
    try:
        model_id = train(
            task_type=5,
            model_type="yolov8",
            train_dataset=manifests["train"],
            validation_dataset=manifests["validation"],
            epochs=1,
            batch=4,
            imgsz=640
        )
        print(f"[OK] Training complete! Model: {model_id}")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Predict on test set
    print("\n[Step 4] Running predictions on test set...")
    try:
        preds = predict(
            model_type="yolov8",
            test_data=manifests["test"],
            labels=["tiger", "lion", "leopard", "cheetah"],
            model_id=model_id,
            confidence_threshold=0.25
        )
        print(f"[OK] Generated {len(preds)} predictions")
        
        # Save predictions
        pred_file = "synthetic_data/outputs/big_cats/predictions.jsonl"
        with open(pred_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved to: {pred_file}")
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return
    
    # Step 5: Evaluate
    print("\n[Step 5] Evaluating model performance...")
    try:
        report = evaluate(
            ground_truths=manifests["test"],
            predictions=pred_file,
            out_dir="synthetic_data/outputs/big_cats/eval"
        )
        print("[OK] Evaluation complete!")
        print(f"  Report: {report}")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Synthetic CV Training Demo Complete!")
    print("=" * 60)


def demo_synthetic_rag_evaluation():
    """Demo: Generate synthetic RAG evaluation set and test"""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic RAG Evaluation")
    print("=" * 60)
    
    # Step 1: Generate evaluation set
    print("\n[Step 1] Generating synthetic RAG evaluation set...")
    eval_path = generate_rag_eval_set(
        num_samples=10,
        output_path="synthetic_data/outputs/rag_eval.json"
    )
    
    # Step 2: Show samples
    print("\n[Step 2] Sample questions:")
    with open(eval_path, "r") as f:
        samples = json.load(f)
    
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"    Q: {sample['question']}")
        print(f"    Gold Answer: {sample['gold_answer']}")
        print(f"    Gold Context IDs: {sample['gold_context_ids']}")
    
    # Step 3: Run evaluation (if RAG pipeline is ready)
    print("\n[Step 3] Running RAG evaluation...")
    print("  (Requires indexed documents - run main.py first)")
    
    try:
        from evaluation_pipeline import EvaluationPipeline, EvalSample
        from indexing.index_pipeline import IndexPipeline
        from query_fusion.query_engine import QueryEngine
        
        # Try to load existing index
        idx = IndexPipeline()
        # Check if index exists
        if os.path.exists("faiss_index"):
            print("  [OK] Found existing index")
            
            qeng = QueryEngine(idx, llm_backend="ollama", model="llama3.2")
            evaluator = EvaluationPipeline(idx, qeng)
            
            # Convert to EvalSample objects
            eval_samples = [EvalSample(**s) for s in samples[:5]]  # Test first 5
            
            print("  Running evaluation on 5 samples...")
            report = evaluator.evaluate_dataset(
                eval_samples,
                out_path="synthetic_data/outputs/rag_eval_report.json"
            )
            
            print(f"\n  Results:")
            print(f"    Avg Recall@5: {report['summary']['avg_recall@5']:.3f}")
            print(f"    Avg MRR: {report['summary']['avg_mrr']:.3f}")
            print(f"    Avg Similarity: {report['summary']['avg_similarity']:.3f}")
            print(f"    Avg Groundedness: {report['summary']['avg_groundedness']:.3f}")
        else:
            print("  [WARNING] No index found. Run main.py first to build index.")
            
    except Exception as e:
        print(f"  [WARNING] Evaluation skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Synthetic RAG Evaluation Demo Complete!")
    print("=" * 60)


def demo_generate_all_modalities():
    """Demo: Generate synthetic data for all modalities"""
    print("\n" + "=" * 60)
    print("DEMO: Generate All Modality Types")
    print("=" * 60)
    
    # Images
    print("\n[1] Generating synthetic images...")
    img_result = generate(
        task_type="image",
        prompt="Various animals in natural habitats",
        num_rows=15,
        columns=["image_path", "bboxes", "labels"],
        params={
            "resolution": "512x512",
            "classes": ["cat", "dog", "bird", "fish"],
            "bbox_augment": True
        },
        media_dir="synthetic_data/outputs/demo_images"
    )
    print(f"  [OK] Generated {img_result['num_generated']} images")
    
    # Text
    print("\n[2] Generating synthetic text documents...")
    text_result = generate(
        task_type="text",
        prompt="Technical articles about AI and ML",
        num_rows=10,
        params={
            "topics": ["machine_learning", "computer_vision", "nlp"],
            "doc_length": "short"
        },
        media_dir="synthetic_data/outputs/demo_text"
    )
    print(f"  [OK] Generated {text_result['num_generated']} documents")
    
    # Audio
    print("\n[3] Generating synthetic audio metadata...")
    audio_result = generate(
        task_type="audio",
        prompt="Podcast conversations",
        num_rows=8,
        params={
            "duration_range": [10, 30],
            "add_noise": False
        },
        media_dir="synthetic_data/outputs/demo_audio"
    )
    print(f"  [OK] Generated {audio_result['num_generated']} audio entries")
    
    # Video
    print("\n[4] Generating synthetic video metadata...")
    video_result = generate(
        task_type="video",
        prompt="Tutorial videos",
        num_rows=5,
        params={
            "fps": 30,
            "duration_range": [20, 60]
        },
        media_dir="synthetic_data/outputs/demo_video"
    )
    print(f"  [OK] Generated {video_result['num_generated']} video entries")
    
    print("\n" + "=" * 60)
    print("All Modality Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION DEMO")
    print("=" * 60)
    print("\nAvailable demos:")
    print("  1. Synthetic CV Training Pipeline")
    print("  2. Synthetic RAG Evaluation")
    print("  3. Generate All Modalities")
    print("  4. Run All Demos")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect demo (1-4, or Enter for all): ").strip() or "4"
    
    if choice == "1":
        demo_synthetic_cv_training()
    elif choice == "2":
        demo_synthetic_rag_evaluation()
    elif choice == "3":
        demo_generate_all_modalities()
    elif choice == "4":
        demo_generate_all_modalities()
        demo_synthetic_cv_training()
        demo_synthetic_rag_evaluation()
    else:
        print("Invalid choice!")
    
    print("\n[OK] Demo complete! Check synthetic_data/outputs/ for generated files.")
