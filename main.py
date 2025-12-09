"""
Main Pipeline - Multimodal RAG System with Computer Vision SDK

This is the main entry point for the Anote-2B Deliverable project.
Running this script demonstrates:

1. Multimodal RAG Pipeline:
   - Ingestion: Text, Images, Audio, Video
   - Indexing: Hybrid retrieval (BM25 + dense embeddings)
   - Query: LLM-grounded answers with citations
   - Evaluation: Retrieval metrics (Recall@k, nDCG@k)

2. Computer Vision SDK:
   - upload(): Register datasets for training
   - train(): Fine-tune object detection models (YOLO/Faster R-CNN)
   - predict(): Run inference on test images
   - evaluate(): Compute mAP, mIoU, confusion matrix

3. Synthetic Data Generation:
   - Generate training/eval datasets
   - Create adversarial test cases
   - Produce evaluation reports

Usage:
    python main.py              # Run full demo (default)
    python main.py --rag-only   # Run RAG pipeline only
    python main.py --cv-only    # Run CV SDK demo only
"""

from ingestion.text_ingest import ingest_text
from ingestion.image_ingest import ingest_images
from ingestion.audio_ingest import ingest_audio
from ingestion.video_ingest import ingest_video
from indexing.index_pipeline import IndexPipeline
from query_fusion.query_engine import QueryEngine
import os
import sys
import json
import time


def create_sample_data():
    """Create sample data files if they don't exist"""
    print("\n[Setup] Creating sample data...")
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Create sample markdown file
    if not os.path.exists("data/test.md"):
        with open("data/test.md", "w", encoding="utf-8") as f:
            f.write("""# Sample Document: The Lantern of Memories

Mira walked along the forest trail, her thoughts wandering. Life had been busy—so busy that she hadn't stopped to reflect in months. As she rounded a corner, she spotted a small, weathered lantern hanging from a low branch.

Curious, she lifted it. The moment her fingers touched the cold metal, the lantern began to glow. Images flickered in the soft light—memories she'd forgotten. Moments of joy, regret, and unfulfilled dreams danced before her eyes.

The lantern showed her the roads she hadn't taken, the words she hadn't said. It illuminated choices she'd made without thinking and paths she'd abandoned out of fear.

Mira stood there for a long time, tears streaming down her face. When the lantern finally dimmed, she carefully hung it back on the branch. She pulled out her phone and made a list—not of things to do, but of things to start again.
""")
        print("  Created: data/test.md")
    
    # Create sample images if they don't exist
    from PIL import Image
    import random
    
    for img_name, color_base in [("image_test_1.jpg", (220, 180, 140)), ("image_test_2.jpg", (180, 140, 100))]:
        img_path = os.path.join("data", img_name)
        if not os.path.exists(img_path):
            # Create a gradient image
            img = Image.new('RGB', (640, 640))
            pixels = img.load()
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    r = int(color_base[0] + (i / img.size[0]) * 35)
                    g = int(color_base[1] + (j / img.size[1]) * 35)
                    b = int(color_base[2] + random.randint(-20, 20))
                    pixels[i, j] = (r, g, b)
            img.save(img_path)
            print(f"  Created: {img_path}")
    
    # Create ground truth for CV demo
    if not os.path.exists("data/test_gt.jsonl"):
        with open("data/test_gt.jsonl", "w") as f:
            f.write(json.dumps({"image_path":"data/image_test_1.jpg","labels":["tiger"],"bboxes":[[50,50,300,300,"tiger"]]}) + "\n")
            f.write(json.dumps({"image_path":"data/image_test_2.jpg","labels":["mug"],"bboxes":[[70,70,200,250,"mug"]]}) + "\n")
        print("  Created: data/test_gt.jsonl")
    
    print("[OK] Sample data ready\n")


def run_multimodal_rag():
    """Run the complete multimodal RAG pipeline"""
    print("\n" + "="*60)
    print("MULTIMODAL RAG PIPELINE")
    print("="*60)
    
    print("\n>> Step 1: Ingesting Data...")
    
    # Text Ingestion
    text_docs = []
    if os.path.exists("data/test.md"):
        text_docs = ingest_text("data/test.md")
        print(f"  [Text] Ingested {len(text_docs)} chunks from Markdown")
    
    # Image Ingestion
    image_files = [f for f in ["data/image_test_1.jpg", "data/image_test_2.jpg"] if os.path.exists(f)]
    image_docs = ingest_images(image_files) if image_files else []
    print(f"  [Images] Ingested {len(image_docs)} images")

    # Audio Ingestion
    audio_docs = ingest_audio("data/")
    if audio_docs:
        print(f"  [Audio] Ingested {len(audio_docs)} audio segments")
    
    # Video Ingestion
    video_docs = ingest_video("data/")
    if video_docs:
        print(f"  [Video] Ingested {len(video_docs)} video frames")

    docs = text_docs + image_docs + audio_docs + video_docs

    print(f"\n  Total: {len(docs)} document chunks from {sum([bool(text_docs), bool(image_docs), bool(audio_docs), bool(video_docs)])} modalities")
    
    if len(docs) == 0:
        print("\n[WARNING] No documents found. Run with sample data creation.")
        return None

    print("\n>> Step 2: Building Hybrid Index...")
    pipeline = IndexPipeline()
    pipeline.index_documents(docs)
    
    # Save index for later use
    pipeline.vector_store.save("vector_index")
    print("[OK] Index saved to vector_index.index")

    print("\n>> Step 3: Query Processing...")
    engine = QueryEngine(
        pipeline,
        llm_backend="ollama",
        model="llama3.2",
    )
    
    # Run multiple queries to showcase capabilities
    queries = [
        "What animal appears in the images and what is the main theme of the text document?",
        "Describe the lantern and its significance.",
        "What did Mira decide to do after seeing the lantern?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        try:
            response = engine.ask(query)
            print("\n=== ANSWER ===")
            print(response["answer"])
            print("\n=== CITATIONS ===")
            print(response["citations"])
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
        
        if i < len(queries):
            print("\n" + "-"*60)
    
    return pipeline


def run_cv_demo():
    """Run CV SDK demonstration"""
    print("\n" + "="*60)
    print("COMPUTER VISION SDK DEMO")
    print("="*60)
    
    try:
        from cv_sdk import upload, train, predict, evaluate
        
        print("\n[1] Uploading Dataset...")
        if not os.path.exists("data/test_gt.jsonl"):
            print("[ERROR] Ground truth file not found. Run with --setup first.")
            return
        
        upload("demo_dataset", "data/test_gt.jsonl", "train")
        upload("demo_dataset", "data/test_gt.jsonl", "validation")
        print("[OK] Dataset uploaded")
        
        print("\n[2] Training Model (Faster R-CNN, 1 epoch demo)...")
        model_id = train(
            task_type=5,
            model_type="faster_rcnn",
            train_dataset="datasets/demo_dataset/train/manifest.jsonl",
            validation_dataset="datasets/demo_dataset/validation/manifest.jsonl",
            epochs=1,
            batch=2
        )
        print(f"[OK] Model trained: {model_id}")
        
        print("\n[3] Running Predictions...")
        preds = predict(
            model_type="faster_rcnn",
            test_data="data/image_test_1.jpg",
            labels=["tiger", "mug"],
            model_id=model_id,
            confidence_threshold=0.2
        )
        print(f"[OK] Detected {len(preds[0]['boxes'])} objects")
        
        print("\n[4] Evaluating Model...")
        all_preds = predict(
            model_type="faster_rcnn",
            test_data="data/test_gt.jsonl",
            labels=["tiger", "mug"],
            model_id=model_id
        )
        
        pred_file = "cv_predictions.jsonl"
        with open(pred_file, "w") as f:
            for p in all_preds:
                f.write(json.dumps(p) + "\n")
        
        report = evaluate("data/test_gt.jsonl", pred_file, out_dir="cv_eval_results")
        print(f"[OK] Evaluation complete: {report}")
        
    except ImportError as e:
        print(f"[ERROR] CV SDK not available: {e}")
    except Exception as e:
        print(f"[ERROR] CV demo failed: {e}")


def print_usage():
    """Print usage information"""
    print("""
Usage: python main.py [OPTIONS]

OPTIONS:
  --help          Show this help message
  --setup         Create sample data files
  --rag-only      Run only multimodal RAG pipeline
  --cv-only       Run only CV SDK demo (upload/train/predict/evaluate)
  
EXAMPLES:
  python main.py                    # Run full demo (RAG + CV) [DEFAULT]
  python main.py --setup            # Setup sample data and run full demo
  python main.py --rag-only         # Run RAG pipeline only
  python main.py --cv-only          # Run CV SDK demo only
  
As per README.md:
  Running `python main.py` will:
    1. Ingest files from /data
    2. Build the hybrid index
    3. Run sample queries
    4. Produce answers with citations
    5. Demonstrate CV SDK capabilities
    """)


def main():
    """
    Main entry point - runs the full demo by default.
    
    As per README requirements, running `python main.py` should:
    1. Ingest files from /data
    2. Build the hybrid index
    3. Run sample queries using retrieved context
    4. Produce answers with citations
    5. Demonstrate CV SDK capabilities
    """
    args = sys.argv[1:]
    
    # Parse arguments
    if "--help" in args or "-h" in args:
        print_usage()
        return
    
    # Setup sample data if needed
    if "--setup" in args or not os.path.exists("data"):
        create_sample_data()
    
    # Determine what to run
    if "--cv-only" in args:
        if not os.path.exists("data/test_gt.jsonl"):
            create_sample_data()
        run_cv_demo()
    elif "--rag-only" in args:
        run_multimodal_rag()
    else:
        # DEFAULT: Run full demo (RAG + CV) as per README requirement
        print("\n" + "="*70)
        print(" "*10 + "ANOTE-2B MULTIMODAL RAG + CV SDK DEMO")
        print("="*70)
        print("\nThis demo showcases:")
        print("  1. Multimodal RAG: Text + Image + Audio + Video retrieval")
        print("  2. CV SDK: Object detection training & evaluation")
        print("  3. Hybrid indexing with BM25 + dense embeddings")
        print("  4. Local LLM inference with grounded citations")
        print("="*70 + "\n")
        
        run_multimodal_rag()
        run_cv_demo()
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "DEMO COMPLETE")
    print("="*70)
    print("\nGenerated Artifacts:")
    if os.path.exists("vector_index.index"):
        print("  [RAG] vector_index.index - FAISS vector store")
    if os.path.exists("vector_index.pkl"):
        print("  [RAG] vector_index.pkl - Document metadata")
    if os.path.exists("cv_eval_results"):
        print("  [CV]  cv_eval_results/ - Evaluation metrics & confusion matrix")
        if os.path.exists("cv_eval_results/metrics.csv"):
            print("        -> metrics.csv: Precision, Recall, F1, mAP, mIoU")
        if os.path.exists("cv_eval_results/confusion_matrix.png"):
            print("        -> confusion_matrix.png: Visual confusion matrix")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Review evaluation metrics in cv_eval_results/metrics.csv")
    print("  2. Add your own multimodal data to data/ directory")
    print("  3. Modify queries in query_fusion/ to test different questions")
    print("  4. Generate synthetic data: python synthetic_data/example_usage.py")
    print("  5. Run full synthetic demo: python demo_synthetic.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
