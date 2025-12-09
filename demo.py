import os
import json
from cv_sdk import upload, train, predict, evaluate
from main import main as run_rag_pipeline

def run_cv_demo():
    print("\n=== CV SDK DEMO ===")
    
    # 1. Upload
    print("\n[1] Uploading dataset...")
    # We use the same file for train/val/test for this tiny demo
    # Ensure data exists
    if not os.path.exists("data/test_gt.jsonl"):
        print("Error: data/test_gt.jsonl not found.")
        return

    upload("demo_dataset", "data/test_gt.jsonl", "train")
    upload("demo_dataset", "data/test_gt.jsonl", "validation")
    upload("demo_dataset", "data/test_gt.jsonl", "test")
    
    # 2. Train (Faster R-CNN)
    print("\n[2] Training Faster R-CNN (1 epoch for demo)...")
    # Point to the uploaded manifest
    train_manifest = "datasets/demo_dataset/train/manifest.jsonl"
    val_manifest = "datasets/demo_dataset/validation/manifest.jsonl"
    
    try:
        model_id = train(
            task_type=5,
            model_type="faster_rcnn",
            train_dataset=train_manifest,
            validation_dataset=val_manifest,
            epochs=1,
            batch=2
        )
        print(f"Training finished. Model saved at: {model_id}")
    except Exception as e:
        print(f"Training failed (expected if no GPU/dependencies or data issues): {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Predict
    print("\n[3] Running Prediction...")
    labels = ["tiger", "mug"] # We know these from the data
    
    # Predict on a single image
    try:
        preds = predict(
            model_type="faster_rcnn",
            test_data="data/image_test_1.jpg",
            labels=labels,
            model_id=model_id,
            confidence_threshold=0.1
        )
        print("Predictions (single image):", json.dumps(preds, indent=2))
    except Exception as e:
        print(f"Prediction failed: {e}")

    # 4. Evaluate
    print("\n[4] Evaluating...")
    try:
        all_preds = predict(
            model_type="faster_rcnn",
            test_data="data/test_gt.jsonl",
            labels=labels,
            model_id=model_id
        )
        
        # Save all preds
        pred_file = "demo_all_preds.jsonl"
        with open(pred_file, "w") as f:
            for p in all_preds:
                f.write(json.dumps(p) + "\n")
                
        report = evaluate("data/test_gt.jsonl", pred_file, out_dir="demo_eval_results")
        print("Evaluation Report:", report)
    except Exception as e:
        print(f"Evaluation failed: {e}")

def run_rag_demo():
    print("\n=== MULTIMODAL RAG DEMO ===")
    try:
        run_rag_pipeline()
    except Exception as e:
        print(f"RAG pipeline failed: {e}")

if __name__ == "__main__":
    # Create dummy data if not exists
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists("data/test_gt.jsonl"):
        print("Creating dummy data...")
        with open("data/test_gt.jsonl", "w") as f:
            f.write(json.dumps({"image_path":"data/image_test_1.jpg","labels":["tiger"],"bboxes":[[50,50,300,300,"tiger"]]}) + "\n")
            f.write(json.dumps({"image_path":"data/image_test_2.jpg","labels":["mug"],"bboxes":[[70,70,200,250,"mug"]]}) + "\n")

    # Create dummy images if not exists
    for img_name in ["image_test_1.jpg", "image_test_2.jpg"]:
        img_path = os.path.join("data", img_name)
        if not os.path.exists(img_path):
            print(f"Creating dummy image: {img_path}")
            from PIL import Image
            # Create a simple RGB image
            img = Image.new('RGB', (640, 640), color = (73, 109, 137))
            img.save(img_path)

    run_cv_demo()
    run_rag_demo()
