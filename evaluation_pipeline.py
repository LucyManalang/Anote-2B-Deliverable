# evaluation_pipeline.py
"""
Evaluation pipeline for multimodal RAG + CV SDK.

Features:
 - Evaluate retrieval (Recall@K, MRR)
 - Evaluate generation (semantic similarity vs. gold)
 - Evaluate CV predictions using cv_sdk.evaluate()
 - Produce combined JSON report and CSV
"""
import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

from indexing.index_pipeline import IndexPipeline
from query_fusion.query_engine import QueryEngine

# Try to import cv_sdk evaluate
try:
    from cv_sdk.evaluate import evaluate as cv_evaluate
except Exception:
    cv_evaluate = None

@dataclass
class EvalSample:
    id: str
    question: str
    gold_answer: str
    gold_context_ids: List[int]
    # optional fields for CV evaluation
    gt_manifest: Optional[str] = None
    label_list: Optional[List[str]] = None
    cv_predictions: Optional[str] = None  # path to preds jsonl


class EvaluationPipeline:
    def __init__(self, pipeline: IndexPipeline, engine: QueryEngine, sim_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pipeline = pipeline
        self.engine = engine
        self.sim = SentenceTransformer(sim_model_name)

    @staticmethod
    def recall_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int = 5) -> float:
        if not expected_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        return len(set(retrieved_k) & set(expected_ids)) / len(set(expected_ids))

    @staticmethod
    def mrr(retrieved_ids: List[int], expected_ids: List[int]) -> float:
        for idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                return 1.0 / idx
        return 0.0

    def evaluate_qa_sample(self, sample: EvalSample) -> Dict[str, Any]:
        out = self.engine.ask(sample.question)
        answer = out["answer"]
        retrieved = out["retrieved"]

        # Map retrieved to "ids" - if your retriever stores chunk IDs add them, else use positional IDs
        retrieved_ids = []
        for i, r in enumerate(retrieved, start=1):
            # try metadata 'chunk_id' or fallback to ordinal index
            mid = r.get("metadata", {}).get("chunk_id") or r.get("metadata", {}).get("id") or i
            retrieved_ids.append(int(mid))

        recall5 = self.recall_at_k(retrieved_ids, sample.gold_context_ids, k=5)
        mrr_v = self.mrr(retrieved_ids, sample.gold_context_ids)

        # Semantic similarity between generated and gold answer
        emb_pred = self.sim.encode(answer, convert_to_tensor=True)
        emb_gold = self.sim.encode(sample.gold_answer, convert_to_tensor=True)
        sim_score = float(util.cos_sim(emb_pred, emb_gold))

        # Citation groundedness: count [1],[2] references that are within retrieved length
        import re
        cited_idx = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
        if cited_idx:
            grounded = sum(1 for c in cited_idx if 1 <= c <= len(retrieved_ids)) / len(cited_idx)
        else:
            grounded = 0.0

        return {
            "id": sample.id,
            "question": sample.question,
            "answer": answer,
            "recall@5": recall5,
            "mrr": mrr_v,
            "similarity": sim_score,
            "citation_groundedness": grounded,
            "retrieved_ids": retrieved_ids
        }

    def evaluate_dataset(self, samples: List[EvalSample], out_path: str = "evaluation_report.json") -> Dict[str, Any]:
        results = []
        for s in samples:
            res = self.evaluate_qa_sample(s)
            results.append(res)

        # aggregate
        summary = {
            "avg_recall@5": float(np.mean([r["recall@5"] for r in results])),
            "avg_mrr": float(np.mean([r["mrr"] for r in results])),
            "avg_similarity": float(np.mean([r["similarity"] for r in results])),
            "avg_groundedness": float(np.mean([r["citation_groundedness"] for r in results])),
            "samples": results
        }

        # Optionally run CV evaluation per-sample if provided
        cv_reports = {}
        if cv_evaluate is not None:
            for s in samples:
                if s.gt_manifest and s.cv_predictions:
                    out = cv_evaluate(s.gt_manifest, s.cv_predictions, out_dir=f"cv_eval_{s.id}")
                    cv_reports[s.id] = out

        report = {
            "summary": summary,
            "cv_reports": cv_reports
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


# CLI example
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", help="path to evaluation json containing samples", required=False)
    parser.add_argument("--output", help="output report path", default="evaluation_report.json")
    parser.add_argument("--generate", help="generate synthetic eval set", action="store_true")
    parser.add_argument("--num_samples", help="number of samples to generate", type=int, default=20)
    args = parser.parse_args()

    # Generate synthetic eval set if requested
    if args.generate:
        try:
            from synthetic_data import generate_rag_eval_set
            print(f"Generating {args.num_samples} synthetic evaluation samples...")
            eval_path = generate_rag_eval_set(
                num_samples=args.num_samples,
                output_path=args.eval_json or "synthetic_rag_eval.json"
            )
            args.eval_json = eval_path
        except Exception as e:
            print(f"Failed to generate synthetic data: {e}")
            print("Falling back to example sample...")

    # Example usage if no eval_json provided: create a tiny sample
    if not args.eval_json:
        samples = [
            EvalSample(id="example1", question="What animal appears in the document?", gold_answer="A tiger.", gold_context_ids=[1])
        ]
    else:
        with open(args.eval_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = [EvalSample(**r) for r in raw]

    # initialize pipeline + engine (assumes indices already built)
    idx = IndexPipeline()
    
    # Try to load existing index
    try:
        if os.path.exists("vector_index.index"):
            idx.vector_store.load("vector_index")
            print("[OK] Loaded existing index")
        else:
            print("[WARNING] No index found. Please run main.py first to build the index.")
    except Exception as e:
        print(f"[WARNING] Could not load index: {e}")
    
    qeng = QueryEngine(idx, llm_backend="ollama", model="llama3.2")

    evaluator = EvaluationPipeline(idx, qeng)
    report = evaluator.evaluate_dataset(samples, out_path=args.output)
    print(json.dumps(report, indent=2))
