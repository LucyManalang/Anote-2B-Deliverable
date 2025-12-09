"""
Query & Fusion Engine
---------------------
Uses:
 - HybridRetriever (BM25 + Vector)
 - Modality-aware retrieval
 - RRF / weighted fusion
 - LLM generation with grounded citations
"""

from typing import List, Dict, Optional
from indexing.index_pipeline import IndexPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QueryEngine:
    def __init__(
        self,
        pipeline: IndexPipeline,
        llm_backend: str = "ollama",
        model: str = "llama3.2",    
        top_k: int = 5
    ):
        """
        Args:
            pipeline: an initialized IndexPipeline
            llm_model: LLM to use for generation
            top_k: number of final retrieval results to feed LLM
        """
        self.pipeline = pipeline
        self.llm_backend = llm_backend
        self.model = model
        self.top_k = top_k

        if llm_backend == "ollama":
            import requests
            self.session = requests.Session()

        # elif llm_backend == "hf":
        #     from transformers import AutoTokenizer, AutoModelForCausalLM
        #     import torch
        #     self.tokenizer = AutoTokenizer.from_pretrained(model)
        #     self.hf_model = AutoModelForCausalLM.from_pretrained(
        #         model,
        #         torch_dtype=torch.float16,
        #         device_map="auto"
        #     )

        else:
            raise ValueError(f"Unknown llm_backend: {llm_backend}")
        

    def retrieve(self, query: str, fusion: str = "rrf", modality: Optional[str] = None):
        """
        Retrieve and fuse results

        Args:
            query: query text
            fusion: "rrf" or "weighted"
            modality: filter, e.g. "image" | "text" | "audio" | "video"

        Returns:
            List of {text, metadata, score}
        """
    
        return self.pipeline.search(
            query=query,
            top_k=self.top_k,
            method=fusion,
            modality_filter=modality
        )

    def format_citations(self, results: List[Dict]) -> str:
        """
        Turn metadata into human-readable grounded citations.
        """
        citation_lines = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            modality = meta.get("modality", "unknown")
            source = meta.get("source", None)
            timestamp = meta.get("timestamp_sec", None)
            frame = meta.get("frame_path", None)
            bbox = meta.get("bbox_id", None)

            line = f"[{i}] ({modality})"

            if source:
                line += f" file='{source}'"
            if timestamp is not None:
                line += f", t={timestamp:.2f}s"
            if frame:
                line += f", frame='{frame}'"
            if bbox:
                line += f", bbox={bbox}"

            citation_lines.append(line)
        return "\n".join(citation_lines)

    def ask(
        self,
        query: str,
        fusion: str = "rrf",
        modality: Optional[str] = None
    ) -> Dict:
        """
        Full query pipeline:
            1. Retrieve fused context
            2. Construct LLM prompt with citations
            3. Ask LLM to answer using ONLY retrieved context + cite sources
        """
        retrieved = self.retrieve(query, fusion=fusion, modality=modality)

        # Build context text
        context_block = "\n\n".join(
            f"[{i}] {r['text']}" for i, r in enumerate(retrieved, 1)
        )
        citations_block = self.format_citations(retrieved)

        prompt = f"""
You are a multimodal RAG assistant. 
You must answer using ONLY the retrieved context below.
When referencing a source, cite it using [1], [2], etc. based on the indexing.

CONTEXT:
{context_block}

CITATION METADATA:
{citations_block}

QUESTION:
{query}

Now provide a grounded answer:
"""

        # Use Ollama API endpoint instead of subprocess
        import requests
        import json
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            output_text = response.json().get("response", "").strip()
            
            if not output_text:
                output_text = "Based on the retrieved context, I cannot provide a complete answer."
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

        return {
            "answer": output_text,
            "retrieved": retrieved,
            "citations": citations_block
        }
