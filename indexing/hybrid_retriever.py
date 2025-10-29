"""
Hybrid Retriever - Combines BM25 and vector retrieval
Uses Reciprocal Rank Fusion (RRF) to fuse results from multiple retrievers
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


class HybridRetriever:
    """
    Hybrid retriever combining sparse retrieval (BM25) and dense retrieval (vectors)
    """
    
    def __init__(
        self, 
        bm25_retriever=None,
        vector_store=None,
        embedder=None,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            bm25_retriever: BM25 retriever instance
            vector_store: Vector store instance
            embedder: Embedder instance (for converting queries to vectors)
            alpha: Weight for BM25 vs vector retrieval
                   0.0 = BM25 only
                   1.0 = vector retrieval only
                   0.5 = balanced
        """
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        method: str = "rrf",
        modality_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid search
        
        Args:
            query: Query text
            top_k: Return top k results
            method: Fusion method
                   - "rrf": Reciprocal Rank Fusion
                   - "weighted": Weighted score fusion
            modality_filter: Modality filter (optional), e.g. "image", "text", "audio"
            
        Returns:
            List of retrieval results
        """
        all_results = []
        
        # BM25 retrieval
        if self.bm25_retriever is not None:
            try:
                bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
                all_results.extend([{**r, "source": "bm25"} for r in bm25_results])
            except Exception as e:
                print(f"BM25 retrieval failed: {e}")
        
        # Vector retrieval
        if self.vector_store is not None and self.embedder is not None:
            try:
                query_embedding = self.embedder.embed(query)
                vector_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
                all_results.extend([{**r, "source": "vector"} for r in vector_results])
            except Exception as e:
                print(f"Vector retrieval failed: {e}")
        
        # Modality filtering
        if modality_filter:
            all_results = [
                r for r in all_results 
                if r.get("metadata", {}).get("modality") == modality_filter
            ]
        
        # Fuse results
        if method == "rrf":
            fused_results = self._reciprocal_rank_fusion(all_results, top_k)
        elif method == "weighted":
            fused_results = self._weighted_fusion(all_results, top_k)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self, 
        results: List[Dict], 
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) algorithm
        
        RRF score = sum(1 / (k + rank)) across all retrievers
        
        Args:
            results: Results from all retrievers
            top_k: Return top k results
            k: RRF parameter (typically 60)
            
        Returns:
            Fused results list
        """
        # Group by source
        results_by_source = defaultdict(list)
        for result in results:
            source = result.get("source", "unknown")
            results_by_source[source].append(result)
        
        # Calculate RRF score for each document
        doc_scores = defaultdict(float)
        doc_info = {}
        
        for source, source_results in results_by_source.items():
            for rank, result in enumerate(source_results, start=1):
                # Use text as document identifier
                doc_id = result["text"]
                
                # RRF formula
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += rrf_score
                
                # Save document info (only keep first occurrence)
                if doc_id not in doc_info:
                    doc_info[doc_id] = result
        
        # Sort and return top-k
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        fused_results = []
        for doc_id, score in sorted_docs:
            result = doc_info[doc_id].copy()
            result["score"] = float(score)
            result["fusion_method"] = "rrf"
            fused_results.append(result)
        
        return fused_results
    
    def _weighted_fusion(
        self, 
        results: List[Dict], 
        top_k: int
    ) -> List[Dict]:
        """
        Weighted score fusion
        
        Normalize and weight-combine BM25 and vector retrieval scores
        
        Args:
            results: Results from all retrievers
            top_k: Return top k results
            
        Returns:
            Fused results list
        """
        # Separate BM25 and vector results
        bm25_results = [r for r in results if r.get("source") == "bm25"]
        vector_results = [r for r in results if r.get("source") == "vector"]
        
        # Normalize scores
        def normalize_scores(results_list):
            if not results_list:
                return {}
            
            scores = [r["score"] for r in results_list]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            
            normalized = {}
            for r in results_list:
                doc_id = r["text"]
                if max_score > min_score:
                    norm_score = (r["score"] - min_score) / (max_score - min_score)
                else:
                    norm_score = 1.0
                normalized[doc_id] = (norm_score, r)
            
            return normalized
        
        bm25_norm = normalize_scores(bm25_results)
        vector_norm = normalize_scores(vector_results)
        
        # Combine scores
        doc_scores = {}
        all_doc_ids = set(bm25_norm.keys()) | set(vector_norm.keys())
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, (0.0, None))[0]
            vector_score = vector_norm.get(doc_id, (0.0, None))[0]
            
            # Weighted combination
            combined_score = (1 - self.alpha) * bm25_score + self.alpha * vector_score
            
            # Get document info
            doc = bm25_norm.get(doc_id, (None, None))[1] or vector_norm.get(doc_id, (None, None))[1]
            
            doc_scores[doc_id] = (combined_score, doc)
        
        # Sort and return top-k
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:top_k]
        
        fused_results = []
        for doc_id, (score, doc) in sorted_docs:
            result = doc.copy()
            result["score"] = float(score)
            result["fusion_method"] = "weighted"
            fused_results.append(result)
        
        return fused_results
    
    def search_by_modality(
        self, 
        query: str, 
        modalities: List[str],
        top_k_per_modality: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search separately by modality
        
        Args:
            query: Query text
            modalities: List of modalities, e.g. ["text", "image", "audio"]
            top_k_per_modality: Number of results to return per modality
            
        Returns:
            Dictionary of results grouped by modality
        """
        results_by_modality = {}
        
        for modality in modalities:
            results = self.search(
                query, 
                top_k=top_k_per_modality,
                modality_filter=modality
            )
            results_by_modality[modality] = results
        
        return results_by_modality


if __name__ == "__main__":
    # Test code
    from embedder import Embedder
    from bm25_retriever import BM25Retriever
    from vector_store import FAISSVectorStore
    
    # Test documents
    documents = [
        {
            "text": "The cat sits on the mat.",
            "metadata": {"source": "doc1.txt", "modality": "text"}
        },
        {
            "text": "A dog is playing in the garden.",
            "metadata": {"source": "doc2.txt", "modality": "text"}
        },
        {
            "text": "Caption: A cat sleeping on a red couch. OCR: ",
            "metadata": {"source": "img1.jpg", "modality": "image"}
        },
        {
            "text": "Computer vision is a fascinating field of artificial intelligence.",
            "metadata": {"source": "doc3.txt", "modality": "text"}
        },
        {
            "text": "Machine learning models can recognize cats and dogs in images.",
            "metadata": {"source": "doc4.txt", "modality": "text"}
        },
        {
            "text": "Caption: Dog running in park with frisbee. OCR: PARK",
            "metadata": {"source": "img2.jpg", "modality": "image"}
        }
    ]
    
    # Create retrievers
    embedder = Embedder()
    bm25_retriever = BM25Retriever()
    vector_store = FAISSVectorStore(dimension=embedder.get_dimension())
    
    # Index documents
    bm25_retriever.index(documents)
    
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed(texts)
    vector_store.add(embeddings, documents)
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        embedder=embedder,
        alpha=0.5
    )
    
    # Test queries
    print("=== Test 1: RRF Fusion ===")
    query = "cat sleeping on furniture"
    results = hybrid_retriever.search(query, top_k=3, method="rrf")
    
    print(f"\nQuery: '{query}'")
    for i, result in enumerate(results, 1):
        modality = result.get("metadata", {}).get("modality", "unknown")
        print(f"  {i}. [{modality}] [Score: {result['score']:.4f}] {result['text'][:60]}...")
    
    print("\n=== Test 2: Weighted Fusion ===")
    results = hybrid_retriever.search(query, top_k=3, method="weighted")
    
    print(f"\nQuery: '{query}'")
    for i, result in enumerate(results, 1):
        modality = result.get("metadata", {}).get("modality", "unknown")
        print(f"  {i}. [{modality}] [Score: {result['score']:.4f}] {result['text'][:60]}...")
    
    print("\n=== Test 3: Modality Filtering ===")
    results = hybrid_retriever.search(query, top_k=3, modality_filter="image")
    
    print(f"\nQuery: '{query}' (images only)")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [Score: {result['score']:.4f}] {result['text'][:60]}...")
    
    print("\n=== Test 4: Search by Modality ===")
    results_by_mod = hybrid_retriever.search_by_modality(
        query, 
        modalities=["text", "image"],
        top_k_per_modality=2
    )
    
    for modality, results in results_by_mod.items():
        print(f"\n{modality.upper()}:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [Score: {result['score']:.4f}] {result['text'][:60]}...")
