"""
Indexing Module - Step 2: Representation & Indexing

This module provides indexing and retrieval capabilities for the multimodal retrieval system, including:
- Text embedding (Embedder)
- BM25 keyword retrieval (BM25Retriever)
- FAISS vector store (FAISSVectorStore)
- Hybrid retriever (HybridRetriever)
- Index pipeline (IndexPipeline)
"""

from .embedder import Embedder
from .bm25_retriever import BM25Retriever
from .vector_store import FAISSVectorStore
from .hybrid_retriever import HybridRetriever
from .index_pipeline import IndexPipeline

__all__ = [
    "Embedder",
    "BM25Retriever",
    "FAISSVectorStore",
    "HybridRetriever",
    "IndexPipeline"
]

__version__ = "0.1.0"
