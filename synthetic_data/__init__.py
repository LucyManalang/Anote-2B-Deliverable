"""
SyntheticDataGen module for generating training/test datasets.

This module provides APIs to generate synthetic data for:
- Images with object detection annotations
- Text documents for RAG evaluation
- Audio transcripts with Q&A pairs
- Video captions with temporal annotations
"""

from .generator import generate, generate_rag_eval_set, generate_cv_dataset

__all__ = ["generate", "generate_rag_eval_set", "generate_cv_dataset"]
