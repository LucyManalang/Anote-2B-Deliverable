"""
Ingestion Module - Step 1: Ingestion / Pre-processing

This module provides multimodal data ingestion capabilities, including:
- Text ingestion (ingest_text) - Markdown/PDF
- Image ingestion (ingest_images) - BLIP + OCR
- Audio ingestion (ingest_audio) - Whisper ASR
- Video ingestion (ingest_video) - Keyframe extraction + BLIP
"""

from .text_ingest import ingest_text
from .image_ingest import ingest_images
from .audio_ingest import ingest_audio
from .video_ingest import ingest_video
from .run_ingestion import run_ingestion, run_all

__all__ = [
    "ingest_text",
    "ingest_images",
    "ingest_audio",
    "ingest_video",
    "run_ingestion",
    "run_all"
]

__version__ = "0.1.0"
