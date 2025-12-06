# Multimodal Retrieval-Augmented Generation (RAG) System

---

### 👥 **Team Members**

| Name          | GitHub Handle | Contribution |
| ------------- | ------------- | ------------ |
| Lucy Manalang | @LucyManalang |              |
| Yidian Chen   | @Llawlietcyd  |              |
|               |               |              |
|               |               |              |

---

## 🎯 **Project Highlights**

- Built a fully local **multimodal Retrieval-Augmented Generation (RAG)** pipeline capable of processing text, images, audio, and video.
- Designed a modular ingestion → indexing → query pipeline using:
  - Chunked text ingestion (Markdown/PDF)
  - Image captioning & OCR
  - Audio transcription (Whisper)
  - Video keyframe extraction + captions
- Implemented **hybrid retrieval (BM25 + dense embeddings)** with optional modality filtering.
- Integrated **local LLM inference via Ollama**, enabling fast, private on-device question answering with grounded citations.
- Created a unified interface that retrieves evidence from multiple modalities and produces cited answers grounded in retrieved content.

---

## **Setup and Installation**

### 1. Clone the Repository

`git clone <your-repo-url>`  
`cd <your-repo-name>`

### 2. Set Up Python Environment

`python3 -m venv venv`  
`source venv/bin/activate`

### 3. Install Dependencies

`pip install -r requirements.txt`

### 4. Install and Run Ollama (for local LLM)

Download for macOS: https://ollama.com/download  
Or install via Homebrew: `brew install ollama`

Start Ollama:  
`ollama serve`

Pull the model used in this project:  
`ollama pull llama3.2`

### 5. Run the Full Pipeline

`python3 main.py`

The system will:

1. Ingest files from `/data`
2. Build the hybrid index
3. Run a sample query using the retrieved context
4. Produce an answer with citations

---

## **Project Overview**

This project is part of the **Break Through Tech AI Studio Challenge**, where students develop practical AI systems for real-world applications.

Our system enables **multimodal information retrieval** across text, images, audio, and video, and answers questions using a **local LLM** grounded in retrieved evidence. This approach reflects real industry needs for systems that:

- Handle unstructured multimodal data
- Provide explainability through citations
- Preserve privacy via local processing
- Offer fast, low-cost inference without cloud APIs

The project demonstrates how organizations can leverage on-device AI for knowledge management, content understanding, and multimodal search.

---

## **Data Exploration**

### Supported Modalities

- **Text** (Markdown, PDF, TXT) — chunked using RecursiveCharacterSplitter
- **Images** (JPG, PNG) — captioned using BLIP + optional OCR
- **Audio** (WAV, MP3) — transcribed using Whisper
- **Video** (MP4) — keyframes + captions extracted from frames

### Preprocessing Highlights

All modalities are converted into normalized textual representations enriched with metadata:

- file_path
- modality
- timestamp (audio/video)
- frame_id (video)
- bbox_id (images with detection)

EDA focused on verifying:

- Chunk boundaries
- Caption quality
- Transcription accuracy
- Keyframe relevance

---

## **Model Development**

### Retrieval

- BM25 (sparse)
- Sentence-Transformers all-MiniLM-L6-v2 (dense)
- Reciprocal Rank Fusion (RRF) for hybrid retrieval

### Indexing

- FAISS vector store
- BM25 term-frequency scoring
- Unified multimodal document schema

### Generation

- Local LLM inference via **Llama 3.2** (Ollama)
- Prompting ensures:
  - grounded answers
  - strict citation format
  - no hallucinations

---

## **Results & Key Findings**

- Hybrid retrieval improves ranking over single-modality approaches.
- The system can answer questions requiring cross-modal reasoning.
- Local inference provides privacy, speed, and reliability without cloud costs.
- Answers consistently reference specific document IDs, timestamps, or frames.

---

## **Next Steps**

- Build a front-end UI for multimodal upload + querying.
- Add object detection metadata to improve vision grounding.
- Integrate a cross-encoder for improved re-ranking.
- Expand to long-context LLMs.
- Explore multimodal embedding models.

---

## **References**

- Sentence-Transformers documentation
- FAISS documentation
- Ollama documentation
- Whisper model paper
- BLIP model paper
- BM25 ranking research

---

## **Acknowledgements**

Thanks to our Challenge Advisor, Abhijay Rane, and Anote AI staff, Rajshri Jain and Natan Vidra, for guidance and support.
