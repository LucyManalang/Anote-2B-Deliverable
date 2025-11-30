"""
Unified Index Pipeline - Indexes ingested data into retrieval system
Integrates the complete flow from ingestion to indexing
"""

import sys
import os
from typing import List, Dict, Optional

# Add parent directory to path for importing ingestion module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import Embedder
from bm25_retriever import BM25Retriever
from vector_store import FAISSVectorStore
from hybrid_retriever import HybridRetriever


class IndexPipeline:
    """
    Complete indexing pipeline from data ingestion to retrieval system
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        alpha: float = 0.5
    ):
        """
        Initialize index pipeline
        
        Args:
            embedding_model: Embedding model name
            index_type: FAISS index type ("flat" or "ivf")
            alpha: Weight parameter for hybrid retrieval
        """
        print("Initializing index pipeline...")
        
        # Create embedder
        self.embedder = Embedder(model_name=embedding_model)
        print(f"✓ Embedder loaded: {embedding_model} (dimension: {self.embedder.get_dimension()})")
        
        # Create retrievers
        self.bm25_retriever = BM25Retriever()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedder.get_dimension(),
            index_type=index_type
        )
        
        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_store=self.vector_store,
            embedder=self.embedder,
            alpha=alpha
        )
        
        print("✓ Index pipeline initialization complete\n")
    
    def index_documents(self, documents: List[Dict]):
        """
        Index document collection
        
        Args:
            documents: List of documents, each containing 'text' and 'metadata'
                      Output format from ingestion module
        """
        if not documents:
            print("Warning: No documents to index")
            return
        
        print(f"Starting to index {len(documents)} documents...")
        
        # 1. Index to BM25
        print("  → Creating BM25 index...")
        self.bm25_retriever.index(documents)
        
        # 2. Generate embeddings and index to vector store
        print("  → Generating text embeddings...")
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        print("  → Adding to vector store...")
        self.vector_store.add(embeddings, documents)
        
        print("✓ Indexing complete!\n")
    
    def index_from_ingestion(
        self,
        file_paths: List[Dict],
        use_ingestion: bool = True
    ):
        """
        Index directly from file paths (calls ingestion module)
        
        Args:
            file_paths: List of files, each element is {'path': str, 'type': str}
                       type can be 'text', 'image', 'audio', 'video'
            use_ingestion: Whether to use ingestion module (default True)
        """
        if not use_ingestion:
            raise ValueError("Currently must use ingestion module")
        
        # Import ingestion module
        try:
            from ingestion.run_ingestion import run_all
        except ImportError:
            raise ImportError(
                "Cannot import ingestion module. Please ensure ingestion directory exists with necessary files."
            )
        
        print(f"Ingesting data from {len(file_paths)} files...")
        
        # Run ingestion
        documents = run_all(file_paths)
        
        print(f"✓ Ingestion complete, obtained {len(documents)} document chunks\n")
        
        # Index documents
        self.index_documents(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "rrf",
        modality_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search query
        
        Args:
            query: Query text
            top_k: Return top k results
            method: Fusion method ("rrf" or "weighted")
            modality_filter: Modality filter (optional)
            
        Returns:
            List of retrieval results
        """
        return self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
            method=method,
            modality_filter=modality_filter
        )
    
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
            modalities: List of modalities
            top_k_per_modality: Number of results per modality
            
        Returns:
            Dictionary of results grouped by modality
        """
        return self.hybrid_retriever.search_by_modality(
            query=query,
            modalities=modalities,
            top_k_per_modality=top_k_per_modality
        )
    
    def save(self, save_dir: str):
        """
        Save index to disk
        
        Args:
            save_dir: Save directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vector store
        vector_path = os.path.join(save_dir, "vector_store")
        self.vector_store.save(vector_path)
        
        # BM25 index can be quickly rebuilt, not saving here
        # Can use pickle to save if needed
        
        print(f"✓ Index saved to: {save_dir}")
    
    def load(self, save_dir: str):
        """
        Load index from disk
        
        Args:
            save_dir: Save directory
        """
        vector_path = os.path.join(save_dir, "vector_store")
        self.vector_store.load(vector_path)
        
        print(f"✓ Index loaded from {save_dir}")
    
    def get_stats(self) -> Dict:
        """
        Get index statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "bm25_documents": self.bm25_retriever.get_corpus_size(),
            "vector_documents": self.vector_store.get_size(),
            "embedding_dimension": self.embedder.get_dimension(),
            "embedding_model": self.embedder.model_name
        }


def demo():
    """Demonstrate index pipeline usage"""
    
    print("=" * 60)
    print("Index Pipeline Demo")
    print("=" * 60 + "\n")
    
    # Create pipeline
    pipeline = IndexPipeline()
    
    # Example documents (simulating ingestion output)
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
        },
        {
            "text": "The weather is sunny and warm today.",
            "metadata": {"source": "doc5.txt", "modality": "text"}
        }
    ]
    
    # Index documents
    pipeline.index_documents(documents)
    
    # Display statistics
    stats = pipeline.get_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test queries
    queries = [
        "cat sleeping on furniture",
        "dogs playing outside",
        "artificial intelligence and machine learning"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        # RRF retrieval
        print("\n[Method: RRF Fusion]")
        results = pipeline.search(query, top_k=3, method="rrf")
        for i, result in enumerate(results, 1):
            modality = result.get("metadata", {}).get("modality", "unknown")
            source = result.get("metadata", {}).get("source", "unknown")
            print(f"  {i}. [{modality}] {source}")
            print(f"     Score: {result['score']:.4f}")
            print(f"     Content: {result['text'][:60]}...")
    
    # Test modality filtering
    print(f"\n{'='*60}")
    print("Modality Filter Test: Images Only")
    print('='*60)
    
    results = pipeline.search("animals", top_k=5, modality_filter="image")
    for i, result in enumerate(results, 1):
        source = result.get("metadata", {}).get("source", "unknown")
        print(f"  {i}. {source}")
        print(f"     Score: {result['score']:.4f}")
        print(f"     Content: {result['text'][:60]}...")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
