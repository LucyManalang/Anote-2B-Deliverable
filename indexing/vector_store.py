"""
Vector Store Module - Dense embedding retrieval using FAISS
Supports efficient similarity search and nearest neighbor retrieval
"""

from typing import List, Dict, Optional
import numpy as np
import pickle
import os


class FAISSVectorStore:
    """
    FAISS-based vector storage and retrieval
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize vector store
        
        Args:
            dimension: Vector dimension
            index_type: FAISS index type
                       - "flat": Exact search (IndexFlatL2)
                       - "ivf": Inverted file index, faster but approximate (IndexIVFFlat)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF index requires training, using default parameters here
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 cluster centers
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.documents = []  # Store original documents
        self.is_trained = (index_type == "flat")  # Flat index doesn't need training
    
    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add vectors and corresponding documents
        
        Args:
            embeddings: Embedding vectors with shape (n_docs, dimension)
            documents: List of documents, each containing 'text' and 'metadata'
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}"
            )
        
        # For IVF index, training is required
        if self.index_type == "ivf" and not self.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings.astype('float32'))
            self.is_trained = True
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents to vector store (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Search for most similar documents
        
        Args:
            query_embedding: Query vector with shape (1, dimension) or (dimension,)
            top_k: Return top k results
            
        Returns:
            List of retrieval results, each containing text, metadata, score
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty, please call add() method to add documents first")
        
        # Ensure query vector has correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # FAISS search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, self.index.ntotal)
        )
        
        # Construct results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for invalid results
                # Convert L2 distance to similarity score (smaller distance = higher score)
                score = 1.0 / (1.0 + dist)
                results.append({
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx].get("metadata", {}),
                    "score": float(score),
                    "retriever": "faiss"
                })
        
        return results
    
    def save(self, filepath: str):
        """
        Save index and documents to disk
        
        Args:
            filepath: Save path (without extension)
        """
        # Save FAISS index
        self.faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents and metadata
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "is_trained": self.is_trained
            }, f)
        
        print(f"Vector store saved to: {filepath}.index and {filepath}.pkl")
    
    def load(self, filepath: str):
        """
        Load index and documents from disk
        
        Args:
            filepath: Load path (without extension)
        """
        # Load FAISS index
        self.index = self.faiss.read_index(f"{filepath}.index")
        
        # Load documents and metadata
        with open(f"{filepath}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
            self.is_trained = data["is_trained"]
        
        print(f"Vector store loaded: {self.index.ntotal} documents")
    
    def get_size(self) -> int:
        """Return the number of documents in the index"""
        return self.index.ntotal


if __name__ == "__main__":
    # Test code
    from embedder import Embedder
    
    # Create embedder
    embedder = Embedder()
    
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
            "text": "Computer vision is a fascinating field of artificial intelligence.",
            "metadata": {"source": "doc3.txt", "modality": "text"}
        },
        {
            "text": "Machine learning models can recognize cats and dogs in images.",
            "metadata": {"source": "doc4.txt", "modality": "text"}
        }
    ]
    
    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed(texts)
    
    # Create vector store
    vector_store = FAISSVectorStore(dimension=embedder.get_dimension())
    vector_store.add(embeddings, documents)
    
    # Test query
    query = "cats and dogs in pictures"
    query_embedding = embedder.embed(query)
    
    print(f"\nQuery: '{query}'")
    results = vector_store.search(query_embedding, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. [Score: {result['score']:.4f}] {result['text']}")
    
    # Test save and load
    vector_store.save("test_index")
    
    new_store = FAISSVectorStore(dimension=embedder.get_dimension())
    new_store.load("test_index")
    
    # Clean up test files
    os.remove("test_index.index")
    os.remove("test_index.pkl")
    print("\nTest complete, temporary files cleaned up")
