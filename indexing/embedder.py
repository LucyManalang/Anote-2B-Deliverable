"""
Embedding Module - Converts text to dense vector representations
Uses sentence-transformers for text embedding
"""

from typing import List, Union
import numpy as np

class Embedder:
    """
    Text embedder for converting text to dense vectors
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder
        
        Args:
            model_name: sentence-transformers model name
                      Default: all-MiniLM-L6-v2 (384 dimensions)
                      Other options:
                      - "all-mpnet-base-v2" (768 dimensions, higher quality)
                      - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual support)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text to embedding vectors
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            numpy array with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100  # Show progress bar for large batches
        )
        
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch process text embeddings (suitable for large-scale data)
        
        Args:
            texts: List of texts
            batch_size: Number of texts to process per batch
            
        Returns:
            numpy array with shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Return the dimension of embedding vectors"""
        return self.dimension


if __name__ == "__main__":
    # Test code
    embedder = Embedder()
    
    # Test single text
    text = "This is a test sentence for embedding."
    embedding = embedder.embed(text)
    print(f"Model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.get_dimension()}")
    print(f"Single text embedding shape: {embedding.shape}")
    
    # Test multiple texts
    texts = [
        "The cat sits on the mat.",
        "A dog is playing in the garden.",
        "Computer vision is a fascinating field."
    ]
    embeddings = embedder.embed(texts)
    print(f"Batch text embedding shape: {embeddings.shape}")
    
    # Calculate similarity example
    from numpy.linalg import norm
    similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    print(f"Cosine similarity between first two sentences: {similarity:.4f}")

