"""
BM25 Retriever - Keyword-based sparse retrieval
Implements traditional term frequency-inverse document frequency retrieval
"""

from typing import List, Dict, Tuple
import numpy as np


class BM25Retriever:
    """
    BM25 keyword retriever
    Uses BM25 algorithm for document ranking and retrieval
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Document length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.corpus_metadata = []
        self.tokenized_corpus = []
        self.bm25 = None
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer (can be replaced with more sophisticated tokenizer)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization + lowercase
        # TODO: Can use nltk, spacy, etc. for more powerful tokenization
        return text.lower().split()
    
    def index(self, documents: List[Dict]):
        """
        Index document collection
        
        Args:
            documents: List of documents, each containing 'text' and 'metadata'
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required. Install with: pip install rank-bm25"
            )
        
        self.corpus = [doc["text"] for doc in documents]
        self.corpus_metadata = [doc.get("metadata", {}) for doc in documents]
        
        # Tokenize
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"BM25 indexing complete: {len(self.corpus)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Query text
            top_k: Return top k results
            
        Returns:
            List of retrieval results, each containing text, metadata, score
        """
        if self.bm25 is None:
            raise ValueError("Please call index() method to index documents first")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Calculate BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Construct results
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only return results with relevance
                results.append({
                    "text": self.corpus[idx],
                    "metadata": self.corpus_metadata[idx],
                    "score": float(scores[idx]),
                    "retriever": "bm25"
                })
        
        return results
    
    def get_corpus_size(self) -> int:
        """Return the number of indexed documents"""
        return len(self.corpus)


if __name__ == "__main__":
    # Test code
    retriever = BM25Retriever()
    
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
        },
        {
            "text": "The garden has many beautiful flowers and trees.",
            "metadata": {"source": "doc5.txt", "modality": "text"}
        }
    ]
    
    # Index documents
    retriever.index(documents)
    
    # Test queries
    queries = [
        "cat in the garden",
        "computer vision machine learning",
        "flowers and plants"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [Score: {result['score']:.4f}] {result['text'][:60]}...")
