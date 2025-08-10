from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
        - all-mpnet-base-v2: Higher quality, 768 dimensions, slower
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode([text])[0]