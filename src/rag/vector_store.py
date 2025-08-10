import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path

from .chunking import TextChunk
from rag.embedding import EmbeddingGenerator

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Vector store using ChromaDB for document retrieval.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "business_documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"ChromaDB initialized. Collection: {collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
    
    def add_documents(self, chunks: List[TextChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of TextChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Prepare data for ChromaDB
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to collection")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with content, metadata, and scores
        """
        logger.info(f"Searching for: '{query}' (top {k} results)")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['documents'][0]:  # Check if results exist
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.embedding_dim
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset collection: {self.collection_name}")