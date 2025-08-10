"""
Simple RAG Pipeline (legacy compatibility)
For the flexible version, use  RAGPipeline from vector_store.py
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .document_reader import DocumentReader
from .chunking import DocumentChunker, TextChunk
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Simple RAG pipeline using ChromaDB (for backward compatibility).
    For more flexibility, use FlexibleRAGPipeline instead.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./data/chromadb"
    ):
        self.document_reader = DocumentReader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        logger.info("Simple RAG Pipeline initialized (ChromaDB embedded)")
    
    def process_documents(
        self,
        document_paths: List[Path],
        chunking_strategy: str = "recursive"
    ) -> int:
        """Process documents and add them to the vector store."""
        all_chunks = []
        
        for doc_path in document_paths:
            logger.info(f"Processing document: {doc_path}")
            
            try:
                content, metadata = self.document_reader.read_document(doc_path)
                metadata['file_path'] = str(doc_path)
                metadata['filename'] = doc_path.name
                
                chunks = self.chunker.chunk_document(
                    content, metadata, strategy=chunking_strategy
                )
                
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {doc_path}")
                
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                continue
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            logger.info(f"Total chunks processed: {len(all_chunks)}")
        
        return len(all_chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        return self.vector_store.similarity_search(query, k, filter_metadata)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        return {
            'chunking': {
                'chunk_size': self.chunker.chunk_size,
                'chunk_overlap': self.chunker.chunk_overlap
            },
            'vector_store': self.vector_store.get_collection_info()
        }