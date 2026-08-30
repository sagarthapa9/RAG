"""
Simple RAG Pipeline (legacy compatibility)
For the flexible version, use RAGPipeline from vector_store.py
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from  langchain_core.documents import Document

from  rag.document_reader import DocumentReader
from rag.chunking import DocumentChunker, TextChunk
from rag.vector_store import ChromaVectorStore

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
        persist_directory: str = "./data/chromadb",
        collection_name: str = "business_documents"
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        try:
            self.document_reader = DocumentReader()
            logger.info("Document reader initialized")
            
            self.chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Document chunker initialized (size: {chunk_size}, overlap: {chunk_overlap})")
            
            self.vector_store = ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_model=embedding_model
            )
            logger.info("Vector store initialized")
            
            logger.info("Simple RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise
    
    @staticmethod
    def _sanitize_metadata_value(value: Any) -> Union[bool, int, float, str, None]:
        """
        Sanitize a single metadata value for ChromaDB compatibility.
        ChromaDB only accepts Bool, Int, Float, or Str types.
        
        Args:
            value: The value to sanitize
            
        Returns:
            Sanitized value or None if value should be excluded
        """
        if value is None:
            return None  # Will be filtered out
        elif isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, dict, tuple)):
            # Convert complex types to string representation
            return str(value)
        else:
            # Convert other types to string
            return str(value)
    
    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[bool, int, float, str]]:
        """
        Sanitize metadata dictionary for ChromaDB compatibility.
        Removes None values and converts unsupported types.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        if not metadata:
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            if key is None:
                continue  # Skip None keys
            
            # Ensure key is string
            key_str = str(key) if key is not None else "unknown_key"
            
            # Sanitize value
            sanitized_value = RAGPipeline._sanitize_metadata_value(value)

            # Empty containers can't be meaningful filters — e.g. Swagger's
            # auto-generated placeholder {"additionalProp1": {}} — so drop them.
            if isinstance(value, (dict, list, tuple)) and not value:
                continue

            # Only include non-None values
            if sanitized_value is not None:
                sanitized[key_str] = sanitized_value
        
        return sanitized
    
    def process_documents(
        self,
        document_paths: List[Path],
        chunking_strategy: str = "recursive"
    ) -> int:
        """
        Process documents and add them to the vector store.
        
        Args:
            document_paths: List of document file paths
            chunking_strategy: Strategy for chunking documents
            
        Returns:
            Number of chunks processed
        """
        all_documents = []
        
        for doc_path in document_paths:
            logger.info(f"Processing document: {doc_path}")
            
            try:
                content, metadata = self.document_reader.read_document(doc_path)
                
                # Add file information to metadata
                base_metadata = {
                    'file_path': str(doc_path),
                    'filename': doc_path.name
                }
                
                # Merge and sanitize metadata
                if metadata:
                    base_metadata.update(metadata)
                
                chunks = self.chunker.chunk_document(
                    content, base_metadata, strategy=chunking_strategy
                )
                
                # Convert TextChunk objects to LangChain Document objects
                for chunk in chunks:
                    if isinstance(chunk, TextChunk):
                        # TextChunk object - extract content and metadata
                        chunk_metadata = {
                            **chunk.metadata,
                            'chunk_id': chunk.chunk_id,
                        }
                        
                        # Add optional attributes if they exist
                        if hasattr(chunk, 'start_char') and chunk.start_char is not None:
                            chunk_metadata['start_char'] = chunk.start_char
                        if hasattr(chunk, 'end_char') and chunk.end_char is not None:
                            chunk_metadata['end_char'] = chunk.end_char
                        
                        # Sanitize metadata before creating Document
                        sanitized_metadata = self._sanitize_metadata(chunk_metadata)
                        
                        doc = Document(
                            page_content=chunk.content,
                            metadata=sanitized_metadata
                        )
                    elif isinstance(chunk, dict):
                        # Dictionary format
                        content = chunk.get('content', '')
                        raw_metadata = chunk.get('metadata', {})
                        sanitized_metadata = self._sanitize_metadata(raw_metadata)
                        
                        doc = Document(
                            page_content=content,
                            metadata=sanitized_metadata
                        )
                    elif isinstance(chunk, Document):
                        # Already a Document object - sanitize its metadata
                        sanitized_metadata = self._sanitize_metadata(chunk.metadata)
                        doc = Document(
                            page_content=chunk.page_content,
                            metadata=sanitized_metadata
                        )
                    else:
                        # String content
                        sanitized_metadata = self._sanitize_metadata(base_metadata)
                        doc = Document(
                            page_content=str(chunk),
                            metadata=sanitized_metadata
                        )
                    
                    all_documents.append(doc)
                
                logger.info(f"Created {len(chunks)} chunks from {doc_path}")
                
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                continue
        
        if all_documents:
            try:
                document_ids = self.vector_store.add_documents(all_documents)
                logger.info(f"Successfully added {len(all_documents)} chunks to vector store")
                logger.info(f"Document IDs: {document_ids[:5]}..." if len(document_ids) > 5 else f"Document IDs: {document_ids}")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                raise
        else:
            logger.warning("No documents were processed successfully")
        
        return len(all_documents)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add raw texts to the vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        try:
            # Sanitize metadatas if provided
            sanitized_metadatas = None
            if metadatas:
                sanitized_metadatas = [
                    self._sanitize_metadata(metadata) for metadata in metadatas
                ]
            
            return self.vector_store.add_texts(texts, sanitized_metadatas)
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Sanitize filter metadata if provided
            sanitized_filter = None
            if filter_metadata:
                sanitized_filter = self._sanitize_metadata(filter_metadata)
                if not sanitized_filter:
                    # Everything dropped (e.g. Swagger's {} placeholder) == no filter
                    sanitized_filter = None
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=sanitized_filter
            )
            logger.info(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.exception("Error searching documents: %s", e)
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for relevant documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Sanitize filter metadata if provided
            sanitized_filter = None
            if filter_metadata:
                sanitized_filter = self._sanitize_metadata(filter_metadata)
                if not sanitized_filter:
                    # Everything dropped (e.g. Swagger's {} placeholder) == no filter
                    sanitized_filter = None
                
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=sanitized_filter
            )
            logger.info(f"Found {len(results)} relevant documents with scores for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents with scores: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a retriever instance for use with LangChain.
        
        Args:
            search_kwargs: Optional search parameters like {"k": 5}
            
        Returns:
            ChromaRetriever instance
        """
        try:
            search_kwargs = search_kwargs or {"k": 4}
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            Dictionary with system information
        """
        try:
            vector_store_info = self.vector_store.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            vector_store_info = {"error": str(e)}
        
        return {
            'pipeline_type': 'Simple RAG Pipeline (ChromaDB)',
            'chunking': {
                'chunk_size': self.chunker.chunk_size,
                'chunk_overlap': self.chunker.chunk_overlap
            },
            'vector_store': vector_store_info
        }
    
    def clear_vector_store(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.vector_store.clear_collection()
            if result:
                logger.info("Vector store cleared successfully")
            return result
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        try:
            info = self.vector_store.get_collection_info()
            return info.get('document_count', 0)
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


# Compatibility function for existing code
def create_rag_pipeline(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_directory: str = "./data/chromadb"
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Name of the embedding model
        persist_directory: Directory to persist ChromaDB data
        
    Returns:
        RAGPipeline instance
    """
    return RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )