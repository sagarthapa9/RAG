"""
Fixed ChromaVectorStore with proper abstract method implementations
"""

import hashlib
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)


class ChromaRetriever(BaseRetriever):
    """Custom retriever for ChromaVectorStore"""

    vector_store: Any
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 4})

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents for a query

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, **self.search_kwargs)


class ChromaVectorStore:
    """
    Custom ChromaDB vector store.

    Plain class (not a LangChain VectorStore subclass): LangChain's base
    became a pydantic model that fights arbitrary attributes across versions.
    The methods LangChain needs are still implemented (add_documents,
    add_texts, similarity_search, similarity_search_with_score, as_retriever),
    so it drops into chains/retrievers that expect that surface.
    """
    
    def __init__(
        self,
        collection_name: str = "business_documents",
        persist_directory: str = "./data/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaVectorStore
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the sentence transformer model
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create persist directory with proper permissions
        try:
            os.makedirs(persist_directory, exist_ok=True, mode=0o755)
            
            # Test write permissions
            test_file = os.path.join(persist_directory, "test_write_permissions")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            logger.info(f"Created persist directory with write permissions: {persist_directory}")
            
        except Exception as e:
            logger.warning(f"Cannot write to {persist_directory}: {e}")
            # Fallback to a writable directory
            import tempfile
            persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db")
            os.makedirs(persist_directory, exist_ok=True, mode=0o755)
            self.persist_directory = persist_directory
            logger.info(f"Using fallback directory: {persist_directory}")
        
        # Initialize embedding model
        try:
            # Set cache directories with proper permissions for HuggingFace models
            cache_dir = os.path.join(persist_directory, "model_cache")
            os.makedirs(cache_dir, exist_ok=True, mode=0o755)
            
            # Set environment variables for HuggingFace cache
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            
            self.embedding_model = SentenceTransformer(embedding_model, cache_folder=cache_dir)
            logger.info(f"Loaded embedding model: {embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Initialize ChromaDB client with proper settings
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {collection_name}")
            logger.info(f"Collection document count: {self.collection.count()}")

            # Warm up: force Chroma to finish loading the HNSW vector index
            # before any query is served. count() reads SQLite metadata and is
            # correct immediately, but query() reads the in-memory index — which
            # can come back empty on a cold start right after a crash/force-kill
            # (the known index/metadata out-of-sync failure mode). get() blocks
            # until the index segments are loaded, so one cheap read here makes
            # the first real query reliable.
            self.collection.get(limit=1)
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            # Try one more fallback approach
            try:
                logger.info("Attempting fallback ChromaDB initialization...")
                fallback_dir = os.path.join(os.path.expanduser("~"), ".chroma_db")
                os.makedirs(fallback_dir, exist_ok=True, mode=0o755)
                
                self.client = chromadb.PersistentClient(
                    path=fallback_dir,
                    settings=Settings(allow_reset=True)
                )
                
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                self.persist_directory = fallback_dir
                logger.info(f"ChromaDB initialized with fallback directory: {fallback_dir}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback ChromaDB initialization also failed: {str(fallback_error)}")
                raise RuntimeError(f"Could not initialize ChromaDB. Original error: {str(e)}, Fallback error: {str(fallback_error)}")
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        try:
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            embeddings_list = embeddings.tolist()
            
            # Generate stable IDs (content hash, not builtin hash() which is
            # salted per-process and would duplicate chunks on re-ingestion)
            ids = [
                f"doc_{i}_{hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
                for i, text in enumerate(texts)
            ]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store (required by VectorStore)
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Convert to Document objects
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        return self.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search (required by VectorStore)
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding_list = query_embedding.tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=filter
            )

            # An empty result needs a reason. With no filter, an empty result on
            # a non-empty collection means the HNSW index wasn't loaded yet (any
            # query gets top-k); retry once. With a filter, 0 hits usually means
            # the filter matched nothing — log it so it's never a silent mystery.
            if not (results['documents'] and results['documents'][0]):
                logger.warning(
                    "Similarity search returned 0 hits: query=%r k=%s where=%r (collection count=%d)",
                    query, k, filter, self.collection.count()
                )
                if not filter and self.collection.count() > 0:
                    logger.warning("Cold-start: retrying once (index may not be loaded yet)")
                    results = self.collection.query(
                        query_embeddings=[query_embedding_list],
                        n_results=k,
                        where=filter
                    )
            
            # Convert results to Document objects
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    
                    # Add distance/score to metadata
                    if results['distances'] and results['distances'][0]:
                        metadata['score'] = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Found {len(documents)} similar documents for query")
            return documents

        except Exception as e:
            # Not silent: log the full traceback so an empty result is never
            # mistaken for "no documents matched" when it's actually a failure.
            logger.exception("Error in similarity search: %s", e)
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding_list = query_embedding.tolist()[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=filter
            )

            # An empty result needs a reason. With no filter, an empty result on
            # a non-empty collection means the HNSW index wasn't loaded yet (any
            # query gets top-k); retry once. With a filter, 0 hits usually means
            # the filter matched nothing — log it so it's never a silent mystery.
            if not (results['documents'] and results['documents'][0]):
                logger.warning(
                    "Similarity search returned 0 hits: query=%r k=%s where=%r (collection count=%d)",
                    query, k, filter, self.collection.count()
                )
                if not filter and self.collection.count() > 0:
                    logger.warning("Cold-start: retrying once (index may not be loaded yet)")
                    results = self.collection.query(
                        query_embeddings=[query_embedding_list],
                        n_results=k,
                        where=filter
                    )
            
            # Convert results to Document objects with scores
            documents_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 1.0
                    
                    # Convert distance to similarity score
                    score = 1 - distance
                    
                    document = Document(page_content=content, metadata=metadata)
                    documents_with_scores.append((document, score))
            
            logger.info(f"Found {len(documents_with_scores)} similar documents with scores")
            return documents_with_scores

        except Exception as e:
            logger.exception("Error in similarity search with score: %s", e)
            return []
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Required abstract method implementation for BaseRetriever compatibility
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        return self.similarity_search(query, k=4)
    
    def as_retriever(self, **kwargs) -> ChromaRetriever:
        """
        Create a retriever from this vector store
        
        Args:
            **kwargs: Arguments to pass to the retriever (e.g., search_kwargs={"k": 5})
            
        Returns:
            ChromaRetriever instance
        """
        search_kwargs = kwargs.pop("search_kwargs", {"k": 4})
        return ChromaRetriever(vector_store=self, search_kwargs=search_kwargs)
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Get all IDs and delete them
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            logger.info("Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        metadatas: Optional[List[Dict]] = None,
        collection_name: str = "business_documents",
        persist_directory: str = "./data/chromadb",
        **kwargs
    ) -> "ChromaVectorStore":
        """
        Create ChromaVectorStore from texts (required by VectorStore)
        
        Args:
            texts: List of texts
            embedding_model: Embedding model name
            metadatas: Optional metadata
            collection_name: Collection name
            persist_directory: Persist directory
            
        Returns:
            ChromaVectorStore instance
        """
        vector_store = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        if texts:
            vector_store.add_texts(texts, metadatas)
        
        return vector_store
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "business_documents",
        persist_directory: str = "./data/chromadb",
        **kwargs
    ) -> "ChromaVectorStore":
        """
        Create ChromaVectorStore from documents
        
        Args:
            documents: List of Document objects
            embedding_model: Embedding model name
            collection_name: Collection name
            persist_directory: Persist directory
            
        Returns:
            ChromaVectorStore instance
        """
        vector_store = cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        if documents:
            vector_store.add_documents(documents)
        
        return vector_store


def create_vector_store(
    collection_name: str = "business_documents",
    persist_directory: str = "./data/chromadb",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> ChromaVectorStore:
    """
    Factory function to create vector store
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data
        embedding_model: Embedding model name
        
    Returns:
        ChromaVectorStore instance
    """
    return ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )


if __name__ == "__main__":
    # Test the vector store
    try:
        # Create vector store
        vector_store = create_vector_store()
        
        # Test adding documents
        test_docs = [
            Document(
                page_content="This is a test document about machine learning.",
                metadata={"source": "test1.txt", "type": "test"}
            ),
            Document(
                page_content="This is another document about artificial intelligence.",
                metadata={"source": "test2.txt", "type": "test"}
            )
        ]
        
        ids = vector_store.add_documents(test_docs)
        print(f"Added documents with IDs: {ids}")
        
        # Test search
        results = vector_store.similarity_search("machine learning", k=2)
        print(f"Search results: {len(results)} documents found")
        
        for doc in results:
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")
        
        # Test retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever._get_relevant_documents("AI")
        print(f"Retrieved via retriever: {len(retrieved_docs)} documents")
        
        # Test collection info
        info = vector_store.get_collection_info()
        print(f"Collection info: {info}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")