#!/usr/bin/env python3
"""
Test script for the RAG vector system.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the complete RAG system."""
    
    # Initialize RAG pipeline
    print(" Initializing RAG Pipeline...")
    rag = RAGPipeline(
        chunk_size=400,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Check for documents
    doc_folder = Path("documents")
    if not doc_folder.exists():
        print(" 'documents' folder not found. Create it and add your business documents.")
        return
    
    # Find documents
    document_files = []
    for ext in ["*.pdf", "*.docx", "*.txt"]:
        document_files.extend(doc_folder.glob(ext))
    
    if not document_files:
        print(" No documents found. Add PDF, DOCX, or TXT files to the 'documents' folder.")
        return
    
    print(f" Found {len(document_files)} documents:")
    for doc in document_files:
        print(f"   - {doc.name}")
    
    # Process documents
    print("\n  Processing documents...")
    total_chunks = rag.process_documents(document_files)
    print(f" Created {total_chunks} chunks")
    
    # Show system info
    info = rag.get_system_info()
    print(f"\n System Info:")
    print(f"   - Chunk size: {info['chunking']['chunk_size']} tokens")
    print(f"   - Chunk overlap: {info['chunking']['chunk_overlap']} tokens")
    print(f"   - Documents in vector store: {info['vector_store']['document_count']}")
    print(f"   - Embedding model: {info['vector_store']['embedding_model']}")
    print(f"   - Embedding dimension: {info['vector_store']['embedding_dimension']}")
    
    # Interactive search
    print(f"\n🔍 Interactive Search (type 'quit' to exit):")
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Search
        results = rag.search(query, k=3)
        
        if not results:
            print("No results found.")
            continue
        
        print(f"\n Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
            print(f"Source: {result['metadata']['filename']}")
            print(f"Content: {result['content'][:200]}...")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()