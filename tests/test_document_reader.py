#!/usr/bin/env python3
"""
Test script for the document reader.
Tests PDF, DOCX, and TXT file processing.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.document_reader import DocumentReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test document reader functionality."""
    
    print("🚀 Document Reader Test")
    print("=" * 40)
    
    # Initialize reader
    reader = DocumentReader()
    
    print(f"📋 Supported extensions: {reader.get_supported_extensions()}")
    
    # Check for documents directory
    doc_folder = Path("documents")
    if not doc_folder.exists():
        print("❌ 'documents' folder not found. Creating it...")
        doc_folder.mkdir()
        print("📁 Created 'documents' folder. Please add some test documents.")
        return
    
    # Scan for documents
    documents = reader.scan_directory(doc_folder)
    
    if not documents:
        print("❌ No supported documents found in 'documents' folder.")
        print("   Add some PDF, DOCX, or TXT files to test.")
        return
    
    print(f"📄 Found {len(documents)} documents:")
    for doc in documents:
        print(f"   - {doc.name} ({doc.suffix})")
    
    # Test each document
    print(f"\n🔄 Testing document reading...")
    
    for doc_path in documents:
        print(f"\n--- Testing: {doc_path.name} ---")
        
        try:
            content, metadata = reader.read_document(doc_path)
            
            # Display metadata
            print(f"✅ Successfully read: {doc_path.name}")
            print(f"   📊 Size: {metadata['file_size_bytes']} bytes")
            print(f"   📝 Content: {metadata['character_count']} chars, {metadata['word_count']} words")
            print(f"   📄 Lines: {metadata['line_count']} total, {metadata['non_empty_line_count']} non-empty")
            
            # Show document-specific metadata
            if doc_path.suffix.lower() == '.pdf':
                if 'page_count' in metadata:
                    print(f"   📄 Pages: {metadata['page_count']}")
                if 'pdf_title' in metadata and metadata['pdf_title']:
                    print(f"   📋 Title: {metadata['pdf_title']}")
                if 'pdf_author' in metadata and metadata['pdf_author']:
                    print(f"   👤 Author: {metadata['pdf_author']}")
            
            elif doc_path.suffix.lower() == '.docx':
                if 'paragraph_count' in metadata:
                    print(f"   📄 Paragraphs: {metadata['paragraph_count']}")
                if 'table_count' in metadata:
                    print(f"   📊 Tables: {metadata['table_count']}")
                if 'docx_title' in metadata and metadata['docx_title']:
                    print(f"   📋 Title: {metadata['docx_title']}")
                if 'docx_author' in metadata and metadata['docx_author']:
                    print(f"   👤 Author: {metadata['docx_author']}")
            
            elif doc_path.suffix.lower() == '.txt':
                if 'text_encoding' in metadata:
                    print(f"   🔤 Encoding: {metadata['text_encoding']}")
            
            # Show content preview
            content_preview = content[:200].replace('\n', ' ').strip()
            if len(content) > 200:
                content_preview += "..."
            print(f"   📖 Preview: {content_preview}")
            
        except Exception as e:
            print(f"❌ Error reading {doc_path.name}: {e}")
    
    print(f"\n✅ Document reader test completed!")
    print("\nNext steps:")
    print("1. Test the full RAG pipeline: python test_vector_system.py")
    print("2. Add more documents to test with different formats")

if __name__ == "__main__":
    main()