#!/usr/bin/env python3
"""
Docker-ready test script for document reader
Save as: test_reader.py
"""
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, '/app/src' if '/app' in os.getcwd() else 'src')

try:
    from rag.simple_reader import SimpleDocumentReader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root or in Docker container")
    sys.exit(1)


def main():
    print("Business RAG Document Reader Test")
    print("=" * 50)
    
    # Initialize reader
    reader = SimpleDocumentReader(max_file_size_mb=50)
    
    # Check documents folder
    docs_folder = Path("documents")
    if not docs_folder.exists():
        docs_folder.mkdir(parents=True)
        print("Created documents folder")
    
    # Look for test documents
    supported_extensions = ['.pdf', '.docx', '.txt']
    found_files = []
    
    for ext in supported_extensions:
        found_files.extend(docs_folder.glob(f'*{ext}'))
    
    if not found_files:
        print("\nNo documents found!")
        print("Please add some test documents to the 'documents' folder:")
        print("  - PDF files (.pdf)")
        print("  - Word documents (.docx)")
        print("  - Text files (.txt)")
        print("\nExample business documents you could use:")
        print("  - Sample contract PDF")
        print("  - Company policy document")
        print("  - Business report")
        return
    
    print(f"\n Found {len(found_files)} document(s):")
    for i, file_path in enumerate(found_files, 1):
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {i}. {file_path.name} ({file_size_mb:.1f} MB)")
    
    # Test each document
    for file_path in found_files:
        print(f"\n{'='*60}")
        print(f"Testing: {file_path.name}")
        print(f"{'='*60}")
        
        # Extract text
        text = reader.read_document(str(file_path))
        
        if text:
            # Show statistics
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            
            print(f"Successfully extracted text!")
            print(f"   Statistics:")
            print(f"      Characters: {char_count:,}")
            print(f"      Words: {word_count:,}")
            print(f"      Lines: {line_count:,}")
            
            # Show preview
            preview_length = 300
            print(f"\n First {preview_length} characters:")
            print("-" * 50)
            print(text[:preview_length])
            if len(text) > preview_length:
                print("...")
            print("-" * 50)
            
            # Show last few lines for completeness check
            lines = text.strip().split('\n')
            if len(lines) > 10:
                print(f"\n Last 3 lines (to check completeness):")
                print("-" * 30)
                for line in lines[-3:]:
                    if line.strip():
                        print(line[:100] + ("..." if len(line) > 100 else ""))
                print("-" * 30)
                
        else:
            print(" Failed to extract text")
            print("   Check file format and try again")
    
    print(f"\n Document reader test complete!")
    print("Next steps:")
    print("  1. Try more documents")
    print("  2. Move to text chunking")
    print("  3. Add embeddings generation")


if __name__ == "__main__":
    main()