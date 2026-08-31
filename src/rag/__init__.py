"""
Business RAG System

A production-ready Retrieval-Augmented Generation system for business documents.
Supports multiple vector databases, document types, and deployment modes.
"""

__version__ = "0.2.0"
__author__ = "Business RAG Team"

# Main exports
from .document_reader import DocumentReader
from .chunking import DocumentChunker, TextChunk
from .vector_store import (
    ChromaVectorStore,

)

__all__ = [
    "DocumentReader",
    "DocumentChunker",
    "TextChunk",
    "ChromaVectorStore"
]