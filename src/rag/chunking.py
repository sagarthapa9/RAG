import tiktoken
from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class TextChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    token_count: int

class DocumentChunker:
    """
    Intelligent text chunking for RAG systems.
    Supports multiple chunking strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        strategy: str = "recursive"
    ) -> List[TextChunk]:
        """
        Chunk document using specified strategy.
        
        Strategies:
        - recursive: Split by paragraphs, then sentences, then words
        - semantic: Split by semantic boundaries (future enhancement)
        - fixed: Fixed-size chunks with overlap
        """
        
        if strategy == "recursive":
            return self._recursive_chunk(text, metadata)
        elif strategy == "fixed":
            return self._fixed_chunk(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _recursive_chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Recursively split text by paragraphs, sentences, then words.
        Maintains semantic coherence.
        """
        chunks = []
        
        # First, split by paragraphs
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_counter = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            combined = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.encoding.encode(combined))
            
            if token_count <= self.chunk_size:
                current_chunk = combined
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, metadata, chunk_counter
                    ))
                    chunk_counter += 1
                
                # Handle large paragraphs that exceed chunk size
                if len(self.encoding.encode(paragraph)) > self.chunk_size:
                    # Split large paragraph by sentences
                    sentence_chunks = self._split_by_sentences(paragraph, metadata, chunk_counter)
                    chunks.extend(sentence_chunks)
                    chunk_counter += len(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, metadata, chunk_counter
            ))
        
        return chunks
    
    def _split_by_sentences(self, text: str, metadata: Dict[str, Any], start_counter: int) -> List[TextChunk]:
        """Split large text by sentences."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        chunk_counter = start_counter
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            combined = current_chunk + ". " + sentence if current_chunk else sentence
            token_count = len(self.encoding.encode(combined))
            
            if token_count <= self.chunk_size:
                current_chunk = combined
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, metadata, chunk_counter
                    ))
                    chunk_counter += 1
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, metadata, chunk_counter
            ))
        
        return chunks
    
    def _fixed_chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Fixed-size chunking with overlap."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        chunk_counter = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append(self._create_chunk(
                chunk_text, metadata, chunk_counter
            ))
            
            # Move start position with overlap
            start += (self.chunk_size - self.chunk_overlap)
            chunk_counter += 1
        
        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_id: int) -> TextChunk:
        """Create a TextChunk object."""
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_index'] = chunk_id
        
        return TextChunk(
            content=text,
            metadata=chunk_metadata,
            chunk_id=f"{metadata.get('filename', 'unknown')}_{chunk_id}",
            token_count=len(self.encoding.encode(text))
        )