"""
Document processing utilities for chunking and preparing text.
"""
from typing import List, Dict, Any
import re


class DocumentProcessor:
    """Process and chunk documents for vector database storage."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(min(100, end - start)):
                    if text[end - i] in '.!?\n':
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_paper(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a paper into chunks with metadata.
        
        Args:
            paper_data: Dictionary containing paper information
            
        Returns:
            List of dictionaries with chunked text and metadata
        """
        # Combine title and summary for chunking
        full_text = f"Title: {paper_data['title']}\n\nAbstract: {paper_data['summary']}"
        
        chunks = self.chunk_text(full_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "title": paper_data["title"],
                    "authors": ", ".join(paper_data["authors"]),
                    "published": paper_data["published"],
                    "source": paper_data["entry_id"],
                    "chunk_index": i
                },
                "id": f"{paper_data['entry_id']}_chunk_{i}"
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks
