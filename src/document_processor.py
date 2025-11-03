"""
Document processing utilities for chunking and preparing text.

This module provides the DocumentProcessor class for splitting documents into
overlapping chunks suitable for vector database storage and retrieval.

Chunking Algorithm:
    The processor splits text into fixed-size chunks with overlap to ensure
    context continuity. It uses a sentence-boundary-aware algorithm that:
    1. Splits text into chunks of approximately chunk_size characters
    2. Attempts to break at sentence boundaries (., !, ?, \n) for readability
    3. Creates overlap between chunks (chunk_overlap characters) to preserve
       context at boundaries
    4. Cleans whitespace (normalizes multiple spaces/newlines to single space)

Overlap Strategy:
    Overlap ensures that important information spanning chunk boundaries isn't
    lost. For example, if a concept is mentioned at the end of one chunk and
    explained at the start of the next, overlap keeps both parts accessible
    during retrieval.

Sentence Boundary Detection:
    When possible, chunks are broken at sentence boundaries within 100
    characters of the target size. This keeps sentences intact, improving
    readability and semantic coherence of retrieved chunks.
"""
from typing import List, Dict, Any
import re


class DocumentProcessor:
    """
    Process and chunk documents for vector database storage.
    
    This class handles the conversion of documents (papers, texts) into
    chunked format suitable for vector database storage. It combines title
    and abstract text, splits into overlapping chunks, and attaches metadata
    to each chunk.
    
    Example:
        .. code-block:: python
        
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            
            paper = {
                "title": "Game Theory Paper",
                "summary": "Long abstract text...",
                "authors": ["Author One", "Author Two"],
                "published": "2024-01-01",
                "entry_id": "http://arxiv.org/abs/1234.5678v1"
            }
            
            chunks = processor.process_paper(paper)
            # Returns list of chunk dicts with text, metadata, and id
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters. Default is 1000.
                Chunks will be approximately this size, but may vary slightly
                due to sentence boundary detection. Larger chunks provide more
                context but may exceed embedding limits.
            chunk_overlap: Number of characters to overlap between consecutive
                chunks. Default is 200. Overlap ensures context continuity and
                prevents information loss at boundaries. Should be less than
                chunk_size.
        
        Note:
            - chunk_overlap should be less than chunk_size
            - Typical values: chunk_size=500-2000, chunk_overlap=50-500
            - Larger chunks work well for detailed technical content
            - Smaller chunks are better for focused retrieval
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Splits input text into chunks of approximately chunk_size characters,
        attempting to break at sentence boundaries when possible. Chunks overlap
        by chunk_overlap characters to preserve context.
        
        Args:
            text: Input text string to chunk. Can be any length. Text is
                cleaned (whitespace normalized) before chunking.
        
        Returns:
            List of text chunk strings. Each chunk is approximately chunk_size
            characters, with chunk_overlap characters shared with adjacent chunks.
            If input text is shorter than chunk_size, returns single-element list.
        
        Example:
            .. code-block:: python
            
                processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
                text = "Sentence one. Sentence two. Sentence three..."
                chunks = processor.chunk_text(text)
                # Returns: ["Sentence one. Sentence two...", "two... three...", ...]
        
        Note:
            - Text is cleaned: multiple spaces/newlines normalized to single space
            - Chunks are stripped of leading/trailing whitespace
            - Sentence boundary detection searches up to 100 chars from target size
            - Overlap is subtracted from next chunk's start position
        """
        # Clean the text: normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                # Search backwards up to 100 characters for sentence markers
                for i in range(min(100, end - start)):
                    if text[end - i] in '.!?\n':
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position: subtract overlap to create overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_paper(
        self,
        paper_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a paper into chunks with metadata.
        
        Takes a paper metadata dictionary, combines title and abstract into
        a single text, chunks it, and attaches metadata to each chunk. This
        format is ready for storage in the vector database.
        
        Args:
            paper_data: Dictionary containing paper information. Required keys:
                - title: str - Paper title
                - summary: str - Paper abstract/summary text
                - authors: List[str] - List of author names
                - published: str - Publication date (YYYY-MM-DD format)
                - entry_id: str - Arxiv entry ID or source identifier
                - pdf_url: str - URL to PDF file (optional, if available)
            Optional keys:
                - pdf_url: str - URL to PDF file (if available)
        
        Returns:
            List of dictionaries, one per chunk. Each dictionary contains:
                - text: str - Chunk text content (from title + abstract)
                - metadata: Dict[str, Any] - Metadata dictionary with:
                    - title: str - Original paper title
                    - authors: str - Comma-separated author names
                    - published: str - Publication date
                    - source: str - Entry ID or source identifier
                    - pdf_url: str - URL to PDF file (if available in paper_data)
                    - chunk_index: int - Zero-based chunk index within paper
                - id: str - Unique chunk identifier
                    Format: "{entry_id}_chunk_{index}"
        
        Example:
            .. code-block:: python
            
                paper = {
                    "title": "Nash Equilibrium in Game Theory",
                    "summary": "This paper discusses Nash equilibrium...",
                    "authors": ["John Doe", "Jane Smith"],
                    "published": "2024-01-15",
                    "entry_id": "http://arxiv.org/abs/1234.5678v1"
                }
                
                chunks = processor.process_paper(paper)
                # Returns:
                # [
                #     {
                #         "text": "Title: Nash Equilibrium...\n\nAbstract: This paper...",
                #         "metadata": {
                #             "title": "Nash Equilibrium in Game Theory",
                #             "authors": "John Doe, Jane Smith",
                #             "published": "2024-01-15",
                #             "source": "http://arxiv.org/abs/1234.5678v1",
                #             "chunk_index": 0
                #         },
                #         "id": "http://arxiv.org/abs/1234.5678v1_chunk_0"
                #     },
                #     ...
                # ]
        
        Note:
            - Only title and summary are processed (not full PDF text)
            - Title and abstract are combined with separator: "Title: ...\n\nAbstract: ..."
            - Authors list is joined with ", " for storage
            - Chunk IDs include entry_id to ensure uniqueness across papers
            - Metadata is preserved for filtering and attribution in queries
            - pdf_url is stored in metadata if available, enabling downstream PDF processing
            - All chunks from the same paper share the same pdf_url
        """
        # Combine title and summary for chunking
        full_text = f"Title: {paper_data['title']}\n\nAbstract: {paper_data['summary']}"
        
        chunks = self.chunk_text(full_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Build metadata dictionary
            metadata = {
                "title": paper_data["title"],
                "authors": ", ".join(paper_data["authors"]),
                "published": paper_data["published"],
                "source": paper_data["entry_id"],
                "chunk_index": i
            }
            # Add pdf_url if available (enables PDF download workflows)
            if "pdf_url" in paper_data and paper_data["pdf_url"]:
                metadata["pdf_url"] = paper_data["pdf_url"]
            
            chunk_data = {
                "text": chunk,
                "metadata": metadata,
                "id": f"{paper_data['entry_id']}_chunk_{i}"
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks
