"""
Unit tests for document processor.
"""
import unittest
from src.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_small_text(self):
        """Test chunking of text smaller than chunk size."""
        text = "This is a small text."
        chunks = self.processor.chunk_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_chunk_large_text(self):
        """Test chunking of text larger than chunk size."""
        text = "A" * 250  # Text larger than chunk_size (100)
        chunks = self.processor.chunk_text(text)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 120)  # Allow some flexibility
    
    def test_process_paper(self):
        """Test processing a paper into chunks."""
        paper_data = {
            "title": "Test Paper",
            "summary": "This is a test paper summary.",
            "authors": ["Author One", "Author Two"],
            "published": "2024-01-01",
            "entry_id": "arxiv:1234.5678"
        }
        
        chunks = self.processor.process_paper(paper_data)
        
        self.assertGreater(len(chunks), 0)
        
        # Check first chunk structure
        first_chunk = chunks[0]
        self.assertIn("text", first_chunk)
        self.assertIn("metadata", first_chunk)
        self.assertIn("id", first_chunk)
        
        # Check metadata
        self.assertEqual(first_chunk["metadata"]["title"], "Test Paper")
        self.assertEqual(first_chunk["metadata"]["authors"], "Author One, Author Two")
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "word " * 100  # Create text with repeated words
        chunks = self.processor.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Last part of current chunk should appear in next chunk
                self.assertTrue(len(chunks[i]) > 0)
                self.assertTrue(len(chunks[i+1]) > 0)


if __name__ == "__main__":
    unittest.main()
