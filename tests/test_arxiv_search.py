"""
Unit tests for arxiv searcher.
"""
import unittest
from unittest.mock import Mock, patch
from src.arxiv_search import ArxivSearcher


class TestArxivSearcher(unittest.TestCase):
    """Test cases for ArxivSearcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.searcher = ArxivSearcher(max_results=1)
    
    def test_initialization(self):
        """Test that arxiv searcher initializes correctly."""
        self.assertEqual(self.searcher.max_results, 1)
    
    @patch('arxiv.Search')
    def test_search_papers_structure(self, mock_search):
        """Test that search_papers returns correct structure."""
        # Create a mock result
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.summary = "Test summary"
        mock_result.authors = [Mock(name="Test Author")]
        mock_result.published = Mock(strftime=lambda x: "2024-01-01")
        mock_result.pdf_url = "http://example.com/paper.pdf"
        mock_result.entry_id = "arxiv:1234.5678"
        
        # Configure mock to return our result
        mock_search_instance = Mock()
        mock_search_instance.results.return_value = [mock_result]
        mock_search.return_value = mock_search_instance
        
        # Test search
        papers = self.searcher.search_papers("game theory")
        
        # Verify structure
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertIn("title", paper)
        self.assertIn("summary", paper)
        self.assertIn("authors", paper)
        self.assertIn("published", paper)
        self.assertIn("pdf_url", paper)
        self.assertIn("entry_id", paper)
    
    def test_max_results_setting(self):
        """Test that max_results can be configured."""
        searcher = ArxivSearcher(max_results=5)
        self.assertEqual(searcher.max_results, 5)


if __name__ == "__main__":
    unittest.main()
