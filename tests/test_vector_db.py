"""
Unit tests for vector database manager.
"""
import unittest
import tempfile
import shutil
from src.vector_db import VectorDBManager


class TestVectorDBManager(unittest.TestCase):
    """Test cases for VectorDBManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.db_manager = VectorDBManager(persist_directory=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that vector DB initializes correctly."""
        self.assertIsNotNone(self.db_manager.client)
        self.assertIsNotNone(self.db_manager.collection)
    
    def test_add_documents(self):
        """Test adding documents to the database."""
        documents = ["Document 1 content", "Document 2 content"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["doc1", "doc2"]
        
        self.db_manager.add_documents(documents, metadatas, ids)
        
        # Check that documents were added
        count = self.db_manager.count()
        self.assertEqual(count, 2)
    
    def test_query_documents(self):
        """Test querying documents from the database."""
        # Add some documents first
        documents = [
            "Game theory is a mathematical framework",
            "Nash equilibrium is a key concept",
            "The weather is sunny today"
        ]
        metadatas = [{"source": f"test{i}"} for i in range(3)]
        ids = [f"doc{i}" for i in range(3)]
        
        self.db_manager.add_documents(documents, metadatas, ids)
        
        # Query for game theory related content
        results = self.db_manager.query("game theory", n_results=2)
        
        self.assertIsNotNone(results)
        self.assertIn("documents", results)
        self.assertEqual(len(results["documents"][0]), 2)
    
    def test_empty_query(self):
        """Test querying an empty database."""
        results = self.db_manager.query("test query", n_results=3)
        
        self.assertIsNotNone(results)
        self.assertIn("documents", results)
        # Empty database should return empty results
        self.assertEqual(len(results["documents"][0]), 0)
    
    def test_add_documents_validation(self):
        """Test that add_documents validates input lengths."""
        documents = ["Doc 1", "Doc 2"]
        metadatas = [{"source": "test1"}]  # Wrong length
        ids = ["id1", "id2"]
        
        with self.assertRaises(ValueError) as context:
            self.db_manager.add_documents(documents, metadatas, ids)
        
        self.assertIn("same length", str(context.exception))


if __name__ == "__main__":
    unittest.main()
