"""
Verification script to check all components are working correctly.
This script can run without an API key to verify the structure.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.vector_db import VectorDBManager
        from src.arxiv_search import ArxivSearcher
        from src.document_processor import DocumentProcessor
        print("✓ All core modules import successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_document_processor():
    """Test document processor functionality."""
    print("\nTesting document processor...")
    from src.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # Test chunking
    text = "This is a test. " * 50
    chunks = processor.chunk_text(text)
    print(f"✓ Text chunking works: {len(chunks)} chunks created")
    
    # Test paper processing
    paper_data = {
        "title": "Test Paper",
        "summary": "This is a test paper summary.",
        "authors": ["Author One", "Author Two"],
        "published": "2024-01-01",
        "entry_id": "arxiv:1234.5678"
    }
    processed = processor.process_paper(paper_data)
    print(f"✓ Paper processing works: {len(processed)} chunks with metadata")
    
    return True


def test_arxiv_searcher():
    """Test arxiv searcher initialization."""
    print("\nTesting arxiv searcher...")
    from src.arxiv_search import ArxivSearcher
    
    searcher = ArxivSearcher(max_results=2)
    print(f"✓ Arxiv searcher initialized with max_results={searcher.max_results}")
    
    return True


def test_workflow_structure():
    """Test workflow can be imported (requires API key to run)."""
    print("\nTesting workflow structure...")
    try:
        from src.workflow import GameTheoryRAG, GraphState
        print("✓ Workflow module imports successfully")
        print("  Note: Running the workflow requires OPENAI_API_KEY in .env")
        return True
    except ImportError as e:
        print(f"✗ Workflow import error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("LangGraph RAG System - Verification")
    print("="*60)
    
    tests = [
        test_imports,
        test_document_processor,
        test_arxiv_searcher,
        test_workflow_structure
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All verification tests passed!")
        print("\nTo run the full system:")
        print("1. Add OPENAI_API_KEY to .env file")
        print("2. Run: python example.py")
    else:
        print("✗ Some tests failed. Please check the output above.")
    print("="*60)


if __name__ == "__main__":
    main()
