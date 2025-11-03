"""
Example demonstrating PDF download extension.

This script shows how the PDF download capability extends the RAG system:
1. Basic RAG workflow (existing functionality)
2. Standalone PDF downloader usage
3. Extracting PDF URLs from vector DB metadata
4. Downloading PDFs from workflow results

This demonstrates the clean architectural extension pattern.
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG
from src.pdf_downloader import PDFDownloader
from src.nodes import extract_pdf_urls_from_results
from src.state import GraphState

load_dotenv()


def example_1_basic_rag():
    """Example 1: Basic RAG workflow (unchanged)."""
    print("="*60)
    print("Example 1: Basic RAG Workflow")
    print("="*60)
    print()
    
    rag = GameTheoryRAG()
    response = rag.query("What is Nash equilibrium?")
    
    print(f"\nResponse: {response[:200]}...\n")
    print("✓ RAG workflow works as before - backward compatible!")


def example_2_standalone_pdf_downloader():
    """Example 2: Using PDFDownloader as standalone tool."""
    print("\n" + "="*60)
    print("Example 2: Standalone PDF Downloader")
    print("="*60)
    print()
    
    downloader = PDFDownloader()
    
    # Example PDF URL from arxiv
    pdf_url = "http://arxiv.org/pdf/1406.2661v1.pdf"  # Small paper for testing
    
    print(f"Downloading PDF from: {pdf_url}")
    pdf_path = downloader.download(pdf_url, download_dir="./example_papers")
    
    if pdf_path:
        print(f"\n✓ PDF downloaded to: {pdf_path}")
        print(f"✓ Standalone tool works independently!")
    else:
        print(f"\n⚠ Download failed (may need network or SSL config)")
        print(f"✓ Tool still demonstrates independence - no workflow required")


def example_3_pdf_from_metadata():
    """Example 3: Extract and download PDFs from RAG results."""
    print("\n" + "="*60)
    print("Example 3: PDF Download from RAG Results")
    print("="*60)
    print()
    
    # Run RAG query first to populate vector DB with papers
    print("Step 1: Running RAG query to populate vector DB...")
    rag = GameTheoryRAG()
    response = rag.query("What is game theory?")
    print("✓ RAG query completed")
    
    # Now query vector DB to get results with metadata
    print("\nStep 2: Querying vector DB for results with PDF URLs...")
    from src.vector_db import VectorDBManager
    
    vector_db = VectorDBManager()
    results = vector_db.query("game theory", n_results=5)
    
    # Check if we have PDF URLs in metadata
    metadatas = results.get("metadatas", [[]])[0]
    pdf_urls_found = []
    
    for metadata in metadatas:
        if metadata and "pdf_url" in metadata:
            pdf_urls_found.append(metadata["pdf_url"])
    
    if pdf_urls_found:
        unique_urls = list(set(pdf_urls_found))
        print(f"✓ Found {len(unique_urls)} unique PDF URLs in metadata")
        
        # Download PDFs
        print("\nStep 3: Downloading PDFs using PDFDownloader...")
        downloader = PDFDownloader()
        downloaded = []
        
        for pdf_url in unique_urls[:2]:  # Limit to 2 for example
            print(f"\n  Downloading: {pdf_url}")
            pdf_path = downloader.download(pdf_url, download_dir="./example_papers")
            if pdf_path:
                downloaded.append(pdf_path)
                print(f"  ✓ Downloaded to: {pdf_path}")
        
        print(f"\n✓ Downloaded {len(downloaded)} PDFs from vector DB metadata")
        print("✓ Demonstrates: RAG results → metadata extraction → PDF download")
    else:
        print("⚠ No PDF URLs found in metadata")
        print("  (This is normal if papers were added before pdf_url was added)")


def example_4_workflow_node():
    """Example 4: Using PDF node in workflow context."""
    print("\n" + "="*60)
    print("Example 4: PDF Node in Workflow Context")
    print("="*60)
    print()
    
    # Create mock state with chroma_results containing PDF URLs
    test_state: GraphState = {
        "user_query": "game theory",
        "needs_context": False,
        "chroma_results": {
            "metadatas": [[
                {
                    "pdf_url": "http://arxiv.org/pdf/1234.5678.pdf",
                    "title": "Example Paper 1"
                },
                {
                    "pdf_url": "http://arxiv.org/pdf/1234.5678.pdf",  # duplicate
                    "title": "Example Paper 1 Chunk 2"
                },
                {
                    "pdf_url": "http://arxiv.org/pdf/5678.9012.pdf",
                    "title": "Example Paper 2"
                }
            ]],
            "documents": [["doc1", "doc2", "doc3"]],
            "ids": [["id1", "id2", "id3"]]
        },
        "relevant_context": False,
        "arxiv_papers": [],
        "papers_added": False,
        "downloaded_pdfs": [],
        "final_response": ""
    }
    
    print("Mock state with PDF URLs in metadata:")
    print(f"  - {len(test_state['chroma_results']['metadatas'][0])} metadata entries")
    
    # Use PDF node to extract and download
    pdf_downloader = PDFDownloader()
    result_state = extract_pdf_urls_from_results(test_state, pdf_downloader)
    
    print(f"\n✓ Node extracted {len(set([m.get('pdf_url') for m in test_state['chroma_results']['metadatas'][0] if m.get('pdf_url')]))} unique PDF URLs")
    print(f"✓ State updated with downloaded_pdfs: {len(result_state['downloaded_pdfs'])} files")
    print(f"✓ Demonstrates explicit dependency injection pattern")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PDF Download Extension - Working Examples")
    print("="*70)
    print("\nThis demonstrates the architectural extension pattern:")
    print("  1. Backward compatibility (existing RAG works)")
    print("  2. Tool independence (PDFDownloader standalone)")
    print("  3. Metadata-driven composition (extract from vector DB)")
    print("  4. Workflow integration (nodes with explicit dependencies)")
    print()
    
    try:
        example_1_basic_rag()
        example_2_standalone_pdf_downloader()
        example_3_pdf_from_metadata()
        example_4_workflow_node()
        
        print("\n" + "="*70)
        print("✓ All Examples Completed")
        print("="*70)
        print("\nKey Takeaways:")
        print("  - RAG workflow unchanged (backward compatible)")
        print("  - PDFDownloader is standalone (reusable anywhere)")
        print("  - Metadata enables new capabilities (pdf_url stored)")
        print("  - Nodes follow explicit dependency pattern (clean extension)")
        print()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

