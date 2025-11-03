"""
PDF download nodes for the LangGraph workflow.

This module contains nodes that download PDF files from URLs found in vector
database metadata. These nodes demonstrate how the modular architecture enables
clean extensions - they use the same chroma_results as the RAG workflow but
branch to PDF processing instead.

The nodes extract pdf_url from metadata and use the standalone PDFDownloader
tool, following the explicit dependency injection pattern.
"""
from typing import List, Set
from src.state import GraphState

# Import for type hints (needed at runtime)
from src.pdf_downloader import PDFDownloader


def extract_pdf_urls_from_results(
    state: GraphState,
    pdf_downloader: PDFDownloader
) -> GraphState:
    """
    Extract PDF URLs from vector DB results and download the PDFs.
    
    This node extracts pdf_url values from the metadata of documents found
    in chroma_results, then downloads those PDFs using the PDFDownloader tool.
    The downloaded file paths are stored in state for use by downstream nodes.
    
    This node demonstrates the branching pattern: it uses the same chroma_results
    as the RAG workflow, but branches to PDF processing instead of response
    generation. This shows how metadata enables tool composition.
    
    Args:
        state: Current workflow state containing:
            - chroma_results: Dict[str, Any] - Results from vector DB query.
              Structure: {"metadatas": [[Dict, ...]], ...}
              Each metadata dict may contain a "pdf_url" key.
        pdf_downloader: PDFDownloader instance for downloading PDF files.
            Should be initialized with appropriate download directory.
    
    Returns:
        Updated state with downloaded_pdfs populated:
            - downloaded_pdfs: List[str] - List of file paths to downloaded PDFs.
              Empty list if no PDFs found or downloads fail.
    
    State Modifications:
        - Sets state["downloaded_pdfs"] to list of downloaded file paths
        - Does NOT modify chroma_results (read-only)
        - Deduplicates PDF URLs (multiple chunks from same paper share PDF)
    
    Example State Transition:
        Input state:
            {"chroma_results": {
                "metadatas": [[
                    {"pdf_url": "http://arxiv.org/pdf/1234.5678.pdf", ...},
                    {"pdf_url": "http://arxiv.org/pdf/1234.5678.pdf", ...},  # duplicate
                    {"pdf_url": "http://arxiv.org/pdf/5678.9012.pdf", ...}
                ]]
            }, ...}
        
        Output state:
            {"chroma_results": {...},  # Unchanged
             "downloaded_pdfs": [
                 "./papers/1234.5678.pdf",
                 "./papers/5678.9012.pdf"
             ], ...}
    
    Note:
        - Deduplicates PDF URLs (multiple chunks from same paper = one download)
        - Only downloads if pdf_url exists in metadata
        - Continues with remaining PDFs if one download fails
        - Downloads are stored in default directory (./papers) unless configured
    """
    print("Extracting PDF URLs from vector DB results...")
    
    # Extract unique PDF URLs from metadata
    pdf_urls: Set[str] = set()
    
    if state["chroma_results"].get("metadatas", [[]])[0]:
        metadatas = state["chroma_results"]["metadatas"][0]
        
        for metadata in metadatas:
            if metadata and "pdf_url" in metadata and metadata["pdf_url"]:
                pdf_urls.add(metadata["pdf_url"])
    
    if not pdf_urls:
        print("No PDF URLs found in metadata")
        state["downloaded_pdfs"] = []
        return state
    
    print(f"Found {len(pdf_urls)} unique PDF URLs to download")
    
    # Download each PDF
    downloaded_paths: List[str] = []
    for pdf_url in pdf_urls:
        try:
            pdf_path = pdf_downloader.download(pdf_url)
            if pdf_path:
                downloaded_paths.append(pdf_path)
        except Exception as e:
            print(f"✗ Error downloading {pdf_url}: {e}")
            # Continue with next PDF
            continue
    
    state["downloaded_pdfs"] = downloaded_paths
    print(f"✓ Downloaded {len(downloaded_paths)} PDFs")
    
    return state


def download_pdfs_from_state(
    state: GraphState,
    pdf_downloader: PDFDownloader
) -> GraphState:
    """
    Download PDFs from URLs stored in state (alternative approach).
    
    This node provides an alternative way to download PDFs when the URLs
    are already present in state (e.g., from arxiv_papers or other sources).
    Unlike extract_pdf_urls_from_results, this doesn't extract from metadata.
    
    Args:
        state: Current workflow state containing:
            - arxiv_papers: List[Dict[str, Any]] - List of paper dicts with pdf_url
              OR any other state field containing PDF URLs
        pdf_downloader: PDFDownloader instance for downloading PDF files.
    
    Returns:
        Updated state with downloaded_pdfs populated.
    
    Note:
        - This is an alternative to extract_pdf_urls_from_results
        - Can be used when PDF URLs come from arxiv_papers instead of metadata
        - Demonstrates flexibility in workflow design
    """
    print("Downloading PDFs from state...")
    
    downloaded_paths: List[str] = []
    
    # Extract PDF URLs from arxiv_papers if available
    if state.get("arxiv_papers"):
        pdf_urls: Set[str] = set()
        for paper in state["arxiv_papers"]:
            if "pdf_url" in paper and paper["pdf_url"]:
                pdf_urls.add(paper["pdf_url"])
        
        for pdf_url in pdf_urls:
            try:
                pdf_path = pdf_downloader.download(pdf_url)
                if pdf_path:
                    downloaded_paths.append(pdf_path)
            except Exception as e:
                print(f"✗ Error downloading {pdf_url}: {e}")
                continue
    
    state["downloaded_pdfs"] = downloaded_paths
    print(f"✓ Downloaded {len(downloaded_paths)} PDFs")
    
    return state

