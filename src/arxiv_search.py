"""
Arxiv search and paper retrieval functionality.

This module provides the ArxivSearcher class for searching and retrieving
papers from arxiv.org, a repository of academic preprints. The searcher uses
the arxiv Python library to query the arxiv API and retrieve paper metadata.

Arxiv API Usage:
    The module uses the official arxiv Python library which queries arxiv.org's
    public API. The API is free to use but has rate limiting considerations:
    - Rate limits are generous but excessive queries should be throttled
    - No API key required
    - Results are sorted by relevance by default

Paper Metadata:
    Papers are returned as dictionaries containing:
    - title: Paper title
    - summary: Abstract text
    - authors: List of author names
    - published: Publication date (YYYY-MM-DD format)
    - pdf_url: Direct URL to PDF
    - entry_id: Arxiv entry ID (e.g., "http://arxiv.org/abs/1234.5678v1")

Search Behavior:
    - Searches are sorted by relevance by default
    - Results are limited by max_results parameter
    - Query string supports arxiv search syntax
"""
import arxiv
from typing import List, Dict, Any, Optional
import os


class ArxivSearcher:
    """
    Search and retrieve papers from arxiv.org.
    
    This class provides methods to search arxiv for academic papers and
    retrieve their metadata. It encapsulates arxiv API interactions and
    provides a clean interface for paper discovery.
    
    Example:
        .. code-block:: python
        
            searcher = ArxivSearcher(max_results=3)
            papers = searcher.search_papers("game theory Nash equilibrium")
            
            for paper in papers:
                print(f"Title: {paper['title']}")
                print(f"Authors: {', '.join(paper['authors'])}")
                print(f"Abstract: {paper['summary'][:100]}...")
    """
    
    def __init__(self, max_results: int = 1):
        """
        Initialize the arxiv searcher.
        
        Args:
            max_results: Maximum number of papers to return per search.
                Default is 1. Higher values return more results but take
                longer and use more resources.
        
        Note:
            - This setting applies to all searches performed by this instance
            - Can be set to higher values (e.g., 5-10) for broader coverage
            - Keep in mind rate limiting and processing time for large values
        """
        self.max_results = max_results
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search arxiv for papers matching the query.
        
        Performs a search on arxiv.org using the provided query string.
        Results are sorted by relevance and limited to max_results papers.
        Returns paper metadata (not full PDFs).
        
        Args:
            query: Search query string. Supports arxiv search syntax.
                Examples:
                    - "game theory"
                    - "Nash equilibrium AND prisoner's dilemma"
                    - "au:Smith AND ti:game theory"
                Query is passed directly to arxiv API.
        
        Returns:
            List of paper metadata dictionaries. Each dictionary contains:
                - title: str - Paper title
                - summary: str - Abstract/summary text
                - authors: List[str] - List of author full names
                - published: str - Publication date in "YYYY-MM-DD" format
                - pdf_url: str - Direct URL to PDF file
                - entry_id: str - Arxiv entry ID (e.g., "http://arxiv.org/abs/1234.5678v1")
        
        Example:
            .. code-block:: python
            
                searcher = ArxivSearcher(max_results=2)
                papers = searcher.search_papers("game theory")
                
                # papers is a list like:
                # [
                #     {
                #         "title": "Game Theory Paper",
                #         "summary": "This paper discusses...",
                #         "authors": ["John Doe", "Jane Smith"],
                #         "published": "2024-01-15",
                #         "pdf_url": "http://arxiv.org/pdf/1234.5678.pdf",
                #         "entry_id": "http://arxiv.org/abs/1234.5678v1"
                #     },
                #     ...
                # ]
        
        Note:
            - Results are sorted by relevance (most relevant first)
            - Returns empty list if no papers found or search fails
            - Only metadata is returned; PDFs are not downloaded
            - Rate limiting may apply with frequent searches
        """
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            paper_data = {
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id
            }
            papers.append(paper_data)
            
        return papers
    
    def download_paper(
        self,
        paper: Dict[str, Any],
        download_dir: str = "./papers"
    ) -> Optional[str]:
        """
        Download a paper PDF from arxiv.
        
        Downloads the full PDF of a paper from arxiv.org and saves it to
        the specified directory. This method is currently unused in the main
        workflow, which only uses paper abstracts.
        
        Intended for future enhancement: The workflow could be extended to
        download full PDFs, extract text using PDF parsing libraries (e.g.,
        PyPDF2 or pdfplumber), and process the full paper text instead of
        just abstracts. This would provide more comprehensive context.
        
        Args:
            paper: Paper metadata dictionary containing at minimum:
                - entry_id: str - Arxiv entry ID used to locate the paper
            download_dir: Directory path where PDF should be saved.
                Directory is created if it doesn't exist. Default is "./papers".
        
        Returns:
            Path to downloaded PDF file as a string, or None if download fails.
            The filename is typically the arxiv ID (e.g., "1234.5678.pdf").
        
        Example:
            .. code-block:: python
            
                paper = {
                    "entry_id": "http://arxiv.org/abs/1234.5678v1",
                    "title": "Game Theory Paper",
                    ...
                }
                
                pdf_path = searcher.download_paper(paper, download_dir="./downloads")
                if pdf_path:
                    print(f"Downloaded to {pdf_path}")
        
        Raises:
            Exception: Various exceptions can occur (network errors, invalid
                entry_id, etc.) but are caught and logged. Method returns None
                on any error.
        
        Note:
            - This method is currently unused in the main workflow
            - Downloads require network connectivity
            - PDFs can be large (several MB)
            - Directory is created automatically if it doesn't exist
            - File is saved with arxiv ID as filename
            - Returns None silently on error (error is printed to console)
        
        Future Enhancement:
            To integrate PDF downloads into the workflow:
            1. Call this method after search_papers
            2. Extract text from downloaded PDF using a PDF library
            3. Process full text instead of just abstract in document_processor
            4. Add node to workflow that uses this method
        
        See Also:
            - search_papers: For finding papers (currently used)
            - DocumentProcessor.process_paper: Currently processes abstracts only
        """
        os.makedirs(download_dir, exist_ok=True)
        
        # Extract arxiv ID from entry_id
        # entry_id format: "http://arxiv.org/abs/1234.5678v1"
        arxiv_id = paper["entry_id"].split("/")[-1]
        
        try:
            # Search for the specific paper to download
            search = arxiv.Search(id_list=[arxiv_id])
            paper_result = next(search.results())
            
            # Download the PDF
            filename = paper_result.download_pdf(dirpath=download_dir)
            return filename
        except Exception as e:
            print(f"Error downloading paper: {e}")
            return None
