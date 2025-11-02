"""
Arxiv search and paper retrieval functionality.
"""
import arxiv
from typing import List, Dict, Any, Optional
import time


class ArxivSearcher:
    """Search and retrieve papers from arxiv.org."""
    
    def __init__(self, max_results: int = 1):
        """Initialize the arxiv searcher."""
        self.max_results = max_results
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search arxiv for papers matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of paper metadata dictionaries
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
    
    def download_paper(self, paper: Dict[str, Any], download_dir: str = "./papers") -> Optional[str]:
        """
        Download a paper PDF.
        
        Args:
            paper: Paper metadata dictionary
            download_dir: Directory to save PDF
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        import os
        os.makedirs(download_dir, exist_ok=True)
        
        # Extract arxiv ID from entry_id
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
