"""
Graph state definition for the LangGraph workflow.

This module defines the GraphState TypedDict which represents the complete
state of the workflow at any given point. The state is passed between nodes
and modified as the workflow progresses.

The state follows a functional programming paradigm where each node receives
the current state and returns an updated state, ensuring clear data flow
through the workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    """
    State container for the game theory RAG workflow.
    
    This TypedDict defines all state fields that flow through the LangGraph
    workflow. Each node can read from and write to this state.
    
    Attributes:
        user_query: The original user query string. Set at workflow start,
            remains constant throughout execution.
            
        needs_context: Boolean flag indicating whether the query requires
            game theory context. Set by check_needs_context node.
            
        chroma_results: Dictionary containing results from vector database
            query. Structure: {
                "documents": [[str, ...]],  # List of document chunks
                "metadatas": [[Dict, ...]], # List of metadata dicts
                "ids": [[str, ...]],        # List of document IDs
                "distances": [[float, ...]] # List of similarity distances
            }
            Set by pull_from_chroma node.
            
        relevant_context: Boolean flag indicating whether retrieved context
            is relevant to answer the query. Set by check_relevance node.
            
        arxiv_papers: List of paper metadata dictionaries found from arxiv
            search. Each dict contains: title, summary, authors, published,
            pdf_url, entry_id. Set by search_arxiv node.
            
        papers_added: Boolean flag indicating whether papers were successfully
            added to the vector database. Set by add_to_chroma node.
            
        downloaded_pdfs: Optional list of file paths to downloaded PDF files.
            Used by PDF download workflows. Empty list by default. Set by PDF
            download nodes when PDFs are downloaded from URLs found in metadata.
            
        final_response: The final answer string generated for the user.
            Set by generate_response node.
    
    Example:
        Initial state for query "What is Nash equilibrium?":
        
        .. code-block:: python
        
            {
                "user_query": "What is Nash equilibrium?",
                "needs_context": False,
                "chroma_results": {},
                "relevant_context": False,
                "arxiv_papers": [],
                "papers_added": False,
                "downloaded_pdfs": [],
                "final_response": ""
            }
        
        After workflow execution, final_response contains the answer and
        chroma_results may contain retrieved documents.
    """
    user_query: str
    needs_context: bool
    chroma_results: List[Dict[str, Any]]
    relevant_context: bool
    arxiv_papers: List[Dict[str, Any]]
    papers_added: bool
    downloaded_pdfs: List[str]  # Optional field for PDF download workflows
    final_response: str

