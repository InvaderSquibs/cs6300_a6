"""
Node implementations for the LangGraph workflow.

This package contains all node functions organized by concern:
- context_nodes: Context checking and relevance evaluation
- retrieval_nodes: Data retrieval from vector DB and arxiv
- processing_nodes: Document processing and storage
- response_nodes: Response generation

All nodes follow a functional pattern:
    (state: GraphState, ...dependencies) -> GraphState

Dependencies are explicitly declared in function signatures, making it clear
what tools each node requires to operate.
"""
from .context_nodes import (
    check_needs_context,
    check_relevance
)
from .retrieval_nodes import (
    pull_from_chroma,
    search_arxiv
)
from .processing_nodes import (
    add_to_chroma
)
from .response_nodes import (
    generate_response
)
from .pdf_nodes import (
    extract_pdf_urls_from_results,
    download_pdfs_from_state
)
from .filter_nodes import (
    filter_game_theory_papers
)

__all__ = [
    "check_needs_context",
    "check_relevance",
    "pull_from_chroma",
    "search_arxiv",
    "filter_game_theory_papers",
    "add_to_chroma",
    "generate_response",
    "extract_pdf_urls_from_results",
    "download_pdfs_from_state",
]

