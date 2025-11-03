"""
Routing functions for conditional edges in the LangGraph workflow.

This module contains functions that determine workflow control flow based
on state conditions. These functions are used with conditional edges to
create dynamic, state-dependent routing.

All routing functions follow the pattern:
    (state: GraphState) -> Literal[str, ...]

They return string literals that map to node names in the workflow graph.
These return values must match the node names exactly for proper routing.
"""
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import GraphState
else:
    from src.state import GraphState


def route_after_relevance_check(
    state: GraphState
) -> Literal["generate_response", "search_arxiv"]:
    """
    Route workflow based on whether retrieved context is relevant.
    
    This router is used after the check_relevance node to determine whether
    the context from the vector database is sufficient to answer the query.
    If relevant, it routes to response generation. If not relevant (or no
    context found), it routes to arxiv search to find additional papers.
    
    Args:
        state: Current workflow state containing:
            - relevant_context: bool - Flag set by check_relevance node
    
    Returns:
        Literal string indicating next node:
            - "generate_response": If relevant_context is True, use the
              retrieved context to generate an answer
            - "search_arxiv": If relevant_context is False, search arxiv
              for papers to expand the knowledge base
    
    Example Usage:
        After check_relevance sets relevant_context=True:
            Input: {"relevant_context": True, ...}
            Output: "generate_response"
        
        After check_relevance sets relevant_context=False:
            Input: {"relevant_context": False, ...}
            Output: "search_arxiv"
    
    Note:
        - Return values must exactly match node names in the workflow graph
        - This router implements the self-improving workflow pattern: if
          existing knowledge isn't sufficient, the system automatically
          searches for and adds new knowledge
        - The workflow loops back after adding papers (add_to_chroma ->
          pull_from_chroma) to re-query with expanded knowledge base
    """
    relevant = state["relevant_context"]
    chroma_results = state.get("chroma_results", {})
    num_docs = len(chroma_results.get("documents", [[]])[0]) if chroma_results.get("documents") else 0
    
    if relevant:
        print(f"→ ROUTING: Context is RELEVANT → routing to 'generate_response'")
        print(f"  Reason: Retrieved {num_docs} document(s) are sufficient to answer the query")
        return "generate_response"
    else:
        print(f"→ ROUTING: Context is NOT RELEVANT (or missing) → routing to 'search_arxiv'")
        print(f"  Reason: Retrieved {num_docs} document(s) insufficient, searching arxiv with 'game theory' prefix")
        return "search_arxiv"


def route_after_paper_filter(
    state: GraphState
) -> Literal["add_to_chroma", "generate_response"]:
    """
    Route workflow based on whether game theory papers were found.
    
    After filtering arxiv papers for game theory relevance, this router
    determines the next step. If game theory papers were found, they're
    added to the vector database. If no game theory papers were found,
    it routes to response generation with a message indicating the query
    is not game theory related.
    
    Args:
        state: Current workflow state containing:
            - arxiv_papers: List[Dict[str, Any]] - Filtered list of papers
              (may be empty if none were game theory related)
            - chroma_results: Dict - Results from vector DB query
    
    Returns:
        Literal string indicating next node:
            - "add_to_chroma": If game theory papers were found, add them
              to the vector database and loop back to retrieval
            - "generate_response": If no game theory papers found, generate
              response indicating query is not game theory related
    
    Note:
        - This router checks if arxiv_papers list has any items
        - Also checks if vector DB had any results
        - Only routes to generate_response if both are empty (not game theory)
    """
    papers = state.get("arxiv_papers", [])
    chroma_results = state.get("chroma_results", {})
    num_docs = len(chroma_results.get("documents", [[]])[0]) if chroma_results.get("documents") else 0
    papers_seen = state.get("papers_seen", [])
    
    if papers:
        print(f"→ ROUTING: Found {len(papers)} NEW game theory paper(s) → routing to 'add_to_chroma'")
        print(f"  Reason: Papers will be added to vector DB and workflow will loop back to retrieve them")
        return "add_to_chroma"
    else:
        # No new game theory papers found
        if num_docs == 0:
            print(f"→ ROUTING: No game theory papers found (checked {len(papers_seen)} total) AND no vector DB results → routing to 'generate_response'")
            print(f"  Reason: Query appears to not be game theory related or no relevant papers exist")
            return "generate_response"
        else:
            # Have vector DB results but no new papers to add - use what we have
            print(f"→ ROUTING: No new game theory papers (already seen {len(papers_seen)}), but have {num_docs} vector DB result(s) → routing to 'generate_response'")
            print(f"  Reason: Will use existing vector DB results for response (no new papers to add)")
            return "generate_response"
