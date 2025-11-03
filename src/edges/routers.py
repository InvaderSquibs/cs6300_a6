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


def route_after_context_check(
    state: GraphState
) -> Literal["pull_from_chroma", "generate_response"]:
    """
    Route workflow based on whether the query needs game theory context.
    
    This router is used after the check_needs_context node to determine
    the next step. If context is needed, it routes to vector DB retrieval.
    Otherwise, it routes directly to response generation (skipping context
    retrieval for non-game-theory queries).
    
    Args:
        state: Current workflow state containing:
            - needs_context: bool - Flag set by check_needs_context node
    
    Returns:
        Literal string indicating next node:
            - "pull_from_chroma": If needs_context is True, route to vector
              DB retrieval to get relevant documents
            - "generate_response": If needs_context is False, skip context
              retrieval and generate response directly
    
    Example Usage:
        After check_needs_context sets needs_context=True:
            Input: {"needs_context": True, ...}
            Output: "pull_from_chroma"
        
        After check_needs_context sets needs_context=False:
            Input: {"needs_context": False, ...}
            Output: "generate_response"
    
    Note:
        - Return values must exactly match node names in the workflow graph
        - This router enables the workflow to handle both game theory and
          non-game-theory queries efficiently
    """
    if state["needs_context"]:
        return "pull_from_chroma"
    else:
        return "generate_response"


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
    if state["relevant_context"]:
        return "generate_response"
    else:
        return "search_arxiv"

