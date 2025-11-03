"""
Filter nodes for post-processing results.

This module contains nodes that filter and validate results after retrieval,
ensuring only game theory relevant content is used.
"""
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

try:
    from langchain_core.language_models import BaseChatModel
except ImportError:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import Any as BaseChatModel
    else:
        BaseChatModel = object


def filter_game_theory_papers(
    state: GraphState,
    llm: BaseChatModel
) -> GraphState:
    """
    Filter arxiv papers to only include those related to game theory.
    
    After searching arxiv, this node evaluates each paper to determine
    if it's actually related to game theory. This ensures that even if
    the search query includes "game theory", only relevant papers are
    added to the vector database.
    
    This filtering happens after retrieval, allowing the system to discover
    new game theory concepts that might not have been recognized in the
    initial query classification.
    
    Args:
        state: Current workflow state containing:
            - arxiv_papers: List[Dict[str, Any]] - Papers found from arxiv search
        llm: BaseChatModel instance for evaluating paper relevance.
    
    Returns:
        Updated state with arxiv_papers filtered to only game theory papers:
            - arxiv_papers: List[Dict[str, Any]] - Filtered list containing only
              papers that are game theory related
    
    State Modifications:
        - Filters state["arxiv_papers"] to only include game theory papers
        - If no papers are game theory related, list becomes empty
    
    Example State Transition:
        Input state:
            {"arxiv_papers": [
                {"title": "Nash Equilibrium in Games", ...},  # Game theory
                {"title": "Quantum Mechanics Basics", ...},     # Not game theory
            ], ...}
        
        Output state:
            {"arxiv_papers": [
                {"title": "Nash Equilibrium in Games", ...},  # Kept
            ], ...}
    
    Note:
        - Evaluates each paper's title and abstract
        - Uses LLM to classify if paper is game theory related
        - Only papers classified as game theory are kept
        - Empty list if no papers are game theory related
    """
    print(f"✓ NODE: filter_game_theory_papers")
    
    papers = state.get("arxiv_papers", [])
    if not papers:
        print(f"  No papers to filter")
        return state
    
    print(f"  Filtering {len(papers)} paper(s) for game theory relevance...")
    
    filtered_papers = []
    
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "")
        summary = paper.get("summary", "")[:500]  # First 500 chars of abstract
        
        # Escape curly braces in title/summary to prevent template variable interpretation
        title_escaped = title.replace("{", "{{").replace("}", "}}")
        summary_escaped = summary.replace("{", "{{").replace("}", "}}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that determines if an academic paper is related to GAME THEORY "
                      "(the mathematical study of strategic decision-making). "
                      "Respond with only 'yes' or 'no'. "
                      "Only say 'yes' if the paper is actually about game theory, Nash equilibrium, "
                      "strategic interactions, etc. Say 'no' for papers about video games, optimization, "
                      "or other unrelated topics even if they mention 'game' or 'theory'."),
            ("user", "Is this paper related to GAME THEORY (mathematical strategic decision-making)?\n\n"
                    "Title: {title}\n\n"
                    "Abstract: {summary}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "title": title_escaped,
            "summary": summary_escaped
        })
        
        is_game_theory = "yes" in response.content.lower()
        
        if is_game_theory:
            filtered_papers.append(paper)
            print(f"    Paper {i}: ✓ Game theory - '{title[:50]}...'")
        else:
            print(f"    Paper {i}: ✗ Not game theory - '{title[:50]}...'")
    
    state["arxiv_papers"] = filtered_papers
    
    # Track seen papers to prevent infinite loops
    papers_seen = state.get("papers_seen", [])
    for paper in papers:
        entry_id = paper.get("entry_id")
        if entry_id and entry_id not in papers_seen:
            papers_seen.append(entry_id)
    state["papers_seen"] = papers_seen
    
    print(f"  Filtered to {len(filtered_papers)} game theory paper(s)")
    print(f"  Total papers seen in this query: {len(papers_seen)}")
    
    return state

