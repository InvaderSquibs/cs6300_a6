"""
Context checking and relevance evaluation nodes.

This module contains nodes that determine whether user queries require
game theory context and whether retrieved context is relevant for answering
queries. Both nodes use LLM-based classification to make routing decisions.
"""
from typing import TYPE_CHECKING
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

# Import BaseChatModel for type hints (needed at runtime for get_type_hints)
try:
    from langchain_core.language_models import BaseChatModel
except ImportError:
    # Fallback for type checking only
    if TYPE_CHECKING:
        from typing import Any as BaseChatModel
    else:
        BaseChatModel = object  # Dummy fallback


def check_needs_context(
    state: GraphState,
    llm: BaseChatModel
) -> GraphState:
    """
    Determine if a user query requires game theory context.
    
    This node uses an LLM to classify whether the user's query is related
    to game theory. It's the first decision point in the workflow, routing
    queries either to context retrieval or direct response generation.
    
    Args:
        state: Current workflow state containing the user_query field.
            Expected fields:
                - user_query: str - The original user question
        llm: BaseChatModel instance to use for classification. Should
            be configured with appropriate temperature and model settings.
    
    Returns:
        Updated state with needs_context field set:
            - needs_context: bool - True if query is game theory related,
              False otherwise
    
    State Modifications:
        - Sets state["needs_context"] to True or False based on LLM response
    
    Example State Transition:
        Input state:
            {"user_query": "What is Nash equilibrium?", ...}
        
        Output state:
            {"user_query": "What is Nash equilibrium?", 
             "needs_context": True, ...}
    
    Note:
        The LLM is prompted to respond with only 'yes' or 'no'. The function
        checks if 'yes' appears in the response (case-insensitive). This is
        a simple heuristic that works well for classification tasks.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that determines if a query is related to game theory. "
                  "Respond with only 'yes' or 'no'."),
        ("user", "Is this query related to game theory? Query: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": state["user_query"]})
    
    needs_context = "yes" in response.content.lower()
    state["needs_context"] = needs_context
    
    print(f"✓ NODE: check_needs_context")
    print(f"  Query: {state['user_query'][:60]}...")
    print(f"  LLM Response: {response.content.strip()}")
    print(f"  Decision: needs_context = {needs_context}")
    return state


def check_relevance(
    state: GraphState,
    llm: BaseChatModel
) -> GraphState:
    """
    Evaluate whether retrieved context from vector DB is relevant to the query.
    
    This node assesses if the documents retrieved from ChromaDB contain
    sufficient and relevant information to answer the user's question. It's
    a quality check before generating a response, ensuring we don't provide
    answers based on irrelevant context.
    
    If no documents were found in chroma_results, this node immediately
    sets relevant_context to False and returns, bypassing LLM evaluation.
    
    Args:
        state: Current workflow state containing:
            - user_query: str - The original user question
            - chroma_results: Dict - Results from vector DB query with
              structure: {"documents": [[str, ...]], ...}
        llm: BaseChatModel instance to use for relevance evaluation.
            Should be configured appropriately for classification tasks.
    
    Returns:
        Updated state with relevant_context field set:
            - relevant_context: bool - True if context is relevant,
              False otherwise
    
    State Modifications:
        - Sets state["relevant_context"] to True or False
        - Early returns if no documents found (sets to False)
    
    Example State Transition:
        Input state:
            {"user_query": "What is Nash equilibrium?",
             "chroma_results": {"documents": [["Game theory text..."]]}, ...}
        
        Output state:
            {"user_query": "What is Nash equilibrium?",
             "chroma_results": {...},
             "relevant_context": True, ...}
    
    Note:
        Only uses the top 2 documents from chroma_results for evaluation
        to keep the prompt size manageable. The LLM is prompted to respond
        with only 'yes' or 'no', and the function checks for 'yes' in the
        response (case-insensitive).
    """
    # Early return if no documents found
    if not state["chroma_results"].get("documents", [[]])[0]:
        state["relevant_context"] = False
        print("No documents found in Chroma, will search arxiv")
        return state
    
    # Extract top documents for evaluation
    docs = state["chroma_results"]["documents"][0]
    combined_docs = "\n\n".join(docs[:2])  # Use top 2 documents
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that determines if provided context is relevant to answer a query. "
                  "Respond with only 'yes' or 'no'."),
        ("user", "Is this context relevant to answer the query?\n\n"
                "Query: {query}\n\nContext: {context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": state["user_query"], "context": combined_docs})
    
    relevant = "yes" in response.content.lower()
    state["relevant_context"] = relevant
    
    print(f"✓ NODE: check_relevance")
    print(f"  Documents evaluated: {len(docs)}")
    print(f"  LLM Response: {response.content.strip()}")
    print(f"  Decision: relevant_context = {relevant}")
    return state

