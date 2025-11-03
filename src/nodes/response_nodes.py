"""
Response generation nodes for the LangGraph workflow.

This module contains nodes that generate the final output for the user.
These nodes combine retrieved context with LLM capabilities to produce
coherent, contextually-aware responses.
"""
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

# Import BaseChatModel for type hints (needed at runtime for get_type_hints)
try:
    from langchain_core.language_models import BaseChatModel
except ImportError:
    # Fallback for type checking only
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import Any as BaseChatModel
    else:
        BaseChatModel = object  # Dummy fallback


def generate_response(
    state: GraphState,
    llm: BaseChatModel
) -> GraphState:
    """
    Generate the final answer to the user's query using retrieved context.
    
    This node is the final step in the workflow. It constructs a prompt
    using context retrieved from the vector database (if available) and
    generates a response using the LLM. If no context is available, it
    returns a fallback message indicating insufficient information.
    
    The node uses up to the top 3 most relevant documents from chroma_results
    to construct the context. The LLM is instructed to use this context
    to answer the user's question about game theory.
    
    Args:
        state: Current workflow state containing:
            - user_query: str - The original user question
            - chroma_results: Dict[str, Any] - Results from vector DB query.
              May contain documents or be empty. Structure:
              {"documents": [[str, ...]], ...}
        llm: BaseChatModel instance configured for response generation.
            Should use appropriate temperature and model settings for
            producing coherent answers.
    
    Returns:
        Updated state with final_response populated:
            - final_response: str - The generated answer string or fallback
              message if no context available
    
    State Modifications:
        - Sets state["final_response"] to the LLM-generated answer or
          fallback message
        - Does NOT modify other state fields (read-only access)
    
    Example State Transition:
        Input state:
            {"user_query": "What is Nash equilibrium?",
             "chroma_results": {
                 "documents": [["Game theory definition...", ...]],
                 ...
             }, ...}
        
        Output state:
            {"user_query": "What is Nash equilibrium?",
             "chroma_results": {...},  # Unchanged
             "final_response": "Nash equilibrium is a concept in game theory...", ...}
    
    Fallback Behavior:
        If chroma_results is empty or contains no documents, the node
        returns a predefined message: "I don't have enough information
        to answer your question about game theory."
    
    Note:
        - Uses top 3 documents from chroma_results for context
        - Context is combined with newlines for readability
        - System prompt instructs LLM to answer about game theory specifically
        - The user query is included in the prompt to ensure relevance
    """
    # Get context from Chroma results
    if state["chroma_results"].get("documents", [[]])[0]:
        docs = state["chroma_results"]["documents"][0]
        context = "\n\n".join(docs[:3])  # Use top 3 documents
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about game theory. "
                      "Use the provided context to answer the user's question."),
            ("user", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": state["user_query"]})
        state["final_response"] = response.content
    else:
        # Fallback if no context available
        state["final_response"] = "I don't have enough information to answer your question about game theory."
    
    print("Generated final response")
    return state

