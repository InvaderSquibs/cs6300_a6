"""
Data retrieval nodes for the LangGraph workflow.

This module contains nodes that retrieve information from external sources:
- Vector database (ChromaDB) queries for semantic search
- Arxiv API searches for academic papers

These nodes are responsible for gathering the raw data that will be
processed and used by downstream nodes in the workflow.
"""
from src.state import GraphState

# Import for type hints (needed at runtime)
from src.vector_db import VectorDBManager
from src.arxiv_search import ArxivSearcher


def pull_from_chroma(
    state: GraphState,
    vector_db: VectorDBManager
) -> GraphState:
    """
    Query the vector database for documents relevant to the user query.
    
    This node performs semantic search on the ChromaDB vector database,
    retrieving the most similar document chunks to the user's query.
    The results are stored in the state for use by downstream nodes.
    
    The node queries for the top 3 most relevant documents by default.
    Results include document text, metadata, IDs, and similarity scores.
    
    Args:
        state: Current workflow state containing:
            - user_query: str - The user's question to search for
        vector_db: VectorDBManager instance configured with the persistent
            ChromaDB collection. Should already be initialized with
            documents from previous workflow executions.
    
    Returns:
        Updated state with chroma_results populated:
            - chroma_results: Dict[str, Any] - Results dictionary with structure:
                {
                    "documents": [[str, ...]],      # List of document chunks
                    "metadatas": [[Dict, ...]],     # List of metadata dicts
                    "ids": [[str, ...]],           # List of document IDs
                    "distances": [[float, ...]]    # Similarity distances
                }
    
    State Modifications:
        - Sets state["chroma_results"] to the query results dictionary
        - Results may be empty if no documents match or database is empty
    
    Example State Transition:
        Input state:
            {"user_query": "What is Nash equilibrium?", ...}
        
        Output state:
            {"user_query": "What is Nash equilibrium?",
             "chroma_results": {
                 "documents": [["Game theory definition..."]],
                 "metadatas": [[{"title": "...", "source": "..."}]],
                 ...
             }, ...}
    
    Note:
        - Uses n_results=3 to get top 3 most relevant documents
        - If database is empty, chroma_results["documents"] will be [[]]
        - Query uses semantic similarity based on embeddings stored in ChromaDB
    """
    print(f"Querying Chroma DB (contains {vector_db.count()} documents)...")
    
    results = vector_db.query(state["user_query"], n_results=3)
    state["chroma_results"] = results
    
    # Log results found
    num_docs = len(results.get("documents", [[]])[0])
    print(f"Found {num_docs} relevant documents in Chroma")
    
    return state


def search_arxiv(
    state: GraphState,
    arxiv_searcher: ArxivSearcher
) -> GraphState:
    """
    Search arxiv.org for academic papers related to the user query.
    
    This node searches arxiv for papers that might contain information
    relevant to the user's question. It augments the search query with
    "game theory" to ensure results are domain-relevant. The papers
    found are stored in state for processing and addition to the vector
    database.
    
    The search returns paper metadata (title, abstract, authors, etc.)
    but does not download full paper PDFs. Only abstracts are used for
    subsequent processing.
    
    Args:
        state: Current workflow state containing:
            - user_query: str - The user's question to search for
        arxiv_searcher: ArxivSearcher instance configured with max_results.
            Should be initialized before workflow execution. The max_results
            setting determines how many papers are returned.
    
    Returns:
        Updated state with arxiv_papers populated:
            - arxiv_papers: List[Dict[str, Any]] - List of paper dictionaries,
              each containing:
                - title: str - Paper title
                - summary: str - Abstract/summary
                - authors: List[str] - Author names
                - published: str - Publication date (YYYY-MM-DD)
                - pdf_url: str - URL to PDF
                - entry_id: str - Arxiv entry ID
    
    State Modifications:
        - Sets state["arxiv_papers"] to the list of found papers
        - List may be empty if no papers found or search fails
    
    Example State Transition:
        Input state:
            {"user_query": "What is Nash equilibrium?", ...}
        
        Output state:
            {"user_query": "What is Nash equilibrium?",
             "arxiv_papers": [
                 {
                     "title": "Nash Equilibrium in Game Theory",
                     "summary": "This paper discusses...",
                     ...
                 },
                 ...
             ], ...}
    
    Note:
        - Search query is constructed as: "game theory {user_query}"
        - Results are sorted by relevance (default arxiv behavior)
        - Only metadata is returned; full PDFs are not downloaded here
        - If search fails or returns no results, arxiv_papers will be []
    """
    print("Searching arxiv for papers on game theory...")
    
    # Create search query by prefixing with "game theory"
    search_query = f"game theory {state['user_query']}"
    papers = arxiv_searcher.search_papers(search_query)
    
    state["arxiv_papers"] = papers
    print(f"Found {len(papers)} papers on arxiv")
    
    return state

