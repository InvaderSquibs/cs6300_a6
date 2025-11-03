"""
LangGraph workflow for game theory RAG system.

This module provides the main GameTheoryRAG class which orchestrates a
LangGraph-based RAG (Retrieval-Augmented Generation) workflow for answering
game theory questions. The workflow uses a modular architecture with
explicit dependency injection.

Architecture:
    The workflow is built using a modular design:
    
    - Nodes: Functional node implementations in src/nodes/
        * Each node has explicit dependencies in its function signature
        * Dependencies are automatically injected by WorkflowBuilder
    
    - Edges: Routing functions in src/edges/
        * Conditional routing based on state conditions
        * Returns node names for dynamic workflow control
    
    - Builder: WorkflowBuilder in src/graph_builder.py
        * Analyzes node function signatures
        * Automatically injects dependencies based on type hints
        * Creates node wrappers compatible with LangGraph
    
    - State: GraphState in src/state.py
        * TypedDict defining all workflow state fields
        * Passed between nodes and modified during execution

Dependency Graph:
    Nodes and their tool dependencies:
    
    - check_needs_context: [llm]
    - check_relevance: [llm]
    - pull_from_chroma: [vector_db]
    - search_arxiv: [arxiv_searcher]
    - add_to_chroma: [vector_db, doc_processor]  # Multi-tool dependency
    - generate_response: [llm]

Shared Tools:
    - llm: Used by 3 nodes (context checking, relevance, response)
    - vector_db: Used by 2 nodes (retrieval, storage)
    - arxiv_searcher: Used by 1 node (search)
    - doc_processor: Used by 1 node (processing)

This architecture makes it easy to:
    - Add new nodes: Just create a function with explicit dependencies
    - Understand dependencies: Function signatures are self-documenting
    - Test nodes: Pure functions with injected dependencies
    - Extend workflow: Add new nodes and wire them to the graph
"""
from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv

from src.state import GraphState
from src.vector_db import VectorDBManager
from src.arxiv_search import ArxivSearcher
from src.document_processor import DocumentProcessor
from src.graph_builder import WorkflowBuilder

# Import nodes
from src.nodes import (
    check_needs_context,
    check_relevance,
    pull_from_chroma,
    search_arxiv,
    add_to_chroma,
    generate_response
)

# Import routers
from src.edges import (
    route_after_context_check,
    route_after_relevance_check
)

# Load environment variables
load_dotenv()


class GameTheoryRAG:
    """
    LangGraph workflow for game theory RAG system.
    
    This class orchestrates a RAG (Retrieval-Augmented Generation) workflow
    that answers game theory questions by:
    1. Determining if a query needs game theory context
    2. Retrieving relevant documents from a vector database
    3. Evaluating if retrieved context is sufficient
    4. Searching arxiv for additional papers if needed
    5. Processing and storing new papers
    6. Generating final answers using retrieved context
    
    The workflow uses a modular architecture with explicit dependency injection,
    making it easy to extend and test. All nodes are pure functions with
    dependencies declared in their signatures.
    
    Example:
        .. code-block:: python
        
            # Initialize with default settings (uses LM Studio if no API key)
            rag = GameTheoryRAG()
            
            # Query the system
            response = rag.query("What is Nash equilibrium?")
            print(response)
            
            # Initialize with OpenAI
            rag = GameTheoryRAG(openai_api_key="sk-...")
            
            # Initialize with custom LLM
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            rag = GameTheoryRAG(llm=llm)
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        openai_api_key: str = None,
        max_arxiv_results: int = 2,
        use_local_llm: bool = False,
        local_llm_model: str = "llama3.2",
        local_llm_base_url: str = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            llm: Optional pre-configured LLM instance. If provided, this
                instance will be used instead of creating a new one.
            openai_api_key: OpenAI API key (if using OpenAI or OpenAI-compatible
                endpoint). If not provided, defaults to LM Studio.
            max_arxiv_results: Maximum number of arxiv papers to fetch per search.
                Default is 2 to balance between coverage and processing time.
            use_local_llm: If True, use Ollama local LLM instead of OpenAI.
                Requires langchain-ollama package and running Ollama service.
            local_llm_model: Model name for local LLM. Default is "llama3.2".
                Must match a model available in your local LLM service.
            local_llm_base_url: Base URL for OpenAI-compatible local LLM
                (e.g., LM Studio). If provided, uses OpenAI-compatible endpoint
                instead of Ollama. Default LM Studio URL is "http://localhost:1234/v1".
        
        Raises:
            ImportError: If langchain-ollama is required but not installed.
            RuntimeError: If Ollama connection fails.
        """
        # Initialize LLM
        if llm:
            self.llm = llm
        elif local_llm_base_url:
            # Use OpenAI-compatible endpoint (e.g., LM Studio)
            from langchain_openai import ChatOpenAI
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or "lm-studio"
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=local_llm_base_url,
                model=local_llm_model,
                temperature=0
            )
            print(f"Using OpenAI-compatible local LLM at {local_llm_base_url} with model {local_llm_model}")
            # Tracing should work automatically with LangChain if env vars are set
        elif use_local_llm:
            try:
                from langchain_ollama import ChatOllama
                # Ollama should also support tracing if LANGCHAIN_TRACING_V2 is set
                self.llm = ChatOllama(model=local_llm_model, temperature=0)
                print(f"Using local LLM: {local_llm_model}")
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for local LLM support. "
                    "Install it with: pip install langchain-ollama"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to Ollama. Make sure Ollama is running and {local_llm_model} is available. "
                    f"Error: {e}"
                )
        else:
            # Default to LM Studio if no API key provided
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                # Default to LM Studio (local LLM)
                default_lm_studio_url = "http://localhost:1234/v1"
                print(f"No OpenAI API key found. Defaulting to LM Studio at {default_lm_studio_url}")
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    api_key="lm-studio",
                    base_url=default_lm_studio_url,
                    model="local-model",
                    temperature=0
                )
                print(f"Using LM Studio (default local LLM) at {default_lm_studio_url}")
            else:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    model="gpt-3.5-turbo",
                    temperature=0
                )
                print("Using OpenAI LLM: gpt-3.5-turbo")
        
        # Initialize component dependencies
        self.vector_db = VectorDBManager()
        self.arxiv_searcher = ArxivSearcher(max_results=max_arxiv_results)
        self.doc_processor = DocumentProcessor()
        
        # Set up dependency registry for builder
        # Keys must match type hint class names in node functions
        self.dependencies = {
            "BaseChatModel": self.llm,
            "VectorDBManager": self.vector_db,
            "ArxivSearcher": self.arxiv_searcher,
            "DocumentProcessor": self.doc_processor,
        }
        
        # Create workflow builder with dependencies
        self.builder = WorkflowBuilder(self.dependencies)
        
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow_uncompiled(self) -> StateGraph:
        """
        Build the LangGraph workflow (uncompiled version for visualization).
        
        This method constructs the workflow graph using the modular node
        functions and builder pattern. Nodes are created with automatic
        dependency injection, and edges are defined for control flow.
        
        Returns:
            Uncompiled StateGraph instance ready for visualization or compilation.
        """
        workflow = StateGraph(GraphState)
        
        # Add nodes with automatic dependency injection
        # The builder automatically injects dependencies based on function signatures
        workflow.add_node(
            "check_needs_context",
            self.builder.create_node(check_needs_context)
            # Dependencies: llm (BaseChatModel)
        )
        
        workflow.add_node(
            "pull_from_chroma",
            self.builder.create_node(pull_from_chroma)
            # Dependencies: vector_db (VectorDBManager)
        )
        
        workflow.add_node(
            "check_relevance",
            self.builder.create_node(check_relevance)
            # Dependencies: llm (BaseChatModel)
        )
        
        workflow.add_node(
            "search_arxiv",
            self.builder.create_node(search_arxiv)
            # Dependencies: arxiv_searcher (ArxivSearcher)
        )
        
        workflow.add_node(
            "add_to_chroma",
            self.builder.create_node(add_to_chroma)
            # Dependencies: vector_db (VectorDBManager), doc_processor (DocumentProcessor)
        )
        
        workflow.add_node(
            "generate_response",
            self.builder.create_node(generate_response)
            # Dependencies: llm (BaseChatModel)
        )
        
        # Set entry point
        workflow.set_entry_point("check_needs_context")
        
        # Add conditional edges with routing functions
        workflow.add_conditional_edges(
            "check_needs_context",
            route_after_context_check,
            {
                "pull_from_chroma": "pull_from_chroma",
                "generate_response": "generate_response"
            }
        )
        
        # After pulling from Chroma, check relevance
        workflow.add_edge("pull_from_chroma", "check_relevance")
        
        # Add conditional edges after relevance check
        workflow.add_conditional_edges(
            "check_relevance",
            route_after_relevance_check,
            {
                "generate_response": "generate_response",
                "search_arxiv": "search_arxiv"
            }
        )
        
        # After searching arxiv, add to chroma
        workflow.add_edge("search_arxiv", "add_to_chroma")
        
        # After adding to chroma, loop back to pull from chroma
        # This creates a self-improving workflow that re-queries after adding papers
        workflow.add_edge("add_to_chroma", "pull_from_chroma")
        
        # Generate response ends the workflow
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    def _build_workflow(self) -> StateGraph:
        """
        Build and compile the LangGraph workflow.
        
        This method creates the workflow graph and compiles it for execution.
        It also enables Phoenix/LangSmith tracing if configured.
        
        Returns:
            Compiled StateGraph ready for execution.
        """
        compiled = self._build_workflow_uncompiled().compile()
        
        # Enable Phoenix/LangSmith tracing if available
        try:
            if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
                print("Tracing enabled (Phoenix/LangSmith compatible)")
        except Exception:
            pass  # Tracing is optional
        
        return compiled
    
    def get_graph(self):
        """
        Get the uncompiled graph for visualization.
        
        This method returns an uncompiled version of the workflow graph
        that can be used for visualization, inspection, or debugging.
        
        Returns:
            Uncompiled StateGraph instance.
        """
        return self._build_workflow_uncompiled()
    
    def query(self, user_query: str) -> str:
        """
        Process a user query through the workflow.
        
        This is the main entry point for querying the RAG system. It
        initializes the workflow state, executes the workflow, and returns
        the final response.
        
        Args:
            user_query: The user's question as a string.
        
        Returns:
            The final answer string generated by the workflow.
        
        Example:
            .. code-block:: python
            
                rag = GameTheoryRAG()
                response = rag.query("What is Nash equilibrium?")
                print(response)
                # Output: "Nash equilibrium is a concept in game theory..."
        """
        initial_state: GraphState = {
            "user_query": user_query,
            "needs_context": False,
            "chroma_results": {},
            "relevant_context": False,
            "arxiv_papers": [],
            "papers_added": False,
            "downloaded_pdfs": [],
            "final_response": ""
        }
        
        print(f"\n{'='*60}")
        print(f"Processing query: {user_query}")
        print(f"{'='*60}\n")
        
        final_state = self.workflow.invoke(initial_state)
        return final_state["final_response"]
