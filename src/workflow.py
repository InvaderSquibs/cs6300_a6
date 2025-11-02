"""
LangGraph workflow for game theory RAG system.
"""
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

from src.vector_db import VectorDBManager
from src.arxiv_search import ArxivSearcher
from src.document_processor import DocumentProcessor


# Load environment variables
load_dotenv()


class GraphState(TypedDict):
    """State for the graph workflow."""
    user_query: str
    needs_context: bool
    chroma_results: List[Dict[str, Any]]
    relevant_context: bool
    arxiv_papers: List[Dict[str, Any]]
    papers_added: bool
    final_response: str


class GameTheoryRAG:
    """LangGraph workflow for game theory RAG system."""
    
    def __init__(self, openai_api_key: str = None, max_arxiv_results: int = 2):
        """Initialize the RAG system."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        self.vector_db = VectorDBManager()
        self.arxiv_searcher = ArxivSearcher(max_results=max_arxiv_results)
        self.doc_processor = DocumentProcessor()
        
        self.workflow = self._build_workflow()
    
    def _check_needs_context(self, state: GraphState) -> GraphState:
        """Check if the user query needs game theory context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that determines if a query is related to game theory. "
                      "Respond with only 'yes' or 'no'."),
            ("user", "Is this query related to game theory? Query: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": state["user_query"]})
        
        needs_context = "yes" in response.content.lower()
        state["needs_context"] = needs_context
        
        print(f"Query needs game theory context: {needs_context}")
        return state
    
    def _pull_from_chroma(self, state: GraphState) -> GraphState:
        """Pull relevant documents from Chroma vector DB."""
        print(f"Querying Chroma DB (contains {self.vector_db.count()} documents)...")
        
        results = self.vector_db.query(state["user_query"], n_results=3)
        state["chroma_results"] = results
        
        # Check if we have any results
        has_results = len(results.get("documents", [[]])[0]) > 0
        print(f"Found {len(results.get('documents', [[]])[0])} relevant documents in Chroma")
        
        return state
    
    def _check_relevance(self, state: GraphState) -> GraphState:
        """Check if retrieved context is relevant."""
        if not state["chroma_results"].get("documents", [[]])[0]:
            state["relevant_context"] = False
            print("No documents found in Chroma, will search arxiv")
            return state
        
        # Extract documents
        docs = state["chroma_results"]["documents"][0]
        combined_docs = "\n\n".join(docs[:2])  # Use top 2 documents
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that determines if provided context is relevant to answer a query. "
                      "Respond with only 'yes' or 'no'."),
            ("user", "Is this context relevant to answer the query?\n\n"
                    "Query: {query}\n\nContext: {context}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": state["user_query"], "context": combined_docs})
        
        relevant = "yes" in response.content.lower()
        state["relevant_context"] = relevant
        
        print(f"Context is relevant: {relevant}")
        return state
    
    def _search_arxiv(self, state: GraphState) -> GraphState:
        """Search arxiv for relevant papers."""
        print("Searching arxiv for papers on game theory...")
        
        # Create search query
        search_query = f"game theory {state['user_query']}"
        papers = self.arxiv_searcher.search_papers(search_query)
        
        state["arxiv_papers"] = papers
        print(f"Found {len(papers)} papers on arxiv")
        
        return state
    
    def _add_to_chroma(self, state: GraphState) -> GraphState:
        """Chunk papers and add them to Chroma DB."""
        print("Processing and adding papers to Chroma DB...")
        
        for paper in state["arxiv_papers"]:
            try:
                # Process paper into chunks
                chunks = self.doc_processor.process_paper(paper)
                
                # Extract data for Chroma
                documents = [chunk["text"] for chunk in chunks]
                metadatas = [chunk["metadata"] for chunk in chunks]
                ids = [chunk["id"] for chunk in chunks]
                
                # Add to vector DB
                self.vector_db.add_documents(documents, metadatas, ids)
                
                print(f"Added {len(chunks)} chunks from paper: {paper['title'][:50]}...")
            except Exception as e:
                print(f"Error adding paper to Chroma: {e}")
                # Continue with next paper even if one fails
                continue
        
        state["papers_added"] = True
        return state
    
    def _generate_response(self, state: GraphState) -> GraphState:
        """Generate final response using retrieved context."""
        # Get context from Chroma results
        if state["chroma_results"].get("documents", [[]])[0]:
            docs = state["chroma_results"]["documents"][0]
            context = "\n\n".join(docs[:3])  # Use top 3 documents
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions about game theory. "
                          "Use the provided context to answer the user's question."),
                ("user", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({"context": context, "query": state["user_query"]})
            state["final_response"] = response.content
        else:
            state["final_response"] = "I don't have enough information to answer your question about game theory."
        
        print("Generated final response")
        return state
    
    def _route_after_context_check(self, state: GraphState) -> Literal["pull_from_chroma", "generate_response"]:
        """Route based on whether context is needed."""
        if state["needs_context"]:
            return "pull_from_chroma"
        else:
            return "generate_response"
    
    def _route_after_relevance_check(self, state: GraphState) -> Literal["generate_response", "search_arxiv"]:
        """Route based on whether context is relevant."""
        if state["relevant_context"]:
            return "generate_response"
        else:
            return "search_arxiv"
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("check_needs_context", self._check_needs_context)
        workflow.add_node("pull_from_chroma", self._pull_from_chroma)
        workflow.add_node("check_relevance", self._check_relevance)
        workflow.add_node("search_arxiv", self._search_arxiv)
        workflow.add_node("add_to_chroma", self._add_to_chroma)
        workflow.add_node("generate_response", self._generate_response)
        
        # Set entry point
        workflow.set_entry_point("check_needs_context")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "check_needs_context",
            self._route_after_context_check,
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
            self._route_after_relevance_check,
            {
                "generate_response": "generate_response",
                "search_arxiv": "search_arxiv"
            }
        )
        
        # After searching arxiv, add to chroma
        workflow.add_edge("search_arxiv", "add_to_chroma")
        
        # After adding to chroma, go back to pull from chroma
        workflow.add_edge("add_to_chroma", "pull_from_chroma")
        
        # Generate response ends the workflow
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def query(self, user_query: str) -> str:
        """
        Process a user query through the workflow.
        
        Args:
            user_query: The user's question
            
        Returns:
            The final response
        """
        initial_state = {
            "user_query": user_query,
            "needs_context": False,
            "chroma_results": {},
            "relevant_context": False,
            "arxiv_papers": [],
            "papers_added": False,
            "final_response": ""
        }
        
        print(f"\n{'='*60}")
        print(f"Processing query: {user_query}")
        print(f"{'='*60}\n")
        
        final_state = self.workflow.invoke(initial_state)
        return final_state["final_response"]
