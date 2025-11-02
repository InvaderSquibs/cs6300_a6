"""
Test the Game Theory RAG system with Phoenix observability/tracing.
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG


def test_with_phoenix():
    """Test the RAG system with Phoenix tracing enabled."""
    print("="*60)
    print("Phoenix Observability Test - Game Theory RAG System")
    print("="*60)
    print()
    
    load_dotenv()
    
    # Check if Phoenix is installed
    try:
        import phoenix as px
        print("✓ Phoenix is installed")
    except ImportError:
        print("✗ Phoenix is not installed")
        print("  Install with: pip install arize-phoenix")
        print("\n  Then start Phoenix server with:")
        print("    phoenix serve")
        return
    
    # Start Phoenix server
    print("\nStarting Phoenix server...")
    print("  (Make sure port 6006 is available)")
    print()
    
    try:
        # Launch Phoenix - it will run in background
        session = px.launch_app(host="0.0.0.0", port=6006)
        print(f"✓ Phoenix UI available at: {session.url}")
        print(f"  Open in browser to see traces in real-time")
        print(f"  Press Ctrl+C to stop Phoenix server")
        print()
    except Exception as e:
        print(f"⚠ Could not auto-start Phoenix: {e}")
        print("  You can start Phoenix server manually:")
        print("    phoenix serve")
        print("  Then open http://localhost:6006")
        print()
        print("  Or set environment variables:")
        print("    export LANGCHAIN_TRACING_V2=true")
        print("    export LANGCHAIN_ENDPOINT=http://localhost:6006")
        print()
        # Continue without auto-start
        session = None
    
    # Initialize RAG system with tracing
    print("-"*60)
    print("Initializing RAG system with Phoenix tracing...")
    print("-"*60)
    
    # Enable LangSmith tracing (Phoenix is compatible)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:6006"
    
    # Check for API key or use local LLM
    api_key = os.getenv("OPENAI_API_KEY")
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL")
    
    if local_base_url:
        print(f"Using local LLM at: {local_base_url}")
        rag = GameTheoryRAG(
            local_llm_base_url=local_base_url,
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "local-model"),
            openai_api_key="lm-studio"
        )
    elif use_local:
        print("Using Ollama local LLM")
        rag = GameTheoryRAG(use_local_llm=True)
    elif api_key:
        print("Using OpenAI API")
        rag = GameTheoryRAG(openai_api_key=api_key)
    else:
        print("✗ No LLM configuration found")
        print("  Set OPENAI_API_KEY, USE_LOCAL_LLM=true, or LOCAL_LLM_BASE_URL")
        return
    
    print("\n" + "="*60)
    print("Running Test Query (check Phoenix UI for traces)")
    print("="*60)
    print()
    
    test_query = "What is game theory?"
    print(f"Query: {test_query}")
    print()
    
    try:
        response = rag.query(test_query)
        print(f"\n{'─'*60}")
        print("RESPONSE:")
        print(f"{'─'*60}")
        print(response)
        print()
        print(f"\n{'─'*60}")
        print("✓ Query completed!")
        phoenix_url = session.url if session else "http://localhost:6006"
        print(f"  Check Phoenix UI at {phoenix_url}")
        print(f"  to see detailed traces of the workflow execution")
        print(f"{'─'*60}")
    except Exception as e:
        print(f"\n✗ Error processing query: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPhoenix features:")
    print("  - View all LLM calls and their inputs/outputs")
    print("  - See workflow node execution times")
    print("  - Track token usage and costs")
    print("  - Analyze prompt effectiveness")
    print("  - Debug routing decisions")


if __name__ == "__main__":
    test_with_phoenix()

