"""
Test the Game Theory RAG system with Phoenix observability/tracing.

This script automatically starts Phoenix and runs with LM Studio (default local LLM).
"""
import os
import sys
import subprocess
import time
import urllib.request
from dotenv import load_dotenv

# Set tracing BEFORE any LangChain imports
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:6006"
os.environ["LANGCHAIN_API_KEY"] = "phoenix"  # Dummy key for Phoenix

from src.workflow import GameTheoryRAG


def test_with_phoenix():
    """Test the RAG system with Phoenix tracing enabled."""
    print("="*60)
    print("Phoenix Observability Test - Game Theory RAG System")
    print("="*60)
    print()
    
    load_dotenv()
    
    print("Starting Phoenix server automatically...")
    print("-" * 60)
    
    # Check if Phoenix is already running
    phoenix_running = False
    try:
        response = urllib.request.urlopen("http://localhost:6006/health", timeout=2)
        if response.status == 200:
            phoenix_running = True
            print("✓ Phoenix server is already running")
    except Exception:
        pass
    
    # Start Phoenix if not running
    phoenix_process = None
    if not phoenix_running:
        print("Starting Phoenix server on port 6006...")
        try:
            # Try using python module approach (most reliable)
            phoenix_process = subprocess.Popen(
                [sys.executable, "-m", "phoenix", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy()
            )
            
            # Wait for Phoenix to start (check health endpoint)
            max_wait = 10
            for i in range(max_wait):
                time.sleep(1)
                try:
                    response = urllib.request.urlopen("http://localhost:6006/health", timeout=1)
                    if response.status == 200:
                        phoenix_running = True
                        print(f"✓ Phoenix server started successfully (took {i+1}s)")
                        break
                except Exception:
                    continue
            
            if not phoenix_running:
                print("⚠ Phoenix server didn't start in time")
                print("  You may need to start it manually: python3 -m phoenix serve")
        except FileNotFoundError:
            print("⚠ Could not start Phoenix (phoenix module not found)")
            print("  Install with: pip install arize-phoenix")
            print("  Then start manually: python3 -m phoenix serve")
        except Exception as e:
            print(f"⚠ Could not auto-start Phoenix: {e}")
            print("  Start manually: python3 -m phoenix serve")
    
    if phoenix_running:
        print(f"✓ Phoenix UI available at: http://localhost:6006")
        print(f"  Open in browser to see traces in real-time")
    else:
        print("⚠ Phoenix server is not running")
        print("  Start it manually: python3 -m phoenix serve")
        print("  Then run this script again")
    
    print()
    
    # Initialize RAG system with tracing
    print("-"*60)
    print("Initializing RAG system with Phoenix tracing...")
    print("-"*60)
    
    # Tracing already enabled at import time
    print("✓ Tracing enabled:")
    print(f"  LANGCHAIN_TRACING_V2={os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"  LANGCHAIN_ENDPOINT={os.getenv('LANGCHAIN_ENDPOINT')}")
    print()
    
    # Default to LM Studio (system default)
    api_key = os.getenv("OPENAI_API_KEY")
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")  # Default to LM Studio
    
    print("🔧 LLM Configuration:")
    print("  Default: LM Studio at http://localhost:1234/v1")
    
    # Use LM Studio by default unless overridden
    if not api_key and not use_local:
        # Default: LM Studio
        model_name = os.getenv("LOCAL_LLM_MODEL", "local-model")
        print(f"  Using: LM Studio at {local_base_url}")
        print(f"  Model: {model_name}")
        print()
        print("💡 Make sure LM Studio is running with a model loaded!")
        print(f"   Verify: curl {local_base_url}/models")
        print()
        
        rag = GameTheoryRAG(
            local_llm_base_url=local_base_url,
            local_llm_model=model_name,
            openai_api_key="lm-studio"
        )
    elif use_local:
        print("  Using: Ollama")
        rag = GameTheoryRAG(use_local_llm=True)
    elif api_key:
        print("  Using: OpenAI API")
        rag = GameTheoryRAG(openai_api_key=api_key)
    else:
        print("✗ Configuration error")
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
        print(f"  Check Phoenix UI at http://localhost:6006")
        print(f"  to see detailed traces of the workflow execution")
        print(f"{'─'*60}")
        
        if phoenix_process:
            print("\n💡 Phoenix server is running in background")
            print("   Press Ctrl+C to stop it when done")
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
    print()
    if phoenix_process:
        try:
            phoenix_process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass  # Process still running, which is expected


if __name__ == "__main__":
    test_with_phoenix()

