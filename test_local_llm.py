"""
Test the Game Theory RAG system with a local LLM (Ollama).
"""
import sys
from src.workflow import GameTheoryRAG


def check_ollama_installed():
    """Check if Ollama is installed and running."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            print(f"Available models:")
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    print(f"  - {line.split()[0]}")
            return True
        else:
            print("✗ Ollama command failed")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed or not in PATH")
        print("  Install from: https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Ollama command timed out")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


def test_local_llm():
    """Test the RAG system with local LLM."""
    print("="*60)
    print("Local LLM Test - Game Theory RAG System")
    print("="*60)
    print()
    
    # Check Ollama
    if not check_ollama_installed():
        print("\nPlease install and run Ollama before testing.")
        print("Visit https://ollama.ai to get started.")
        sys.exit(1)
    
    # Check for langchain-ollama
    try:
        import langchain_ollama
        print("✓ langchain-ollama is installed")
    except ImportError:
        print("✗ langchain-ollama is not installed")
        print("  Install with: pip install langchain-ollama")
        sys.exit(1)
    
    print("\n" + "-"*60)
    print("Initializing RAG system with local LLM...")
    print("-"*60)
    
    # Initialize with local LLM
    # Try common model names
    models_to_try = ["llama3.2", "llama3.2:1b", "llama3.1", "llama3", "mistral", "phi3"]
    
    rag = None
    model_used = None
    
    for model in models_to_try:
        try:
            print(f"\nTrying model: {model}...")
            rag = GameTheoryRAG(use_local_llm=True, local_llm_model=model)
            model_used = model
            print(f"✓ Successfully initialized with {model}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if not rag:
        print("\n✗ Could not initialize with any model.")
        print("  Make sure at least one model is available.")
        print("  Pull a model with: ollama pull llama3.2")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Running Test Queries")
    print("="*60)
    print()
    
    # Test queries (simple ones to start)
    test_queries = [
        "What is game theory?",  # Should trigger arxiv search on first run
        "Explain the prisoner's dilemma"  # Should use cached data
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print("="*60)
        print()
        
        try:
            response = rag.query(query)
            print(f"\n{'─'*60}")
            print("RESPONSE:")
            print(f"{'─'*60}")
            print(response)
            print()
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"\nUsed model: {model_used}")
    print("\nYou can visualize the graph by running:")
    print("  python visualize_graph.py")


if __name__ == "__main__":
    test_local_llm()

