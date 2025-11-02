"""
Test the Game Theory RAG system with LM Studio (OpenAI-compatible endpoint).
"""
import sys
from src.workflow import GameTheoryRAG


def test_lm_studio():
    """Test the RAG system with LM Studio."""
    print("="*60)
    print("LM Studio Test - Game Theory RAG System")
    print("="*60)
    print()
    
    # LM Studio default settings
    base_url = "http://localhost:1234/v1"  # LM Studio default
    model = "local-model"  # Change to your model name in LM Studio
    
    print("LM Studio Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print("\nMake sure LM Studio is running with a model loaded!")
    print("  - Open LM Studio")
    print("  - Load a model")
    print("  - Start the local server (usually on port 1234)")
    print()
    
    try:
        response = input("Press Enter to continue or Ctrl+C to cancel... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    print("\n" + "-"*60)
    print("Initializing RAG system with LM Studio...")
    print("-"*60)
    
    try:
        rag = GameTheoryRAG(
            local_llm_base_url=base_url,
            local_llm_model=model,
            openai_api_key="lm-studio"  # LM Studio doesn't require real key
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure LM Studio is running")
        print("  2. Make sure a model is loaded in LM Studio")
        print("  3. Check that the server is on http://localhost:1234/v1")
        print("  4. Try updating the model name to match what's in LM Studio")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Running Test Queries")
    print("="*60)
    print()
    
    # Test queries (simple ones to start)
    test_queries = [
        "What is game theory?",  # Should trigger arxiv search on first run
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
            break
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_lm_studio()

