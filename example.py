"""
Example usage of the Game Theory RAG system.
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG


def run_example():
    """Run a simple example of the RAG system."""
    load_dotenv()
    
    # Initialize
    print("="*60)
    print("Game Theory RAG System - Example")
    print("="*60)
    print()
    
    rag = GameTheoryRAG()
    
    # First query - will search arxiv since DB is empty
    print("Example 1: Query on empty database")
    print("-" * 60)
    response1 = rag.query("What is game theory?")
    print(f"\nResponse: {response1}\n")
    
    # Second query - should use cached data
    print("\nExample 2: Query with populated database")
    print("-" * 60)
    response2 = rag.query("What are the main concepts in game theory?")
    print(f"\nResponse: {response2}\n")
    
    # Non-game-theory query
    print("\nExample 3: Non-game-theory query")
    print("-" * 60)
    response3 = rag.query("How do I bake a cake?")
    print(f"\nResponse: {response3}\n")


if __name__ == "__main__":
    run_example()
