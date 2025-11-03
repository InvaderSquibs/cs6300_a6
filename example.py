"""
Example usage of the Game Theory RAG system.

This demonstrates basic usage patterns. For more examples, see:
- query.py - Command-line interface
- verify_all_components.py - Component verification
- tests/ - Comprehensive test suite
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG


def run_example():
    """Run simple examples of the RAG system."""
    load_dotenv()
    
    print("="*60)
    print("Game Theory RAG System - Examples")
    print("="*60)
    print()
    
    rag = GameTheoryRAG()
    
    # Example 1: Basic game theory question
    print("Example 1: Basic game theory question")
    print("-" * 60)
    response1 = rag.query("What is Nash equilibrium?")
    print(f"\nResponse: {response1[:200]}...\n")
    
    # Example 2: Related follow-up
    print("\nExample 2: Related follow-up question")
    print("-" * 60)
    response2 = rag.query("What are the main concepts in game theory?")
    print(f"\nResponse: {response2[:200]}...\n")
    
    print("="*60)
    print("For more examples, use: python3 query.py")
    print("="*60)


if __name__ == "__main__":
    run_example()
