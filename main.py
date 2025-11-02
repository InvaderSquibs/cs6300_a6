"""
Main entry point for the Game Theory RAG system.
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG


def main():
    """Run the Game Theory RAG system with example queries."""
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        print("Example: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize the RAG system
    print("Initializing Game Theory RAG system...")
    rag = GameTheoryRAG(api_key)
    
    # Example queries
    queries = [
        "What is the Nash equilibrium?",
        "Explain the prisoner's dilemma in game theory",
        "What's the weather like today?"  # Non-game-theory query
    ]
    
    for query in queries:
        response = rag.query(query)
        print(f"\nQuestion: {query}")
        print(f"Answer: {response}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
