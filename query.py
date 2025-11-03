#!/usr/bin/env python3
"""
Command-line script to ask questions to the Game Theory RAG system.

Usage:
    python query.py "What is Nash equilibrium?"
    python query.py "Explain the prisoner's dilemma"
    python query.py --interactive  # Interactive mode
"""
import sys
import argparse
from src.workflow import GameTheoryRAG


def main():
    """Main entry point for command-line queries."""
    parser = argparse.ArgumentParser(
        description="Ask questions to the Game Theory RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 %(prog)s "What is Nash equilibrium?"
  python3 %(prog)s "Explain the prisoner's dilemma"
  python3 %(prog)s --interactive
  python3 %(prog)s -i
        """
    )
    
    parser.add_argument(
        'question',
        nargs='?',
        help='The question to ask'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (ask multiple questions)'
    )
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    try:
        rag = GameTheoryRAG(
            local_llm_base_url='http://localhost:1234/v1',
            local_llm_model='local-model',
            openai_api_key='lm-studio'
        )
    except Exception as e:
        print(f"✗ Failed to initialize: {e}", file=sys.stderr)
        print("\nMake sure LM Studio is running on localhost:1234", file=sys.stderr)
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("="*70)
        print("Game Theory RAG System - Interactive Mode")
        print("="*70)
        print("(Type 'quit' or 'exit' to stop, or press Ctrl+C)\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                print(f"\nProcessing: {question}")
                print("-" * 70)
                
                response = rag.query(question)
                
                print("\n" + "="*70)
                print("ANSWER:")
                print("="*70)
                print(response)
                print("\n" + "="*70)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n", file=sys.stderr)
    
    # Single question mode
    elif args.question:
        question = args.question
        print(f"Question: {question}")
        print("="*70)
        print()
        
        try:
            response = rag.query(question)
            print("ANSWER:")
            print("="*70)
            print(response)
        except Exception as e:
            print(f"✗ Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # No question provided
    else:
        parser.print_help()
        print("\nExample:")
        print('  python query.py "What is Nash equilibrium?"')
        sys.exit(1)


if __name__ == "__main__":
    main()

