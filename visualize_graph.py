"""
Visualize the LangGraph workflow.
"""
import os
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG


def visualize_graph():
    """Generate visualization of the LangGraph workflow."""
    load_dotenv()
    
    # Try to initialize with local LLM first (doesn't need API key)
    try:
        print("Attempting to use local LLM for visualization...")
        rag = GameTheoryRAG(use_local_llm=True)
    except Exception as e:
        print(f"Could not use local LLM: {e}")
        # Try with OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("Using OpenAI API key for visualization...")
            rag = GameTheoryRAG(openai_api_key=api_key)
        else:
            print("Warning: No API key or local LLM available.")
            print("Creating dummy LLM for visualization only...")
            # Create a minimal mock just for graph structure
            from langchain_core.language_models import BaseChatModel
            from langchain_core.messages import BaseMessage
            
            class DummyLLM(BaseChatModel):
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    from langchain_core.outputs import ChatGeneration, ChatResult
                    return ChatResult(generations=[ChatGeneration(message=BaseMessage(content="yes"))])
                
                @property
                def _llm_type(self):
                    return "dummy"
            
            rag = GameTheoryRAG(llm=DummyLLM())
    
    # Get the compiled graph for visualization (has draw_mermaid method)
    graph = rag.workflow.get_graph()
    
    print("\n" + "="*60)
    print("LangGraph Workflow Visualization")
    print("="*60 + "\n")
    
    # Print ASCII visualization
    print("ASCII Graph Structure:")
    print("-" * 60)
    try:
        graph.print_ascii()
    except (AttributeError, ImportError) as e:
        # Some versions might use different method or need grandalf
        if "grandalf" in str(e):
            print("(ASCII visualization requires 'grandalf' package)")
            print("  Install with: pip install grandalf")
        else:
            print("(ASCII visualization not available in this version)")
    print()
    
    # Generate Mermaid diagram
    print("Mermaid Diagram:")
    print("-" * 60)
    try:
        mermaid_diagram = graph.draw_mermaid()
        print(mermaid_diagram)
        
        # Save Mermaid diagram to file
        mermaid_file = "workflow_diagram.mmd"
        with open(mermaid_file, "w") as f:
            f.write(mermaid_diagram)
        print(f"\n✓ Saved Mermaid diagram to {mermaid_file}")
        print(f"  You can view it at https://mermaid.live/ or in any Mermaid-compatible viewer")
        
        # Try to generate PNG if possible
        try:
            png_data = graph.draw_mermaid_png()
            if png_data:
                # draw_mermaid_png returns bytes, save to file
                png_file = "workflow_diagram.png"
                if isinstance(png_data, bytes):
                    with open(png_file, "wb") as f:
                        f.write(png_data)
                    print(f"✓ Saved PNG diagram to {png_file}")
                elif isinstance(png_data, str):
                    # If it returns a path, just report it
                    print(f"✓ PNG diagram at {png_data}")
                else:
                    print(f"Note: PNG data format unexpected: {type(png_data)}")
        except Exception as e:
            print(f"Note: PNG generation not available ({e})")
            print("  Use the Mermaid file with an online viewer or VS Code extension")
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        print("\nGraph structure:")
        print(f"Nodes: {list(graph.nodes.keys()) if hasattr(graph, 'nodes') else 'N/A'}")


if __name__ == "__main__":
    visualize_graph()

