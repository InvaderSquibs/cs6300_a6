# cs6300_a6 - Game Theory RAG System

A LangGraph-based RAG system that:
1. Checks if a conversation needs game theory context
2. Pulls relevant context from ChromaDB vector database
3. If context is relevant, uses it to answer questions
4. If not relevant, searches arxiv.org for papers
5. Chunks and adds papers to the vector database
6. Re-queries the database with updated context

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

Run the example script:

```bash
python example.py
```

Or use it in your own code:

```python
from src.workflow import GameTheoryRAG

rag = GameTheoryRAG()
response = rag.query("What is the Nash equilibrium?")
print(response)
```

### Visualization

Generate visualizations of the workflow graph:

```bash
python3 visualize_graph.py
```

This creates:
- `workflow_diagram.mmd` - Mermaid diagram
- `workflow_diagram.png` - PNG image
- ASCII visualization in terminal

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for details.

### Local LLM Testing

Test with local LLMs (LM Studio or Ollama):

```bash
# LM Studio (OpenAI-compatible)
python3 test_lm_studio.py

# Ollama
python3 test_local_llm.py
```

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for full instructions.

### Observability with Phoenix

Monitor and trace your pipeline with Phoenix:

```bash
# Install Phoenix
pip install arize-phoenix

# Run with tracing
python3 test_with_phoenix.py
```

Phoenix provides:
- Real-time workflow tracing
- LLM call monitoring
- Performance metrics
- Cost tracking

See [LOCAL_TESTING.md](LOCAL_TESTING.md#observability-with-phoenix) for details.

## Architecture

The system uses a LangGraph workflow with the following nodes:

1. **check_needs_context**: Determines if query is game theory related
2. **pull_from_chroma**: Retrieves relevant documents from vector DB
3. **check_relevance**: Evaluates if retrieved context is relevant
4. **search_arxiv**: Searches arxiv.org for papers (if no relevant context)
5. **add_to_chroma**: Processes and stores papers in vector DB
6. **generate_response**: Creates final answer using context

## Components

- `src/workflow.py`: Main LangGraph workflow
- `src/vector_db.py`: ChromaDB integration
- `src/arxiv_search.py`: Arxiv paper search
- `src/document_processor.py`: Document chunking utilities
