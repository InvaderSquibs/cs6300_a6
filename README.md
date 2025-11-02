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

The system defaults to using **LM Studio** (local LLM) at `http://localhost:1234/v1`.

### Option 1: LM Studio (Default - Recommended)

1. Install LM Studio from https://lmstudio.ai
2. Load a model in LM Studio
3. Start the local server (usually on port 1234)
4. That's it! The system will automatically use LM Studio.

### Option 2: OpenAI API

If you prefer OpenAI, create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Option 3: Ollama

Set environment variable:
```
USE_LOCAL_LLM=true
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

### Local LLM Testing (Default: LM Studio)

The system **defaults to LM Studio** for local LLM testing:

```bash
# Test with LM Studio (default)
python3 test_with_phoenix.py  # Also includes observability

# Or test LM Studio specifically
python3 test_lm_studio.py

# Test with Ollama (optional)
python3 test_local_llm.py
```

**LM Studio Setup:**
1. Install from https://lmstudio.ai
2. Load a model in LM Studio
3. Start local server (default port 1234)
4. Run the test scripts - they'll automatically connect!

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

## Testing & Verification

### Vector DB Growth Verification

Test that the vector database grows when new papers are found:

```bash
# Run growth test (resets DB to empty baseline)
python3 test_vector_db_growth.py

# Save current DB state as baseline
python3 test_vector_db_growth.py save

# Reset DB to baseline
python3 test_vector_db_growth.py reset
```

The test will:
1. Reset DB to baseline state (empty or saved)
2. Run a query that triggers arxiv search
3. Verify DB count growth
4. Show before/after document counts
