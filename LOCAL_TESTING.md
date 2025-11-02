# Local LLM Testing & Visualization Guide

This guide explains how to visualize the agent graph and test the system with local LLMs (Ollama or LM Studio).

## Graph Visualization

### Quick Start

Run the visualization script to generate visualizations of the workflow:

```bash
python3 visualize_graph.py
```

This will generate:
- **ASCII visualization**: Text-based diagram in the terminal (requires `grandalf` package)
- **Mermaid diagram**: Saved to `workflow_diagram.mmd`
- **PNG image**: Saved to `workflow_diagram.png`

### Viewing the Diagrams

1. **ASCII**: Displayed directly in the terminal
2. **Mermaid**: 
   - Copy contents of `workflow_diagram.mmd` and paste into [mermaid.live](https://mermaid.live)
   - VS Code: Install "Markdown Preview Mermaid Support" extension
   - GitHub: Automatically renders `.mmd` files
3. **PNG**: Open `workflow_diagram.png` in any image viewer

### Graph Structure

The workflow includes:
- **check_needs_context**: Routes based on whether query needs game theory context
- **pull_from_chroma**: Retrieves documents from vector DB
- **check_relevance**: Evaluates if retrieved context is sufficient
- **search_arxiv**: Searches arxiv.org for papers (if needed)
- **add_to_chroma**: Processes and stores papers
- **generate_response**: Creates final answer

The graph shows conditional routing with dotted lines (dashed edges) and regular flow with solid lines.

## Local LLM Testing

The system supports two local LLM options:

### Option 1: LM Studio (Recommended for OpenAI-compatible endpoints)

LM Studio provides an OpenAI-compatible API endpoint, making it easy to test with any model.

**Setup:**
1. **Install LM Studio**: Download from https://lmstudio.ai
2. **Load a Model**: 
   - Open LM Studio
   - Download and load a model (e.g., Llama 3.2, Mistral, etc.)
   - Start the local server (usually on port 1234)

**Test with LM Studio:**
```bash
python3 test_lm_studio.py
```

**Use in Code:**
```python
from src.workflow import GameTheoryRAG

# Initialize with LM Studio
rag = GameTheoryRAG(
    local_llm_base_url="http://localhost:1234/v1",  # LM Studio default
    local_llm_model="local-model",  # Your model name in LM Studio
    openai_api_key="lm-studio"  # Can be any string
)

response = rag.query("What is game theory?")
print(response)
```

### Option 2: Ollama

Ollama is a lightweight local LLM runner.

**Setup:**
1. **Install Ollama**: Visit https://ollama.ai and install
2. **Install a Model**:
   ```bash
   ollama pull llama3.2
   # Or try: llama3.2:1b, llama3.1, mistral, phi3
   ```
3. **Install Python Package**:
   ```bash
   pip install langchain-ollama
   ```

**Test with Ollama:**
```bash
python3 test_local_llm.py
```

**Use in Code:**
```python
from src.workflow import GameTheoryRAG

# Initialize with Ollama
rag = GameTheoryRAG(
    use_local_llm=True,
    local_llm_model="llama3.2"  # or any model you have
)

response = rag.query("What is game theory?")
print(response)
```

### Test Scripts

Both test scripts will:
- Check if the local LLM is running
- Initialize the RAG system
- Run test queries through the workflow
- Show the full execution flow

### Available Models (Ollama)

The Ollama test script tries these models in order:
- `llama3.2`
- `llama3.2:1b` (smaller, faster)
- `llama3.1`
- `llama3`
- `mistral`
- `phi3`

Check available models:
```bash
ollama list
```

### Troubleshooting

**LM Studio Connection Issues:**
- Make sure LM Studio is running and server is started
- Check that a model is loaded in LM Studio
- Verify the server URL (default: `http://localhost:1234/v1`)
- Update the model name to match what's shown in LM Studio

**Ollama Connection Issues:**
- Make sure Ollama is installed and in your PATH: `ollama --version`
- Ensure Ollama is running: `ollama serve` (usually automatic)
- Check available models: `ollama list`
- Pull missing models: `ollama pull <model-name>`
- Install Python package: `pip install langchain-ollama`

**General Model Issues:**
- Try a different model (some are better than others)
- Smaller models are faster but less capable
- Larger models give better results but are slower
- For LM Studio, check the model name matches exactly

## Benefits of Local LLM Testing

1. **No API Costs**: Test without using OpenAI credits
2. **Privacy**: Your queries stay local
3. **Offline**: Works without internet (after initial model download)
4. **Development**: Faster iteration during development
5. **Debugging**: Easier to debug with local execution

## Quick Reference

### Visualization
```bash
# Generate all visualizations (ASCII, Mermaid, PNG)
python3 visualize_graph.py
```

### Testing Local LLMs
```bash
# Test with LM Studio (OpenAI-compatible)
python3 test_lm_studio.py

# Test with Ollama
python3 test_local_llm.py
```

### Usage Examples

**LM Studio:**
```python
from src.workflow import GameTheoryRAG

rag = GameTheoryRAG(
    local_llm_base_url="http://localhost:1234/v1",
    local_llm_model="local-model",
    openai_api_key="lm-studio"
)
```

**Ollama:**
```python
from src.workflow import GameTheoryRAG

rag = GameTheoryRAG(
    use_local_llm=True,
    local_llm_model="llama3.2"
)
```

**OpenAI (default):**
```python
from src.workflow import GameTheoryRAG

rag = GameTheoryRAG()  # Uses OPENAI_API_KEY from .env
```

## Observability with Phoenix

Phoenix provides detailed tracing and observability for your LLM pipeline.

### Setup

1. **Install Phoenix:**
   ```bash
   pip install arize-phoenix
   ```

2. **Start Phoenix Server:**
   ```bash
   phoenix serve
   ```
   This starts a web UI at `http://localhost:6006`

### Running with Tracing

```bash
python3 test_with_phoenix.py
```

Or manually enable tracing:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=http://localhost:6006
python3 example.py
```

### Phoenix Features

- **Workflow Tracing**: See each node's execution
- **LLM Call Details**: Inputs, outputs, tokens, latency
- **Prompt Analysis**: Evaluate prompt effectiveness
- **Cost Tracking**: Monitor API usage
- **Debugging**: Understand routing decisions

### Phoenix UI

Once running, open `http://localhost:6006` to see:
- Real-time traces of your workflow
- Detailed LLM call information
- Performance metrics
- Error tracking

## Next Steps

- Visualize the graph: `python3 visualize_graph.py`
- Test with LM Studio: `python3 test_lm_studio.py`
- Test with Ollama: `python3 test_local_llm.py`
- Monitor with Phoenix: `python3 test_with_phoenix.py`
- Customize the workflow: Edit `src/workflow.py`
- Add your own queries: Modify the test scripts

