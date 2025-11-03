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

**Option 1: Command-Line Query (Recommended)**

Ask a single question:
```bash
python3 query.py "What is Nash equilibrium?"
```

Interactive mode:
```bash
python3 query.py --interactive
# or
python3 query.py -i
```

**Option 2: Example Script**

```bash
python3 example.py
```

Runs a few example queries automatically.

**Option 4: Use in Your Code**

```python
from src.workflow import GameTheoryRAG

# Initialize the system
rag = GameTheoryRAG()

# Ask a question
response = rag.query("What is the Nash equilibrium?")
print(response)

# Ask another question
response = rag.query("Explain the prisoner's dilemma")
print(response)
```

**Example Questions:**
- "What is Nash equilibrium?"
- "Explain the prisoner's dilemma"
- "What are mixed strategies in game theory?"
- "How does game theory apply to economics?"
- "What is a dominant strategy?"

### PDF Download Extension

The PDF downloader can be used standalone:

```python
from src.pdf_downloader import PDFDownloader

downloader = PDFDownloader()
pdf_path = downloader.download("http://arxiv.org/pdf/1234.5678.pdf")

# PDFs are saved to ./papers/ directory by default
# You can specify a custom directory:
pdf_path = downloader.download(url, download_dir="./my_downloads")
```

**PDF Storage Location:**
- Default directory: `./papers/` (in project root)
- Directory is created automatically if it doesn't exist
- Filename is extracted from URL (e.g., arxiv ID like `1406.2661v1.pdf`)
- Full path: `{project_root}/papers/{filename}.pdf`

### Visualization

Generate visualizations of the workflow graph:

```bash
python3 visualize_graph.py
```

This creates:
- `workflow_diagram.mmd` - Mermaid diagram
- `workflow_diagram.png` - PNG image
- ASCII visualization in terminal

### Testing & Verification

```bash
# Comprehensive verification
python3 verify_all_components.py

# Run test suite
pytest tests/

# Collect evaluation metrics
python3 evaluation_metrics.py
```

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for details.

### Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Complete design and evaluation report
- **[PEAS_ANALYSIS.md](PEAS_ANALYSIS.md)** - PEAS framework analysis
- **[ENVIRONMENT_AGENT_LOOP.md](ENVIRONMENT_AGENT_LOOP.md)** - Environment-agent loop documentation
- **[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)** - Quantitative evaluation metrics
- **[WORKFLOW.md](WORKFLOW.md)** - Detailed workflow documentation

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

The system uses a modular LangGraph workflow with explicit dependency injection. The architecture is designed for clarity, testability, and extensibility.

### Modular Design

The workflow is organized into distinct modules:

- **Nodes** (`src/nodes/`): Functional node implementations organized by concern
  - `context_nodes.py`: Context checking and relevance evaluation
  - `retrieval_nodes.py`: Data retrieval from vector DB and arxiv
  - `processing_nodes.py`: Document processing and storage
  - `response_nodes.py`: Response generation

- **Edges** (`src/edges/`): Routing functions for conditional workflow control
  - `routers.py`: Conditional routing based on state

- **State** (`src/state.py`): GraphState TypedDict defining workflow state

- **Builder** (`src/graph_builder.py`): WorkflowBuilder with automatic dependency injection

### Workflow Nodes

The workflow consists of 6 nodes:

1. **check_needs_context**: Determines if query is game theory related (uses: `llm`)
2. **pull_from_chroma**: Retrieves relevant documents from vector DB (uses: `vector_db`)
3. **check_relevance**: Evaluates if retrieved context is relevant (uses: `llm`)
4. **search_arxiv**: Searches arxiv.org for papers (uses: `arxiv_searcher`)
5. **add_to_chroma**: Processes and stores papers in vector DB (uses: `vector_db`, `doc_processor`)
6. **generate_response**: Creates final answer using context (uses: `llm`)

### Dependency Injection

Each node function explicitly declares its dependencies in its signature:

```python
def add_to_chroma(
    state: GraphState,
    vector_db: VectorDBManager,      # Explicit dependency
    doc_processor: DocumentProcessor  # Explicit dependency
) -> GraphState:
    ...
```

The `WorkflowBuilder` automatically injects dependencies by analyzing function signatures and type hints, making dependencies self-documenting and easy to test.

### Adding New Nodes

To add a new node:

1. Create a function with explicit dependencies:
```python
def my_new_node(
    state: GraphState,
    vector_db: VectorDBManager,
    llm: BaseChatModel
) -> GraphState:
    """Node that uses vector_db and llm."""
    # Implementation
    return state
```

2. Register the dependency in workflow.py (if new):
```python
self.dependencies = {
    ...
    "VectorDBManager": self.vector_db,
}
```

3. Add to workflow:
```python
workflow.add_node(
    "my_new_node",
    self.builder.create_node(my_new_node)
)
```

The builder automatically injects `vector_db` and `llm` based on type hints!

### Extension Pattern: Adding PDF Download Capability

The architecture demonstrates clean extensibility through the PDF download extension:

**1. Metadata Enhancement** (additive, non-breaking):
- `pdf_url` is stored in vector DB metadata alongside documents
- Existing workflow continues to work unchanged
- New metadata enables new capabilities

**2. Standalone Tool Creation**:
```python
# src/pdf_downloader.py - Independent tool
downloader = PDFDownloader()
pdf_path = downloader.download("http://arxiv.org/pdf/1234.5678.pdf")
```

**3. Workflow Branching**:
Same RAG results can branch to different actions:

```
Main RAG Path:
pull_from_chroma → check_relevance → generate_response

PDF Download Path:
pull_from_chroma → extract_pdf_urls_from_results → download_pdfs
```

Both paths use the same `chroma_results`, but metadata enables different downstream actions.

**4. Explicit Dependencies**:
```python
def extract_pdf_urls_from_results(
    state: GraphState,
    pdf_downloader: PDFDownloader  # Clear dependency
) -> GraphState:
    # Extract pdf_url from metadata, download PDFs
    ...
```

**Key Benefits**:
- ✓ Backward compatible: Existing workflow unchanged
- ✓ Tool independence: PDFDownloader usable standalone
- ✓ Metadata-driven: Vector DB metadata acts as contract
- ✓ Clean composition: Same results, different actions
- ✓ Easy testing: Pure functions with explicit dependencies

## Components

- `src/workflow.py`: Main LangGraph workflow orchestrator
- `src/state.py`: GraphState TypedDict definition
- `src/nodes/`: Node function implementations
- `src/edges/`: Routing functions
- `src/graph_builder.py`: Dependency injection builder
- `src/vector_db.py`: ChromaDB integration
- `src/arxiv_search.py`: Arxiv paper search
- `src/document_processor.py`: Document chunking utilities
- `src/pdf_downloader.py`: Standalone PDF download tool (optional extension)
- `src/nodes/pdf_nodes.py`: PDF workflow nodes (optional extension)

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
