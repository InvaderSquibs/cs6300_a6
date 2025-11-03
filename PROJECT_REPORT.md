# Game Theory RAG Agent: Design & Evaluation Report

## 1. Introduction

This report documents the design, implementation, and evaluation of a Game Theory RAG (Retrieval-Augmented Generation) agent built using LangGraph. The agent combines ReAct-style reasoning, tool use, and retrieval to answer game theory questions by dynamically expanding its knowledge base from academic papers.

### Project Goals

- Implement a ReAct-style agent that reasons iteratively
- Use multiple tools (vector database, Arxiv search, document processing)
- Demonstrate clear environment-agent-environment loops
- Evaluate agent performance with quantitative metrics
- Create a self-improving system that expands its knowledge base

### System Overview

The agent is a domain-specific research assistant that:
1. Queries a vector database for relevant game theory context
2. Evaluates whether retrieved context is sufficient
3. Searches Arxiv for additional papers when needed
4. Filters papers for game theory relevance
5. Adds relevant papers to the knowledge base
6. Generates answers using retrieved context

## 2. PEAS Analysis

See [PEAS_ANALYSIS.md](PEAS_ANALYSIS.md) for the complete PEAS framework analysis.

### Summary

- **Performance**: Answer quality, knowledge base growth, response relevance, system efficiency
- **Environment**: Vector database (ChromaDB), Arxiv API, user query space
- **Actuators**: Vector DB query/add tools, Arxiv search, document processor, LLM reasoning
- **Sensors**: State fields (`chroma_results`, `arxiv_papers`, `relevant_context`, etc.)

The agent operates in a sequential, partially observable, dynamic environment with both persistent (vector DB) and external (Arxiv) components.

## 3. Architecture

### System Architecture

The system uses a modular LangGraph workflow with explicit dependency injection:

```
┌─────────────────────────────────────────────────────────┐
│                    GameTheoryRAG                        │
│  (Main Orchestrator)                                    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│ VectorDB    │   │ ArxivSearcher│   │ DocumentProc │
│ Manager     │   │              │   │              │
└─────────────┘   └──────────────┘   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ BaseChatModel│
                   │    (LLM)     │
                   └──────────────┘
```

### Workflow Architecture

The workflow is implemented as a LangGraph state machine:

```
User Query
    ↓
[Pull from Chroma DB]  ← Entry point
    ↓
[Check relevance of results]
    ↓
    ├─ RELEVANT → [Generate Response] → END
    │
    └─ NOT RELEVANT → [Search arxiv.org]
                          ↓
                    [Filter papers for game theory relevance]
                          ↓
                          ├─ Game theory papers found → [Add to Chroma]
                          │                                 ↓
                          │                          [Pull from Chroma DB] ← Loop back
                          │
                          └─ No game theory papers → [Generate Response] → END
```

See [WORKFLOW.md](WORKFLOW.md) for detailed workflow documentation.

### Modular Design

The system is organized into distinct modules:

- **Nodes** (`src/nodes/`): Functional node implementations
  - `retrieval_nodes.py`: Vector DB and Arxiv queries
  - `context_nodes.py`: Relevance evaluation
  - `filter_nodes.py`: Paper filtering
  - `processing_nodes.py`: Document processing and storage
  - `response_nodes.py`: Response generation

- **Edges** (`src/edges/`): Routing functions for conditional control flow
  - `routers.py`: State-based routing decisions

- **State** (`src/state.py`): `GraphState` TypedDict defining workflow state

- **Builder** (`src/graph_builder.py`): Automatic dependency injection

### Dependency Injection

Each node function explicitly declares its dependencies:

```python
def add_to_chroma(
    state: GraphState,
    vector_db: VectorDBManager,      # Explicit dependency
    doc_processor: DocumentProcessor # Explicit dependency
) -> GraphState:
    ...
```

The `WorkflowBuilder` automatically injects dependencies based on type hints, making the system:
- **Testable**: Pure functions with injected dependencies
- **Extensible**: Easy to add new nodes
- **Readable**: Dependencies are self-documenting

## 4. Tool Contracts

### Tool 1: Vector Database Manager (`VectorDBManager`)

**Location**: `src/vector_db.py`

**Purpose**: Manages persistent vector database for storing and retrieving document chunks.

**Input Contract**:
- `query(query: str, n_results: int = 3) -> Dict[str, Any]`
  - `query`: Search query string
  - `n_results`: Number of results to return (default: 3)
- `add_documents(documents: List[str], metadatas: List[Dict], ids: List[str]) -> None`
  - `documents`: List of document text strings
  - `metadatas`: List of metadata dictionaries
  - `ids`: List of unique document IDs

**Output Contract**:
- `query()` returns:
  ```python
  {
      "documents": [[str, ...]],      # List of document chunks
      "metadatas": [[Dict, ...]],     # List of metadata dicts
      "ids": [[str, ...]],            # List of document IDs
      "distances": [[float, ...]]     # Similarity distances
  }
  ```
- `add_documents()` returns `None` (void, modifies database)

**Preconditions**:
- Database directory exists or can be created
- Documents are non-empty strings
- IDs are unique within collection

**Postconditions**:
- `query()` returns results sorted by similarity (best matches first)
- `add_documents()` adds documents to persistent storage
- Database state persists across invocations

**Error Handling**:
- Invalid input: Raises `ValueError`
- Database errors: Logs error and continues

### Tool 2: Arxiv Searcher (`ArxivSearcher`)

**Location**: `src/arxiv_search.py`

**Purpose**: Searches arxiv.org for academic papers matching a query.

**Input Contract**:
- `search_papers(query: str) -> List[Dict[str, Any]]`
  - `query`: Search query string (supports arxiv search syntax)

**Output Contract**:
Returns list of paper dictionaries:
```python
[
    {
        "title": str,              # Paper title
        "summary": str,            # Abstract text
        "authors": List[str],      # Author names
        "published": str,          # Date (YYYY-MM-DD)
        "pdf_url": str,           # Direct URL to PDF
        "entry_id": str           # Arxiv entry ID
    },
    ...
]
```

**Preconditions**:
- Query string is non-empty
- Arxiv API is accessible (network connection)

**Postconditions**:
- Returns up to `max_results` papers (default: 2)
- Results sorted by relevance (arxiv default)
- Papers may be empty if no matches found

**Error Handling**:
- Network errors: Returns empty list, logs warning
- API errors: Returns empty list, continues execution

### Tool 3: Document Processor (`DocumentProcessor`)

**Location**: `src/document_processor.py`

**Purpose**: Processes papers into chunks with metadata for vector storage.

**Input Contract**:
- `process_paper(paper: Dict[str, Any]) -> List[Dict[str, Any]]`
  - `paper`: Dictionary with `title`, `summary`, `authors`, `published`, `entry_id`, `pdf_url`

**Output Contract**:
Returns list of chunk dictionaries:
```python
[
    {
        "text": str,              # Chunk text (title + abstract portion)
        "metadata": {
            "title": str,
            "authors": List[str],
            "published": str,
            "source": str,
            "chunk_index": int,
            "pdf_url": str
        },
        "id": str                 # Unique chunk ID
    },
    ...
]
```

**Preconditions**:
- Paper dictionary contains required fields
- `chunk_size` and `chunk_overlap` are positive integers

**Postconditions**:
- Text is chunked with specified size and overlap
- Each chunk has unique ID: `{entry_id}_chunk_{index}`
- Metadata preserved for all chunks

**Error Handling**:
- Missing fields: Uses defaults or empty strings
- Processing errors: Returns empty list, logs warning

### Tool 4: LLM Reasoning (`BaseChatModel`)

**Location**: LangChain integration (various nodes)

**Purpose**: Performs reasoning tasks (relevance evaluation, paper filtering, response generation).

**Input Contract**:
- Prompt strings (varies by task)
- Context documents (for response generation)

**Output Contract**:
- Classification tasks: "yes" or "no" responses
- Generation tasks: Full text responses

**Preconditions**:
- LLM service is accessible (local or API)
- Prompt is well-formed

**Postconditions**:
- Returns text response
- Classification responses are parsed to boolean

**Error Handling**:
- API errors: Logs error, uses fallback/default
- Timeout: Logs warning, retries or uses default

## 5. Environment-Agent-Environment Loop

See [ENVIRONMENT_AGENT_LOOP.md](ENVIRONMENT_AGENT_LOOP.md) for complete loop documentation.

### Loop Summary

The agent demonstrates a clear environment-agent-environment loop:

1. **Percept**: Agent reads state fields (e.g., `chroma_results`, `arxiv_papers`)
2. **Reasoning**: Agent evaluates state (e.g., `check_relevance` node)
3. **Action**: Agent calls tools (e.g., `vector_db.query()`, `arxiv_searcher.search_papers()`)
4. **Outcome**: Environment updates state (e.g., `chroma_results` populated, vector DB grows)
5. **New Percept**: Agent reads updated state
6. **Loop**: Process repeats until goal achieved

### Key Loop Examples

**Loop 1: Vector DB Retrieval**
- Percept: `user_query` in state
- Action: `VectorDBManager.query()`
- Outcome: `chroma_results` populated in state
- New Percept: Agent reads `chroma_results`

**Loop 2: Knowledge Base Expansion**
- Percept: `arxiv_papers` with filtered papers
- Action: `VectorDBManager.add_documents()`
- Outcome: Vector database grows (persistent environment change)
- New Percept: Re-query returns better results

**Loop 3: Self-Improving Cycle**
- Add papers → Re-query → Better results → Generate response
- Demonstrates the self-improving nature of the system

## 6. Evaluation Results

See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for detailed metrics.

### Summary

**Query Performance**:
- Success Rate: 100% (5/5 test queries)
- All queries produced non-empty responses

**Knowledge Base Growth**:
- System successfully adds papers from Arxiv
- Vector database grows from 0 to 10+ documents over test queries
- Average 2-5 document chunks added per growth event

**Tool Usage**:
- Vector DB queries: 100% (always called)
- Arxiv searches: 40-60% (used when context insufficient)
- Papers added: 2-3 events per test run
- Demonstrates adaptive tool usage

**Response Quality**:
- Average response length: 200-400 characters
- Context usage rate: 60-100%
- Responses are substantive and informative

**Loop Efficiency**:
- Average iterations: 1.4-2.0 per query
- Loop rate: 40-60% (queries requiring Arxiv search)
- System efficiently handles both simple and complex queries

### Strengths

1. **Self-Improving**: Knowledge base grows over time
2. **Adaptive**: Uses Arxiv only when needed
3. **Domain-Focused**: Filters papers for game theory relevance
4. **Robust**: Handles edge cases gracefully
5. **Efficient**: Avoids unnecessary external API calls

### Limitations

1. **Abstract-Only**: Uses paper abstracts, not full PDF text
2. **Limited Context Window**: Only uses top 3 documents
3. **No Conversation History**: Each query is independent
4. **Fixed Domain**: Specifically designed for game theory

## 7. Team Collaboration

### Development Process

The project was developed using:
- **Version Control**: Git for code management
- **Modular Architecture**: Separated concerns into distinct modules
- **Documentation**: Comprehensive docstrings and markdown documentation
- **Testing**: Unit tests and integration tests

### Code Quality

- **Modularity**: Clear separation of concerns (nodes, edges, state, tools)
- **Documentation**: Comprehensive docstrings on all functions and classes
- **Error Handling**: Try-except blocks for external dependencies
- **Type Hints**: Explicit type annotations for clarity
- **Reproducibility**: `requirements.txt` and clear setup instructions

### Collaboration Evidence

- **Git History**: Shows iterative development and refinements
- **Code Structure**: Modular design allows independent work on different components
- **Documentation**: Clear documentation enables collaboration
- **Testing**: Test suite ensures code quality and compatibility

### Individual Contributions

(If applicable, document individual contributions here. For solo projects, document the development process and decision-making.)

## 8. Conclusion

### Summary

The Game Theory RAG agent successfully demonstrates:

1. **ReAct-Style Reasoning**: Iterative workflow with reasoning at each step
2. **Tool Use**: Multiple tools (vector DB, Arxiv search, document processing, LLM)
3. **Retrieval**: Semantic search from vector database with dynamic expansion
4. **Environment-Agent Loop**: Clear percept → action → outcome cycles
5. **Evaluation**: Quantitative metrics show 100% success rate and efficient operation

### Key Achievements

- **Working System**: All test queries produce successful responses
- **Self-Improving**: Knowledge base grows over time
- **Adaptive**: Uses tools strategically based on context
- **Robust**: Handles edge cases gracefully
- **Well-Documented**: Comprehensive documentation for all components

### Future Improvements

1. **Full PDF Processing**: Extract text from downloaded PDFs
2. **Conversation History**: Maintain context across queries
3. **Query Refinement**: Expand/narrow queries based on results
4. **Multi-Domain Support**: Extend beyond game theory
5. **Response Quality Scoring**: Automated evaluation metrics

### Lessons Learned

1. **Modular Architecture**: Enables easy extension and testing
2. **Explicit Dependencies**: Makes code more readable and maintainable
3. **State Management**: TypedDict provides clear state contracts
4. **Iterative Design**: Self-improving loops create value over time
5. **Documentation**: Comprehensive docs essential for complex systems

## References

- **PEAS Analysis**: [PEAS_ANALYSIS.md](PEAS_ANALYSIS.md)
- **Environment Loop**: [ENVIRONMENT_AGENT_LOOP.md](ENVIRONMENT_AGENT_LOOP.md)
- **Evaluation Results**: [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)
- **Workflow Documentation**: [WORKFLOW.md](WORKFLOW.md)
- **Implementation Details**: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Usage Guide**: [README.md](README.md)

## Appendix

### Architecture Diagram

See `workflow_diagram.png` for visual workflow representation.

### Code Examples

See `example.py` and `query.py` for usage examples.

### Test Results

Run `evaluation_metrics.py` to collect new metrics.

