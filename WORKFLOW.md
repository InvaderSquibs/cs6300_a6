# Game Theory RAG Workflow

## Workflow Diagram

```
User Query
    ↓
[Pull from Chroma DB]  ← ALWAYS starts here (no initial gate)
    ↓
[Check relevance of results]
    ↓
    ├─ RELEVANT → [Generate Response] → END
    │
    └─ NOT RELEVANT → [Search arxiv.org with "game theory" prefix]
                          ↓
                    [Filter papers for game theory relevance]
                          ↓
                          ├─ Game theory papers found → [Chunk & Add to Chroma]
                          │                                 ↓
                          │                          [Pull from Chroma DB] ← (loop back)
                          │
                          └─ No game theory papers → [Generate Response] → END
                                                       (indicates not game theory related)
```

**Key Architectural Improvement:**
- Removed initial hard gate at `check_needs_context`
- Always searches vector DB first (more permissive)
- Searches arxiv with "game theory" prefix when needed
- Filters papers AFTER retrieval (can discover new concepts)
- Only rejects if: no vector DB results AND no game theory papers found

## Node Descriptions

### 1. Pull from Chroma
- **Location**: `src/nodes/retrieval_nodes.py::pull_from_chroma`
- **Dependencies**: `vector_db` (VectorDBManager)
- Queries ChromaDB vector database for relevant documents
- Returns top 3 most relevant chunks

### 2. Check Relevance
- **Location**: `src/nodes/context_nodes.py::check_relevance`
- **Dependencies**: `llm` (BaseChatModel)
- Uses LLM to evaluate if retrieved context is sufficient
- Routes to either response generation or arxiv search

### 3. Search Arxiv
- **Location**: `src/nodes/retrieval_nodes.py::search_arxiv`
- **Dependencies**: `arxiv_searcher` (ArxivSearcher)
- Searches arxiv.org with query prefixed by "game theory"
- Retrieves paper metadata (title, abstract, authors, pdf_url, etc.)
- Filters out papers already seen to prevent infinite loops

### 4. Filter Game Theory Papers
- **Location**: `src/nodes/filter_nodes.py::filter_game_theory_papers`
- **Dependencies**: `llm` (BaseChatModel)
- Filters arxiv results to only include papers actually related to game theory
- Uses LLM to evaluate each paper's title and abstract
- Only keeps papers classified as game theory related
- Tracks seen papers to prevent re-processing

### 5. Add to Chroma
- **Location**: `src/nodes/processing_nodes.py::add_to_chroma`
- **Dependencies**: `vector_db` (VectorDBManager), `doc_processor` (DocumentProcessor)
- Processes papers into chunks with overlap
- Stores chunks with metadata in ChromaDB
- Loops back to "Pull from Chroma" with updated database

### 6. Generate Response
- **Location**: `src/nodes/response_nodes.py::generate_response`
- **Dependencies**: `llm` (BaseChatModel)
- Uses retrieved context to generate final answer
- Falls back to message indicating query is not game theory related if no context available
- Ends the workflow

## Routing Functions

### Route After Relevance Check
- **Location**: `src/edges/routers.py::route_after_relevance_check`
- Routes to "generate_response" if context is relevant
- Routes to "search_arxiv" if context is not relevant

### Route After Paper Filter
- **Location**: `src/edges/routers.py::route_after_paper_filter`
- Routes to "add_to_chroma" if game theory papers found
- Routes to "generate_response" if no game theory papers found
- Prevents infinite loops by checking papers_seen

## Dependency Relationships

**Shared Tools:**
- `llm`: Used by nodes (check_relevance, filter_game_theory_papers, generate_response)
- `vector_db`: Used by 2 nodes (pull_from_chroma, add_to_chroma)
- `arxiv_searcher`: Used by 1 node (search_arxiv)
- `doc_processor`: Used by 1 node (add_to_chroma)

All dependencies are explicitly declared in node function signatures and automatically injected by `WorkflowBuilder` in `src/graph_builder.py`.

## Key Features

- **Modular Architecture**: Nodes organized by concern in separate files
- **Explicit Dependencies**: Function signatures make dependencies clear
- **Automatic Injection**: WorkflowBuilder handles dependency wiring
- **Conditional Routing**: Dynamically routes based on query type and context availability
- **Self-Improving**: Automatically expands knowledge base when needed
- **Context-Aware**: Only fetches external data when necessary
- **Efficient**: Reuses existing knowledge from vector database

## Related Documentation

- **[PROJECT_REPORT.md](../PROJECT_REPORT.md)** - Complete design and evaluation report
- **[PEAS_ANALYSIS.md](../PEAS_ANALYSIS.md)** - PEAS framework analysis
- **[ENVIRONMENT_AGENT_LOOP.md](../ENVIRONMENT_AGENT_LOOP.md)** - Detailed environment-agent loop documentation
- **[EVALUATION_RESULTS.md](../EVALUATION_RESULTS.md)** - Quantitative evaluation results
