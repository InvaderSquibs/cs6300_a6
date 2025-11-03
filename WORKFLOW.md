# Game Theory RAG Workflow

## Workflow Diagram

```
User Query
    ↓
[Check if needs game theory context]
    ↓
    ├─ YES → [Pull from Chroma DB]
    │            ↓
    │        [Check relevance of results]
    │            ↓
    │            ├─ RELEVANT → [Generate Response] → END
    │            │
    │            └─ NOT RELEVANT → [Search arxiv.org]
    │                                   ↓
    │                              [Chunk & Add to Chroma]
    │                                   ↓
    │                              [Pull from Chroma DB] ← (loop back)
    │
    └─ NO → [Generate Response] → END
```

## Node Descriptions

### 1. Check Needs Context
- **Location**: `src/nodes/context_nodes.py::check_needs_context`
- **Dependencies**: `llm` (BaseChatModel)
- Uses LLM to determine if query is related to game theory
- Routes to either Chroma DB query or direct response

### 2. Pull from Chroma
- **Location**: `src/nodes/retrieval_nodes.py::pull_from_chroma`
- **Dependencies**: `vector_db` (VectorDBManager)
- Queries ChromaDB vector database for relevant documents
- Returns top 3 most relevant chunks

### 3. Check Relevance
- **Location**: `src/nodes/context_nodes.py::check_relevance`
- **Dependencies**: `llm` (BaseChatModel)
- Uses LLM to evaluate if retrieved context is sufficient
- Routes to either response generation or arxiv search

### 4. Search Arxiv
- **Location**: `src/nodes/retrieval_nodes.py::search_arxiv`
- **Dependencies**: `arxiv_searcher` (ArxivSearcher)
- Searches arxiv.org for relevant game theory papers
- Retrieves paper metadata (title, abstract, authors, etc.)

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
- Ends the workflow

## Routing Functions

### Route After Context Check
- **Location**: `src/edges/routers.py::route_after_context_check`
- Routes based on `needs_context` flag
- Returns: "pull_from_chroma" or "generate_response"

### Route After Relevance Check
- **Location**: `src/edges/routers.py::route_after_relevance_check`
- Routes based on `relevant_context` flag
- Returns: "generate_response" or "search_arxiv"

## Dependency Relationships

**Shared Tools:**
- `llm`: Used by 3 nodes (check_needs_context, check_relevance, generate_response)
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
