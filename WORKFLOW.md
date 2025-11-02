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
- Uses LLM to determine if query is related to game theory
- Routes to either Chroma DB query or direct response

### 2. Pull from Chroma
- Queries ChromaDB vector database for relevant documents
- Returns top 3 most relevant chunks

### 3. Check Relevance
- Uses LLM to evaluate if retrieved context is sufficient
- Routes to either response generation or arxiv search

### 4. Search Arxiv
- Searches arxiv.org for relevant game theory papers
- Retrieves paper metadata (title, abstract, authors, etc.)

### 5. Add to Chroma
- Processes papers into chunks with overlap
- Stores chunks with metadata in ChromaDB
- Loops back to "Pull from Chroma" with updated database

### 6. Generate Response
- Uses retrieved context to generate final answer
- Ends the workflow

## Key Features

- **Conditional Routing**: Dynamically routes based on query type and context availability
- **Self-Improving**: Automatically expands knowledge base when needed
- **Context-Aware**: Only fetches external data when necessary
- **Efficient**: Reuses existing knowledge from vector database
