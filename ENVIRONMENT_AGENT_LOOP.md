# Environment-Agent-Environment Loop Documentation

## Overview

This document explicitly maps the environment-agent-environment loop in the Game Theory RAG system, showing how percepts (observations) trigger actions that modify the environment, which then produces new percepts.

## Loop Structure

The agent operates in a sequential, iterative loop:

```
[Environment] → [Percept] → [Agent Reasoning] → [Action] → [Environment Update] → [Percept] → ...
```

## Detailed Loop Mappings

### Loop 1: Vector Database Retrieval

**Percept (Sensor Reading):**
- **State Field**: `state["user_query"]`
- **Source**: User input
- **What Agent Perceives**: User's question about game theory
- **Node**: Workflow entry point

**Agent Reasoning:**
- **Node**: `pull_from_chroma`
- **Reasoning**: "I need to search my knowledge base for relevant information"
- **Decision**: Query vector database with user query

**Action (Tool Call):**
- **Tool**: `VectorDBManager.query(query, n_results=3)`
- **Input**: User query string
- **What It Does**: Searches vector database using semantic similarity
- **Node**: `pull_from_chroma`

**Environment Update:**
- **State Field**: `state["chroma_results"]`
- **What Changed**: 
  - If documents found: `chroma_results` contains documents, metadata, IDs, distances
  - If empty: `chroma_results["documents"]` is `[[]]`
- **Environment**: Vector database (read-only, but state reflects its contents)

**New Percept:**
- **State Field**: `state["chroma_results"]`
- **What Agent Perceives**: Either relevant documents found, or empty results
- **Next Node**: `check_relevance`

**Example State Transition:**
```python
# Before action
state = {
    "user_query": "What is Nash equilibrium?",
    "chroma_results": {},  # Empty
    ...
}

# After pull_from_chroma action
state = {
    "user_query": "What is Nash equilibrium?",
    "chroma_results": {
        "documents": [["Nash equilibrium is a concept in game theory...", ...]],
        "metadatas": [[{"title": "Game Theory Paper", ...}]],
        "ids": [["1406.2661v1_chunk_0", ...]],
        "distances": [[0.15, 0.23, ...]]
    },
    ...
}
```

---

### Loop 2: Context Relevance Evaluation

**Percept (Sensor Reading):**
- **State Field**: `state["chroma_results"]`
- **Source**: Output of previous loop (vector DB query)
- **What Agent Perceives**: Retrieved documents and their relevance scores
- **Node**: `check_relevance`

**Agent Reasoning:**
- **Node**: `check_relevance`
- **Reasoning**: "Are these retrieved documents sufficient to answer the user's question?"
- **Decision Logic**: Uses LLM to evaluate if context is relevant
- **Prompt**: "Is the provided context sufficient to answer the query?"

**Action (Tool Call):**
- **Tool**: `BaseChatModel` (LLM)
- **Input**: User query + retrieved context
- **What It Does**: Classifies whether context is relevant (yes/no)
- **Node**: `check_relevance`

**Environment Update:**
- **State Field**: `state["relevant_context"]`
- **What Changed**: 
  - `True`: Context is sufficient
  - `False`: Context is insufficient or missing
- **Environment**: Agent's internal state (decision/classification)

**New Percept:**
- **State Field**: `state["relevant_context"]`
- **What Agent Perceives**: Boolean flag indicating whether to proceed or search for more
- **Next Node**: Router function `route_after_relevance_check`

**Example State Transition:**
```python
# Before action
state = {
    "chroma_results": {"documents": [["..."], ...]},
    "relevant_context": False,  # Unknown
    ...
}

# After check_relevance action
state = {
    "chroma_results": {"documents": [["..."], ...]},  # Unchanged
    "relevant_context": True,  # LLM determined context is sufficient
    ...
}
```

---

### Loop 3: Arxiv Search (When Context Insufficient)

**Percept (Sensor Reading):**
- **State Field**: `state["relevant_context"]` = `False`
- **Source**: Output of relevance evaluation
- **What Agent Perceives**: "I don't have enough information, need to search external source"
- **Node**: Router `route_after_relevance_check`

**Agent Reasoning:**
- **Node**: Router function
- **Reasoning**: "Context is not relevant, I should search Arxiv for more papers"
- **Decision**: Route to `search_arxiv` node

**Action (Tool Call):**
- **Tool**: `ArxivSearcher.search_papers(query)`
- **Input**: Search query (prefixed with "game theory")
- **What It Does**: Searches arxiv.org API for papers matching query
- **Node**: `search_arxiv`

**Environment Update:**
- **State Field**: `state["arxiv_papers"]`
- **What Changed**: List of paper dictionaries found from Arxiv
- **Environment**: Arxiv API (external, but results stored in state)
- **Also Updates**: `state["papers_seen"]` - tracks which papers have been processed

**New Percept:**
- **State Field**: `state["arxiv_papers"]`
- **What Agent Perceives**: List of papers found (may be empty, may contain non-game-theory papers)
- **Next Node**: `filter_game_theory_papers`

**Example State Transition:**
```python
# Before action
state = {
    "user_query": "What is Rocco's Basalisk?",
    "arxiv_papers": [],  # Empty
    "papers_seen": [],
    ...
}

# After search_arxiv action
state = {
    "user_query": "What is Rocco's Basalisk?",
    "arxiv_papers": [
        {
            "title": "Game Theory Paper",
            "summary": "...",
            "entry_id": "http://arxiv.org/abs/1406.2661v1",
            ...
        },
        ...
    ],
    "papers_seen": ["http://arxiv.org/abs/1406.2661v1", ...],  # Updated
    ...
}
```

---

### Loop 4: Paper Filtering

**Percept (Sensor Reading):**
- **State Field**: `state["arxiv_papers"]`
- **Source**: Output of Arxiv search
- **What Agent Perceives**: List of papers (may include non-game-theory papers)
- **Node**: `filter_game_theory_papers`

**Agent Reasoning:**
- **Node**: `filter_game_theory_papers`
- **Reasoning**: "Which of these papers are actually about game theory?"
- **Decision Logic**: Uses LLM to classify each paper
- **Prompt**: "Is this paper related to GAME THEORY (mathematical strategic decision-making)?"

**Action (Tool Call):**
- **Tool**: `BaseChatModel` (LLM)
- **Input**: Paper title and abstract for each paper
- **What It Does**: Classifies each paper as game theory or not
- **Node**: `filter_game_theory_papers`

**Environment Update:**
- **State Field**: `state["arxiv_papers"]` (filtered)
- **What Changed**: 
  - Papers list reduced to only game theory related papers
  - `state["papers_seen"]` updated with all processed papers
- **Environment**: Agent's internal state (filtered list)

**New Percept:**
- **State Field**: `state["arxiv_papers"]` (now filtered)
- **What Agent Perceives**: Only game theory papers (may be empty if none found)
- **Next Node**: Router `route_after_paper_filter`

**Example State Transition:**
```python
# Before action
state = {
    "arxiv_papers": [
        {"title": "Game Theory Paper", ...},  # Game theory
        {"title": "Video Game AI", ...},      # Not game theory
    ],
    "papers_seen": [],
    ...
}

# After filter_game_theory_papers action
state = {
    "arxiv_papers": [
        {"title": "Game Theory Paper", ...},  # Only game theory kept
    ],  # Video Game AI removed
    "papers_seen": [
        "http://arxiv.org/abs/1406.2661v1",  # Game theory paper
        "http://arxiv.org/abs/1234.5678v1",  # Video game paper (seen but not added)
    ],
    ...
}
```

---

### Loop 5: Knowledge Base Expansion

**Percept (Sensor Reading):**
- **State Field**: `state["arxiv_papers"]` (non-empty, filtered list)
- **Source**: Output of paper filtering
- **What Agent Perceives**: "I have new game theory papers to add to my knowledge base"
- **Node**: Router `route_after_paper_filter`

**Agent Reasoning:**
- **Node**: Router function
- **Reasoning**: "I found game theory papers, I should add them to my vector database"
- **Decision**: Route to `add_to_chroma` node

**Action (Tool Call):**
- **Tool 1**: `DocumentProcessor.process_paper(paper)` - Chunks each paper
- **Tool 2**: `VectorDBManager.add_documents(documents, metadatas, ids)` - Stores chunks
- **Input**: Paper dictionaries from `state["arxiv_papers"]`
- **What It Does**: 
  1. Processes papers into chunks with metadata
  2. **Modifies environment** by adding documents to vector database
- **Node**: `add_to_chroma`

**Environment Update:**
- **State Field**: `state["papers_added"]` = `True`
- **What Changed**: 
  - **Vector Database**: **Actually modified** - new document chunks added to persistent storage
  - Database count increases (e.g., from 10 to 15 documents)
- **Environment**: Vector database (persistent environment state changed)

**New Percept:**
- **State Field**: `state["papers_added"]` = `True`
- **What Agent Perceives**: "I've successfully added papers to my knowledge base"
- **Next Node**: Loop back to `pull_from_chroma` (re-query with expanded knowledge)

**Example State Transition:**
```python
# Before action
state = {
    "arxiv_papers": [{"title": "Game Theory Paper", ...}],
    "papers_added": False,
    ...
}
# Vector DB count: 10 documents

# After add_to_chroma action
state = {
    "arxiv_papers": [{"title": "Game Theory Paper", ...}],  # Unchanged
    "papers_added": True,
    ...
}
# Vector DB count: 15 documents (5 new chunks added)
```

---

### Loop 6: Re-Query with Expanded Knowledge

**Percept (Sensor Reading):**
- **State Field**: `state["papers_added"]` = `True`
- **Source**: Output of knowledge base expansion
- **What Agent Perceives**: "I've added new papers, I should re-query my knowledge base"
- **Node**: Edge from `add_to_chroma` to `pull_from_chroma`

**Agent Reasoning:**
- **Node**: `pull_from_chroma` (called again)
- **Reasoning**: "My knowledge base has been updated, let me search again with the new information"
- **Decision**: Query vector database again (same query, but database now has more documents)

**Action (Tool Call):**
- **Tool**: `VectorDBManager.query(query, n_results=3)` (same as Loop 1, but on updated database)
- **Input**: Same user query
- **What It Does**: Searches vector database again (now includes newly added papers)
- **Node**: `pull_from_chroma`

**Environment Update:**
- **State Field**: `state["chroma_results"]` (updated)
- **What Changed**: 
  - May now include documents from newly added papers
  - Results may be more relevant or more comprehensive
- **Environment**: Vector database (same database, but now contains more documents)

**New Percept:**
- **State Field**: `state["chroma_results"]` (updated with new results)
- **What Agent Perceives**: Potentially better results from expanded knowledge base
- **Next Node**: Continue to `check_relevance` (Loop 2)

**This creates the self-improving loop:**
```
[Add Papers] → [Re-Query] → [Better Results] → [Generate Response]
```

---

### Loop 7: Response Generation

**Percept (Sensor Reading):**
- **State Field**: `state["chroma_results"]` (with relevant documents)
- **State Field**: `state["relevant_context"]` = `True`
- **Source**: Output of relevance evaluation (Loop 2)
- **What Agent Perceives**: "I have sufficient context to answer the question"
- **Node**: Router `route_after_relevance_check`

**Agent Reasoning:**
- **Node**: Router function
- **Reasoning**: "Context is relevant, I can generate a response now"
- **Decision**: Route to `generate_response` node

**Action (Tool Call):**
- **Tool**: `BaseChatModel` (LLM)
- **Input**: User query + retrieved context documents
- **What It Does**: Generates final answer using retrieved context
- **Node**: `generate_response`

**Environment Update:**
- **State Field**: `state["final_response"]`
- **What Changed**: Contains the generated answer string
- **Environment**: Agent's output (final result)

**New Percept:**
- **State Field**: `state["final_response"]`
- **What Agent Perceives**: Final answer ready for user
- **Next Node**: END (workflow terminates)

**Example State Transition:**
```python
# Before action
state = {
    "user_query": "What is Nash equilibrium?",
    "chroma_results": {
        "documents": [["Nash equilibrium is a concept...", ...]],
        ...
    },
    "final_response": "",  # Empty
    ...
}

# After generate_response action
state = {
    "user_query": "What is Nash equilibrium?",
    "chroma_results": {...},  # Unchanged
    "final_response": "Nash equilibrium is a concept in game theory where...",  # Generated
    ...
}
```

---

## Complete Loop Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT-AGENT LOOP                        │
└─────────────────────────────────────────────────────────────────┘

1. INITIAL PERCEPT
   User Query → state["user_query"]
                    ↓
2. ACTION: Query Vector DB
   pull_from_chroma → VectorDBManager.query()
                    ↓
3. ENVIRONMENT UPDATE: Results Retrieved
   state["chroma_results"] populated
                    ↓
4. PERCEPT: Check Results
   state["chroma_results"] → Agent sees documents (or empty)
                    ↓
5. REASONING: Evaluate Relevance
   check_relevance → LLM classifies context
                    ↓
6. DECISION: Relevant?
   ├─ YES → Loop 7 (Generate Response) → END
   └─ NO → Continue to Loop 3
                    ↓
7. ACTION: Search Arxiv
   search_arxiv → ArxivSearcher.search_papers()
                    ↓
8. ENVIRONMENT UPDATE: Papers Found
   state["arxiv_papers"] populated
                    ↓
9. PERCEPT: Papers Available
   state["arxiv_papers"] → Agent sees papers
                    ↓
10. REASONING: Filter Papers
    filter_game_theory_papers → LLM classifies each paper
                    ↓
11. ENVIRONMENT UPDATE: Filtered List
    state["arxiv_papers"] filtered (only game theory)
                    ↓
12. ACTION: Add to Knowledge Base
    add_to_chroma → VectorDBManager.add_documents()
                    ↓
13. ENVIRONMENT UPDATE: Vector DB Modified
    Vector database grows (persistent change)
    state["papers_added"] = True
                    ↓
14. LOOP BACK: Re-Query with Expanded Knowledge
    → Returns to Step 2 (pull_from_chroma)
                    ↓
15. NEW PERCEPT: Better Results
    state["chroma_results"] (now includes new papers)
                    ↓
16. REASONING: Evaluate Again
    check_relevance → Context now relevant?
                    ↓
17. DECISION: Generate Response
    generate_response → LLM creates answer
                    ↓
18. FINAL OUTPUT
    state["final_response"] → User receives answer
```

## Key Observations

1. **Environment Modifications**: Only one action actually modifies persistent environment:
   - `VectorDBManager.add_documents()` - Adds documents to vector database
   - All other actions are read-only (queries, searches, classifications)

2. **State as Percept Container**: All percepts flow through `GraphState`:
   - Environment observations stored in state fields
   - Agent reads state to perceive environment
   - Agent writes to state to record decisions/outcomes

3. **Iterative Improvement**: The loop back mechanism (add papers → re-query) creates a self-improving system:
   - Knowledge base grows over time
   - Subsequent queries benefit from expanded knowledge
   - System gets better at answering questions as it processes more queries

4. **Loop Prevention**: `papers_seen` tracking prevents infinite loops:
   - Papers already processed are filtered out
   - Prevents re-adding the same papers repeatedly
   - Ensures workflow terminates

5. **Multiple Environment Types**: Agent interacts with:
   - **Persistent Environment**: Vector database (changes persist)
   - **External Environment**: Arxiv API (read-only, external state)
   - **Internal Environment**: Agent's state (decisions, classifications)

## State Flow Summary

```
Initial State:
  user_query, chroma_results={}, relevant_context=False, arxiv_papers=[], ...

Percept → Action → Outcome → Percept → Action → ...

Loop 1: chroma_results (empty) → search_arxiv → arxiv_papers (populated)
Loop 2: arxiv_papers → filter → arxiv_papers (filtered)
Loop 3: arxiv_papers → add_to_chroma → papers_added=True, Vector DB grows
Loop 4: papers_added → pull_from_chroma → chroma_results (updated with new docs)
Loop 5: chroma_results (relevant) → check_relevance → relevant_context=True
Loop 6: relevant_context → generate_response → final_response (answer)
```

This demonstrates a clear environment-agent-environment loop where:
- **Percepts** (state observations) trigger **reasoning**
- **Reasoning** leads to **actions** (tool calls)
- **Actions** modify or query the **environment**
- **Environment changes** produce new **percepts**
- The cycle repeats until the goal is achieved

