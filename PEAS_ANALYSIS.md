# PEAS Analysis: Game Theory RAG Agent

## Overview

This document provides a formal PEAS (Performance, Environment, Actuators, Sensors) analysis for the Game Theory RAG agent system. This framework helps clarify the agent's design, capabilities, and operating context.

## Performance Measure

The agent's performance is measured by:

1. **Answer Quality**
   - Relevance of responses to user queries
   - Accuracy of game theory information provided
   - Completeness of answers (covers query scope)
   - Use of retrieved context vs. hallucination

2. **Knowledge Base Growth**
   - Number of game theory papers successfully added to vector database
   - Diversity of papers added (different topics, authors, time periods)
   - Vector database document count growth over time

3. **Response Relevance**
   - Whether retrieved context is actually relevant to the query
   - Success rate of context retrieval vs. needing external search
   - Efficiency of workflow (fewer iterations = better)

4. **System Efficiency**
   - Query processing time
   - Number of tool calls required per query
   - Success rate of vector DB retrieval vs. Arxiv search fallback

**Success Criteria:**
- Agent provides accurate, relevant answers to game theory questions
- Vector database grows with relevant papers over time
- System avoids infinite loops and handles edge cases gracefully
- Retrieved context is used effectively in responses

## Environment

The agent operates in a multi-component environment:

### Environment Components

1. **Vector Database (ChromaDB)**
   - **Type**: Persistent, local vector database
   - **State**: Collection of document chunks with embeddings
   - **Observability**: Fully observable (agent can query all contents)
   - **Determinism**: Deterministic (same query returns same results)
   - **Dynamics**: Changes when new documents are added by the agent
   - **Location**: `./chroma_db/` directory on local filesystem

2. **Arxiv API**
   - **Type**: External API service (arxiv.org)
   - **State**: Repository of academic papers (changing over time)
   - **Observability**: Partially observable (agent can search, but doesn't know all papers)
   - **Determinism**: Stochastic (search results may vary, new papers added over time)
   - **Dynamics**: External system, changes independently of agent
   - **Access**: HTTP API calls to arxiv.org

3. **User Query Space**
   - **Type**: Input space of possible questions
   - **State**: Current query string
   - **Observability**: Fully observable (agent receives complete query)
   - **Determinism**: Deterministic (same query = same input)
   - **Dynamics**: Static during workflow execution, changes per query

### Environment Characteristics

- **Observability**: Mixed
  - Vector DB: Fully observable
  - Arxiv API: Partially observable (search results only)
  - User queries: Fully observable

- **Determinism**: Mixed
  - Vector DB: Deterministic
  - Arxiv API: Stochastic (new papers, varying results)
  - LLM responses: Stochastic (temperature-based variation)

- **Episodic vs. Sequential**: Sequential
  - Agent maintains state across workflow steps
  - Vector database persists across queries
  - Knowledge base grows over time

- **Static vs. Dynamic**: Dynamic
  - Vector database changes as agent adds papers
  - Arxiv content changes externally
  - Agent's knowledge base evolves

## Actuators

Actuators are the tools/actions the agent can use to affect the environment:

### 1. Vector Database Query Tool (`VectorDBManager.query`)
- **Purpose**: Retrieve relevant documents from vector database
- **Input**: Query string (user's question)
- **Output**: Dictionary with documents, metadata, IDs, and similarity scores
- **Effect**: Pure read operation (doesn't modify environment, but provides percepts)
- **Location**: `src/vector_db.py::VectorDBManager.query()`

### 2. Vector Database Add Tool (`VectorDBManager.add_documents`)
- **Purpose**: Add new document chunks to vector database
- **Input**: List of documents, metadata dictionaries, and IDs
- **Output**: None (void operation)
- **Effect**: **Modifies environment** - adds documents to persistent storage
- **Location**: `src/vector_db.py::VectorDBManager.add_documents()`

### 3. Arxiv Search Tool (`ArxivSearcher.search_papers`)
- **Purpose**: Search arxiv.org for academic papers
- **Input**: Search query string
- **Output**: List of paper dictionaries (title, summary, authors, pdf_url, etc.)
- **Effect**: Pure read operation (queries external API, doesn't modify local environment)
- **Location**: `src/arxiv_search.py::ArxivSearcher.search_papers()`

### 4. Document Processor Tool (`DocumentProcessor.process_paper`)
- **Purpose**: Chunk paper text and extract metadata
- **Input**: Paper dictionary (title, summary, authors, etc.)
- **Output**: List of chunk dictionaries with text and metadata
- **Effect**: Pure transformation (processes data, doesn't modify environment directly)
- **Location**: `src/document_processor.py::DocumentProcessor.process_paper()`

### 5. LLM Reasoning Tool (`BaseChatModel`)
- **Purpose**: Evaluate context relevance and generate responses
- **Input**: Prompt strings (queries, context)
- **Output**: Text responses (yes/no for classification, full answers for generation)
- **Effect**: Pure computation (doesn't modify environment, but influences agent decisions)
- **Location**: Used in `check_relevance`, `filter_game_theory_papers`, `generate_response` nodes

### Actuator Categories

**Environment-Modifying Actuators:**
- `VectorDBManager.add_documents()` - Only actuator that modifies persistent environment state

**Information-Gathering Actuators:**
- `VectorDBManager.query()` - Observes vector database state
- `ArxivSearcher.search_papers()` - Observes external Arxiv state

**Processing Actuators:**
- `DocumentProcessor.process_paper()` - Transforms data for storage
- `BaseChatModel` (LLM) - Performs reasoning and generation

## Sensors

Sensors are the mechanisms by which the agent perceives the environment. In this system, sensors are implemented through state observations in the `GraphState` TypedDict.

### Sensor Mappings

1. **Vector Database Sensor**
   - **State Field**: `chroma_results`
   - **Type**: Dictionary with documents, metadata, IDs, distances
   - **Source**: Output of `VectorDBManager.query()`
   - **What it Perceives**: 
     - Whether relevant documents exist in vector database
     - Content of retrieved document chunks
     - Metadata about document sources
     - Similarity scores (relevance metrics)
   - **Node**: `pull_from_chroma`
   - **Example Percept**: 
     ```python
     {
         "documents": [["Nash equilibrium is a concept...", "..."]],
         "metadatas": [[{"title": "Game Theory Paper", "source": "arxiv"}], ...],
         "ids": [["1406.2661v1_chunk_0", ...]],
         "distances": [[0.15, 0.23, ...]]
     }
     ```

2. **Arxiv Search Sensor**
   - **State Field**: `arxiv_papers`
   - **Type**: List of paper dictionaries
   - **Source**: Output of `ArxivSearcher.search_papers()`
   - **What it Perceives**:
     - Available papers matching search query
     - Paper titles and abstracts
     - Author information
     - PDF URLs for download
     - Publication dates
   - **Node**: `search_arxiv`
   - **Example Percept**:
     ```python
     [
         {
             "title": "Nash Equilibrium in Game Theory",
             "summary": "This paper discusses...",
             "authors": ["Author One", "Author Two"],
             "pdf_url": "http://arxiv.org/pdf/1406.2661v1.pdf",
             "entry_id": "http://arxiv.org/abs/1406.2661v1",
             "published": "2023-06-15"
         },
         ...
     ]
     ```

3. **Relevance Evaluation Sensor**
   - **State Field**: `relevant_context`
   - **Type**: Boolean flag
   - **Source**: Output of LLM reasoning in `check_relevance` node
   - **What it Perceives**:
     - Whether retrieved context is sufficient to answer the query
     - Quality/relevance of vector DB results
   - **Node**: `check_relevance`
   - **Example Percept**: `True` (context is relevant) or `False` (need more information)

4. **Paper Filtering Sensor**
   - **State Field**: `arxiv_papers` (filtered)
   - **Type**: List of paper dictionaries (subset)
   - **Source**: Output of LLM classification in `filter_game_theory_papers` node
   - **What it Perceives**:
     - Which papers from Arxiv are actually game theory related
     - Quality of paper matches for the domain
   - **Node**: `filter_game_theory_papers`
   - **Example Percept**: Filtered list containing only game theory papers

5. **Loop Prevention Sensor**
   - **State Field**: `papers_seen`
   - **Type**: List of paper entry_ids (strings)
   - **Source**: Accumulated across workflow execution
   - **What it Perceives**:
     - Which papers have already been processed
     - Prevents infinite loops by tracking seen papers
   - **Nodes**: `search_arxiv`, `filter_game_theory_papers`
   - **Example Percept**: `["http://arxiv.org/abs/1406.2661v1", ...]`

6. **User Query Sensor**
   - **State Field**: `user_query`
   - **Type**: String
   - **Source**: Direct input from user
   - **What it Perceives**:
     - The user's question or request
     - Initial goal for the agent
   - **Node**: Set at workflow start
   - **Example Percept**: `"What is Nash equilibrium?"`

### Sensor Characteristics

- **Observability**: All sensors are fully observable within the agent's state
- **Update Frequency**: Sensors update at each node execution
- **Persistence**: Some percepts persist across workflow (e.g., `papers_seen`), others are transient (e.g., `chroma_results`)

## Agent Capabilities and Limitations

### Capabilities

1. **Semantic Search**: Can find relevant documents using vector similarity
2. **External Knowledge Retrieval**: Can search Arxiv for new papers
3. **Context Evaluation**: Can assess whether retrieved context is sufficient
4. **Knowledge Base Expansion**: Can add new papers to vector database
5. **Iterative Reasoning**: Can loop back and re-query after adding papers
6. **Domain Filtering**: Can filter papers to ensure game theory relevance

### Limitations

1. **No Direct PDF Processing**: Can download PDFs but doesn't extract full text (only abstracts)
2. **Limited Context Window**: Only uses top 3 documents from vector DB
3. **No Multi-turn Conversation**: Each query is independent (no conversation history)
4. **No Real-time Updates**: Arxiv search reflects current state, but agent doesn't monitor for new papers
5. **Fixed Domain**: Specifically designed for game theory, not general-purpose
6. **No Query Refinement**: Uses user query directly, doesn't refine or expand it
7. **LLM Dependency**: Relies on LLM for classification and generation (subject to LLM limitations)

## Environment-Agent Interaction Summary

```
[Environment: Vector DB] 
    ↓ (perceived via chroma_results sensor)
[Agent: pull_from_chroma node]
    ↓ (reads state, queries tool)
[Environment: Vector DB returns results]
    ↓ (chroma_results updated in state)
[Agent: check_relevance node]
    ↓ (reads chroma_results, uses LLM)
[Agent Decision: relevant_context = True/False]
    ↓ (if False, agent acts on Arxiv environment)
[Environment: Arxiv API]
    ↓ (perceived via arxiv_papers sensor)
[Agent: search_arxiv node]
    ↓ (reads state, queries Arxiv)
[Environment: Arxiv returns papers]
    ↓ (arxiv_papers updated in state)
[Agent: filter_game_theory_papers node]
    ↓ (filters papers using LLM)
[Agent: add_to_chroma node]
    ↓ (modifies Vector DB environment)
[Environment: Vector DB updated with new documents]
    ↓ (environment state changed)
[Agent: Loops back to pull_from_chroma]
    ↓ (perceives updated environment)
```

This PEAS analysis demonstrates that the agent operates in a sequential, partially observable, dynamic environment with clear actuators for information gathering and environment modification, and sensors that provide percepts through the state dictionary.

