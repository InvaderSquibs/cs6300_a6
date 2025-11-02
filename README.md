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
