"""
Document processing and storage nodes.

This module contains nodes that process raw data (papers, documents) into
a format suitable for storage and retrieval. These nodes coordinate between
document processors and vector databases to expand the knowledge base.

Processing nodes typically use multiple tools (e.g., both document processor
and vector DB), making their dependencies explicit in function signatures.
"""
from src.state import GraphState

# Import for type hints (needed at runtime)
from src.vector_db import VectorDBManager
from src.document_processor import DocumentProcessor


def add_to_chroma(
    state: GraphState,
    vector_db: VectorDBManager,
    doc_processor: DocumentProcessor
) -> GraphState:
    """
    Process arxiv papers and add them to the vector database.
    
    This node takes papers found by the arxiv search and processes them
    for storage in ChromaDB. It:
    1. Chunks each paper (title + abstract) into processable pieces
    2. Extracts metadata from each chunk
    3. Adds chunks to the vector database with unique IDs
    4. Handles errors gracefully, continuing with remaining papers if one fails
    
    This node demonstrates multi-tool dependency: it requires both
    document_processor (for chunking) and vector_db (for storage).
    
    Args:
        state: Current workflow state containing:
            - arxiv_papers: List[Dict[str, Any]] - Papers found from arxiv
              search. Each dict contains: title, summary, authors, published,
              entry_id, pdf_url.
        vector_db: VectorDBManager instance for storing processed chunks.
            Should be the same instance used throughout the workflow to
            ensure persistence.
        doc_processor: DocumentProcessor instance configured with chunk_size
            and chunk_overlap. Used to break papers into searchable chunks
            while preserving metadata.
    
    Returns:
        Updated state with papers_added flag set:
            - papers_added: bool - Always True if node completes successfully
              (even if some papers failed to process)
    
    State Modifications:
        - Sets state["papers_added"] to True
        - Does NOT modify arxiv_papers (read-only from this node)
        - Vector database is modified (chunks added)
    
    Example State Transition:
        Input state:
            {"arxiv_papers": [
                {"title": "Game Theory Paper", "summary": "...", ...},
                ...
            ], ...}
        
        Output state:
            {"arxiv_papers": [...],  # Unchanged
             "papers_added": True, ...}
        
        Additionally, vector_db now contains new document chunks.
    
    Error Handling:
        - If processing a paper fails, logs error and continues with next paper
        - Node always sets papers_added=True even if some papers failed
        - This ensures workflow can continue even with partial failures
    
    Note:
        - Processes papers sequentially (one at a time)
        - Each paper may produce multiple chunks depending on abstract length
        - Chunk IDs are constructed as: "{entry_id}_chunk_{index}"
        - Metadata includes: title, authors, published date, source, chunk_index
        - Database count is logged before and after for verification
    """
    count_before = vector_db.count()
    print(f"Processing and adding papers to Chroma DB...")
    print(f"  DB count before adding: {count_before} documents")
    
    total_chunks_added = 0
    for paper in state["arxiv_papers"]:
        try:
            # Process paper into chunks using document processor
            chunks = doc_processor.process_paper(paper)
            
            # Extract data for Chroma (list of lists format)
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]
            
            # Add to vector DB using vector_db manager
            vector_db.add_documents(documents, metadatas, ids)
            total_chunks_added += len(chunks)
            
            print(f"  ✓ Added {len(chunks)} chunks from paper: {paper['title'][:50]}...")
        except Exception as e:
            print(f"  ✗ Error adding paper to Chroma: {e}")
            # Continue with next paper even if one fails
            continue
    
    count_after = vector_db.count()
    print(f"  DB count after adding: {count_after} documents")
    print(f"  Total chunks added in this run: {total_chunks_added}")
    print(f"  DB growth: {count_after - count_before} documents")
    
    state["papers_added"] = True
    return state

