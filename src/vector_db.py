"""
Vector database manager using ChromaDB for storing and retrieving documents.

This module provides a VectorDBManager class that encapsulates all interactions
with ChromaDB, a vector database optimized for semantic search. The manager
handles document storage, retrieval, and persistence.

ChromaDB Integration:
    The module uses ChromaDB's PersistentClient for persistent storage. Documents
    are stored with their embeddings (computed automatically by ChromaDB), metadata,
    and unique IDs. The database persists to disk, allowing the knowledge base to
    grow and persist across workflow executions.

Persistence Strategy:
    - Database is stored in a configurable directory (default: ./chroma_db)
    - All documents are persisted to disk automatically
    - The same collection is reused across instances pointing to the same directory
    - This enables the workflow to build a persistent knowledge base over time

Collection Management:
    - Collection name: "game_theory_docs"
    - Collection is created automatically on first use
    - Metadata includes description for collection tracking
    - Anonymized telemetry is disabled for privacy

Vector Search:
    - Uses semantic similarity based on document embeddings
    - Query results are sorted by relevance (similarity score)
    - Results include documents, metadata, IDs, and similarity distances
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os


class VectorDBManager:
    """
    Manages interactions with ChromaDB vector database.
    
    This class provides a high-level interface for storing and retrieving
    documents in ChromaDB. It handles collection management, document
    validation, and query operations.
    
    The manager uses ChromaDB's persistent client, which automatically:
    - Computes embeddings for documents
    - Stores embeddings, documents, and metadata
    - Persists data to disk
    - Enables semantic search queries
    
    Example:
        .. code-block:: python
        
            # Initialize with default directory
            db = VectorDBManager()
            
            # Add documents
            db.add_documents(
                documents=["Document text 1", "Document text 2"],
                metadatas=[{"source": "paper1"}, {"source": "paper2"}],
                ids=["doc1", "doc2"]
            )
            
            # Query for relevant documents
            results = db.query("game theory", n_results=2)
            
            # Check document count
            count = db.count()
    
    Attributes:
        persist_directory: Path to directory where database is stored
        client: ChromaDB PersistentClient instance
        collection: ChromaDB collection for game theory documents
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client with persistence.
        
        Creates or connects to a persistent ChromaDB database at the specified
        directory. The collection "game_theory_docs" is created if it doesn't
        exist, or connected to if it does.
        
        Args:
            persist_directory: Path to directory for database storage.
                Directory is created if it doesn't exist. Default is "./chroma_db".
                All database files (SQLite, embeddings, etc.) are stored here.
        
        Note:
            - Directory is created automatically if it doesn't exist
            - Multiple instances pointing to the same directory share the same database
            - Anonymized telemetry is disabled for privacy
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for game theory documents
        self.collection = self.client.get_or_create_collection(
            name="game_theory_docs",
            metadata={"description": "Game theory papers and documents"}
        )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents to the vector database.
        
        Stores documents along with their metadata and unique IDs in ChromaDB.
        ChromaDB automatically computes embeddings for each document to enable
        semantic search. All lists must have the same length.
        
        Args:
            documents: List of document text strings to store. Each string
                represents a document chunk or full document.
            metadatas: List of metadata dictionaries, one per document. Each dict
                can contain arbitrary key-value pairs for filtering or tracking.
                Common keys: source, title, authors, published, chunk_index.
            ids: List of unique identifier strings, one per document. IDs must be
                unique across the collection. Format is typically:
                "{source_id}_chunk_{index}" for chunked documents.
        
        Raises:
            ValueError: If the three input lists have different lengths.
                All lists must have exactly the same number of elements.
        
        Note:
            - ChromaDB automatically computes embeddings using a default embedder
            - Documents are immediately persisted to disk
            - If an ID already exists, the document is updated (not duplicated)
            - Metadata can be used for filtering queries later
        """
        # Validate that all lists have the same length
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError(
                f"All input lists must have the same length. "
                f"Got documents={len(documents)}, metadatas={len(metadatas)}, ids={len(ids)}"
            )
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the vector database for relevant documents.
        
        Performs semantic search on the stored documents using the query text.
        Results are ranked by similarity to the query and returned with their
        metadata, IDs, and similarity scores.
        
        Args:
            query_text: Search query string. The query is embedded and compared
                against all stored document embeddings using cosine similarity.
            n_results: Number of top results to return. Default is 3. Results are
                sorted by similarity (most similar first).
        
        Returns:
            Dictionary containing query results with structure:
            {
                "ids": [[str, ...]],           # List of document IDs (nested list format)
                "documents": [[str, ...]],     # List of document texts (nested list format)
                "metadatas": [[Dict, ...]],    # List of metadata dicts (nested list format)
                "distances": [[float, ...]]    # List of similarity distances (nested list format)
            }
            
            Note: Results are in nested list format because ChromaDB supports
            multiple queries. For single queries, access with [0] index:
            results["documents"][0] gets the list of document strings.
        
        Example:
            .. code-block:: python
            
                results = db.query("Nash equilibrium", n_results=2)
                docs = results["documents"][0]  # Get top 2 documents
                ids = results["ids"][0]        # Get their IDs
                scores = results["distances"][0]  # Get similarity scores
        
        Note:
            - Lower distance values indicate higher similarity
            - If fewer than n_results documents exist, all available are returned
            - Empty list is returned if no documents match or database is empty
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    
    def count(self) -> int:
        """
        Return the number of documents in the collection.
        
        Returns:
            Integer count of documents currently stored in the collection.
        
        Example:
            .. code-block:: python
            
                count = db.count()
                print(f"Database contains {count} documents")
        
        Note:
            - Count includes all documents ever added (including updated ones)
            - Returns 0 if collection is empty
        """
        return self.collection.count()
