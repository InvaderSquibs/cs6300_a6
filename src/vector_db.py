"""
Vector database manager using ChromaDB for storing and retrieving documents.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os


class VectorDBManager:
    """Manages interactions with ChromaDB vector database."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistence."""
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
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add documents to the vector database."""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the vector database for relevant documents."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
