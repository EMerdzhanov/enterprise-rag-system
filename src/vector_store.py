"""
Vector store module using ChromaDB for efficient similarity search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collection_name = collection_name

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents and their embeddings to the store.

        Args:
            embeddings: numpy array of embeddings.
            documents: List of document texts.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of document IDs.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional filter conditions.

        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', and 'ids'.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    # Quick test
    store = VectorStore()
    test_embeddings = np.random.rand(3, 384).astype(np.float32)
    test_docs = ["Doc 1", "Doc 2", "Doc 3"]
    store.add(test_embeddings, test_docs)
    print(f"Store contains {store.count()} documents")
