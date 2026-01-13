"""
Retriever module for semantic search over documents.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


class Retriever:
    """Retrieve relevant documents using semantic search."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k: int = 3,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize the retriever.

        Args:
            embedding_generator: Embedding generator instance.
            vector_store: Vector store instance.
            top_k: Number of results to retrieve.
            similarity_threshold: Minimum similarity score (0-1).
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Override default top_k.
            filter_metadata: Optional metadata filter.

        Returns:
            List of results with 'document', 'metadata', 'score', and 'id'.
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"]):
            # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
            distance = results["distances"][i] if results["distances"] else 0
            similarity = 1 - distance

            if similarity >= self.similarity_threshold:
                formatted_results.append({
                    "document": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "score": round(similarity, 4),
                    "id": results["ids"][i] if results["ids"] else f"doc_{i}"
                })

        return formatted_results

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Retrieve and format as a single context string.

        Args:
            query: Search query.
            top_k: Override default top_k.
            separator: String to join documents.

        Returns:
            Concatenated context string.
        """
        results = self.retrieve(query, top_k)
        documents = [r["document"] for r in results]
        return separator.join(documents)


if __name__ == "__main__":
    # Quick test
    print("Retriever module loaded successfully")
