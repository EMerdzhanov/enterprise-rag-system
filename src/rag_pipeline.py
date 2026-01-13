"""
Main RAG Pipeline orchestration module.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .document_loader import DocumentLoader
from .retriever import Retriever
from .llm_client import LLMClient


class RAGPipeline:
    """
    End-to-end RAG pipeline for document Q&A.

    This pipeline handles document ingestion, embedding generation,
    vector storage, retrieval, and LLM-based response generation.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "claude-3-sonnet-20240229",
        top_k: int = 3,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            embedding_model: Sentence transformer model name.
            llm_model: Claude model name.
            top_k: Number of documents to retrieve.
            collection_name: ChromaDB collection name.
            persist_directory: Directory to persist vector store.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize components
        self.document_loader = DocumentLoader(chunk_size, chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = VectorStore(collection_name, persist_directory)
        self.retriever = Retriever(
            self.embedding_generator,
            self.vector_store,
            top_k=top_k
        )
        self.llm_client = LLMClient(model=llm_model)

        self._document_count = 0
        self._chunk_count = 0

    def ingest_documents(self, source: str) -> Dict[str, int]:
        """
        Ingest documents from a file or directory.

        Args:
            source: Path to file or directory.

        Returns:
            Dictionary with ingestion statistics.
        """
        path = Path(source)

        if path.is_file():
            documents = [self.document_loader.load_file(str(path))]
        elif path.is_dir():
            documents = self.document_loader.load_directory(str(path))
        else:
            raise ValueError(f"Invalid source: {source}")

        # Collect all chunks with metadata
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                all_chunks.append(chunk)
                all_metadatas.append({
                    **doc["metadata"],
                    "chunk_index": chunk_idx
                })
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

        if all_chunks:
            # Generate embeddings
            embeddings = self.embedding_generator.generate(all_chunks)

            # Store in vector database
            self.vector_store.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

        self._document_count += len(documents)
        self._chunk_count += len(all_chunks)

        return {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "total_documents": self._document_count,
            "total_chunks": self._chunk_count
        }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User's question.
            top_k: Override default top_k.
            return_sources: Whether to return source documents.

        Returns:
            Dictionary with 'answer' and optionally 'sources'.
        """
        # Retrieve relevant context
        k = top_k or self.top_k
        results = self.retriever.retrieve(question, top_k=k)

        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [] if return_sources else None
            }

        # Build context from retrieved documents
        context = "\n\n---\n\n".join([r["document"] for r in results])

        # Generate response
        answer = self.llm_client.generate(question, context)

        response = {"answer": answer}

        if return_sources:
            response["sources"] = [
                {
                    "content": r["document"][:200] + "..." if len(r["document"]) > 200 else r["document"],
                    "score": r["score"],
                    "metadata": r["metadata"]
                }
                for r in results
            ]

        return response

    def clear(self) -> None:
        """Clear all documents from the pipeline."""
        self.vector_store.clear()
        self._document_count = 0
        self._chunk_count = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Return pipeline statistics."""
        return {
            "documents": self._document_count,
            "chunks": self._chunk_count,
            "vectors": self.vector_store.count()
        }


if __name__ == "__main__":
    print("RAG Pipeline module loaded successfully")
    print("Initialize with: pipeline = RAGPipeline()")
