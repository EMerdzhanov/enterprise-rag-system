#!/usr/bin/env python3
"""
Enterprise RAG System Demo

This script demonstrates the full RAG pipeline:
1. Loading sample documents
2. Generating embeddings
3. Storing in vector database
4. Querying with natural language
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline


def print_header():
    """Print demo header."""
    print("=" * 60)
    print("         ENTERPRISE RAG SYSTEM - DEMO")
    print("=" * 60)
    print()


def print_section(number: int, title: str):
    """Print section header."""
    print(f"[{number}] {title}")


def run_demo():
    """Run the RAG system demo."""
    print_header()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your API key: export ANTHROPIC_API_KEY=your_key")
        return

    # Initialize pipeline
    print_section(1, "Initializing RAG Pipeline...")
    start_time = time.time()

    pipeline = RAGPipeline(
        chunk_size=500,
        chunk_overlap=50,
        top_k=3
    )

    print(f"    ✓ Pipeline initialized in {time.time() - start_time:.2f}s")
    print()

    # Load documents
    print_section(2, "Loading documents from data/sample_docs/...")
    docs_path = Path(__file__).parent.parent / "data" / "sample_docs"

    if not docs_path.exists() or not any(docs_path.iterdir()):
        print("    Creating sample documents...")
        create_sample_documents(docs_path)

    start_time = time.time()
    stats = pipeline.ingest_documents(str(docs_path))
    print(f"    ✓ Loaded {stats['documents_processed']} documents ({stats['chunks_created']} chunks)")
    print(f"    ✓ Processing time: {time.time() - start_time:.2f}s")
    print()

    # Run sample queries
    print_section(3, "Running sample queries...")
    print()

    sample_queries = [
        "What are the main AI trends for 2024?",
        "How does RAG architecture work?",
        "What are the benefits of vector databases?"
    ]

    for query in sample_queries:
        print(f"Query: \"{query}\"")
        print("-" * 50)

        start_time = time.time()
        result = pipeline.query(query, return_sources=True)

        print(f"\nAnswer:")
        print(result["answer"])

        if result.get("sources"):
            print(f"\nSources ({len(result['sources'])} retrieved):")
            for i, source in enumerate(result["sources"], 1):
                print(f"  [{i}] Score: {source['score']} - {source['metadata'].get('source', 'unknown')}")

        print(f"\nResponse time: {time.time() - start_time:.2f}s")
        print("=" * 60)
        print()

    print("Demo completed successfully!")


def create_sample_documents(docs_path: Path):
    """Create sample documents for demo."""
    docs_path.mkdir(parents=True, exist_ok=True)

    # Sample document 1: AI Trends
    ai_trends = """# AI Trends for 2024

## Executive Summary

The artificial intelligence landscape continues to evolve rapidly in 2024.
This report outlines the key trends shaping the industry.

## Key Trends

### 1. Agentic AI Systems

Autonomous AI agents capable of multi-step reasoning and task execution
are becoming mainstream. These systems can plan, execute, and adapt to
achieve complex goals without constant human intervention.

### 2. RAG Architecture Adoption

Retrieval-Augmented Generation (RAG) has become the standard approach for
enterprise AI applications. By combining retrieval systems with large
language models, organizations can build accurate, grounded AI systems.

### 3. Multimodal AI

Models that can process and generate multiple types of content - text,
images, audio, and video - are seeing widespread adoption in creative
and analytical applications.

### 4. AI Safety and Governance

Organizations are investing heavily in AI safety measures, including
red teaming, alignment research, and governance frameworks.

## Conclusion

2024 marks a pivotal year for AI adoption, with enterprise-grade
solutions becoming accessible to organizations of all sizes.
"""

    # Sample document 2: RAG Architecture
    rag_doc = """# Understanding RAG Architecture

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture that enhances
large language models by providing them with relevant context from a
knowledge base before generating responses.

## How RAG Works

1. **Document Ingestion**: Documents are loaded and split into chunks
2. **Embedding Generation**: Each chunk is converted to a vector embedding
3. **Vector Storage**: Embeddings are stored in a vector database
4. **Retrieval**: When a query arrives, similar documents are retrieved
5. **Generation**: The LLM generates a response using retrieved context

## Benefits of RAG

- **Accuracy**: Responses are grounded in actual documents
- **Up-to-date**: Knowledge can be updated without retraining
- **Transparent**: Sources can be cited for verification
- **Cost-effective**: No need for expensive model fine-tuning

## Best Practices

- Use appropriate chunk sizes (300-500 tokens)
- Implement chunk overlap for context preservation
- Choose embedding models suited to your domain
- Regularly evaluate retrieval quality
"""

    # Sample document 3: Vector Databases
    vector_db_doc = """# Vector Databases for AI Applications

## Introduction

Vector databases are specialized storage systems designed to efficiently
store and search high-dimensional vectors (embeddings).

## Popular Vector Databases

### ChromaDB
- Open-source and easy to use
- Great for prototyping and small-scale applications
- Supports persistent and in-memory storage

### Pinecone
- Fully managed cloud service
- Excellent scalability
- Enterprise-grade security

### Weaviate
- Open-source with cloud options
- Supports hybrid search (vector + keyword)
- GraphQL API

## Key Features

1. **Similarity Search**: Find vectors close to a query vector
2. **Filtering**: Combine vector search with metadata filters
3. **Scalability**: Handle millions of vectors efficiently
4. **Real-time Updates**: Add and remove vectors dynamically

## Use Cases

- Semantic search
- Recommendation systems
- Image similarity
- Anomaly detection
- Question answering (RAG)
"""

    # Write sample documents
    (docs_path / "ai_trends_2024.md").write_text(ai_trends)
    (docs_path / "rag_architecture.md").write_text(rag_doc)
    (docs_path / "vector_databases.md").write_text(vector_db_doc)

    print(f"    ✓ Created 3 sample documents")


if __name__ == "__main__":
    run_demo()
