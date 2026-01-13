# Enterprise RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude API](https://img.shields.io/badge/LLM-Claude%20API-orange.svg)](https://www.anthropic.com/)

A production-ready Retrieval-Augmented Generation (RAG) system designed for enterprise document Q&A. Built with Claude API, ChromaDB vector store, and sentence-transformers for embeddings.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Documents│───▶│ Text Chunker │───▶│ Embedding Generator │   │
│  └──────────┘    └──────────────┘    └──────────┬──────────┘   │
│                                                  │               │
│                                                  ▼               │
│                                        ┌─────────────────┐      │
│                                        │   ChromaDB      │      │
│                                        │  Vector Store   │      │
│                                        └────────┬────────┘      │
│                                                  │               │
│  ┌──────────┐    ┌──────────────┐    ┌─────────▼─────────┐     │
│  │  Query   │───▶│  Retriever   │───▶│ Context + Query   │     │
│  └──────────┘    └──────────────┘    └─────────┬─────────┘     │
│                                                  │               │
│                                                  ▼               │
│                                        ┌─────────────────┐      │
│                                        │   Claude LLM    │      │
│                                        │   Generation    │      │
│                                        └─────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Document Ingestion**: Support for PDF, TXT, and Markdown files
- **Smart Chunking**: Configurable chunk size with overlap for context preservation
- **Local Embeddings**: sentence-transformers for fast, free embedding generation
- **Vector Storage**: ChromaDB for efficient similarity search
- **LLM Integration**: Claude API for high-quality response generation
- **Configurable Retrieval**: Adjustable top-k and similarity thresholds

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EMerdzhanov/enterprise-rag-system.git
cd enterprise-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Add your Anthropic API key
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

### Run Demo

```bash
python examples/demo.py
```

## Usage

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Ingest documents
rag.ingest_documents("data/sample_docs/")

# Query the system
response = rag.query("What are the key findings in the quarterly report?")
print(response)
```

### Advanced Configuration

```python
from src.rag_pipeline import RAGPipeline

# Custom configuration
rag = RAGPipeline(
    chunk_size=500,
    chunk_overlap=50,
    top_k=5,
    embedding_model="all-MiniLM-L6-v2"
)
```

## Demo Output

```
============================================================
         ENTERPRISE RAG SYSTEM - DEMO
============================================================

[1] Loading documents from data/sample_docs/...
    ✓ Loaded 3 documents (15 chunks)

[2] Generating embeddings...
    ✓ Embeddings generated in 1.2s

[3] Storing in vector database...
    ✓ Stored 15 vectors in ChromaDB

[4] Running sample queries...

Query: "What are the main AI trends for 2024?"
─────────────────────────────────────────────
Retrieved 3 relevant chunks (similarity: 0.89, 0.85, 0.82)

Response:
Based on the documents, the main AI trends for 2024 include:

1. **Agentic AI Systems**: Autonomous AI agents capable of
   multi-step reasoning and task execution
2. **RAG Architecture Adoption**: Enterprise adoption of
   retrieval-augmented generation for knowledge management
3. **Multimodal Models**: Integration of text, image, and
   audio processing in unified models

Response time: 2.3s
============================================================
```

## Project Structure

```
enterprise-rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── embeddings.py      # Embedding generation
│   ├── vector_store.py    # ChromaDB operations
│   ├── retriever.py       # Similarity search
│   ├── llm_client.py      # Claude API client
│   ├── document_loader.py # Document ingestion
│   └── rag_pipeline.py    # Main orchestration
├── data/
│   └── sample_docs/       # Sample documents
├── examples/
│   └── demo.py            # Runnable demo
└── docs/
    └── architecture.md    # Design documentation
```

## Performance

| Metric | Value |
|--------|-------|
| Embedding Generation | ~100 docs/sec |
| Vector Search (10k docs) | <50ms |
| End-to-end Query | ~2-3s |
| Memory Usage | ~500MB base |

## Tech Stack

- **Python 3.10+**
- **Anthropic Claude API** - LLM generation
- **ChromaDB** - Vector database
- **sentence-transformers** - Embedding models
- **PyPDF2** - PDF processing

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Emil T Merdzhanov**
- LinkedIn: [emil-t-merdzhanov](https://www.linkedin.com/in/emil-t-merdzhanov/)
- GitHub: [EMerdzhanov](https://github.com/EMerdzhanov)
