"""
Document loading and chunking module.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re


class TextChunker:
    """Split text into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)

                if break_point > start + self.chunk_size // 2:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks


class DocumentLoader:
    """Load and process documents from various formats."""

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document loader.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
        """
        self.chunker = TextChunker(chunk_size, chunk_overlap)

    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single file.

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary with 'content', 'metadata', and 'chunks'.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")

        if extension == '.pdf':
            content = self._load_pdf(path)
        else:
            content = self._load_text(path)

        chunks = self.chunker.chunk(content)

        return {
            "content": content,
            "metadata": {
                "source": str(path.name),
                "extension": extension,
                "size": len(content)
            },
            "chunks": chunks
        }

    def load_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load all supported files from a directory.

        Args:
            directory: Path to the directory.

        Returns:
            List of document dictionaries.
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents = []
        for file_path in path.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")

        return documents

    def _load_text(self, path: Path) -> str:
        """Load text from a text file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_pdf(self, path: Path) -> str:
        """Load text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return '\n\n'.join(text_parts)
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")


if __name__ == "__main__":
    # Quick test
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    test_text = "This is a test. " * 50
    chunks = chunker.chunk(test_text)
    print(f"Split into {len(chunks)} chunks")
