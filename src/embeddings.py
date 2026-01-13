"""
Embedding generation module using sentence-transformers.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for processing.

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )

        return embeddings

    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode(text, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dimension


if __name__ == "__main__":
    # Quick test
    generator = EmbeddingGenerator()
    test_texts = [
        "This is a test sentence.",
        "Another example for embedding."
    ]
    embeddings = generator.generate(test_texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {generator.dimension}")
