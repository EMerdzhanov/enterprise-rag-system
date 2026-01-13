"""
LLM client module for Claude API integration.
"""

import os
from typing import Optional
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Claude API client for response generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229"
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model name to use.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using the provided context.

        Args:
            query: User's question.
            context: Retrieved context from documents.
            system_prompt: Optional system prompt override.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Generated response string.
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain
relevant information, say so. Be concise and accurate."""

        user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided above."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return response.content[0].text

    def generate_simple(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a simple response without RAG context.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Generated response string.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text


if __name__ == "__main__":
    # Quick test (requires valid API key)
    try:
        client = LLMClient()
        response = client.generate_simple("Say hello in one word.")
        print(f"Response: {response}")
    except ValueError as e:
        print(f"Skipping test: {e}")
