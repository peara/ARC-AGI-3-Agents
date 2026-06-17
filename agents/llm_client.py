"""OpenAI-compatible chat client for LLM calls.

Wraps the ``openai`` SDK to provide a thin, synchronous HTTP layer that
planning modules can use without importing network concerns.  The client
is configurable via constructor arguments or environment variables
``LLM_BASE_URL``, ``LLM_MODEL``, and ``LLM_API_KEY``.
"""

from __future__ import annotations

import os

import openai
from openai import OpenAI as OpenAIClient

__all__ = ["LLMClient", "LLMCallError"]


class LLMCallError(Exception):
    """Raised when an LLM API call fails (connection, auth, rate-limit, etc.)."""


class LLMClient:
    """Synchronous OpenAI-compatible chat client.

    Parameters
    ----------
    base_url:
        Base URL of the OpenAI-compatible API server.  Falls back to the
        ``LLM_BASE_URL`` environment variable.  Required — raises
        :class:`ValueError` if neither is set.
    model:
        Model identifier to send with each request.  Falls back to the
        ``LLM_MODEL`` environment variable.  Required — raises
        :class:`ValueError` if neither is set.
    api_key:
        API key for authentication.  Falls back to the ``LLM_API_KEY``
        environment variable.  If neither is provided the key defaults to
        an empty string, which is correct for no-auth servers (llama.cpp,
        ollama, vLLM).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        _base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        if not _base_url:
            raise ValueError(
                "base_url is required: pass it explicitly or set LLM_BASE_URL"
            )
        self.base_url: str = _base_url

        _model = model or os.environ.get("LLM_MODEL", "")
        if not _model:
            raise ValueError(
                "model is required: pass it explicitly or set LLM_MODEL"
            )
        self.model: str = _model

        _api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.api_key: str = _api_key

        self._client: OpenAIClient = OpenAIClient(
            base_url=self.base_url,
            api_key=self.api_key or "no-key",
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request and return the assistant content string.

        Parameters
        ----------
        messages:
            A list of message dicts with ``role`` and ``content`` keys,
            compatible with the OpenAI chat API.

        Returns
        -------
        str
            The assistant's reply text.

        Raises
        ------
        LLMCallError
            On any ``openai.OpenAIError`` (connection, auth, rate-limit, etc.).
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
            )
        except openai.OpenAIError as exc:
            raise LLMCallError(str(exc)) from exc

        content: str | None = response.choices[0].message.content
        return content or ""