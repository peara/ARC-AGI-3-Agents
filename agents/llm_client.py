"""OpenAI-compatible chat client for LLM calls.

Wraps the ``openai`` SDK to provide a thin, synchronous HTTP layer that
planning modules can use without importing network concerns.  The client
is configurable via constructor arguments or environment variables
``LLM_BASE_URL``, ``LLM_MODEL``, and ``LLM_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any

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

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content string.

        Parameters
        ----------
        messages:
            A list of message dicts with ``role`` and ``content`` keys,
            compatible with the OpenAI chat API.
        thinking:
            Override the server's reasoning/thinking behaviour. ``None`` falls
            back to the ``LLM_ENABLE_THINKING`` env var (unset = server default).
            ``True``/``False`` forces thinking on/off via
            ``extra_body={"chat_template_kwargs": {"enable_thinking": ...}}``,
            the control used by vLLM / LM Studio / TGI for Qwen3-style hybrid
            reasoning models. Disabling thinking typically yields a large
            latency reduction for tasks that don't need multi-step deduction.
        max_tokens:
            Cap on generated tokens. ``None`` falls back to the ``LLM_MAX_TOKENS``
            env var (unset = no cap). Always recommended as a safety net —
            without it a thinking-on call can burn an unbounded budget.

        Returns
        -------
        str
            The assistant's reply text.

        Raises
        ------
        LLMCallError
            On any ``openai.OpenAIError`` (connection, auth, rate-limit, etc.).
        """
        # Resolve env defaults: explicit arg > env var > server default.
        if thinking is None:
            env_think = os.environ.get("LLM_ENABLE_THINKING", "").strip().lower()
            if env_think in ("true", "1", "yes", "on"):
                thinking = True
            elif env_think in ("false", "0", "no", "off"):
                thinking = False
        if max_tokens is None:
            env_max = os.environ.get("LLM_MAX_TOKENS", "").strip()
            if env_max:
                try:
                    max_tokens = int(env_max)
                except ValueError:
                    pass

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,  # type: ignore[arg-type]
            "timeout": float(os.environ.get("LLM_TIMEOUT", "120")),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if thinking is not None:
            # extra_body merges into the top-level HTTP request body — the SDK
            # rejects unknown kwargs, but extra_body is forwarded verbatim.
            # LM Studio honors chat_template_kwargs at the body's top level
            # (which is where extra_body lands). Do NOT pass it as a bare
            # kwarg — the OpenAI SDK rejects it as an unknown parameter.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMCallError(str(exc)) from exc

        content: str | None = response.choices[0].message.content
        return content or ""