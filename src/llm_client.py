"""Unified LLM client supporting Ollama (local) and DeepSeek (API) backends.

Auto-selects the provider based on ``LLM_PROVIDER`` env var.  When set to
``"auto"`` (default), tries Ollama first and falls back to DeepSeek if the
Ollama server is unreachable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import ollama
from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    thinking: Optional[str] = None


@dataclass
class LLMClient:
    """Thin wrapper that dispatches chat() to Ollama or DeepSeek."""

    provider: str = "auto"
    model_name: str = "nemotron-3-nano:4b"
    ollama_base_url: str = "http://localhost:11434"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    _resolved_provider: Optional[str] = field(default=None, init=False, repr=False)

    def _resolve_provider(self) -> str:
        """Decide which backend to use (cached after first call)."""
        if self._resolved_provider is not None:
            return self._resolved_provider

        if self.provider in ("ollama", "deepseek"):
            self._resolved_provider = self.provider
            return self._resolved_provider

        # auto: try ollama, fall back to deepseek
        try:
            c = ollama.Client(host=self.ollama_base_url)
            c.list()
            self._resolved_provider = "ollama"
        except Exception:
            if self.deepseek_api_key:
                self._resolved_provider = "deepseek"
            else:
                raise RuntimeError(
                    "Ollama is unreachable and DEEPSEEK_API_KEY is not set. "
                    "Start Ollama or set DEEPSEEK_API_KEY in .env."
                )
        return self._resolved_provider

    def chat(
        self,
        messages: list[dict],
        *,
        think: bool = False,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        provider = self._resolve_provider()
        if provider == "ollama":
            return self._chat_ollama(messages, think=think, temperature=temperature,
                                     top_p=top_p, max_tokens=max_tokens)
        return self._chat_deepseek(messages, think=think, temperature=temperature,
                                   top_p=top_p, max_tokens=max_tokens)

    # ── Ollama ────────────────────────────────────────────────────────────

    def _chat_ollama(self, messages, *, think, temperature, top_p, max_tokens) -> LLMResponse:
        client = ollama.Client(host=self.ollama_base_url)
        resp = client.chat(
            model=self.model_name,
            messages=messages,
            think=think,
            options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
        )
        return LLMResponse(
            content=resp.message.content or "",
            thinking=getattr(resp.message, "thinking", None),
        )

    # ── DeepSeek (OpenAI-compatible) ──────────────────────────────────────

    def _chat_deepseek(self, messages, *, think, temperature, top_p, max_tokens) -> LLMResponse:
        client = OpenAI(
            api_key=self.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=self.deepseek_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        return LLMResponse(content=content, thinking=reasoning)
