"""Unified LLM client supporting Ollama, OpenRouter, and HuggingFace backends.

Auto-selects the provider based on ``LLM_PROVIDER`` env var.  When set to
``"auto"`` (default), tries Ollama first and falls back to OpenRouter.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
import ollama
from openai import OpenAI, RateLimitError


@dataclass
class LLMResponse:
    content: str
    thinking: Optional[str] = None


@dataclass
class LLMClient:
    """Thin wrapper that dispatches chat() to Ollama, OpenRouter, or HuggingFace."""

    provider: str = "auto"
    model_name: str = "nemotron-3-nano:4b"
    ollama_base_url: str = "http://localhost:11434"
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    hf_token: str = ""
    hf_model: str = "nvidia/OpenMath-Nemotron-14B-Kaggle"

    _resolved_provider: Optional[str] = field(default=None, init=False, repr=False)

    def _resolve_provider(self) -> str:
        """Decide which backend to use (cached after first call)."""
        if self._resolved_provider is not None:
            return self._resolved_provider

        if self.provider in ("ollama", "openrouter", "huggingface"):
            self._resolved_provider = self.provider
            return self._resolved_provider

        # auto: try ollama, fall back to openrouter
        try:
            c = ollama.Client(host=self.ollama_base_url)
            c.list()
            self._resolved_provider = "ollama"
        except Exception:
            if self.openrouter_api_key:
                self._resolved_provider = "openrouter"
            else:
                raise RuntimeError(
                    "Ollama is unreachable and OPENROUTER_API_KEY is not set."
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
        if provider == "openrouter":
            return self._chat_openrouter(messages, temperature=temperature,
                                         top_p=top_p, max_tokens=max_tokens)
        if provider == "huggingface":
            return self._chat_huggingface(messages, temperature=temperature,
                                          top_p=top_p, max_tokens=max_tokens)
        raise RuntimeError(f"Unknown provider: {provider!r}")

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

    # ── OpenRouter (OpenAI-compatible) ────────────────────────────────────

    _OPENROUTER_RATE_LIMIT_DELAY = 60.0  # 429: wait a full minute before retry
    _OPENROUTER_MAX_RETRIES = 2

    def _chat_openrouter(self, messages, *, temperature, top_p, max_tokens) -> LLMResponse:
        client = OpenAI(
            api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        last_exc: Exception | None = None
        for attempt in range(self._OPENROUTER_MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                return LLMResponse(content=content)
            except RateLimitError as exc:
                last_exc = exc
                # Per-day limits won't recover within a retry window — raise immediately.
                if "per-day" in str(exc).lower() or "per_day" in str(exc).lower():
                    raise
                if attempt < self._OPENROUTER_MAX_RETRIES:
                    time.sleep(self._OPENROUTER_RATE_LIMIT_DELAY)
                    continue
        raise last_exc

    # ── HuggingFace Inference (via featherless-ai router, OpenAI-compatible) ──

    _HF_BASE_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"
    _HF_503_DELAY = 20.0
    _HF_429_DELAY = 10.0
    _HF_MAX_RETRIES = 3

    def _chat_huggingface(self, messages, *, temperature, top_p, max_tokens) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.hf_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        for attempt in range(self._HF_MAX_RETRIES + 1):
            resp = requests.post(self._HF_BASE_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                if attempt < self._HF_MAX_RETRIES:
                    time.sleep(self._HF_503_DELAY)
                    continue
                resp.raise_for_status()
            if resp.status_code == 429:
                if attempt < self._HF_MAX_RETRIES:
                    time.sleep(self._HF_429_DELAY)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            return LLMResponse(content=content)
        raise RuntimeError("HuggingFace: max retries exceeded")
