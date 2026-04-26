"""Unified LLM client supporting Ollama, OpenRouter, and DeepSeek backends.

Auto-selects the provider based on ``LLM_PROVIDER`` env var.  When set to
``"auto"`` (default), tries Ollama first, then OpenRouter, then DeepSeek.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import ollama
from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    thinking: Optional[str] = None


@dataclass
class LLMClient:
    """Thin wrapper that dispatches chat() to Ollama, OpenRouter, or DeepSeek."""

    provider: str = "auto"
    model_name: str = "nemotron-3-nano:4b"
    ollama_base_url: str = "http://localhost:11434"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    hf_base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_lora_path: str = "models/planner-lora"
    hf_max_new_tokens: int = 384
    hf_load_4bit: bool = True

    _resolved_provider: Optional[str] = field(default=None, init=False, repr=False)
    _hf_tokenizer: Any = field(default=None, init=False, repr=False)
    _hf_model: Any = field(default=None, init=False, repr=False)
    _hf_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _resolve_provider(self) -> str:
        """Decide which backend to use (cached after first call)."""
        if self._resolved_provider is not None:
            return self._resolved_provider

        if self.provider in ("ollama", "openrouter", "deepseek", "hf_lora"):
            self._resolved_provider = self.provider
            return self._resolved_provider

        # auto: try ollama → openrouter → deepseek
        try:
            c = ollama.Client(host=self.ollama_base_url)
            c.list()
            self._resolved_provider = "ollama"
        except Exception:
            if self.openrouter_api_key:
                self._resolved_provider = "openrouter"
            elif self.deepseek_api_key:
                self._resolved_provider = "deepseek"
            else:
                raise RuntimeError(
                    "Ollama is unreachable and no API key is set. "
                    "Start Ollama or set OPEN_ROUTER_API_KEY / DEEPSEEK_API_KEY in .env."
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
            return self._chat_openrouter(messages, think=think, temperature=temperature,
                                         top_p=top_p, max_tokens=max_tokens)
        if provider == "hf_lora":
            return self._chat_hf_lora(messages, think=think, temperature=temperature,
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

    # ── OpenRouter (OpenAI-compatible) ────────────────────────────────────

    def _chat_openrouter(self, messages, *, think, temperature, top_p, max_tokens) -> LLMResponse:
        client = OpenAI(
            api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=self.openrouter_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        return LLMResponse(content=content, thinking=reasoning)

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

    # ── Local HuggingFace + PEFT LoRA ─────────────────────────────────────

    def _load_hf_lora(self) -> None:
        """Lazy-load the local planner LoRA adapter."""
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        adapter_config_path = os.path.join(self.hf_lora_path, "adapter_config.json")
        base_model = self.hf_base_model
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path") or base_model

        tokenizer = AutoTokenizer.from_pretrained(self.hf_lora_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if self.hf_load_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, self.hf_lora_path)
        model.eval()

        self._hf_tokenizer = tokenizer
        self._hf_model = model

    def _chat_hf_lora(self, messages, *, think, temperature, top_p, max_tokens) -> LLMResponse:
        if think:
            # Local HF generation has no separate "thinking" channel.
            think = False
        with self._hf_lock:
            self._load_hf_lora()

            import torch

            tokenizer = self._hf_tokenizer
            model = self._hf_model
            encoded = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            if hasattr(encoded, "data"):
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
            elif isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
            else:
                input_ids = encoded
                attention_mask = None

            device = model.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            do_sample = temperature is not None and temperature > 0
            gen_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": min(max_tokens, self.hf_max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with torch.inference_mode():
                output = model.generate(**gen_kwargs)
            generated = output[0][input_ids.shape[-1]:]
            content = tokenizer.decode(generated, skip_special_tokens=True).strip()
            return LLMResponse(content=content, thinking=None)
