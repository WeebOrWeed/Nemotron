"""Classifier that identifies the puzzle type AND produces an execution plan.

All six puzzle types use the ``QueryPlanner`` (see ``src/planner.py``).
The planner uses a dedicated cloud LLMClient (OpenRouter or DeepSeek)
so it always gets a capable model, while execution nodes use whatever
``LLM_PROVIDER`` is configured.
"""
from __future__ import annotations

from src.config import (
    OPEN_ROUTER_API_KEY, OPEN_ROUTER_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    PLANNER_PROVIDER, MODEL_NAME, OLLAMA_BASE_URL,
    HF_PLANNER_BASE_MODEL, HF_PLANNER_LORA_PATH,
    HF_PLANNER_LOAD_4BIT, HF_PLANNER_MAX_NEW_TOKENS,
)
from src.state import GraphState, ThoughtNode
from src.llm_client import LLMClient
from src.planner import QueryPlanner

# ── keyword → puzzle type mapping ──────────────────────────────────────

PUZZLE_SIGNATURES: dict[str, str] = {
    "bit manipulation": "bit_manipulation",
    "numeral system": "numeral_conversion",
    "unit conversion": "unit_conversion",
    "encryption rules": "cipher_decryption",
    "transformation rules": "equation_transform",
    "gravitational constant": "gravity_physics",
}


def _make_planner_llm() -> LLMClient:
    """Create an LLMClient for DAG planning.

    Honours ``PLANNER_PROVIDER`` env var when set (``ollama``, ``openrouter``,
    ``deepseek``, or ``hf_lora``); otherwise picks the best available cloud key.
    """
    forced = PLANNER_PROVIDER.lower().strip()
    if forced == "hf_lora":
        return LLMClient(
            provider="hf_lora",
            hf_base_model=HF_PLANNER_BASE_MODEL,
            hf_lora_path=HF_PLANNER_LORA_PATH,
            hf_max_new_tokens=HF_PLANNER_MAX_NEW_TOKENS,
            hf_load_4bit=HF_PLANNER_LOAD_4BIT,
        )
    if forced == "ollama":
        return LLMClient(
            provider="ollama", model_name=MODEL_NAME, ollama_base_url=OLLAMA_BASE_URL,
        )
    if forced == "openrouter" or (forced == "" and OPEN_ROUTER_API_KEY):
        if not OPEN_ROUTER_API_KEY:
            raise RuntimeError("PLANNER_PROVIDER=openrouter but OPEN_ROUTER_API_KEY is not set.")
        return LLMClient(
            provider="openrouter",
            openrouter_api_key=OPEN_ROUTER_API_KEY,
            openrouter_model=OPEN_ROUTER_MODEL,
        )
    if forced == "deepseek" or (forced == "" and DEEPSEEK_API_KEY):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("PLANNER_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set.")
        return LLMClient(
            provider="deepseek",
            deepseek_api_key=DEEPSEEK_API_KEY,
            deepseek_model=DEEPSEEK_MODEL,
        )
    raise RuntimeError(
        "No planner backend available. Set PLANNER_PROVIDER=ollama/hf_lora or "
        "set OPEN_ROUTER_API_KEY / DEEPSEEK_API_KEY in .env."
    )


def make_classify_node(llm: LLMClient):
    """Factory that returns the classify node function with the LLM client bound.

    A separate cloud-backed LLMClient is created for the planner so
    DAG generation always uses a capable model.
    """
    planner_llm = _make_planner_llm()
    planner = QueryPlanner(planner_llm)

    def classify_node(state: GraphState) -> dict:
        """Classify the puzzle and produce an execution plan.

        Returns ``puzzle_type`` and ``thought_dag``.  The planner calls
        the LLM to compose a DAG from the tool catalogue.
        """
        prompt = state["prompt"]
        prompt_lower = prompt.lower()

        puzzle_type = "unknown"
        for signature, ptype in PUZZLE_SIGNATURES.items():
            if signature in prompt_lower:
                puzzle_type = ptype
                break

        dag = planner.plan(puzzle_type, prompt) if puzzle_type != "unknown" else None

        return {"puzzle_type": puzzle_type, "thought_dag": dag}

    return classify_node
