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
    """Create the best available cloud LLMClient for DAG planning."""
    if OPEN_ROUTER_API_KEY:
        return LLMClient(
            provider="openrouter",
            openrouter_api_key=OPEN_ROUTER_API_KEY,
            openrouter_model=OPEN_ROUTER_MODEL,
        )
    if DEEPSEEK_API_KEY:
        return LLMClient(
            provider="deepseek",
            deepseek_api_key=DEEPSEEK_API_KEY,
            deepseek_model=DEEPSEEK_MODEL,
        )
    raise RuntimeError(
        "No planner API key set. Set OPEN_ROUTER_API_KEY or DEEPSEEK_API_KEY in .env."
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
