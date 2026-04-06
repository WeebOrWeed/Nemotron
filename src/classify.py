"""Classifier that identifies the puzzle type AND produces an execution plan.

All six puzzle types use the ``QueryPlanner`` (see ``src/planner.py``).

The classify node outputs both ``puzzle_type`` and ``thought_dag`` so the
decompose step can use the plan directly -- the LLM only kicks in on retries.
"""
from __future__ import annotations

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


def make_classify_node(llm: LLMClient):
    """Factory that returns the classify node function with the LLM client bound.

    All puzzle types use the ``QueryPlanner``.
    """
    planner = QueryPlanner(llm)

    plan_builders: dict[str, callable] = {
        "gravity_physics": planner.plan_gravity,
        "unit_conversion": planner.plan_unit,
        "numeral_conversion": planner.plan_numeral,
        "cipher_decryption": planner.plan_cipher,
        "bit_manipulation": planner.plan_bit,
        "equation_transform": planner.plan_equation,
    }

    def classify_node(state: GraphState) -> dict:
        """Classify the puzzle and emit a recommended execution plan.

        Returns ``puzzle_type`` and ``thought_dag``.  For known types the
        DAG is built by a plan builder.  For unknown types the DAG is left
        empty so decompose will generate one via the LLM.
        """
        prompt = state["prompt"]
        prompt_lower = prompt.lower()

        puzzle_type = "unknown"
        for signature, ptype in PUZZLE_SIGNATURES.items():
            if signature in prompt_lower:
                puzzle_type = ptype
                break

        builder = plan_builders.get(puzzle_type)
        dag = builder(prompt) if builder else None

        return {"puzzle_type": puzzle_type, "thought_dag": dag}

    return classify_node
