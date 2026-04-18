"""Classifier that identifies the puzzle type AND produces an execution plan.

All six puzzle types use the ``QueryPlanner`` (see ``src/planner.py``).

The classify node outputs both ``puzzle_type`` and ``thought_dag`` so the
decompose step can use the plan directly -- the LLM only kicks in on retries.
"""
from __future__ import annotations

import re

from src.state import GraphState, ThoughtNode
from src.llm_client import LLMClient
from src.planner import QueryPlanner

# ── keyword → puzzle type mapping ──────────────────────────────────────

def classify_equation_subtype(prompt: str) -> str:
    """Classify equation_transform into one of four subtypes.

    Returns one of: equation_numeric_deduce, equation_numeric_guess,
    cryptarithm_deduce, cryptarithm_guess.
    """
    try:
        after_header = prompt.split("Below are a few examples:\n", 1)[1]
        examples_text, rest = after_header.split("\nNow, determine the result for: ", 1)
        question_text = rest.strip()
    except (IndexError, ValueError):
        return "cryptarithm_guess"

    if any(c.isdigit() for c in examples_text):
        q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
        if q_match and re.search(
            r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text
        ):
            return "equation_numeric_deduce"
        return "equation_numeric_guess"

    if len(question_text) == 5:
        q_op = question_text[2]
        for ex_line in examples_text.strip().splitlines():
            inp = ex_line.split(" = ")[0].strip()
            if len(inp) == 5 and inp[2] == q_op:
                return "cryptarithm_deduce"

    return "cryptarithm_guess"


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
