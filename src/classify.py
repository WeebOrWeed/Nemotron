"""Deterministic classifier that identifies the puzzle type AND produces a
recommended execution plan (sub-questions + tool assignments).

The classify node outputs both ``puzzle_type`` and ``thought_dag`` so the
decompose step can use the plan directly -- the LLM only kicks in on retries.
"""
from __future__ import annotations

import json

from src.state import GraphState, ThoughtNode

# ── keyword → puzzle type mapping ──────────────────────────────────────

PUZZLE_SIGNATURES: dict[str, str] = {
    "bit manipulation": "bit_manipulation",
    "numeral system": "numeral_conversion",
    "unit conversion": "unit_conversion",
    "encryption rules": "cipher_decryption",
    "transformation rules": "equation_transform",
    "gravitational constant": "gravity_physics",
}

# ── per-type DAG templates ─────────────────────────────────────────────
# Each builder receives the raw prompt and returns a list[ThoughtNode].
# The ``tool`` field tells the solver which deterministic tool (or LLM
# fallback) to use for that step.

_EQ_TRANSFORM_QUESTION = """\
{prompt}

INSTRUCTIONS FOR SOLVING:
1. Each input is 5 chars: chars 0-1 are LEFT, char 2 is the OPERATOR, chars 3-4 are RIGHT.
2. Group examples by their operator (char at position 2).
3. Focus on examples sharing the SAME operator as the target.
4. Look for a consistent transformation from (LEFT, RIGHT) → RESULT:
   - If chars are digits: try arithmetic (add, subtract, multiply, reverse digits, etc.)
   - If chars are symbols: compare ASCII/ordinal values; try per-character operations
     like (ord(L[i]) + ord(R[i])) mod N, XOR, subtraction, etc.
   - Check if result length varies — it may depend on the operation.
5. Apply the discovered rule to the target's LEFT and RIGHT operands.

Answer with ONLY the result string on the last line, nothing else."""


def _plan_gravity(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question="Solve gravity_physics puzzle",
            depends_on=[],
            tool="solve_gravity_physics",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


def _plan_unit(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question="Solve unit_conversion puzzle",
            depends_on=[],
            tool="solve_unit_conversion",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


def _plan_numeral(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question="Solve numeral_conversion puzzle",
            depends_on=[],
            tool="solve_numeral_conversion",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


def _plan_cipher(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question="Solve cipher_decryption puzzle",
            depends_on=[],
            tool="solve_cipher_decryption",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


def _plan_bit(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question="Solve bit_manipulation puzzle",
            depends_on=[],
            tool="solve_bit_manipulation",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


def _plan_equation(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="solve",
            question=_EQ_TRANSFORM_QUESTION.format(prompt=prompt),
            depends_on=[],
            tool="solve_equation_transform",
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


_PLAN_BUILDERS: dict[str, callable] = {
    "gravity_physics": _plan_gravity,
    "unit_conversion": _plan_unit,
    "numeral_conversion": _plan_numeral,
    "cipher_decryption": _plan_cipher,
    "bit_manipulation": _plan_bit,
    "equation_transform": _plan_equation,
}


def classify_node(state: GraphState) -> dict:
    """Classify the puzzle and emit a recommended execution plan.

    Returns ``puzzle_type`` and ``thought_dag``.  For known types the DAG
    contains the deterministic solver tool with the appropriate LLM-fallback
    question already baked in.  For unknown types the DAG is left empty so
    decompose will generate one via the LLM.
    """
    prompt = state["prompt"]
    prompt_lower = prompt.lower()

    puzzle_type = "unknown"
    for signature, ptype in PUZZLE_SIGNATURES.items():
        if signature in prompt_lower:
            puzzle_type = ptype
            break

    builder = _PLAN_BUILDERS.get(puzzle_type)
    dag = builder(prompt) if builder else None

    return {"puzzle_type": puzzle_type, "thought_dag": dag}
