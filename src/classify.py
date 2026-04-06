"""Classifier that identifies the puzzle type AND produces an execution plan.

For ``gravity_physics``, ``unit_conversion``, ``numeral_conversion``, and
``cipher_decryption`` the plan is produced by the LLM-based ``QueryPlanner``
(see ``src/planner.py``), which outputs a mermaid-style topology +
node-metadata dict and builds ThoughtNodes.  Other types use deterministic
DAG templates.

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

# ── per-type DAG templates ─────────────────────────────────────────────
# Each builder receives the raw prompt and returns a list[ThoughtNode].
# The ``tool`` field tells the solver which deterministic tool (or LLM
# fallback) to use for that step.

def _plan_bit(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="identify_rule",
            question=(
                "Bit-manipulation puzzle:\n\n"
                f"{prompt}\n\n"
                "Analyze ALL given input → output pairs (8-bit binary strings). "
                "The same boolean/bit rule must map every input to its output. "
                "Consider: XOR with a constant, AND/OR with a mask, NOT, left/right "
                "shift, rotation, and 2-input or 3-input combinations (e.g. "
                "MUX-style).\n"
                "State the exact rule and any constants. Show verification for "
                "each pair.\n"
                "End with a single clear line: RULE = <concise description>."
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="apply_rule",
            question=(
                "Original puzzle:\n"
                f"{prompt}\n\n"
                "Inferred rule from analysis:\n{identify_rule}\n\n"
                "Apply this rule to the target 8-bit input the puzzle asks for "
                "(the unknown output case).\n"
                "Output ONLY the 8-bit binary result on the last line — exactly "
                "8 characters, each 0 or 1, no spaces or prefix."
            ),
            depends_on=["identify_rule"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


def _plan_equation(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="extract_examples",
            question=(
                "Equation-transformation puzzle. Each example is a 5-character "
                "left-hand side (chars 0-1 = left operand, char 2 = operator, "
                "chars 3-4 = right operand), an equals sign, and a result.\n\n"
                f"{prompt}\n\n"
                "Extract every example as [left5, result] and the 5-character "
                "target expression to evaluate (often after "
                "\"determine the result for:\").\n\n"
                "Output ONLY valid JSON, nothing else:\n"
                '{"examples": [["12+34", "46"], ...], "target": "90+12"}'
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="infer_rule",
            question=(
                "Structured puzzle data (JSON):\n{extract_examples}\n\n"
                "Each left token is exactly 5 characters: LEFT=chars 0-1, "
                "OPERATOR=char 2, RIGHT=chars 3-4.\n"
                "1) Group examples by operator (char at index 2).\n"
                "2) Keep only examples whose operator matches the target's "
                "operator.\n"
                "3) Find one consistent rule mapping (LEFT, RIGHT) → result. "
                "Try: digit arithmetic (+, -, *, //, %, gcd, xor…), string "
                "concatenation or reversal of parts, per-character ordinal ops "
                "with mod/offset, or a fixed char substitution over the four "
                "operand characters.\n"
                "Show that the rule matches every kept example.\n"
                "End with a single line: RULE = <concise description>."
            ),
            depends_on=["extract_examples"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="apply_rule",
            question=(
                "Data (JSON):\n{extract_examples}\n\n"
                "Discovered rule:\n{infer_rule}\n\n"
                "Apply the rule to the target's left operand (chars 0-1), "
                "operator (char 2), and right operand (chars 3-4). "
                "Match the result style of the examples (length and format).\n"
                "Output ONLY the result string on the last line, nothing else."
            ),
            depends_on=["infer_rule", "extract_examples"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


def make_classify_node(llm: LLMClient):
    """Factory that returns the classify node function with the LLM client bound.

    Gravity, unit conversion, numeral conversion, and cipher decryption use the
    LLM-based ``QueryPlanner``; other types use deterministic DAG templates.
    """
    planner = QueryPlanner(llm)

    plan_builders: dict[str, callable] = {
        "gravity_physics": planner.plan_gravity,
        "unit_conversion": planner.plan_unit,
        "numeral_conversion": planner.plan_numeral,
        "cipher_decryption": planner.plan_cipher,
        "bit_manipulation": _plan_bit,
        "equation_transform": _plan_equation,
    }

    def classify_node(state: GraphState) -> dict:
        """Classify the puzzle and emit a recommended execution plan.

        Returns ``puzzle_type`` and ``thought_dag``.  For known types the
        DAG is built by a plan builder (LLM-based for gravity, deterministic
        for others).  For unknown types the DAG is left empty so decompose
        will generate one via the LLM.
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
