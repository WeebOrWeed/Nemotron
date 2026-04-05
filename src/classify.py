"""Classifier that identifies the puzzle type AND produces an execution plan.

For ``gravity_physics`` and ``unit_conversion`` the plan is produced by the
LLM-based ``QueryPlanner`` (see ``src/planner.py``), which outputs a
mermaid-style topology + node-metadata dict and builds per-pair ThoughtNodes.
Other types use deterministic DAG templates.

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

def _plan_numeral(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="extract_data",
            question=(
                "Read the numeral-system puzzle below. Extract every example that "
                "relates a conventional number (usually Arabic digits) to the "
                "puzzle's numeral notation, and the target decimal integer you "
                "must express in that notation.\n\n"
                f"{prompt}\n\n"
                "Output ONLY valid JSON, nothing else:\n"
                '{"examples": [["12", "XII"], ["5", "V"]], "target_decimal": 2024}\n'
                "(Use [] for examples if the puzzle only states the target number.)"
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="infer_system",
            question=(
                "Puzzle data (JSON):\n{extract_data}\n\n"
                "Which numeral system does the puzzle use (e.g. Roman, another "
                "base)? Summarize rules visible in the examples. "
                "End with a line: TARGET_DECIMAL=<integer> matching target_decimal."
            ),
            depends_on=["extract_data"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="express_target",
            question=(
                "Data:\n{extract_data}\n\nSystem summary:\n{infer_system}\n\n"
                "Express the target integer in the required notation only "
                "(e.g. standard Roman: I,V,X,L,C,D,M). "
                "Output ONLY that representation on the last line, no words."
            ),
            depends_on=["infer_system", "extract_data"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


def _plan_cipher(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="extract_word_pairs",
            question=(
                "Read the cipher puzzle below. From each example line, list "
                "aligned (encrypted_word, plain_word) pairs in order—every pair "
                "that shows the encryption rule.\n\n"
                f"{prompt}\n\n"
                "Output ONLY valid JSON, nothing else:\n"
                '{"pairs": [["encrypted_word", "plain_word"], ...]}'
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="extract_cipher",
            question=(
                "Read the cipher puzzle below. What is the exact ciphertext "
                "string you must decrypt (the line after \"decrypt\" or similar)?\n\n"
                f"{prompt}\n\n"
                "Output ONLY that ciphertext on the last line—same spelling and "
                "spacing as in the puzzle."
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="decrypt",
            question=(
                "Example alignments (JSON):\n{extract_word_pairs}\n\n"
                "Ciphertext to decrypt:\n{extract_cipher}\n\n"
                "Infer a consistent per-letter substitution from the examples "
                "(lowercase English as in the plaintexts). Decrypt the ciphertext. "
                "Output ONLY the decrypted plaintext on the last line "
                "(lowercase words separated by spaces)."
            ),
            depends_on=["extract_word_pairs", "extract_cipher"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


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

    Gravity and unit conversion use the LLM-based ``QueryPlanner``; other
    types use deterministic DAG templates.
    """
    planner = QueryPlanner(llm)

    plan_builders: dict[str, callable] = {
        "gravity_physics": planner.plan_gravity,
        "unit_conversion": planner.plan_unit,
        "numeral_conversion": _plan_numeral,
        "cipher_decryption": _plan_cipher,
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
