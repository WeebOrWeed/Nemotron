"""Deterministic classifier that identifies the puzzle type AND produces a
recommended execution plan (sub-questions + tool assignments).

The classify node outputs both ``puzzle_type`` and ``thought_dag`` so the
decompose step can use the plan directly -- the LLM only kicks in on retries.
"""
from __future__ import annotations

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

def _plan_gravity(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="extract_spec",
            question=(
                "Read the gravitational / falling-body puzzle below.\n\n"
                f"{prompt}\n\n"
                "Extract and output ONLY valid JSON, nothing else:\n"
                "- \"pairs\": list of [t, d] observation pairs (time and distance, "
                "numbers only, consistent units).\n"
                "- \"gravitation_function\": the governing relation as a string. "
                "In almost all cases use \"d = 0.5*g*t^2\" (or equivalent "
                "d = (1/2)*g*t^2). If the prompt states a different law, use that "
                "exact form.\n"
                "- \"query_t\": the time t for which you must find the distance d.\n"
                'Example shape: {"pairs": [[3.88, 109.74], ...], '
                '"gravitation_function": "d = 0.5*g*t^2", "query_t": 4.41}'
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="fill_equations",
            question=(
                "Structured spec (JSON):\n{extract_spec}\n\n"
                "Using the gravitation_function, substitute each observation pair "
                "(t, d) into the formula so d and t are numeric and g is still "
                "unknown. One equation per line.\n"
                "Use one line per observation, e.g. 14.92 = 0.5*g*1.37^2 or "
                "already-expanded 14.92 = 0.5*g*1.37*1.37 if you prefer.\n"
                "Show every observation as its own filled equation."
            ),
            depends_on=["extract_spec"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="expand_products",
            question=(
                "Filled equations:\n{fill_equations}\n\n"
                "This step is **expand only**—do **not** multiply numeric factors "
                "together (do **not** compute t*t, do **not** fold 0.5 with "
                "anything). Preserve full precision by leaving every factor "
                "separate.\n"
                "**Required:** replace each t^2 with two copies of t joined by "
                "* only: t^2 -> t*t using the numeric t for that line.\n"
                "**No parentheses.** Output must look exactly like:\n"
                "14.92 = 0.5*g*1.37*1.37\n"
                "one line per observation, same pattern, nothing else—no extra "
                "blocks, no evaluated products, no combining 0.5 with t terms."
            ),
            depends_on=["fill_equations"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="g_unevaluated",
            question=(
                "Expanded equations (must stay unevaluated):\n{expand_products}\n\n"
                "For **each** line d = 0.5*g*t*t, isolate g by dividing both "
                "sides, writing g as a **chain of divisions** with **each factor "
                "separate**—same style as:\n"
                "g = 14.92 / 0.5 / 1.37 / 1.37\n"
                "Use the actual d, 0.5, and the two t values from that line. "
                "**No parentheses.** **Do not** multiply or divide numerically; "
                "leave every slash as literal structure only. One g=... line per "
                "observation."
            ),
            depends_on=["expand_products"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="d_per_pair_symbolic",
            question=(
                "Spec (JSON):\n{extract_spec}\n\n"
                "Unevaluated g expressions (one per observation):\n"
                "{g_unevaluated}\n\n"
                "For **each** observation, derive the distance at **query_t** "
                "from the gravitation_function by substituting that row's g "
                "expression—keep the result as **one symbolic formula** d_i "
                "(only multiplications/divisions/powers, still **no** decimal "
                "evaluation). When the law is d = 0.5*g*t^2, prefer the fully "
                "expanded chain form with **no** parentheses and **no** "
                "partial products, e.g. "
                "d_i = d_obs * query_t * query_t / t_obs / t_obs (use numeric "
                "query_t and t_obs from the JSON and each g-line).\n"
                "Label d_1, d_2, ... matching each pair. **Do not** compute "
                "numeric d_i yet."
            ),
            depends_on=["g_unevaluated", "extract_spec"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="d_i_representations",
            question=(
                "Symbolic per-pair distances (unevaluated):\n{d_per_pair_symbolic}\n\n"
                "Re-output **only** the same ``d_i`` lines as above—**same count** "
                "as in that block (often five, but puzzles may have fewer "
                "observations). **Do not** add ``d_{n+1}`` or any extra line; "
                "**do not** drop a line.\n"
                "Each line: ``d_k = <decimals with only * and />`` — **no** "
                "parentheses, **no** powers, **no** letters (especially **no** "
                "``g`` on the right-hand side), **no** JSON.\n"
                "**Do not** compute numeric values. No blank lines, markdown, or "
                "commentary."
            ),
            depends_on=["d_per_pair_symbolic"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="eval_geom_mean_d",
            question=(
                "Python: parse d_k lines, multiply the evaluated d values together, "
                "then take the n-th root (geometric mean, n = line count). "
                "Return that value rounded to 2dp."
            ),
            depends_on=["d_i_representations"],
            tool="gravity_geom_mean_chain_exprs",
            tool_input="{d_i_representations}",
            answer=None,
        ),
    ]


def _plan_unit(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="extract_pairs",
            question=(
                "Read the unit conversion puzzle below. Extract ALL example "
                "conversions as [from_value, to_value] pairs (numbers only), "
                "and the target value the puzzle asks you to convert.\n\n"
                f"{prompt}\n\n"
                "Output ONLY valid JSON in this exact format, nothing else:\n"
                '{"pairs": [[from1, to1], [from2, to2], ...], "target": <number>}'
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="compute_factor",
            question=(
                "Given these unit-conversion example pairs (linear scale in the "
                "same unit system):\n"
                "{extract_pairs}\n\n"
                "For each pair [from, to], compute factor = to / from.\n"
                "Then compute the average factor across all pairs.\n"
                "Show your work briefly, then output ONLY the average factor as a "
                "number on the last line."
            ),
            depends_on=["extract_pairs"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        ThoughtNode(
            id="apply_convert",
            question=(
                "The linear conversion factor is: {compute_factor}\n"
                "Full context (JSON with pairs and target): {extract_pairs}\n\n"
                "Compute converted = factor * target using the \"target\" field "
                "from the JSON. Match the precision of the examples (typically "
                "two decimal places).\n"
                "Output ONLY the final number on the last line."
            ),
            depends_on=["compute_factor", "extract_pairs"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


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
    is an LLM-per-step chain (or mixed with tools after retries from
    ``decompose``).  For unknown types the DAG is left empty so
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
