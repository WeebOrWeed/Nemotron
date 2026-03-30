from __future__ import annotations

import json
import re

import ollama

from src.state import GraphState, ThoughtNode, FailureRecord

# ---------------------------------------------------------------------------
# Compact system prompt with one few-shot example per puzzle type
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = """\
You decompose puzzles into a DAG of steps. Output ONLY a JSON array.

Each node: {"id": "...", "question": "...", "depends_on": [...], "tool": "...", "tool_input": "..."}

- "tool": tool name or "ask_llm" for reasoning. ALWAYS set this field.
- "tool_input": JSON string. Use {parent_id} to reference a parent's answer.
- "depends_on": parent IDs this node waits for. [] = root.
- Last node = final answer.
- ask_llm nodes: question field is sent to the LLM. Include full context.
- tool nodes: question field is a short description; tool_input does the work.

TOOLS: ask_llm, eval_math, apply_formula, round_number, average, regex_extract,
xor_binary, and_binary, or_binary, not_binary, shift_left, shift_right,
rotate_left, rotate_right, substitute_chars, build_char_map, to_roman,
from_roman, linear_factor, compute_gravity_g, compute_gravity_d.

=== EXAMPLE FOR gravity_physics ===
Prompt: "...For t = 1.37s, distance = 14.92 m\\nFor t = 4.27s, distance = 144.96 m...determine the falling distance for t = 4.41s..."
[
  {"id": "extract", "question": "Extract all (t, d) pairs and the target t as JSON. Format: {observations: [[t,d],...], target_t: n}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "get_g", "question": "Compute g from observations", "depends_on": ["extract"], "tool": "ask_llm", "tool_input": null},
  {"id": "result", "question": "Compute final d", "depends_on": ["get_g", "extract"], "tool": "ask_llm", "tool_input": null}
]
Note: get_g node should ask LLM to compute g=2d/t^2 for each pair and average. result node should compute d=0.5*g*t^2 and round to 2 decimals.

=== EXAMPLE FOR numeral_conversion ===
Prompt: "...numbers are secretly converted...11 -> XI, 15 -> XV...write the number 38..."
[
  {"id": "find_number", "question": "What number needs to be converted? Look at the last line of the puzzle. Output ONLY the integer.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "convert", "question": "Convert to Roman", "depends_on": ["find_number"], "tool": "to_roman", "tool_input": "{\\"number\\": {find_number}}"}
]

=== EXAMPLE FOR unit_conversion ===
Prompt: "...10.08 m becomes 6.69, 17.83 m becomes 11.83...convert 25.09 m..."
[
  {"id": "extract", "question": "List all (input, output) pairs and target value from the puzzle. Format answer as JSON: {pairs: [[in,out],...], target: n}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "get_factor", "question": "Compute conversion factor", "depends_on": ["extract"], "tool": "ask_llm", "tool_input": null},
  {"id": "result", "question": "Multiply factor by target and round to 2 decimals", "depends_on": ["get_factor", "extract"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR cipher_decryption ===
Prompt: "...encryption rules...ucoov pwgtfyoqg -> queen discovers...decrypt: trb wzrswvog"
[
  {"id": "extract_pairs", "question": "From the examples, list all aligned (encrypted_word, plain_word) pairs. Format: [[encrypted, plain], ...]. Include every word pair.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "build_map", "question": "Build char substitution map", "depends_on": ["extract_pairs"], "tool": "build_char_map", "tool_input": "{\\"pairs\\": {extract_pairs}}"},
  {"id": "get_target", "question": "What is the encrypted text to decrypt? Output ONLY the encrypted text.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "decrypt", "question": "Apply substitution", "depends_on": ["build_map", "get_target"], "tool": "substitute_chars", "tool_input": "{\\"text\\": \\"{get_target}\\", \\"mapping\\": {build_map}}"}
]

=== EXAMPLE FOR bit_manipulation ===
Prompt: "...bit manipulation rule transforms 8-bit binary...01010001 -> 11011101...determine output for 00110100"
[
  {"id": "identify_rule", "question": "Analyze ALL input->output pairs. Try XOR, NOT, shifts, rotations. Which single operation transforms every input to its output? State the operation and any constant. Show your work for each pair.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "apply_rule", "question": "Using the rule: {identify_rule}\\n\\nApply it to the target input from the puzzle. Output ONLY the 8-bit binary result.", "depends_on": ["identify_rule"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR equation_transform ===
Prompt: "...transformation rules applied to equations...`!*[{ = '\"[`...determine result for: [[-!'"
[
  {"id": "identify_rule", "question": "Analyze the input->output examples character by character. Build a substitution map or identify the transformation pattern. Show the mapping for each character.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "apply_rule", "question": "Using the transformation: {identify_rule}\\n\\nApply it to the target expression. Output ONLY the transformed result.", "depends_on": ["identify_rule"], "tool": "ask_llm", "tool_input": null}
]

Output ONLY the JSON array for the given puzzle. No commentary.\
"""

RETRY_ADDENDUM = """

PREVIOUS ATTEMPT FAILED:
{failure_details}

Generate a DIFFERENT decomposition. Try simpler steps or different tools.\
"""


def _build_user_prompt(prompt: str, puzzle_type: str, failure_log: list[FailureRecord]) -> str:
    msg = f"Puzzle type: {puzzle_type}\n\nPuzzle prompt:\n{prompt}"
    if failure_log:
        details = "\n".join(
            f"- Node '{f['node_id']}': {f['error']}" for f in failure_log
        )
        msg += RETRY_ADDENDUM.format(failure_details=details)
    return msg


def _parse_dag_json(raw: str) -> list[ThoughtNode]:
    """Extract JSON array from LLM response, tolerating markdown fences."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find JSON array in the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        text = text[bracket_start:bracket_end + 1]

    nodes_raw = json.loads(text)
    if not isinstance(nodes_raw, list) or len(nodes_raw) == 0:
        raise ValueError("LLM returned empty or non-list JSON")

    dag: list[ThoughtNode] = []
    seen_ids: set[str] = set()
    for item in nodes_raw:
        node_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        depends_on = item.get("depends_on", [])
        tool = item.get("tool") or None
        tool_input = item.get("tool_input") or None

        if not node_id or not question:
            continue
        if not isinstance(depends_on, list):
            depends_on = []
        depends_on = [str(d) for d in depends_on]
        if node_id in seen_ids:
            node_id = f"{node_id}_{len(dag)}"
        seen_ids.add(node_id)

        if isinstance(tool, str):
            tool = tool.strip() or None
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input)
        elif isinstance(tool_input, str):
            tool_input = tool_input.strip() or None

        dag.append(ThoughtNode(
            id=node_id,
            question=question,
            depends_on=depends_on,
            tool=tool,
            tool_input=tool_input,
            answer=None,
        ))

    if not dag:
        raise ValueError("No valid nodes parsed from LLM output")

    valid_ids = {n["id"] for n in dag}
    for node in dag:
        node["depends_on"] = [d for d in node["depends_on"] if d in valid_ids]

    return dag


def _fallback_dag(prompt: str) -> list[ThoughtNode]:
    return [
        ThoughtNode(
            id="direct_answer",
            question=(
                "Solve the following problem. Output ONLY the final answer "
                "with no explanation.\n\n" + prompt
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
    ]


_SOLVER_TOOLS = {
    "gravity_physics": "solve_gravity_physics",
    "unit_conversion": "solve_unit_conversion",
    "numeral_conversion": "solve_numeral_conversion",
    "cipher_decryption": "solve_cipher_decryption",
    "bit_manipulation": "solve_bit_manipulation",
}


def _solver_dag(prompt: str, puzzle_type: str) -> list[ThoughtNode]:
    """Single-node DAG that calls the type-specific solver tool."""
    tool_name = _SOLVER_TOOLS[puzzle_type]
    return [
        ThoughtNode(
            id="solve",
            question=f"Solve {puzzle_type} puzzle",
            depends_on=[],
            tool=tool_name,
            tool_input=json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


_EQ_TRANSFORM_PROMPT = """\
Analyze this puzzle step by step. The examples show a transformation rule applied to expressions.

IMPORTANT INSTRUCTIONS:
1. The middle character of each expression is likely the OPERATOR
2. Group examples by their operator character
3. For each operator, figure out EXACTLY what it does to the left and right parts
4. Consider: character substitution, ASCII math, concatenation, deletion, pairwise operations
5. Apply the discovered rule to the target expression
6. Output ONLY the final answer string on the last line, nothing else

Puzzle:
{prompt}

Think step by step. What is the operator in the target? Which example(s) use the same operator? What does that operator do? Apply it and give ONLY the final answer on the last line."""


def make_decompose_node(model_name: str, base_url: str):
    """Factory that returns the decompose node function with the LLM client bound."""
    client = ollama.Client(host=base_url)

    def decompose_node(state: GraphState) -> dict:
        prompt = state["prompt"]
        puzzle_type = state.get("puzzle_type") or "unknown"
        failure_log: list[FailureRecord] = state.get("failure_log") or []

        # For types with deterministic solvers, skip LLM decompose entirely
        if puzzle_type in _SOLVER_TOOLS and not failure_log:
            return {"thought_dag": _solver_dag(prompt, puzzle_type)}

        # For equation_transform, use a specialized direct prompt
        if puzzle_type == "equation_transform":
            return {"thought_dag": [
                ThoughtNode(
                    id="solve",
                    question=_EQ_TRANSFORM_PROMPT.format(prompt=prompt),
                    depends_on=[],
                    tool="ask_llm",
                    tool_input=None,
                    answer=None,
                ),
            ]}

        # On solver failure (retry), or for unknown types, use LLM decompose
        user_msg = _build_user_prompt(prompt, puzzle_type, failure_log)

        try:
            resp = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                think=False,
                options={"temperature": 0.3, "num_predict": 2048},
            )
            raw = resp.message.content or ""
            dag = _parse_dag_json(raw)
        except Exception:
            dag = _fallback_dag(prompt)

        return {"thought_dag": dag}

    return decompose_node
