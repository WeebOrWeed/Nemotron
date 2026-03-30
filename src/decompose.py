"""Decompose node: on first pass, uses the DAG already built by classify.
On retries (solver failure), generates a new plan via LLM or majority-vote.
"""
from __future__ import annotations

import json
import re

import ollama

from src.state import GraphState, ThoughtNode, FailureRecord

# ── LLM decompose prompt (used only on retries / unknown types) ────────

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


def make_decompose_node(model_name: str, base_url: str):
    """Factory: returns the decompose node with the LLM client bound."""
    client = ollama.Client(host=base_url)

    def decompose_node(state: GraphState) -> dict:
        prompt = state["prompt"]
        puzzle_type = state.get("puzzle_type") or "unknown"
        failure_log: list[FailureRecord] = state.get("failure_log") or []
        existing_dag = state.get("thought_dag")

        # ── first pass: classifier already built the DAG ───────────
        if existing_dag and not failure_log:
            return {"thought_dag": existing_dag}

        # ── retry for equation_transform: majority-vote LLM ───────
        if puzzle_type == "equation_transform" and failure_log:
            eq_question = (
                f"{prompt}\n\n"
                "Look at the examples above. Each 5-character input has an "
                "operator at position 2 (like * or -).\n"
                "The target uses the same type of operator as one of the "
                "examples.\nFind the pattern and apply it.\n\n"
                "Answer with ONLY the result string, nothing else."
            )
            return {"thought_dag": [
                ThoughtNode(
                    id="solve",
                    question=eq_question,
                    depends_on=[],
                    tool="majority_vote_llm",
                    tool_input="7",
                    answer=None,
                ),
            ]}

        # ── retry / unknown: ask LLM to generate a new DAG ────────
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
