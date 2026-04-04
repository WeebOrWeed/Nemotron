"""Decompose node: on first pass, uses the DAG already built by classify.
On retries (solver failure), generates a new plan via LLM or majority-vote.
"""
from __future__ import annotations

import json
import re

from src.state import GraphState, ThoughtNode, FailureRecord
from src.llm_client import LLMClient

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

TOOLS: ask_llm, eval_math, apply_formula, round_number, average, divide_sum_n_json, gravity_geom_mean_chain_exprs, regex_extract,
xor_binary, and_binary, or_binary, not_binary, shift_left, shift_right,
rotate_left, rotate_right, substitute_chars, build_char_map, to_roman,
from_roman, linear_factor, compute_gravity_g, compute_gravity_d.

=== EXAMPLE FOR gravity_physics ===
Prompt: "...For t = 1.37s, distance = 14.92 m\\nFor t = 4.27s, distance = 144.96 m...determine the falling distance for t = 4.41s..."
[
  {"id": "extract_spec", "question": "Extract JSON: {\\\"pairs\\\": [[t,d],...], \\\"gravitation_function\\\": \\\"d = 0.5*g*t^2\\\", \\\"query_t\\\": number}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "fill_equations", "question": "From:\\n{extract_spec}\\n\\nOne equation per pair, numbers substituted (e.g. 14.92 = 0.5*g*(1.37)^2).", "depends_on": ["extract_spec"], "tool": "ask_llm", "tool_input": null},
  {"id": "expand_products", "question": "Expand t^2 to t*t only; no parens; no numeric multiply: d = 0.5*g*t*t per line:\\n{fill_equations}", "depends_on": ["fill_equations"], "tool": "ask_llm", "tool_input": null},
  {"id": "g_unevaluated", "question": "g = d/0.5/t/t chains, unevaluated, no parens:\\n{expand_products}", "depends_on": ["expand_products"], "tool": "ask_llm", "tool_input": null},
  {"id": "d_per_pair_symbolic", "question": "Spec:\\n{extract_spec}\\n\\n{g_unevaluated}\\n\\nSymbolic d_i at query_t per pair; no decimals.", "depends_on": ["g_unevaluated", "extract_spec"], "tool": "ask_llm", "tool_input": null},
  {"id": "d_i_representations", "question": "Echo same count of d_k lines as above, * / and decimals only, no g:\\n{d_per_pair_symbolic}", "depends_on": ["d_per_pair_symbolic"], "tool": "ask_llm", "tool_input": null},
  {"id": "eval_geom_mean_d", "question": "geometric mean (product then n-th root) via tool", "depends_on": ["d_i_representations"], "tool": "gravity_geom_mean_chain_exprs", "tool_input": "{d_i_representations}"}
]

=== EXAMPLE FOR unit_conversion ===
Prompt: "...3.2 m becomes 10.5 ft\\n5.1 m becomes 16.77 ft...convert measurement: 2.0 m"
[
  {"id": "extract_pairs", "question": "Extract ALL example conversions as [from, to] number pairs and the target value to convert. Output ONLY JSON: {\\\"pairs\\\": [[from1,to1],...], \\\"target\\\": <number>}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "compute_factor", "question": "Given:\\n{extract_pairs}\\n\\nFor each pair compute factor = to/from, then average. Output ONLY the average factor on the last line.", "depends_on": ["extract_pairs"], "tool": "ask_llm", "tool_input": null},
  {"id": "apply_convert", "question": "factor = {compute_factor}\\nContext: {extract_pairs}\\n\\nCompute factor * target from JSON. Match example precision. Output ONLY the number.", "depends_on": ["compute_factor", "extract_pairs"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR numeral_conversion ===
Prompt: "...numeral system...12 is written as XII...express the number 44..."
[
  {"id": "extract_data", "question": "Extract examples [[decimal_or_label, notation], ...] and target_decimal. Output ONLY JSON: {\\\"examples\\\": [[\\\"12\\\",\\\"XII\\\"]], \\\"target_decimal\\\": 44}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "infer_system", "question": "Data:\\n{extract_data}\\n\\nWhich numeral system? Rules from examples. End with TARGET_DECIMAL=n.", "depends_on": ["extract_data"], "tool": "ask_llm", "tool_input": null},
  {"id": "express_target", "question": "Data:\\n{extract_data}\\n\\n{infer_system}\\n\\nOutput ONLY the target in that notation on the last line.", "depends_on": ["infer_system", "extract_data"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR cipher_decryption ===
Prompt: "...encryption rules...ucoov pwgtfyoqg -> queen discovers...decrypt: trb wzrswvog"
[
  {"id": "extract_word_pairs", "question": "List aligned [encrypted_word, plain_word] from every example. Output ONLY JSON: {\\\"pairs\\\": [[\\\"ucoov\\\",\\\"queen\\\"], ...]}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "extract_cipher", "question": "What exact ciphertext must be decrypted? Output ONLY that string on the last line.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "decrypt", "question": "Pairs:\\n{extract_word_pairs}\\n\\nCipher:\\n{extract_cipher}\\n\\nDecrypt; output ONLY plaintext on the last line.", "depends_on": ["extract_word_pairs", "extract_cipher"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR bit_manipulation ===
Prompt: "...bit manipulation rule transforms 8-bit binary...01010001 -> 11011101...determine output for 00110100"
[
  {"id": "identify_rule", "question": "Analyze ALL input->output pairs. Try XOR, NOT, shifts, rotations. Which single operation transforms every input to its output? State the operation and any constant. Show your work for each pair.", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "apply_rule", "question": "Using the rule: {identify_rule}\\n\\nApply it to the target input from the puzzle. Output ONLY the 8-bit binary result.", "depends_on": ["identify_rule"], "tool": "ask_llm", "tool_input": null}
]

=== EXAMPLE FOR equation_transform ===
Prompt: "...transformation rules...12+34 = 46...determine the result for: 90+12"
[
  {"id": "extract_examples", "question": "Extract each example as [left5, result] and the 5-char target. Output ONLY JSON: {\\\"examples\\\": [[\\\"12+34\\\",\\\"46\\\"]], \\\"target\\\": \\\"90+12\\\"}", "depends_on": [], "tool": "ask_llm", "tool_input": null},
  {"id": "infer_rule", "question": "Data:\\n{extract_examples}\\n\\nGroup by operator at index 2; infer consistent (LEFT,RIGHT)→result. End with RULE = ...", "depends_on": ["extract_examples"], "tool": "ask_llm", "tool_input": null},
  {"id": "apply_rule", "question": "Data:\\n{extract_examples}\\n\\n{infer_rule}\\n\\nApply to target; output ONLY result on last line.", "depends_on": ["infer_rule", "extract_examples"], "tool": "ask_llm", "tool_input": null}
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


def make_decompose_node(llm: LLMClient):
    """Factory: returns the decompose node with the LLM client bound."""

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
            resp = llm.chat(
                [
                    {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                think=False,
                temperature=0.3,
                max_tokens=2048,
            )
            raw = resp.content
            dag = _parse_dag_json(raw)
        except Exception:
            dag = _fallback_dag(prompt)

        return {"thought_dag": dag}

    return decompose_node
