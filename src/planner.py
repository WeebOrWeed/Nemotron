"""LLM-based query planner for all six puzzle types.

``QueryPlanner.plan(puzzle_type, prompt)`` sends a unified system prompt
(``PLANNER_SYSTEM``) to the LLM which composes a DAG from the tool
catalogue.  The generic ``_build_dag`` builder converts the parsed
MERMAID + NODES output into ``ThoughtNode`` objects.

The LLM is the sole DAG builder — no hardcoded per-type fallbacks.
"""
from __future__ import annotations

import json
import re

from src.state import ThoughtNode
from src.llm_client import LLMClient


# ── Unified planner system prompt ────────────────────────────────────

PLANNER_SYSTEM = """\
You are a unified DAG planner for reasoning puzzles.  You receive a
PUZZLE_TYPE and a PROMPT, and output an execution plan.

OUTPUT FORMAT (strictly two sections, nothing else):

MERMAID:
START --> N1
N1 --> N2
...
Nk --> END

NODES:
{"N1": {"id": "...", "question": "...", "tool": "...", "tool_input": "..."}, ...}

FIELD RULES:
- id        : unique snake_case identifier for the node.
- question  : one-sentence description of what the node does.
- tool      : exact tool function name from the catalogue below.
- tool_input: a JSON *string*.  Use "__PROMPT__" (with quotes) wherever
              the full puzzle prompt text is needed — it will be replaced
              at runtime.  Use {parent_id} for dependency interpolation
              and {parent_id_field} for JSON sub-field access.
- depends_on is NOT in NODES — it is derived from the MERMAID edges.

TOOL CATALOGUE AND PLANS PER PUZZLE TYPE:

=== gravity_physics (3 nodes) ===
N1  id=extract_obs     tool=extract_gravity_obs
    tool_input = {"prompt": "__PROMPT__"}
N2  id=compute_g       tool=compute_gravity_g       depends on N1
    tool_input = {extract_obs}
N3  id=predict_d       tool=compute_gravity_d       depends on N1, N2
    tool_input = {"g": "{compute_g}", "t": "{extract_obs_target_t}"}

=== unit_conversion (3 nodes) ===
N1  id=extract_pairs   tool=extract_unit_pairs
    tool_input = {"prompt": "__PROMPT__"}
N2  id=compute_factor  tool=geometric_mean_factor   depends on N1
    tool_input = {extract_pairs}
N3  id=predict         tool=apply_factor_round      depends on N1, N2
    tool_input = {"factor": "{compute_factor}", "target": "{extract_pairs_target}"}

=== numeral_conversion (2 nodes) ===
N1  id=detect_system   tool=detect_numeral_system
    tool_input = {"pairs": [[<decimal_from_example_1>, "<notation_from_example_1>"], ...]}
N2  id=convert_target  tool=convert_numeral        depends on N1
    tool_input = {"number": <target_decimal>, "system": "{detect_system_system}"}

=== cipher_decryption (2*N + 2 nodes) ===
For EACH example line i (i = 1..N) in the prompt:
  N(2i-1) id=extract_pairs_i   tool=split_word_pairs     root node
          tool_input = {"encrypted": "<line i encrypted text>", "plaintext": "<line i plain text>"}
  N(2i)   id=create_mapping_i  tool=build_char_map       depends on extract_pairs_i
          tool_input = {extract_pairs_i}
Then two final nodes:
  merge_mapping   tool=merge_char_maps        depends on ALL create_mapping_i
          tool_input = newline-joined {create_mapping_i} refs
  translate       tool=decrypt_substitution   depends on merge_mapping
          tool_input = {"ciphertext": "<target ciphertext>", "mapping": {merge_mapping}}
Read the actual encrypted/plain/ciphertext from the PROMPT.

=== bit_manipulation (variable strategy DAG, 5+ nodes) ===
Always start by extracting the bit task, then choose MULTIPLE independent
strategy nodes. Do not collapse the strategies into one generator node.

Required root:
  id=extract_bits      tool=extract_bit_task
      tool_input = {"prompt": "__PROMPT__"}

Available strategy nodes (choose 2-6 based on the prompt):
  id=try_byte          tool=try_byte_ops_bit_rule           depends on extract_bits
      tool_input = {extract_bits}
  id=try_gf2           tool=try_gf2_affine_bit_rule         depends on extract_bits
      tool_input = {extract_bits}
  id=try_bruteforce    tool=try_per_bit_bruteforce_rule     depends on extract_bits
      tool_input = {extract_bits}
  id=try_tt3_input     tool=try_shifted_truth_table_rule    depends on extract_bits
      tool_input = {"task": {extract_bits}, "arity": 3, "unknown_policy": "input"}
  id=try_tt2_majority  tool=try_shifted_truth_table_rule    depends on extract_bits
      tool_input = {"task": {extract_bits}, "arity": 2, "unknown_policy": "majority"}
  id=try_tt4_input     tool=try_shifted_truth_table_rule    depends on extract_bits
      tool_input = {"task": {extract_bits}, "arity": 4, "unknown_policy": "input"}

Required final nodes:
  id=select_bits       tool=select_bit_strategy_candidate   depends on ALL chosen strategy nodes and extract_bits
      tool_input = {"candidates": [{try_byte}, {try_gf2}, ...], "bits": "{extract_bits_bits}"}
  id=normalize_bits    tool=normalize_binary_answer         depends on select_bits and extract_bits
      tool_input = {"answer": "{select_bits}", "bits": "{extract_bits_bits}"}

=== equation_transform (1 node, no deterministic type solver) ===
N1  id=solve_equation  tool=ask_llm
    tool_input = "1"


=== bit_manipulation (variable strategy example, reward=1.0) ===
Input: PUZZLE_TYPE: bit_manipulation
Output:
MERMAID:
START --> extract_bits
extract_bits --> try_byte
extract_bits --> try_gf2
extract_bits --> try_tt3_input
try_byte --> select_bits
try_gf2 --> select_bits
try_tt3_input --> select_bits
extract_bits --> select_bits
select_bits --> normalize_bits
extract_bits --> normalize_bits
normalize_bits --> END

NODES:
{"extract_bits": {"id": "extract_bits", "question": "Extract bit examples and target from the prompt.", "tool": "extract_bit_task", "tool_input": "{\"prompt\": \"__PROMPT__\"}"}, "try_byte": {"id": "try_byte", "question": "Try whole-byte transformations on the extracted examples.", "tool": "try_byte_ops_bit_rule", "tool_input": "{extract_bits}"}, "try_gf2": {"id": "try_gf2", "question": "Try an affine GF(2) transformation.", "tool": "try_gf2_affine_bit_rule", "tool_input": "{extract_bits}"}, "try_tt3_input": {"id": "try_tt3_input", "question": "Try a shifted arity-3 truth table with input fallback.", "tool": "try_shifted_truth_table_rule", "tool_input": "{\"task\": {extract_bits}, \"arity\": 3, \"unknown_policy\": \"input\"}"}, "select_bits": {"id": "select_bits", "question": "Select the best answer from the independent strategy candidates.", "tool": "select_bit_strategy_candidate", "tool_input": "{\"candidates\": [{try_byte}, {try_gf2}, {try_tt3_input}], \"bits\": \"{extract_bits_bits}\"}"}, "normalize_bits": {"id": "normalize_bits", "question": "Normalize the selected binary answer.", "tool": "normalize_binary_answer", "tool_input": "{\"answer\": \"{select_bits}\", \"bits\": \"{extract_bits_bits}\"}"}}


=== bit_manipulation (variable strategy example, reward=1.0) ===
Input: PUZZLE_TYPE: bit_manipulation
Output:
MERMAID:
START --> extract_bits
extract_bits --> try_gf2
extract_bits --> try_bruteforce
extract_bits --> try_tt2_majority
extract_bits --> try_tt4_input
try_gf2 --> select_bits
try_bruteforce --> select_bits
try_tt2_majority --> select_bits
try_tt4_input --> select_bits
extract_bits --> select_bits
select_bits --> normalize_bits
extract_bits --> normalize_bits
normalize_bits --> END

NODES:
{"extract_bits": {"id": "extract_bits", "question": "Extract bit examples and target from the prompt.", "tool": "extract_bit_task", "tool_input": "{\"prompt\": \"__PROMPT__\"}"}, "try_gf2": {"id": "try_gf2", "question": "Try an affine GF(2) transformation.", "tool": "try_gf2_affine_bit_rule", "tool_input": "{extract_bits}"}, "try_bruteforce": {"id": "try_bruteforce", "question": "Try per-bit brute-force rules.", "tool": "try_per_bit_bruteforce_rule", "tool_input": "{extract_bits}"}, "try_tt2_majority": {"id": "try_tt2_majority", "question": "Try a shifted arity-2 truth table with majority fallback.", "tool": "try_shifted_truth_table_rule", "tool_input": "{\"task\": {extract_bits}, \"arity\": 2, \"unknown_policy\": \"majority\"}"}, "try_tt4_input": {"id": "try_tt4_input", "question": "Try a shifted arity-4 truth table with input fallback.", "tool": "try_shifted_truth_table_rule", "tool_input": "{\"task\": {extract_bits}, \"arity\": 4, \"unknown_policy\": \"input\"}"}, "select_bits": {"id": "select_bits", "question": "Select the best answer from the independent strategy candidates.", "tool": "select_bit_strategy_candidate", "tool_input": "{\"candidates\": [{try_gf2}, {try_bruteforce}, {try_tt2_majority}, {try_tt4_input}], \"bits\": \"{extract_bits_bits}\"}"}, "normalize_bits": {"id": "normalize_bits", "question": "Normalize the selected binary answer.", "tool": "normalize_binary_answer", "tool_input": "{\"answer\": \"{select_bits}\", \"bits\": \"{extract_bits_bits}\"}"}}



RULES:
- Match the plan to PUZZLE_TYPE exactly.
- Use exact tool names from the catalogue. For bit_manipulation, choose a
  variable set of independent strategy nodes; do not always emit the same DAG.
- Never use type-specific deterministic solver tools such as solve_bit_manipulation,
  solve_composite_bit_rule, solve_numeral_conversion, solve_cipher_decryption, or
  solve_equation_transform. Plans must be composed from smaller strategy tools or ask_llm.
- Output ONLY MERMAID and NODES sections — no commentary.\
"""


KNOWN_TYPES = frozenset({
    "gravity_physics",
    "unit_conversion",
    "numeral_conversion",
    "cipher_decryption",
    "bit_manipulation",
    "equation_transform",
})

FORBIDDEN_MONOLITHIC_TOOLS = frozenset({
    "solve_bit_manipulation",
    "solve_composite_bit_rule",
    "solve_numeral_conversion",
    "solve_cipher_decryption",
    "solve_equation_transform",
    "generate_bit_rule_candidates",
    "select_bit_candidate",
})


# ── Planner output parser ───────────────────────────────────────────

def _parse_nodes_loose(json_blob: str) -> dict[str, dict]:
    """Best-effort parser for nearly-valid NODES output.

    Local SFT adapters sometimes learn the right node shape but emit
    ``tool_input`` as an unescaped JSON string, e.g.
    ``"tool_input": "{"prompt": "..."}"``. Strict JSON parsing fails there,
    but the node metadata is still recoverable.
    """
    node_key_match = re.search(r'"([^"]+)"\s*:\s*\{', json_blob)
    node_id_match = re.search(r'"id"\s*:\s*"([^"]+)"', json_blob)
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', json_blob)
    question_match = re.search(r'"question"\s*:\s*"([^"]*)"', json_blob)

    if not node_id_match or not tool_match:
        raise ValueError("Could not recover node metadata from malformed NODES JSON")

    node_id = node_id_match.group(1)
    tool = tool_match.group(1)
    node_key = node_key_match.group(1) if node_key_match else node_id
    question = question_match.group(1) if question_match else ""

    tool_input = ""
    if "__PROMPT__" in json_blob or "PROMPT:" in json_blob:
        tool_input = '{"prompt": "__PROMPT__"}'
    else:
        tool_input_match = re.search(
            r'"tool_input"\s*:\s*("(?:\\.|[^"\\])*"|\{.*?\})',
            json_blob,
            re.DOTALL,
        )
        if tool_input_match:
            tool_input = tool_input_match.group(1)
            if tool_input.startswith('"') and tool_input.endswith('"'):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    pass
    if not tool_input and tool == "solve_bit_manipulation":
        tool_input = '{"prompt": "__PROMPT__"}'

    return {
        node_key: {
            "id": node_id,
            "question": question,
            "tool": tool,
            "tool_input": tool_input,
        }
    }


def _parse_planner_output(
    raw: str,
) -> tuple[list[tuple[str, str]], dict[str, dict]]:
    """Parse LLM planner output into (mermaid_edges, nodes_dict).

    Handles variations in section headers (``MERMAID:``, ``**MERMAID**``,
    etc.) and NODES as either a ``{"N1": {...}}`` dict or a
    ``[{"id": "...", ...}]`` array.

    Returns
    -------
    edges : list of (src, dst) tuples, e.g. [("N1", "N2"), ...]
    nodes_dict : {"N1": {"id": "extract_spec", ...}, ...}
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    mermaid_match = re.search(
        r"\*{0,2}MERMAID\*{0,2}:?\s*\n(?:```[^\n]*\n)?(.*?)(?=\n\*{0,2}NODES\*{0,2}|\n```\s*\n\*{0,2}NODES\*{0,2}|\Z)",
        raw, re.DOTALL,
    )
    nodes_match = re.search(
        r"\*{0,2}NODES\*{0,2}:?\s*\n?(.*)", raw, re.DOTALL,
    )

    if not mermaid_match or not nodes_match:
        raise ValueError(
            "Could not find MERMAID and NODES sections in planner output"
        )

    # --- edges (accept any alphanumeric/underscore node names) ---
    edges: list[tuple[str, str]] = []
    for line in mermaid_match.group(1).strip().splitlines():
        m = re.match(r"\s*([\w]+)\s*-->\s*([\w]+)", line.strip())
        if m:
            edges.append((m.group(1), m.group(2)))

    # --- nodes (dict or array) ---
    nodes_text = nodes_match.group(1).strip()
    for fence in ("```json", "```"):
        nodes_text = nodes_text.replace(fence, "")
    nodes_text = nodes_text.strip()

    json_start = min(
        (nodes_text.find(c) for c in ("{", "[") if nodes_text.find(c) >= 0),
        default=-1,
    )
    if json_start < 0:
        raise ValueError("No JSON found in NODES section")

    json_blob = nodes_text[json_start:]
    try:
        parsed, _ = json.JSONDecoder().raw_decode(json_blob)
    except json.JSONDecodeError:
        open_b = json_blob.count("{") - json_blob.count("}")
        open_s = json_blob.count("[") - json_blob.count("]")
        repaired = json_blob + ("}" * max(open_b, 0)) + ("]" * max(open_s, 0))
        try:
            parsed, _ = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError:
            parsed = _parse_nodes_loose(json_blob)

    if isinstance(parsed, list):
        nodes_dict: dict[str, dict] = {}
        for i, entry in enumerate(parsed, 1):
            n_key = entry.pop("n_key", None) or f"N{i}"
            nodes_dict[n_key] = entry
    elif isinstance(parsed, dict):
        nodes_dict = parsed
    else:
        raise ValueError(f"Unexpected JSON type in NODES: {type(parsed)}")

    if not edges:
        raise ValueError("No edges found in MERMAID section")
    if not nodes_dict:
        raise ValueError("Empty NODES dict")

    return edges, nodes_dict


# ── Generic DAG builder ─────────────────────────────────────────────

def _build_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Convert parsed LLM output into ThoughtNodes.

    * ``depends_on`` is derived from the mermaid edges.
    * Node keys can be ``N1``-style or semantic names (``extract_obs``).
    * Nodes are topologically sorted via Kahn's algorithm.
    * ``"__PROMPT__"`` in ``tool_input`` is replaced with the actual
      prompt text (JSON-escaped).
    """
    from collections import deque

    parent_map: dict[str, list[str]] = {nk: [] for nk in nodes_dict}
    child_map: dict[str, list[str]] = {nk: [] for nk in nodes_dict}
    for src, dst in edges:
        if dst in nodes_dict and src in nodes_dict:
            parent_map[dst].append(src)
            child_map[src].append(dst)

    nkey_to_id = {nk: node["id"] for nk, node in nodes_dict.items()}

    # Topological sort (Kahn's algorithm)
    in_degree = {nk: len(parent_map[nk]) for nk in nodes_dict}
    queue = deque(nk for nk, d in in_degree.items() if d == 0)
    ordered: list[str] = []
    while queue:
        nk = queue.popleft()
        ordered.append(nk)
        for child in child_map.get(nk, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    # Append any remaining nodes not reached by edges
    for nk in nodes_dict:
        if nk not in ordered:
            ordered.append(nk)

    prompt_json = json.dumps(prompt)

    dag: list[ThoughtNode] = []
    for nk in ordered:
        node = nodes_dict[nk]
        tool_name = node.get("tool")
        if tool_name in FORBIDDEN_MONOLITHIC_TOOLS:
            raise ValueError(f"Forbidden monolithic tool in DAG: {tool_name}")
        depends_on = [
            nkey_to_id[p] for p in parent_map[nk] if p in nkey_to_id
        ]

        tool_input = node.get("tool_input", "")
        tool_input = tool_input.replace('"__PROMPT__"', prompt_json)

        dag.append(ThoughtNode(
            id=node["id"],
            question=node.get("question", ""),
            depends_on=depends_on,
            tool=tool_name,
            tool_input=tool_input,
            answer=None,
        ))

    return dag


# ── QueryPlanner class ──────────────────────────────────────────────

class QueryPlanner:
    """LLM-based query planner.

    ``plan(puzzle_type, prompt)`` sends the unified ``PLANNER_SYSTEM``
    prompt with ``PUZZLE_TYPE: <type>`` in the user message.  The LLM
    composes a DAG from the tool catalogue, outputting MERMAID edges and
    JSON node definitions which are parsed into ``ThoughtNode`` objects.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, puzzle_type: str, prompt: str) -> list[ThoughtNode]:
        """Generate a DAG for *puzzle_type* via LLM.

        The LLM selects composable tools from the catalogue and wires
        them into a DAG.  No hardcoded per-type logic.
        """
        if puzzle_type not in KNOWN_TYPES:
            raise ValueError(f"Unknown puzzle type: {puzzle_type}")

        resp = self.llm.chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {
                    "role": "user",
                    "content": f"PUZZLE_TYPE: {puzzle_type}\n\nPROMPT:\n{prompt}",
                },
            ],
            think=False,
            temperature=0.3,
            max_tokens=8192,
        )
        raw = (resp.content or "").strip()
        if not raw:
            raise ValueError("Empty planner response")

        edges, nodes_dict = _parse_planner_output(raw)
        return _build_dag(edges, nodes_dict, prompt)
