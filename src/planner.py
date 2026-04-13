"""Unified LLM-based query planner for all six puzzle types.

A single system prompt (``PLANNER_SYSTEM``) describes every puzzle type,
its composable tools, and the expected DAG topology.  The caller passes
``puzzle_type`` as the intent; the LLM reads it and generates a
mermaid DAG + JSON nodes.

The generic ``_build_dag`` builder converts the parsed LLM output
directly into ``ThoughtNode`` objects — no per-type hardcoded builders.
``depends_on`` is derived from the mermaid edges, and a ``__PROMPT__``
placeholder in ``tool_input`` is substituted with the actual prompt.
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

=== numeral_conversion (1 node) ===
N1  id=solve_numeral   tool=solve_numeral_conversion
    tool_input = {"prompt": "__PROMPT__"}

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

=== bit_manipulation (1 node) ===
N1  id=solve_bits      tool=solve_bit_manipulation
    tool_input = {"prompt": "__PROMPT__"}

=== equation_transform (1 node) ===
N1  id=solve_equation  tool=solve_equation_transform
    tool_input = {"prompt": "__PROMPT__"}

RULES:
- Match the plan to PUZZLE_TYPE exactly.
- Use the EXACT tool names and node IDs shown above.
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


# ── Planner output parser ───────────────────────────────────────────

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
        parsed, _ = json.JSONDecoder().raw_decode(repaired)

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
        depends_on = [
            nkey_to_id[p] for p in parent_map[nk] if p in nkey_to_id
        ]

        tool_input = node.get("tool_input", "")
        tool_input = tool_input.replace('"__PROMPT__"', prompt_json)

        dag.append(ThoughtNode(
            id=node["id"],
            question=node.get("question", ""),
            depends_on=depends_on,
            tool=node.get("tool"),
            tool_input=tool_input,
            answer=None,
        ))

    return dag


# ── QueryPlanner class ──────────────────────────────────────────────

class QueryPlanner:
    """Unified LLM-based query planner.

    ``plan(puzzle_type, prompt)`` sends the unified ``PLANNER_SYSTEM``
    prompt with ``PUZZLE_TYPE: <type>`` in the user message.  The LLM
    generates MERMAID + NODES which are parsed and converted into
    ``ThoughtNode`` objects by the generic ``_build_dag`` builder.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, puzzle_type: str, prompt: str) -> list[ThoughtNode]:
        """Generate a DAG for *puzzle_type* by asking the LLM.

        Parameters
        ----------
        puzzle_type : str
            One of the six known types (e.g. ``"gravity_physics"``).
        prompt : str
            The full puzzle prompt text.

        Returns
        -------
        list[ThoughtNode]
            Ready-to-execute DAG nodes.
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
