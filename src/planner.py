"""LLM-based query planner for gravity, unit, numeral, and cipher puzzles.

The planner calls the LLM to produce a structured execution plan with two parts:
  1. A mermaid-style directed graph (N1 --> N2, etc.) defining the topology
  2. A JSON dict mapping each N-key to a ThoughtNode-shaped object
     (id, question, tool, tool_input) — depends_on is derived from node IDs

Dependencies and tool assignments are enforced by type-specific rules
(not the LLM's mermaid) to guarantee correct parallel topology.
Question text always comes from deterministic templates.  If the planner
output is entirely unparseable, the error propagates.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from src.state import ThoughtNode
from src.llm_client import LLMClient


# ── Gravity planner LLM prompt ───────────────────────────────────────

GRAVITY_PLANNER_SYSTEM = """\
You are a query planner for gravity physics puzzles.

Read the puzzle and produce an execution plan with two sections.

OUTPUT FORMAT (output ONLY these two sections, nothing else):

MERMAID:
START --> N1
N1 --> N2
N1 --> N3
...
Nk --> END

NODES:
Each value is a ThoughtNode with id, question, tool, and tool_input.
Use {parent_id} placeholders in question/tool_input to reference a parent node's answer at runtime.
{"N1": {"id": "...", "question": "...", "tool": "...", "tool_input": null}, ...}

RULES:
- N1 = extract_spec (tool: ask_llm). Extracts pairs, formula, query_t as JSON.
- For EACH observation pair i, create a chain of 5 nodes:
  fill_eq_i -> expand_i -> g_i -> d_symbolic_i -> d_repr_i
  All fill_eq nodes depend on N1 (extract_spec).
- Last node = eval_geom_mean_d (tool: gravity_geom_mean_chain_exprs).
  It depends on ALL d_repr nodes.  tool_input = all {d_repr_i} joined by newlines.
- fill_eq_i question: state the formula and pair values, ask to substitute.
- expand_i question: reference {fill_eq_i}, ask to replace t^2 with t*t.
- g_i question: reference {expand_i}, ask to isolate g as division chain.
- d_symbolic_i question: reference {g_i} and query_t, derive d_i symbolically.
- d_repr_i question: reference {d_symbolic_i}, output clean d_i = <decimals * / only>.
- If the formula is not d = 0.5*g*t^2, include it in extract_spec question.

EXAMPLE (2 pairs: t=3.88/d=109.74 and t=4.27/d=144.96, query_t=4.41):

MERMAID:
START --> N1
N1 --> N2
N1 --> N3
N2 --> N4
N3 --> N5
N4 --> N6
N5 --> N7
N6 --> N8
N7 --> N9
N8 --> N10
N9 --> N11
N10 --> N12
N11 --> N12
N12 --> END

NODES:
{"N1": {"id": "extract_spec", "question": "Extract pairs, gravitation_function, query_t as JSON from the puzzle.", "tool": "ask_llm", "tool_input": null}, "N2": {"id": "fill_eq_1", "question": "Formula: d = 0.5*g*t^2. Pair 1: t=3.88, d=109.74. Substitute so g is unknown. Output one equation.", "tool": "ask_llm", "tool_input": null}, "N3": {"id": "fill_eq_2", "question": "Formula: d = 0.5*g*t^2. Pair 2: t=4.27, d=144.96. Substitute so g is unknown. Output one equation.", "tool": "ask_llm", "tool_input": null}, "N4": {"id": "expand_1", "question": "Equation:\\n{fill_eq_1}\\n\\nReplace t^2 with t*t. No parentheses. Do not multiply factors.", "tool": "ask_llm", "tool_input": null}, "N5": {"id": "expand_2", "question": "Equation:\\n{fill_eq_2}\\n\\nReplace t^2 with t*t. No parentheses. Do not multiply factors.", "tool": "ask_llm", "tool_input": null}, "N6": {"id": "g_1", "question": "Equation:\\n{expand_1}\\n\\nIsolate g as a division chain: g = d / 0.5 / t / t. One line only.", "tool": "ask_llm", "tool_input": null}, "N7": {"id": "g_2", "question": "Equation:\\n{expand_2}\\n\\nIsolate g as a division chain: g = d / 0.5 / t / t. One line only.", "tool": "ask_llm", "tool_input": null}, "N8": {"id": "d_symbolic_1", "question": "g expression:\\n{g_1}\\n\\nquery_t=4.41. Derive d_1 = d_obs * query_t * query_t / t_obs / t_obs. Do not evaluate.", "tool": "ask_llm", "tool_input": null}, "N9": {"id": "d_symbolic_2", "question": "g expression:\\n{g_2}\\n\\nquery_t=4.41. Derive d_2 = d_obs * query_t * query_t / t_obs / t_obs. Do not evaluate.", "tool": "ask_llm", "tool_input": null}, "N10": {"id": "d_repr_1", "question": "Symbolic:\\n{d_symbolic_1}\\n\\nRe-output: d_1 = <decimals * / only>. No letters, no parentheses. One line.", "tool": "ask_llm", "tool_input": null}, "N11": {"id": "d_repr_2", "question": "Symbolic:\\n{d_symbolic_2}\\n\\nRe-output: d_2 = <decimals * / only>. No letters, no parentheses. One line.", "tool": "ask_llm", "tool_input": null}, "N12": {"id": "eval_geom_mean_d", "question": "Geometric mean of d_k values, rounded to 2dp.", "tool": "gravity_geom_mean_chain_exprs", "tool_input": "{d_repr_1}\\n{d_repr_2}"}}\
"""


# ── Question templates (keyed by node-id prefix) ────────────────────

def _gravity_question(
    node_id: str,
    meta: dict,
    prompt: str,
    func: str,
    query_t: float,
) -> str:
    """Generate the full question text for a gravity DAG node."""

    if node_id == "extract_spec":
        return (
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
        )

    m = re.match(r"fill_eq_(\d+)", node_id)
    if m:
        i = m.group(1)
        ts, ds = str(meta.get("t", "?")), str(meta.get("d", "?"))
        return (
            f"Gravitation formula: {func}\n"
            f"Observation pair {i}: t = {ts}, d = {ds}\n\n"
            f"Substitute this single observation into the formula so d "
            f"and t are numeric and g is still unknown.\n"
            f"Output ONLY one filled equation, nothing else."
        )

    m = re.match(r"expand_(\d+)", node_id)
    if m:
        i = m.group(1)
        ts, ds = str(meta.get("t", "?")), str(meta.get("d", "?"))
        fill_ph = "{fill_eq_" + i + "}"
        return (
            f"Filled equation:\n{fill_ph}\n\n"
            f"This step is expand only — do NOT multiply numeric "
            f"factors together. Replace t^2 with t*t using the "
            f"numeric t. No parentheses.\n"
            f"Output exactly like: {ds} = 0.5*g*{ts}*{ts}"
        )

    m = re.match(r"g_(\d+)", node_id)
    if m:
        i = m.group(1)
        ts, ds = str(meta.get("t", "?")), str(meta.get("d", "?"))
        expand_ph = "{expand_" + i + "}"
        return (
            f"Expanded equation:\n{expand_ph}\n\n"
            f"Isolate g by dividing both sides, writing g as a "
            f"chain of divisions with each factor separate:\n"
            f"g = {ds} / 0.5 / {ts} / {ts}\n"
            f"No parentheses. Do not compute numerically. "
            f"One g=... line only."
        )

    m = re.match(r"d_symbolic_(\d+)", node_id)
    if m:
        i = m.group(1)
        ts, ds = str(meta.get("t", "?")), str(meta.get("d", "?"))
        qts = str(query_t)
        g_ph = "{g_" + i + "}"
        return (
            f"g expression for observation {i}:\n{g_ph}\n\n"
            f"query_t = {qts}\n\n"
            f"Derive d_{i} at query_t from {func} by substituting "
            f"this observation's g expression. Write as a fully "
            f"expanded chain with no parentheses and no partial "
            f"products:\n"
            f"d_{i} = {ds} * {qts} * {qts} / {ts} / {ts}\n"
            f"Use the actual numeric values. Do not evaluate."
        )

    m = re.match(r"d_repr_(\d+)", node_id)
    if m:
        i = m.group(1)
        d_sym_ph = "{d_symbolic_" + i + "}"
        return (
            f"Symbolic distance:\n{d_sym_ph}\n\n"
            f"Re-output as exactly one line: d_{i} = <decimals with "
            f"only * and />\nNo parentheses, no powers, no letters "
            f"(no g on the right-hand side). Do not compute numeric "
            f"values. One line only, no commentary."
        )

    if node_id == "eval_geom_mean_d":
        return (
            "Python: parse d_k lines, multiply the evaluated d values "
            "together, then take the n-th root (geometric mean, n = line "
            "count). Return that value rounded to 2dp."
        )

    return f"Solve this step: {node_id}"


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

    # --- edges ---
    edges: list[tuple[str, str]] = []
    for line in mermaid_match.group(1).strip().splitlines():
        m = re.match(
            r"\s*(START|END|N\d+)\s*-->\s*(START|END|N\d+)", line.strip()
        )
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
        # Try to repair truncated / slightly malformed JSON by balancing braces
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


# ── DAG builder ─────────────────────────────────────────────────────

def _gravity_depends_on(node_id: str, d_repr_ids: list[str]) -> list[str]:
    """Derive correct depends_on for a gravity node from its ID.

    The topology for gravity is fully determined by the node naming
    convention, so we don't rely on the LLM's mermaid edges (which the
    small model frequently gets wrong).
    """
    if node_id == "extract_spec":
        return []
    m = re.match(r"fill_eq_(\d+)", node_id)
    if m:
        return ["extract_spec"]
    m = re.match(r"expand_(\d+)", node_id)
    if m:
        return [f"fill_eq_{m.group(1)}"]
    m = re.match(r"g_(\d+)", node_id)
    if m:
        return [f"expand_{m.group(1)}"]
    m = re.match(r"d_symbolic_(\d+)", node_id)
    if m:
        return [f"g_{m.group(1)}"]
    m = re.match(r"d_repr_(\d+)", node_id)
    if m:
        return [f"d_symbolic_{m.group(1)}"]
    if node_id == "eval_geom_mean_d":
        return list(d_repr_ids)
    return []


def _build_gravity_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Combine mermaid edges + ThoughtNode metadata into ThoughtNodes.

    Each entry in *nodes_dict* is already ThoughtNode-shaped (id, question,
    tool, tool_input).  If the LLM omitted ``question`` or ``tool_input``,
    we fall back to deterministic templates.

    Dependencies are derived from gravity naming rules (not the LLM mermaid)
    to guarantee correct parallel topology.
    """

    # Extract formula and query_t
    func = "d = 0.5*g*t^2"
    query_t = 0.0
    for meta in nodes_dict.values():
        if meta.get("id") == "extract_spec" and "formula" in meta:
            func = meta["formula"]
        if "query_t" in meta:
            query_t = float(meta["query_t"])
    if query_t == 0.0:
        qt_match = re.search(r"query_t\s*[=:]\s*(\d+(?:\.\d+)?)", str(nodes_dict))
        if qt_match:
            query_t = float(qt_match.group(1))

    # Collect pair data from fill_eq nodes so we can propagate t,d downstream.
    # Try explicit t/d fields first; fall back to extracting from question text
    # using a pattern that matches "t=X, d=Y" or "t=X ... d=Y" after "Pair".
    pair_data: dict[str, dict] = {}
    for meta in nodes_dict.values():
        nid = meta.get("id", "")
        m = re.match(r"fill_eq_(\d+)", nid)
        if m:
            t_val = meta.get("t")
            d_val = meta.get("d")
            if t_val is None or d_val is None:
                q = meta.get("question", "")
                pair_m = re.search(
                    r"[Pp]air\s*\d+\s*:\s*t\s*[=:]\s*(\d+(?:\.\d+)?)\s*[,;]\s*d\s*[=:]\s*(\d+(?:\.\d+)?)",
                    q,
                )
                if pair_m:
                    if t_val is None:
                        t_val = float(pair_m.group(1))
                    if d_val is None:
                        d_val = float(pair_m.group(2))
            pair_data[m.group(1)] = {"t": t_val, "d": d_val}

    # Pre-collect d_repr IDs for eval_geom_mean_d
    d_repr_ids = sorted(
        [meta["id"] for meta in nodes_dict.values()
         if meta.get("id", "").startswith("d_repr_")],
        key=lambda s: int(re.search(r"\d+", s).group()),
    )

    def _n_sort(key: str) -> int:
        m = re.match(r"N(\d+)", key)
        return int(m.group(1)) if m else 0

    dag: list[ThoughtNode] = []

    # Known tool overrides — the LLM sometimes assigns the wrong tool
    _TOOL_OVERRIDE: dict[str, str] = {
        "eval_geom_mean_d": "gravity_geom_mean_chain_exprs",
    }

    for n_key in sorted(nodes_dict, key=_n_sort):
        meta = nodes_dict[n_key]
        node_id = meta["id"]
        tool = _TOOL_OVERRIDE.get(node_id, meta.get("tool") or "ask_llm")

        depends_on = _gravity_depends_on(node_id, d_repr_ids)

        # Always use detailed templates for question text — they embed
        # concrete pair values and worked-example output that the small
        # execution model needs to follow symbolic manipulation correctly.
        enriched = dict(meta)
        m_idx = re.match(r"(?:fill_eq|expand|g|d_symbolic|d_repr)_(\d+)", node_id)
        if m_idx:
            pd = pair_data.get(m_idx.group(1), {})
            enriched.setdefault("t", pd.get("t"))
            enriched.setdefault("d", pd.get("d"))
        question = _gravity_question(node_id, enriched, prompt, func, query_t)

        tool_input = meta.get("tool_input")

        if node_id == "eval_geom_mean_d":
            if not tool_input:
                tool_input = "\n".join("{" + did + "}" for did in d_repr_ids)

        dag.append(ThoughtNode(
            id=node_id,
            question=question,
            depends_on=depends_on,
            tool=tool,
            tool_input=tool_input,
            answer=None,
        ))

    return dag


# ═══════════════════════════════════════════════════════════════════
#  UNIT CONVERSION planner
# ═══════════════════════════════════════════════════════════════════

UNIT_PLANNER_SYSTEM = """\
You are a DAG planner for unit-conversion puzzles.

A unit conversion puzzle provides example (from, to) pairs and asks you to
convert a target measurement.  The relationship may be multiplicative (y = a*x)
OR affine (y = a*x + b, e.g. Fahrenheit to Celsius).  We handle both via
least-squares linear regression.

Plan a DAG with EXACTLY these 3 nodes:

1. extract_pairs  (tool: ask_llm) — extract all [from, to] pairs and target.
2. linear_fit     (tool: linear_fit) — fit y = a*x + b to the pairs.
   tool_input = {"pairs": [{extract_pairs}]}  (interpolated at runtime).
3. apply_convert  (tool: eval_math) — compute result from slope, intercept, target.
   tool_input = {"expr": "round(<slope> * <target> + <intercept>, 2)"}.

Output EXACTLY two sections (no other text):

MERMAID:
START --> N1
N1 --> N2
N2 --> N3
N3 --> END

NODES:
{"N1": {"id": "extract_pairs", ...}, "N2": {"id": "linear_fit", ...}, "N3": {"id": "apply_convert", ...}}

RULES:
- N1 = extract_pairs (tool: ask_llm).
- N2 = linear_fit (tool: linear_fit). Depends on N1.
- N3 = apply_convert (tool: eval_math). Depends on N2.
- Use the actual target number from the puzzle.
- ALWAYS output exactly 3 nodes, nothing more.

EXAMPLE (3 pairs: 10.08→6.69, 17.83→11.83, 35.85→23.79, target=25.09):

MERMAID:
START --> N1
N1 --> N2
N2 --> N3
N3 --> END

NODES:
{"N1": {"id": "extract_pairs", "question": "Extract pairs and target as JSON.", "tool": "ask_llm", "tool_input": null}, "N2": {"id": "linear_fit", "question": "Fit y = ax + b to all pairs.", "tool": "linear_fit", "tool_input": "{\\"pairs\\": []}"}, "N3": {"id": "apply_convert", "question": "Compute result from slope, intercept, target.", "tool": "eval_math", "tool_input": "{\\"expr\\": \\"round(0 * 25.09 + 0, 2)\\"}"}}\
"""


def _extract_prompt_pairs(prompt: str) -> list[tuple[float, float]]:
    """Extract (from, to) pairs from a unit conversion prompt via regex."""
    return [
        (float(f), float(t))
        for f, t in re.findall(
            r"(\d+(?:\.\d+)?)\s*m?\s*becomes\s*(\d+(?:\.\d+)?)", prompt
        )
    ]


def _extract_target(prompt: str) -> float:
    """Extract the target value to convert from the prompt."""
    tgt_m = re.search(
        r"convert.*?measurement[:\s]+(\d+(?:\.\d+)?)", prompt, re.IGNORECASE
    )
    if not tgt_m:
        tgt_m = re.search(r"convert.*?(\d+(?:\.\d+)?)\s*m", prompt, re.IGNORECASE)
    if tgt_m:
        return float(tgt_m.group(1))
    return 0.0


def _build_unit_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Build a 3-node unit conversion DAG using affine linear regression.

    The DAG is always: extract_pairs → linear_fit → apply_convert.
    This handles both multiplicative (y=ax) and affine (y=ax+b) conversions.
    The LLM planner output is used only for the target value; node structure
    is fully deterministic.
    """

    target = _extract_target(prompt)

    # Also try extracting from apply_convert tool_input in planner output
    if target == 0.0:
        for meta in nodes_dict.values():
            if meta.get("id") == "apply_convert" and meta.get("tool_input"):
                tgt_m = re.search(r"\*\s*(\d+(?:\.\d+)?)", meta["tool_input"])
                if tgt_m:
                    target = float(tgt_m.group(1))
                    break

    prompt_pairs = _extract_prompt_pairs(prompt)
    pairs_json = json.dumps(
        [[fv, tv] for fv, tv in prompt_pairs]
    )

    dag: list[ThoughtNode] = [
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
            id="linear_fit",
            question="Fit y = ax + b to all extracted (from, to) pairs using least-squares regression.",
            depends_on=["extract_pairs"],
            tool="linear_fit",
            tool_input="{extract_pairs}",
            answer=None,
        ),
        ThoughtNode(
            id="apply_convert",
            question=f"Apply the fitted linear model (slope, intercept) to convert target={target}, round to 2dp.",
            depends_on=["linear_fit"],
            tool="eval_math",
            tool_input=json.dumps(
                {"expr": f"round({{linear_fit_slope}} * {target} + {{linear_fit_intercept}}, 2)"}
            ),
            answer=None,
        ),
    ]

    return dag


# ═══════════════════════════════════════════════════════════════════
#  NUMERAL CONVERSION planner  (generalized: Roman, base-N, custom)
# ═══════════════════════════════════════════════════════════════════

NUMERAL_PLANNER_SYSTEM = """\
You are a DAG planner for numeral-system conversion puzzles.

The puzzle gives examples of numbers converted into an unknown numeral
system (could be Roman, binary, hex, or any base 2-36) and asks you to
convert a target number.

Plan a DAG with EXACTLY these 5 nodes:

1. extract_data    (tool: ask_llm)  — extract [decimal, notation] pairs,
   target number, and list unique symbols seen in the notations.
2. detect_system   (tool: detect_numeral_system) — deterministic: tries
   Roman + all bases 2-36 against the examples.  Runs in PARALLEL with N3.
3. llm_analyze     (tool: ask_llm) — LLM reasons about the patterns.
   Runs in PARALLEL with N2.
4. reconcile       (tool: ask_llm) — combines results from N2 and N3,
   picks the system.  If detect_system found a perfect match, trust it.
   Output ONE line: SYSTEM=roman  or  SYSTEM=base_N  (e.g. SYSTEM=base_16).
5. convert_target  (tool: convert_numeral) — deterministic conversion.

Output EXACTLY two sections (no other text):

MERMAID:
START --> N1
N1 --> N2
N1 --> N3
N2 --> N4
N3 --> N4
N4 --> N5
N5 --> END

NODES:
{"N1": {...}, "N2": {...}, "N3": {...}, "N4": {...}, "N5": {...}}

- ALWAYS output exactly 5 nodes with this topology.
- Use the actual target number from the puzzle.\
"""


def _extract_target_number(prompt: str) -> int | None:
    """Extract the target decimal number from a numeral conversion prompt."""
    m = re.search(r"write the number\s+(\d+)", prompt, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"convert.*?(\d+)", prompt, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_numeral_pairs(prompt: str) -> list[list]:
    """Extract (decimal, notation) example pairs from the prompt."""
    pairs = []
    for m in re.finditer(r"(\d+)\s*->\s*(\S+)", prompt):
        pairs.append([int(m.group(1)), m.group(2)])
    return pairs


def _build_numeral_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Build a 5-node numeral conversion DAG with parallel hypothesis testing.

    Structure:
      extract_data → (detect_system ∥ llm_analyze) → reconcile → convert_target

    detect_system deterministically tries Roman + bases 2-36.
    llm_analyze lets the LLM reason about patterns/custom systems.
    reconcile picks the best system.  convert_target applies it.
    """
    target = _extract_target_number(prompt)
    if target is None:
        raise ValueError(
            "Could not extract target number from numeral conversion prompt"
        )

    pairs = _extract_numeral_pairs(prompt)
    pairs_json = json.dumps(pairs)

    dag: list[ThoughtNode] = [
        # N1: extract data
        ThoughtNode(
            id="extract_data",
            question=(
                "Read the numeral-system puzzle below. Extract:\n"
                "1. Every example as [decimal_number, notation_string]\n"
                "2. The target decimal number to convert\n"
                "3. All unique symbols you see in the notation outputs\n\n"
                f"{prompt}\n\n"
                "Output ONLY valid JSON:\n"
                '{"pairs": [[38, "XXXVIII"], ...], "target": 38, '
                '"symbols": ["I","V","X","L","C","D","M"]}'
            ),
            depends_on=[],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        # N2: deterministic detection (parallel with N3)
        ThoughtNode(
            id="detect_system",
            question=(
                "Detect the numeral system by testing all known systems "
                "(Roman, base 2-36) against the example pairs."
            ),
            depends_on=["extract_data"],
            tool="detect_numeral_system",
            tool_input=json.dumps({"pairs": pairs}),
            answer=None,
        ),
        # N3: LLM analysis (parallel with N2)
        ThoughtNode(
            id="llm_analyze",
            question=(
                "Analyze these numeral conversion examples:\n"
                f"{pairs_json}\n\n"
                "For each example, study the notation output:\n"
                "- What unique symbols appear? How many?\n"
                "- Is it positional (like binary/hex) or additive/subtractive "
                "(like Roman)?\n"
                "- If positional: what base? The number of unique symbols "
                "often equals the base.\n"
                "- If Roman-like: I=1, V=5, X=10, L=50, C=100, D=500, M=1000?\n"
                "- Could it be a custom system with non-standard symbols?\n\n"
                "Conclude with ONE line:\n"
                "SYSTEM=roman  OR  SYSTEM=base_N  (e.g. SYSTEM=base_16)\n"
                "If unsure, give your best guess."
            ),
            depends_on=["extract_data"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        # N4: reconcile
        ThoughtNode(
            id="reconcile",
            question=(
                "Deterministic detection result:\n{detect_system}\n\n"
                "LLM analysis:\n{llm_analyze}\n\n"
                "Pick the numeral system. Rules:\n"
                "- If detect_system shows all_correct=true, TRUST it.\n"
                "- If detect_system failed (all_correct=false), consider "
                "the LLM analysis.\n"
                "- If both disagree and neither is confident, go with "
                "detect_system's best match.\n\n"
                "Output ONLY one line: SYSTEM=roman  or  SYSTEM=base_N"
            ),
            depends_on=["detect_system", "llm_analyze"],
            tool="ask_llm",
            tool_input=None,
            answer=None,
        ),
        # N5: convert target
        ThoughtNode(
            id="convert_target",
            question=f"Convert {target} using the reconciled system.",
            depends_on=["reconcile"],
            tool="convert_numeral",
            tool_input=json.dumps({
                "number": target,
                "system": "{reconcile}",
            }),
            answer=None,
        ),
    ]

    return dag


# ═══════════════════════════════════════════════════════════════════
#  CIPHER DECRYPTION planner
# ═══════════════════════════════════════════════════════════════════

CIPHER_PLANNER_SYSTEM = """\
You are a DAG planner for cipher / encryption puzzles.

The puzzle gives N example lines (encrypted text -> plaintext) and asks
you to decrypt a target ciphertext.  The encryption rule is UNKNOWN.

Plan a DAG with this topology:

1. extract_cases (ask_llm) — discover all routes (example lines + target).
2. For EACH case i (parallel, deterministic):
   a. extract_pairs_i (split_word_pairs) — split into word pairs.
   b. create_mapping_i (build_char_map) — char alignment.
3. merge_mapping (merge_char_maps) — majority-vote merge.
4. translate (decrypt_substitution) — apply map to ciphertext.

Only extract_cases uses the LLM.  Everything else is deterministic.\
"""


def _extract_cipher_cases(prompt: str) -> list[tuple[str, str]]:
    """Extract (encrypted_line, plain_line) example pairs from the prompt."""
    return [
        (enc.strip(), plain.strip())
        for enc, plain in re.findall(r"(.+?)\s*->\s*(.+)", prompt)
    ]


def _extract_ciphertext(prompt: str) -> str:
    """Extract the target ciphertext from the prompt."""
    m = re.search(r"decrypt the following text:\s*(.+)", prompt, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _build_cipher_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Build a cipher DAG: one LLM call to discover routes, then fully deterministic.

    Structure:
      extract_cases (ask_llm)
        ├── extract_pairs_1 (split_word_pairs)  → create_mapping_1 (build_char_map) ─┐
        ├── extract_pairs_2 (split_word_pairs)  → create_mapping_2 (build_char_map) ─┤
        ├── ...                                                                      ─┤
        └── extract_pairs_N (split_word_pairs)  → create_mapping_N (build_char_map) ─┘
                                                                                      ↓
                                                        merge_mapping (merge_char_maps)
                                                                                      ↓
                                                    translate (decrypt_substitution)

    Only extract_cases uses the LLM.  Everything after is deterministic.
    """
    cases = _extract_cipher_cases(prompt)
    ciphertext = _extract_ciphertext(prompt)

    dag: list[ThoughtNode] = []

    mapping_ids: list[str] = []
    for i, (enc_line, plain_line) in enumerate(cases, 1):
        pairs_id = f"extract_pairs_{i}"
        map_id = f"create_mapping_{i}"
        mapping_ids.append(map_id)

        enc_esc = enc_line.replace('"', '\\"')
        plain_esc = plain_line.replace('"', '\\"')
        dag.append(ThoughtNode(
            id=pairs_id,
            question=f"Split words: {enc_line} | {plain_line}",
            depends_on=[],
            tool="split_word_pairs",
            tool_input=f'{{"encrypted": "{enc_esc}", "plaintext": "{plain_esc}"}}',
            answer=None,
        ))

        dag.append(ThoughtNode(
            id=map_id,
            question="Build character substitution map from word pairs.",
            depends_on=[pairs_id],
            tool="build_char_map",
            tool_input="{" + pairs_id + "}",
            answer=None,
        ))

    maps_input = "\n".join("{" + mid + "}" for mid in mapping_ids)
    dag.append(ThoughtNode(
        id="merge_mapping",
        question="Merge all per-case character mappings via majority vote.",
        depends_on=list(mapping_ids),
        tool="merge_char_maps",
        tool_input=maps_input,
        answer=None,
    ))

    ct_esc = ciphertext.replace('"', '\\"')
    dag.append(ThoughtNode(
        id="translate",
        question="Apply merged char map to ciphertext.",
        depends_on=["merge_mapping"],
        tool="decrypt_substitution",
        tool_input='{"ciphertext": "' + ct_esc + '", "mapping": {merge_mapping}}',
        answer=None,
    ))

    return dag


# ═══════════════════════════════════════════════════════════════════
#  BIT MANIPULATION planner
# ═══════════════════════════════════════════════════════════════════

def _build_bit_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Build a single-node deterministic DAG for bit manipulation.

    solve_bit_manipulation handles everything: regex-extract binary pairs
    and target, brute-force per-bit boolean function search, apply to target.
    """
    import json as _json
    return [
        ThoughtNode(
            id="solve_bits",
            question="Brute-force bit manipulation rule and apply to target.",
            depends_on=[],
            tool="solve_bit_manipulation",
            tool_input=_json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
#  EQUATION TRANSFORM planner
# ═══════════════════════════════════════════════════════════════════

def _build_equation_dag(
    edges: list[tuple[str, str]],
    nodes_dict: dict[str, dict],
    prompt: str,
) -> list[ThoughtNode]:
    """Build a single-node deterministic DAG for equation transformation.

    solve_equation_transform handles everything: parse examples and target,
    try numeric/symbolic/ordinal strategies per operator group.
    """
    import json as _json
    return [
        ThoughtNode(
            id="solve_equation",
            question="Solve equation transformation deterministically.",
            depends_on=[],
            tool="solve_equation_transform",
            tool_input=_json.dumps({"prompt": prompt}),
            answer=None,
        ),
    ]


# ── QueryPlanner class ──────────────────────────────────────────────

class QueryPlanner:
    """LLM-based query planner for all puzzle types.

    Calls the LLM to generate a mermaid-style execution plan (topology)
    plus a ThoughtNode dict, then combines them into ThoughtNodes.
    Raises on failure — no silent fallback.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ── gravity ────────────────────────────────────────────────────

    def plan_gravity(self, prompt: str) -> list[ThoughtNode]:
        """Plan a gravity physics puzzle via LLM planner.

        Raises if the LLM output cannot be parsed into a valid DAG.
        """
        return self._llm_plan(GRAVITY_PLANNER_SYSTEM, prompt, _build_gravity_dag)

    # ── unit conversion ────────────────────────────────────────────

    def plan_unit(self, prompt: str) -> list[ThoughtNode]:
        """Plan a unit conversion puzzle via LLM planner.

        Raises if the LLM output cannot be parsed into a valid DAG.
        """
        return self._llm_plan(UNIT_PLANNER_SYSTEM, prompt, _build_unit_dag)

    # ── numeral conversion ────────────────────────────────────────

    def plan_numeral(self, prompt: str) -> list[ThoughtNode]:
        """Plan a numeral conversion puzzle via LLM planner.

        Falls back to building the DAG directly from the prompt if the
        planner output can't be parsed — the DAG shape is fixed (5 nodes)
        so no planner information is actually lost.
        """
        try:
            return self._llm_plan(
                NUMERAL_PLANNER_SYSTEM, prompt, _build_numeral_dag
            )
        except (ValueError, KeyError):
            return _build_numeral_dag([], {}, prompt)

    # ── cipher decryption ─────────────────────────────────────────

    def plan_cipher(self, prompt: str) -> list[ThoughtNode]:
        """Plan a cipher decryption puzzle via LLM planner.

        Falls back to building the DAG directly from the prompt if the
        planner output can't be parsed.
        """
        try:
            return self._llm_plan(
                CIPHER_PLANNER_SYSTEM, prompt, _build_cipher_dag
            )
        except (ValueError, KeyError):
            return _build_cipher_dag([], {}, prompt)

    # ── bit manipulation ─────────────────────────────────────────

    def plan_bit(self, prompt: str) -> list[ThoughtNode]:
        """Plan a bit manipulation puzzle — fully deterministic."""
        return _build_bit_dag([], {}, prompt)

    # ── equation transform ─────────────────────────────────────────

    def plan_equation(self, prompt: str) -> list[ThoughtNode]:
        """Plan an equation transformation puzzle — fully deterministic."""
        return _build_equation_dag([], {}, prompt)

    # ── shared LLM call ───────────────────────────────────────────

    def _llm_plan(
        self,
        system_prompt: str,
        prompt: str,
        builder,
    ) -> list[ThoughtNode]:
        resp = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            think=False,
            temperature=0.3,
            max_tokens=16384,
        )
        raw = (resp.content or "").strip()
        if not raw:
            raise ValueError("Empty planner response")

        edges, nodes_dict = _parse_planner_output(raw)
        return builder(edges, nodes_dict, prompt)
