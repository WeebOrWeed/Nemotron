"""Full local RL iteration: collect -> analyze -> generate tools -> apply -> retrain.

Mirrors the Kaggle notebook flow (`notebooks/grpo_planner.ipynb`) end-to-end
but runs entirely on the local machine using Ollama as the LLM backend.

Pipeline
--------
Phase 1: Data collection
    Run the planner over N puzzles (N candidates each), execute each DAG,
    score the results, and write `kaggle_output/planner_scores.jsonl`.
Phase 2: Failure analysis
    For each VALID-but-wrong puzzle, ask the LLM to classify the failure
    (BAD_PLAN / TOOL_LIMITATION / TOOL_BUG / BAD_INPUT / MISSING_TOOL) and
    propose a fix.
Phase 3: Tool generation
    For each NEW_TOOL or TOOL_LIMITATION analysis, ask the LLM to write a
    concrete Python function. Validate (syntax, signature, uniqueness),
    install into runtime, and append to `kaggle_output/tools_generated.py`.
    Save full analysis to `kaggle_output/failure_analysis.json`.
Phase 4: Apply artifacts
    Call apply_artifacts.merge_tools + update_planner_prompt to update
    `src/tools.py` and `src/planner.py` in place (with .bak backups).
Phase 5: Retrain LoRA
    Call train_lora.main on the new winners.

Usage
-----
    python iterate_local.py
    python iterate_local.py --types bit_manipulation --limit 30 --n 2
    python iterate_local.py --skip-train  (skip Phase 5)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.classify import PUZZLE_SIGNATURES
from src.config import (
    MODEL_NAME, OLLAMA_BASE_URL, TRAIN_PATH,
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    OPEN_ROUTER_API_KEY, OPEN_ROUTER_MODEL,
)
from src.llm_client import LLMClient
from src.planner import PLANNER_SYSTEM, _parse_planner_output, _build_dag
from src.solver import _solve_single_node, _find_sink, _extract_final_answer

import src.tools as tools_module


HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE_OUT = os.path.join(HERE, "kaggle_output")
SCORES_PATH = os.path.join(KAGGLE_OUT, "planner_scores.jsonl")
TOOLS_GEN_PATH = os.path.join(KAGGLE_OUT, "tools_generated.py")
ANALYSIS_PATH = os.path.join(KAGGLE_OUT, "failure_analysis.json")
FORBIDDEN_MONOLITHIC_TOOLS = {
    "solve_bit_manipulation",
    "solve_composite_bit_rule",
    "solve_numeral_conversion",
    "solve_cipher_decryption",
    "solve_equation_transform",
    "generate_bit_rule_candidates",
    "select_bit_candidate",
}
BIT_TOOL_PREFIX_ALLOWLIST = (
    "extract_",
    "try_",
    "score_",
    "rank_",
    "select_",
    "apply_",
    "normalize_",
    "validate_",
)


def _has_forbidden_monolith(dag) -> bool:
    return any((node.get("tool") or "") in FORBIDDEN_MONOLITHIC_TOOLS for node in dag or [])


# ─── Phase 1: Data collection ──────────────────────────────────────────

def _detect_type(prompt: str) -> str:
    lower = prompt.lower()
    for sig, ptype in PUZZLE_SIGNATURES.items():
        if sig in lower:
            return ptype
    return "unknown"


def _compute_reward(expected: str, got: str, dag_valid: bool) -> float:
    if not dag_valid:
        return -1.0
    a, b = got.strip(), expected.strip()
    if a == b:
        return 1.0
    try:
        if abs(float(a) - float(b)) <= 1e-2 + 1e-9:
            return 0.5
    except (ValueError, TypeError):
        pass
    return -0.5


def _execute_dag(dag, llm, prompt) -> tuple[str, list[dict]]:
    """Run a DAG to completion, returning (final_answer, node_trace)."""
    trace: list[dict] = []
    max_rounds = len(dag) + 2
    for _ in range(max_rounds):
        answered = {n["id"] for n in dag if n["answer"] is not None}
        ready = [
            n for n in dag
            if n["answer"] is None and all(p in answered for p in n["depends_on"])
        ]
        if not ready:
            break
        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            futs = {pool.submit(_solve_single_node, llm, n, dag, prompt): n for n in ready}
            for fut in as_completed(futs):
                node = futs[fut]
                err = None
                try:
                    ans = fut.result(timeout=120)
                except Exception as e:
                    ans = ""
                    err = str(e)
                for n in dag:
                    if n["id"] == node["id"]:
                        n["answer"] = ans
                        break
                trace.append({
                    "id": node["id"],
                    "tool": node.get("tool"),
                    "tool_input": (node.get("tool_input") or "")[:500],
                    "answer": (ans or "")[:500],
                    "error": err,
                })
    sink = _find_sink(dag)
    raw = sink.get("answer") or ""
    return _extract_final_answer(raw) or raw, trace


def _generate_candidate(planner_llm, puzzle_type, prompt, temperature):
    resp = planner_llm.chat(
        [{"role": "system", "content": PLANNER_SYSTEM},
         {"role": "user", "content": f"PUZZLE_TYPE: {puzzle_type}\n\nPROMPT:\n{prompt}"}],
        think=False, temperature=temperature, max_tokens=4096,
    )
    raw = (resp.content or "").strip()
    if not raw:
        return "", None
    try:
        edges, nodes_dict = _parse_planner_output(raw)
        dag = _build_dag(edges, nodes_dict, prompt)
        if _has_forbidden_monolith(dag):
            return raw, None
        return raw, dag
    except Exception:
        return raw, None


COMPOSABLE_BIT_PLANNER_OUTPUT = r"""MERMAID:
START --> extract_bits
extract_bits --> gen_candidates
gen_candidates --> select_bits
select_bits --> normalize_bits
extract_bits --> normalize_bits
normalize_bits --> END

NODES:
{"extract_bits": {"id": "extract_bits", "question": "Extract bit examples and target from the prompt.", "tool": "extract_bit_task", "tool_input": "{\"prompt\": \"__PROMPT__\"}"}, "gen_candidates": {"id": "gen_candidates", "question": "Generate candidate bit-rule predictions from the extracted examples.", "tool": "generate_bit_rule_candidates", "tool_input": "{extract_bits}"}, "select_bits": {"id": "select_bits", "question": "Select the most reliable candidate answer.", "tool": "select_bit_candidate", "tool_input": "{gen_candidates}"}, "normalize_bits": {"id": "normalize_bits", "question": "Normalize the selected binary answer.", "tool": "normalize_binary_answer", "tool_input": "{\"answer\": \"{select_bits}\", \"bits\": \"{extract_bits_bits}\"}"}}"""


def _generate_forced_composable_bit_plan(prompt: str):
    edges, nodes_dict = _parse_planner_output(COMPOSABLE_BIT_PLANNER_OUTPUT)
    return COMPOSABLE_BIT_PLANNER_OUTPUT, _build_dag(edges, nodes_dict, prompt)


def phase1_collect(args, planner_llm, exec_llm) -> list[dict]:
    print("\n" + "=" * 60)
    print("Phase 1: Data Collection")
    print("=" * 60)

    df = pd.read_csv(TRAIN_PATH)
    df["puzzle_type"] = df["prompt"].apply(_detect_type)
    if args.types:
        df = df[df["puzzle_type"].isin(args.types)]
    df = df.reset_index(drop=True)
    start = args.offset
    end = start + args.limit
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Slice: rows {start}..{end-1} of {args.types or 'ALL'} ({len(df)} puzzles)")

    step = 0.7 / max(args.n - 1, 1)
    temps = [round(0.2 + i * step, 2) for i in range(args.n)]
    print(f"Puzzles: {len(df)}  |  Candidates/puzzle: {args.n}  |  Temps: {temps}")

    all_records: list[dict] = []
    stats = {"total": 0, "exact": 0, "valid": 0, "reward_sum": 0.0}

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        row_id = row["id"]
        prompt = row["prompt"]
        expected = str(row.get("answer", ""))
        puzzle_type = row["puzzle_type"]
        print(f"[{idx}/{len(df)}] {puzzle_type} {row_id} expected={expected!r}")

        for ci, temp in enumerate(temps):
            t0 = time.time()
            if args.force_composable_bit_plan and puzzle_type == "bit_manipulation":
                raw_plan, dag = _generate_forced_composable_bit_plan(prompt)
            else:
                raw_plan, dag = _generate_candidate(planner_llm, puzzle_type, prompt, temp)
            dag_valid = dag is not None
            got, trace = "", []
            if dag_valid:
                try:
                    got, trace = _execute_dag(dag, exec_llm, prompt)
                except Exception:
                    got = ""
            reward = _compute_reward(expected, got, dag_valid)
            elapsed = round(time.time() - t0, 1)

            record = {
                "puzzle_id": row_id,
                "puzzle_type": puzzle_type,
                "candidate": ci,
                "temperature": temp,
                "prompt": prompt,
                "planner_output": raw_plan[:4000],
                "dag_valid": dag_valid,
                "dag_nodes": len(dag) if dag else 0,
                "got": got,
                "expected": expected,
                "reward": reward,
                "elapsed_s": elapsed,
                "node_trace": trace,
            }
            all_records.append(record)
            stats["total"] += 1
            stats["reward_sum"] += reward
            if reward >= 1.0:
                stats["exact"] += 1
            if dag_valid:
                stats["valid"] += 1
            tag = "OK" if reward >= 0.5 else ("VALID" if dag_valid else "BAD_DAG")
            print(f"  c{ci} T={temp} {tag} got={got!r} reward={reward} ({elapsed}s)")

    os.makedirs(KAGGLE_OUT, exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = stats["total"] or 1
    print(f"\n=== Collection Summary ===")
    print(f"Total samples:  {stats['total']}")
    print(f"Valid DAGs:     {stats['valid']} ({100*stats['valid']/n:.1f}%)")
    print(f"Exact matches:  {stats['exact']} ({100*stats['exact']/n:.1f}%)")
    print(f"Mean reward:    {stats['reward_sum']/n:.3f}")
    print(f"Saved: {SCORES_PATH}")
    return all_records


# ─── Phase 2: Failure analysis ─────────────────────────────────────────

ANALYSIS_SYSTEM_TMPL = """\
You are a debugging assistant for a DAG-based puzzle solver.

You receive: the puzzle prompt, the DAG plan, a per-node trace, and the
expected vs actual answer.

THE FOLLOWING TOOLS ARE ALREADY DEFINED AND AVAILABLE in the registry --
DO NOT classify any of them as "missing" or "not defined":
{tool_registry}

Classify the root cause as ONE of:
1. BAD_PLAN        wrong tools chosen or wrong DAG structure
2. TOOL_LIMITATION tool exists but its algorithm cannot handle this case
3. TOOL_BUG        simple code bug in an existing tool
4. BAD_INPUT       wrong tool_input wiring/interpolation in the DAG
5. MISSING_TOOL    no relevant tool exists at all (CHECK THE LIST ABOVE FIRST)

If fix_type is NEW_TOOL, the new_tool_name MUST be a fresh snake_case name
NOT in the list above. Be specific (e.g. "extract_pairwise_xor_pattern",
not "solve_bit_manipulation").

ARCHITECTURE RULE:
- Never propose or repair a type-specific end-to-end solver, especially any
  tool named solve_*.
- New tools must be small DAG steps such as extract_*, generate_*,
  score_*, rank_*, select_*, apply_*, or normalize_*.
- If a failure was caused by a monolithic solve_* node, classify it as BAD_PLAN
  and propose replacing it with composable nodes.

Output STRICT JSON only (no commentary):
{{
  "failure_type": "BAD_PLAN|TOOL_LIMITATION|TOOL_BUG|BAD_INPUT|MISSING_TOOL",
  "failed_node_id": "<which node caused the error>",
  "explanation": "<2-3 sentences>",
  "fix_type": "UPDATE_PROMPT|NEW_TOOL|FIX_TOOL",
  "new_tool_name": "<snake_case if NEW_TOOL else null>",
  "proposal": "<specific proposal>"
}}\
"""


def _existing_tool_names() -> list[str]:
    return sorted(getattr(tools_module, "TOOL_REGISTRY", {}).keys())


def _analysis_system() -> str:
    names = _existing_tool_names()
    listing = "\n".join(f"  - {n}" for n in names)
    return ANALYSIS_SYSTEM_TMPL.format(tool_registry=listing)


def phase2_analyze(records, llm, max_analyses=30) -> list[dict]:
    print("\n" + "=" * 60)
    print("Phase 2: Failure Analysis")
    print("=" * 60)

    failures = [r for r in records if r["dag_valid"] and -1.0 < r["reward"] < 1.0]
    fail_per_puzzle: dict[str, dict] = {}
    for r in failures:
        if r["puzzle_id"] not in fail_per_puzzle:
            fail_per_puzzle[r["puzzle_id"]] = r

    print(f"Unique failed puzzles: {len(fail_per_puzzle)}")
    if not fail_per_puzzle:
        return []

    existing = set(_existing_tool_names())
    system_prompt = _analysis_system()
    analyses: list[dict] = []
    items = list(fail_per_puzzle.items())[:max_analyses]
    for i, (pid, r) in enumerate(items, 1):
        trace_str = json.dumps(r.get("node_trace", []), indent=2)[:2000]
        user_msg = (
            f"PUZZLE_TYPE: {r['puzzle_type']}\n\n"
            f"PROMPT (truncated):\n{r['prompt'][:1000]}\n\n"
            f"DAG PLAN:\n{r['planner_output'][:1500]}\n\n"
            f"NODE EXECUTION TRACE:\n{trace_str}\n\n"
            f"EXPECTED: {r['expected']}\nGOT:      {r['got']}"
        )
        try:
            resp = llm.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg}],
                think=False, temperature=0.1, max_tokens=1024,
            )
            raw = (resp.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            try:
                analysis = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    print(f"[{i}] {pid}: no JSON in response")
                    continue
                analysis = json.loads(m.group(0))
            # Sanity: if the LLM proposed a tool name that already exists,
            # downgrade to FIX_TOOL or BAD_INPUT to avoid wasting Phase 3.
            ntn = analysis.get("new_tool_name")
            if (
                analysis.get("fix_type") == "NEW_TOOL"
                and isinstance(ntn, str)
                and ntn in existing
            ):
                analysis["original_fix_type"] = "NEW_TOOL"
                analysis["original_new_tool_name"] = ntn
                analysis["fix_type"] = "FIX_TOOL"
                analysis["new_tool_name"] = None
                analysis["explanation"] += (
                    f" [auto-downgraded: '{ntn}' already exists]"
                )
            analysis["puzzle_id"] = pid
            analysis["puzzle_type"] = r["puzzle_type"]
            analyses.append(analysis)
            print(f"[{i}/{len(items)}] {pid}: {analysis['failure_type']} -> {analysis['fix_type']}")
        except Exception as e:
            print(f"[{i}] {pid}: analysis failed ({e})")

    print(f"\nAnalyzed {len(analyses)} failures")
    if analyses:
        ft = Counter(a["failure_type"] for a in analyses)
        fx = Counter(a["fix_type"] for a in analyses)
        print(f"By failure_type: {dict(ft)}")
        print(f"By fix_type:     {dict(fx)}")
    return analyses


# ─── Phase 3: Tool generation ──────────────────────────────────────────

TOOL_GEN_SYSTEM_TMPL = """\
You are a Python developer writing deterministic tool functions for a DAG puzzle solver.

Given a tool proposal (name, purpose, expected I/O), write a complete Python function.

STRICT REQUIREMENTS:
1. Function name must be snake_case and UNIQUE -- it MUST NOT be any of:
{forbidden}
2. Function signature: def tool_name(raw: str) -> str:
3. First line inside the function: params = json.loads(raw)
4. Returns a string result
5. Completely deterministic (NO LLM calls, NO randomness, NO network)
6. Only use standard library imports (json, math, re, itertools, etc.)
7. Include a one-line docstring
8. Implement REAL logic. Do NOT return a placeholder or hardcoded answer.
9. For bit_manipulation, the function must be a composable step named with
   one of these prefixes: extract_, try_, score_, rank_, select_, apply_,
   normalize_, validate_. Do NOT use broad names like infer_complex_bit_rule.

Output format -- output EXACTLY this, nothing else:

TOOL_NAME: <function_name>
DESCRIPTION: <one-line description of what it does>
CATALOGUE_ENTRY: <tool_name> | <input format example>

```python
import json

def function_name(raw: str) -> str:
    \"\"\"One-line docstring.\"\"\"
    params = json.loads(raw)
    # implementation
    return str(result)
```\
"""


def _tool_gen_system() -> str:
    forbidden = "\n".join(f"  - {n}" for n in _existing_tool_names())
    return TOOL_GEN_SYSTEM_TMPL.format(forbidden=forbidden)


def phase3_generate_tools(analyses, llm, max_tools=15) -> list[dict]:
    print("\n" + "=" * 60)
    print("Phase 3: Tool Generation")
    print("=" * 60)

    new_tool_proposals = [
        a for a in analyses
        if a["fix_type"] == "NEW_TOOL" or a["failure_type"] == "TOOL_LIMITATION"
    ]
    print(f"NEW_TOOL/TOOL_LIMITATION proposals: {len(new_tool_proposals)}")

    system_prompt = _tool_gen_system()
    installed: list[dict] = []
    for a in new_tool_proposals[:max_tools]:
        user_msg = (
            f"Failure context:\n"
            f"- Puzzle type: {a['puzzle_type']}\n"
            f"- Failed node: {a.get('failed_node_id')}\n"
            f"- Root cause: {a['explanation']}\n"
            f"- Proposal: {a['proposal']}\n"
            f"- Suggested name (you may rename): {a.get('new_tool_name')}\n\n"
            f"Write the tool function with a UNIQUE name not in the forbidden list."
        )
        try:
            resp = llm.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg}],
                think=False, temperature=0.2, max_tokens=2048,
            )
            raw_resp = (resp.content or "").strip()

            tool_name = description = catalogue_entry = ""
            for line in raw_resp.split("\n"):
                if line.startswith("TOOL_NAME:"):
                    tool_name = line.split(":", 1)[1].strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("CATALOGUE_ENTRY:"):
                    catalogue_entry = line.split(":", 1)[1].strip()

            code = raw_resp
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                compile(code, f"<gen_{tool_name}>", "exec")
            except SyntaxError as e:
                print(f"  SKIP {tool_name}: syntax error: {e}")
                continue

            fn_match = re.search(r"def\s+(\w+)\s*\(", code)
            if not fn_match:
                print("  SKIP: no function found in generated code")
                continue
            actual_fn_name = fn_match.group(1)
            if actual_fn_name in FORBIDDEN_MONOLITHIC_TOOLS or actual_fn_name.startswith("solve_"):
                print(f"  SKIP {actual_fn_name}: type-specific monolithic/fixed solver names are forbidden")
                continue
            if a["puzzle_type"] == "bit_manipulation" and not actual_fn_name.startswith(BIT_TOOL_PREFIX_ALLOWLIST):
                print(f"  SKIP {actual_fn_name}: bit tools must be composable strategy-step names")
                continue
            if hasattr(tools_module, actual_fn_name):
                print(f"  SKIP {actual_fn_name}: already exists in tools.py")
                continue

            exec_globals = {
                "json": json, "math": __import__("math"),
                "re": re, "itertools": __import__("itertools"),
            }
            exec(code, exec_globals)
            fn = exec_globals[actual_fn_name]
            setattr(tools_module, actual_fn_name, fn)

            installed.append({
                "puzzle_type": a["puzzle_type"],
                "failed_node": a.get("failed_node_id"),
                "explanation": a["explanation"],
                "tool_name": actual_fn_name,
                "description": description or actual_fn_name,
                "catalogue_entry": catalogue_entry or f"{actual_fn_name} | <input>",
                "code": code,
                "status": "installed",
            })
            print(f"  INSTALLED {actual_fn_name} for [{a['puzzle_type']}]")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nInstalled {len(installed)} new tools")

    os.makedirs(KAGGLE_OUT, exist_ok=True)
    with open(TOOLS_GEN_PATH, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated tools from failure analysis.\n\n')
        f.write('Copy the functions you want to keep into src/tools.py.\n"""\n')
        f.write("import json\nimport math\nimport re\nimport itertools\n\n")
        for t in installed:
            f.write(f"\n# --- [{t['puzzle_type']}] {t['description']} ---\n")
            f.write(t["code"])
            f.write("\n\n")
    print(f"Saved: {TOOLS_GEN_PATH}")

    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "analyses": analyses,
            "generated_tools": installed,
            "prompt_update_proposals": [
                {"puzzle_type": a["puzzle_type"], "explanation": a["explanation"],
                 "proposal": a["proposal"]}
                for a in analyses if a["fix_type"] == "UPDATE_PROMPT"
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {ANALYSIS_PATH}")
    return installed


# ─── Phase 4: Apply artifacts ──────────────────────────────────────────

def phase4_apply(args):
    print("\n" + "=" * 60)
    print("Phase 4: Apply Artifacts to src/")
    print("=" * 60)
    import apply_artifacts
    allowed = set(args.types) if args.types else None
    result = apply_artifacts.merge_tools(allowed, dry_run=False)
    apply_artifacts.update_planner_prompt(allowed, result["added"], dry_run=False)


# ─── Phase 5: Retrain LoRA ─────────────────────────────────────────────

def phase5_retrain(args):
    print("\n" + "=" * 60)
    print("Phase 5: Retrain LoRA")
    print("=" * 60)
    cmd = [sys.executable, "train_lora.py"]
    if args.types:
        cmd += ["--types", *args.types]
    cmd += ["--epochs", str(args.epochs)]
    if args.lora_model:
        cmd += ["--model", args.lora_model]
    print("Running:", " ".join(cmd))
    import subprocess
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"WARNING: train_lora.py exited with code {rc}")
    return rc


# ─── Main ──────────────────────────────────────────────────────────────

def _make_llm(role: str) -> LLMClient:
    """Pick the LLM for a given role (planner / executor / analyzer / generator).

    Strategy: prefer cloud LLM (OpenRouter, then DeepSeek) for analysis +
    tool generation since they need stronger reasoning; fall back to local
    Ollama. Use Ollama for executor since tools handle most of the work.
    """
    if role in ("analyzer", "generator", "planner"):
        if OPEN_ROUTER_API_KEY:
            return LLMClient(
                provider="openrouter",
                openrouter_api_key=OPEN_ROUTER_API_KEY,
                openrouter_model=OPEN_ROUTER_MODEL,
            )
        if DEEPSEEK_API_KEY:
            return LLMClient(
                provider="deepseek",
                deepseek_api_key=DEEPSEEK_API_KEY,
                deepseek_model=DEEPSEEK_MODEL,
            )
    return LLMClient(
        provider="ollama", model_name=MODEL_NAME, ollama_base_url=OLLAMA_BASE_URL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local end-to-end RL iteration")
    parser.add_argument("--types", nargs="*", default=None,
                        help="Restrict to puzzle types (e.g. bit_manipulation)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max puzzles to collect (default: 30)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N puzzles (after type filter) -- "
                             "use to iterate over fresh slices (default: 0)")
    parser.add_argument("--n", type=int, default=2,
                        help="Candidate DAGs per puzzle (default: 2)")
    parser.add_argument("--max-analyses", type=int, default=30,
                        help="Max failures to analyze (default: 30)")
    parser.add_argument("--max-tools", type=int, default=15,
                        help="Max new tools to generate (default: 15)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="LoRA training epochs (default: 3)")
    parser.add_argument("--lora-model", default=None,
                        help="Override base model for train_lora.py")
    parser.add_argument("--planner-llm", default="ollama",
                        choices=["ollama", "openrouter", "deepseek"],
                        help="Planner backend (default: ollama)")
    parser.add_argument("--analyzer-llm", default="ollama",
                        choices=["ollama", "openrouter", "deepseek"],
                        help="Analyzer + tool-gen backend (default: ollama)")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Reuse existing kaggle_output/planner_scores.jsonl")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Reuse existing kaggle_output/failure_analysis.json")
    parser.add_argument("--skip-apply", action="store_true",
                        help="Don't update src/tools.py and src/planner.py")
    parser.add_argument("--skip-train", action="store_true",
                        help="Don't run train_lora.py at the end")
    parser.add_argument("--force-composable-bit-plan", action="store_true",
                        help="For bit_manipulation, collect data with the enforced composable DAG instead of calling the planner")
    args = parser.parse_args()

    print("=" * 60)
    print("Local RL Iteration")
    print(f"  types       = {args.types or 'ALL'}")
    print(f"  limit       = {args.limit}")
    print(f"  candidates  = {args.n}")
    print(f"  planner_llm = {args.planner_llm}")
    print(f"  analyzer_llm= {args.analyzer_llm}")
    print(f"  force_bit   = {args.force_composable_bit_plan}")
    print("=" * 60)

    def _build(provider: str) -> LLMClient:
        if provider == "openrouter":
            if not OPEN_ROUTER_API_KEY:
                sys.exit("OPEN_ROUTER_API_KEY not set in .env")
            return LLMClient(provider="openrouter",
                             openrouter_api_key=OPEN_ROUTER_API_KEY,
                             openrouter_model=OPEN_ROUTER_MODEL)
        if provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                sys.exit("DEEPSEEK_API_KEY not set in .env")
            return LLMClient(provider="deepseek",
                             deepseek_api_key=DEEPSEEK_API_KEY,
                             deepseek_model=DEEPSEEK_MODEL)
        return LLMClient(provider="ollama",
                         model_name=MODEL_NAME, ollama_base_url=OLLAMA_BASE_URL)

    planner_llm = _build(args.planner_llm)
    exec_llm = _build("ollama")
    analyzer_llm = _build(args.analyzer_llm)

    if args.skip_collect:
        print(f"\n[skip-collect] reading existing {SCORES_PATH}")
        records = []
        with open(SCORES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"  {len(records)} records loaded")
    else:
        records = phase1_collect(args, planner_llm, exec_llm)

    if args.skip_analyze and os.path.exists(ANALYSIS_PATH):
        print(f"\n[skip-analyze] reading existing {ANALYSIS_PATH}")
        analyses = json.load(open(ANALYSIS_PATH, encoding="utf-8"))["analyses"]
    else:
        analyses = phase2_analyze(records, analyzer_llm, args.max_analyses)
        phase3_generate_tools(analyses, analyzer_llm, args.max_tools)

    if not args.skip_apply:
        phase4_apply(args)
    else:
        print("\n[skip-apply] src/ unchanged")

    if not args.skip_train:
        phase5_retrain(args)
    else:
        print("\n[skip-train] LoRA not retrained")

    print("\n" + "=" * 60)
    print("Iteration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
