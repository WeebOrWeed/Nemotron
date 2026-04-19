"""RL scoring harness for the DAG planner.

For each puzzle in train.csv, generates N candidate DAGs via the LLM
planner (temperature sampling), executes each through the solver, and
logs ``(puzzle_id, puzzle_type, planner_output, dag_valid, final_answer,
expected, reward)`` to a JSONL file.

The scored data can be used for:
  - Few-shot prompt optimization (add highest-reward DAGs to PLANNER_SYSTEM)
  - GRPO fine-tuning (with TRL / OpenRLHF on a local model)

Usage
-----
    python train_planner.py                           # defaults: N=4, 50 puzzles
    python train_planner.py --n 8 --limit 200         # 8 candidates, 200 puzzles
    python train_planner.py --types gravity_physics    # only gravity
    python train_planner.py --output scored.jsonl      # custom output path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.classify import PUZZLE_SIGNATURES
from src.config import (
    MODEL_NAME, OLLAMA_BASE_URL, TRAIN_PATH,
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    OPEN_ROUTER_API_KEY, OPEN_ROUTER_MODEL,
)
from src.llm_client import LLMClient
from src.planner import QueryPlanner, PLANNER_SYSTEM, _parse_planner_output, _build_dag
from src.solver import (
    _solve_single_node, _find_sink, _extract_final_answer, _interpolate_parents,
)
from src.state import ThoughtNode

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# -- Reward function ----------------------------------------------------------

def compute_reward(expected: str, got: str, dag_valid: bool) -> float:
    """Score a single DAG execution attempt.

    +1.0  exact match
    +0.5  numeric match within 0.01
    -0.5  valid DAG but wrong answer
    -1.0  malformed / unparseable DAG
    """
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


# -- Detect puzzle type -------------------------------------------------------

def _detect_type(prompt: str) -> str:
    lower = prompt.lower()
    for sig, ptype in PUZZLE_SIGNATURES.items():
        if sig in lower:
            return ptype
    return "unknown"


# -- Execute a DAG (single pass, no retries) ----------------------------------

def _execute_dag(
    dag: list[ThoughtNode],
    exec_llm: LLMClient,
    prompt: str,
) -> str:
    """Run a DAG to completion (single pass).

    Iterates over nodes in topological order, solving each one whose
    parents are all answered.  Returns the final answer string.
    """
    max_rounds = len(dag) + 2
    for _ in range(max_rounds):
        answered_ids = {n["id"] for n in dag if n["answer"] is not None}
        ready = [
            n for n in dag
            if n["answer"] is None
            and all(p in answered_ids for p in n["depends_on"])
        ]
        if not ready:
            break
        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            future_to_node = {
                pool.submit(_solve_single_node, exec_llm, n, dag, prompt): n
                for n in ready
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    answer = future.result(timeout=120)
                    for n in dag:
                        if n["id"] == node["id"]:
                            n["answer"] = answer
                            break
                except Exception:
                    for n in dag:
                        if n["id"] == node["id"]:
                            n["answer"] = ""
                            break

    sink = _find_sink(dag)
    raw = sink.get("answer") or ""
    return _extract_final_answer(raw) or raw


# -- Generate one candidate DAG -----------------------------------------------

def _generate_candidate(
    planner_llm: LLMClient,
    puzzle_type: str,
    prompt: str,
    temperature: float,
) -> tuple[str, list[ThoughtNode] | None]:
    """Call the planner LLM and return (raw_output, dag_or_None)."""
    resp = planner_llm.chat(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"PUZZLE_TYPE: {puzzle_type}\n\nPROMPT:\n{prompt}"},
        ],
        think=False,
        temperature=temperature,
        max_tokens=8192,
    )
    raw = (resp.content or "").strip()
    if not raw:
        return "", None
    try:
        edges, nodes_dict = _parse_planner_output(raw)
        dag = _build_dag(edges, nodes_dict, prompt)
        return raw, dag
    except Exception:
        return raw, None


# -- Main loop ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RL scoring harness for DAG planner")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of candidate DAGs per puzzle (default: 4)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max puzzles to process (default: 50)")
    parser.add_argument("--types", nargs="*", default=None,
                        help="Puzzle types to include (default: all)")
    parser.add_argument("--output", default="data/planner_scores.jsonl",
                        help="Output JSONL path (default: data/planner_scores.jsonl)")
    parser.add_argument("--temps", nargs="*", type=float, default=None,
                        help="Custom temperature list (default: spread 0.2..0.9)")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_PATH)
    df["puzzle_type"] = df["prompt"].apply(_detect_type)
    if args.types:
        df = df[df["puzzle_type"].isin(args.types)]
    df = df.head(args.limit)

    if args.temps:
        temps = args.temps
    else:
        step = 0.7 / max(args.n - 1, 1)
        temps = [round(0.2 + i * step, 2) for i in range(args.n)]

    planner_llm = LLMClient(
        provider="openrouter",
        openrouter_api_key=OPEN_ROUTER_API_KEY,
        openrouter_model=OPEN_ROUTER_MODEL,
    )
    exec_llm = LLMClient(
        provider=LLM_PROVIDER,
        model_name=MODEL_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        deepseek_api_key=DEEPSEEK_API_KEY,
        deepseek_model=DEEPSEEK_MODEL,
        openrouter_api_key=OPEN_ROUTER_API_KEY,
        openrouter_model=OPEN_ROUTER_MODEL,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    total = len(df)
    stats = {"total_samples": 0, "reward_sum": 0.0, "exact": 0, "valid_dag": 0}

    print(f"=== RL Planner Harness ===")
    print(f"Puzzles: {total}  |  Candidates/puzzle: {args.n}  |  Temps: {temps}")
    print(f"Output: {args.output}\n")

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        row_id = row["id"]
        prompt = row["prompt"]
        expected = str(row.get("answer", ""))
        puzzle_type = row["puzzle_type"]

        print(f"[{idx}/{total}] {puzzle_type} {row_id} expected={expected!r}")

        for ci, temp in enumerate(temps):
            t0 = time.time()
            raw_plan, dag = _generate_candidate(planner_llm, puzzle_type, prompt, temp)
            dag_valid = dag is not None

            got = ""
            if dag_valid:
                try:
                    got = _execute_dag(dag, exec_llm, prompt)
                except Exception as exc:
                    got = ""

            reward = compute_reward(expected, got, dag_valid)
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
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["total_samples"] += 1
            stats["reward_sum"] += reward
            if reward >= 1.0:
                stats["exact"] += 1
            if dag_valid:
                stats["valid_dag"] += 1

            tag = "OK" if reward >= 0.5 else ("VALID" if dag_valid else "BAD_DAG")
            print(f"  c{ci} T={temp} {tag} got={got!r} reward={reward} ({elapsed}s)")

    out_f.close()

    n = stats["total_samples"] or 1
    print(f"\n=== Summary ===")
    print(f"Total samples:  {stats['total_samples']}")
    print(f"Valid DAGs:     {stats['valid_dag']} ({100*stats['valid_dag']/n:.1f}%)")
    print(f"Exact matches:  {stats['exact']} ({100*stats['exact']/n:.1f}%)")
    print(f"Mean reward:    {stats['reward_sum']/n:.3f}")
    print(f"Output:         {args.output}")


if __name__ == "__main__":
    main()
