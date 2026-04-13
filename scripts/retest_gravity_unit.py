"""Re-test unit_conversion and gravity_physics mismatches, remove solved ones."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "deepseek"
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from src.graph import build_graph
from src.llm_client import LLMClient
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

MISMATCHES_CSV = "results/misses/mismatches.csv"
TRAIN_CSV = "data/train.csv"


def answers_match(answer, expected) -> bool:
    a, b = str(answer).strip(), str(expected).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-2 + 1e-9
    except (ValueError, TypeError):
        return False


mismatches = pd.read_csv(MISMATCHES_CSV)
train = pd.read_csv(TRAIN_CSV)

target_types = {"unit_conversion", "gravity_physics"}
subset = mismatches[mismatches["puzzle_type"].isin(target_types)]
target_ids = set(subset["id"])
filtered = train[train["id"].isin(target_ids)].reset_index(drop=True)

print(f"Retesting {len(filtered)} IDs:")
for t in target_types:
    n = (subset["puzzle_type"] == t).sum()
    print(f"  {t}: {n}")

llm = LLMClient(provider="deepseek", deepseek_api_key=DEEPSEEK_API_KEY, deepseek_model=DEEPSEEK_MODEL)
graph = build_graph(llm)

solved_ids = set()
for i, (_, row) in enumerate(filtered.iterrows(), 1):
    expected = row["answer"]
    miss_row = subset[subset["id"] == row["id"]].iloc[0]
    ptype = miss_row["puzzle_type"]

    t0 = time.time()
    try:
        output = graph.invoke({
            "prompt": row["prompt"],
            "answer": None,
            "puzzle_type": None,
            "thought_dag": None,
            "retries": 0,
            "failure_log": [],
        })
        answer = output.get("answer") or ""
    except Exception as exc:
        answer = ""
        print(f"[{i}/{len(filtered)}] {row['id']} ({ptype}) ERROR: {exc}", flush=True)
        continue

    elapsed = time.time() - t0
    match = answers_match(answer, expected)
    tag = "MATCH" if match else "MISS"
    print(f"[{i}/{len(filtered)}] {row['id']} ({ptype}) {tag} ({elapsed:.1f}s)  expected={expected}  actual={answer}", flush=True)
    if match:
        solved_ids.add(row["id"])

print(f"\n{'='*60}")
print(f"Solved: {len(solved_ids)}/{len(filtered)}")

if solved_ids:
    print(f"\nRemoving {len(solved_ids)} solved IDs from {MISMATCHES_CSV}")
    updated = mismatches[~mismatches["id"].isin(solved_ids)].reset_index(drop=True)
    updated.to_csv(MISMATCHES_CSV, index=False)
    print(f"Updated CSV: {len(updated)} remaining mismatches (was {len(mismatches)})")

    print("\nSolved:")
    for sid in sorted(solved_ids):
        r = subset[subset["id"] == sid].iloc[0]
        print(f"  {sid} ({r['puzzle_type']}) expected={r['expected']}")
else:
    print("No new solves.")
