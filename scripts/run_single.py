"""Run a single puzzle ID through the full pipeline with verbose output."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "deepseek"
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from src.graph import build_graph
from src.llm_client import LLMClient
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

TARGET_ID = sys.argv[1] if len(sys.argv) > 1 else "10d29630"

train = pd.read_csv("data/train.csv")
row = train[train["id"] == TARGET_ID].iloc[0]
expected = row["answer"]

print(f"ID: {row['id']}")
print(f"Expected: {expected}")
print(f"Prompt:\n{row['prompt'][:500]}...")

llm = LLMClient(provider="deepseek", deepseek_api_key=DEEPSEEK_API_KEY, deepseek_model=DEEPSEEK_MODEL)
graph = build_graph(llm)

t0 = time.time()
output = graph.invoke({
    "prompt": row["prompt"],
    "answer": None,
    "puzzle_type": None,
    "thought_dag": None,
    "retries": 0,
    "failure_log": [],
})
elapsed = time.time() - t0

answer = output.get("answer") or ""
dag = output.get("thought_dag") or []
ptype = output.get("puzzle_type", "unknown")
retries = output.get("retries", 0)

print(f"\nPuzzle type: {ptype}")
print(f"Retries: {retries}")
print(f"Time: {elapsed:.1f}s")
print(f"\nDAG ({len(dag)} nodes):")
for n in dag:
    status = "OK" if n["answer"] is not None else "UNSOLVED"
    deps = ",".join(n["depends_on"]) or "(root)"
    print(f"\n  [{status}] {n['id']}")
    print(f"    tool: {n.get('tool', 'llm')}")
    print(f"    deps: {deps}")
    print(f"    input: {(n.get('tool_input') or '')[:200]}")
    print(f"    answer: {n.get('answer', '')}")

print(f"\nFinal answer: {answer}")
print(f"Expected:     {expected}")

a, b = str(answer).strip(), str(expected).strip()
match = a == b
if not match:
    try:
        match = abs(float(a) - float(b)) <= 1e-2 + 1e-9
    except (ValueError, TypeError):
        pass
print(f"Match: {match}")
