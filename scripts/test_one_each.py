"""Run 1 mismatch from each puzzle type through the full pipeline."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "deepseek"
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
from src.graph import build_graph
from src.llm_client import LLMClient
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

mismatches = pd.read_csv("results/misses/mismatches.csv")
train = pd.read_csv("data/train.csv")

sample_ids = mismatches.groupby("puzzle_type").first()["id"].tolist()
print(f"Testing {len(sample_ids)} IDs (1 per type):\n")
for sid in sample_ids:
    row = mismatches[mismatches["id"] == sid].iloc[0]
    print(f"  {sid}  {row['puzzle_type']}  expected={row['expected']}")

llm = LLMClient(provider="deepseek", deepseek_api_key=DEEPSEEK_API_KEY, deepseek_model=DEEPSEEK_MODEL)
graph = build_graph(llm)

filtered = train[train["id"].isin(sample_ids)]

for _, row in filtered.iterrows():
    miss_row = mismatches[mismatches["id"] == row["id"]].iloc[0]
    ptype = miss_row["puzzle_type"]
    expected = row["answer"]
    print(f"\n{'='*60}")
    print(f"ID: {row['id']}  TYPE: {ptype}")
    print(f"Expected: {expected}")

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
        dag = output.get("thought_dag") or []

        elapsed = time.time() - t0
        a, b = str(answer).strip(), str(expected).strip()
        match = a == b
        if not match:
            try:
                match = abs(float(a) - float(b)) <= 1e-2 + 1e-9
            except (ValueError, TypeError):
                pass

        print(f"Actual:   {answer}")
        print(f"Result:   {'MATCH' if match else 'MISS'}")
        print(f"Time:     {elapsed:.1f}s")
        print(f"DAG nodes: {len(dag)}")
        for n in dag:
            status = "OK" if n["answer"] is not None else "UNSOLVED"
            deps = ",".join(n["depends_on"]) or "(root)"
            ans = (n["answer"] or "")[:80]
            print(f"  [{status}] {n['id']} (tool:{n.get('tool','llm')}) deps={deps}")
            print(f"           -> {ans}")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"ERROR: {exc}")
        print(f"Time: {elapsed:.1f}s")

print(f"\n{'='*60}")
print("Done.")
