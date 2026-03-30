"""Test end-to-end solvers against first 10 rows of train data."""
import json
import pandas as pd
from src.classify import classify_node
from src.tools import run_tool

df = pd.read_csv("data/train.csv").head(10)

for idx, row in df.iterrows():
    state = {"prompt": row["prompt"], "puzzle_type": None, "thought_dag": None,
             "retries": 0, "failure_log": [], "answer": None}
    ptype = classify_node(state)["puzzle_type"]
    solver = f"solve_{ptype}"
    expected = str(row["answer"])

    try:
        result = run_tool(solver, json.dumps({"prompt": row["prompt"]}))
        match = result.strip() == expected.strip()
        status = "MATCH" if match else "MISS"
        print(f"[{status}] id={row['id']} type={ptype}")
        if not match:
            print(f"    got:      {result}")
            print(f"    expected: {expected}")
    except Exception as e:
        print(f"[ERROR] id={row['id']} type={ptype}: {e}")
