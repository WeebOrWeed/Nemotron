"""Quick test: run one puzzle per type through the planner and print the DAG."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "deepseek"

import pandas as pd
from src.llm_client import LLMClient
from src.planner import QueryPlanner
from src.classify import PUZZLE_SIGNATURES
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

llm = LLMClient(provider="deepseek", deepseek_api_key=DEEPSEEK_API_KEY, deepseek_model=DEEPSEEK_MODEL)
planner = QueryPlanner(llm)
df = pd.read_csv("data/train.csv")

for sig, ptype in PUZZLE_SIGNATURES.items():
    row = df[df["prompt"].str.contains(sig, case=False, na=False)].iloc[0]
    print(f"\n{'='*60}")
    print(f"TYPE: {ptype}  ID: {row['id']}")
    try:
        dag = planner.plan(ptype, row["prompt"])
        print(f"  nodes: {len(dag)}")
        for n in dag:
            deps = ",".join(n["depends_on"]) or "(root)"
            print(f"  [{n['id']}] tool={n['tool']}  deps={deps}")
            ti = n["tool_input"][:80] + ("..." if len(n["tool_input"]) > 80 else "")
            print(f"    input: {ti}")
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
