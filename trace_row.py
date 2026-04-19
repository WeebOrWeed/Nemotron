"""Run graph for a single train id and print full DAG step outputs."""
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    TRAIN_PATH,
    LLM_PROVIDER,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    OPEN_ROUTER_API_KEY,
    OPEN_ROUTER_MODEL,
)
from src.graph import build_graph
from src.llm_client import LLMClient
from src.solver import _interpolate_parents

ROW_ID = sys.argv[1] if len(sys.argv) > 1 else "0040ff76"


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    row = df.loc[df["id"].astype(str) == ROW_ID]
    if row.empty:
        print(f"No row with id={ROW_ID!r} in {TRAIN_PATH}")
        sys.exit(1)
    row = row.iloc[0]
    prompt = row["prompt"]
    expected = row.get("answer", "")

    llm = LLMClient(
        provider=LLM_PROVIDER,
        model_name=MODEL_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        deepseek_api_key=DEEPSEEK_API_KEY,
        deepseek_model=DEEPSEEK_MODEL,
        openrouter_api_key=OPEN_ROUTER_API_KEY,
        openrouter_model=OPEN_ROUTER_MODEL,
    )
    print(f"id={ROW_ID}  provider={llm._resolve_provider()}  expected={expected!r}\n")
    print("=" * 72)

    graph = build_graph(llm)
    out = graph.invoke({
        "prompt": prompt,
        "answer": None,
        "puzzle_type": None,
        "thought_dag": None,
        "retries": 0,
        "failure_log": [],
    })

    print(f"puzzle_type: {out.get('puzzle_type')}")
    print(f"retries: {out.get('retries', 0)}")
    print(f"final answer: {out.get('answer')!r}")
    fl = out.get("failure_log") or []
    if fl:
        print(f"failure_log ({len(fl)}):")
        for f in fl:
            print(f"  - {f['node_id']}: {f['error'][:200]}")
    print("=" * 72)

    dag = out.get("thought_dag") or []
    for i, node in enumerate(dag, 1):
        nid = node["id"]
        tool = node.get("tool") or "ask_llm"
        deps = node.get("depends_on") or []
        ans = node.get("answer") or ""
        q = node.get("question") or ""
        print(f"\n### Step {i}: {nid}  (tool={tool}, depends_on={deps})\n")
        q_sent = _interpolate_parents(q, dag)
        full_user = (
            f"Original puzzle:\n{prompt}\n\nYour task:\n{q_sent}"
        )
        print("--- user message (matches solver: system + this user block) ---")
        print(full_user[:12000] + ("\n[truncated]\n" if len(full_user) > 12000 else ""))
        print("\n--- model output (full) ---")
        print(ans if ans else "(empty)")
        print("-" * 72)

    print(f"\n### Summary: predicted={out.get('answer')!r}  expected={expected!r}")


if __name__ == "__main__":
    main()
