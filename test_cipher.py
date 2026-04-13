"""Single test with FULL node output for id=00189f6a."""
import sys, time, json
import pandas as pd
from src.llm_client import LLMClient
from src.planner import QueryPlanner
from src.solver import _interpolate_parents, _extract_final_answer
from src.tools import run_tool

sys.stdout.reconfigure(line_buffering=True)

llm = LLMClient()
planner = QueryPlanner(llm)
SYSTEM = "You are a precise problem-solving assistant. Output ONLY your final answer on the last line with no extra text."

def solve_node(node, dag, prompt):
    if node["answer"] is not None:
        return
    t0 = time.time()
    tool = node.get("tool")
    if not tool or tool == "ask_llm":
        question = _interpolate_parents(node["question"], dag)
        resp = llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Original puzzle:\n" + prompt + "\n\nYour task:\n" + question},
            ],
            think=True, temperature=0.6, max_tokens=4096,
        )
        node["answer"] = (resp.content or "").strip()
    else:
        raw_input = _interpolate_parents(node.get("tool_input") or "{}", dag)
        node["answer"] = run_tool(tool, raw_input)
    elapsed = time.time() - t0
    print(f"\n=== {node['id']} ({tool}) [{elapsed:.0f}s] ===", flush=True)
    print(node["answer"], flush=True)
    print("=" * 60, flush=True)


df = pd.read_csv("data/train.csv")
row = df[df["id"] == "00189f6a"].iloc[0]
print("ID:", row.id, flush=True)
print("Expected:", row.answer, flush=True)

dag = planner.plan("cipher_decryption", row.prompt)
print(f"DAG: {[n['id'] for n in dag]}", flush=True)

while True:
    answered = {n["id"] for n in dag if n["answer"] is not None}
    ready = [n for n in dag if n["answer"] is None and all(p in answered for p in n["depends_on"])]
    if not ready:
        break
    for n in ready:
        solve_node(n, dag, row.prompt)

sink = dag[-1]
answer = _extract_final_answer(sink["answer"] or "")
print(f"\nFINAL: {answer}", flush=True)
print(f"EXPECTED: {row.answer}", flush=True)
print(f"MATCH: {answer.strip() == row.answer.strip()}", flush=True)
