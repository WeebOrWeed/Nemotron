"""Re-test all IDs from mismatches.csv and remove solved ones."""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(line_buffering=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
from src.graph import build_graph
from src.llm_client import LLMClient
from src.config import (
    MODEL_NAME, OLLAMA_BASE_URL, LLM_PROVIDER,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
)

MISMATCHES_CSV = "results/misses/mismatches.csv"
TRAIN_CSV = "data/train.csv"
BATCH_SIZE = 20


def answers_match(answer, expected) -> bool:
    a, b = str(answer).strip(), str(expected).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-2 + 1e-9
    except (ValueError, TypeError):
        return False


def run_single(graph, row_id, prompt, expected, idx, total):
    t0 = time.time()
    try:
        output = graph.invoke({
            "prompt": prompt,
            "answer": None,
            "puzzle_type": None,
            "thought_dag": None,
            "retries": 0,
            "failure_log": [],
        })
        answer = output.get("answer") or ""
    except Exception as exc:
        answer = ""
        print(f"[{idx}/{total}] id={row_id} ERROR: {exc}", flush=True)
        return row_id, answer, False

    elapsed = time.time() - t0
    match = answers_match(answer, expected)
    tag = "MATCH" if match else "MISS"
    print(f"[{idx}/{total}] id={row_id} {tag} ({elapsed:.1f}s)", flush=True)
    if not match:
        print(f"    expected={expected}  actual={answer}", flush=True)
    return row_id, answer, match


def main():
    mismatches = pd.read_csv(MISMATCHES_CSV)
    train = pd.read_csv(TRAIN_CSV)

    miss_ids = set(mismatches["id"])
    filtered = train[train["id"].isin(miss_ids)].reset_index(drop=True)
    total = len(filtered)
    print(f"Retesting {total} mismatched IDs")

    type_counts = mismatches["puzzle_type"].value_counts()
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    llm = LLMClient(
        provider=LLM_PROVIDER,
        model_name=MODEL_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        deepseek_api_key=DEEPSEEK_API_KEY,
        deepseek_model=DEEPSEEK_MODEL,
    )
    graph = build_graph(llm)

    solved_ids = set()
    all_results = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = filtered.iloc[batch_start:batch_end]
        batch_t0 = time.time()

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {}
            for i, (_, row) in enumerate(batch.iterrows()):
                idx = batch_start + i + 1
                fut = pool.submit(
                    run_single, graph, row["id"], row["prompt"],
                    row["answer"], idx, total,
                )
                futures[fut] = row["id"]

            for fut in as_completed(futures):
                row_id, answer, match = fut.result()
                all_results[row_id] = answer
                if match:
                    solved_ids.add(row_id)

        elapsed = time.time() - batch_t0
        print(f"  batch {batch_start+1}-{batch_end}/{total} done ({elapsed:.1f}s)", flush=True)

    print(f"\n{'='*60}")
    print(f"Solved: {len(solved_ids)}/{total}")

    if solved_ids:
        print(f"Removing {len(solved_ids)} solved IDs from {MISMATCHES_CSV}")
        updated = mismatches[~mismatches["id"].isin(solved_ids)].reset_index(drop=True)
        updated.to_csv(MISMATCHES_CSV, index=False)
        print(f"Updated {MISMATCHES_CSV}: {len(updated)} remaining mismatches")

        print("\nSolved IDs:")
        for sid in sorted(solved_ids):
            ptype = mismatches.loc[mismatches["id"] == sid, "puzzle_type"].iloc[0]
            print(f"  {sid} ({ptype})")
    else:
        print("No new solves.")


if __name__ == "__main__":
    main()
