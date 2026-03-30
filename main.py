import argparse
import os
import time

import pandas as pd

from src.config import MODEL_NAME, OLLAMA_BASE_URL, TRAIN_PATH, TEST_PATH, RESULTS_DIR
from src.graph import build_graph


def _answers_match(answer, expected) -> bool:
    """Compare answers: exact string match, then numeric fallback for floats."""
    a, b = str(answer).strip(), str(expected).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 0.01
    except (ValueError, TypeError):
        return False


def _print_dag_trace(output: dict) -> None:
    dag = output.get("thought_dag") or []
    puzzle_type = output.get("puzzle_type", "unknown")
    retries = output.get("retries", 0)
    failure_log = output.get("failure_log") or []

    print(f"    type={puzzle_type}  nodes={len(dag)}  retries={retries}")
    for node in dag:
        status = "OK" if node["answer"] is not None else "UNSOLVED"
        answer_preview = (node["answer"] or "")[:60]
        deps = ",".join(node["depends_on"]) if node["depends_on"] else "(root)"
        executor = f"tool:{node['tool']}" if node.get("tool") else "llm"
        print(f"    [{status}] {node['id']} ({executor}) deps={deps} -> {answer_preview}")
    if failure_log:
        print(f"    failures: {len(failure_log)}")
        for f in failure_log:
            print(f"      - {f['node_id']}: {f['error'][:80]}")


def run(
    dataset_path: str,
    output_path: str,
    limit: int | None = None,
    verbose: bool = False,
) -> None:
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)

    print(f"Loaded {len(df)} rows from {dataset_path}")
    print(f"Model: {MODEL_NAME}  |  Ollama: {OLLAMA_BASE_URL}")

    graph = build_graph(MODEL_NAME, OLLAMA_BASE_URL)

    has_expected = "answer" in df.columns

    results = []
    for idx, row in df.iterrows():
        row_id = row["id"]
        prompt = row["prompt"]
        expected = row["answer"] if has_expected else None

        print(f"[{idx + 1}/{len(df)}] id={row_id} ...", end=" ", flush=True)
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
            print(f"ERROR: {exc}")
            answer = ""
            output = {}

        elapsed = time.time() - t0
        match = _answers_match(answer, expected) if has_expected else None
        status = f"{'MATCH' if match else 'MISS'} " if match is not None else ""
        print(f"{status}done ({elapsed:.1f}s)")

        if verbose and output:
            _print_dag_trace(output)

        entry = {"id": row_id, "answer": answer}
        if has_expected:
            entry["expected"] = expected
        results.append(entry)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {output_path}")

    if has_expected:
        matches = sum(
            1 for r in results if _answers_match(r["answer"], r["expected"])
        )
        print(f"Accuracy: {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron DAG-of-Thoughts Pipeline")
    parser.add_argument(
        "--dataset",
        default=TRAIN_PATH,
        help=f"Path to input CSV (default: {TRAIN_PATH})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "predictions.csv"),
        help="Path to output CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to process (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print DAG execution trace for each question",
    )
    args = parser.parse_args()
    run(args.dataset, args.output, args.limit, args.verbose)


if __name__ == "__main__":
    main()
