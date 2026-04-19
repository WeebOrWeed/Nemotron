import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.classify import PUZZLE_SIGNATURES
from src.config import (
    MODEL_NAME, OLLAMA_BASE_URL, TRAIN_PATH, TEST_PATH, RESULTS_DIR,
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    OPEN_ROUTER_API_KEY, OPEN_ROUTER_MODEL,
)
from src.graph import build_graph
from src.llm_client import LLMClient


def _detect_type(prompt: str) -> str:
    """Quick keyword-based puzzle type detection (mirrors classify_node)."""
    lower = prompt.lower()
    for sig, ptype in PUZZLE_SIGNATURES.items():
        if sig in lower:
            return ptype
    return "unknown"


def _answers_match(answer, expected) -> bool:
    """Compare answers: exact string match, then numeric within 10^-2 absolute."""
    a, b = str(answer).strip(), str(expected).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-2 + 1e-9
    except (ValueError, TypeError):
        return False



BATCH_SIZE = 20


def _run_single(graph, row_id, prompt, expected, has_expected, verbose, idx, total):
    """Run a single puzzle through the graph. Returns (idx, entry, log_lines)."""
    lines = []
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
        output = {}
        lines.append(f"[{idx}/{total}] id={row_id} ... ERROR: {exc}")

    elapsed = time.time() - t0
    match = _answers_match(answer, expected) if has_expected else None
    status = f"{'MATCH' if match else 'MISS'} " if match is not None else ""
    lines.append(f"[{idx}/{total}] id={row_id} ... {status}done ({elapsed:.1f}s)")
    if match is False:
        lines.append(f"    expected: {expected}  |  actual: {answer}")
    if verbose and output:
        lines.extend(_format_dag_trace(output))

    entry = {"id": row_id, "answer": answer}
    if has_expected:
        entry["expected"] = expected
    return idx, entry, lines


def _format_dag_trace(output: dict) -> list[str]:
    """Format DAG trace as a list of strings (for deferred printing)."""
    lines = []
    dag = output.get("thought_dag") or []
    puzzle_type = output.get("puzzle_type", "unknown")
    retries = output.get("retries", 0)
    failure_log = output.get("failure_log") or []

    lines.append(f"    type={puzzle_type}  nodes={len(dag)}  retries={retries}")
    for node in dag:
        status = "OK" if node["answer"] is not None else "UNSOLVED"
        answer_preview = (node["answer"] or "")[:60].encode("ascii", "replace").decode()
        deps = ",".join(node["depends_on"]) if node["depends_on"] else "(root)"
        executor = f"tool:{node['tool']}" if node.get("tool") else "llm"
        lines.append(f"    [{status}] {node['id']} ({executor}) deps={deps} -> {answer_preview}")
    if failure_log:
        lines.append(f"    failures: {len(failure_log)}")
        for f in failure_log:
            lines.append(f"      - {f['node_id']}: {f['error'][:80]}")
    return lines


def run(
    dataset_path: str,
    output_path: str,
    limit: int | None = None,
    verbose: bool = False,
    types: list[str] | None = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    df = pd.read_csv(dataset_path)
    if types:
        allowed = set(types)
        mask = df["prompt"].apply(lambda p: _detect_type(p) in allowed)
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} rows matching types: {', '.join(types)}")
    if limit is not None:
        df = df.head(limit)

    llm = LLMClient(
        provider=LLM_PROVIDER,
        model_name=MODEL_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        deepseek_api_key=DEEPSEEK_API_KEY,
        deepseek_model=DEEPSEEK_MODEL,
        openrouter_api_key=OPEN_ROUTER_API_KEY,
        openrouter_model=OPEN_ROUTER_MODEL,
    )
    provider = llm._resolve_provider()
    model_display = {
        "ollama": MODEL_NAME,
        "openrouter": OPEN_ROUTER_MODEL,
        "deepseek": DEEPSEEK_MODEL,
    }.get(provider, MODEL_NAME)

    total = len(df)
    print(f"Loaded {total} rows from {dataset_path}")
    print(f"Provider: {provider}  |  Model: {model_display}")
    print(f"Batch size: {batch_size}")

    graph = build_graph(llm)
    has_expected = "answer" in df.columns

    results = [None] * total
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = df.iloc[batch_start:batch_end]
        batch_t0 = time.time()

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {}
            for i, (_, row) in enumerate(batch.iterrows()):
                idx = batch_start + i + 1
                expected = row["answer"] if has_expected else None
                fut = pool.submit(
                    _run_single, graph, row["id"], row["prompt"],
                    expected, has_expected, verbose, idx, total,
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                idx, entry, lines = fut.result()
                results[idx - 1] = entry
                for line in lines:
                    print(line)

        batch_elapsed = time.time() - batch_t0
        print(f"  batch {batch_start + 1}-{batch_end}/{total} done ({batch_elapsed:.1f}s)")

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
    parser.add_argument(
        "--types",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=None,
        help="Comma-separated puzzle types to run (e.g. gravity_physics,cipher_decryption)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of puzzles to run in parallel per batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()
    run(args.dataset, args.output, args.limit, args.verbose, args.types, args.batch_size)


if __name__ == "__main__":
    main()
