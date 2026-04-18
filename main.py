import argparse
import os
import time

import pandas as pd

from src.classify import PUZZLE_SIGNATURES
from src.config import (
    MODEL_NAME, OLLAMA_BASE_URL, TRAIN_PATH, TEST_PATH, RESULTS_DIR,
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    HF_TOKEN, HF_MODEL,
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


def _print_dag_trace(output: dict) -> None:
    dag = output.get("thought_dag") or []
    puzzle_type = output.get("puzzle_type", "unknown")
    retries = output.get("retries", 0)
    failure_log = output.get("failure_log") or []

    print(f"    type={puzzle_type}  nodes={len(dag)}  retries={retries}")
    for node in dag:
        status = "OK" if node["answer"] is not None else "UNSOLVED"
        answer_preview = (node["answer"] or "")[:60].encode("ascii", "replace").decode()
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
    types: list[str] | None = None,
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
        openrouter_api_key=OPENROUTER_API_KEY,
        openrouter_model=OPENROUTER_MODEL,
        hf_token=HF_TOKEN,
        hf_model=HF_MODEL,
    )
    provider = llm._resolve_provider()

    model_display = {
        "ollama": MODEL_NAME,
        "openrouter": OPENROUTER_MODEL,
        "deepseek": DEEPSEEK_MODEL,
        "huggingface": HF_MODEL,
    }.get(provider, MODEL_NAME)

    print(f"Loaded {len(df)} rows from {dataset_path}")
    print(f"Provider: {provider}  |  Model: {model_display}")

    graph = build_graph(llm)

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
        if match is False:
            print(f"    expected: {expected}  |  actual: {answer}")

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
    parser.add_argument(
        "--types",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=None,
        help="Comma-separated puzzle types to run (e.g. gravity_physics,cipher_decryption)",
    )
    args = parser.parse_args()
    run(args.dataset, args.output, args.limit, args.verbose, args.types)


if __name__ == "__main__":
    main()
