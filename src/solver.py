from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.state import GraphState, ThoughtNode, FailureRecord
from src.config import MAX_RETRIES
from src.tools import run_tool
from src.llm_client import LLMClient

SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Output ONLY your final answer on the last line with no extra text."
)


def _interpolate_parents(template: str, dag: list[ThoughtNode]) -> str:
    """Replace {parent_id} placeholders with parent answers."""
    answered = {n["id"]: (n["answer"] or "") for n in dag if n["answer"] is not None}
    for node_id, answer in answered.items():
        template = template.replace("{" + node_id + "}", answer)
    return template


def _extract_final_answer(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _solve_single_node(
    llm: LLMClient,
    node: ThoughtNode,
    dag: list[ThoughtNode],
    original_prompt: str,
) -> str:
    tool_name = node.get("tool")
    if not tool_name or tool_name == "ask_llm":
        return _run_llm_node(llm, node, dag, original_prompt)
    try:
        return _run_tool_node(tool_name, node, dag)
    except Exception:
        if tool_name == "solve_equation_transform":
            return _eq_transform_llm_fallback(llm, node, dag, original_prompt)
        return _run_llm_node(llm, node, dag, original_prompt)


def _run_tool_node(tool_name: str, node: ThoughtNode, dag: list[ThoughtNode]) -> str:
    raw_input = node.get("tool_input") or "{}"
    raw_input = _interpolate_parents(raw_input, dag)
    return run_tool(tool_name, raw_input)


def _run_llm_node(
    llm: LLMClient,
    node: ThoughtNode,
    dag: list[ThoughtNode],
    original_prompt: str,
) -> str:
    question = node["question"]
    question = _interpolate_parents(question, dag)

    full_question = (
        f"Original puzzle:\n{original_prompt}\n\n"
        f"Your task:\n{question}"
    )

    n_votes = int(node.get("tool_input") or "1") if node.get("tool") == "majority_vote_llm" else 1

    if n_votes > 1:
        return _majority_vote(llm, full_question, n_votes)

    resp = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_question},
        ],
        think=True,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )
    text = (resp.content or "").strip()
    if not text:
        raise ValueError("Empty response from model")
    # Keep full text for DAG parents; final graph answer uses last line via
    # _extract_final_answer on the sink only.
    return text


def _eq_transform_llm_fallback(
    llm: LLMClient,
    node: ThoughtNode,
    dag: list[ThoughtNode],
    original_prompt: str,
) -> str:
    """Specialized LLM fallback for equation_transform: majority vote with 5 calls."""
    question = node["question"]
    question = _interpolate_parents(question, dag)
    full_question = (
        f"Original puzzle:\n{original_prompt}\n\n"
        f"Your task:\n{question}"
    )
    return _majority_vote(llm, full_question, 5)


def _majority_vote(
    llm: LLMClient,
    full_question: str,
    n_votes: int,
) -> str:
    """Make multiple LLM calls with varying temperatures and return the most common answer."""
    from collections import Counter
    temps = [0.3, 0.5, 0.7, 0.9, 1.0, 0.4, 0.6, 0.8][:n_votes]
    answers: list[str] = []
    for temp in temps:
        try:
            resp = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_question},
                ],
                think=True,
                temperature=temp,
                top_p=0.95,
                max_tokens=4096,
            )
            ans = _extract_final_answer(resp.content)
            if ans:
                answers.append(ans)
        except Exception:
            continue
    if not answers:
        raise ValueError("All majority vote attempts failed")
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def make_solve_next(llm: LLMClient):
    """Factory that returns the solve_next node function with the LLM client bound."""

    def solve_next(state: GraphState) -> dict:
        dag: list[ThoughtNode] = list(state["thought_dag"])
        failure_log: list[FailureRecord] = list(state.get("failure_log") or [])
        retries: int = state.get("retries", 0)
        original_prompt: str = state["prompt"]

        answered_ids = {n["id"] for n in dag if n["answer"] is not None}

        ready = [
            n for n in dag
            if n["answer"] is None
            and all(p in answered_ids for p in n["depends_on"])
        ]

        if not ready:
            unsolved = [n for n in dag if n["answer"] is None]
            if unsolved:
                for n in unsolved:
                    failure_log.append(FailureRecord(
                        node_id=n["id"],
                        question=n["question"][:200],
                        error="No ready nodes; possible broken dependency",
                    ))
                return {
                    "thought_dag": dag,
                    "failure_log": failure_log,
                    "retries": retries + 1,
                }
            sink = _find_sink(dag)
            raw = sink.get("answer") or ""
            return {
                "thought_dag": dag,
                "answer": _extract_final_answer(raw) or raw,
            }

        failures_this_batch: list[FailureRecord] = []

        with ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            future_to_node = {
                pool.submit(
                    _solve_single_node, llm, n, dag, original_prompt
                ): n
                for n in ready
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    answer = future.result(timeout=120)
                    for n in dag:
                        if n["id"] == node["id"]:
                            n["answer"] = answer
                            break
                except Exception as exc:
                    failures_this_batch.append(FailureRecord(
                        node_id=node["id"],
                        question=node["question"][:200],
                        error=str(exc)[:300],
                    ))

        if failures_this_batch:
            failure_log.extend(failures_this_batch)
            return {
                "thought_dag": dag,
                "failure_log": failure_log,
                "retries": retries + 1,
            }

        unsolved = [n for n in dag if n["answer"] is None]
        if not unsolved:
            sink = _find_sink(dag)
            raw = sink.get("answer") or ""
            return {
                "thought_dag": dag,
                "answer": _extract_final_answer(raw) or raw,
            }

        return {"thought_dag": dag, "failure_log": failure_log}

    return solve_next


def _find_sink(dag: list[ThoughtNode]) -> ThoughtNode:
    """Find the sink node (the node that no other node depends on)."""
    all_ids = {n["id"] for n in dag}
    depended_on = set()
    for n in dag:
        depended_on.update(n["depends_on"])
    sink_ids = all_ids - depended_on
    for n in reversed(dag):
        if n["id"] in sink_ids:
            return n
    return dag[-1]
