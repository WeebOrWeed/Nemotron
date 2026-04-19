from langgraph.graph import StateGraph, START, END

from src.state import GraphState
from src.classify import make_classify_node
from src.decompose import make_decompose_node
from src.solver import make_solve_next
from src.config import MAX_RETRIES
from src.llm_client import LLMClient


def _route_after_solve(state: GraphState) -> str:
    dag = state.get("thought_dag") or []
    unsolved = [n for n in dag if n["answer"] is None]
    retries = state.get("retries", 0)
    failure_log = state.get("failure_log") or []

    if not unsolved:
        return "done"

    if failure_log and retries <= MAX_RETRIES:
        return "retry"

    if failure_log and retries > MAX_RETRIES:
        return "done"

    return "continue"


def build_graph(llm: LLMClient):
    classify_node = make_classify_node(llm)
    decompose_node = make_decompose_node(llm)
    solve_next = make_solve_next(llm)

    graph = StateGraph(GraphState)

    graph.add_node("classify", classify_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("solve_next", solve_next)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "decompose")
    graph.add_edge("decompose", "solve_next")

    graph.add_conditional_edges(
        "solve_next",
        _route_after_solve,
        {
            "continue": "solve_next",
            "retry": "decompose",
            "done": END,
        },
    )

    return graph.compile()
