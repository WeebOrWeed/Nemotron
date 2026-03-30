from typing import Optional
from typing_extensions import TypedDict


class ThoughtNode(TypedDict):
    id: str
    question: str
    depends_on: list[str]
    tool: Optional[str]
    tool_input: Optional[str]
    answer: Optional[str]


class FailureRecord(TypedDict):
    node_id: str
    question: str
    error: str


class GraphState(TypedDict):
    prompt: str
    puzzle_type: Optional[str]
    thought_dag: Optional[list[ThoughtNode]]
    retries: int
    failure_log: Optional[list[FailureRecord]]
    answer: Optional[str]
