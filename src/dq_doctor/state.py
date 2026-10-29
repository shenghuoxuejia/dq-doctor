import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class DQState(TypedDict):
    messages: Annotated[list, add_messages]
    df: Any
    target_col: str | None
    time_col: str | None
    profile: dict # 将在里面存重复行数
    diagnoses: Annotated[dict[str, dict], operator.or_]
    severity_order: list[str]
    remediation: dict | None
    human_approved: bool | None
