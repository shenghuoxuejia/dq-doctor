import operator
from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages

class DQState(TypedDict):
    messages: Annotated[list, add_messages]
    df: Any
    target_col: str | None
    time_col: str | None
    profile: dict
    
    # 关键修改：使用 Annotated 和 operator.or_（字典合并）
    # 这告诉 LangGraph：如果有新结果，请把它和旧结果合并，而不是直接覆盖！
    diagnoses: Annotated[dict[str, dict], operator.or_]
    
    severity_order: list[str]
    remediation: dict | None
    human_approved: bool | None