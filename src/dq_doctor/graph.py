import pickle
from typing import Any, Tuple

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.graph import END, START, StateGraph

from .nodes import synthesize_node  # <--- 重新接入大脑
from .state import DQState
from .tools.leakage import scan_leakage
from .tools.missingness import diagnose_missingness


class CustomSerializer(SerializerProtocol):
    def dumps(self, obj: Any) -> bytes:
        if isinstance(obj, pd.DataFrame):
            return pickle.dumps(("DataFrame", obj))
        return pickle.dumps(obj)
    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        if isinstance(obj, pd.DataFrame):
            return "DataFrame", pickle.dumps(obj)
        return "pickle", pickle.dumps(obj)
    def loads(self, data: bytes) -> Any:
        obj = pickle.loads(data)
        if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "DataFrame":
            return obj[1]
        return obj
    def loads_typed(self, data: Tuple[str, bytes]) -> Any:
        return pickle.loads(data[1])

def profile_node(state: DQState) -> dict:
    df = state["df"]
    target_col = state.get("target_col")
    duplicate_count = df.duplicated().sum()

    return {
        "profile": {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "has_target": target_col is not None and target_col in df.columns,
            "duplicate_rows": int(duplicate_count),
            "columns": list(df.columns),
        }
    }

def missingness_node(state: DQState) -> dict:
    target_col = state.get("target_col")
    if not target_col:
        return {"diagnoses": {"missingness": {"error": "未提供目标列"}}}
    return {"diagnoses": {"missingness": diagnose_missingness(state["df"], target_col)}}

def leakage_node(state: DQState) -> dict:
    target_col = state.get("target_col")
    if not target_col:
        return {"diagnoses": {"leakage": {"error": "未提供目标列"}}}
    return {"diagnoses": {"leakage": scan_leakage(state["df"], target_col)}}

def build_graph():
    builder = StateGraph(DQState)
    builder.add_node("profile", profile_node)
    builder.add_node("missingness", missingness_node)
    builder.add_node("leakage", leakage_node)
    builder.add_node("synthesize", synthesize_node) # <--- 加回节点

    builder.add_edge(START, "profile")
    builder.add_edge("profile", "missingness")
    builder.add_edge("missingness", "leakage")
    builder.add_edge("leakage", "synthesize")       # <--- 连向大脑
    builder.add_edge("synthesize", END)             # <--- 结束

    return builder.compile(checkpointer=MemorySaver(serde=CustomSerializer()))
