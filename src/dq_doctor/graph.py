import pickle
import pandas as pd
from .nodes import synthesize_node
from typing import Any, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from .state import DQState
from .tools.missingness import diagnose_missingness
from .tools.leakage import scan_leakage

class CustomSerializer(SerializerProtocol):
    """自定义序列化器，专门处理 Pandas DataFrame"""
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
        type_str, bytes_data = data
        if type_str == "DataFrame":
            return pickle.loads(bytes_data)
        return pickle.loads(bytes_data)


def profile_node(state: DQState) -> dict:
    df = state["df"]
    target_col = state.get("target_col")
    return {
        "profile": {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "has_target": target_col is not None and target_col in df.columns,
            "columns": list(df.columns),
        }
    }

def missingness_node(state: DQState) -> dict:
    target_col = state.get("target_col")
    if not target_col:
        return {"diagnoses": {"missingness": {"error": "未提供目标列"}}}
    report = diagnose_missingness(state["df"], target_col)
    return {"diagnoses": {"missingness": report}}

def leakage_node(state: DQState) -> dict:
    target_col = state.get("target_col")
    if not target_col:
        return {"diagnoses": {"leakage": {"error": "未提供目标列"}}}
    report = scan_leakage(state["df"], target_col)
    return {"diagnoses": {"leakage": report}}

def build_graph():
    builder = StateGraph(DQState)
    
    builder.add_node("profile", profile_node)
    builder.add_node("missingness", missingness_node)
    builder.add_node("leakage", leakage_node)
    # 新增节点
    builder.add_node("synthesize", synthesize_node)
    
    builder.add_edge(START, "profile")
    builder.add_edge("profile", "missingness")
    builder.add_edge("missingness", "leakage")
    # 修改连线：leakage -> synthesize -> END
    builder.add_edge("leakage", "synthesize")
    builder.add_edge("synthesize", END)
    
    # 保留你的自定义序列化器
    return builder.compile(checkpointer=MemorySaver(serde=CustomSerializer()))