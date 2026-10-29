"""synthesize_node 的降级路径与确定性规则测试。

重点覆盖「没有 LLM 也能用」这一行为，因为它决定了 CI 与陌生评审者
能否在不配置 API Key 的情况下跑通整个项目。
"""

from pathlib import Path

import pandas as pd

import dq_doctor.nodes as nodes

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _unhealthy_diagnoses() -> dict:
    return {
        "missingness": {
            "mechanism_global": "拒绝 MCAR",
            "mcar_p_value": 0.0,
            "per_column_missing_rate": {"income": 0.2},
            "target_correlated": [{"column": "income", "effect_size": 0.42, "p_value_corrected": 0.001}],
        },
        "leakage": {
            "skipped": False,
            "summary": "发现 1 个高风险特征",
            "constant_columns": [],
            "id_like_columns": [],
            "perfect_predictors": [{"column": "leak_score", "single_feature_auc": 1.0}],
        },
    }


def _healthy_diagnoses() -> dict:
    return {
        "missingness": {
            "mechanism_global": "数据完美，无任何缺失值",
            "mcar_p_value": None,
            "per_column_missing_rate": {"a": 0.0, "b": 0.0},
            "target_correlated": [],
        },
        "leakage": {
            "skipped": False,
            "summary": "未发现明显泄漏",
            "constant_columns": [],
            "id_like_columns": [],
            "perfect_predictors": [],
        },
    }


# --------------------------------------------------------------------------- #
# 惰性构造：没有 Key 时不应尝试建立客户端
# --------------------------------------------------------------------------- #
def test_get_llm_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert nodes._get_llm() is None


def test_import_does_not_require_api_key(monkeypatch):
    """模块级不应持有 LLM 实例——这是「无 Key 也能 import」的根因。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert not hasattr(nodes, "llm")


# --------------------------------------------------------------------------- #
# 规则引擎：没有 Key 时给出可用结论
# --------------------------------------------------------------------------- #
def test_offline_fallback_returns_actionable_report(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    out = nodes.synthesize_node({"diagnoses": _unhealthy_diagnoses()})

    text = out["messages"][0][1]
    assert "规则引擎" in text
    assert "income" in text          # 缺失嫌疑列被点出
    assert "leak_score" in text      # 泄漏候选列被点出
    assert out["severity_order"][0] == "leakage"


def test_offline_fallback_orders_missingness_first_when_it_dominates(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    diagnoses = _unhealthy_diagnoses()
    diagnoses["leakage"]["perfect_predictors"] = []

    out = nodes.synthesize_node({"diagnoses": diagnoses})
    assert out["severity_order"][0] == "missingness"


def test_healthy_data_short_circuits_without_llm(monkeypatch):
    """数据干净时直接返回确定性结论，连 _get_llm 都不该被调用。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(nodes, "_get_llm", lambda: calls.append(1))

    out = nodes.synthesize_node({"diagnoses": _healthy_diagnoses()})
    assert out["severity_order"] == ["healthy", "missingness", "leakage"]
    assert "健康体检" in out["messages"][0][1]
    assert calls == []


def test_skipped_leakage_scan_is_flagged(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    diagnoses = _unhealthy_diagnoses()
    diagnoses["leakage"] = {"skipped": True, "skip_reason": "目标列有 3 个类别", "perfect_predictors": []}

    out = nodes.synthesize_node({"diagnoses": diagnoses})
    text = out["messages"][0][1]
    assert "未执行" in text
    assert "3 个类别" in text


def test_empty_diagnoses_does_not_crash(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    out = nodes.synthesize_node({"diagnoses": {}})
    assert out["messages"][0][1]
    assert out["severity_order"]


# --------------------------------------------------------------------------- #
# LLM 输出解析的健壮性
# --------------------------------------------------------------------------- #
def test_extract_json_handles_code_fence():
    raw = '```json\n{"severity_order": ["leakage"], "top_insight": "hi"}\n```'
    assert nodes._extract_json(raw)["top_insight"] == "hi"


def test_extract_json_handles_surrounding_prose():
    raw = '好的，分析如下：\n{"severity_order": ["missingness"], "top_insight": "x"}\n以上。'
    assert nodes._extract_json(raw)["severity_order"] == ["missingness"]


def test_extract_json_raises_without_object():
    try:
        nodes._extract_json("抱歉，我无法完成。")
    except ValueError:
        return
    raise AssertionError("预期抛出 ValueError")


# --------------------------------------------------------------------------- #
# 端到端：没有 Key 时整条图仍能跑通
# --------------------------------------------------------------------------- #
def test_graph_runs_end_to_end_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from dq_doctor.graph import build_graph

    csv = DATA_DIR / "toxic_data.csv"
    if not csv.exists():  # 数据集未随仓库分发时跳过
        return
    df = pd.read_csv(csv)

    graph = build_graph()
    res = graph.invoke(
        {"df": df, "target_col": "target", "diagnoses": {}, "messages": []},
        config={"configurable": {"thread_id": "test_offline_e2e"}},
    )

    assert res["messages"]
    assert res["messages"][-1].content
    assert "leakage" in res["severity_order"]
