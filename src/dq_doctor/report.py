"""把工具层返回的结构化条目渲染成「列名 + 解释」的可读文本。

这些条目的共同要求是**必须带列名**：只告诉用户「发现 1 个高风险特征」而不说是哪一列，
用户无法采取任何行动。把格式化逻辑单独放一层，界面代码只负责摆放，
也让这段契约可以被单元测试覆盖（见 tests/test_report.py）。
"""

from __future__ import annotations

from typing import Any, Mapping


def _column(item: Mapping[str, Any]) -> str:
    return str(item.get("column") or "?")


def _number(value: Any, fmt: str, prefix: str, suffix: str = "") -> str:
    """数值存在时拼成「prefix+数字+suffix」，否则返回空串。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{prefix}{value:{fmt}}{suffix}"
    return ""


def format_constant_item(item: Mapping[str, Any]) -> str:
    """全空列 / 常量列：`列名 — 风险`。"""
    return f"{_column(item)} — {item.get('risk', '')}"


def format_id_item(item: Mapping[str, Any]) -> str:
    """高基数 ID 列：附上唯一值占比，便于判断是不是主键。"""
    ratio = _number(item.get("unique_ratio"), ".1%", "，唯一值占比 ")
    return f"{_column(item)}{ratio} — {item.get('risk', '')}"


def format_perfect_item(item: Mapping[str, Any]) -> str:
    """高单变量 AUC 特征：附上 AUC，便于和 0.95 阈值对照。"""
    auc = _number(item.get("single_feature_auc"), ".3f", "（AUC=", "）")
    return f"{_column(item)}{auc} — {item.get('risk', '')}"
