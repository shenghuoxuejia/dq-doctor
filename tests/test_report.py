from dq_doctor.report import format_constant_item, format_id_item, format_perfect_item


def test_perfect_item_carries_column_and_auc():
    item = {"column": "leak_score", "single_feature_auc": 1.0, "risk": "疑似目标泄漏。"}
    text = format_perfect_item(item)
    assert text.startswith("leak_score（AUC=1.000）")
    assert "疑似目标泄漏。" in text


def test_perfect_item_without_auc():
    text = format_perfect_item({"column": "x", "risk": "r"})
    assert text == "x — r"


def test_id_item_carries_unique_ratio():
    item = {"column": "customer_id", "unique_ratio": 0.998, "risk": "主键列。"}
    assert format_id_item(item).startswith("customer_id，唯一值占比 99.8% — ")


def test_constant_item_carries_column():
    assert format_constant_item({"column": "all_null", "risk": "全空列。"}) == "all_null — 全空列。"


def test_missing_column_falls_back_to_placeholder():
    """条目里没有列名时不能渲染成空字符串，否则用户看到一条无主文案。"""
    assert format_perfect_item({"risk": "r"}).startswith("? — ")


def test_bool_is_not_treated_as_number():
    """True 是 int 的子类，不能被格式化成一个百分比。"""
    assert format_id_item({"column": "c", "unique_ratio": True, "risk": "r"}) == "c — r"
