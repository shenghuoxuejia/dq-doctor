"""界面冒烟测试：用 Streamlit 官方 AppTest 真跑一遍脚本。

覆盖目标：脚本能在「无 API Key」环境下完整渲染首屏而不抛异常——
这正是改版前会因为模块级 ChatOpenAI 直接崩溃的场景。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
