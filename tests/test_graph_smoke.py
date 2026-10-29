import numpy as np
import pandas as pd

from dq_doctor.graph import build_graph


def test_graph_outputs_message():
    # 这里用最简单的数据
    df = pd.DataFrame({'target': [0, 1] * 50, 'feature': np.random.randn(100)})
    graph = build_graph()

    # 使用 mock 的 thread_id 防止冲突
    res = graph.invoke(
        {'df': df, 'target_col': 'target', 'diagnoses': {}, 'messages': []},
        config={'configurable': {'thread_id': 'test_smoke_1'}}
    )

    # 断言：必须产生非空的 messages，且 severity_order 必须存在
    assert len(res['messages']) > 0
    assert res['messages'][0].content != ""
    assert res['severity_order'] is not None
