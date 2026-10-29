import numpy as np
import pandas as pd

from dq_doctor.tools.missingness import diagnose_missingness, littles_mcar_test


def test_mcar_rejects_non_random():
    # 生成一个含信息性缺失的数据集
    np.random.seed(42)
    n = 500
    target = np.random.binomial(1, 0.5, n)
    # 当 target=1 时，有 80% 的概率缺失
    feature = np.where(target == 1, np.where(np.random.rand(n) < 0.8, np.nan, 1.0), 1.0)
    df = pd.DataFrame({'target': target, 'feature': feature})

    p_val = littles_mcar_test(df)
    # 断言：p 值应该远小于 0.05（拒绝 MCAR）
    assert p_val is not None
    assert p_val < 0.05

def test_missingness_interpretation():
    df = pd.DataFrame({'target': [1, 1, 0, 0, 1], 'f1': [np.nan, 1, 1, 1, np.nan]})
    res = diagnose_missingness(df, 'target')
    # 只要不报错且包含必备 key 即可
    assert "mcar_p_value" in res
    assert "mechanism_global" in res
