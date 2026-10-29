import numpy as np
import pandas as pd

from dq_doctor.tools.leakage import scan_leakage


def test_offset_index_leakage():
    """测试索引偏移情况下，泄漏扫描是否依然准确"""
    np.random.seed(42)
    n = 500
    target = np.random.binomial(1, 0.5, n)
    leak = target + np.random.randn(n) * 0.01
    df = pd.DataFrame({'target': target, 'leak_col': leak})

    # 正常索引
    res_normal = scan_leakage(df, 'target')

    # 制造索引偏移（比如删除了前 100 行，或者 set_index）
    df_offset = df.copy()
    df_offset.index = np.arange(1000, 1500)
    res_offset = scan_leakage(df_offset, 'target')

    # 断言：无论索引怎么变，抓到的异常特征数量必须一致，且都为 1
    assert len(res_normal['perfect_predictors']) == 1
    assert len(res_offset['perfect_predictors']) == 1
    assert res_offset['perfect_predictors'][0]['column'] == 'leak_col'

def test_string_target_with_nan():
    """测试目标列含 NaN 的字符串情况"""
    df = pd.DataFrame({
        'target': ['Yes', 'No', np.nan, 'Yes', 'No'] * 100,
        'feature': [1, 0, 1, 1, 0] * 100
    })
    res = scan_leakage(df, 'target')
    # 哪怕有 NaN，也应该能抓出 feature 是异常特征
    assert len(res['perfect_predictors']) == 1
def test_clustering_and_merge_info():
    """测试高相关性聚类，并验证保留的是 AUC 最高的，且带有 merged 信息"""
    n = 200
    target = np.random.binomial(1, 0.5, n)
    # f1, f2 都泄漏，且高度相关
    f1 = target + np.random.randn(n) * 0.01
    f2 = target + np.random.randn(n) * 0.05
    df = pd.DataFrame({'target': target, 'f1': f1, 'f2': f2})

    res = scan_leakage(df, 'target')
    # 因为 f1 噪声小，AUC 更高，所以保留 f1，删掉 f2
    assert len(res['perfect_predictors']) == 1
    assert res['perfect_predictors'][0]['column'] == 'f1'
    assert "已合并 1 个" in res['perfect_predictors'][0]['risk']

def test_multi_class_skipped():
    """测试多分类目标能否显式跳过"""
    df = pd.DataFrame({'target': [0, 1, 2] * 50, 'leak': [0, 1, 2] * 50})
    res = scan_leakage(df, 'target')
    assert res['skipped'] is True
    assert res['perfect_predictors'] == []

def test_constant_column_detected():
    """测试常量列能否被揪出"""
    df = pd.DataFrame({'target': [0, 1] * 50, 'const_col': [5] * 100})
    res = scan_leakage(df, 'target')
    assert len(res['constant_columns']) == 1
    assert res['constant_columns'][0]['column'] == 'const_col'
