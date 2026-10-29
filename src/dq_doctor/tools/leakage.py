import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def scan_leakage(
    df: pd.DataFrame,
    target_col: str,
    id_threshold: float = 0.95,
    auc_threshold: float = 0.95,
) -> dict:
    """
    扫描数据集中可能导致“目标泄漏”的特征。
    1. 极高基数的 ID 类特征（唯一值超过 95%）。
    2. 单特征预测能力极强（AUC >= 0.95）的数值特征。
    """
    target = df[target_col]
    n = len(df)
    perfect_predictors = []
    id_like = []

    for col in df.columns:
        if col == target_col:
            continue
            
        series = df[col]
        # 忽略全空的列
        if series.isnull().all():
            continue
            
        nunique_ratio = series.nunique(dropna=True) / n
        
        # 1. 扫描 ID 类特征（高基数）
        if nunique_ratio > id_threshold and series.dtype.kind in "iO":
            id_like.append({
                "column": col,
                "unique_ratio": round(float(nunique_ratio), 3),
                "risk": "高基数 ID 类列，可能是主键/流水号，建模时应直接排除。",
            })
            
        # 2. 扫描单特征高 AUC（强泄漏嫌疑）
        if series.dtype.kind in "if" and target.dtype.kind in "if":
            valid = series.notna() & target.notna()
            if valid.sum() > 30 and series[valid].nunique() > 1:
                try:
                    # 将 target 编码为 0/1 计算 AUC
                    y_true = LabelEncoder().fit_transform(target[valid])
                    if len(np.unique(y_true)) == 2: # 确保是二分类
                        auc = roc_auc_score(y_true, series[valid])
                        # 无论是正相关还是负相关，取绝对值最大的 AUC
                        auc = max(auc, 1 - auc) 
                        if auc >= auc_threshold:
                            perfect_predictors.append({
                                "column": col,
                                "single_feature_auc": round(float(auc), 3),
                                "risk": f"单特征 AUC={auc:.3f}。预测能力异常高，疑似目标泄漏或后见之明偏差。",
                            })
                except Exception:
                    pass

    return {
        "id_like_columns": id_like,
        "perfect_predictors": sorted(
            perfect_predictors, key=lambda x: -x["single_feature_auc"]
        ),
        "summary": (
            f"发现 {len(id_like)} 个高基数 ID 类列，"
            f"{len(perfect_predictors)} 个单特征预测能力异常（AUC≥{auc_threshold}）的特征。"
        ),
    }