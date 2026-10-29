import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def scan_leakage(
    df: pd.DataFrame,
    target_col: str,
    id_threshold: float = 0.95,
    auc_threshold: float = 0.95,
) -> dict:
    n = len(df)
    perfect_predictors = []
    id_like = []
    constant_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]

        # 1. 全空列检测
        if series.isnull().all():
            constant_cols.append({
                "column": col,
                "risk": "全空列（100% 缺失），无任何信息量，建议直接删除。"
            })
            continue

        # 2. 常量列检测
        if series.nunique(dropna=True) <= 1:
            constant_cols.append({
                "column": col,
                "risk": "零方差常量列，对模型预测无任何信息增益，建议直接删除。"
            })
            continue

        # 3. 高基数 ID 检测
        nunique_ratio = series.nunique(dropna=True) / n
        if nunique_ratio > id_threshold and series.dtype.kind in "iO":
            id_like.append({
                "column": col,
                "unique_ratio": round(float(nunique_ratio), 3),
                "risk": "高基数 ID 类列，可能是主键/流水号，建模时应直接排除。",
            })

    target = df[target_col]
    target_encoded = np.full(n, np.nan)
    target_valid_mask = target.notna().to_numpy()

    if target.dtype.kind in 'O' or target.dtype.kind == 'b':
        target_encoded[target_valid_mask] = LabelEncoder().fit_transform(target[target_valid_mask].astype(str))
    else:
        target_encoded[target_valid_mask] = target[target_valid_mask].to_numpy()

    unique_targets = np.unique(target_encoded[~np.isnan(target_encoded)])

    if len(unique_targets) > 2:
        return {
            "id_like_columns": id_like,
            "perfect_predictors": [],
            "constant_columns": constant_cols,
            "skipped": True,
            "skip_reason": f"目标列有 {len(unique_targets)} 个类别（多分类/回归）。当前版本仅支持二分类。",
            "summary": f"⚠️ 跳过高 AUC 扫描。发现 {len(id_like)} 个高基数 ID，{len(constant_cols)} 个无效列。",
        }
    elif len(unique_targets) < 2:
        return {
            "id_like_columns": id_like,
            "perfect_predictors": [],
            "constant_columns": constant_cols,
            "skipped": True,
            "skip_reason": "目标列类别不足 2 个（或全为空），无法进行 AUC 扫描。",
            "summary": f"⚠️ 跳过扫描。发现 {len(id_like)} 个高基数 ID，{len(constant_cols)} 个无效列。",
        }

    candidate_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if series.isnull().all() or series.dtype.kind not in "if":
            continue

        series_arr = series.to_numpy()
        valid_mask = (~pd.isna(series_arr)) & (~np.isnan(target_encoded))

        if valid_mask.sum() > 30:
            X_feat = series_arr[valid_mask].astype(float)
            y_true = target_encoded[valid_mask]

            # 单类目标（该子集里只有 0 或只有 1）时 ROC-AUC 无定义，直接跳过，
            # 否则 sklearn 会刷 UndefinedMetricWarning 并返回 nan。
            if len(np.unique(X_feat)) > 1 and len(np.unique(y_true)) == 2:
                try:
                    auc = roc_auc_score(y_true, X_feat)
                    auc = max(auc, 1 - auc)
                    if auc >= auc_threshold:
                        perfect_predictors.append({
                            "column": col,
                            "single_feature_auc": round(float(auc), 3),
                            "risk": "预测能力异常高，高度疑似目标泄漏。",
                        })
                        candidate_cols.append(col)
                except Exception:
                    pass

    if len(perfect_predictors) > 1:
        perfect_predictors.sort(key=lambda x: -x["single_feature_auc"])
        corr_matrix = df[candidate_cols].corr().abs()
        dropped_cols = set()

        for i in range(len(perfect_predictors)):
            col_a = perfect_predictors[i]["column"]
            if col_a in dropped_cols:
                continue
            merged_count = 0
            for j in range(i + 1, len(perfect_predictors)):
                col_b = perfect_predictors[j]["column"]
                if col_b in dropped_cols:
                    continue
                if corr_matrix.loc[col_a, col_b] > 0.9:
                    dropped_cols.add(col_b)
                    merged_count += 1
            if merged_count > 0:
                perfect_predictors[i]["risk"] += f" （已合并 {merged_count} 个高度相关的同源特征）"

        perfect_predictors = [p for p in perfect_predictors if p["column"] not in dropped_cols]

    return {
        "id_like_columns": id_like,
        "perfect_predictors": perfect_predictors,
        "constant_columns": constant_cols,
        "skipped": False,
        "summary": f"发现 {len(id_like)} 个高基数 ID，{len(constant_cols)} 个无效列，以及 {len(perfect_predictors)} 个高 AUC 异常特征。",
    }
