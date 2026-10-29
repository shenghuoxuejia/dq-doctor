import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.imputation.mice import MICEData

def littles_mcar_test(df: pd.DataFrame) -> float | None:
    numeric_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    if numeric_df.empty or not numeric_df.isnull().any().any():
        return None
    try:
        mice = MICEData(numeric_df)
        return float(mice.mcar_test())
    except Exception:
        return None

def target_correlated_missingness(df: pd.DataFrame, target_col: str) -> list[dict]:
    results = []
    target = df[target_col]
    
    for col in df.columns:
        if col == target_col or not df[col].isnull().any():
            continue
            
        missing_indicator = df[col].isnull().astype(int)
        
        try:
            if target.dtype.kind in "if":
                valid_mask = target.notna()
                if target[valid_mask].nunique() <= 2:
                    contingency = pd.crosstab(missing_indicator[valid_mask], target[valid_mask])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    chi2, p, _, _ = stats.chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    effect = np.sqrt(chi2 / n) if n > 0 else 0
                    test = "cramers_v (binary target)"
                else:
                    r, p = stats.pointbiserialr(missing_indicator[valid_mask], target[valid_mask])
                    effect, test = abs(r), "pointbiserial"
            else:
                contingency = pd.crosstab(missing_indicator, target)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                n = contingency.sum().sum()
                effect = np.sqrt(chi2 / n) if n > 0 else 0
                test = "cramers_v"
                
            if p < 0.05:
                results.append({
                    "column": col,
                    "missing_rate": float(df[col].isnull().mean()),
                    "effect_size": float(effect),
                    "p_value": float(p),
                    "test": test,
                    "interpretation": (
                        f"缺失指示变量与 {target_col} 显著关联"
                        f"（{test}, effect={effect:.3f}, p={p:.4f}）。"
                        "该缺失是信息性的（MNAR嫌疑），删除缺失行可能引入选择偏差。"
                    ),
                })
        except Exception:
            continue
            
    return sorted(results, key=lambda x: -x["effect_size"])

def diagnose_missingness(df: pd.DataFrame, target_col: str) -> dict:
    mcar_p = littles_mcar_test(df)
    target_corr = target_correlated_missingness(df, target_col)
    
    per_column = (
        df.isnull().mean()
        .sort_values(ascending=False)
        .loc[lambda s: s > 0]
        .to_dict()
    )
    
    # ===== 修改在这里：加入对“完全无缺失”的判断 =====
    if not per_column:
        mechanism_global = "数据完美，无任何缺失值"
    elif mcar_p is None:
        mechanism_global = "有缺失值，但无法执行 MCAR 检验（数据类型不支持或样本太少）"
    elif mcar_p > 0.05:
        mechanism_global = "不能拒绝 MCAR（数据可能完全随机缺失）"
    else:
        mechanism_global = "拒绝 MCAR（提示非随机缺失 MAR/MNAR）"
        
    return {
        "per_column_missing_rate": per_column,
        "mcar_p_value": mcar_p,
        "mechanism_global": mechanism_global,
        "target_correlated": target_corr,
    }