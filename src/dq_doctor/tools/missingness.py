import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def littles_mcar_test(df: pd.DataFrame) -> float | None:
    """
    手写 Little's MCAR 检验。
    修正了自由度计算：df_chi = Σ p_j - p
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    if numeric_df.empty or not numeric_df.isnull().any().any():
        return None

    try:
        mu = numeric_df.mean().values
        cov = numeric_df.cov().values
        p = numeric_df.shape[1]

        patterns = numeric_df.isnull().apply(lambda x: tuple(x), axis=1)
        d2 = 0.0
        sum_p_j = 0  # 记录所有模式中观测到的变量总数

        for pattern, group in numeric_df.groupby(patterns):
            observed_idx = [i for i, val in enumerate(pattern) if not val]
            p_j = len(observed_idx)
            if p_j == 0:
                continue

            sum_p_j += p_j  # 累加

            mu_obs = mu[observed_idx]
            cov_obs = cov[np.ix_(observed_idx, observed_idx)]

            if np.linalg.det(cov_obs) == 0:
                continue

            cov_obs_inv = np.linalg.pinv(cov_obs)
            diff = group.iloc[:, observed_idx].mean().values - mu_obs
            n_j = len(group)
            d2 += n_j * np.dot(np.dot(diff, cov_obs_inv), diff.T)

        # 修正：自由度 = 观测变量总数 - 变量总数
        df_chi = sum_p_j - p
        if df_chi <= 0:
            return None

        p_value = 1 - stats.chi2.cdf(d2, df_chi)
        return float(p_value)
    except Exception:
        return None

def target_correlated_missingness(df: pd.DataFrame, target_col: str) -> list[dict]:
    """
    对每一列进行检验，收集所有 p 值，整体做 FDR 校正。
    """
    target = df[target_col]
    all_tests = [] # 记录所有参与检验的结果

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

            # 收集所有合法检验结果，不管 p 值大小
            all_tests.append({
                "column": col,
                "missing_rate": float(df[col].isnull().mean()),
                "effect_size": float(effect),
                "p_value": float(p),
                "test": test,
            })
        except Exception:
            continue

    if not all_tests:
        return []

    # 整体 FDR 校正 (Benjamini-Hochberg)
    p_values = [res["p_value"] for res in all_tests]
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    final_results = []
    for i, res in enumerate(all_tests):
        if reject[i]:  # 只保留校正后依然显著的
            res["p_value_corrected"] = float(pvals_corrected[i])
            res["interpretation"] = (
                f"缺失指示变量与 {target_col} 显著关联"
                f"（{res['test']}, effect={res['effect_size']:.3f}, FDR校正后 p={pvals_corrected[i]:.4f}）。"
                "该缺失是信息性的（MNAR嫌疑），删除缺失行可能引入选择偏差。"
            )
            # 清理中间变量
            del res["test"]
            del res["p_value"]
            final_results.append(res)

    return sorted(final_results, key=lambda x: -x["effect_size"])

def diagnose_missingness(df: pd.DataFrame, target_col: str) -> dict:
    mcar_p = littles_mcar_test(df)
    target_corr = target_correlated_missingness(df, target_col)

    per_column = (
        df.isnull().mean()
        .sort_values(ascending=False)
        .loc[lambda s: s > 0]
        .to_dict()
    )

    if not per_column:
        mechanism_global = "数据完美，无任何缺失值"
    elif mcar_p is None:
        mechanism_global = "有缺失值，但无法执行 MCAR 检验"
    elif mcar_p > 0.05:
        mechanism_global = "不能拒绝 MCAR（数据可能完全随机缺失）"
    else:
        mechanism_global = "拒绝 MCAR（数据存在非随机缺失特征，需进一步排查 MAR/MNAR）"

    return {
        "per_column_missing_rate": per_column,
        "mcar_p_value": mcar_p,
        "mechanism_global": mechanism_global,
        "target_correlated": target_corr,
    }
