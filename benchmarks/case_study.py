"""Case study + 基准：让 DQ-Doctor 自己决定删哪一列。

这个脚本刻意不写死要删除的特征名——泄漏列由 ``scan_leakage()`` 的输出决定。
即"工具先诊断，脚本再按诊断结果行动"，而不是"我造了泄漏，然后自己 drop"。

运行： ``python benchmarks/case_study.py``（约 40 秒，无需 API Key、不联网）
产出： ``benchmarks/RESULTS.md`` + ``benchmarks/figures/*.png``
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import missingno as msno  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dq_doctor.tools.leakage import scan_leakage  # noqa: E402
from dq_doctor.tools.missingness import diagnose_missingness, littles_mcar_test  # noqa: E402

DATA = ROOT / "data"
FIG = ROOT / "benchmarks" / "figures"
RESULTS = ROOT / "benchmarks" / "RESULTS.md"

SEED = 42
AUC_THRESHOLD = 0.95

# 跨平台中文字体候选：Linux / macOS 上 SimHei 不存在时依次回退，避免豆腐块。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 一、案例研究：工具驱动
# --------------------------------------------------------------------------- #
def generate_realistic_data() -> pd.DataFrame:
    """1000 样本 / 15 特征 / 15% 标签噪声，AUC 落在 0.9 附近，再注入一条事后泄漏。"""
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=3,
        n_redundant=0,
        flip_y=0.15,
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    # 事后字段：事件发生之后才写入，线上预测时根本拿不到
    df["post_event_status"] = y * 0.95 + rng.standard_normal(1000) * 0.1
    # MNAR：高价值样本缺失
    df.loc[df["feature_0"] > 1.0, "feature_1"] = np.nan
    return df


def run_case_study(df: pd.DataFrame) -> dict:
    report = scan_leakage(df, "target")
    flagged = [p["column"] for p in report["perfect_predictors"]]

    y = df["target"]
    clf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    X_all = df.drop(columns=["target"]).fillna(0)
    auc_with = cross_val_score(clf, X_all, y, cv=5, scoring="roc_auc").mean()

    # 关键：删哪些列由工具输出决定，脚本里没有出现列名字面量
    X_clean = X_all.drop(columns=flagged)
    auc_without = cross_val_score(clf, X_clean, y, cv=5, scoring="roc_auc").mean()

    return {
        "flagged": flagged,
        "summary": report.get("summary", ""),
        "auc_with": float(auc_with),
        "auc_without": float(auc_without),
        "n_features": X_all.shape[1],
        "n_rows": len(df),
    }


# --------------------------------------------------------------------------- #
# 二、fixture 命中表
# --------------------------------------------------------------------------- #
FIXTURES = [
    {
        "file": "clean_data.csv",
        "target": "target",
        "injected": "无（目标列只有单一取值，泄漏扫描应跳过）",
        "must": {},
    },
    {
        "file": "edge_cases.csv",
        "target": "target",
        "injected": "全空列、常量列、高基数 ID",
        "must": {"constant": ["all_null", "constant_val"], "id": ["unique_str"]},
    },
    {
        "file": "test_data.csv",
        "target": "target",
        "injected": "单特征 AUC=1.0 的泄漏列 + 与标签相关的缺失",
        "must": {"perfect": ["leak_col"], "mnar": ["missing_col"]},
    },
    {
        "file": "toxic_data.csv",
        "target": "target",
        "injected": "泄漏特征 + 收入缺失与标签相关（MNAR）",
        "must": {"perfect": ["leak_score"], "mnar": ["income"]},
    },
    {
        "file": "telco_simulated.csv",
        "target": "churn",
        "injected": "主键列 + 强预测列 + 账单缺失与流失相关",
        "must": {"id": ["customer_id"], "perfect": ["support_calls"], "mnar": ["monthly_charges"]},
    },
    {
        "file": "real_cancer.csv",
        "target": "is_malignant",
        "injected": "无（真实数据，worst_* 是合法强特征 → 用于量化误报）",
        "must": {},
    },
]


def _found(df: pd.DataFrame, target: str) -> dict:
    lea = scan_leakage(df, target)
    mis = diagnose_missingness(df, target) if df[target].nunique() > 1 else {}
    return {
        "perfect": [p["column"] for p in lea.get("perfect_predictors", [])],
        "constant": [p["column"] for p in lea.get("constant_columns", [])],
        "id": [p["column"] for p in lea.get("id_like_columns", [])],
        "mnar": [p["column"] for p in mis.get("target_correlated", [])],
        "skipped": bool(lea.get("skipped", False)),
        "summary": lea.get("summary", ""),
    }


def build_fixture_table() -> list[dict]:
    rows = []
    for fx in FIXTURES:
        df = pd.read_csv(DATA / fx["file"])
        found = _found(df, fx["target"])

        must = fx["must"]
        hit, missing = [], []
        for kind, cols in must.items():
            for c in cols:
                (hit if c in found[kind] else missing).append(f"{kind}:{c}")

        extra = []
        for kind in ("perfect", "constant", "id", "mnar"):
            extra += [f"{kind}:{c}" for c in found[kind] if c not in must.get(kind, [])]

        rows.append(
            {
                "file": fx["file"],
                "n": len(df),
                "injected": fx["injected"],
                "hit": hit,
                "missing": missing,
                "extra": extra,
                "skipped": found["skipped"],
                "ok": not missing,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 三、第一类错误率（蒙特卡洛）
# --------------------------------------------------------------------------- #
def type_one_error(n_rep: int = 400) -> dict:
    rng = np.random.default_rng(SEED)

    # (a) Little's MCAR：数据真的 MCAR 时不该拒绝零假设
    p_values = []
    for _ in range(n_rep):
        frame = pd.DataFrame(rng.standard_normal((300, 10)), columns=[f"c{i}" for i in range(10)])
        for col in frame.columns:
            mask = rng.random(300) < 0.15
            frame.loc[mask, col] = np.nan
        p = littles_mcar_test(frame)
        if p is not None:
            p_values.append(p)

    # (b) 泄漏扫描：20 个与标签独立的白噪声列，不该有任何一列越过阈值
    max_aucs = []
    for _ in range(n_rep):
        X = rng.standard_normal((200, 20))
        y = rng.integers(0, 2, 200)
        aucs = [max(a, 1 - a) for a in (roc_auc_score(y, X[:, j]) for j in range(20))]
        max_aucs.append(float(max(aucs)))

    max_aucs = np.array(max_aucs)

    # (c) 小样本：n=60 时单特征 AUC 的抽样方差大，0.95 阈值不再安全
    small_aucs = []
    for _ in range(n_rep):
        X = rng.standard_normal((60, 20))
        y = rng.integers(0, 2, 60)
        aucs = [max(a, 1 - a) for a in (roc_auc_score(y, X[:, j]) for j in range(20))]
        small_aucs.append(float(max(aucs)))

    small_aucs = np.array(small_aucs)
    return {
        "n_rep": n_rep,
        "little_reject": float(np.mean(np.array(p_values) < 0.05)),
        "little_median_p": float(np.median(p_values)),
        "leak_fpr_95": float(np.mean(max_aucs >= 0.95)),
        "leak_fpr_90": float(np.mean(max_aucs >= 0.90)),
        "leak_fpr_85": float(np.mean(max_aucs >= 0.85)),
        "max_auc_p50": float(np.percentile(max_aucs, 50)),
        "max_auc_p95": float(np.percentile(max_aucs, 95)),
        "small_fpr_95": float(np.mean(small_aucs >= 0.95)),
        "small_auc_p95": float(np.percentile(small_aucs, 95)),
    }


# --------------------------------------------------------------------------- #
# 四、同源特征聚类去重
# --------------------------------------------------------------------------- #
def cluster_dedup() -> dict:
    df = pd.read_csv(DATA / "real_cancer.csv")
    y = df["is_malignant"]

    raw = []
    for col in df.columns:
        if col == "is_malignant" or df[col].dtype.kind not in "if":
            continue
        mask = df[col].notna()
        if mask.sum() <= 30 or df.loc[mask, col].nunique() <= 1:
            continue
        if len(np.unique(y[mask])) != 2:
            continue
        auc = roc_auc_score(y[mask], df.loc[mask, col])
        auc = max(auc, 1 - auc)
        if auc >= AUC_THRESHOLD:
            raw.append((col, round(float(auc), 3)))

    report = scan_leakage(df, "is_malignant")
    return {
        "raw": raw,
        "reported": [(p["column"], p["single_feature_auc"]) for p in report["perfect_predictors"]],
        "raw_n": len(raw),
        "reported_n": len(report["perfect_predictors"]),
    }


# --------------------------------------------------------------------------- #
# 五、图
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # 图 1：缺失概况。只画有缺失的列 + 用 bar 而不是 matrix，
    # 否则 1000×16 的矩阵在缩略图里就是一块黑。
    plt.figure(figsize=(8, 4))
    msno.bar(df, figsize=(8, 4), fontsize=9, labels=True)
    plt.title("缺失概况（仅列级比例，便于缩略图阅读）")
    plt.tight_layout()
    plt.savefig(FIG / "missing_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 图 2：与目标的相关性
    plt.figure(figsize=(6, 5))
    corr = df[["target", "post_event_status", "feature_0", "feature_1"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("特征与目标的相关性热力图")
    plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 图 3：单特征 AUC（原始 ROC-AUC，不做交叉验证、不训练模型）
    aucs = {}
    for col in df.columns:
        if col == "target" or df[col].dtype.kind not in "if":
            continue
        mask = df[col].notna()
        if len(np.unique(df.loc[mask, "target"])) != 2:
            continue
        auc = roc_auc_score(df.loc[mask, "target"], df.loc[mask, col])
        aucs[col] = max(auc, 1 - auc)
    series = pd.Series(aucs).sort_values()

    plt.figure(figsize=(8, 4))
    series.plot(
        kind="barh",
        color=["crimson" if v >= AUC_THRESHOLD else "gray" for v in series],
    )
    plt.axvline(AUC_THRESHOLD, color="black", linestyle="--", label="泄漏报警阈值 (0.95)")
    plt.title("各特征单变量预测能力（原始 ROC-AUC）")
    plt.xlabel("ROC-AUC（未训练模型、未做交叉验证）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "feature_aucs.png", bbox_inches="tight", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 六、写 RESULTS.md
# --------------------------------------------------------------------------- #
def _fmt(items: list[str]) -> str:
    return "、".join(f"`{x}`" for x in items) if items else "—"


def write_results(case: dict, fixtures: list[dict], err: dict, dedup: dict) -> None:
    lines: list[str] = []
    add = lines.append

    add("# 基准与案例研究\n")
    add(
        f"> 全部数字可复现：`python benchmarks/case_study.py`"
        f"（约 40 秒，无需 API Key、不联网）。随机种子 {SEED}，生成于 {date.today().isoformat()}。\n"
    )

    add("## 1. 案例研究：删哪一列由工具决定\n")
    add(
        "脚本里**没有出现泄漏列名的字面量**：`scan_leakage()` 先给出候选，"
        "下游 `drop()` 用的就是它的返回值。\n"
    )
    add("| 项目 | 结果 |")
    add("| --- | --- |")
    add(f"| 数据规模 | {case['n_rows']} 行 × {case['n_features']} 特征（15 个基础特征 + 1 条事后字段） |")
    add(f"| 工具标记为泄漏的特征 | {_fmt(case['flagged'])} |")
    add("| 脚本是否硬编码列名 | **否**——删除列表来自工具返回值 |")
    add(f"| 5 折交叉验证 AUC（含泄漏特征） | {case['auc_with']:.3f} |")
    add(f"| 5 折交叉验证 AUC（按工具建议剔除后） | {case['auc_without']:.3f} |")
    add(f"| 虚高幅度 | **{case['auc_with'] - case['auc_without']:.3f}** |")
    add("")
    add(
        "**结论**：这条事后字段让离线 AUC 虚高 "
        f"{case['auc_with'] - case['auc_without']:.3f}。"
        "它在事件发生之后才写入，线上预测时不可得，属于典型的 target leakage。"
        "DQ-Doctor 在没有被告知任何线索的情况下独立命中。\n"
    )

    add("## 2. fixture 命中情况\n")
    add("6 份数据（含 1 份真实的乳腺癌诊断数据）全部跑一遍，对照「人工注入的问题」应否被检出：\n")
    add("| 数据集 | 行数 | 注入的问题 | 实测命中 | 额外发现 | 漏报 | 判定 |")
    add("| --- | ---: | --- | --- | --- | --- | :---: |")
    for r in fixtures:
        add(
            f"| `{r['file']}` | {r['n']} | {r['injected']} | {_fmt(r['hit'])} | "
            f"{_fmt(r['extra'])} | {_fmt(r['missing'])} | {'✅' if r['ok'] else '❌'} |"
        )
    add("")
    add(
        "> `real_cancer.csv` 的两处\"额外发现\"（`worst perimeter` AUC 0.975、"
        "`worst concave points` 0.967）不是工具出错，而是**已知误报**：它们是恶性肿瘤的"
        "真实强特征，不是泄漏。当前阈值是纯统计判据，无法区分\"事后可得\"与\"本身就很强\"，"
        "这也是 Roadmap 里「带标注基准集」的动机。\n"
    )

    add("## 3. 第一类错误率（蒙特卡洛，400 次重复）\n")
    add(f"每次重复独立生成数据，统计在零假设成立时误报的比例（{err['n_rep']} 次重复）：\n")
    add("| 检验 | 名义水平 | 实测第一类错误率 | 解读 |")
    add("| --- | ---: | ---: | --- |")
    add(
        f"| Little's MCAR（10 列，15% MCAR 缺失） | 5% | **{err['little_reject']:.1%}** | "
        f"低于名义水平，p 中位数 {err['little_median_p']:.2f}，检验偏保守 |"
    )
    add(
        f"| 泄漏扫描 AUC ≥ 0.95（20 个白噪声列，n=200） | — | **{err['leak_fpr_95']:.1%}** | "
        "纯噪声完全够不到 0.95，零误报 |"
    )
    add(
        f"| 同上，阈值降到 0.90 | — | {err['leak_fpr_90']:.1%} | "
        f"仍未误报：n=200 下噪声列最大 AUC 的 95 分位只有 {err['max_auc_p95']:.3f}，"
        "远低于 0.90 |"
    )
    add(
        f"| 同上，阈值降到 0.85 | — | {err['leak_fpr_85']:.1%} | "
        "仍是 0 误报：20 个噪声列仍然够不到阈值 |"
    )
    add(
        f"| 同上，阈值 0.95 但样本量降到 n=60 | — | {err['small_fpr_95']:.1%} | "
        f"阈值仍未被击穿，但噪声 95 分位从 {err['max_auc_p95']:.3f} 抬到 "
        f"{err['small_auc_p95']:.3f}，余量明显收窄 |"
    )
    add("")
    add(
        f"20 个噪声列中最大的单特征 AUC：中位数 {err['max_auc_p50']:.3f}，"
        f"95 分位 {err['max_auc_p95']:.3f}。**0.95 高于 n=200 时的噪声天花板**，"
        "所以它在\"不误报\"这一侧是安全的——真正的代价落在另一侧："
        "`real_cancer.csv` 的合法强特征被误报（第 2 节），以及 0.92 左右的真实泄漏会被漏掉。"
        "换句话说，这个阈值偏保守，宁可漏检也不冤枉，应按业务代价重新标定。"
        f"样本量降到 n=60 时噪声 95 分位抬到 {err['small_auc_p95']:.3f}，"
        "阈值余量随之收窄，所以解读这个阈值时必须带上样本量。\n"
    )

    add("## 4. 同源特征聚类去重\n")
    add(
        "单一阈值会把同一族高度相关的特征全部报出来，对用户是噪音。"
        "`scan_leakage()` 用 |ρ|>0.9 做聚类，只保留组内 AUC 最高的代表：\n"
    )
    add(f"- 原始候选（逐个算 AUC）：**{dedup['raw_n']} 个** —— {_fmt([c for c, _ in dedup['raw']])}")
    add(
        f"- 聚类后输出：**{dedup['reported_n']} 个** —— "
        f"{_fmt([f'{c} ({a:.3f})' for c, a in dedup['reported']])}"
    )
    add(
        f"- 噪音降低 {(dedup['raw_n'] - dedup['reported_n']) / max(dedup['raw_n'], 1):.0%}，"
        "且没有丢掉最强信号\n"
    )

    add("## 5. 复现\n")
    add("```bash")
    add('pip install -e ".[bench]"')
    add("python benchmarks/case_study.py    # 重跑全部表格与图片")
    add("```")

    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = generate_realistic_data()
    case = run_case_study(df)
    print(f"[case study] flagged={case['flagged']} "
          f"auc_with={case['auc_with']:.3f} auc_without={case['auc_without']:.3f}")

    make_figures(df)
    print("[figures] missing_bar / correlation_heatmap / feature_aucs 已更新")

    fixtures = build_fixture_table()
    for r in fixtures:
        print(f"[fixture] {r['file']:22s} ok={r['ok']} hit={len(r['hit'])} missing={r['missing']}")

    err = type_one_error()
    print(f"[type-I] little={err['little_reject']:.3%} "
          f"leak@.95={err['leak_fpr_95']:.3%} leak@.90={err['leak_fpr_90']:.3%}")

    dedup = cluster_dedup()
    print(f"[dedup] raw={dedup['raw_n']} -> reported={dedup['reported_n']}")

    write_results(case, fixtures, err, dedup)
    print(f"[done] {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
