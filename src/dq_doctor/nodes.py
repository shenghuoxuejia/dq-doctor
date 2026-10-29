"""LLM 汇总节点：把结构化诊断结果翻译成业务洞察。

设计要点
--------
1. **LLM 惰性构造**：模块导入阶段不读取 ``DEEPSEEK_API_KEY``，因此
   没有 Key 时 ``pytest`` / ``streamlit run app.py`` 依然可以正常启动，
   只有真正走到 LLM 那一步才会用到凭证。
2. **无 Key 也能用**：缺少 Key 或 LLM 调用失败时，回退到**规则引擎**
   生成确定性的中文报告，而不是抛异常或返回一句"请看面板"。
3. **防幻觉短路**：数据确实干净时直接返回确定性结论，不消耗 token。
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .state import DQState

load_dotenv()

MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
KNOWN_MODULES = ("missingness", "leakage")

SYNTH_PROMPT = """你是一个资深的数据科学专家。下面是自动化工具对一份建模数据集的结构化诊断结果。

诊断结果 JSON:
{diagnoses}

请完成以下任务：
1. 按“对建模决策的威胁程度”从高到低对各个诊断进行排序。
2. 针对最严重的问题写一段解释。

⚠️ 极其重要的规则：
如果诊断结果显示“数据完美”、“未发现目标相关的信息性缺失”、“0个单特征预测能力异常”，代表数据质量优秀。你绝对不可以虚构风险。请直接表扬数据质量很高，提醒防过拟合即可。

⚠️ 关于高 AUC 特征：
算法报告的 AUC 异常特征只是“数学候选”。如果特征名包含 "worst", "total" 等后置统计词，可能是合法的强预测因子，需提醒用户核实“预测时点是否可得”；如果包含 "id", "index" 或与标签同义，则必定是严重泄漏。

⚠️ 关于扫描被跳过 (skipped)：
如果 `leakage` 模块显示 `"skipped": true`，说明目标列不支持（如多分类/回归）。请你在解释中**务必明确指出**：“由于任务是 [多分类/回归]，系统未执行高 AUC 泄漏扫描，请用户自行警惕数据泄漏风险”。

请严格输出以下 JSON 格式：
{{"severity_order": ["排第一的模块名", "排第二的模块名"], "top_insight": "你的专业解释..."}}
"""


# --------------------------------------------------------------------------- #
# LLM 客户端：惰性构造
# --------------------------------------------------------------------------- #
def _get_llm() -> ChatOpenAI | None:
    """按需构造 LLM 客户端；未配置 API Key 时返回 None（走规则引擎）。"""
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        return None
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )


# --------------------------------------------------------------------------- #
# 确定性规则：排序 + 报告
# --------------------------------------------------------------------------- #
def _collect(diagnoses: dict) -> dict:
    """把嵌套的诊断字典摊平成规则引擎与 LLM 分支共用的标量。"""
    m = diagnoses.get("missingness") or {}
    lea = diagnoses.get("leakage") or {}

    rates = m.get("per_column_missing_rate") or {}
    return {
        "m_correlated": m.get("target_correlated") or [],
        "m_mechanism": m.get("mechanism_global") or "未知",
        "m_mcar_p": m.get("mcar_p_value"),
        "m_max_rate": max(rates.values()) if rates else 0.0,
        "l_perfect": lea.get("perfect_predictors") or [],
        "l_id": lea.get("id_like_columns") or [],
        "l_const": lea.get("constant_columns") or [],
        "l_skipped": bool(lea.get("skipped", False)),
        "l_skip_reason": lea.get("skip_reason") or "",
        "l_summary": lea.get("summary") or "",
    }


def _is_healthy(f: dict) -> bool:
    """数据确实干净：无泄漏候选、无 MNAR 嫌疑、无高缺失，且泄漏扫描确实跑过。"""
    return (
        not f["m_correlated"]
        and not f["l_perfect"]
        and not f["l_id"]
        and not f["l_skipped"]
        and f["m_max_rate"] < 0.05
    )


HEALTHY_INSIGHT = (
    "【健康体检】数据质量极高！经全面扫描，未发现目标泄漏（Leakage）、高基数 ID 干扰，"
    "且缺失值与标签无显著关联（无 MNAR 嫌疑）。该数据集可以直接进入特征工程与建模阶段，"
    "建议重点关注模型的交叉验证与过拟合问题。"
)


def _rule_based_severity(diagnoses: dict) -> list[str]:
    """没有 LLM 时的确定性威胁度排序。"""
    f = _collect(diagnoses)
    if _is_healthy(f):
        return ["healthy", "missingness", "leakage"]

    leakage_score = 3 * len(f["l_perfect"]) + 2 * len(f["l_id"]) + len(f["l_const"])
    missingness_score = 2 * len(f["m_correlated"])

    if leakage_score >= missingness_score:
        order = ["leakage", "missingness"]
    else:
        order = ["missingness", "leakage"]

    if f["l_skipped"]:
        # 扫描被跳过时它不构成"已确认风险"：排到最后，但保留以便 UI 提示"未扫描"。
        order = [m for m in order if m != "leakage"] + ["leakage"]
    return order


def _rule_based_report(diagnoses: dict | None = None, reason: str = "") -> str:
    """规则引擎报告：零 token、零网络，仍给出可执行的结论。"""
    diagnoses = diagnoses or {}
    f = _collect(diagnoses)
    lines: list[str] = ["**规则引擎报告**（未启用大模型，结论由确定性规则生成）"]

    if _is_healthy(f):
        lines.append(HEALTHY_INSIGHT)
        return "\n\n".join(lines)

    # --- 缺失 ---
    lines.append(f"**① 缺失模式**：{f['m_mechanism']}")
    if f["m_max_rate"] > 0:
        lines.append(f"- 最高单列缺失率：{f['m_max_rate']:.1%}")
    if f["m_correlated"]:
        lines.append(f"- 发现 {len(f['m_correlated'])} 个与目标相关的信息性缺失特征（MNAR 嫌疑）：")
        for item in f["m_correlated"]:
            col = item.get("column", "?")
            eff = item.get("effect_size")
            eff_txt = f"{eff:.3f}" if isinstance(eff, (int, float)) else "n/a"
            lines.append(f"  - `{col}`（效应量 {eff_txt}）建议保留缺失指示变量，不要直接整行删除。")
    else:
        lines.append("- 未发现与目标显著相关的信息性缺失。")

    # --- 泄漏 ---
    if f["l_skipped"]:
        lines.append(f"**② 泄漏扫描**：已跳过（{f['l_skip_reason'] or '目标列不支持'}）。")
        lines.append("- 注意：本次**未执行**高 AUC 泄漏扫描，请自行警惕时间穿越、后置统计量等泄漏风险。")
    else:
        lines.append("**② 泄漏扫描**：")
        if f["l_const"]:
            lines.append(f"- {len(f['l_const'])} 个全空/常量列，可直接删除：")
            for item in f["l_const"]:
                lines.append(f"  - `{item.get('column', '?')}`")
        if f["l_id"]:
            lines.append(f"- {len(f['l_id'])} 个高基数 ID 列，应排除出特征集：")
            for item in f["l_id"]:
                lines.append(f"  - `{item.get('column', '?')}`")
        if f["l_perfect"]:
            lines.append(f"- {len(f['l_perfect'])} 个单特征 AUC 异常特征（**需结合业务核实**，不必然泄漏）：")
            for item in f["l_perfect"]:
                auc = item.get("single_feature_auc")
                auc_txt = f"{auc:.3f}" if isinstance(auc, (int, float)) else "n/a"
                lines.append(f"  - `{item.get('column', '?')}` 单特征 AUC={auc_txt}")
            lines.append(
                "  - 若特征名带 `worst` / `total` 等聚合统计词，请核实在**预测时点**是否已经可得；"
                "若与标签同源或事后产生，则必须剔除。"
            )
        if not (f["l_const"] or f["l_id"] or f["l_perfect"]):
            lines.append("- 未发现明显的数据泄漏。")

    if reason:
        lines.append(f"> {reason}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    """从可能包裹了 ``` 代码块或前后缀说明的文本里抽出第一个 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("响应中没有 JSON 对象")
    return json.loads(cleaned[start : end + 1])


# --------------------------------------------------------------------------- #
# 节点
# --------------------------------------------------------------------------- #
def synthesize_node(state: DQState) -> dict:
    diagnoses = state.get("diagnoses") or {}
    if not diagnoses:
        return {
            "severity_order": _rule_based_severity({}),
            "messages": [("ai", "未发现任何明显的数据质量问题。")],
        }

    # 决定性短路：数据确实干净时不调用 LLM（省钱 + 杜绝无中生有）
    if _is_healthy(_collect(diagnoses)):
        return {
            "severity_order": ["healthy", "missingness", "leakage"],
            "messages": [("ai", HEALTHY_INSIGHT)],
        }

    llm = _get_llm()
    if llm is None:
        return {
            "severity_order": _rule_based_severity(diagnoses),
            "messages": [
                (
                    "ai",
                    _rule_based_report(
                        diagnoses,
                        reason="未检测到 `DEEPSEEK_API_KEY`，已使用规则引擎；"
                        "配置 Key 后可获得大模型的业务解读。",
                    ),
                )
            ],
        }

    prompt = SYNTH_PROMPT.format(diagnoses=json.dumps(diagnoses, ensure_ascii=False, indent=2))
    try:
        response = llm.invoke(prompt)
        parsed = _extract_json(str(response.content))
        severity_order = [m for m in parsed.get("severity_order", []) if m in KNOWN_MODULES]
        top_insight = str(parsed.get("top_insight") or "").strip()
        if not top_insight:
            raise ValueError("top_insight 为空")
        if not severity_order:
            severity_order = _rule_based_severity(diagnoses)
    except Exception as exc:  # noqa: BLE001 - 任何 LLM 故障都必须降级而非崩溃
        return {
            "severity_order": _rule_based_severity(diagnoses),
            "messages": [
                (
                    "ai",
                    _rule_based_report(
                        diagnoses,
                        reason=f"大模型调用失败（`{type(exc).__name__}`），已自动回退到规则引擎结论。",
                    ),
                )
            ],
        }

    return {"severity_order": severity_order, "messages": [("ai", top_insight)]}
