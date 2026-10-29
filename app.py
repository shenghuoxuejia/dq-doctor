import uuid

import pandas as pd
import streamlit as st

from dq_doctor.graph import build_graph
from dq_doctor.report import format_constant_item, format_id_item, format_perfect_item

st.set_page_config(page_title="DQ-Doctor", page_icon="🩺", layout="wide")
st.title("🩺 DQ-Doctor：建模感知的数据质量诊断 Agent")
st.markdown("上传你的建模数据集，Agent 将自动执行缺失检验与泄漏扫描，并由大模型生成业务洞察。")

with st.sidebar:
    st.header("配置面板")
    uploaded_file = st.file_uploader("上传 CSV 数据文件", type=["csv"])

    target_col = None
    df = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='gbk')

        st.write("数据预览：")
        st.dataframe(df.head(3))
        target_col = st.selectbox("请选择目标列 (Target)", df.columns)

    run_btn = st.button("🚀 运行诊断", type="primary", use_container_width=True)

if run_btn:
    if uploaded_file is None or df is None:
        st.error("请先在左侧上传 CSV 文件！")
    elif target_col is None:
        st.error("请选择目标列！")
    else:
        with st.spinner("Agent 正在深入诊断数据，大模型正在生成报告..."):
            initial_state = {
                "df": df,
                "target_col": target_col,
                "diagnoses": {},
                "messages": []
            }
            config = {"configurable": {"thread_id": f"streamlit_{uuid.uuid4().hex[:8]}"}}

            try:
                graph = build_graph()
                result = graph.invoke(initial_state, config=config)

                diagnoses = result.get("diagnoses", {})
                profile = result.get("profile", {})
                llm_insight = result["messages"][-1].content if result.get("messages") else "无分析结果"

                severity_order = result.get("severity_order", [])
                known_modules = ["missingness", "leakage"]
                for m in known_modules:
                    if m not in severity_order:
                        severity_order.append(m)

                st.success("✅ 诊断完成！")
                st.subheader("🧠 Agent 专家洞察")
                st.info(llm_insight)

                st.subheader("📊 质量指标概览")

                cols = st.columns(2)
                col_idx = 0

                for module in severity_order:
                    if module == "healthy" or module not in known_modules:
                        continue
                    if col_idx >= 2:
                        break

                    with cols[col_idx]:
                        if module == "missingness":
                            st.markdown("### 🕳️ 缺失与数据概况")
                            m = diagnoses.get("missingness", {})
                            if not m:
                                st.write("未执行缺失诊断")
                            else:
                                st.write(f"**全局机制:** {m.get('mechanism_global', '未知')}")

                                # 新增：展示重复行
                                dup_cnt = profile.get("duplicate_rows", 0)
                                if dup_cnt > 0:
                                    st.warning(f"发现 {dup_cnt} 行完全重复的数据，建议去重。")
                                else:
                                    st.text("未发现重复行。")

                                rates = m.get("per_column_missing_rate", {})
                                if rates:
                                    st.write("**列缺失率:**")
                                    for k, v in rates.items():
                                        st.text(f" - {k}: {v:.1%}")
                                target_corr = m.get("target_correlated", [])
                                if target_corr:
                                    st.warning(f"发现 {len(target_corr)} 个信息性缺失特征！")
                                    for item in target_corr:
                                        st.code(item["interpretation"], language="text")

                        elif module == "leakage":
                            st.markdown("### 🚨 数据泄漏扫描")
                            lea = diagnoses.get("leakage", {})
                            if not lea:
                                st.write("未执行泄漏扫描")
                            else:
                                if lea.get("skipped", False):
                                    st.warning(lea.get("summary", "扫描已跳过"))
                                else:
                                    st.write(f"**扫描摘要:** {lea.get('summary', '')}")

                                    # 展示无效列 / ID 列 / 泄漏候选时都带上列名：
                                    # 只说「发现 1 个高风险特征」而不给列名，用户没法行动。
                                    consts = lea.get("constant_columns", [])
                                    if consts:
                                        st.error(f"发现 {len(consts)} 个全空/常量列（建议删除）：")
                                        for item in consts:
                                            st.code(format_constant_item(item), language="text")

                                    ids = lea.get("id_like_columns", [])
                                    if ids:
                                        st.error(f"发现 {len(ids)} 个高基数 ID 列（建议排除出特征集）：")
                                        for item in ids:
                                            st.code(format_id_item(item), language="text")

                                    perfect = lea.get("perfect_predictors", [])
                                    if perfect:
                                        st.warning(f"发现 {len(perfect)} 个高风险预测特征（请结合业务核实）：")
                                        for item in perfect:
                                            st.code(format_perfect_item(item), language="text")
                                    else:
                                        st.write("未发现明显的数据泄漏。")
                    col_idx += 1

            except Exception as e:
                st.error(f"运行过程中发生错误：{e}")
