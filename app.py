import streamlit as st
import pandas as pd
import uuid
from dq_doctor.graph import build_graph

# 1. 页面基础设置
st.set_page_config(page_title="DQ-Doctor", page_icon="🩺", layout="wide")
st.title("🩺 DQ-Doctor：建模感知的数据质量诊断 Agent")
st.markdown("上传你的建模数据集，Agent 将自动执行缺失检验与泄漏扫描，并由大模型生成业务洞察。")

# 2. 侧边栏：文件上传与参数配置
with st.sidebar:
    st.header("配置面板")
    uploaded_file = st.file_uploader("上传 CSV 数据文件", type=["csv"])
    
    target_col = None
    df = None
    
    # 注意这里的缩进，所有内容都在 if 里面
    if uploaded_file is not None:
        try:
            # 尝试用 utf-8 读取
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果报错，说明是 GBK 编码（比如 Excel 存的中文 CSV），重置指针再读
            uploaded_file.seek(0) 
            df = pd.read_csv(uploaded_file, encoding='gbk')
            
        # 这一部分必须在 try-except 外面，保证读取成功后才预览
        st.write("数据预览：")
        st.dataframe(df.head(3))
        
        # 选择目标列
        target_col = st.selectbox("请选择目标列 (Target)", df.columns)
        
    run_btn = st.button("🚀 运行诊断", type="primary", use_container_width=True)

# 3. 主区域：执行逻辑与结果展示
if run_btn:
    if uploaded_file is None or df is None:
        st.error("请先在左侧上传 CSV 文件！")
    elif target_col is None:
        st.error("请选择目标列！")
    else:
        with st.spinner("Agent 正在深入诊断数据，大模型正在生成报告..."):
            # 准备初始状态
            initial_state = {
                "df": df,
                "target_col": target_col,
                "diagnoses": {},
                "messages": []
            }
            
            # 每次运行使用新的 thread_id，避免缓存干扰
            config = {"configurable": {"thread_id": f"streamlit_{uuid.uuid4().hex[:8]}"}}
            
            try:
                # 构建并执行图
                graph = build_graph()
                result = graph.invoke(initial_state, config=config)
                
                diagnoses = result.get("diagnoses", {})
                llm_insight = result["messages"][-1].content if result.get("messages") else "无分析结果"
                
                # --- 渲染结果 ---
                st.success("✅ 诊断完成！")
                
                # 区域 A：大模型的核心洞察
                st.subheader("🧠 Agent 专家洞察")
                st.info(llm_insight)
                
                # 区域 B：诊断总览
                st.subheader("📊 质量指标概览")
                col1, col2 = st.columns(2)
                
                # 缺失诊断面板
                with col1:
                    st.markdown("### 🕳️ 缺失模式诊断")
                    m = diagnoses.get("missingness", {})
                    if not m:
                        st.write("未执行缺失诊断")
                    else:
                        st.write(f"**全局机制:** {m.get('mechanism_global', '未知')}")
                        target_corr = m.get("target_correlated", [])
                        if target_corr:
                            st.warning(f"发现 {len(target_corr)} 个信息性缺失特征！")
                            for item in target_corr:
                                st.code(item["interpretation"], language="text")
                        else:
                            st.write("未发现目标相关的信息性缺失。")
                            
                # 泄漏诊断面板
                with col2:
                    st.markdown("### 🚨 数据泄漏扫描")
                    l = diagnoses.get("leakage", {})
                    if not l:
                        st.write("未执行泄漏扫描")
                    else:
                        st.write(f"**扫描摘要:** {l.get('summary', '')}")
                        perfect = l.get("perfect_predictors", [])
                        if perfect:
                            st.error(f"发现 {len(perfect)} 个严重泄漏特征！")
                            for item in perfect:
                                st.code(item["risk"], language="text")
                        else:
                            st.write("未发现明显的数据泄漏。")
                            
            except Exception as e:
                st.error(f"运行过程中发生错误：{e}")