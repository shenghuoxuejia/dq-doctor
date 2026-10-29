\# 🩺 DQ-Doctor: 建模感知的数据质量诊断 Agent



\*\*DQ-Doctor\*\* 是一个基于 \*\*LangGraph\*\* 和大语言模型（LLM）的数据质量诊断 Agent。传统的质量检查工具只会告诉你“缺失率 35%”或“存在 ID 列”，但 DQ-Doctor 专为\*\*建模前（Pre-modeling）\*\*设计，它会告诉你：\*\*“这个缺失是信息性的，直接删除会引入选择偏差”\*\* 以及 \*\*“这个特征的 AUC 太高，存在目标泄漏，上线必崩溃”\*\*。



\## ✨ 核心特性 (Features)



\- 🕳️ \*\*缺失模式诊断\*\*：不仅计算缺失率，更通过 Little's MCAR 检验和相关性分析，判断缺失是否与 Target 强相关（MNAR 嫌疑）。

\- 🚨 \*\*数据泄漏扫描\*\*：自动抓取高基数 ID 列以及单特征异常高 AUC 的“内鬼”特征，防范后见之明偏差（Leakage）。

\- 🧠 \*\*LLM 业务洞察\*\*：基于 DeepSeek / OpenAI 等大模型，将冷冰冰的统计检验结果转化为有业务深度的专家诊断报告，并按威胁程度自动排序。

\- 🛡️ \*\*防幻觉硬拦截\*\*：针对大语言模型容易“无中生有”的问题，设计了基于规则的确定性兜底机制，保证结论严谨。

\- 🎨 \*\*可视化交互界面\*\*：基于 Streamlit，支持拖拽上传 CSV，一键生成诊断面板。



\## 🏗️ 技术架构 (Architecture)



\- \*\*Agent 编排\*\*: \[LangGraph](https://github.com/langchain-ai/langgraph) (状态机编排与状态合并)

\- \*\*数据与统计\*\*: Pandas, NumPy, SciPy, Statsmodels (MICE 算法)

\- \*\*大语言模型\*\*: LangChain + DeepSeek API (兼容 OpenAI)

\- \*\*前端 UI\*\*: Streamlit



\## 🚀 快速开始 (Quick Start)



1\. \*\*克隆项目并安装依赖\*\*

&#x20;  ```bash

&#x20;  git clone https://github.com/yourname/dq-doctor.git

&#x20;  cd dq-doctor

&#x20;  uv venv \&\& uv pip install -e ".\[dev]"

&#x20;  # 或使用 pip install -e .

